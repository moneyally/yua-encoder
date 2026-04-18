#!/usr/bin/env bash
# auto_phase2.sh — auto_continue.sh 완주 후 성능 올리기 실험 순차 실행.
#
# 목표: ViT 단일 제출 성능 극대 + 앙상블 best config 재탐색
#
# 순서:
#   0) auto_continue.sh 완주 대기
#   1) ablation_eval.py on E5 ViT — face-crop × TTA 12 config grid (최적 조합 찾기)
#   2) ensemble_search.py 재실행 — E9 KD best 포함하여 새 ensemble weight 탐색
#   3) commit + 아침 리포트 업데이트

set -uo pipefail

PROJECT="/workspace/user4/emotion-project"
cd "$PROJECT"

LOG="$PROJECT/logs/auto_phase2.log"
note() { echo "[phase2 $(TZ=Asia/Seoul date +'%H:%M:%S')] $*" | tee -a "$LOG"; }

if [ -f /home/user4/miniconda3/etc/profile.d/conda.sh ]; then
    # shellcheck disable=SC1091
    source /home/user4/miniconda3/etc/profile.d/conda.sh
    conda activate user4_env 2>/dev/null || note "[WARN] conda activate 실패"
fi

note "=== auto_phase2 START ==="

# ===== 0) auto_continue.sh 완주 대기 =====
AC_PID_FILE="$PROJECT/logs/auto_continue.pid"
if [ -f "$AC_PID_FILE" ]; then
    ac_pid=$(cat "$AC_PID_FILE")
    note "[wait] auto_continue PID=$ac_pid 완주 대기..."
    while kill -0 "$ac_pid" 2>/dev/null; do
        sleep 120
    done
    note "[wait] auto_continue 종료 감지"
fi

# ===== 1) ablation_eval.py — ViT crop/TTA 12 config =====
note ""
note "────────── [1/2] ablation_eval (ViT-B/16 crop × TTA 12 config) ──────────"
OUT1="$PROJECT/results/ablation_exp05_vit_b16_two_stage"
if [ -d "$OUT1" ] && [ -f "$OUT1/summary.md" ]; then
    note "  [skip] ${OUT1}/summary.md 이미 존재"
else
    python scripts/ablation_eval.py \
        --model "$PROJECT/models/exp05_vit_b16_two_stage.pt" \
        --data-root "$PROJECT/data_rot" \
        --val-split val \
        --device auto \
        --log-every 200 \
        > "$PROJECT/logs/ablation_eval_e5.stdout" 2>&1
    rc=$?
    note "  → ablation_eval E5 rc=$rc"
    if [ $rc -ne 0 ]; then
        note "  [WARN] ablation_eval 실패 — 다음 진행"
    else
        best_line=$(grep -E "Best:" "$OUT1/summary.md" 2>/dev/null | head -1)
        note "  best: $best_line"
    fi
fi

# ===== 2) Ensemble re-search — E9 KD best 포함 =====
note ""
note "────────── [2/2] Ensemble re-search (E9 KD 포함) ──────────"

# E9 KD best 선정 — 완료된 exp09 ckpt 중 가장 val_acc 높은 것
python - <<'PY' 2>&1 | tee -a "$LOG"
import json, glob
from pathlib import Path

metas = sorted(glob.glob('/workspace/user4/emotion-project/logs/exp09_*.meta.json'))
best_name, best_acc = None, -1.0
rows = []
for p in metas:
    try:
        m = json.load(open(p))
        b = (m.get('best') or {}).get('val_accuracy')
        if b is None:
            continue
        name = Path(p).stem.replace('.meta', '')
        rows.append((name, b, m.get('save_gate_pass')))
        if m.get('save_gate_pass') and b > best_acc:
            best_name, best_acc = name, b
    except Exception as e:
        print(f"skip {p}: {e}")

print("\n## E9 KD 결과")
for n, a, g in sorted(rows, key=lambda x: -x[1]):
    print(f"  {n:60s}  val_acc={a:.4f}  gate={g}")
print(f"\n[best] {best_name}  val_acc={best_acc:.4f}")

# 저장 — phase2 에서 사용
with open('/workspace/user4/emotion-project/logs/e9_kd_best.json', 'w') as f:
    json.dump({'name': best_name, 'val_acc': best_acc}, f)
PY

BEST_JSON="$PROJECT/logs/e9_kd_best.json"
BEST_NAME=$(python -c "import json; d=json.load(open('$BEST_JSON')); print(d.get('name') or '')" 2>/dev/null)

if [ -z "$BEST_NAME" ] || [ ! -f "$PROJECT/models/${BEST_NAME}.pt" ]; then
    note "  [skip] KD best ckpt 없음 — ensemble re-search skip"
else
    note "  KD best = ${BEST_NAME}"
    OUT2="$PROJECT/models/ensemble_with_kd.json"
    python scripts/ensemble_search.py \
        --models \
            "$PROJECT/models/exp02_resnet50_ft_crop_aug.h5" \
            "$PROJECT/models/exp04_effnet_ft_balanced.h5" \
            "$PROJECT/models/exp05_vit_b16_two_stage.pt" \
            "$PROJECT/models/${BEST_NAME}.pt" \
        --val-dir "$PROJECT/data_rot/img/val" \
        --val-label-root "$PROJECT/data_rot/label" \
        --crop-mode bbox \
        --output-config "$OUT2" \
        --output-report "$PROJECT/results/ensemble_with_kd_report.md" \
        --cache-dir "$PROJECT/results/ensemble_cache_phase2" \
        --seed 42 --de-maxiter 60 \
        > "$PROJECT/logs/ensemble_search_phase2.stdout" 2>&1
    rc=$?
    note "  → ensemble re-search rc=$rc"
    if [ -f "$OUT2" ]; then
        new_acc=$(python -c "import json; d=json.load(open('$OUT2')); print(d.get('_val_acc'))" 2>/dev/null)
        note "  new ensemble val_acc = $new_acc"
    fi
fi

# ===== 3) commit =====
note ""
note "────────── [3/3] commit ──────────"
cd "$PROJECT"
git add results/ablation_exp05_vit_b16_two_stage/ models/ensemble_with_kd.json \
        results/ensemble_with_kd_report.md logs/auto_phase2.log 2>/dev/null || true

if git diff --cached --quiet; then
    note "  [skip] 변경 없음"
else
    git commit -m "$(cat <<'EOF'
exp: phase2 ablation — ViT TTA grid + ensemble re-search with KD

- results/ablation_exp05_vit_b16_two_stage/: crop × TTA 12 config val
- models/ensemble_with_kd.json: E2 + E4 + E5 + E9(best KD) weighted voting
- Phase 2 자율 실행 로그 (auto_phase2.log)
EOF
)" 2>&1 | tee -a "$LOG" || note "  [WARN] commit 실패"
fi

# ===== 4) 아침 리포트 갱신 =====
REPORT="$PROJECT/logs/morning_report.md"
{
    echo ""
    echo "---"
    echo "## Phase 2 결과 ($(TZ=Asia/Seoul date +'%H:%M KST'))"
    echo ""
    echo "### Ablation (E5 ViT, crop × TTA)"
    if [ -f "$PROJECT/results/ablation_exp05_vit_b16_two_stage/summary.md" ]; then
        head -30 "$PROJECT/results/ablation_exp05_vit_b16_two_stage/summary.md"
    else
        echo "(결과 없음 — 실패 or skip)"
    fi
    echo ""
    echo "### Ensemble re-search (with KD)"
    if [ -f "$PROJECT/models/ensemble_with_kd.json" ]; then
        cat "$PROJECT/models/ensemble_with_kd.json"
    else
        echo "(결과 없음)"
    fi
} >> "$REPORT"

note "=== auto_phase2 DONE ==="
