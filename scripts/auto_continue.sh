#!/usr/bin/env bash
# auto_continue.sh — 정원 자는 동안 KD ablation 순차 자동 실행.
#
# 실행 순서:
#   0) 현 FULL KD sweep (Stage 2-d) 완료 대기
#   1) CE-only control (α=0) — ChatGPT 권장 #1 (attribution gap 해소)
#   2) Unfreeze=4 ablation — capacity scaling law 확장
#   3) experiments.md 반영 + local commit (push 는 정원이 아침에)
#
# 정책:
#   - 각 단계 실패해도 다음 단계 진행 시도 (로그로 알림)
#   - destructive 작업 (rm/push/force) 없음
#   - token / secret 파일에 안 남김
#   - 모든 출력은 logs/auto_*.log 에 누적

set -uo pipefail

PROJECT="/workspace/user4/emotion-project"
cd "$PROJECT"

LOG="$PROJECT/logs/auto_continue.log"
mkdir -p "$PROJECT/logs"

note() {
    echo "[auto $(TZ=Asia/Seoul date +'%H:%M:%S')] $*" | tee -a "$LOG"
}

# conda env
if [ -f /home/user4/miniconda3/etc/profile.d/conda.sh ]; then
    # shellcheck disable=SC1091
    source /home/user4/miniconda3/etc/profile.d/conda.sh
    conda activate user4_env 2>/dev/null || note "[WARN] conda activate 실패"
fi

note "=== auto_continue START ==="
note "[env] python=$(which python)  torch=$(python -c 'import torch; print(torch.__version__)' 2>&1)"

# ===== 0) FULL KD sweep 완료 대기 =====
SWEEP_PID_FILE="$PROJECT/logs/kd_sweep_full.pid"
if [ -f "$SWEEP_PID_FILE" ]; then
    sweep_pid=$(cat "$SWEEP_PID_FILE")
    note "[wait] FULL sweep PID=$sweep_pid 완료 대기..."
    while kill -0 "$sweep_pid" 2>/dev/null; do
        sleep 60
    done
    note "[wait] FULL sweep 종료 감지"
else
    note "[wait] SWEEP_PID_FILE 없음 — 즉시 다음 단계"
fi

# 앙상블 teacher npz 존재 확인 (제1헌법: FAIL 즉시 알림)
TEACHER_NPZ="$PROJECT/results/teacher_soft/train_teacher_tsoff.npz"
if [ ! -f "$TEACHER_NPZ" ]; then
    note "[FAIL] teacher npz 없음: $TEACHER_NPZ — 이후 단계 skip"
    exit 1
fi

# 공통 학습 인자 (distill_siglip 에 넘길 것)
COMMON_ARGS=(
    --src-ckpt "$PROJECT/models/exp06_siglip_linear_probe.pt"
    --teacher-soft "$TEACHER_NPZ"
    --data-root "$PROJECT/data_rot"
    --lr 5e-5 --lr-backbone 5e-6 --weight-decay 1e-4
    --epochs 5 --patience 3
    --batch-size 48 --num-workers 20 --prefetch-factor 4
    --amp bf16 --seed 42
    --crop --augment
    --grad-clip-norm 1.0
    --label-smoothing 0.0
)

# ===== 1) CE-only control (α=0) =====
note ""
note "────────── [1/2] CE-only control (α=0, attribution gap 해소) ──────────"
NAME1="exp09_ce_only_control"
if [ -f "$PROJECT/models/${NAME1}.pt" ]; then
    note "  [skip] ${NAME1} ckpt 이미 존재"
else
    python scripts/distill_siglip.py \
        --name "$NAME1" \
        --temperature 4.0 --alpha 0.0 --unfreeze-last-n 0 \
        --note "CE-only control (α=0, hard CE only)" \
        "${COMMON_ARGS[@]}" \
        > "$PROJECT/logs/${NAME1}.stdout" 2>&1
    rc=$?
    note "  → ${NAME1} rc=$rc"
    if [ $rc -ne 0 ]; then
        note "  [WARN] ${NAME1} 실패 (prev_val_acc gate 일 수도) — 다음 진행"
    fi
fi

# ===== 2) Unfreeze=4 ablation =====
note ""
note "────────── [2/2] Unfreeze=4 (capacity scaling 확장) ──────────"
NAME2="exp09_siglip_kd_tsoff_T4_a07_uf4"
if [ -f "$PROJECT/models/${NAME2}.pt" ]; then
    note "  [skip] ${NAME2} ckpt 이미 존재"
else
    python scripts/distill_siglip.py \
        --name "$NAME2" \
        --temperature 4.0 --alpha 0.7 --unfreeze-last-n 4 \
        --note "KD unfreeze scaling: last 4 blocks" \
        "${COMMON_ARGS[@]}" \
        > "$PROJECT/logs/${NAME2}.stdout" 2>&1
    rc=$?
    note "  → ${NAME2} rc=$rc"
fi

# ===== 3) local commit (push 는 정원이 아침에) =====
note ""
note "────────── [3/3] local commit ──────────"
cd "$PROJECT"
git add experiments.md logs/exp09_*.csv logs/exp09_*.meta.json scripts/auto_continue.sh 2>/dev/null || true
git diff --cached --stat | head -10 | tee -a "$LOG"
if git diff --cached --quiet; then
    note "  [skip] 변경 없음"
else
    git commit -m "$(cat <<'EOF'
exp: KD ablation (CE-only control + unfreeze=4)

- exp09_ce_only_control: α=0 → attribution gap 검증 (KD 고유 효과 정량화)
- exp09_siglip_kd_tsoff_T4_a07_uf4: capacity scaling last-4
- auto_continue.sh: 정원 자는 동안 순차 자동 실행 스크립트
EOF
)" 2>&1 | tee -a "$LOG" || note "  [WARN] commit 실패"
fi

# ===== 4) 간이 아침 리포트 =====
REPORT="$PROJECT/logs/morning_report.md"
{
    echo "# 아침 리포트 ($(TZ=Asia/Seoul date +'%Y-%m-%d %H:%M KST'))"
    echo ""
    echo "## 간밤 실행 결과 (val_acc)"
    echo ""
    echo '| exp | val_acc | 비고 |'
    echo '|---|---:|---|'
    for f in "$PROJECT"/logs/exp09_*.meta.json; do
        [ -f "$f" ] || continue
        name=$(basename "$f" .meta.json)
        acc=$(python -c "
import json
m=json.load(open('$f'))
best=m.get('best') or {}
va=best.get('val_accuracy')
print(f'{va:.4f}' if va else 'N/A')
" 2>/dev/null)
        gate=$(python -c "
import json
m=json.load(open('$f'))
print('pass' if m.get('save_gate_pass') else 'fail')
" 2>/dev/null)
        echo "| $name | $acc | save_gate=$gate |"
    done
    echo ""
    echo "## 비교 기준선"
    echo ""
    echo "- E6 SigLIP 원본: 0.8192"
    echo "- **E5 ViT 단일**: **0.8458** ← 제출 기준선"
    echo "- 앙상블 상한: 0.8600"
    echo ""
    echo "## 다음 단계 (정원 아침에 할 일)"
    echo ""
    echo "1. 수치 보고 KD best 선정"
    echo "2. CE-only 대비 KD 고유 이득 계산"
    echo "3. git push (토큰 제공 시) + README/보고서 반영"
    echo "4. 외부 데이터 추가 실험 (Kaggle/HF) 여부 결정"
    echo ""
    echo "## 로그"
    echo ""
    echo "\`\`\`"
    tail -30 "$LOG"
    echo "\`\`\`"
} > "$REPORT"

note ""
note "=== auto_continue DONE ==="
note "아침 리포트: $REPORT"
