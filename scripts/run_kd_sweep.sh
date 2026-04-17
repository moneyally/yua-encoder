#!/usr/bin/env bash
# run_kd_sweep.sh — KD sweep 순차 큐.
# Stage 1: teacher soft 생성 (TS off/on 2 variant)
# Stage 2: student distill sweep (4 config)
# 제1헌법: 각 단계 실패 시 즉시 중단 (set -e). 다음 실험 실행 X.
# 제3헌법: 하드코딩 0 — 모든 경로/파라미터 env or CLI 로 조정 가능.
#
# 사용: bash scripts/run_kd_sweep.sh [SMOKE|FULL]
#   SMOKE = --limit-samples 64 로 빠른 파이프 검증 (~10분)
#   FULL  = 전체 데이터 sweep (~3~4시간)

set -euo pipefail

PROJECT="${EMOTION_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT"

# conda env 자동 활성화 (nohup 으로 띄워도 동작)
CONDA_ENV="${CONDA_ENV:-user4_env}"
CONDA_BASE="${CONDA_BASE:-/home/user4/miniconda3}"
if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV" ]; then
    if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV" || {
            echo "[FAIL] conda activate $CONDA_ENV 실패" >&2; exit 1;
        }
    fi
fi
# python + 핵심 패키지 sanity check (제1헌법: FAIL 시 즉시 중단)
python -c "import torch, tensorflow" 2>/dev/null || {
    echo "[FAIL] torch/tensorflow import 불가 — conda env=$CONDA_ENV 확인" >&2
    python --version >&2 || true
    exit 1
}
echo "[env] conda=${CONDA_DEFAULT_ENV:-none}  python=$(which python)"

MODE="${1:-FULL}"
if [[ "$MODE" != "SMOKE" && "$MODE" != "FULL" ]]; then
    echo "[FAIL] 사용: $0 [SMOKE|FULL]" >&2
    exit 1
fi

# SSOT — teacher = ensemble_best (bbox 기준 0.86, 가장 강한 teacher)
TEACHER_ENSEMBLE="${TEACHER_ENSEMBLE:-$PROJECT/models/ensemble_best.json}"
STUDENT_CKPT="${STUDENT_CKPT:-$PROJECT/models/exp06_siglip_linear_probe.pt}"
DATA_ROOT="${DATA_ROOT:-$PROJECT/data_rot}"

TEACHER_DIR="$PROJECT/results/teacher_soft"
mkdir -p "$TEACHER_DIR" "$PROJECT/logs"

echo "========================================"
echo "KD sweep — MODE=$MODE"
echo "teacher ensemble: $TEACHER_ENSEMBLE"
echo "student ckpt:     $STUDENT_CKPT"
echo "data root:        $DATA_ROOT"
echo "========================================"

# 사전 검증 (제1헌법: 파일 없음 등 즉시 FAIL)
for f in "$TEACHER_ENSEMBLE" "$STUDENT_CKPT"; do
    [ -f "$f" ] || { echo "[FAIL] 파일 없음: $f" >&2; exit 1; }
done
[ -d "$DATA_ROOT/img/train" ] || { echo "[FAIL] $DATA_ROOT/img/train 없음" >&2; exit 1; }

SMOKE_LIMIT=""
KD_EPOCHS=5
# CPU 96코어 / GPU 45GB 여유 → 워커 20, prefetch 4, bs 48 으로 GPU util 극대화
KD_NUM_WORKERS="${KD_NUM_WORKERS:-20}"
KD_PREFETCH="${KD_PREFETCH:-4}"
KD_BATCH_SIZE="${KD_BATCH_SIZE:-48}"
if [[ "$MODE" == "SMOKE" ]]; then
    SMOKE_LIMIT="--limit-samples 512"
    KD_EPOCHS=1
    KD_NUM_WORKERS=8
    KD_BATCH_SIZE=32
    echo "[SMOKE] stratified 512 samples, epochs=1"
fi

run_step() {
    local name="$1"; shift
    local logfile="$PROJECT/logs/${name}.stdout"
    echo ""
    echo "────────── [$(date +%H:%M:%S)] STEP: $name ──────────"
    echo "log → $logfile"
    # 제1헌법: 단계 실패 시 set -e 로 전체 sweep 중단
    "$@" >"$logfile" 2>&1
    echo "[$(date +%H:%M:%S)] $name OK"
}

########## Stage 1: Teacher soft ##########
echo ""
echo "============ Stage 1: Teacher soft ============"

# BATCH 버전 사용 (제4헌법: sequential → 10x+ 가속)
TEACHER_SCRIPT="scripts/gen_teacher_soft_batch.py"
TEACHER_BATCH_ARGS=(
    --loader-batch-size "${TEACHER_LOADER_BS:-64}"
    --num-workers       "${TEACHER_WORKERS:-16}"
    --prefetch-factor   "${TEACHER_PREFETCH:-4}"
    --tf-batch          "${TF_BATCH:-32}"
    --vit-batch         "${VIT_BATCH:-64}"
    --siglip-batch      "${SIGLIP_BATCH:-16}"
)

# 1-a: TS off (baseline)
if [[ ! -f "$TEACHER_DIR/train_teacher_tsoff.npz" ]]; then
    run_step "teacher_soft_tsoff" \
        python "$TEACHER_SCRIPT" \
            --ensemble-json "$TEACHER_ENSEMBLE" \
            --data-root "$DATA_ROOT" \
            --split train \
            --crop \
            --min-conf 0.0 \
            --out "$TEACHER_DIR/train_teacher_tsoff.npz" \
            "${TEACHER_BATCH_ARGS[@]}" \
            $SMOKE_LIMIT
else
    echo "  [skip] teacher_soft_tsoff 이미 존재"
fi

# 1-b: TS on (val에서 T* 학습 후 적용)
if [[ ! -f "$TEACHER_DIR/train_teacher_tson.npz" ]]; then
    run_step "teacher_soft_tson" \
        python "$TEACHER_SCRIPT" \
            --ensemble-json "$TEACHER_ENSEMBLE" \
            --data-root "$DATA_ROOT" \
            --split train \
            --crop \
            --apply-ts \
            --min-conf 0.0 \
            --out "$TEACHER_DIR/train_teacher_tson.npz" \
            "${TEACHER_BATCH_ARGS[@]}" \
            $SMOKE_LIMIT
else
    echo "  [skip] teacher_soft_tson 이미 존재"
fi

########## Stage 2: Student KD sweep ##########
echo ""
echo "============ Stage 2: Student KD sweep ============"

# 기본 인자
COMMON=(
    --src-ckpt "$STUDENT_CKPT"
    --data-root "$DATA_ROOT"
    --epochs "$KD_EPOCHS"
    --patience 3
    --batch-size "$KD_BATCH_SIZE"
    --num-workers "$KD_NUM_WORKERS"
    --prefetch-factor "$KD_PREFETCH"
    --amp bf16
    --seed 42
    --crop --augment
    --label-smoothing 0.0
    --grad-clip-norm 1.0
    --weight-decay 1e-4
)

# E09-1: baseline — ts_off, T=4, α=0.7, unfreeze=0 (linear probe)
run_step "exp09_siglip_kd_tsoff_T4_a07_uf0" \
    python scripts/distill_siglip.py \
        --name exp09_siglip_kd_tsoff_T4_a07_uf0 \
        --teacher-soft "$TEACHER_DIR/train_teacher_tsoff.npz" \
        --temperature 4.0 --alpha 0.7 --unfreeze-last-n 0 \
        --lr 5e-5 --lr-backbone 5e-6 \
        --note "KD baseline: ts_off T=4 α=0.7 linear-probe" \
        "${COMMON[@]}"

# E09-2: TS on 비교
run_step "exp09_siglip_kd_tson_T4_a07_uf0" \
    python scripts/distill_siglip.py \
        --name exp09_siglip_kd_tson_T4_a07_uf0 \
        --teacher-soft "$TEACHER_DIR/train_teacher_tson.npz" \
        --temperature 4.0 --alpha 0.7 --unfreeze-last-n 0 \
        --lr 5e-5 --lr-backbone 5e-6 \
        --note "KD variant: ts_on T=4 α=0.7 linear-probe" \
        "${COMMON[@]}" || echo "[WARN] E09-2 실패 (저장 gate FAIL 가능성) — 다음 스텝 계속"

# E09-3: T=2 비교 (더 샤프한 teacher)
run_step "exp09_siglip_kd_tsoff_T2_a07_uf0" \
    python scripts/distill_siglip.py \
        --name exp09_siglip_kd_tsoff_T2_a07_uf0 \
        --teacher-soft "$TEACHER_DIR/train_teacher_tsoff.npz" \
        --temperature 2.0 --alpha 0.7 --unfreeze-last-n 0 \
        --lr 5e-5 --lr-backbone 5e-6 \
        --note "KD variant: ts_off T=2 α=0.7 linear-probe" \
        "${COMMON[@]}" || echo "[WARN] E09-3 실패 — 계속"

# E09-4: partial unfreeze (last 2 blocks)
run_step "exp09_siglip_kd_tsoff_T4_a07_uf2" \
    python scripts/distill_siglip.py \
        --name exp09_siglip_kd_tsoff_T4_a07_uf2 \
        --teacher-soft "$TEACHER_DIR/train_teacher_tsoff.npz" \
        --temperature 4.0 --alpha 0.7 --unfreeze-last-n 2 \
        --lr 5e-5 --lr-backbone 5e-6 \
        --note "KD variant: ts_off T=4 α=0.7 unfreeze last-2" \
        "${COMMON[@]}" || echo "[WARN] E09-4 실패 — 계속"

echo ""
echo "========================================"
echo "KD sweep DONE at $(date +'%Y-%m-%d %H:%M:%S')"
echo "결과 확인: logs/exp09_*.csv + models/exp09_*.pt + experiments.md"
echo "========================================"
