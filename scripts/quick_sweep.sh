#!/usr/bin/env bash
# Phase 0 + Phase 1: 빠른 모델 비교 스윕
# 6개 모델 × 3 epoch, 전체 5996장. 결과는 experiments.md + 로그.
set -euo pipefail
# 팀원 환경 호환: 환경변수 우선 > 스크립트 위치 기반 자동 감지.
PROJECT="${EMOTION_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${EMOTION_PY:-python}"
cd "$PROJECT"

# 소규모 5996장엔 bs=32~64 적합.
BS="${BS:-32}"
EP="${EP:-3}"

echo "=== Phase 0: GPU 속도 프로파일 (bs=$BS, 1 epoch 첫 실험에서 측정) ==="
echo "=== Phase 1: Quick sweep 6 models × ${EP} epoch ==="
date

MODELS=(
  "q1_custom_cnn       custom_cnn"
  "q2_vgg16            vgg16"
  "q3_resnet50_frozen  resnet50_frozen"
  "q4_resnet50_ft      resnet50_ft"
  "q5_effnet_frozen    efficientnet"
  "q6_effnet_ft        efficientnet_ft"
)

mkdir -p "$PROJECT/logs/sweep"

# 실험별 결과 추적 (요약 테이블용)
declare -a SUMMARY_NAMES=()
declare -a SUMMARY_RCS=()
declare -a SUMMARY_LOGS=()

# for-loop 안에서 한 모델 실패해도 전체 스윕을 계속 돌리기 위해 set +e 로 감쌈
set +e
for entry in "${MODELS[@]}"; do
  name=$(echo "$entry" | awk '{print $1}')
  model=$(echo "$entry" | awk '{print $2}')
  LOG="$PROJECT/logs/sweep/$name.log"
  echo ""
  echo "=== [$(date +%H:%M:%S)] $name ($model) ==="
  t0=$(date +%s)

  # tee 로 전체 로그 보존. 콘솔엔 핵심 이벤트만 표시.
  # PIPESTATUS[0] 로 train.py 의 실제 exit code 확인 (tail/grep 의 0 에 가려지지 않게).
  "$PY" scripts/train.py \
    --name "$name" \
    --model "$model" \
    --epochs "$EP" \
    --batch-size "$BS" \
    --patience 5 \
    --note "phase1-quick-sweep" \
    2>&1 | tee "$LOG" \
    | grep -E "val_acc|Epoch|Error|Traceback|OOM|\[DONE\]|\[FAIL\]" || true
  rc=${PIPESTATUS[0]}

  t1=$(date +%s)
  if [ "$rc" -eq 0 ]; then
    echo "[$(date +%H:%M:%S)] [DONE] $name done in $((t1 - t0))s (rc=0)"
  else
    echo "[$(date +%H:%M:%S)] [FAIL] $name (rc=$rc) after $((t1 - t0))s — 로그: $LOG"
  fi

  SUMMARY_NAMES+=("$name")
  SUMMARY_RCS+=("$rc")
  SUMMARY_LOGS+=("$LOG")
done
set -e

echo ""
echo "=== [$(date +%H:%M:%S)] ALL SWEEP DONE ==="
echo ""
echo "=== 실험별 요약 테이블 ==="
printf "%-22s %-6s %s\n" "NAME" "RC" "VAL_ACC(last)"
printf "%-22s %-6s %s\n" "----" "--" "-------------"
for i in "${!SUMMARY_NAMES[@]}"; do
  n="${SUMMARY_NAMES[$i]}"
  r="${SUMMARY_RCS[$i]}"
  l="${SUMMARY_LOGS[$i]}"
  if [ -f "$l" ]; then
    va=$(grep -E "val_acc" "$l" | tail -1 | sed 's/^[[:space:]]*//')
  else
    va="(no log)"
  fi
  if [ "$r" -eq 0 ]; then
    tag="OK"
  else
    tag="FAIL"
  fi
  printf "%-22s %-6s %s\n" "$n" "${tag}:${r}" "${va:-(none)}"
done

echo ""
echo "=== 결과 요약 (experiments.md 최근 라인) ==="
tail -20 "$PROJECT/experiments.md"
