#!/usr/bin/env bash
# kd_monitor.sh — KD sweep 진행 모니터 (크론잡 2분 주기).
# 출력: logs/kd_monitor.log (누적, tail 로 실시간 관찰 가능)
# 제1헌법: GPU OOM / python 프로세스 이상 감지 시 로그에 [ALERT] 남김.
# 하드코딩 0 — PROJECT 는 EMOTION_PROJECT_ROOT env 또는 스크립트 위치 기반.

set -uo pipefail

PROJECT="${EMOTION_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG="$PROJECT/logs/kd_monitor.log"
mkdir -p "$PROJECT/logs"

ts=$(date +'%Y-%m-%d %H:%M:%S')
{
    echo "=== $ts ==="
    echo "[gpu]"
    nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null | sed 's/^/    /'
    echo "[python procs]"
    ps -eo pid,pcpu,pmem,rss,etime,cmd --sort=-%mem 2>/dev/null \
        | awk 'NR==1 || /python/' \
        | grep -v 'awk' \
        | head -6 | sed 's/^/    /'
    echo "[latest kd logs]"
    for f in "$PROJECT"/logs/kd_*.stdout "$PROJECT"/logs/teacher_soft_*.stdout \
             "$PROJECT"/logs/exp09_*.stdout; do
        [ -f "$f" ] || continue
        echo "  --- $(basename "$f") (last 3) ---"
        tail -3 "$f" 2>/dev/null | sed 's/^/      /'
    done
    # OOM / FAIL 감지
    if grep -qiE '(CUDA out of memory|OutOfMemory|\[FAIL\]|killed)' \
        "$PROJECT"/logs/kd_*.stdout "$PROJECT"/logs/teacher_soft_*.stdout \
        "$PROJECT"/logs/exp09_*.stdout 2>/dev/null; then
        echo "[ALERT] OOM / FAIL 감지됨!"
        grep -iEn '(CUDA out of memory|OutOfMemory|\[FAIL\]|killed)' \
            "$PROJECT"/logs/kd_*.stdout "$PROJECT"/logs/teacher_soft_*.stdout \
            "$PROJECT"/logs/exp09_*.stdout 2>/dev/null | head -5 | sed 's/^/    /'
    fi
    echo ""
} >> "$LOG" 2>&1
