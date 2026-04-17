#!/bin/bash
# Pod 재구성 시 환경 복구 스크립트
set -euo pipefail
ENV_NAME="${1:-user4_env}"
PROJECT="/workspace/user4/emotion-project"

echo "[1/4] conda env 생성: $ENV_NAME (python 3.10)"
conda create -y -n "$ENV_NAME" python=3.10

echo "[2/4] pip 업그레이드"
conda run -n "$ENV_NAME" pip install --upgrade pip

echo "[3/4] requirements_lock.txt 설치"
conda run -n "$ENV_NAME" pip install -r "$PROJECT/requirements_lock.txt"

echo "[4/4] GPU 라이브러리 확인"
conda run -n "$ENV_NAME" python -c "import tensorflow as tf; import torch; print('TF:', tf.__version__, 'GPUs:', tf.config.list_physical_devices('GPU')); print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"

echo "[완료] $ENV_NAME 환경 복구됨. bash activate 후 학습 재개 가능."
