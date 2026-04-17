"""Shim module for YUA vision_encoder compatibility.

yua-encoder (vision_encoder.py) 가 원래 YUA VLM 레포의 `src.token_protocol` 모듈에서
`IGNORE_INDEX` 를 import 하도록 돼 있어, 감정 분류 환경에서 직접 실행 가능하게
최소한의 shim 을 제공한다.

IGNORE_INDEX = -100 은 PyTorch CrossEntropyLoss 의 기본값과 동일하다.
감정 분류 파이프라인에선 라벨이 항상 0~3 범위이므로 실질적으로 쓰일 일은 없지만,
multimodal splice 함수들이 참조하므로 정의만 맞춰둔다.
"""
from __future__ import annotations

IGNORE_INDEX: int = -100

__all__ = ["IGNORE_INDEX"]
