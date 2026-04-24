"""emotion-project / predict.py — 감정 4분류 추론 엔트리포인트 (제출용).

================================================================================
사용법
================================================================================

라이브러리로 import:
    from predict import load_model, predict
    model = load_model("models/best_model.h5")            # 또는 .pt / 앙상블 .json
    label, conf = predict(model, "path/to/img.jpg")
    # label: "anger"|"happy"|"panic"|"sadness"
    # conf:  0.0 ~ 1.0

CLI:
    python predict.py --model models/best_model.h5 --image path/to/img.jpg
    # stdout: {"label": "happy", "confidence": 0.9213, "probs": {...}}

    # 기본 (auto-crop ON): 외부 평가 환경 권장. MTCNN 자동 얼굴 crop 으로 학습 조건(annot_A bbox) 재현
    python predict.py --model models/X.pt --image img.jpg

    # 학습 시 bbox 와 같은 조건 (auto-crop ON + TTA):
    python predict.py --model models/X.pt --image img.jpg --tta

    # 전체 이미지로만 추론 (debug/비교):
    python predict.py --model models/X.pt --image img.jpg --auto-face-crop off

================================================================================
지원 모델 (자동 감지)
================================================================================

- .h5 (TF/Keras)
  scripts/train.py 가 저장한 모델. 클래스: custom_cnn / vgg16 / resnet50_frozen /
  resnet50_ft / efficientnet / efficientnet_ft. preprocess family 는 같은 logs/
  디렉터리의 <name>.meta.json 이 있으면 거기서 읽고, 없으면 모델의 input shape +
  간단한 heuristic 으로 resnet50 기본값 적용.

- .pt (PyTorch, ViT)
  scripts/train_vit.py 가 저장한 체크포인트 (dict: model state_dict + args + meta).
  timm.create_model(meta.model_name, pretrained=False, num_classes=4) 로 재생성 후
  state_dict load. 전처리: Resize(256) → CenterCrop(224) → ImageNet normalize.

- .pt (PyTorch, SigLIP)
  scripts/train_siglip.py 가 저장한 체크포인트. meta.siglip_name 존재 여부로
  SigLIP 판별. VisionEncoder + VisionProjection + classifier 재조립 후 load.
  전처리: SiglipImageProcessor (384 또는 siglip_name 에 따라 자동).

- .json (앙상블 설정)
  {
    "models": [
      {"path": "models/exp04.h5", "weight": 1.0},
      {"path": "models/exp05_vit.pt", "weight": 1.2}
    ]
  }
  각 모델의 softmax 확률을 weight 곱한 뒤 평균 → argmax + max prob.

================================================================================
제출 (CLAUDE.md #8, 4/30 18시)
================================================================================

- models/best_model.h5 (or .pt / 앙상블 .json)
- predict.py — 본 파일 (load_model + predict 2줄 호출 가능)
- README.md

================================================================================
주의
================================================================================

- bbox crop 이 필요한 모델이라도 predict.py 는 crop 없이 **이미지 전체** 로 돌린다
  (test set 의 annot_A 를 모르는 상황 가정). crop 기반 학습 모델은 여기서 약간의
  성능 하락이 있을 수 있음. 학습 시 aug 로 전체 얼굴 prior 를 일부 학습시킨 모델이
  제출용으로 더 안전함.
- EXIF orientation 은 `ImageOps.exif_transpose` 로 자동 정규화 (data_rot 와 동일).
- CPU-only 환경에서도 동작 (torch: map_location="cpu" 지정).
- TF 또는 torch 한쪽만 설치된 환경도 lazy import 로 지원 — 쓰지 않는 쪽은 import
  실패 시 해당 모델 type 만 에러를 낸다.

================================================================================
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import numpy as np
from PIL import Image, ImageOps

# 프로젝트 루트: 환경변수 > __file__ (predict.py 는 프로젝트 루트에 있음)
PROJECT = Path(os.environ.get("EMOTION_PROJECT_ROOT",
                              str(Path(__file__).resolve().parent)))

# 고정 클래스 순서 (train.py / train_vit.py / train_siglip.py 와 100% 정합)
CLASSES = ["anger", "happy", "panic", "sadness"]
NUM_CLASSES = len(CLASSES)

# ==========================================================================
# 공통 유틸
# ==========================================================================

ImageLike = Union[str, Path, Image.Image]


def _load_pil(image: ImageLike) -> Image.Image:
    """PIL.Image 로 로드 + EXIF orientation 정규화 + RGB.

    학습 파이프라인(data_rot 기준)과 동일한 전처리. 추론 시 들어오는 raw JPEG 은
    EXIF orientation 태그가 남아있을 수 있으므로 exif_transpose 반드시 필요.
    """
    if isinstance(image, Image.Image):
        img = image
    else:
        path = Path(image)
        if not path.is_file():
            raise FileNotFoundError(f"이미지 파일 없음: {path}")
        img = Image.open(path)
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        # EXIF 파싱 실패 시 그냥 원본 사용 (data_rot 는 이미 정규화 됐으므로 영향 없음)
        pass
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _read_meta_json(model_path: Path) -> dict:
    """같은 이름의 logs/<name>.meta.json 을 찾아 dict 반환. 없으면 {}."""
    name = model_path.stem  # 'exp04_effnet_ft_balanced'
    # 우선 logs/<name>.meta.json
    candidates = [
        PROJECT / "logs" / f"{name}.meta.json",
        model_path.parent.parent / "logs" / f"{name}.meta.json",
        model_path.with_suffix(".meta.json"),
    ]
    for c in candidates:
        if c.is_file():
            try:
                with open(c, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                continue
    return {}


def _softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return (e / np.sum(e, axis=axis, keepdims=True)).astype(np.float32)


def _wrap_tta(predict_fn: Callable[[Image.Image], np.ndarray]) -> Callable[[Image.Image], np.ndarray]:
    """hflip TTA wrapper.  원본 PIL + 좌우반전 PIL 각각 예측 후 softmax 평균.

    얼굴 좌우대칭 가정 하 감정분류에서 통계적으로 소폭 개선 (~+0.3~0.7%p).
    극소수 비대칭 케이스(한쪽 윙크 등)엔 영향 있지만 평균 성능엔 유리.
    추론 비용 2배.
    """
    def predict_fn_tta(pil: Image.Image) -> np.ndarray:
        p1 = np.asarray(predict_fn(pil), dtype=np.float32).reshape(-1)
        pil_flip = pil.transpose(Image.FLIP_LEFT_RIGHT)
        p2 = np.asarray(predict_fn(pil_flip), dtype=np.float32).reshape(-1)
        return ((p1 + p2) / 2.0).astype(np.float32)
    return predict_fn_tta


# --------------------------------------------------------------------------
# Multi-TTA (5crop / multi-scale / hflip 조합)
# --------------------------------------------------------------------------
# 기존 _wrap_tta 는 단순 hflip 1회. 이 wrapper 는 다음 3가지를 orthogonal 하게 조합:
#   (1) crops: "none" | "5crop" — 중심+4모서리 (각 crop 은 side x side)
#   (2) scales: [s1, s2, ...] — 각 scale 로 먼저 전체 이미지를 resize (s x s)
#   (3) hflip: 각 view 마다 좌우반전 추가 여부
# 최종 view 수 = len(crops_list) × len(scales) × (2 if hflip else 1)
#
# 수학:
#   p_avg = (1/N) * Σ softmax(model(aug_i(x)))
#   여기서 aug_i 는 (scale, crop, flip) 조합의 deterministic 변환.
#   각 predict_fn 은 이미 softmax 출력 (합=1) 이므로 평균도 합=1 보장.
#   argmax(p_avg) = 최종 class.
#
# 설계 주의:
# - predict_fn 자체가 내부에서 학습시 preprocess (Resize→CenterCrop→Normalize) 를
#   수행하므로, 여기서 이미지를 미리 resize/crop 해서 건네줘도 predict_fn 이 다시
#   자기 기준으로 처리함. 즉 scale=224 지정 + predict_fn 이 Resize(256) 이면
#   실제 입력은 predict_fn 의 Resize 가 한 번 더 걸린 결과. 이는 "약한 noise" 로
#   작용해 TTA 효과를 얻는 설계. 학습 때 쓰던 preprocess 는 그대로 유지되므로
#   분포 drift 없음.
# - Image.BILINEAR 로 일관 (학습 파이프라인과 동일).


def _five_crop(pil: Image.Image, side: int) -> List[Image.Image]:
    """PIL (W,H) → 5개 crop (PIL) 리스트: [center, tl, tr, bl, br].

    crop size = (side, side). 이미지가 side 보다 작으면 가능한 만큼 잘라내고
    그대로 반환 (부족한 쪽은 이미지 경계가 crop 경계가 됨).
    torch transforms.FiveCrop 과 동일 시맨틱.
    """
    W, H = pil.size
    s = int(side)
    if s <= 0:
        return [pil]
    # 이미지가 side 보다 작으면 전체 사용 (경계 clip)
    cw = min(s, W)
    ch = min(s, H)
    # center
    cx1 = max(0, (W - cw) // 2)
    cy1 = max(0, (H - ch) // 2)
    center = pil.crop((cx1, cy1, cx1 + cw, cy1 + ch))
    tl = pil.crop((0, 0, cw, ch))
    tr = pil.crop((max(0, W - cw), 0, W, ch))
    bl = pil.crop((0, max(0, H - ch), cw, H))
    br = pil.crop((max(0, W - cw), max(0, H - ch), W, H))
    return [center, tl, tr, bl, br]


def _multi_tta_views(pil: Image.Image,
                     crops: str = "none",
                     scales: Optional[List[int]] = None,
                     hflip: bool = True) -> List[Image.Image]:
    """multi-TTA view 목록 생성.

    Args:
      pil: 원본 (이미 face crop 끝난) PIL RGB.
      crops: "none" | "5crop".
      scales: 각 scale 로 (scale x scale) resize 후 crop 적용. None/[] → 원본 그대로.
      hflip: True 면 각 view 별 PIL.FLIP_LEFT_RIGHT 추가.

    view 수 = max(1, len(scales_list)) × (5 if crops=='5crop' else 1) × (2 if hflip else 1)
    """
    crops = (crops or "none").lower()
    if crops not in ("none", "5crop"):
        raise ValueError(f"tta crops 는 'none'|'5crop': {crops!r}")
    scale_list: List[Optional[int]]
    if scales is None or len(scales) == 0:
        scale_list = [None]
    else:
        scale_list = [int(s) for s in scales]
        for s in scale_list:
            if s is None or s <= 0:
                raise ValueError(f"tta scales 값은 양의 정수: {scales!r}")

    views: List[Image.Image] = []
    for s in scale_list:
        if s is None:
            base = pil
        else:
            base = pil.resize((s, s), Image.BILINEAR)
        if crops == "5crop":
            # crop side 기본 224 (학습 표준). 이미지가 그보다 작으면 _five_crop 이 경계 clip.
            crop_side = 224 if s is None else min(224, s)
            view_list = _five_crop(base, crop_side)
        else:
            view_list = [base]
        for v in view_list:
            views.append(v)
            if hflip:
                views.append(v.transpose(Image.FLIP_LEFT_RIGHT))
    return views


def _wrap_tta_multi(predict_fn: Callable[[Image.Image], np.ndarray],
                    crops: str = "none",
                    scales: Optional[List[int]] = None,
                    hflip: bool = True) -> Callable[[Image.Image], np.ndarray]:
    """Multi-TTA wrapper: 5crop + multi-scale + hflip 조합의 softmax 평균.

    crops="none", scales in (None, [x]), hflip=False  → view 1개 (원본만)
    crops="none", hflip=True                          → view 2개 (기존 _wrap_tta 와 동등)
    crops="5crop", hflip=True, scales=[224,256]       → view 5*2*2 = 20개

    각 view 별 predict_fn 호출 → softmax 벡터 반환 가정. 평균은 단순 산술평균
    (각 view 가 softmax 결과이므로 합=1 보장, 평균도 합=1).
    """
    crops_norm = (crops or "none").lower()
    # 사전 validation (호출마다 반복 경고 방지)
    if crops_norm not in ("none", "5crop"):
        raise ValueError(f"tta crops 는 'none'|'5crop': {crops!r}")
    scale_list = None if (scales is None or len(scales) == 0) else [int(s) for s in scales]
    if scale_list is not None:
        for s in scale_list:
            if s <= 0:
                raise ValueError(f"tta scales 값은 양의 정수: {scales!r}")

    def predict_fn_multi(pil: Image.Image) -> np.ndarray:
        views = _multi_tta_views(pil, crops=crops_norm, scales=scale_list, hflip=hflip)
        if not views:
            # 방어적 fallback — 이론상 _multi_tta_views 는 항상 최소 1개 반환
            return np.asarray(predict_fn(pil), dtype=np.float32).reshape(-1)
        agg = np.zeros(NUM_CLASSES, dtype=np.float64)
        n = 0
        for v in views:
            p = np.asarray(predict_fn(v), dtype=np.float64).reshape(-1)
            if p.shape[0] != NUM_CLASSES:
                raise RuntimeError(
                    f"multi-TTA: predict_fn 가 {p.shape[0]} 차원 반환 (expected {NUM_CLASSES})"
                )
            agg += p
            n += 1
        avg = (agg / float(n)).astype(np.float32)
        # 각 p 가 softmax (합=1) 라 avg 합도 1 (float 오차 제외).
        # softmax 는 logit→prob 변환이므로 probs 합 보정에 쓰면 분포 왜곡.
        # 합산 기반 재정규화 (avg / avg.sum()).
        s = float(avg.sum())
        if not np.isfinite(s) or s <= 0:
            raise RuntimeError(
                f"multi-TTA avg probs invalid (sum={s}). "
                f"predict_fn 반환값이 NaN/0 가능성."
            )
        if abs(s - 1.0) > 1e-6:
            avg = (avg / s).astype(np.float32)
        return avg
    return predict_fn_multi


def _select_tta_wrapper(crops: str,
                        scales: Optional[List[int]],
                        hflip: bool) -> Optional[Callable[[Callable[[Image.Image], np.ndarray]],
                                                          Callable[[Image.Image], np.ndarray]]]:
    """주어진 TTA 파라미터 조합에 대해 가장 적합한 wrapper factory 선택.

    반환:
      None  — TTA 불필요 (crops=none, scales 단일/없음, hflip=False)
      callable(base_fn) → wrapped_fn
    """
    crops_norm = (crops or "none").lower()
    scales_list = [] if scales is None else [int(s) for s in scales]
    multi_scale = len(scales_list) > 1
    use_5crop = (crops_norm == "5crop")
    if not (multi_scale or use_5crop):
        # crops/scales 확장 없음 → hflip 만 필요 여부로 갈림
        if hflip:
            return _wrap_tta
        return None
    # multi-TTA 활성
    return lambda fn: _wrap_tta_multi(fn, crops=crops_norm,
                                      scales=(scales_list or None),
                                      hflip=bool(hflip))


# ==========================================================================
# MTCNN auto face crop (facenet-pytorch)
# ==========================================================================
# 학습 시 --crop 플래그로 annot_A bbox 얼굴만 잘라 학습한 모델들은 inference 에서
# 전체 이미지를 넣으면 val_acc 0.8 → 0.3~0.6 급락. 외부 평가셋은 bbox 없으므로
# 추론 시점에 MTCNN 으로 자동 face detect+crop 이 필수. MTCNN 은 validate_data_rot.py
# 에서 annot_A 와 IoU mean 0.959 로 검증됨 (240/240 매칭).

_MTCNN_INSTANCE: dict = {
    "obj": None,          # MTCNN
    "device": None,       # str
    "min_face_size": None,
    "import_failed": False,
    "import_error": None,
}


def _get_mtcnn(min_face_size: int = 20, device: str = "cpu"):
    """MTCNN 인스턴스 lazy-load + cache.
    같은 (min_face_size, device) 요청은 재사용. 다르면 재생성.
    import 실패 시 None 반환 (+ 경고는 최초 1회만).
    """
    global _MTCNN_INSTANCE
    if _MTCNN_INSTANCE["import_failed"]:
        return None
    cached = _MTCNN_INSTANCE["obj"]
    if (cached is not None
            and _MTCNN_INSTANCE["device"] == device
            and _MTCNN_INSTANCE["min_face_size"] == min_face_size):
        return cached
    try:
        from facenet_pytorch import MTCNN  # type: ignore
    except Exception as e:
        _MTCNN_INSTANCE["import_failed"] = True
        _MTCNN_INSTANCE["import_error"] = repr(e)
        print(
            f"[warn] facenet-pytorch import 실패: {e!r}. "
            f"auto_face_crop 비활성화 (전체 이미지로 추론).",
            file=sys.stderr,
        )
        return None
    try:
        # keep_all=False → 가장 큰/확률 높은 얼굴 1개 선택
        mtcnn = MTCNN(
            image_size=160,           # 내부 썸네일용, 우리는 bbox 만 쓰니까 무관
            margin=0,                 # margin 은 우리가 별도로 관리
            min_face_size=int(min_face_size),
            keep_all=False,
            post_process=False,
            device=device,
        )
    except Exception as e:
        _MTCNN_INSTANCE["import_failed"] = True
        _MTCNN_INSTANCE["import_error"] = repr(e)
        print(
            f"[warn] MTCNN 초기화 실패: {e!r}. auto_face_crop 비활성화.",
            file=sys.stderr,
        )
        return None
    _MTCNN_INSTANCE["obj"] = mtcnn
    _MTCNN_INSTANCE["device"] = device
    _MTCNN_INSTANCE["min_face_size"] = int(min_face_size)
    return mtcnn


def _auto_face_crop_pil(pil: Image.Image,
                        margin: float = 0.1,
                        min_face_size: int = 20,
                        device: str = "cpu") -> Image.Image:
    """MTCNN 으로 face detect → bbox crop 하여 PIL 반환.

    얼굴 0개 or 에러면 원본 반환 (fallback).  margin 비율만큼 bbox 확장 (bbox 의
    width/height 각각에 대해 margin 만큼 좌우/상하 padding 후 이미지 경계 clip).

    여러 얼굴 감지 시 MTCNN keep_all=False 라 가장 큰/확률 높은 1개만 나옴.
    그래도 방어적으로 면적 최대 bbox 재선택.
    """
    if pil is None:
        return pil
    mtcnn = _get_mtcnn(min_face_size=min_face_size, device=device)
    if mtcnn is None:
        return pil  # import/init 실패 — 원본 fallback

    W, H = pil.size
    if W <= 0 or H <= 0:
        return pil

    try:
        boxes, probs = mtcnn.detect(pil)
    except Exception as e:
        print(f"[warn] MTCNN detect crash: {e!r}. 원본 이미지로 fallback.",
              file=sys.stderr)
        return pil

    if boxes is None or len(boxes) == 0:
        return pil

    # 면적 최대 bbox 선택 (keep_all=False 여도 방어)
    best_idx = 0
    best_area = -1.0
    for i, b in enumerate(boxes):
        if b is None:
            continue
        try:
            x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        except Exception:
            continue
        if not all(np.isfinite([x1, y1, x2, y2])):
            continue
        area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        if area > best_area:
            best_area = area
            best_idx = i
    if best_area <= 0:
        return pil

    x1, y1, x2, y2 = [float(v) for v in boxes[best_idx][:4]]

    # margin 확장
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    mx = bw * max(0.0, float(margin))
    my = bh * max(0.0, float(margin))
    x1 -= mx
    x2 += mx
    y1 -= my
    y2 += my

    # clip to image bounds + int cast
    x1 = int(max(0, min(W, round(x1))))
    y1 = int(max(0, min(H, round(y1))))
    x2 = int(max(0, min(W, round(x2))))
    y2 = int(max(0, min(H, round(y2))))

    # 비정상 (음수 면적 / 너무 작음) → fallback
    if x2 - x1 < 4 or y2 - y1 < 4:
        return pil

    try:
        return pil.crop((x1, y1, x2, y2))
    except Exception as e:
        print(f"[warn] PIL crop 실패: {e!r}. 원본으로 fallback.", file=sys.stderr)
        return pil


def _wrap_face_crop(predict_fn: Callable[[Image.Image], np.ndarray],
                    margin: float = 0.1,
                    min_face_size: int = 20,
                    device: str = "cpu") -> Callable[[Image.Image], np.ndarray]:
    """predict_fn 에 auto face crop 전처리 wrapper 씌움.
    wrap 순서는 호출자가 결정: crop → (그다음 TTA 가 원한다면 hflip → predict).
    """
    def predict_fn_cropped(pil: Image.Image) -> np.ndarray:
        cropped = _auto_face_crop_pil(
            pil, margin=margin, min_face_size=min_face_size, device=device,
        )
        return predict_fn(cropped)
    return predict_fn_cropped


# ==========================================================================
# TF (Keras .h5)
# ==========================================================================

def _resolve_tf_preprocess(model_family: str) -> Callable:
    """train.py 의 _resolve_preprocess 와 동일. lazy import."""
    from tensorflow.keras.applications import resnet50, vgg16, efficientnet
    if model_family.startswith("resnet50"):
        return resnet50.preprocess_input
    if model_family == "vgg16":
        return vgg16.preprocess_input
    if model_family.startswith("efficientnet"):
        return efficientnet.preprocess_input
    # custom_cnn
    return lambda x: x.astype(np.float32) / 255.0


def _infer_tf_family(model, meta: dict) -> tuple[str, int]:
    """(model_family, img_size) 추론.

    우선순위:
      1) meta.json 의 args.model 이 있으면 거기서 family 추출
      2) Keras model.name (ResNet50 내부 모델 이름은 'resnet50' 이 박힘)
      3) 기본값 'resnet50' + 224
    """
    img_size = 224
    # 1) meta.args.model
    args_model = (meta.get("args") or {}).get("model")
    if args_model in ("custom_cnn",):
        family = "custom_cnn"
    elif args_model == "vgg16":
        family = "vgg16"
    elif args_model in ("resnet50_frozen", "resnet50_ft"):
        family = "resnet50"
    elif args_model in ("efficientnet", "efficientnet_ft"):
        family = "efficientnet"
    else:
        # 2) 모델 이름 / 레이어 이름 기반 추론
        mname = (getattr(model, "name", "") or "").lower()
        lowered = mname
        # 내부 backbone 찾기 (첫번째 depth 의 functional layer 이름 체크)
        try:
            for lyr in model.layers:
                ln = (getattr(lyr, "name", "") or "").lower()
                if ln:
                    lowered += " " + ln
        except Exception:
            pass
        if "resnet" in lowered:
            family = "resnet50"
        elif "vgg" in lowered:
            family = "vgg16"
        elif "efficient" in lowered:
            family = "efficientnet"
        else:
            # 알려진 backbone 키워드 없음 — 우리 custom_cnn 은 Conv2D + Dense 만이라 이 경로로 옴.
            # 안전한 기본으로 custom_cnn (/255 정규화) 채택. 다른 모델일 경우 경고 출력.
            family = "custom_cnn"
            print(
                f"[warn] preprocess family 자동감지 실패 (model.name={mname!r}). "
                f"custom_cnn (/255) 로 fallback. 만약 ResNet/VGG/EfficientNet 이면 "
                f"logs/<name>.meta.json 에 args.model 을 명시하여 정확히 지정할 것.",
                file=sys.stderr,
            )

    # img_size: meta 우선, 없으면 모델 input_shape
    if (meta.get("args") or {}).get("img_size"):
        try:
            img_size = int(meta["args"]["img_size"])
        except Exception:
            pass
    else:
        try:
            shape = model.input_shape   # (None, H, W, 3)
            if shape and len(shape) >= 3 and shape[1] and shape[2]:
                img_size = int(shape[1])
        except Exception:
            pass

    return family, img_size


def _load_tf(model_path: Path,
             tta: bool = False,
             auto_face_crop: bool = True,
             face_crop_margin: float = 0.1,
             face_crop_min_size: int = 20,
             face_crop_device: str = "cpu",
             tta_crops: str = "none",
             tta_scales: Optional[List[int]] = None,
             tta_hflip: bool = True) -> dict:
    """TF .h5 load → predict_fn dict 반환.  tta=True 면 hflip TTA 적용.
    auto_face_crop=True 면 MTCNN 으로 자동 얼굴 crop 후 resize (학습 조건 일치).
    tta=True 이고 tta_crops/tta_scales 확장 지정 시 multi-TTA 적용."""
    try:
        import tensorflow as tf
    except ImportError as e:
        raise ImportError(
            "TF 모델(.h5) 추론을 위해 tensorflow 설치 필요: pip install tensorflow"
        ) from e

    # GPU OOM/CUDA 환경 문제 시에도 load 는 되도록 memory_growth 시도
    try:
        for g in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
    except Exception:
        pass

    model = tf.keras.models.load_model(str(model_path), compile=False)
    meta = _read_meta_json(model_path)
    family, img_size = _infer_tf_family(model, meta)
    preprocess = _resolve_tf_preprocess(family)

    def predict_fn(pil: Image.Image) -> np.ndarray:
        """PIL → softmax probs (4,). 학습 파이프라인과 동일한 resize/preprocess."""
        # train.py: tf.keras.utils.image_dataset_from_directory 가 내부에서
        # tf.image.resize (bilinear) 로 리사이즈 한 뒤 preprocess_input 적용.
        # 동일하게 PIL.resize (BILINEAR) → numpy float32 → preprocess_input.
        img = pil.resize((img_size, img_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32)   # (H, W, 3) [0, 255]
        arr = preprocess(arr.copy())             # ResNet/EffNet 각자 방식
        batch = np.expand_dims(arr, axis=0)
        out = model.predict(batch, verbose=0)    # (1, 4) — 학습 시 softmax 있음
        probs = np.asarray(out[0], dtype=np.float32)
        # 대부분 train.py 의 마지막 layer 가 Dense(softmax) 라 정규화 돼 있지만,
        # 합이 1 이 아니면 (custom model 등) 방어적 softmax.
        s = float(probs.sum())
        if not np.isfinite(s) or abs(s - 1.0) > 1e-3:
            probs = _softmax_np(probs)
        return probs

    # 배선 순서: (원본 PIL) → face crop → [TTA wrap(hflip/multi)] → predict_fn(core)
    # TTA 는 crop 된 얼굴 PIL 을 변형하는 편이 자연스러움 (얼굴 기준 좌우대칭/zoom).
    predict_fn_core = predict_fn
    predict_fn_base = predict_fn_core
    if auto_face_crop:
        predict_fn_base = _wrap_face_crop(
            predict_fn_core,
            margin=face_crop_margin,
            min_face_size=face_crop_min_size,
            device=face_crop_device,
        )
    if tta:
        factory = _select_tta_wrapper(tta_crops, tta_scales, tta_hflip)
        predict_fn_public = factory(predict_fn_base) if factory else predict_fn_base
    else:
        predict_fn_public = predict_fn_base

    return {
        "predict_fn": predict_fn_public,
        "predict_fn_base": predict_fn_base,   # TTA 없이 쓰고 싶을 때 / predict() override 용
        "predict_fn_raw": predict_fn_core,    # crop/tta 전부 없는 원본 (debug)
        "tta": bool(tta),
        "auto_face_crop": bool(auto_face_crop),
        "type": f"tf_{family}"
                + ("_crop" if auto_face_crop else "")
                + ("_tta" if tta else ""),
        "meta": {
            "path": str(model_path),
            "family": family,
            "img_size": img_size,
            "keras_name": getattr(model, "name", ""),
            "args": (meta.get("args") or {}),
            "tta": bool(tta),
            "tta_crops": str(tta_crops),
            "tta_scales": list(tta_scales) if tta_scales else None,
            "tta_hflip": bool(tta_hflip),
            "auto_face_crop": bool(auto_face_crop),
            "face_crop_margin": float(face_crop_margin),
            "face_crop_min_size": int(face_crop_min_size),
        },
        "_model": model,   # GC 방지 + reuse
    }


# ==========================================================================
# torch (ViT / SigLIP)  — lazy import
# ==========================================================================

def _load_torch_checkpoint(model_path: Path) -> tuple[Any, dict, dict, "Any"]:
    """torch.load → (ckpt, args, meta, torch_module)."""
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "torch 모델(.pt) 추론을 위해 torch 설치 필요: pip install torch"
        ) from e
    try:
        # torch >= 2.6 에서 weights_only=True 가 기본. 우리 ckpt 는 dict 라서 허용됐던
        # 대부분의 scalar/tensor 만 들어있지만, args (dict) 등 때문에 False 로 강제.
        ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
    except TypeError:
        # torch 구버전은 weights_only 인자 없음
        ckpt = torch.load(str(model_path), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError(
            f"예상치 못한 torch ckpt 구조: {type(ckpt)}. "
            f"train_vit.py/train_siglip.py 로 저장된 dict 포맷이 필요함."
        )
    args = ckpt.get("args") or {}
    meta = ckpt.get("meta") or {}
    return ckpt, args, meta, torch


def _is_siglip_ckpt(args: dict, meta: dict, model_path: Optional[Path] = None) -> bool:
    """체크포인트 내 메타로 SigLIP 판별.
    1) args/meta 에 siglip_name 키 존재
    2) meta.model_name 이 siglip 로 시작
    3) distill_siglip.py 산출물: args.src_ckpt 파일명에 siglip
    4) ckpt 파일명 자체에 'siglip'
    5) state_dict 에 'vision_encoder.siglip' prefix 키 존재 (최종 fallback)
    """
    if "siglip_name" in args or "siglip_name" in meta:
        return True
    mn = meta.get("model_name", "") or ""
    if mn.startswith("siglip"):
        return True
    # distill 산출물 — src_ckpt 추적
    src = args.get("src_ckpt") or args.get("src-ckpt") or ""
    if isinstance(src, str) and "siglip" in src.lower():
        return True
    # 파일명 heuristic (exp09_siglip_kd_*.pt 같은 distill 산출물)
    if model_path is not None and "siglip" in Path(model_path).name.lower():
        return True
    return False


def _load_torch_vit(model_path: Path,
                    force_cpu: bool = False,
                    tta: bool = False,
                    auto_face_crop: bool = True,
                    face_crop_margin: float = 0.1,
                    face_crop_min_size: int = 20,
                    tta_crops: str = "none",
                    tta_scales: Optional[List[int]] = None,
                    tta_hflip: bool = True) -> dict:
    """train_vit.py 의 timm ViT 체크포인트 load.  tta=True 면 hflip TTA 적용.
    auto_face_crop=True 면 MTCNN 으로 자동 얼굴 crop 후 resize (학습 조건 일치).
    tta=True 이고 tta_crops/tta_scales 확장 지정 시 multi-TTA 적용."""
    ckpt, args, meta, torch = _load_torch_checkpoint(model_path)
    try:
        import timm
    except ImportError as e:
        raise ImportError("timm 필요: pip install timm") from e

    # fullcombo 계열은 args["backbone"] 에 timm model_name 을 저장한다 (DINOv3 등).
    # 이전 계열은 args["model"] 또는 meta["model_name"] 을 썼으므로 폴백 체인을 함께 둔다.
    model_name = (meta.get("model_name")
                  or args.get("backbone")
                  or args.get("model")
                  or "vit_base_patch16_224")
    img_size = int(meta.get("img_size") or args.get("img_size") or 224)
    num_classes = int(len(meta.get("classes") or CLASSES))

    # DINOv3 / DINOv2 등 img_size 인자를 받는 백본은 학습 시 img_size 그대로 전달해야
    # positional embedding 이 맞는다. 일부 백본은 미지원 → TypeError fallback.
    try:
        model = timm.create_model(model_name, pretrained=False,
                                  num_classes=num_classes, img_size=img_size)
    except TypeError:
        model = timm.create_model(model_name, pretrained=False,
                                  num_classes=num_classes)
    state = ckpt.get("model")
    if state is None:
        raise RuntimeError("ckpt['model'] state_dict 가 없음.")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        # strict=False 로 넘어가지만 경고
        print(f"[warn] timm load state_dict unexpected keys: {len(unexpected)} (첫 3: {unexpected[:3]})",
              file=sys.stderr)

    if force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    # train_vit.py: Resize(short=256) → CenterCrop(224). ratio 유지.
    resize_short = max(256, int(round(img_size * (256 / 224))))

    def _resize_short_side(pil: Image.Image, short: int) -> Image.Image:
        w, h = pil.size
        if w <= h:
            new_w = short
            new_h = int(round(h * short / w))
        else:
            new_h = short
            new_w = int(round(w * short / h))
        return pil.resize((new_w, new_h), Image.BILINEAR)

    def _center_crop(pil: Image.Image, side: int) -> Image.Image:
        w, h = pil.size
        left = max(0, (w - side) // 2)
        top = max(0, (h - side) // 2)
        right = left + side
        bottom = top + side
        return pil.crop((left, top, right, bottom))

    def predict_fn(pil: Image.Image) -> np.ndarray:
        img = _resize_short_side(pil, resize_short)
        img = _center_crop(img, img_size)
        arr = np.asarray(img, dtype=np.float32) / 255.0   # (H, W, 3) [0,1]
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        arr = arr.transpose(2, 0, 1)                      # (3, H, W)
        tensor = torch.from_numpy(arr).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
        return probs.astype(np.float32)

    # MTCNN device: 모델이 cuda 에 있으면 MTCNN 도 cuda, 아니면 cpu
    mtcnn_device = "cuda" if device.type == "cuda" else "cpu"

    predict_fn_core = predict_fn
    predict_fn_base = predict_fn_core
    if auto_face_crop:
        predict_fn_base = _wrap_face_crop(
            predict_fn_core,
            margin=face_crop_margin,
            min_face_size=face_crop_min_size,
            device=mtcnn_device,
        )
    if tta:
        factory = _select_tta_wrapper(tta_crops, tta_scales, tta_hflip)
        predict_fn_public = factory(predict_fn_base) if factory else predict_fn_base
    else:
        predict_fn_public = predict_fn_base

    return {
        "predict_fn": predict_fn_public,
        "predict_fn_base": predict_fn_base,
        "predict_fn_raw": predict_fn_core,
        "tta": bool(tta),
        "auto_face_crop": bool(auto_face_crop),
        "type": f"torch_{model_name}"
                + ("_crop" if auto_face_crop else "")
                + ("_tta" if tta else ""),
        "meta": {
            "path": str(model_path),
            "model_name": model_name,
            "img_size": img_size,
            "device": str(device),
            "classes": meta.get("classes") or CLASSES,
            "tta": bool(tta),
            "tta_crops": str(tta_crops),
            "tta_scales": list(tta_scales) if tta_scales else None,
            "tta_hflip": bool(tta_hflip),
            "auto_face_crop": bool(auto_face_crop),
            "face_crop_margin": float(face_crop_margin),
            "face_crop_min_size": int(face_crop_min_size),
        },
        "_model": model,
        "_device": device,
    }


def _load_torch_siglip(model_path: Path,
                       force_cpu: bool = False,
                       tta: bool = False,
                       auto_face_crop: bool = True,
                       face_crop_margin: float = 0.1,
                       face_crop_min_size: int = 20,
                       tta_crops: str = "none",
                       tta_scales: Optional[List[int]] = None,
                       tta_hflip: bool = True) -> dict:
    """train_siglip.py 의 SigLIP wrapper 체크포인트 load.  tta=True 면 hflip TTA 적용.
    auto_face_crop=True 면 MTCNN 으로 자동 얼굴 crop 후 SiglipImageProcessor 입력.
    tta=True 이고 tta_crops/tta_scales 확장 지정 시 multi-TTA 적용."""
    ckpt, args, meta, torch = _load_torch_checkpoint(model_path)

    # models_custom import 를 위해 PROJECT 를 sys.path 에 올림
    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))

    try:
        from models_custom.vision_encoder import VisionEncoder, VisionConfig
    except ImportError as e:
        raise ImportError(
            "SigLIP 모델 추론을 위해 models_custom/vision_encoder.py 필요. "
            "프로젝트 루트에서 실행 중인지 확인."
        ) from e

    # 모든 VisionConfig 필드를 ckpt meta/args 에서 dynamic 추출 (state_dict shape mismatch 방지).
    # 미지정 필드는 train_siglip.py 기본값과 동일하게 fallback.
    def _pick(key, default):
        # meta 우선 → args → default
        if key in meta and meta[key] is not None:
            return meta[key]
        if key in args and args[key] is not None:
            return args[key]
        return default
    siglip_name = _pick("siglip_name", "google/siglip-base-patch16-384")
    projection_dim = int(_pick("projection_dim", 2048))
    use_pixel_shuffle = bool(_pick("use_pixel_shuffle", True))
    pixel_shuffle_ratio = int(_pick("pixel_shuffle_ratio", 4))
    projection_layers = int(_pick("projection_layers", 2))
    use_anyres = bool(_pick("use_anyres", False))
    use_ocr_augment = bool(_pick("use_ocr_augment", False))
    use_tile_position_embedding = bool(_pick("use_tile_position_embedding", False))
    img_size = int(_pick("img_size", 384))
    num_classes = int(len(meta.get("classes") or CLASSES))
    dropout = float(args.get("dropout") or 0.1)

    cfg = VisionConfig(
        encoder_name=siglip_name,
        image_size=img_size,
        freeze_encoder=True,           # inference 는 항상 freeze
        use_pixel_shuffle=use_pixel_shuffle,
        pixel_shuffle_ratio=pixel_shuffle_ratio,
        projection_layers=projection_layers,
        projection_dim=projection_dim,
        use_anyres=use_anyres,
        use_ocr_augment=use_ocr_augment,
        use_tile_position_embedding=use_tile_position_embedding,
    )
    enc = VisionEncoder(cfg)
    if force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc.load_siglip(device=str(device))

    # SiglipEmotionClassifier 을 여기 로컬로 재구성 (train_siglip 과 동일 구조)
    import torch.nn as nn

    class _SiglipEmotionClassifier(nn.Module):
        def __init__(self, vision_encoder, num_classes, dropout):
            super().__init__()
            self.vision_encoder = vision_encoder
            self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
            self.classifier = nn.Linear(vision_encoder.config.projection_dim, num_classes)

        def forward(self, pixel_values):
            embeds = self.vision_encoder.encode_image(pixel_values)
            if embeds.ndim != 3:
                raise RuntimeError(f"encode_image shape invalid: {tuple(embeds.shape)}")
            pooled = embeds.mean(dim=1)
            pooled = self.dropout(pooled)
            return self.classifier(pooled)

    model = _SiglipEmotionClassifier(enc, num_classes=num_classes, dropout=dropout)
    state = ckpt.get("model")
    if state is None:
        raise RuntimeError("ckpt['model'] state_dict 가 없음.")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f"[warn] siglip load unexpected keys: {len(unexpected)} (첫 3: {unexpected[:3]})",
              file=sys.stderr)

    model.to(device).eval()
    processor = enc.processor
    if processor is None:
        raise RuntimeError("SiglipImageProcessor load 실패.")

    def predict_fn(pil: Image.Image) -> np.ndarray:
        # SiglipImageProcessor 가 resize + normalize 전부 담당 (384×384 기본)
        inputs = processor(images=[pil], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            logits = model(pixel_values)
            probs = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
        return probs.astype(np.float32)

    mtcnn_device = "cuda" if device.type == "cuda" else "cpu"

    predict_fn_core = predict_fn
    predict_fn_base = predict_fn_core
    if auto_face_crop:
        predict_fn_base = _wrap_face_crop(
            predict_fn_core,
            margin=face_crop_margin,
            min_face_size=face_crop_min_size,
            device=mtcnn_device,
        )
    if tta:
        factory = _select_tta_wrapper(tta_crops, tta_scales, tta_hflip)
        predict_fn_public = factory(predict_fn_base) if factory else predict_fn_base
    else:
        predict_fn_public = predict_fn_base

    return {
        "predict_fn": predict_fn_public,
        "predict_fn_base": predict_fn_base,
        "predict_fn_raw": predict_fn_core,
        "tta": bool(tta),
        "auto_face_crop": bool(auto_face_crop),
        "type": f"torch_siglip({siglip_name.rsplit('/',1)[-1]})"
                + ("_crop" if auto_face_crop else "")
                + ("_tta" if tta else ""),
        "meta": {
            "path": str(model_path),
            "siglip_name": siglip_name,
            "projection_dim": projection_dim,
            "use_pixel_shuffle": use_pixel_shuffle,
            "pixel_shuffle_ratio": pixel_shuffle_ratio,
            "img_size": img_size,
            "device": str(device),
            "classes": meta.get("classes") or CLASSES,
            "tta": bool(tta),
            "tta_crops": str(tta_crops),
            "tta_scales": list(tta_scales) if tta_scales else None,
            "tta_hflip": bool(tta_hflip),
            "auto_face_crop": bool(auto_face_crop),
            "face_crop_margin": float(face_crop_margin),
            "face_crop_min_size": int(face_crop_min_size),
        },
        "_model": model,
        "_device": device,
    }


def _load_torch(model_path: Path,
                force_cpu: bool = False,
                tta: bool = False,
                auto_face_crop: bool = True,
                face_crop_margin: float = 0.1,
                face_crop_min_size: int = 20,
                tta_crops: str = "none",
                tta_scales: Optional[List[int]] = None,
                tta_hflip: bool = True) -> dict:
    """torch .pt 체크포인트 — ViT / SigLIP 자동 감지.  tta / auto_face_crop flag 전달."""
    ckpt, args, meta, _torch = _load_torch_checkpoint(model_path)
    if _is_siglip_ckpt(args, meta, model_path=model_path):
        return _load_torch_siglip(
            model_path, force_cpu=force_cpu, tta=tta,
            auto_face_crop=auto_face_crop,
            face_crop_margin=face_crop_margin,
            face_crop_min_size=face_crop_min_size,
            tta_crops=tta_crops, tta_scales=tta_scales, tta_hflip=tta_hflip,
        )
    # 기본: ViT
    return _load_torch_vit(
        model_path, force_cpu=force_cpu, tta=tta,
        auto_face_crop=auto_face_crop,
        face_crop_margin=face_crop_margin,
        face_crop_min_size=face_crop_min_size,
        tta_crops=tta_crops, tta_scales=tta_scales, tta_hflip=tta_hflip,
    )


# ==========================================================================
# 앙상블 (.json)
# ==========================================================================

def _load_ensemble(cfg_path: Path,
                   force_cpu: bool = False,
                   tta: bool = False,
                   auto_face_crop: bool = True,
                   face_crop_margin: float = 0.1,
                   face_crop_min_size: int = 20,
                   tta_crops: str = "none",
                   tta_scales: Optional[List[int]] = None,
                   tta_hflip: bool = True) -> dict:
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    members = cfg.get("models")
    if not members or not isinstance(members, list):
        raise ValueError(f"ensemble config 에 'models' 리스트 필요: {cfg_path}")

    loaded = []
    base_dir = cfg_path.parent
    for i, m in enumerate(members):
        if not isinstance(m, dict):
            raise ValueError(f"ensemble models[{i}] 는 dict 여야 함.")
        p = m.get("path")
        if not p:
            raise ValueError(f"ensemble models[{i}] 에 'path' 필요.")
        model_path = Path(p)
        if not model_path.is_absolute():
            # 프로젝트 루트 기준으로 먼저 해석 (일반적으로 "models/..." 상대 경로)
            candidate = PROJECT / model_path
            if not candidate.is_file():
                # fallback: json 파일 위치 기준
                candidate = base_dir / model_path
            model_path = candidate
        if not model_path.is_file():
            raise FileNotFoundError(f"ensemble member {i} 파일 없음: {model_path}")
        weight = float(m.get("weight", 1.0))
        # MTCNN + TTA 는 ensemble 레벨에서 1회만 수행. sub 는 raw core (crop/tta off) 로 load.
        # 같은 cropped PIL 을 모든 sub 가 공유 → MTCNN N회 중복 제거.
        sub = _load_single(
            model_path,
            force_cpu=force_cpu,
            tta=False,
            auto_face_crop=False,
            face_crop_margin=face_crop_margin,
            face_crop_min_size=face_crop_min_size,
            tta_crops="none",
            tta_scales=None,
            tta_hflip=False,
        )
        sub["_weight"] = weight
        loaded.append(sub)
        print(
            f"[ensemble] [{i}] loaded {model_path.name}  type={sub['type']}  "
            f"w={weight}  (sub tta/crop OFF — ensemble 에서 1회 공유)",
            file=sys.stderr,
        )

    def predict_fn_core(pil: Image.Image) -> np.ndarray:
        """ensemble weighted sum. pil 은 이미 cropped + TTA-wrapped 상태에서 들어옴."""
        agg = np.zeros(NUM_CLASSES, dtype=np.float64)
        total_w = 0.0
        for sub in loaded:
            probs = sub["predict_fn_raw"](pil)
            probs = np.asarray(probs, dtype=np.float64).reshape(-1)
            if probs.shape[0] != NUM_CLASSES:
                raise RuntimeError(
                    f"ensemble member {sub['type']} 가 {probs.shape[0]} classes 반환 "
                    f"(expected {NUM_CLASSES})"
                )
            w = float(sub.get("_weight", 1.0))
            agg += probs * w
            total_w += w
        if total_w <= 0:
            raise ValueError("ensemble 총 weight 가 0 이하.")
        agg = agg / total_w
        # softmax 재정규화 (각 member 가 softmax 결과라면 이미 합=1, weighted avg 도 합=1.
        # 하지만 float 오차 누적 방어)
        s = float(agg.sum())
        if not np.isfinite(s) or abs(s - 1.0) > 1e-3:
            agg = _softmax_np(agg.astype(np.float32))
        return agg.astype(np.float32)

    # ensemble 레벨 공통 wrap: face crop 1회 → TTA 1회
    # force_cpu=True 면 무조건 CPU. 아니면 torch sub 중 GPU 쓰는 게 있으면 GPU, 없으면 CPU.
    if force_cpu:
        mtcnn_device = "cpu"
    else:
        mtcnn_device = "cpu"
        for sub in loaded:
            dev = sub.get("_device")
            if dev is not None and getattr(dev, "type", None) == "cuda":
                mtcnn_device = "cuda"
                break

    predict_fn_base = predict_fn_core
    if auto_face_crop:
        predict_fn_base = _wrap_face_crop(
            predict_fn_core,
            margin=face_crop_margin,
            min_face_size=face_crop_min_size,
            device=mtcnn_device,
        )
    if tta:
        factory = _select_tta_wrapper(tta_crops, tta_scales, tta_hflip)
        predict_fn_public = factory(predict_fn_base) if factory else predict_fn_base
    else:
        predict_fn_public = predict_fn_base

    return {
        "predict_fn": predict_fn_public,
        "predict_fn_base": predict_fn_base,   # TTA 없이 (crop 만) — predict() override 용
        "predict_fn_raw": predict_fn_core,    # crop/tta 전부 없는 원본 (debug)
        "tta": bool(tta),
        "auto_face_crop": bool(auto_face_crop),
        "type": f"ensemble(n={len(loaded)})"
                + ("_crop" if auto_face_crop else "")
                + ("_tta" if tta else ""),
        "meta": {
            "path": str(cfg_path),
            "members": [{"path": str(sub["meta"].get("path")),
                         "type": sub["type"],
                         "weight": sub.get("_weight", 1.0)} for sub in loaded],
            "tta": bool(tta),
            "tta_crops": str(tta_crops),
            "tta_scales": list(tta_scales) if tta_scales else None,
            "tta_hflip": bool(tta_hflip),
            "auto_face_crop": bool(auto_face_crop),
            "face_crop_margin": float(face_crop_margin),
            "face_crop_min_size": int(face_crop_min_size),
        },
        "_members": loaded,
    }


# ==========================================================================
# public API
# ==========================================================================

def _torch_cuda_available() -> bool:
    """torch 가 설치돼 있고 cuda 쓸 수 있으면 True. 미설치여도 예외 안 남."""
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _load_single(model_path: Path,
                 force_cpu: bool = False,
                 tta: bool = False,
                 auto_face_crop: bool = True,
                 face_crop_margin: float = 0.1,
                 face_crop_min_size: int = 20,
                 tta_crops: str = "none",
                 tta_scales: Optional[List[int]] = None,
                 tta_hflip: bool = True) -> dict:
    """단일 모델 load (.h5 / .pt). ensemble 이 내부적으로 호출.
    tta=True 시 각 _load_* 가 hflip TTA wrapper 적용.
    tta_crops/tta_scales 지정 시 multi-TTA 로 승격.
    auto_face_crop=True 시 MTCNN face crop wrapper 추가."""
    suf = model_path.suffix.lower()
    # TF 는 force_cpu 를 직접 못 받음 → TF 쪽 MTCNN device 결정 위해 힌트 전달.
    tf_crop_device = "cpu" if force_cpu else ("cuda"
                                              if _torch_cuda_available()
                                              else "cpu")
    if suf == ".h5":
        return _load_tf(
            model_path,
            tta=tta,
            auto_face_crop=auto_face_crop,
            face_crop_margin=face_crop_margin,
            face_crop_min_size=face_crop_min_size,
            face_crop_device=tf_crop_device,
            tta_crops=tta_crops, tta_scales=tta_scales, tta_hflip=tta_hflip,
        )
    if suf in (".pt", ".pth"):
        return _load_torch(
            model_path,
            force_cpu=force_cpu,
            tta=tta,
            auto_face_crop=auto_face_crop,
            face_crop_margin=face_crop_margin,
            face_crop_min_size=face_crop_min_size,
            tta_crops=tta_crops, tta_scales=tta_scales, tta_hflip=tta_hflip,
        )
    raise ValueError(f"지원하지 않는 확장자: {suf} (예상: .h5, .pt, .json)")


def load_model(path: Union[str, Path],
               device: str = "auto",
               tta: bool = False,
               auto_face_crop: bool = True,
               face_crop_margin: float = 0.1,
               face_crop_min_size: int = 20,
               tta_crops: str = "none",
               tta_scales: Optional[List[int]] = None,
               tta_hflip: bool = True) -> dict:
    """모델 로드.

    Args:
        path: .h5 (TF) / .pt (torch ViT or SigLIP) / .json (ensemble config)
        device: 'auto' | 'cuda' | 'cpu'. torch 모델 load 시 실제 반영됨.
                'cpu' 면 CUDA 가능해도 CPU 강제. TF 는 자체 GPU 관리 (TF visible env 로 따로).
        tta: True 면 hflip Test-Time Augmentation (원본+좌우반전 softmax 평균).
             추론 2배 느려지지만 소폭 정확도 ↑ (+0.3~0.7%p). 앙상블에도 전파됨.
             False 면 tta_crops/tta_scales 값은 무시.
        auto_face_crop: True (기본) 면 MTCNN 으로 자동 얼굴 detect + crop 후 resize.
             학습 시 --crop 을 쓴 모델은 이걸 반드시 켜야 val 조건 재현 가능.
             얼굴 detect 실패 시 원본 이미지로 graceful fallback.
        face_crop_margin: crop bbox 확장 비율 (0.1 = 10%). 학습 bbox 가 약간 여유 있음 반영.
        face_crop_min_size: MTCNN min_face_size 인자 (작은 얼굴까지 잡으려면 ↓).
        tta_crops: "none" (기본, view 수 불변) | "5crop" (중심+4모서리).
             tta=True 인 경우에만 의미 있음.
        tta_scales: 각 scale (정수) 로 이미지를 (s, s) 로 resize 후 crop 적용.
             None 또는 [] 또는 단일값 → 원본 resize 단계 생략 (predict_fn 내부 resize 만).
             예: [224, 256] → 2개 scale, len>1 이면 multi-TTA 활성.
        tta_hflip: True (기본) 면 각 view 별 좌우반전 추가. tta=True 일 때만 영향.

    view 수 = max(1, len(tta_scales_or_1)) × (5 if 5crop else 1) × (2 if tta_hflip else 1)

    기본값 호환: tta=True, tta_crops="none", tta_scales=None, tta_hflip=True
    → 기존 hflip TTA 와 완전 동일 (view 2개).

    Returns:
        dict:
          predict_fn: Callable[[PIL.Image], np.ndarray(shape=(4,), dtype=float32)]
          type:      str (e.g. 'tf_resnet50_crop_tta', 'torch_vit_base_patch16_224_crop', ...)
          meta:      dict (모델 타입별 정보)
    """
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"모델 경로 없음: {model_path}")

    if device not in ("auto", "cuda", "cpu"):
        raise ValueError(f"device 는 'auto'|'cuda'|'cpu' 여야 함: {device!r}")
    force_cpu = (device == "cpu")

    if float(face_crop_margin) < 0:
        raise ValueError(f"face_crop_margin 은 0 이상: {face_crop_margin!r}")
    if int(face_crop_min_size) < 8:
        raise ValueError(f"face_crop_min_size 는 8 이상: {face_crop_min_size!r}")

    # multi-TTA 인자 validation (tta=False 여도 미리 체크: 잘못된 값 조용히 넘기지 않음)
    tc = (tta_crops or "none").lower()
    if tc not in ("none", "5crop"):
        raise ValueError(f"tta_crops 는 'none'|'5crop': {tta_crops!r}")
    if tta_scales is not None:
        if not isinstance(tta_scales, (list, tuple)):
            raise ValueError(f"tta_scales 는 list[int] 여야 함: {tta_scales!r}")
        tta_scales = [int(s) for s in tta_scales]
        for s in tta_scales:
            if s <= 0:
                raise ValueError(f"tta_scales 값은 양의 정수: {tta_scales!r}")

    suf = model_path.suffix.lower()
    if suf == ".json":
        return _load_ensemble(
            model_path,
            force_cpu=force_cpu,
            tta=tta,
            auto_face_crop=auto_face_crop,
            face_crop_margin=face_crop_margin,
            face_crop_min_size=face_crop_min_size,
            tta_crops=tc, tta_scales=tta_scales, tta_hflip=bool(tta_hflip),
        )

    return _load_single(
        model_path,
        force_cpu=force_cpu,
        tta=tta,
        auto_face_crop=auto_face_crop,
        face_crop_margin=face_crop_margin,
        face_crop_min_size=face_crop_min_size,
        tta_crops=tc, tta_scales=tta_scales, tta_hflip=bool(tta_hflip),
    )


def _choose_predict_fn(model_obj: dict, tta: Optional[bool]) -> Callable[[Image.Image], np.ndarray]:
    """predict() 시점 tta override 를 반영하여 실제 쓸 predict_fn 선택.

    tta=None  → load_model() 시 세팅(model_obj['predict_fn']) 그대로.
    tta=True  → 무조건 TTA 적용. base predict_fn(있으면) 위에 hflip wrap.
                 ensemble 의 경우 load 시 tta off 였으면 ensemble 레벨에서 hflip 씌움
                 (sub 전파는 load 시점에만 가능).
    tta=False → TTA 없이. load 가 TTA 로 돼 있으면 base 로 되돌림.
    """
    if tta is None:
        return model_obj["predict_fn"]
    base = model_obj.get("predict_fn_base") or model_obj["predict_fn"]
    if tta:
        return _wrap_tta(base)
    return base


def predict_probs(model_obj: dict, image: ImageLike, tta: Optional[bool] = None) -> np.ndarray:
    """단일 이미지 → (4,) softmax probabilities. 앙상블도 투명하게 지원.

    tta: None → load_model 시 세팅 따름. True/False 면 이 호출에서 강제 override.
    """
    if not isinstance(model_obj, dict) or "predict_fn" not in model_obj:
        raise TypeError(
            "model_obj 는 load_model() 반환 dict 여야 함. "
            f"got type={type(model_obj).__name__}"
        )
    pil = _load_pil(image)
    fn = _choose_predict_fn(model_obj, tta)
    probs = fn(pil)
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    if probs.shape[0] != NUM_CLASSES:
        raise RuntimeError(
            f"predict_fn 가 {probs.shape[0]} 차원 반환 (expected {NUM_CLASSES})"
        )
    return probs


def predict(model_obj: dict, image: ImageLike, tta: Optional[bool] = None) -> tuple[str, float]:
    """단일 이미지 → (label, confidence).

    label: CLASSES 중 하나
    confidence: argmax 클래스의 softmax 확률 (float, 0~1)
    tta: None → load_model 시 세팅, True/False → 호출 레벨 override.
    """
    probs = predict_probs(model_obj, image, tta=tta)
    idx = int(np.argmax(probs))
    return CLASSES[idx], float(probs[idx])


# ==========================================================================
# Batch API — 여러 장 한 번에 처리 (튜터/외부 평가자가 3~6장 보내는 경우 대비)
# ==========================================================================

def predict_probs_batch(model_obj: dict,
                        images: list,
                        tta: Optional[bool] = None) -> np.ndarray:
    """여러 장 → probabilities (N, NUM_CLASSES).

    images: list[str | Path | PIL.Image]
    내부적으로 predict_probs 순차 호출 (단일 이미지 predict_fn 래핑).
    앙상블 / TTA / face crop 전부 그대로 전파.

    Returns: np.ndarray shape (N, 4), sum=1 각 row.
    Raises:
      TypeError — images 가 list 아니거나 비어있음
      RuntimeError — 개별 predict 실패 시 즉시 중단 (제1헌법)
    """
    if not isinstance(images, (list, tuple)):
        raise TypeError(
            f"images must be list/tuple, got {type(images).__name__}"
        )
    if len(images) == 0:
        raise ValueError("images 가 비어있음")
    probs_list = []
    for i, img in enumerate(images):
        p = predict_probs(model_obj, img, tta=tta)
        probs_list.append(p)
    probs = np.stack(probs_list, axis=0).astype(np.float32)
    if probs.shape != (len(images), NUM_CLASSES):
        raise RuntimeError(
            f"batch predict shape mismatch: {probs.shape} "
            f"(expected ({len(images)},{NUM_CLASSES}))"
        )
    return probs


# ==========================================================================
# Multi-face API — 한 이미지에 얼굴 여러 개인 경우 각각 분류
# ==========================================================================

def detect_all_faces(image: ImageLike,
                     margin: float = 0.1,
                     min_face_size: int = 20,
                     min_conf: float = 0.9,
                     device: str = "auto") -> list:
    """모든 얼굴 검출 + margin crop 한 PIL 리스트 반환.
    얼굴 0건이면 빈 리스트. (classifier 에 전체 이미지 fallback 은 호출자 책임)

    Args:
        image: str/Path/PIL.Image
        margin: bbox 확장 비율 (0.1 = 10%)
        min_face_size: MTCNN min_face_size
        min_conf: 얼굴 confidence 하한 (0.9 추천)
        device: 'auto'|'cuda'|'cpu'
    """
    import math  # top-level import 없음 (lazy)
    import torch
    from facenet_pytorch import MTCNN
    pil = _load_pil(image)
    dev = device if device != "auto" else \
          ("cuda" if torch.cuda.is_available() else "cpu")
    # keep_all=True 전용 인스턴스 (기존 single-face MTCNN 캐시와 분리)
    mtcnn = MTCNN(
        image_size=224,
        margin=max(0, int(margin * 224)),
        min_face_size=int(min_face_size),
        thresholds=[0.6, 0.7, 0.7],
        keep_all=True,
        device=dev,
        post_process=False,
    )
    W, H = pil.size
    boxes, probs = mtcnn.detect(pil)
    if boxes is None or len(boxes) == 0:
        return []
    faces = []
    for box, conf in zip(boxes, probs):
        if conf is None or conf < min_conf:
            continue
        x0, y0, x1, y1 = map(float, box)
        bw = max(1.0, x1 - x0)
        bh = max(1.0, y1 - y0)
        mx = bw * margin
        my = bh * margin
        x0 = max(0.0, x0 - mx)
        y0 = max(0.0, y0 - my)
        x1 = min(float(W), x1 + mx)
        y1 = min(float(H), y1 + my)
        if (x1 - x0) <= 1 or (y1 - y0) <= 1:
            continue
        faces.append({
            "crop": pil.crop((int(x0), int(y0),
                              int(math.ceil(x1)), int(math.ceil(y1)))),
            "bbox": (x0, y0, x1, y1),
            "confidence": float(conf),
        })
    return faces


def predict_multi_face(model_obj: dict,
                       image: ImageLike,
                       tta: Optional[bool] = None,
                       min_conf: float = 0.9,
                       margin: float = 0.1,
                       min_face_size: int = 20) -> list:
    """여러 얼굴이 있는 이미지 → 각 얼굴별 (label, confidence, bbox, probs) 리스트.

    얼굴 0건 (MTCNN 실패) 시 **전체 이미지 fallback** 1건 반환 (기존 single predict 동작 유지).
    """
    faces = detect_all_faces(image, margin=margin,
                             min_face_size=min_face_size,
                             min_conf=min_conf)
    if not faces:
        # fallback: 전체 이미지 단일 분류
        probs = predict_probs(model_obj, image, tta=tta)
        idx = int(np.argmax(probs))
        return [{
            "label": CLASSES[idx],
            "confidence": float(probs[idx]),
            "bbox": None,
            "face_confidence": None,
            "probs": {c: float(p) for c, p in zip(CLASSES, probs)},
            "fallback": True,
        }]
    out = []
    for f in faces:
        probs = predict_probs(model_obj, f["crop"], tta=tta)
        idx = int(np.argmax(probs))
        out.append({
            "label": CLASSES[idx],
            "confidence": float(probs[idx]),
            "bbox": [round(v, 2) for v in f["bbox"]],
            "face_confidence": round(f["confidence"], 4),
            "probs": {c: float(p) for c, p in zip(CLASSES, probs)},
            "fallback": False,
        })
    return out


def predict_batch(model_obj: dict,
                  images: list,
                  tta: Optional[bool] = None) -> list[tuple[str, float]]:
    """여러 장 → list[(label, confidence)].

    사용 예)
        model = load_model("models/exp05_vit_b16_two_stage.pt")
        results = predict_batch(model, ["a.jpg", "b.jpg", "c.jpg"])
        for (lbl, conf), path in zip(results, paths):
            print(f"{path}: {lbl} ({conf:.3f})")
    """
    probs = predict_probs_batch(model_obj, images, tta=tta)
    out = []
    for row in probs:
        idx = int(np.argmax(row))
        out.append((CLASSES[idx], float(row[idx])))
    return out


# ==========================================================================
# CLI
# ==========================================================================

def _cli():
    ap = argparse.ArgumentParser(
        description="emotion-project 감정 4분류 추론 (anger/happy/panic/sadness)"
    )
    ap.add_argument("--model", required=True,
                    help="모델 경로: .h5 (TF) / .pt (torch ViT/SigLIP) / .json (앙상블)")
    # 이미지 입력 — 3 모드 중 하나 (mutually exclusive)
    img_group = ap.add_mutually_exclusive_group(required=True)
    img_group.add_argument("--image", help="단일 이미지 경로")
    img_group.add_argument("--images", nargs="+",
                           help="여러 이미지 경로 (공백 구분). 예: --images a.jpg b.jpg c.jpg")
    img_group.add_argument("--image-dir",
                           help="디렉터리 — 하위의 .jpg/.jpeg/.png 전부 추론")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu"],
                    help="torch 쪽만 영향. 기본 'auto'.")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="stderr 에 모델 메타 출력")
    ap.add_argument("--tta", action="store_true",
                    help="Test-Time Augmentation (hflip 평균). 추론 2배 느려지나 +0.3~0.7%%p 기대.")
    ap.add_argument("--auto-face-crop", default="on", choices=["on", "off", "auto"],
                    help="MTCNN 자동 얼굴 crop. on=항상 ON (기본, 외부 평가 환경 권장 / 학습 조건 일치), "
                         "off=전체 이미지 사용, auto=on 과 동일 (MTCNN 실패 시 자동 fallback 은 항상 적용됨).")
    ap.add_argument("--face-crop-margin", type=float, default=0.1,
                    help="face crop bbox 확장 비율 (default 0.1 = 10%%).")
    ap.add_argument("--face-crop-min-size", type=int, default=20,
                    help="MTCNN min_face_size (default 20).")
    # multi-TTA 확장. --tta 가 on 이어야 실제로 적용됨.
    ap.add_argument("--tta-crops", default="none", choices=["none", "5crop"],
                    help="TTA crop 전략. none (기본) | 5crop (중심+4모서리). "
                         "--tta 와 같이 써야 효과.")
    ap.add_argument("--tta-scales", default="",
                    help="TTA scale CSV (예: '224,256'). 각 scale 로 (s,s) 리사이즈 후 crop. "
                         "빈 문자열 또는 단일값이면 scale 확장 없음. --tta 와 같이 써야 효과.")
    ap.add_argument("--tta-hflip", default="on", choices=["on", "off"],
                    help="각 view 별 좌우반전 추가 여부. on (기본) | off. "
                         "기존 --tta 와 동일한 hflip 동작. --tta 가 off 이면 무시.")
    ap.add_argument("--multi-face", action="store_true",
                    help="이미지 1장에 얼굴 여러 개 검출 + 각각 분류. "
                         "얼굴 0건일 땐 전체 이미지 fallback 1건 반환.")
    ap.add_argument("--multi-min-conf", type=float, default=0.9,
                    help="multi-face 모드의 얼굴 검출 confidence 하한.")
    args = ap.parse_args()

    auto_face_crop = (args.auto_face_crop in ("on", "auto"))

    # --tta-scales CSV → List[int] (빈값은 None)
    tta_scales_parsed: Optional[List[int]] = None
    if args.tta_scales and args.tta_scales.strip():
        try:
            tta_scales_parsed = [int(s.strip()) for s in args.tta_scales.split(",")
                                 if s.strip()]
        except ValueError as e:
            ap.error(f"--tta-scales 파싱 실패 ({args.tta_scales!r}): {e}")
        if tta_scales_parsed and any(s <= 0 for s in tta_scales_parsed):
            ap.error(f"--tta-scales 값은 양의 정수: {tta_scales_parsed}")
    tta_hflip_bool = (args.tta_hflip == "on")

    model_obj = load_model(
        args.model,
        device=args.device,
        tta=args.tta,
        auto_face_crop=auto_face_crop,
        face_crop_margin=args.face_crop_margin,
        face_crop_min_size=args.face_crop_min_size,
        tta_crops=args.tta_crops,
        tta_scales=tta_scales_parsed,
        tta_hflip=tta_hflip_bool,
    )
    if args.verbose:
        print(f"[i] type={model_obj['type']}  meta={model_obj.get('meta')}",
              file=sys.stderr)

    # 이미지 목록 수집 (단일 / 여러 / 디렉터리)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if args.image:
        image_paths = [args.image]
    elif args.images:
        image_paths = list(args.images)
    else:  # args.image_dir
        d = Path(args.image_dir)
        if not d.is_dir():
            ap.error(f"--image-dir 디렉터리 아님: {d}")
        image_paths = sorted(
            str(p) for p in d.rglob("*")
            if p.is_file() and p.suffix.lower() in image_exts
        )
        if not image_paths:
            ap.error(f"--image-dir 에 이미지 없음: {d}")

    # --multi-face 모드 (단일 이미지에 얼굴 여러 개)
    if args.multi_face:
        if len(image_paths) != 1:
            ap.error("--multi-face 는 --image 단일 경로와만 조합 가능")
        results = predict_multi_face(
            model_obj, image_paths[0],
            min_conf=args.multi_min_conf,
            margin=args.face_crop_margin,
            min_face_size=args.face_crop_min_size,
        )
        print(json.dumps({
            "image": image_paths[0],
            "n_faces": sum(1 for r in results if not r.get("fallback")),
            "fallback_used": any(r.get("fallback") for r in results),
            "faces": results,
        }, ensure_ascii=False))
        return

    if len(image_paths) == 1:
        # 단일 — 기존 출력 포맷 유지 (backward compatible)
        probs = predict_probs(model_obj, image_paths[0])
        idx = int(np.argmax(probs))
        out = {
            "label": CLASSES[idx],
            "confidence": round(float(probs[idx]), 6),
            "probs": {c: round(float(p), 6) for c, p in zip(CLASSES, probs)},
            "model_type": model_obj["type"],
        }
        print(json.dumps(out, ensure_ascii=False))
    else:
        # 여러 장 — JSON list (한 줄씩 ndjson)
        probs_mat = predict_probs_batch(model_obj, image_paths)
        for path, probs in zip(image_paths, probs_mat):
            idx = int(np.argmax(probs))
            out = {
                "image": path,
                "label": CLASSES[idx],
                "confidence": round(float(probs[idx]), 6),
                "probs": {c: round(float(p), 6) for c, p in zip(CLASSES, probs)},
                "model_type": model_obj["type"],
            }
            print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    _cli()
