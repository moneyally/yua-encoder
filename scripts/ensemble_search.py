"""emotion-project / scripts/ensemble_search.py

여러 훈련된 모델(.h5/.pt) 의 val set softmax 예측을 수집해서 **최적 앙상블 방식을
자동 탐색** 하고, 최종 json config + markdown report 를 출력한다.

================================================================================
파이프라인
================================================================================

1. argparse 로 모델 경로 리스트 + --val-dir + --output-* 받음
2. 각 모델 load (predict.py 의 load_model 재사용) — .h5 / .pt 자동 분기
3. val 전체 이미지 N 장 softmax probs 추출 → cache_dir/<name>_val_probs.npz 캐시
   - probs cache 있으면 스킵 (재실행 빠름)
4. 여러 앙상블 방법 비교 및 val_acc / val_macro_f1 / val_NLL 측정:
     (A) 단일 모델 val_acc — 기준점 (per-model)
     (B) Uniform soft voting (weight 1.0 씩)
     (C) Per-model Temperature Scaling (NLL 최소화, scipy.optimize.minimize_scalar)
     (D) (C) + Weight optimization (scipy.optimize.differential_evolution, bounds=(0,1))
     (E) Stacking — sklearn LogisticRegression (multinomial) with 5-fold out-of-fold
         (옵션, --skip-stacking 로 끄기 가능)
5. 최고 val_acc 방법 선택 → models/ensemble_best.json (predict.py 포맷 호환)
6. results/ensemble_report.md 작성 (방법별 표 + 최종 추천)

================================================================================
수식 및 정합성 증명 (주석으로 남김)
================================================================================

[Temperature Scaling (Guo 2017)]
softmax(z)_i = exp(z_i) / Σ exp(z_j)  =  p_i
TS: softmax(z/T)_i = exp(z_i/T) / Σ exp(z_j/T)

주장: 우리는 probs (p_i, 합=1) 만 갖고 있어도 TS 결과를 정확히 복원 가능.
증명:
    z_i = log(p_i) + C  (C는 상수, softmax 는 shift-invariant)
    z_i / T = (log(p_i) + C) / T = log(p_i)/T + C/T
    softmax 는 상수 shift 에 invariant 이므로 C/T 제거:
        softmax(z/T)_i = exp(log(p_i)/T) / Σ exp(log(p_j)/T)
                       = p_i^(1/T)      / Σ p_j^(1/T)
    □
즉 `probs_T(p, T) = p^(1/T) / sum(p^(1/T))` 식이면 충분.
이 코드의 `apply_temperature()` 가 바로 이 식을 쓴다 (clip 1e-12 for log-safety).

[NLL]
NLL(T) = - (1/N) Σ log p_{y_i, after T}  (y_i 는 true class)
scipy.optimize.minimize_scalar(..., bounds=(0.05, 10), method="bounded") 로 1D 탐색.

[Weight optimization]
목적함수: 1 - val_acc(softmax avg with normalized weights)
각 모델의 TS-적용 probs 를 w_k 가중치로 합산 후 argmax.
    agg_i = Σ_k (w_k / Σ w) * p_k,i
경계: w_k ∈ [0, 1]. (모두 0 방지: 함수 안에서 sum ≤ 1e-8 이면 penalty)
differential_evolution(seed=SEED) — 전역 탐색.

[Stacking (옵션)]
5-fold StratifiedKFold. 각 fold 에서 train-fold probs concat 으로 LR fit,
val-fold 에 predict → OOF probs. 마지막으로 OOF 기반 val_acc 측정.
(실제 inference-time LR 배포는 predict.py 확장 필요하므로 보고용 수치만.)

================================================================================
JSON 출력 포맷 (predict.py 와 호환)
================================================================================

{
  "_method": "<best method name>",
  "_val_acc": float,
  "_val_macro_f1": float,
  "_val_nll": float,
  "_per_model": {<name>: {"val_acc": ..., "val_macro_f1": ..., "temperature": ...}},
  "_notes": "...",
  "models": [
     {"path": "models/...", "weight": float, "temperature": float},
     ...
  ]
}

predict.py 는 현재 `weight` 키만 사용하고 `temperature` 는 무시한다.
기본 모드는 `--no-apply-ts` 로, TS 는 탐색 보조(NLL 판단)만 하고
최종 json 의 `weight` 은 raw softmax weighted voting 기준으로 최적화한다
(= predict.py 실 배포 정합). `--apply-ts` 를 켜면 TS 적용 상태에서
weight 를 최적화해 temperature 도 함께 기록하지만 이 경우 predict.py 확장 필요.

================================================================================
CLI 예시
================================================================================

python scripts/ensemble_search.py \\
  --models models/exp02_resnet50_ft_crop_aug.h5 \\
           models/exp04_effnet_ft_balanced.h5 \\
           models/exp05_vit_b16_two_stage.pt \\
           models/exp06_siglip_linear_probe.pt \\
  --val-dir data_rot/img/val \\
  --output-config models/ensemble_best.json \\
  --output-report results/ensemble_report.md \\
  --cache-dir results/ensemble_cache \\
  --seed 42

옵션:
  --force-cpu        : TF/torch CPU 강제 (GPU 경쟁 회피)
  --skip-stacking    : LR 스택 방법 건너뛰기
  --apply-ts         : 최종 config 에 TS 포함 (predict.py 확장 필요)
  --de-maxiter 60    : differential_evolution 반복 수
  --limit-per-class N: 클래스당 최대 N장만 (빠른 디버그용)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ------------------------------------------------------------------
# 프로젝트 루트 자동 결정 (CLAUDE.md #7: 하드코딩 금지)
# ------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
# scripts/ 상위가 프로젝트 루트
PROJECT = Path(os.environ.get("EMOTION_PROJECT_ROOT", str(THIS_FILE.parent.parent)))
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

# predict.py 는 프로젝트 루트에 있음. 재사용.
import predict as predict_mod  # noqa: E402  (sys.path 세팅 후 import)

CLASSES: List[str] = list(predict_mod.CLASSES)
NUM_CLASSES: int = predict_mod.NUM_CLASSES
assert NUM_CLASSES == 4 and CLASSES == ["anger", "happy", "panic", "sadness"], (
    f"predict.CLASSES 가 기대와 다름: {CLASSES}"
)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ------------------------------------------------------------------
# 로깅
# ------------------------------------------------------------------

def _log(msg: str) -> None:
    print(f"[ensemble_search] {msg}", file=sys.stderr, flush=True)


# ------------------------------------------------------------------
# val set 수집
# ------------------------------------------------------------------

def _load_val_bboxes(label_root: Path, split: str = "val") -> dict:
    """data_rot/label/{split}/<split>_<cls>.json 로드 → {(class, filename): bbox}.

    학습 때 `--crop` 으로 얼굴 bbox 를 잘라 학습했으므로, 앙상블 평가도
    동일한 crop 조건에서 해야 학습 성능과 일치한다 (predict.py 기본은 no-crop).
    key 는 (class, filename) 튜플 — 클래스 간 동명이인 충돌 방어.
    """
    bbox_by_key: dict = {}
    for cls in CLASSES:
        p = label_root / split / f"{split}_{cls}.json"
        if not p.is_file():
            continue
        loaded = None
        for enc in ("euc-kr", "utf-8"):
            try:
                with open(p, encoding=enc) as f:
                    loaded = json.load(f)
                break
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        for item in loaded or []:
            if not isinstance(item, dict):
                continue
            fname = item.get("filename")
            box = (item.get("annot_A") or {}).get("boxes") if isinstance(item.get("annot_A"), dict) else None
            if not fname or not isinstance(box, dict):
                continue
            try:
                x0 = float(box["minX"]); y0 = float(box["minY"])
                x1 = float(box["maxX"]); y1 = float(box["maxY"])
            except (KeyError, TypeError, ValueError):
                continue
            bbox_by_key[(cls, fname)] = (x0, y0, x1, y1)
    return bbox_by_key


def collect_val_samples(
    val_dir: Path,
    limit_per_class: int = 0,
    label_root: Optional[Path] = None,
) -> Tuple[List[Path], np.ndarray, List[Optional[Tuple[float, float, float, float]]]]:
    """val_dir/<class>/<img> 구조에서 (paths, y_true, bboxes) 반환.

    label_root 지정되면 각 이미지에 해당하는 annot_A bbox 도 같이 반환.
    bbox 없는 이미지는 None.
    CLASSES 순서에 따라 y_true 는 0~3 정수. 정렬된 파일 순서 — 재현성.
    """
    if not val_dir.is_dir():
        raise FileNotFoundError(f"val-dir 없음: {val_dir}")
    bbox_by_key: dict = {}
    if label_root is not None:
        if not label_root.is_dir():
            raise FileNotFoundError(f"label-root 없음: {label_root}")
        bbox_by_key = _load_val_bboxes(label_root, split="val")
    paths: List[Path] = []
    y_true: List[int] = []
    bboxes: List[Optional[Tuple[float, float, float, float]]] = []
    for idx, cls in enumerate(CLASSES):
        cls_dir = val_dir / cls
        if not cls_dir.is_dir():
            raise FileNotFoundError(f"클래스 디렉토리 없음: {cls_dir}")
        files = sorted(p for p in cls_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in IMG_EXT)
        if limit_per_class > 0:
            files = files[:limit_per_class]
        paths.extend(files)
        y_true.extend([idx] * len(files))
        for f in files:
            # (class, filename) 튜플로 조회 — 동명이인 방어
            bboxes.append(bbox_by_key.get((cls, f.name)))
    y = np.asarray(y_true, dtype=np.int64)
    n_with_box = sum(1 for b in bboxes if b is not None)
    _log(f"val 수집: 총 {len(paths)}장 (per-class counts: "
         f"{[int((y == i).sum()) for i in range(NUM_CLASSES)]}), "
         f"bbox 매칭 {n_with_box}/{len(paths)}")
    return paths, y, bboxes


# ------------------------------------------------------------------
# 모델 load & probs 수집 (캐시)
# ------------------------------------------------------------------

def model_cache_key(model_path: Path) -> str:
    """캐시 파일명 key: stem + mtime_ns (파일 변경 감지)."""
    st = model_path.stat()
    # mtime 까지 포함해서 모델 덮어쓰기 됐을 때 캐시 무효화
    return f"{model_path.stem}.mtime{st.st_mtime_ns}"


def collect_probs_for_model(
    model_path: Path,
    paths: List[Path],
    cache_dir: Path,
    force_cpu: bool,
    force_recompute: bool,
    bboxes: Optional[List[Optional[Tuple[float, float, float, float]]]] = None,
    crop_mode: str = "raw",
) -> np.ndarray:
    """모델 1개 → probs (N, 4). 캐시 지원.

    crop_mode:
      - "raw": bbox 무시, 전체 이미지 (predict.py 기본)
      - "bbox": bboxes 리스트 기반 PIL crop 후 predict (학습 조건 일치)
    캐시 포맷: npz with probs + paths. crop_mode 가 캐시 파일명에 박혀 있어 자동 무효화.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = model_cache_key(model_path)
    cache_file = cache_dir / f"{key}_{crop_mode}_val_probs.npz"

    if cache_file.is_file() and not force_recompute:
        try:
            # paths 는 fixed-length unicode ndarray 로 저장하기 때문에 pickle 불필요.
            dat = np.load(cache_file, allow_pickle=False)
            cached_probs = dat["probs"]
            cached_paths = dat["paths"]
            if (cached_probs.shape == (len(paths), NUM_CLASSES)
                    and len(cached_paths) == len(paths)
                    and all(str(cached_paths[i]) == str(paths[i]) for i in range(len(paths)))):
                _log(f"[cache HIT] {model_path.name} ← {cache_file.name}")
                return cached_probs.astype(np.float32)
            else:
                _log(f"[cache MISS] {model_path.name} (shape/path 불일치, 재계산)")
        except Exception as e:
            _log(f"[cache LOAD FAIL] {model_path.name}: {e} → 재계산")

    # 실제 로드 — crop_mode 에 따라 predict.py 의 auto_face_crop 분기 제어
    #   bbox: 이미 ensemble_search 가 JSON bbox 로 PIL crop → predict.py MTCNN 중복 금지
    #   raw:  predict.py 의 MTCNN auto-crop 을 사용 (외부 평가 환경 시뮬)
    device = "cpu" if force_cpu else "auto"
    auto_crop_for_load = (crop_mode != "bbox")
    _log(f"[load] {model_path.name}  device={device}  auto_face_crop={auto_crop_for_load}")
    t0 = time.time()
    try:
        model_obj = predict_mod.load_model(
            str(model_path), device=device, tta=False,
            auto_face_crop=auto_crop_for_load,
        )
    except TypeError:
        # predict.py 가 auto_face_crop 인자를 아직 안 받는 구버전 fallback
        model_obj = predict_mod.load_model(str(model_path), device=device, tta=False)
    _log(f"[load ok] {model_path.name}  type={model_obj['type']}  ({time.time()-t0:.1f}s)")

    from PIL import Image, ImageOps

    probs = np.zeros((len(paths), NUM_CLASSES), dtype=np.float32)
    t0 = time.time()
    log_every = max(50, len(paths) // 20)
    use_bbox = (crop_mode == "bbox" and bboxes is not None
                and len(bboxes) == len(paths))
    if crop_mode == "bbox" and not use_bbox:
        _log(f"[warn] crop_mode=bbox 지정됐으나 bboxes 리스트가 None/길이 불일치 → raw 로 fallback")
    for i, p in enumerate(paths):
        try:
            if use_bbox:
                # predict_probs 에 PIL (bbox crop 된) 직접 전달. 학습 조건과 일치.
                pil = Image.open(p).convert("RGB")
                try:
                    pil = ImageOps.exif_transpose(pil)
                except Exception:
                    pass
                bbox = bboxes[i]
                if bbox is not None:
                    W, H = pil.size
                    x0, y0, x1, y1 = bbox
                    x0 = max(0.0, min(float(W), float(x0)))
                    x1 = max(0.0, min(float(W), float(x1)))
                    y0 = max(0.0, min(float(H), float(y0)))
                    y1 = max(0.0, min(float(H), float(y1)))
                    if (x1 - x0) > 1 and (y1 - y0) > 1:
                        pil = pil.crop((int(x0), int(y0),
                                        int(round(x1)), int(round(y1))))
                pr = predict_mod.predict_probs(model_obj, pil)
            else:
                pr = predict_mod.predict_probs(model_obj, p)
        except Exception as e:
            raise RuntimeError(f"predict_probs 실패 at {p}: {e}") from e
        pr = np.asarray(pr, dtype=np.float32).reshape(-1)
        if pr.shape[0] != NUM_CLASSES:
            raise RuntimeError(f"probs 차원 불일치: {pr.shape} (expected {NUM_CLASSES},)")
        probs[i] = pr
        if (i + 1) % log_every == 0:
            _log(f"  {model_path.name}  {i+1}/{len(paths)}  ({time.time()-t0:.1f}s)")

    # 합계 1 근처인지 sanity
    sums = probs.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=5e-3):
        bad = int((np.abs(sums - 1.0) > 5e-3).sum())
        _log(f"[warn] {model_path.name} probs 합이 1 아닌 행 {bad}/{len(paths)} — renormalize")
        probs = probs / np.clip(sums[:, None], 1e-8, None)

    # 캐시 저장 — paths 는 고정길이 unicode (object/pickle 회피)
    try:
        # numpy 가 unicode 최대 길이 자동 결정 (U<maxlen>), pickle 불필요 → allow_pickle=False load 가능
        path_arr = np.asarray([str(p) for p in paths])   # dtype='<Uxxx'
        np.savez(cache_file, probs=probs, paths=path_arr)
        _log(f"[cache SAVE] {cache_file.name}")
    except Exception as e:
        _log(f"[cache SAVE FAIL] {e}")

    # 모델 해제 시도 (메모리). TF 는 clear_session, torch 는 cuda empty_cache.
    try:
        del model_obj
        # TF
        try:
            import tensorflow as tf  # 이미 load 시 import 됐으면 cached
            tf.keras.backend.clear_session()
        except Exception:
            pass
        # torch
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        import gc
        gc.collect()
    except Exception:
        pass

    return probs


# ------------------------------------------------------------------
# 메트릭
# ------------------------------------------------------------------

def acc(probs: np.ndarray, y: np.ndarray) -> float:
    """argmax 정확도."""
    assert probs.ndim == 2 and probs.shape[1] == NUM_CLASSES
    assert y.shape[0] == probs.shape[0]
    return float((probs.argmax(axis=1) == y).mean())


def macro_f1(probs: np.ndarray, y: np.ndarray) -> float:
    """sklearn 없이 직접 macro-F1 계산.

    수식: F1_c = 2·P_c·R_c / (P_c+R_c), macro = mean over c.
    P_c = TP_c / (TP_c + FP_c), R_c = TP_c / (TP_c + FN_c).
    클래스 support 0 이면 F1 = 0 (sklearn zero_division=0 과 동일).
    """
    pred = probs.argmax(axis=1)
    f1s = []
    for c in range(NUM_CLASSES):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
            continue
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        if p + r == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * p * r / (p + r))
    return float(np.mean(f1s))


def nll(probs: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """Categorical cross-entropy NLL (mean).
    -(1/N) Σ log p_{y_i}  (clip eps for log-safety)."""
    idx = np.arange(len(y))
    p_true = probs[idx, y]
    return float(-np.log(np.clip(p_true, eps, 1.0)).mean())


# ------------------------------------------------------------------
# Temperature Scaling
# ------------------------------------------------------------------

def apply_temperature(probs: np.ndarray, T: float, eps: float = 1e-12) -> np.ndarray:
    """probs^(1/T) renormalized → softmax(logits/T) 와 동일 (증명은 파일 상단 주석 참조).

    T > 0 필요. T = 1.0 면 identity.
    """
    if T <= 0:
        raise ValueError(f"temperature 는 > 0 이어야 함: {T}")
    p = np.clip(probs, eps, 1.0).astype(np.float64)
    # (1/T) 거듭제곱 시 underflow 방지: log-space
    #   log_q_i = (1/T) * log p_i ;  q = softmax_over_axis1(log_q)
    log_q = np.log(p) / T
    log_q = log_q - log_q.max(axis=1, keepdims=True)
    e = np.exp(log_q)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def optimize_temperature(probs: np.ndarray, y: np.ndarray,
                         bounds: Tuple[float, float] = (0.05, 10.0)) -> float:
    """NLL 최소화하는 T ∈ bounds.  scipy.optimize.minimize_scalar bounded."""
    from scipy.optimize import minimize_scalar

    def obj(T):
        return nll(apply_temperature(probs, float(T)), y)

    res = minimize_scalar(obj, bounds=bounds, method="bounded",
                          options={"xatol": 1e-3, "maxiter": 100})
    T_opt = float(res.x)
    # Sanity: T=1 과 비교 (항상 1.0 대비 NLL 같거나 낮아야 함)
    nll_1 = obj(1.0)
    nll_opt = float(res.fun)
    if nll_opt > nll_1 + 1e-6:
        # 수치적 이슈. 1.0 반환.
        return 1.0
    return T_opt


# ------------------------------------------------------------------
# Weight 탐색 (differential_evolution)
# ------------------------------------------------------------------

def optimize_weights_de(probs_list: List[np.ndarray], y: np.ndarray,
                         seed: int, maxiter: int) -> Tuple[np.ndarray, float]:
    """각 모델별 probs 리스트 → best weights (shape (M,), sum=1), best val_acc.

    목적: 1 - val_acc. bounds: w ∈ [0,1]^M. sum=0 이면 big penalty.
    """
    from scipy.optimize import differential_evolution

    M = len(probs_list)
    N = probs_list[0].shape[0]
    # stack (M, N, C) 로 미리 쌓아두면 각 eval 빠름
    P = np.stack(probs_list, axis=0).astype(np.float64)   # (M, N, 4)

    def obj(w):
        w = np.asarray(w, dtype=np.float64)
        s = w.sum()
        if s < 1e-8:
            return 1.0   # full failure
        wn = w / s
        agg = np.tensordot(wn, P, axes=([0], [0]))  # (N, C)
        acc_v = (agg.argmax(axis=1) == y).mean()
        return 1.0 - float(acc_v)

    bounds = [(0.0, 1.0)] * M
    res = differential_evolution(
        obj, bounds,
        seed=seed, maxiter=maxiter, tol=1e-5, polish=True,
        popsize=max(15, 5 * M), mutation=(0.5, 1.0), recombination=0.7,
        init="sobol" if M >= 2 else "latinhypercube",
    )
    w_opt = np.asarray(res.x, dtype=np.float64)
    s = w_opt.sum()
    if s < 1e-8:
        # 모든 모델 동등 가중 fallback
        w_opt = np.ones(M, dtype=np.float64) / M
    else:
        w_opt = w_opt / s
    best_acc = 1.0 - float(res.fun)
    return w_opt.astype(np.float32), best_acc


# ------------------------------------------------------------------
# Stacking (LR) — OOF val_acc
# ------------------------------------------------------------------

def stacking_oof_acc(probs_list: List[np.ndarray], y: np.ndarray,
                     seed: int) -> Tuple[float, float, float]:
    """5-fold StratifiedKFold OOF LogisticRegression.

    Feature: concat(probs_1, probs_2, ..., probs_M)  shape (N, M*4)
    Label: y (N,)
    반환: (val_acc, val_macro_f1, val_nll)  — 모두 OOF 기준
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    X = np.concatenate(probs_list, axis=1).astype(np.float64)  # (N, M*4)
    y = np.asarray(y, dtype=np.int64)
    N = len(y)

    oof_probs = np.zeros((N, NUM_CLASSES), dtype=np.float64)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        # sklearn 1.7+ 에서 multi_class 인자 제거됨 — lbfgs 는 자동으로 multinomial 처리
        clf = LogisticRegression(
            max_iter=2000, C=1.0, solver="lbfgs",
            random_state=seed,
        )
        clf.fit(X[tr], y[tr])
        # classes_ 가 0..3 순서 보장되는지 확인
        expected = np.arange(NUM_CLASSES)
        if not np.array_equal(clf.classes_, expected):
            # 재배열 (학습 fold 에 클래스 4개 다 있으면 항상 0,1,2,3 이지만 방어)
            proba = clf.predict_proba(X[va])
            remap = np.zeros((len(va), NUM_CLASSES))
            for i_out, c in enumerate(clf.classes_):
                remap[:, int(c)] = proba[:, i_out]
            oof_probs[va] = remap
        else:
            oof_probs[va] = clf.predict_proba(X[va])

    return (acc(oof_probs.astype(np.float32), y),
            macro_f1(oof_probs.astype(np.float32), y),
            nll(oof_probs.astype(np.float32), y))


# ------------------------------------------------------------------
# 리포트 / json 출력
# ------------------------------------------------------------------

def summarize(
    methods_table: List[dict],
    per_model_info: List[dict],
    best_method: dict,
    args: argparse.Namespace,
    val_count: int,
) -> str:
    """markdown report 본문 생성."""
    lines = []
    lines.append("# Ensemble Search Report")
    lines.append("")
    lines.append(f"- val samples: **{val_count}** ({args.val_dir})")
    lines.append(f"- seed: {args.seed}")
    lines.append(f"- de_maxiter: {args.de_maxiter}")
    lines.append(f"- apply_ts (final json): {args.apply_ts}")
    lines.append(f"- skip_stacking: {args.skip_stacking}")
    lines.append("")
    lines.append("## Per-model val metrics (no ensemble)")
    lines.append("")
    lines.append("| # | model | val_acc | val_macro_f1 | val_nll | T* (NLL opt) |")
    lines.append("|:-:|---|---:|---:|---:|---:|")
    for i, m in enumerate(per_model_info):
        lines.append(f"| {i} | `{m['name']}` | {m['val_acc']:.4f} | "
                     f"{m['val_macro_f1']:.4f} | {m['val_nll']:.4f} | {m['T']:.3f} |")
    lines.append("")
    lines.append("## Ensemble methods")
    lines.append("")
    lines.append("| method | val_acc | val_macro_f1 | val_nll | notes |")
    lines.append("|---|---:|---:|---:|---|")
    for row in methods_table:
        lines.append(f"| {row['method']} | {row['val_acc']:.4f} | "
                     f"{row['val_macro_f1']:.4f} | {row['val_nll']:.4f} | {row.get('notes','')} |")
    lines.append("")
    lines.append("## Best method")
    lines.append("")
    lines.append(f"- **{best_method['method']}**")
    lines.append(f"- val_acc = {best_method['val_acc']:.4f}")
    lines.append(f"- val_macro_f1 = {best_method['val_macro_f1']:.4f}")
    lines.append(f"- val_nll = {best_method['val_nll']:.4f}")
    if "weights" in best_method:
        lines.append(f"- weights = {[round(float(w),4) for w in best_method['weights']]}")
    if "Ts" in best_method:
        lines.append(f"- temperatures = {[round(float(t),3) for t in best_method['Ts']]}")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `apply_temperature(p, T) = p^(1/T) / sum(p^(1/T))` — softmax(z/T) 와 수학적으로 동일 "
                 "(증명: `logits = log(probs) + C`, softmax 는 상수 shift invariant).")
    lines.append("- TS 는 NLL (calibration) 기준 최적. val_acc 에는 영향 거의 없음 (argmax 동일).")
    lines.append("- 최종 json 의 `weight` 은 raw softmax weighted voting 기준으로 predict.py 와 호환.")
    if args.apply_ts:
        lines.append("- `--apply-ts` 켜짐: `temperature` 도 json 에 기록. predict.py 확장 필요.")
    else:
        lines.append("- `--apply-ts` 꺼짐 (기본): `temperature` 는 참고만 기록, weight 만 inference 에 사용.")
    return "\n".join(lines) + "\n"


def write_json_config(
    model_paths: List[Path],
    weights: np.ndarray,
    Ts: np.ndarray,
    best_method_name: str,
    best_val_acc: float,
    best_val_f1: float,
    best_val_nll: float,
    per_model_info: List[dict],
    output_path: Path,
    apply_ts: bool,
) -> None:
    """predict.py 호환 json 저장."""
    assert len(model_paths) == len(weights) == len(Ts)

    # 경로 상대화 (프로젝트 루트 기준) — predict.py 가 "models/xxx" 상대 경로를 PROJECT 기준으로 해석함
    members = []
    for p, w, T in zip(model_paths, weights, Ts):
        try:
            rel = p.resolve().relative_to(PROJECT.resolve())
            rel_str = str(rel)
        except ValueError:
            rel_str = str(p)
        entry = {"path": rel_str, "weight": round(float(w), 6)}
        if apply_ts:
            entry["temperature"] = round(float(T), 4)
        members.append(entry)

    out = {
        "_method": best_method_name,
        "_val_acc": round(float(best_val_acc), 6),
        "_val_macro_f1": round(float(best_val_f1), 6),
        "_val_nll": round(float(best_val_nll), 6),
        "_per_model": {
            m["name"]: {
                "val_acc": round(float(m["val_acc"]), 6),
                "val_macro_f1": round(float(m["val_macro_f1"]), 6),
                "val_nll": round(float(m["val_nll"]), 6),
                "temperature": round(float(m["T"]), 4),
            } for m in per_model_info
        },
        "_notes": (
            "ensemble_search.py 가 선택한 최적 방법. "
            + ("temperature 는 predict.py 가 현재 무시 — 확장 필요."
               if apply_ts else
               "temperature 는 참고값. predict.py 는 weight 만 사용 (raw weighted voting).")
        ),
        "models": members,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"[save json] {output_path}")


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="여러 모델 val 예측 수집 → 최적 앙상블 탐색 → json + md 출력",
    )
    ap.add_argument("--models", nargs="+", required=True,
                    help="모델 경로 여러 개 (.h5/.pt). 2개 이상 권장.")
    ap.add_argument("--val-dir", required=True,
                    help="val 이미지 디렉토리 (하위에 anger/happy/panic/sadness 폴더)")
    ap.add_argument("--output-config", default=str(PROJECT / "models" / "ensemble_best.json"),
                    help="최종 앙상블 json 저장 경로")
    ap.add_argument("--output-report", default=str(PROJECT / "results" / "ensemble_report.md"),
                    help="리포트 markdown 저장 경로")
    ap.add_argument("--cache-dir", default=str(PROJECT / "results" / "ensemble_cache"),
                    help="per-model probs npz 캐시 디렉토리")
    ap.add_argument("--force-cpu", action="store_true",
                    help="TF/torch CPU 강제 (GPU 경쟁 회피)")
    ap.add_argument("--force-recompute", action="store_true",
                    help="캐시 무시하고 전부 재예측")
    ap.add_argument("--seed", type=int, default=42, help="reproducibility")
    ap.add_argument("--de-maxiter", type=int, default=60,
                    help="differential_evolution maxiter")
    ap.add_argument("--skip-stacking", action="store_true",
                    help="sklearn LogisticRegression stacking 건너뛰기")
    ap.add_argument("--apply-ts", action="store_true",
                    help="TS 적용 probs 로 weight 최적화 후 temperature 까지 json 기록 "
                         "(predict.py 확장 필요; 기본 꺼짐)")
    ap.add_argument("--limit-per-class", type=int, default=0,
                    help="클래스당 이미지 최대 수 (0=제한없음). 빠른 디버그용.")
    ap.add_argument("--tie-epsilon", type=float, default=1e-6,
                    help="val_acc 가 이 값 이내로 같으면 더 단순한 방법을 우선 선택")
    ap.add_argument("--crop-mode", default="bbox",
                    choices=["raw", "bbox"],
                    help="bbox: annot_A bbox 로 val 이미지 crop 후 평가 "
                         "(학습 조건 일치, default). raw: 전체 이미지.")
    ap.add_argument("--label-root", default=None,
                    help="annot_A bbox 로드용 라벨 루트 "
                         "(예: data_rot/label). 미지정 시 --val-dir 의 "
                         "../label 자동 감지.")
    args = ap.parse_args()

    # -------- force_cpu 실제 적용 --------
    if args.force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        _log("force_cpu: CUDA_VISIBLE_DEVICES=''")

    # -------- 경로 정규화 --------
    model_paths: List[Path] = []
    for m in args.models:
        p = Path(m)
        if not p.is_absolute():
            cand = PROJECT / p
            p = cand if cand.is_file() else p
        if not p.is_file():
            raise FileNotFoundError(f"모델 파일 없음: {p}")
        model_paths.append(p.resolve())

    if len(set(model_paths)) != len(model_paths):
        raise ValueError(f"중복 모델 경로: {model_paths}")
    if len(model_paths) < 2:
        _log("[warn] 모델 1개만 지정됨. 앙상블 의미 없음. 그래도 진행 (TS 확인용).")

    val_dir = Path(args.val_dir)
    if not val_dir.is_absolute():
        val_dir = PROJECT / val_dir
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = PROJECT / cache_dir
    output_config = Path(args.output_config)
    if not output_config.is_absolute():
        output_config = PROJECT / output_config
    output_report = Path(args.output_report)
    if not output_report.is_absolute():
        output_report = PROJECT / output_report

    # -------- label_root 자동 감지 (bbox 모드) --------
    label_root: Optional[Path] = None
    if args.crop_mode == "bbox":
        if args.label_root:
            label_root = Path(args.label_root)
            if not label_root.is_absolute():
                label_root = PROJECT / label_root
        else:
            # --val-dir 에서 자동: data_rot/img/val 이면 → data_rot/label
            candidate = val_dir.parent.parent / "label"
            if candidate.is_dir():
                label_root = candidate
                _log(f"[auto] label_root = {label_root}")
            else:
                _log(f"[warn] label_root 자동감지 실패 ({candidate}) → crop_mode raw 로 강등")
                args.crop_mode = "raw"

    # -------- val 수집 --------
    paths, y, bboxes = collect_val_samples(
        val_dir, limit_per_class=args.limit_per_class,
        label_root=label_root,
    )
    if len(paths) == 0:
        raise RuntimeError("val 이미지 0장 — 디렉토리/구조 확인 필요")

    _log(f"[crop_mode] {args.crop_mode}  (학습 시 --crop 쓴 모델은 bbox 필수)")

    # -------- 모델별 probs 수집 --------
    probs_list: List[np.ndarray] = []
    per_model_info: List[dict] = []
    for mp in model_paths:
        probs = collect_probs_for_model(
            mp, paths, cache_dir,
            force_cpu=args.force_cpu,
            force_recompute=args.force_recompute,
            bboxes=bboxes,
            crop_mode=args.crop_mode,
        )
        probs_list.append(probs)

    # -------- per-model metric + 최적 T --------
    for mp, probs in zip(model_paths, probs_list):
        a = acc(probs, y)
        f1 = macro_f1(probs, y)
        nl = nll(probs, y)
        T = optimize_temperature(probs, y)
        nl_T = nll(apply_temperature(probs, T), y)
        per_model_info.append({
            "name": mp.stem,
            "path": str(mp),
            "val_acc": a,
            "val_macro_f1": f1,
            "val_nll": nl,
            "T": T,
            "val_nll_T": nl_T,
        })
        _log(f"[model] {mp.stem}  acc={a:.4f}  f1={f1:.4f}  "
             f"nll={nl:.4f} → nll(T={T:.3f})={nl_T:.4f}")

    # -------- 앙상블 방법별 평가 --------
    methods_table: List[dict] = []
    M = len(probs_list)

    # (B) Uniform soft voting
    agg_uniform = np.mean(np.stack(probs_list, axis=0), axis=0)
    methods_table.append({
        "method": "uniform_soft_voting",
        "val_acc": acc(agg_uniform, y),
        "val_macro_f1": macro_f1(agg_uniform, y),
        "val_nll": nll(agg_uniform, y),
        "notes": f"M={M} 균등 평균",
        "_weights": np.ones(M, dtype=np.float32) / M,
        "_Ts": np.ones(M, dtype=np.float32),
        "_agg": agg_uniform,
    })

    # TS-applied probs (각 모델별)
    Ts_arr = np.array([info["T"] for info in per_model_info], dtype=np.float32)
    probs_list_ts = [apply_temperature(p, float(T)) for p, T in zip(probs_list, Ts_arr)]

    # (C) TS + uniform
    agg_ts_uniform = np.mean(np.stack(probs_list_ts, axis=0), axis=0)
    methods_table.append({
        "method": "ts_uniform_soft_voting",
        "val_acc": acc(agg_ts_uniform, y),
        "val_macro_f1": macro_f1(agg_ts_uniform, y),
        "val_nll": nll(agg_ts_uniform, y),
        "notes": f"TS per-model → uniform avg",
        "_weights": np.ones(M, dtype=np.float32) / M,
        "_Ts": Ts_arr,
        "_agg": agg_ts_uniform,
    })

    # (D-raw) Weight opt on raw probs (predict.py 기본 호환 — 이 방법이 최종 json 기준)
    _log(f"[opt] weight DE (raw)  M={M}  maxiter={args.de_maxiter}")
    w_raw, _ = optimize_weights_de(probs_list, y,
                                    seed=args.seed, maxiter=args.de_maxiter)
    # 실제 agg 재계산 (정확한 metric)
    agg_w_raw = np.tensordot(w_raw, np.stack(probs_list, axis=0), axes=([0], [0])).astype(np.float32)
    methods_table.append({
        "method": "weight_opt_raw",
        "val_acc": acc(agg_w_raw, y),
        "val_macro_f1": macro_f1(agg_w_raw, y),
        "val_nll": nll(agg_w_raw, y),
        "notes": f"DE on raw probs, w={[round(float(x),3) for x in w_raw]}",
        "_weights": w_raw,
        "_Ts": np.ones(M, dtype=np.float32),
        "_agg": agg_w_raw,
    })

    # (D-ts) Weight opt on TS probs
    _log(f"[opt] weight DE (TS applied)  M={M}  maxiter={args.de_maxiter}")
    w_ts, _ = optimize_weights_de(probs_list_ts, y,
                                   seed=args.seed, maxiter=args.de_maxiter)
    agg_w_ts = np.tensordot(w_ts, np.stack(probs_list_ts, axis=0), axes=([0], [0])).astype(np.float32)
    methods_table.append({
        "method": "weight_opt_ts",
        "val_acc": acc(agg_w_ts, y),
        "val_macro_f1": macro_f1(agg_w_ts, y),
        "val_nll": nll(agg_w_ts, y),
        "notes": f"DE on TS probs, w={[round(float(x),3) for x in w_ts]}",
        "_weights": w_ts,
        "_Ts": Ts_arr,
        "_agg": agg_w_ts,
    })

    # (E) Stacking LR (옵션)
    if not args.skip_stacking:
        try:
            _log("[opt] stacking 5-fold LR")
            st_acc, st_f1, st_nll = stacking_oof_acc(probs_list, y, seed=args.seed)
            methods_table.append({
                "method": "stacking_lr_oof",
                "val_acc": st_acc,
                "val_macro_f1": st_f1,
                "val_nll": st_nll,
                "notes": "참고용; inference 배포시 predict.py 에 LR 통합 필요 — 최종 json 후보 제외",
                "_skip_as_final": True,
            })
        except Exception as e:
            _log(f"[warn] stacking 실패: {e}")
            methods_table.append({
                "method": "stacking_lr_oof",
                "val_acc": 0.0, "val_macro_f1": 0.0, "val_nll": float("inf"),
                "notes": f"실패: {e}",
                "_skip_as_final": True,
            })

    # per-model 자기 성능도 candidate 로 추가 (앙상블보다 단일이 나을 수도 있으니)
    for idx, (info, probs) in enumerate(zip(per_model_info, probs_list)):
        methods_table.append({
            "method": f"single:{info['name']}",
            "val_acc": info["val_acc"],
            "val_macro_f1": info["val_macro_f1"],
            "val_nll": info["val_nll"],
            "notes": f"단일 모델 기준점",
            "_single_idx": idx,
        })

    # -------- Best 선택 --------
    # 후보 기준: _skip_as_final 아닌 것 중 val_acc 최대, tie 시 nll 낮은 쪽, 또 tie 시 단순 방법 선호 순위
    method_priority = {
        "uniform_soft_voting": 0,
        "ts_uniform_soft_voting": 1,
        "weight_opt_raw": 2,
        "weight_opt_ts": 3,
    }

    def _key(row):
        # 우선 단일/stacking 제외 후보만 priority 사용
        prio = method_priority.get(row["method"], 99)
        return (-row["val_acc"], row["val_nll"], prio)

    eligible = [r for r in methods_table if not r.get("_skip_as_final") and not r["method"].startswith("single:")]
    eligible_sorted = sorted(eligible, key=_key)

    # single 이 eligible 보다 크게 높으면 경고
    best_single = max(per_model_info, key=lambda m: m["val_acc"])
    best_ens = eligible_sorted[0]
    if best_single["val_acc"] > best_ens["val_acc"] + args.tie_epsilon:
        _log(f"[WARN] 단일 모델 '{best_single['name']}' val_acc={best_single['val_acc']:.4f} 가 "
             f"최고 앙상블 '{best_ens['method']}' {best_ens['val_acc']:.4f} 보다 높음.")
        _log(f"       그래도 앙상블 json 출력은 진행 (weight 최적화 기반). single 만 쓰려면 predict.py "
             f"에 해당 .h5/.pt 직접 지정 권장.")

    best_method = best_ens
    _log(f"[BEST] method={best_method['method']}  val_acc={best_method['val_acc']:.4f}  "
         f"val_macro_f1={best_method['val_macro_f1']:.4f}  val_nll={best_method['val_nll']:.4f}")

    # -------- json 출력 결정 --------
    # apply_ts=True 면 best 가 weight_opt_ts / ts_uniform 중에서 선택될 확률 높음.
    # apply_ts=False 면 TS 가 반영된 best (weight_opt_ts, ts_uniform) 를 raw 로 대체.
    if not args.apply_ts and best_method["method"] in ("weight_opt_ts", "ts_uniform_soft_voting"):
        # weight_opt_raw 와 비교해서 val_acc 더 높은 쪽을 raw 모드로 강제
        raw_row = next(r for r in methods_table if r["method"] == "weight_opt_raw")
        uni_row = next(r for r in methods_table if r["method"] == "uniform_soft_voting")
        raw_pick = raw_row if raw_row["val_acc"] >= uni_row["val_acc"] else uni_row
        _log(f"[note] --no-apply-ts 모드 → best({best_method['method']}) 대신 "
             f"{raw_pick['method']} 로 대체 (raw weighted voting 만 predict.py 호환).")
        best_method_for_json = raw_pick
    else:
        best_method_for_json = best_method

    # weights/Ts 해결
    w_out = best_method_for_json.get("_weights", np.ones(M, dtype=np.float32) / M)
    if args.apply_ts:
        T_out = best_method_for_json.get("_Ts", np.ones(M, dtype=np.float32))
    else:
        # predict.py 는 T 무시하지만 per-model optimal T 는 기록만 해둠
        T_out = Ts_arr.copy()

    # -------- 리포트 + json 저장 --------
    # weights info 반영
    best_for_report = dict(best_method)
    best_for_report["weights"] = list(w_out)
    if args.apply_ts:
        best_for_report["Ts"] = list(T_out)

    md = summarize(methods_table, per_model_info, best_for_report, args, val_count=len(paths))
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(md, encoding="utf-8")
    _log(f"[save report] {output_report}")

    write_json_config(
        model_paths=model_paths,
        weights=w_out,
        Ts=T_out,
        best_method_name=best_method_for_json["method"],
        best_val_acc=best_method_for_json["val_acc"],
        best_val_f1=best_method_for_json["val_macro_f1"],
        best_val_nll=best_method_for_json["val_nll"],
        per_model_info=per_model_info,
        output_path=output_config,
        apply_ts=args.apply_ts,
    )

    # -------- 자체 재검증 (bug 0 확인) --------
    # 출력한 json 을 실제로 다시 읽어 weights 대로 agg 해보기
    with open(output_config, encoding="utf-8") as f:
        written = json.load(f)
    # weights normalize
    mem_w = np.array([float(m["weight"]) for m in written["models"]], dtype=np.float64)
    if mem_w.sum() <= 0:
        raise RuntimeError("저장된 json weight 합 ≤ 0")
    wn = mem_w / mem_w.sum()
    if args.apply_ts and "temperature" in written["models"][0]:
        probs_for_check = [apply_temperature(p, float(m["temperature"]))
                           for p, m in zip(probs_list, written["models"])]
    else:
        probs_for_check = probs_list
    agg_chk = np.tensordot(wn, np.stack(probs_for_check, axis=0), axes=([0], [0])).astype(np.float32)
    chk_acc = acc(agg_chk, y)
    _log(f"[self-check] json 재적용 val_acc = {chk_acc:.4f} "
         f"(recorded: {best_method_for_json['val_acc']:.4f})")
    # 허용 오차 (weight rounding 6자리) — 이내면 통과
    if abs(chk_acc - best_method_for_json["val_acc"]) > 5e-3:
        _log(f"[WARN] 재검증 val_acc 큰 오차. 저장된 weight rounding 영향 — "
             f"rounding 자릿수 늘리거나 실증값 재확인 필요.")

    _log("[done]")


if __name__ == "__main__":
    main()
