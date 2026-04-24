#!/usr/bin/env python
"""전체 모델 val 1200장 재평가 — 문서 수치 팩트체크.

E1~E9 + 앙상블 전부 동일한 val set (1200장) 에서 accuracy, Macro F1, NLL 재계산.
이미 cache(`results/ensemble_cache_phase2/*.npz`) 가 있는 모델은 즉시 계산,
없는 모델은 실제 inference 로 돌림.

결과: results/reeval_all_models.json
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT = Path("/workspace/user4/emotion-project")
sys.path.insert(0, str(PROJECT))

from predict import CLASSES, load_model, predict_probs  # noqa: E402

CLS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
CACHE_DIR = PROJECT / "results/ensemble_cache_phase2"

# (name, ckpt_path, cache_npz_basename_or_None, claim, auto_face_crop)
MODELS_TO_EVAL = [
    # 학습 시 crop 설정과 일치시켜 평가
    ("E1_resnet50_frozen",   "models/exp01_resnet50_baseline.h5",                  None,                                                            "0.4042",        False),   # 학습 crop=0
    ("E2_resnet50_ft",       "models/exp02_resnet50_ft_crop_aug.h5",               "exp02_resnet50_ft_crop_aug.mtime1776410889000000000_bbox_val_probs.npz",  "0.7733",        True),
    ("E3_adv_adamw",         "models/exp03_adv_adamw_cosine.h5",                   None,                                                            "0.7617",        True),
    ("E4_effnet_ft",         "models/exp04_effnet_ft_balanced.h5",                 "exp04_effnet_ft_balanced.mtime1776415932000000000_bbox_val_probs.npz",    "0.7958",        True),
    ("E5_vit_b16",           "models/exp05_vit_b16_two_stage.pt",                  "exp05_vit_b16_two_stage.mtime1776419368000000000_bbox_val_probs.npz",     "0.8458/0.8383", True),
    ("E6_siglip_lp",         "models/exp06_siglip_linear_probe.pt",                None,                                                            "0.8192",        True),
    ("E9_siglip_kd_uf4",     "models/exp09_siglip_kd_tsoff_T4_a07_uf4.pt",         "exp09_siglip_kd_tsoff_T4_a07_uf4.mtime1776447103000000000_bbox_val_probs.npz", "0.8383/0.8333", True),
    ("ENSEMBLE_FULL",        "models/ensemble_with_kd.json",                       None,                                                            "0.8692",        True),
]


def load_val_labels() -> tuple[list[str], np.ndarray]:
    """val 에서 .jpg / .jpeg / .png 전부 수집."""
    val_dir = PROJECT / "data_rot/img/val"
    IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = []
    labels = []
    for cls in CLASSES:
        items = [p for p in (val_dir / cls).iterdir()
                 if p.is_file() and p.suffix.lower() in IMG_EXT]
        items = sorted(items)
        for p in items:
            paths.append(str(p))
            labels.append(CLS_TO_IDX[cls])
    return paths, np.array(labels, dtype=np.int64)


def compute_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    preds = np.argmax(probs, axis=1)
    acc = float((preds == labels).mean())
    # Macro F1
    f1s = []
    for c in range(len(CLASSES)):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s))
    # NLL
    eps = 1e-8
    p_true = probs[np.arange(len(labels)), labels]
    nll = float(-np.log(np.clip(p_true, eps, 1.0)).mean())
    return {"acc": acc, "macro_f1": macro_f1, "nll": nll}


def eval_from_cache(npz_name: str) -> tuple[dict, float]:
    """cache 의 paths 로 labels 를 재구성 (val 폴더와 다를 수 있음)."""
    t0 = time.perf_counter()
    d = np.load(CACHE_DIR / npz_name)
    probs = d["probs"].astype(np.float32)
    paths = d["paths"]
    # path 에서 class 추출 (data_rot/img/val/<class>/file.jpg)
    cache_labels = np.array(
        [CLS_TO_IDX[Path(p).parent.name] for p in paths], dtype=np.int64)
    dt = time.perf_counter() - t0
    return compute_metrics(probs, cache_labels), dt, len(paths)


def eval_from_model(model_path: str, paths: list[str], labels: np.ndarray,
                    auto_face_crop: bool) -> tuple[dict, float, float]:
    print(f"  [load] {model_path}  (auto_face_crop={auto_face_crop})", file=sys.stderr)
    t_load = time.perf_counter()
    model = load_model(str(PROJECT / model_path), tta=False, auto_face_crop=auto_face_crop)
    load_sec = time.perf_counter() - t_load
    print(f"  [load] done {load_sec:.1f}s", file=sys.stderr)

    # warmup 2회 (XLA 컴파일 있으면 여기서 먹힘)
    for i in range(2):
        _ = predict_probs(model, paths[i])

    t0 = time.perf_counter()
    probs_list = []
    for i, p in enumerate(paths):
        probs = predict_probs(model, p)
        probs_list.append(probs)
        if (i + 1) % 200 == 0:
            elapsed = time.perf_counter() - t0
            eta = elapsed / (i + 1) * (len(paths) - i - 1)
            print(f"    [{i+1}/{len(paths)}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s",
                  file=sys.stderr)
    probs = np.stack(probs_list, axis=0)
    dt = time.perf_counter() - t0

    metrics = compute_metrics(probs, labels)

    # 메모리 정리
    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except Exception:
        pass

    return metrics, load_sec, dt


def main():
    print("=" * 90, file=sys.stderr)
    print("전체 모델 val 1200 재평가 — 팩트체크", file=sys.stderr)
    print("=" * 90, file=sys.stderr)

    paths, labels = load_val_labels()
    print(f"val {len(paths)} 장 로드", file=sys.stderr)

    results = []
    for name, ckpt, cache, claimed, crop in MODELS_TO_EVAL:
        print(f"\n--- {name}  (claim: {claimed}, crop={crop}) ---", file=sys.stderr)
        row = {"name": name, "ckpt": ckpt, "claimed": claimed, "auto_face_crop": crop}

        if cache is not None and (CACHE_DIR / cache).is_file():
            metrics, dt, n = eval_from_cache(cache)
            row.update(metrics)
            row["method"] = "cache"
            row["n"] = n
            row["time_sec"] = round(dt, 3)
            print(f"  [cache] n={n} acc={metrics['acc']:.4f}  f1={metrics['macro_f1']:.4f}  "
                  f"nll={metrics['nll']:.4f}  ({dt:.3f}s)", file=sys.stderr)
        else:
            ckpt_path = PROJECT / ckpt
            if not ckpt_path.is_file():
                print(f"  [SKIP] ckpt not found: {ckpt_path}", file=sys.stderr)
                row.update({"method": "skip_no_ckpt"})
                results.append(row)
                continue
            metrics, load_sec, dt = eval_from_model(ckpt, paths, labels, crop)
            row.update(metrics)
            row["method"] = "inference"
            row["n"] = len(paths)
            row["load_sec"] = round(load_sec, 1)
            row["time_sec"] = round(dt, 1)
            print(f"  [infer] n={len(paths)} acc={metrics['acc']:.4f}  f1={metrics['macro_f1']:.4f}  "
                  f"nll={metrics['nll']:.4f}  (load {load_sec:.1f}s, infer {dt:.0f}s)",
                  file=sys.stderr)
        results.append(row)

    # 최종 표
    print("\n" + "=" * 100, file=sys.stderr)
    print(f"{'Model':<22s} {'Method':<10s} {'claim':<18s} {'acc':>8s} {'F1':>8s} {'NLL':>8s} {'Δclaim':>10s}",
          file=sys.stderr)
    print("-" * 100, file=sys.stderr)
    for r in results:
        if "acc" not in r:
            continue
        claim = r["claimed"]
        # claim 첫 값만 파싱 (예: "0.8458/0.8383" → 0.8458)
        first_claim = float(claim.split("/")[0]) if claim else 0.0
        delta = r["acc"] - first_claim
        print(f"{r['name']:<22s} {r['method']:<10s} {claim:<18s} "
              f"{r['acc']:>8.4f} {r['macro_f1']:>8.4f} {r['nll']:>8.4f} "
              f"{delta:>+9.4f}", file=sys.stderr)

    out = {"val_n": len(paths), "results": results}
    out_path = PROJECT / "results/reeval_all_models.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
