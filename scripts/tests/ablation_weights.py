#!/usr/bin/env python
"""Ablation: cache 재활용해서 weight 조합별 성능 비교."""
import json
from pathlib import Path
import numpy as np

CLASSES = ["anger", "happy", "panic", "sadness"]
CLS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
CACHE = Path("/workspace/user4/emotion-project/results/ensemble_cache_phase2")

# 모델 순서 고정
CKPTS = [
    ("E2_resnet50",   "exp02_resnet50_ft_crop_aug.mtime1776410889000000000_bbox_val_probs.npz"),
    ("E4_effnet",     "exp04_effnet_ft_balanced.mtime1776415932000000000_bbox_val_probs.npz"),
    ("E5_vit",        "exp05_vit_b16_two_stage.mtime1776419368000000000_bbox_val_probs.npz"),
    ("E9_siglip_kd",  "exp09_siglip_kd_tsoff_T4_a07_uf4.mtime1776447103000000000_bbox_val_probs.npz"),
]

# load
probs_list, paths = [], None
for name, fname in CKPTS:
    d = np.load(CACHE / fname)
    probs_list.append(d["probs"])
    if paths is None:
        paths = d["paths"]
    else:
        assert np.array_equal(paths, d["paths"]), f"{name} paths mismatch"

P = np.stack(probs_list, axis=0)  # (4_models, 1200, 4_classes)
# labels from path (data_rot/img/val/<class>/xxx.jpg)
labels = np.array([CLS_TO_IDX[Path(p).parent.name] for p in paths], dtype=np.int64)
print(f"P shape {P.shape}  labels {labels.shape}  classes {CLASSES}")

def eval_weights(weights: np.ndarray) -> dict:
    w = np.asarray(weights, dtype=np.float64)
    if w.sum() <= 0:
        return {"acc": 0.0, "nll": float("nan"), "f1": 0.0}
    w = w / w.sum()
    agg = np.einsum("m,mnc->nc", w, P)  # (1200, 4)
    preds = np.argmax(agg, axis=1)
    acc = float((preds == labels).mean())
    # NLL
    p_true = agg[np.arange(len(labels)), labels]
    nll = float(-np.log(np.clip(p_true, 1e-8, 1.0)).mean())
    # macro F1
    f1s = []
    for c in range(4):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        f1s.append(f1)
    return {"acc": acc, "nll": nll, "f1": float(np.mean(f1s))}

# configs
CONFIGS = {
    "Single E2 (ResNet50)":    [1, 0, 0, 0],
    "Single E4 (EffNet)":      [0, 1, 0, 0],
    "Single E5 (ViT)":         [0, 0, 1, 0],
    "Single E9 (SigLIP KD)":   [0, 0, 0, 1],
    "CNN only (E2+E4)":        [0.5, 0.5, 0, 0],
    "No E9 (E2+E4+E5)":        [0.041151, 0.073014, 0.522546, 0],
    "No E2 (E4+E5+E9)":        [0, 0.073014, 0.522546, 0.36329],
    "No CNN (E5+E9)":          [0, 0, 0.522546, 0.36329],
    "Uniform 4":               [1, 1, 1, 1],
    "Uniform 3 (no E9)":       [1, 1, 1, 0],
    "FULL (DE optimized)":     [0.041151, 0.073014, 0.522546, 0.36329],
}

print(f"\n{'Config':<32s} {'acc':>8s} {'f1':>8s} {'nll':>8s} {'vs FULL':>10s}")
print("-" * 72)
results = {}
for name, w in CONFIGS.items():
    r = eval_weights(np.asarray(w, dtype=np.float64))
    results[name] = r
full_acc = results["FULL (DE optimized)"]["acc"]
for name, r in results.items():
    delta = r["acc"] - full_acc
    print(f"{name:<32s} {r['acc']:>8.4f} {r['f1']:>8.4f} {r['nll']:>8.4f} {delta:>+10.4f}")

# 추가: E9 ablation 인사이트
print(f"\nE9 기여도 분석:")
print(f"  FULL vs No E9      : {results['FULL (DE optimized)']['acc'] - results['No E9 (E2+E4+E5)']['acc']:+.4f}")
print(f"  E5 single vs FULL  : {results['Single E5 (ViT)']['acc'] - full_acc:+.4f}  ← ViT만 쓰면 얼마 손해?")
print(f"  No CNN (E5+E9)     : {results['No CNN (E5+E9)']['acc']:.4f}  ← CNN 빼도 괜찮?")

# JSON 저장
out = Path("/workspace/user4/emotion-project/results/ablation_weights.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump({"n": int(len(labels)), "classes": CLASSES, "configs": {
        name: {"weights": list(map(float, w)), **results[name]}
        for name, w in CONFIGS.items()
    }}, f, indent=2, ensure_ascii=False)
print(f"\n[saved] {out}")
