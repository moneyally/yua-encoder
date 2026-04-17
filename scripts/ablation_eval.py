"""ablation_eval.py — 단일/앙상블 모델에 대한 face-crop × TTA ablation.

목적: 메인 제출 모델 (e.g. E5 ViT) 의 성능을 다양한 조건에서 측정.
      - Face detection: no-crop / annot_A bbox / MTCNN auto-crop
      - TTA: none / hflip / 5crop+hflip / 5crop+multi-scale+hflip

재사용 (SSOT):
  - predict.load_model / predict.predict_probs
  - train_vit.build_crop_records / build_folder_records

출력:
  - results/ablation_<name>/config_<id>.json        (per-config)
  - results/ablation_<name>/summary.md              (표 + 요약)

제1헌법: FAIL 즉시 중단.
제3헌법: 하드코딩 0 — 모든 조합/경로 CLI + env.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

PROJECT = Path(os.environ.get(
    "EMOTION_PROJECT_ROOT",
    str(Path(__file__).resolve().parent.parent),
))
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from predict import load_model, predict_probs, CLASSES  # noqa: E402
from train_vit import build_crop_records, build_folder_records  # noqa: E402

NUM_CLASSES = len(CLASSES)

# ablation 축 정의 — CLI 로 subset 선택 가능
CROP_MODES = ("none", "bbox", "mtcnn")
TTA_CONFIGS = {
    "none":  dict(tta=False, tta_crops="none",  tta_scales=None,          tta_hflip=True),
    "hflip": dict(tta=True,  tta_crops="none",  tta_scales=None,          tta_hflip=True),
    "5crop": dict(tta=True,  tta_crops="5crop", tta_scales=None,          tta_hflip=True),
    "5crop_mscale": dict(tta=True, tta_crops="5crop", tta_scales=[224, 256], tta_hflip=True),
}


# --------------------------------------------------------------------------
# Image loader — crop_mode 별 이미지 변환
# --------------------------------------------------------------------------
def _apply_bbox(img: Image.Image, bbox) -> Image.Image:
    if bbox is None:
        return img
    W, H = img.size
    x0, y0, x1, y1 = bbox
    x0 = max(0.0, min(float(W), float(x0)))
    x1 = max(0.0, min(float(W), float(x1)))
    y0 = max(0.0, min(float(H), float(y0)))
    y1 = max(0.0, min(float(H), float(y1)))
    if (x1 - x0) <= 1 or (y1 - y0) <= 1:
        return img
    return img.crop((int(x0), int(y0),
                     int(math.ceil(x1)), int(math.ceil(y1))))


# --------------------------------------------------------------------------
# 단일 config 평가
# --------------------------------------------------------------------------
def evaluate_config(model_path: Path,
                    records: list[dict],
                    crop_mode: str,
                    tta_name: str,
                    device: str,
                    log_every: int = 200) -> dict:
    """records 전체 val set 추론 → accuracy/F1/NLL/confmat/latency 반환."""
    tta_kw = TTA_CONFIGS[tta_name]
    auto_face_crop = (crop_mode == "mtcnn")
    # bbox 모드 = 우리가 직접 crop, predict.load_model 의 auto_face_crop 은 off
    # none 모드 = 그대로, auto_face_crop off
    model = load_model(
        str(model_path),
        device=device,
        auto_face_crop=auto_face_crop,
        tta=tta_kw["tta"],
        tta_crops=tta_kw["tta_crops"],
        tta_scales=tta_kw["tta_scales"],
        tta_hflip=tta_kw["tta_hflip"],
    )

    probs = np.zeros((len(records), NUM_CLASSES), dtype=np.float32)
    gt = np.zeros(len(records), dtype=np.int64)
    t0 = time.time()
    per_img_times = []
    for i, r in enumerate(records):
        img = Image.open(r["path"]).convert("RGB")
        if crop_mode == "bbox":
            img = _apply_bbox(img, r.get("bbox_xyxy"))
        # mtcnn/none 은 predict.py 내부 처리
        ts = time.time()
        p = predict_probs(model, img)
        per_img_times.append(time.time() - ts)
        p = np.asarray(p, dtype=np.float32).reshape(-1)
        if p.shape[0] != NUM_CLASSES:
            raise RuntimeError(
                f"[FAIL] predict_probs shape {p.shape} (expected ({NUM_CLASSES},))"
            )
        if not np.all(np.isfinite(p)):
            raise RuntimeError(f"[FAIL] NaN/Inf probs at rec={i} path={r['path']}")
        probs[i] = p
        gt[i] = int(r["class_idx"])
        if (i + 1) % log_every == 0:
            el = time.time() - t0
            rate = (i + 1) / max(1e-6, el)
            eta = (len(records) - i - 1) / max(1e-6, rate)
            print(f"    [{crop_mode}/{tta_name}] {i+1}/{len(records)} "
                  f"{rate:.1f} img/s  ETA {eta:.0f}s")
    elapsed = time.time() - t0

    # Metrics
    y_pred = probs.argmax(axis=1)
    acc = float((y_pred == gt).mean())
    eps = 1e-12
    nll = float(-np.mean(np.log(np.clip(probs[np.arange(len(gt)), gt], eps, 1.0))))

    # macro F1, per-class P/R/F1
    from sklearn.metrics import f1_score, classification_report, confusion_matrix
    macro_f1 = float(f1_score(
        gt, y_pred, labels=list(range(NUM_CLASSES)),
        average="macro", zero_division=0,
    ))
    report = classification_report(
        gt, y_pred, labels=list(range(NUM_CLASSES)),
        target_names=CLASSES, digits=4, zero_division=0, output_dict=True,
    )
    cm = confusion_matrix(
        gt, y_pred, labels=list(range(NUM_CLASSES)),
    ).tolist()

    per_img_times = np.asarray(per_img_times)
    return {
        "crop_mode": crop_mode,
        "tta_name": tta_name,
        "n": len(records),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "nll": nll,
        "per_class": {
            CLASSES[c]: {
                "precision": report[CLASSES[c]]["precision"],
                "recall": report[CLASSES[c]]["recall"],
                "f1": report[CLASSES[c]]["f1-score"],
                "support": int(report[CLASSES[c]]["support"]),
            } for c in range(NUM_CLASSES)
        },
        "confusion_matrix": cm,
        "latency_ms_per_img": {
            "mean": float(per_img_times.mean() * 1000),
            "median": float(np.median(per_img_times) * 1000),
            "p95": float(np.percentile(per_img_times, 95) * 1000),
        },
        "elapsed_sec": elapsed,
    }


# --------------------------------------------------------------------------
# Summary markdown
# --------------------------------------------------------------------------
def write_summary(results: list[dict], out_md: Path, model_path: Path) -> None:
    lines = [
        f"# Ablation Results — `{model_path.name}`",
        "",
        f"val set: {results[0]['n']} images × {len(results)} configs",
        "",
        "## Accuracy / F1 / NLL / Latency 요약",
        "",
        "| # | crop | TTA | accuracy | macro F1 | NLL | ms/img (p95) |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(results, 1):
        lines.append(
            f"| {i} | {r['crop_mode']} | {r['tta_name']} | "
            f"{r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['nll']:.4f} | "
            f"{r['latency_ms_per_img']['mean']:.1f} ({r['latency_ms_per_img']['p95']:.1f}) |"
        )

    # Best row
    best = max(results, key=lambda x: x["accuracy"])
    lines += [
        "",
        f"**Best**: `{best['crop_mode']}` + `{best['tta_name']}` → "
        f"accuracy **{best['accuracy']:.4f}**, F1 **{best['macro_f1']:.4f}**",
        "",
        "## Per-class F1 — best config",
        "",
        "| class | precision | recall | F1 | support |",
        "|---|---:|---:|---:|---:|",
    ]
    for c in CLASSES:
        pc = best["per_class"][c]
        lines.append(
            f"| {c} | {pc['precision']:.4f} | {pc['recall']:.4f} | "
            f"{pc['f1']:.4f} | {pc['support']} |"
        )

    # Confusion matrix
    lines += [
        "",
        "## Confusion matrix — best config (rows=gt, cols=pred)",
        "",
        "```",
        "       " + "  ".join(f"{c:>8s}" for c in CLASSES),
    ]
    for i, c in enumerate(CLASSES):
        row = best["confusion_matrix"][i]
        lines.append(f"{c:>6s} " + "  ".join(f"{v:>8d}" for v in row))
    lines += ["```", ""]

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Face-crop × TTA ablation eval (객체탐지 + crop + 고성능 테스트)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", required=True,
                    help=".h5 / .pt / .json (ensemble)")
    ap.add_argument("--data-root", default=str(PROJECT / "data_rot"))
    ap.add_argument("--val-split", default="val", choices=["val", "train"])
    ap.add_argument("--crop-modes", nargs="+", default=list(CROP_MODES),
                    choices=list(CROP_MODES))
    ap.add_argument("--tta-names", nargs="+", default=list(TTA_CONFIGS.keys()),
                    choices=list(TTA_CONFIGS.keys()))
    ap.add_argument("--out-dir", default=None,
                    help="기본: results/ablation_<model_stem>/")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--limit-samples", type=int, default=0,
                    help=">0 이면 stratified N 샘플 (smoke)")
    ap.add_argument("--log-every", type=int, default=200)
    args = ap.parse_args()
    return args


def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.is_file():
        raise SystemExit(f"[FAIL] model 없음: {model_path}")
    data_root = Path(args.data_root)
    if not (data_root / "img" / args.val_split).is_dir():
        raise SystemExit(f"[FAIL] {data_root}/img/{args.val_split} 없음")

    out_dir = Path(args.out_dir) if args.out_dir else \
        PROJECT / "results" / f"ablation_{model_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # records — bbox mode 에서 필요하므로 crop_records 전체 준비
    records_crop = build_crop_records(data_root, args.val_split)
    # none/mtcnn 모드는 folder records 도 가능하지만, crop_records 가 이미 bbox 정보 + 필터
    # 통일. (bbox 없는 샘플은 bbox mode 에서 _apply_bbox 가 원본 유지)
    records = records_crop

    if args.limit_samples > 0:
        # stratified
        from collections import defaultdict
        per_class = defaultdict(list)
        for r in records:
            per_class[int(r["class_idx"])].append(r)
        each = max(1, args.limit_samples // NUM_CLASSES)
        strat = []
        for c in range(NUM_CLASSES):
            strat.extend(per_class.get(c, [])[:each])
        records = strat
        print(f"[limit] stratified {each}/class = {len(records)}")

    print(f"[ablation] model={model_path.name}  n={len(records)}")
    print(f"[ablation] crop_modes={args.crop_modes}  tta={args.tta_names}")
    print(f"[ablation] → {out_dir}")

    results = []
    for cm in args.crop_modes:
        for tn in args.tta_names:
            tag = f"{cm}__{tn}"
            print(f"\n━━━ config: {tag} ━━━")
            r = evaluate_config(
                model_path=model_path,
                records=records,
                crop_mode=cm,
                tta_name=tn,
                device=args.device,
                log_every=args.log_every,
            )
            results.append(r)
            # per-config 저장
            with open(out_dir / f"config_{tag}.json", "w", encoding="utf-8") as f:
                json.dump(r, f, indent=2, ensure_ascii=False)
            print(f"  → acc={r['accuracy']:.4f} F1={r['macro_f1']:.4f} "
                  f"NLL={r['nll']:.4f} ms/img={r['latency_ms_per_img']['mean']:.1f}")

    # summary
    summary_md = out_dir / "summary.md"
    write_summary(results, summary_md, model_path)
    summary_json = out_dir / "summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[DONE] summary → {summary_md}")
    print(f"[DONE] json    → {summary_json}")


if __name__ == "__main__":
    main()
