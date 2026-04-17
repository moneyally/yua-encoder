"""gen_teacher_soft.py — 앙상블 teacher soft target 생성.

설계 (ChatGPT/설계 리뷰 반영):
- Teacher = ensemble_json (predict.load_model 가 weighted softmax voting 수행)
- Apply TS (optional): val 에서 T* 학습 → train probs 에 scalar temperature scaling
- Confidence filter (optional):
    --min-conf 0.0  (default, off)
    --min-conf 0.4  : top-1 < 0.4 샘플 drop
    --conf-weight   : drop 대신 per-sample weight = top-1 conf (soft filter)
- Key = (class, filename) 튜플 (동명 파일 방어)
- Row 검증: shape (N,4), finite, non-negative, sum≈1
- 출력: npz (allow_pickle=False 호환, dtype=str for names) + meta json

하드코딩 0 — 모든 경로/임계값 argparse, PROJECT env 기준.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

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

# 재사용 — predict.py + train_vit.py
from predict import load_model, predict_probs, CLASSES  # noqa: E402
from train_vit import build_crop_records, build_folder_records  # noqa: E402

NUM_CLASSES = len(CLASSES)


# --------------------------------------------------------------------------
# TS — Temperature Scaling (Guo 2017)
# --------------------------------------------------------------------------
def fit_temperature(probs: np.ndarray, gt: np.ndarray,
                    lo: float = 0.5, hi: float = 10.0) -> float:
    """val probs (N,4) + hard gt (N,) → T* s.t. NLL 최소.
    probs → logits (log of clipped probs) → /T → softmax → NLL on gt.
    scipy scalar minimizer 사용 (경계 [lo,hi]).
    """
    from scipy.optimize import minimize_scalar
    eps = 1e-12
    logits = np.log(np.clip(probs, eps, 1.0))

    def nll(T: float) -> float:
        if T <= 0:
            return float("inf")
        z = logits / float(T)
        z = z - z.max(axis=1, keepdims=True)
        ez = np.exp(z)
        p = ez / ez.sum(axis=1, keepdims=True)
        p_gt = np.clip(p[np.arange(len(gt)), gt], eps, 1.0)
        return float(-np.mean(np.log(p_gt)))

    res = minimize_scalar(nll, bounds=(float(lo), float(hi)), method="bounded",
                          options={"xatol": 1e-4})
    return float(res.x)


def apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    """probs (N,4) → softmax(log(probs)/T)."""
    if T is None or abs(T - 1.0) < 1e-6:
        return probs.astype(np.float32)
    eps = 1e-12
    logits = np.log(np.clip(probs, eps, 1.0))
    z = logits / float(T)
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return (ez / ez.sum(axis=1, keepdims=True)).astype(np.float32)


# --------------------------------------------------------------------------
# Teacher probs 수집 (ensemble_json 기반)
# --------------------------------------------------------------------------
def collect_teacher_probs(ensemble_json: Path,
                          records: list[dict],
                          device: str,
                          log_every: int = 500) -> np.ndarray:
    """records 별 teacher probs (weighted softmax voting via predict.load_model).

    records 는 build_crop_records 출력 형식:
      [{"path": str, "class_idx": int, "bbox_xyxy": (x0,y0,x1,y1), ...}, ...]

    predict.load_model(ensemble_json, auto_face_crop=False) 를 쓰므로,
    bbox crop 은 우리가 직접 수행 (학습 조건 bbox 재현).
    """
    print(f"[teacher] loading ensemble: {ensemble_json}")
    model = load_model(
        str(ensemble_json),
        device=device,
        tta=False,
        auto_face_crop=False,   # bbox 우리가 수행 (중복 crop 방지)
    )
    probs = np.zeros((len(records), NUM_CLASSES), dtype=np.float32)
    t0 = time.time()
    # 제1헌법: WARN/FAIL/CRITICAL 즉시 raise. fallback 으로 감추지 않음.
    for i, rec in enumerate(records):
        with Image.open(rec["path"]) as im:
            im.load()
            if im.mode != "RGB":
                im = im.convert("RGB")
            bbox = rec.get("bbox_xyxy")
            if bbox is not None:
                W, H = im.size
                x0, y0, x1, y1 = bbox
                x0 = max(0.0, min(float(W), float(x0)))
                x1 = max(0.0, min(float(W), float(x1)))
                y0 = max(0.0, min(float(H), float(y0)))
                y1 = max(0.0, min(float(H), float(y1)))
                if (x1 - x0) > 1 and (y1 - y0) > 1:
                    im = im.crop((
                        int(x0), int(y0),
                        int(math.ceil(x1)), int(math.ceil(y1)),
                    ))
            p = predict_probs(model, im)
        p = np.asarray(p, dtype=np.float32).reshape(-1)
        if p.shape[0] != NUM_CLASSES:
            raise RuntimeError(
                f"[FAIL] teacher prob shape: {p.shape} (expected ({NUM_CLASSES},)) "
                f"— rec={i} path={rec.get('path')}"
            )
        if not np.all(np.isfinite(p)):
            raise RuntimeError(
                f"[FAIL] teacher probs NaN/Inf at rec={i} path={rec.get('path')}"
            )
        probs[i] = p
        if (i + 1) % log_every == 0:
            el = time.time() - t0
            rate = (i + 1) / max(1e-6, el)
            eta = (len(records) - i - 1) / max(1e-6, rate)
            print(f"  [teacher] {i+1}/{len(records)}  {rate:.1f} img/s  ETA {eta:.0f}s")
    print(f"[teacher] done. elapsed={time.time()-t0:.1f}s")
    return probs


# --------------------------------------------------------------------------
# 검증
# --------------------------------------------------------------------------
def validate_probs(probs: np.ndarray, tag: str = "probs") -> None:
    """shape (N,4), finite, non-neg, row sum≈1. 실패 시 SystemExit."""
    if probs.ndim != 2 or probs.shape[1] != NUM_CLASSES:
        raise SystemExit(
            f"[{tag}] shape 오류: {tuple(probs.shape)} (expected (N,{NUM_CLASSES}))"
        )
    if not np.all(np.isfinite(probs)):
        raise SystemExit(f"[{tag}] NaN/Inf 포함")
    if not np.all(probs >= 0):
        raise SystemExit(f"[{tag}] 음수 확률 포함: min={probs.min()}")
    sums = probs.sum(axis=1)
    bad = np.where(np.abs(sums - 1.0) > 1e-3)[0]
    if bad.size > 0:
        raise SystemExit(
            f"[{tag}] row sum != 1: {bad.size}건 "
            f"(first 5 sums: {sums[bad[:5]].tolist()})"
        )
    print(f"[{tag}] validate OK — shape={probs.shape}, "
          f"sum∈[{sums.min():.6f},{sums.max():.6f}]")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="앙상블 teacher soft target 생성 (KD 용)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--ensemble-json", required=True,
                    help="models/ensemble_*.json")
    ap.add_argument("--data-root", default=str(PROJECT / "data_rot"))
    ap.add_argument("--split", default="train", choices=["train", "val"],
                    help="soft target 생성 대상 split")
    ap.add_argument("--crop", action="store_true", default=True,
                    help="bbox crop 사용 (학습 조건 재현, 기본 ON)")
    ap.add_argument("--no-crop", dest="crop", action="store_false",
                    help="bbox crop 끄기 (전체 이미지)")
    ap.add_argument("--apply-ts", action="store_true",
                    help="val split 에서 T* 학습 후 train probs 에 적용. "
                         "이 옵션 사용 시 --val-split-for-ts 함께 지정 권장.")
    ap.add_argument("--val-split-for-ts", default="val",
                    help="TS 학습용 split (기본 val).")
    ap.add_argument("--ts-bounds", nargs=2, type=float, default=[0.5, 10.0],
                    help="TS 탐색 구간 [lo hi]")
    ap.add_argument("--min-conf", type=float, default=0.0,
                    help="top-1 < min-conf 샘플 drop (0=off, 0.4~0.5 권장 범위)")
    ap.add_argument("--conf-weight", action="store_true",
                    help="drop 대신 per-sample weight=top1_conf 를 meta 에 저장")
    ap.add_argument("--limit-samples", type=int, default=0,
                    help="상위 N 샘플만 처리 (smoke 용, 0=전체)")
    ap.add_argument("--out", required=True, help="output npz path")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--log-every", type=int, default=500)
    args = ap.parse_args()

    # 경계 체크
    if args.min_conf < 0 or args.min_conf > 1:
        raise SystemExit(f"--min-conf 는 [0,1]: {args.min_conf}")
    if args.ts_bounds[0] <= 0 or args.ts_bounds[1] <= args.ts_bounds[0]:
        raise SystemExit(f"--ts-bounds 이상: {args.ts_bounds}")
    return args


def main():
    args = parse_args()
    ensemble_json = Path(args.ensemble_json)
    data_root = Path(args.data_root)
    out_path = Path(args.out)
    if not ensemble_json.is_file():
        raise SystemExit(f"[!] ensemble json 없음: {ensemble_json}")
    if not (data_root / "img" / args.split).is_dir():
        raise SystemExit(f"[!] data_root/img/{args.split} 없음: {data_root}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ===== records 구성 (crop=True 기본) =====
    if args.crop:
        records = build_crop_records(data_root, args.split)
    else:
        records = build_folder_records(data_root, args.split)
    if args.limit_samples > 0:
        # 계층 샘플링 — 클래스별 균등 분포 유지 (SMOKE 의미 보존, QA-Warning 반영)
        from collections import defaultdict
        per_class: dict[int, list] = defaultdict(list)
        for r in records:
            per_class[int(r["class_idx"])].append(r)
        each = max(1, args.limit_samples // NUM_CLASSES)
        strat = []
        for c in range(NUM_CLASSES):
            strat.extend(per_class.get(c, [])[:each])
        records = strat
        print(f"[limit-samples] stratified: {each}/class × {NUM_CLASSES} = {len(records)}")
    print(f"[records] {args.split}: N={len(records)}  crop={args.crop}")

    # class-wise 분포 확인
    cls_counts = np.bincount(
        [r["class_idx"] for r in records], minlength=NUM_CLASSES,
    )
    print(f"[records] per-class: {dict(zip(CLASSES, cls_counts.tolist()))}")

    # ===== teacher probs 수집 =====
    probs = collect_teacher_probs(
        ensemble_json=ensemble_json,
        records=records,
        device=args.device,
        log_every=args.log_every,
    )
    validate_probs(probs, tag=f"teacher_raw[{args.split}]")

    # ===== TS (optional) =====
    T_star = 1.0
    if args.apply_ts:
        print(f"\n[TS] fit on split={args.val_split_for_ts}")
        if args.crop:
            ts_records = build_crop_records(data_root, args.val_split_for_ts)
        else:
            ts_records = build_folder_records(data_root, args.val_split_for_ts)
        ts_probs = collect_teacher_probs(
            ensemble_json=ensemble_json,
            records=ts_records,
            device=args.device,
            log_every=args.log_every,
        )
        validate_probs(ts_probs, tag=f"teacher_raw[{args.val_split_for_ts}]")
        ts_gt = np.array([r["class_idx"] for r in ts_records], dtype=np.int64)
        T_star = fit_temperature(
            ts_probs, ts_gt,
            lo=args.ts_bounds[0], hi=args.ts_bounds[1],
        )
        print(f"[TS] T* = {T_star:.4f}  (bounds={args.ts_bounds})")
        probs = apply_temperature(probs, T_star)
        validate_probs(probs, tag=f"teacher_ts[{args.split}]")

    # ===== Confidence filter =====
    gt_arr = np.array([r["class_idx"] for r in records], dtype=np.int64)
    top1_conf = probs.max(axis=1)
    drop_mask = np.zeros(len(records), dtype=bool)
    if args.min_conf > 0:
        drop_mask = top1_conf < args.min_conf
        drop_count = int(drop_mask.sum())
        drop_by_class = {
            CLASSES[c]: int(((gt_arr == c) & drop_mask).sum())
            for c in range(NUM_CLASSES)
        }
        print(f"[conf-filter] min_conf={args.min_conf}  drop={drop_count}/{len(records)}")
        print(f"  per-class drop: {drop_by_class}")
    else:
        drop_count = 0
        drop_by_class = {c: 0 for c in CLASSES}

    # per-sample weight (conf-weight 모드)
    sample_weight = np.ones(len(records), dtype=np.float32)
    if args.conf_weight:
        sample_weight = top1_conf.astype(np.float32)
        print(f"[conf-weight] weight = top1 conf. "
              f"mean={sample_weight.mean():.4f}, min={sample_weight.min():.4f}")

    # drop 적용
    keep_mask = ~drop_mask
    filenames_kept = []
    classes_kept = []
    for i, r in enumerate(records):
        if keep_mask[i]:
            filenames_kept.append(Path(r["path"]).name)
            classes_kept.append(CLASSES[int(r["class_idx"])])
    probs_kept = probs[keep_mask]
    gt_kept = gt_arr[keep_mask]
    weight_kept = sample_weight[keep_mask]
    conf_kept = top1_conf[keep_mask]

    # ===== hard GT vs teacher argmax 불일치 (drop 안 함, meta 만) =====
    teacher_argmax = probs_kept.argmax(axis=1)
    mismatch = (teacher_argmax != gt_kept).sum()
    print(f"[mismatch] teacher argmax vs hard gt: {mismatch}/{len(gt_kept)} "
          f"({mismatch/max(1,len(gt_kept))*100:.2f}%)")

    # entropy (분포 다양성 지표)
    eps = 1e-12
    entropy = -np.sum(
        probs_kept * np.log(np.clip(probs_kept, eps, 1.0)), axis=1,
    ).mean()
    print(f"[entropy] mean = {entropy:.4f}  (uniform={math.log(NUM_CLASSES):.4f})")

    # ===== 저장 (dtype=str 고정폭, allow_pickle=False 호환) =====
    np.savez(
        str(out_path),
        filenames=np.array(filenames_kept, dtype=str),
        classes=np.array(classes_kept, dtype=str),
        teacher_probs=probs_kept.astype(np.float32),
        class_idx_gt=gt_kept.astype(np.int64),
        sample_weight=weight_kept.astype(np.float32),
        teacher_conf=conf_kept.astype(np.float32),
        cls_names=np.array(CLASSES, dtype=str),
    )

    # meta json (재현성)
    meta = {
        "ensemble_json": str(ensemble_json),
        "data_root": str(data_root),
        "split": args.split,
        "crop": bool(args.crop),
        "apply_ts": bool(args.apply_ts),
        "ts_star": float(T_star),
        "ts_bounds": list(args.ts_bounds) if args.apply_ts else None,
        "val_split_for_ts": args.val_split_for_ts if args.apply_ts else None,
        "min_conf": float(args.min_conf),
        "conf_weight": bool(args.conf_weight),
        "limit_samples": int(args.limit_samples),
        "classes": CLASSES,
        "n_total_records": len(records),
        "n_kept": int(keep_mask.sum()),
        "n_dropped": int(drop_count),
        "drop_by_class": drop_by_class,
        "mismatch_teacher_vs_hard": int(mismatch),
        "entropy_mean": float(entropy),
        "top1_conf_stats": {
            "mean": float(top1_conf.mean()),
            "min": float(top1_conf.min()),
            "max": float(top1_conf.max()),
            "median": float(np.median(top1_conf)),
        },
        "per_class_kept": {
            CLASSES[c]: int((gt_kept == c).sum()) for c in range(NUM_CLASSES)
        },
    }
    meta_path = out_path.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] npz → {out_path}")
    print(f"[DONE] meta → {meta_path}")


if __name__ == "__main__":
    main()
