#!/usr/bin/env python
"""bench_latency.py — ViT 단일 vs 앙상블 실측 latency + accuracy.

val 20장 (4 class × 5) 샘플링 → warmup 3회 후 20회 측정.
mean / median / p95 latency 와 top-1 accuracy 출력.
두 모델 예측이 같은 class 인 비율 (agreement) 도 체크해서 최적화 전후 동등성 검증.

사용:
    python scripts/bench_latency.py
    python scripts/bench_latency.py --n-per-class 10 --warmup 5
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

# 레포 루트 sys.path
PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from predict import CLASSES, load_model, predict_probs   # noqa: E402


def _sample_val(val_root: Path, n_per_class: int, seed: int) -> list[tuple[str, str]]:
    """(image_path, ground_truth_class) 목록 리턴."""
    rng = random.Random(seed)
    out: list[tuple[str, str]] = []
    for cls in CLASSES:
        cls_dir = val_root / cls
        if not cls_dir.is_dir():
            raise FileNotFoundError(f"val 디렉터리 없음: {cls_dir}")
        imgs = sorted(cls_dir.glob("*.jpg"))
        if len(imgs) < n_per_class:
            raise RuntimeError(f"{cls}: {len(imgs)} < {n_per_class} 장")
        picks = rng.sample(imgs, n_per_class)
        for p in picks:
            out.append((str(p), cls))
    return out


def _stats(values_ms: list[float]) -> dict:
    arr = sorted(values_ms)
    n = len(arr)
    return {
        "n": n,
        "mean_ms": round(statistics.mean(arr), 1),
        "median_ms": round(statistics.median(arr), 1),
        "p95_ms": round(arr[max(0, int(round(0.95 * (n - 1))))], 1) if n > 0 else 0.0,
        "min_ms": round(arr[0], 1) if n > 0 else 0.0,
        "max_ms": round(arr[-1], 1) if n > 0 else 0.0,
    }


def _bench_one(name: str, model: dict, samples: list[tuple[str, str]],
               warmup: int) -> dict:
    times: list[float] = []
    preds: list[str] = []
    probs_list: list[np.ndarray] = []

    # GPU async 보장 — np.asarray 변환이 D2H sync 을 강제하지만 명시적으로도 한번 더.
    try:
        import torch as _t
        if _t.cuda.is_available():
            _t.cuda.synchronize()
    except Exception:
        _t = None

    # warmup (첫 N 샘플 반복 — cudnn benchmark / XLA compile 등)
    print(f"  [{name}] warmup {warmup}회...", file=sys.stderr)
    for i in range(warmup):
        _ = predict_probs(model, samples[i % len(samples)][0])
    if _t is not None and _t.cuda.is_available():
        _t.cuda.synchronize()

    print(f"  [{name}] measure {len(samples)}회...", file=sys.stderr)
    for path, _gt in samples:
        t0 = time.perf_counter()
        p = predict_probs(model, path)
        if _t is not None and _t.cuda.is_available():
            _t.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        times.append(dt)
        idx = int(np.argmax(p))
        preds.append(CLASSES[idx])
        probs_list.append(np.asarray(p, dtype=np.float32))

    correct = sum(1 for (_, gt), pred in zip(samples, preds) if pred == gt)
    return {
        "name": name,
        "latency": _stats(times),
        "correct": correct,
        "total": len(samples),
        "accuracy": round(correct / len(samples), 4),
        "preds": preds,
        "probs": np.stack(probs_list, axis=0),   # (N, 4)
        "times_ms": times,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-root", default=str(PROJECT / "data_rot/img/val"))
    ap.add_argument("--n-per-class", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vit-path", default=str(PROJECT / "models/exp05_vit_b16_two_stage.pt"))
    ap.add_argument("--ens-path", default=str(PROJECT / "models/ensemble_with_kd.json"))
    ap.add_argument("--output", default=str(PROJECT / "results/bench_latency.json"))
    ap.add_argument("--skip-ensemble", action="store_true")
    ap.add_argument("--skip-vit", action="store_true")
    args = ap.parse_args()

    val_root = Path(args.val_root)
    if not val_root.is_dir():
        print(f"ERROR: val 디렉터리 없음: {val_root}", file=sys.stderr)
        return 1

    samples = _sample_val(val_root, args.n_per_class, args.seed)
    print(f"[samples] n={len(samples)}  classes={CLASSES}  n_per_class={args.n_per_class}",
          file=sys.stderr)

    results: dict[str, dict] = {}

    if not args.skip_vit:
        vit_path = Path(args.vit_path)
        if not vit_path.is_file():
            print(f"ERROR: ViT ckpt 없음: {vit_path}", file=sys.stderr)
            return 1
        print(f"\n[load] ViT 단일: {vit_path.name}", file=sys.stderr)
        t0 = time.perf_counter()
        vit = load_model(str(vit_path), tta=False, auto_face_crop=True)
        load_s = time.perf_counter() - t0
        print(f"  load {load_s:.2f}s  type={vit['type']}", file=sys.stderr)
        r = _bench_one("ViT-B/16", vit, samples, warmup=args.warmup)
        r["load_sec"] = round(load_s, 2)
        results["vit"] = r

    if not args.skip_ensemble:
        ens_path = Path(args.ens_path)
        if not ens_path.is_file():
            print(f"ERROR: ensemble config 없음: {ens_path}", file=sys.stderr)
            return 1
        print(f"\n[load] 앙상블: {ens_path.name}", file=sys.stderr)
        t0 = time.perf_counter()
        ens = load_model(str(ens_path), tta=False, auto_face_crop=True)
        load_s = time.perf_counter() - t0
        print(f"  load {load_s:.2f}s  type={ens['type']}", file=sys.stderr)
        r = _bench_one("Ensemble", ens, samples, warmup=args.warmup)
        r["load_sec"] = round(load_s, 2)
        results["ensemble"] = r

    # agreement (ViT vs Ensemble 같은 class 비율)
    if "vit" in results and "ensemble" in results:
        agree = sum(1 for a, b in zip(results["vit"]["preds"], results["ensemble"]["preds"]) if a == b)
        total = len(results["vit"]["preds"])
        results["agreement"] = {
            "same_class": agree,
            "total": total,
            "rate": round(agree / total, 4) if total else 0.0,
        }

    # 보고서 프린트
    print("\n" + "=" * 76)
    print(f"{'model':12s} {'n':>3s} {'acc':>6s} {'mean':>8s} {'median':>8s} {'p95':>8s} {'min':>7s} {'max':>8s}")
    print("-" * 76)
    for key, r in results.items():
        if key == "agreement":
            continue
        s = r["latency"]
        print(f"{r['name']:12s} {r['total']:>3d} {r['accuracy']:>6.3f} "
              f"{s['mean_ms']:>7.0f}ms {s['median_ms']:>7.0f}ms {s['p95_ms']:>7.0f}ms "
              f"{s['min_ms']:>6.0f}ms {s['max_ms']:>7.0f}ms")
    if "agreement" in results:
        ag = results["agreement"]
        print(f"\n[agreement] ViT vs Ensemble 같은 class: {ag['same_class']}/{ag['total']} ({ag['rate']:.1%})")
    print("=" * 76)

    # JSON 저장 (probs / times 는 용량 위해 뺌)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_json = {}
    for key, r in results.items():
        if key == "agreement":
            out_json[key] = r
            continue
        out_json[key] = {
            "name": r["name"],
            "load_sec": r["load_sec"],
            "latency": r["latency"],
            "accuracy": r["accuracy"],
            "correct": r["correct"],
            "total": r["total"],
            "preds": r["preds"],
        }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
