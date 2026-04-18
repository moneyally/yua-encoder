#!/usr/bin/env python
"""A. 객체 탐지 실측 테스트.

1. val 20장 (4 class × 5) 얼굴 탐지 성공률
2. 합성 multi-face 이미지 (val 2장 가로 concat) → 얼굴 2개 검출되는지
3. 빈 배경 (solid color) → fallback 동작
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT = Path("/workspace/user4/emotion-project")
sys.path.insert(0, str(PROJECT))

from predict import CLASSES, detect_all_faces  # noqa: E402


def pick_samples(n_per_class: int = 5, seed: int = 42):
    rng = random.Random(seed)
    out = []
    for cls in CLASSES:
        imgs = sorted((PROJECT / "data_rot/img/val" / cls).glob("*.jpg"))
        for p in rng.sample(imgs, n_per_class):
            out.append((str(p), cls))
    return out


def test_single_face(samples):
    """각 val 이미지에서 얼굴 1개 검출 기대."""
    print("\n" + "=" * 72)
    print("[A-1] 단일 얼굴 탐지 (val 20장)")
    print("=" * 72)
    rows = []
    n_detected = 0
    n_empty = 0
    all_conf = []
    for path, cls in samples:
        t0 = time.perf_counter()
        img = Image.open(path).convert("RGB")
        faces = detect_all_faces(img, margin=0.1, min_face_size=20, min_conf=0.9, device="cuda")
        dt_ms = (time.perf_counter() - t0) * 1000
        n = len(faces)
        if n > 0:
            n_detected += 1
            best_conf = max(f["confidence"] for f in faces)
            all_conf.append(best_conf)
        else:
            n_empty += 1
            best_conf = 0.0
        rows.append({"class": cls, "file": Path(path).name[:40],
                     "n_faces": n, "best_conf": round(best_conf, 3),
                     "time_ms": round(dt_ms, 1)})
    print(f"\n{'class':10s} {'file':42s} {'n':>3s} {'conf':>7s} {'time':>8s}")
    print("-" * 72)
    for r in rows[:8]:
        print(f"{r['class']:10s} {r['file']:42s} {r['n_faces']:>3d} {r['best_conf']:>7.3f} {r['time_ms']:>6.1f}ms")
    print(f"... (+{len(rows) - 8} 더)")
    print(f"\n탐지 성공: {n_detected}/{len(samples)} ({n_detected/len(samples)*100:.1f}%)")
    if all_conf:
        print(f"confidence: mean={np.mean(all_conf):.3f}  min={min(all_conf):.3f}  max={max(all_conf):.3f}")
    return {"rows": rows, "detected": n_detected, "total": len(samples),
            "mean_conf": round(float(np.mean(all_conf)), 3) if all_conf else 0.0}


def test_multi_face(samples):
    """val 2장 가로 concat → 얼굴 2개 검출되는지."""
    print("\n" + "=" * 72)
    print("[A-2] 합성 multi-face (val anger + happy 가로 concat)")
    print("=" * 72)
    # class 다른 2장 고름
    anger_path = next(p for p, c in samples if c == "anger")
    happy_path = next(p for p, c in samples if c == "happy")
    a = Image.open(anger_path).convert("RGB")
    h = Image.open(happy_path).convert("RGB")
    # 높이 맞춤 (min height 기준 resize)
    H = min(a.height, h.height, 600)
    a2 = a.resize((int(a.width * H / a.height), H), Image.LANCZOS)
    h2 = h.resize((int(h.width * H / h.height), H), Image.LANCZOS)
    combined = Image.new("RGB", (a2.width + h2.width, H), (128, 128, 128))
    combined.paste(a2, (0, 0))
    combined.paste(h2, (a2.width, 0))
    combined_path = "/tmp/multiface_test.jpg"
    combined.save(combined_path, quality=92)
    print(f"  합성 이미지: {combined.size}  저장: {combined_path}")

    t0 = time.perf_counter()
    faces = detect_all_faces(combined, margin=0.1, min_face_size=20, min_conf=0.9, device="cuda")
    dt_ms = (time.perf_counter() - t0) * 1000
    print(f"\n탐지 {len(faces)}개 얼굴  (소요 {dt_ms:.1f}ms):")
    for i, f in enumerate(faces):
        x1, y1, x2, y2 = f["bbox"]
        w, h_ = x2 - x1, y2 - y1
        print(f"  [{i+1}] conf={f['confidence']:.3f}  bbox=({x1},{y1})-({x2},{y2})  size={w}×{h_}")
    return {"n_faces": len(faces), "faces": [{k: v for k, v in f.items() if k != "crop"} for f in faces],
            "time_ms": round(dt_ms, 1)}


def test_empty_bg():
    """회색 배경만 → MTCNN 0개 검출."""
    print("\n" + "=" * 72)
    print("[A-3] 얼굴 없는 이미지 (solid gray 512×512) fallback")
    print("=" * 72)
    blank = Image.new("RGB", (512, 512), (200, 200, 200))
    t0 = time.perf_counter()
    faces = detect_all_faces(blank, margin=0.1, min_face_size=20, min_conf=0.9, device="cuda")
    dt_ms = (time.perf_counter() - t0) * 1000
    print(f"  탐지: {len(faces)}개  소요: {dt_ms:.1f}ms")
    print(f"  → fallback 트리거 기대 (predict_multi_face 에서 원본 이미지 전체 사용)")
    return {"n_faces": len(faces), "time_ms": round(dt_ms, 1)}


if __name__ == "__main__":
    samples = pick_samples(n_per_class=5, seed=42)
    print(f"samples: {len(samples)} (class 5×4)")
    r1 = test_single_face(samples)
    r2 = test_multi_face(samples)
    r3 = test_empty_bg()

    print("\n" + "=" * 72)
    print("SUMMARY — 객체 탐지 (MTCNN)")
    print("=" * 72)
    print(f"  단일 val 20장 탐지 성공: {r1['detected']}/{r1['total']} (mean conf {r1['mean_conf']})")
    print(f"  합성 multi-face 2얼굴:   {r2['n_faces']}개 검출")
    print(f"  얼굴 없는 배경:          {r3['n_faces']}개 (0 기대)")

    out = {
        "single_face": r1,
        "multi_face": r2,
        "empty_bg": r3,
    }
    # rows 의 numpy 제거 (json serializable)
    out_path = PROJECT / "results/face_detection_test.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {out_path}")
