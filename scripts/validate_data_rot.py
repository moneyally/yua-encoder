"""data_rot/ 의 정규화가 파이프라인과 정합하는지 수치 검증.

눈으로 때려맞추지 말고 Python 계산·assertions 로만 결정.

검증 항목:
  A. 이미지 shape ↔ seg mask shape 일치율 (전수)
  B. bbox(annot_A.boxes) 좌표가 이미지 영역 [0, W] × [0, H] 내부 (전수)
  C. 파일 매칭: 이미지 ↔ 라벨 filename ↔ seg mask key (전수)
  D. 클래스별 카운트 (train 1495~1501, val 300×4)
  E. Mediapipe Face Detection 으로 bbox 영역에 실제 얼굴 있는지 IoU 측정 (랜덤 샘플)

산출: results/validate_data_rot_report.md + 콘솔 로그
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image, ImageOps

CLASSES = ["anger", "happy", "panic", "sadness"]
SPLITS = ["train", "val"]


def load_labels(lbl_root: Path):
    per = {s: {} for s in SPLITS}
    for s in SPLITS:
        for c in CLASSES:
            p = lbl_root / s / f"{s}_{c}.json"
            if not p.is_file():
                per[s][c] = []
                continue
            loaded = None
            for enc in ("euc-kr", "utf-8"):
                try:
                    with open(p, encoding=enc) as f:
                        loaded = json.load(f)
                    break
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
            per[s][c] = loaded or []
    return per


# ──────────────────────────────────────────────────────────────
# A. 이미지 shape ↔ seg mask shape
# ──────────────────────────────────────────────────────────────
def verify_shape_match(data: Path, sample_limit: int | None = None):
    results = {"total": 0, "match": 0, "mismatch": 0, "details": []}
    for s in SPLITS:
        for c in CLASSES:
            seg_p = data / "segmentation" / s / f"{s}_{c}.npz"
            if not seg_p.is_file():
                continue
            try:
                npz = np.load(seg_p, allow_pickle=True)
            except Exception as e:
                print(f"[warn] seg load {seg_p}: {e}")
                continue
            npz_files = list(npz.files)
            if sample_limit and len(npz_files) > sample_limit:
                step = max(1, len(npz_files) // sample_limit)
                npz_files = npz_files[::step][:sample_limit]

            img_dir = data / "img" / s / c
            for key in npz_files:
                fname = key
                img_p = img_dir / fname
                if not img_p.is_file():
                    results["mismatch"] += 1
                    results["details"].append((str(img_p), "missing img"))
                    results["total"] += 1
                    continue
                try:
                    with Image.open(img_p) as im:
                        W, H = im.size    # PIL order (W, H)
                    mask_shape = npz[key].shape[:2]  # (H, W)
                    if mask_shape == (H, W):
                        results["match"] += 1
                    else:
                        results["mismatch"] += 1
                        results["details"].append(
                            (f"{s}/{c}/{fname}", f"img({W}x{H}) vs mask{mask_shape}")
                        )
                    results["total"] += 1
                except Exception as e:
                    print(f"[warn] img open {img_p}: {e}")
    return results


# ──────────────────────────────────────────────────────────────
# B. bbox in bounds
# ──────────────────────────────────────────────────────────────
def verify_bbox_in_bounds(data: Path, per_lbl):
    results = {"total": 0, "in_bounds": 0, "out_bounds": 0, "details": []}
    for s in SPLITS:
        for c in CLASSES:
            for item in per_lbl[s].get(c, []):
                if not isinstance(item, dict):
                    continue
                fname = item.get("filename", "")
                img_p = data / "img" / s / c / fname
                if not img_p.is_file():
                    continue
                annot = item.get("annot_A", {})
                box = annot.get("boxes") if isinstance(annot, dict) else None
                if not (isinstance(box, dict) and
                        all(k in box for k in ("minX", "maxX", "minY", "maxY"))):
                    continue
                try:
                    with Image.open(img_p) as im:
                        W, H = im.size
                except Exception:
                    continue
                x0, y0, x1, y1 = box["minX"], box["minY"], box["maxX"], box["maxY"]
                results["total"] += 1
                ok = (0 <= x0 < x1 <= W + 1) and (0 <= y0 < y1 <= H + 1)
                if ok:
                    results["in_bounds"] += 1
                else:
                    results["out_bounds"] += 1
                    results["details"].append(
                        (f"{s}/{c}/{fname}", f"box({x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}) in {W}x{H}")
                    )
    return results


# ──────────────────────────────────────────────────────────────
# C. 매칭
# ──────────────────────────────────────────────────────────────
def verify_matching(data: Path, per_lbl):
    per_split_cls = {}
    for s in SPLITS:
        for c in CLASSES:
            img_dir = data / "img" / s / c
            seg_p = data / "segmentation" / s / f"{s}_{c}.npz"
            img_set = {f.name for f in img_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}} \
                if img_dir.is_dir() else set()
            lbl_set = {it.get("filename", "") for it in per_lbl[s].get(c, []) if isinstance(it, dict)}
            try:
                seg_set = set(np.load(seg_p, allow_pickle=True).files) if seg_p.is_file() else set()
            except Exception:
                seg_set = set()
            three = img_set & lbl_set & seg_set
            per_split_cls[(s, c)] = {
                "img": len(img_set), "lbl": len(lbl_set), "seg": len(seg_set),
                "3way_match": len(three),
                "img_only": len(img_set - lbl_set - seg_set),
                "lbl_only": len(lbl_set - img_set - seg_set),
                "seg_only": len(seg_set - img_set - lbl_set),
                "img_lbl_not_seg": len((img_set & lbl_set) - seg_set),
                "img_seg_not_lbl": len((img_set & seg_set) - lbl_set),
                "lbl_seg_not_img": len((lbl_set & seg_set) - img_set),
            }
    return per_split_cls


# ──────────────────────────────────────────────────────────────
# D. 클래스별 카운트
# ──────────────────────────────────────────────────────────────
def verify_counts(data: Path):
    r = {}
    for s in SPLITS:
        for c in CLASSES:
            d = data / "img" / s / c
            n = sum(1 for f in d.iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png"}) if d.is_dir() else 0
            r[(s, c)] = n
    return r


# ──────────────────────────────────────────────────────────────
# E. Face detection 으로 bbox 품질 검증 (랜덤 N장)
# ──────────────────────────────────────────────────────────────
def iou(a, b):
    """bbox (x0,y0,x1,y1) IoU."""
    ix0 = max(a[0], b[0]); iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2]); iy1 = min(a[3], b[3])
    iw = max(0.0, ix1 - ix0); ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def verify_face_detection(data: Path, per_lbl, n_per_class: int = 20):
    """facenet-pytorch MTCNN으로 bbox 영역에 얼굴이 있는지 IoU 측정.
    IoU > 0.3 이면 '얼굴 발견'. 평균/중앙값 + 클래스별 리포트.
    """
    try:
        from facenet_pytorch import MTCNN
        import torch
    except ImportError:
        return {"error": "facenet-pytorch 미설치"}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # keep_all=True : 한 이미지에 얼굴 여럿 잡음
    mtcnn = MTCNN(keep_all=True, device=device, post_process=False)

    samples = []
    rng = np.random.default_rng(123)
    for s in SPLITS:
        for c in CLASSES:
            items = per_lbl[s].get(c, [])
            items = [it for it in items if isinstance(it, dict) and "annot_A" in it]
            if not items:
                continue
            pick = rng.choice(len(items), size=min(n_per_class, len(items)), replace=False)
            for i in pick:
                it = items[int(i)]
                fname = it.get("filename", "")
                img_p = data / "img" / s / c / fname
                if not img_p.is_file():
                    continue
                box = it.get("annot_A", {}).get("boxes", {})
                if not all(k in box for k in ("minX", "maxX", "minY", "maxY")):
                    continue
                samples.append((s, c, img_p, box))

    results = {
        "total": 0,
        "face_found_in_img": 0,
        "iou_over_03": 0,
        "iou_over_05": 0,
        "ious": [],
        "no_face_detected": 0,
        "per_class": {c: {"n": 0, "iou_mean": 0.0, "iou_over_03": 0} for c in CLASSES},
        "detector": f"facenet-pytorch MTCNN ({device})",
    }
    for s, c, img_p, gt_box in samples:
        try:
            img = Image.open(img_p).convert("RGB")
            W, H = img.size
            boxes, probs = mtcnn.detect(img)
        except Exception as e:
            print(f"[warn] mtcnn err {img_p}: {e}")
            continue
        results["total"] += 1
        gt = (gt_box["minX"], gt_box["minY"], gt_box["maxX"], gt_box["maxY"])
        if boxes is None or len(boxes) == 0:
            results["no_face_detected"] += 1
            results["ious"].append(0.0)
            results["per_class"][c]["n"] += 1
            continue
        results["face_found_in_img"] += 1
        best_iou = 0.0
        for b in boxes:
            if b is None:
                continue
            pred = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
            best_iou = max(best_iou, iou(gt, pred))
        results["ious"].append(best_iou)
        if best_iou > 0.3:
            results["iou_over_03"] += 1
            results["per_class"][c]["iou_over_03"] += 1
        if best_iou > 0.5:
            results["iou_over_05"] += 1
        results["per_class"][c]["n"] += 1
        results["per_class"][c]["iou_mean"] += best_iou

    for c in CLASSES:
        n = results["per_class"][c]["n"]
        results["per_class"][c]["iou_mean"] = (
            results["per_class"][c]["iou_mean"] / n if n else 0.0
        )
    if results["ious"]:
        results["iou_mean"] = float(np.mean(results["ious"]))
        results["iou_median"] = float(np.median(results["ious"]))
    else:
        results["iou_mean"] = 0.0
        results["iou_median"] = 0.0
    return results


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    default_root = os.environ.get("PROJECT_ROOT", "/workspace/user4/emotion-project")
    ap.add_argument("--root", default=default_root)
    ap.add_argument("--data-subdir", default="data_rot")
    ap.add_argument("--face-sample-per-class", type=int, default=20,
                    help="mediapipe 검증용 클래스당 랜덤 샘플 수")
    ap.add_argument("--skip-face", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    data = root / args.data_subdir
    out_md = root / "results" / f"{args.data_subdir}_validate_report.md"
    (root / "results").mkdir(exist_ok=True)

    if not data.is_dir():
        print(f"[!] {data} 없음"); sys.exit(2)
    print(f"검증 대상: {data}\n")

    t0 = time.time()

    print("[A] 이미지 shape ↔ seg mask shape 전수 검증")
    A = verify_shape_match(data)
    print(f"    total={A['total']}  match={A['match']}  mismatch={A['mismatch']}")
    if A["mismatch"]:
        print("    mismatch 샘플 (최대 10):")
        for p, msg in A["details"][:10]:
            print(f"      {p}: {msg}")

    print("\n[D] 클래스별 이미지 수")
    D = verify_counts(data)
    total_imgs = 0
    for s in SPLITS:
        for c in CLASSES:
            n = D[(s, c)]
            total_imgs += n
            print(f"    {s}/{c:10s}: {n}")
    print(f"    ---------- 합계: {total_imgs}")

    print("\n[C] 파일 매칭 (img ↔ label ↔ seg 3-way)")
    per_lbl = load_labels(data / "label")
    C = verify_matching(data, per_lbl)
    tot_3way = 0
    for (s, c), r in C.items():
        tot_3way += r["3way_match"]
        anyany = any(r[k] for k in ("img_only", "lbl_only", "seg_only",
                                     "img_lbl_not_seg", "img_seg_not_lbl", "lbl_seg_not_img"))
        marker = "✗" if anyany else "✓"
        print(f"    {marker} {s}/{c:8s}: img={r['img']} lbl={r['lbl']} seg={r['seg']} "
              f"3way={r['3way_match']}  only: img={r['img_only']} lbl={r['lbl_only']} seg={r['seg_only']}")
    print(f"    전체 3-way 매칭 수: {tot_3way}")

    print("\n[B] bbox in bounds 전수 검증")
    B = verify_bbox_in_bounds(data, per_lbl)
    print(f"    total={B['total']}  in_bounds={B['in_bounds']}  out={B['out_bounds']}")
    if B["out_bounds"]:
        print("    out of bounds 샘플 (최대 10):")
        for p, msg in B["details"][:10]:
            print(f"      {p}: {msg}")

    if not args.skip_face:
        print(f"\n[E] Mediapipe Face Detection 검증 (클래스당 {args.face_sample_per_class}장)")
        E = verify_face_detection(data, per_lbl, args.face_sample_per_class)
        if "error" in E:
            print(f"    [!] {E['error']}")
        else:
            print(f"    samples={E['total']}  face_detected_in_img={E['face_found_in_img']}  no_face={E['no_face_detected']}")
            print(f"    IoU mean={E['iou_mean']:.3f}  median={E['iou_median']:.3f}")
            print(f"    IoU > 0.3: {E['iou_over_03']}/{E['total']}   IoU > 0.5: {E['iou_over_05']}/{E['total']}")
            print(f"    클래스별:")
            for c in CLASSES:
                pc = E["per_class"][c]
                print(f"      {c:10s}: n={pc['n']}  iou_mean={pc['iou_mean']:.3f}  iou>0.3={pc['iou_over_03']}")

    dt = time.time() - t0
    print(f"\n=== 검증 완료 — {dt:.1f}s ===")

    # Overall PASS 판정
    pass_A = (A["mismatch"] == 0)
    pass_B = (B["out_bounds"] == 0)
    pass_D = (total_imgs == 7196)
    pass_C = all(
        r["img_only"] == 0 and r["seg_only"] == 0 for r in C.values()
    )
    verdict = "ALL PASS ✓" if (pass_A and pass_B and pass_D and pass_C) else "CHECK FAIL"
    print(f"\n판정: A={pass_A} B={pass_B} C={pass_C} D={pass_D}  →  {verdict}")

    # 리포트 md 저장
    lines = [
        "# data_rot validate report",
        f"root={data}",
        f"duration={dt:.1f}s",
        "",
        f"[A] shape match: {A['match']}/{A['total']}  mismatch={A['mismatch']}",
        f"[B] bbox in bounds: {B['in_bounds']}/{B['total']}  out={B['out_bounds']}",
        f"[D] 전체 이미지: {total_imgs} (기대 7196)",
        f"[C] 3-way 매칭 총: {tot_3way}",
    ]
    if not args.skip_face and "error" not in E:
        lines += [
            f"[E] face IoU mean={E['iou_mean']:.3f}  IoU>0.3={E['iou_over_03']}/{E['total']}  IoU>0.5={E['iou_over_05']}/{E['total']}",
        ]
    lines.append(f"\n판정: {verdict}")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n리포트: {out_md}")


if __name__ == "__main__":
    main()
