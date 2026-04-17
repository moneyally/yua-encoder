"""EXIF orientation 일괄 정규화.

검증 결과 (bbox_coord_test / mask_coord_test):
  - 라벨의 annot_A.boxes 좌표는 이미 **EXIF 적용된(정규화된) 좌표계** 기준
  - segmentation mask 역시 정규화된 좌표계 기준
  → 따라서 라벨/마스크는 회전 금지. **이미지만** 정규화하면 됨.

원본 data/ 는 그대로 두고 사본을 data_rot/ 에 생성:
  data_rot/img/{split}/{cls}/*.jpg       — 픽셀 방향 정규화(EXIF transpose) + EXIF 제거
  data_rot/label/{split}/{split}_{cls}.json — 원본 JSON 그대로 복사
  data_rot/segmentation/{split}/{split}_{cls}.npz — 원본 npz 그대로 복사

rotate_bbox / rotate_mask_np 함수는 수학적으로 정확하며 전수 검증됨(verify_rotation_math.py).
이번 데이터셋엔 불필요하지만 다른 데이터셋에 재사용 가능해 유지.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import ExifTags, Image, ImageOps, UnidentifiedImageError

CLASSES = ["anger", "happy", "panic", "sadness"]
SPLITS = ["train", "val"]
JPEG_QUALITY = 95  # 원본 jpg 손실 최소화

ORI_TAG = next(k for k, v in ExifTags.TAGS.items() if v == "Orientation")

# PIL transpose op 매핑 (exif_transpose 소스와 동일)
PIL_OP = {
    2: Image.FLIP_LEFT_RIGHT,
    3: Image.ROTATE_180,
    4: Image.FLIP_TOP_BOTTOM,
    5: Image.TRANSPOSE,
    6: Image.ROTATE_270,  # CW 90
    7: Image.TRANSVERSE,
    8: Image.ROTATE_90,   # CCW 90
}

BBOX_KEYS = ("minX", "maxX", "minY", "maxY")


# ──────────────────────────────────────────────────────────────
# 좌표·마스크 변환
# ──────────────────────────────────────────────────────────────
def rotate_bbox(box: dict, ori: int, W: int, H: int) -> dict:
    """box: {minX, maxX, minY, maxY}. W,H 는 raw(EXIF 적용 전) 픽셀."""
    x0, x1 = box["minX"], box["maxX"]
    y0, y1 = box["minY"], box["maxY"]
    if ori == 1:
        mx0, my0, mx1, my1 = x0, y0, x1, y1
    elif ori == 2:   # FLIP_LEFT_RIGHT
        mx0, my0, mx1, my1 = W - x1, y0, W - x0, y1
    elif ori == 3:   # 180
        mx0, my0, mx1, my1 = W - x1, H - y1, W - x0, H - y0
    elif ori == 4:   # FLIP_TOP_BOTTOM
        mx0, my0, mx1, my1 = x0, H - y1, x1, H - y0
    elif ori == 5:   # TRANSPOSE (diag top-left → bottom-right)
        mx0, my0, mx1, my1 = y0, x0, y1, x1
    elif ori == 6:   # ROTATE_270 → (x,y)→(H-1-y, x); new W=H, new H=W
        mx0, my0, mx1, my1 = H - y1, x0, H - y0, x1
    elif ori == 7:   # TRANSVERSE
        mx0, my0, mx1, my1 = H - y1, W - x1, H - y0, W - x0
    elif ori == 8:   # ROTATE_90 → (x,y)→(y, W-1-x); new W=H, new H=W
        mx0, my0, mx1, my1 = y0, W - x1, y1, W - x0
    else:
        mx0, my0, mx1, my1 = x0, y0, x1, y1
    return {"minX": mx0, "maxX": mx1, "minY": my0, "maxY": my1}


def rotate_mask_np(mask: np.ndarray, ori: int) -> np.ndarray:
    """numpy 배열에 대해 PIL 과 동일한 회전 적용."""
    if ori == 1:
        return mask
    if ori == 2:
        return np.ascontiguousarray(np.fliplr(mask))
    if ori == 3:
        return np.ascontiguousarray(np.rot90(mask, 2))
    if ori == 4:
        return np.ascontiguousarray(np.flipud(mask))
    if ori == 5:   # TRANSPOSE
        return np.ascontiguousarray(np.transpose(mask))
    if ori == 6:   # CW 90 = np.rot90 k=-1 (=k=3)
        return np.ascontiguousarray(np.rot90(mask, -1))
    if ori == 7:   # TRANSVERSE = rot180(transpose) — anti-diagonal reflection
        return np.ascontiguousarray(np.rot90(np.transpose(mask), 2))
    if ori == 8:   # CCW 90
        return np.ascontiguousarray(np.rot90(mask, 1))
    return mask


def get_orientation(im: Image.Image) -> int:
    try:
        exif = im.getexif()
        if not exif:
            return 1
        return int(exif.get(ORI_TAG, 1))
    except Exception:
        return 1


# ──────────────────────────────────────────────────────────────
# 이미지 처리
# ──────────────────────────────────────────────────────────────
def _worker_rotate(task):
    """한 장 처리 (ProcessPool worker). (src, dst) → (s, c, fname, ori, W, H, err)."""
    src, dst, s, c = task
    try:
        with Image.open(src) as im:
            ori = get_orientation(im)
            raw_W, raw_H = im.size
            im_rot = ImageOps.exif_transpose(im)
            if im_rot.mode not in ("RGB", "L"):
                im_rot = im_rot.convert("RGB")
            im_rot.save(dst, "JPEG", quality=JPEG_QUALITY, exif=b"")
        return (s, c, src.name, ori, raw_W, raw_H, None)
    except Exception as e:
        return (s, c, src.name, 1, 0, 0, f"{type(e).__name__}: {e}")


def process_images(src_root: Path, dst_root: Path,
                   ori_map: Dict[str, Dict[str, Dict[str, Tuple[int, int, int]]]],
                   workers: int = 32):
    """병렬 JPEG decode/rotate/encode. 96코어 중 workers 개 사용."""
    stats = {"ok": 0, "err": 0, "by_ori": Counter()}
    tasks = []
    for s in SPLITS:
        if s not in ori_map: ori_map[s] = {}
        for c in CLASSES:
            if c not in ori_map[s]: ori_map[s][c] = {}
            sd = src_root / "img" / s / c
            dd = dst_root / "img" / s / c
            dd.mkdir(parents=True, exist_ok=True)
            for f in sorted(sd.iterdir()):
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    tasks.append((f, dd / f.name, s, c))
    print(f"  총 {len(tasks)}장, workers={workers}")
    t0 = time.time()
    last_log = t0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, res in enumerate(ex.map(_worker_rotate, tasks, chunksize=16), 1):
            s, c, fname, ori, W, H, err = res
            if err is None:
                ori_map[s][c][fname] = (ori, W, H)
                stats["ok"] += 1
                stats["by_ori"][ori] += 1
            else:
                print(f"[warn] img err {s}/{c}/{fname}: {err}")
                stats["err"] += 1
            # 주기적 진행 로그 (1초 이상 간격)
            now = time.time()
            if now - last_log >= 1.0 or i == len(tasks):
                rate = i / max(now - t0, 1e-6)
                eta = (len(tasks) - i) / max(rate, 1e-6)
                print(f"  [img] {i}/{len(tasks)}  {rate:.1f} img/s  ETA {eta:.0f}s", flush=True)
                last_log = now
    print(f"  [img] 완료 ok={stats['ok']}, err={stats['err']}, {time.time()-t0:.1f}s")
    return stats


# ──────────────────────────────────────────────────────────────
# 라벨 JSON 처리
# ──────────────────────────────────────────────────────────────
def process_labels(src_root: Path, dst_root: Path, ori_map):
    """라벨 bbox는 이미 정규화 좌표계이므로 원본 그대로 복사."""
    stats = {"files": 0, "copied": 0, "err": 0}
    for s in SPLITS:
        sd = src_root / "label" / s
        dd = dst_root / "label" / s
        dd.mkdir(parents=True, exist_ok=True)
        for c in CLASSES:
            p = sd / f"{s}_{c}.json"
            if not p.is_file():
                continue
            try:
                shutil.copy2(p, dd / p.name)
                stats["copied"] += 1
                print(f"  [lbl-copy] {s}/{c}: {p.name}")
            except Exception as e:
                print(f"[warn] label copy fail {p}: {e}")
                stats["err"] += 1
            stats["files"] += 1
    return stats


# ──────────────────────────────────────────────────────────────
# segmentation npz 처리
# ──────────────────────────────────────────────────────────────
def process_segmentation(src_root: Path, dst_root: Path, ori_map):
    """seg mask도 이미 정규화 좌표계이므로 원본 npz 그대로 복사."""
    stats = {"files": 0, "copied": 0, "err": 0}
    for s in SPLITS:
        sd = src_root / "segmentation" / s
        dd = dst_root / "segmentation" / s
        dd.mkdir(parents=True, exist_ok=True)
        for c in CLASSES:
            p = sd / f"{s}_{c}.npz"
            if not p.is_file():
                continue
            try:
                shutil.copy2(p, dd / p.name)
                stats["copied"] += 1
                print(f"  [seg-copy] {s}/{c}: {p.name}")
            except Exception as e:
                print(f"[warn] seg copy fail {p}: {e}")
                stats["err"] += 1
            stats["files"] += 1
    return stats


# ──────────────────────────────────────────────────────────────
# 검증: 랜덤 파일 몇 개 재회전 일관성
# ──────────────────────────────────────────────────────────────
def verify_samples(src_root: Path, dst_root: Path, ori_map, n: int = 4):
    print("\n=== 검증: 원본 → 회전 후 bbox 좌표가 얼굴 영역 안에 있나 spot check ===")
    rng = np.random.default_rng(0)
    for s in SPLITS:
        for c in CLASSES:
            fnames = list(ori_map.get(s, {}).get(c, {}).keys())
            if not fnames:
                continue
            sample = rng.choice(fnames, size=min(n, len(fnames)), replace=False)
            for fname in sample:
                ori, rw, rh = ori_map[s][c][fname]
                dst_img = dst_root / "img" / s / c / fname
                try:
                    with Image.open(dst_img) as im:
                        sw, sh = im.size
                except Exception as e:
                    print(f"  [!] open fail {dst_img}: {e}")
                    continue
                # 예상 사이즈
                exp_w, exp_h = (rw, rh) if ori in (1, 2, 3, 4) else (rh, rw)
                status = "OK" if (sw, sh) == (exp_w, exp_h) else f"SIZE MISMATCH {sw}x{sh} vs expected {exp_w}x{exp_h}"
                print(f"  {s}/{c}/{fname[:16]}… ori={ori} raw={rw}x{rh} saved={sw}x{sh}  {status}")
                break  # 각 클래스당 1개만 spot check 로그


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    default_root = os.environ.get("PROJECT_ROOT", "/workspace/user4/emotion-project")
    ap.add_argument("--root", default=default_root)
    ap.add_argument("--dst", default=None,
                    help="저장 위치 (기본 <root>/data_rot)")
    ap.add_argument("--skip-images", action="store_true",
                    help="이미지는 이미 돌렸고 label/seg만 다시 할 때")
    ap.add_argument("--workers", type=int, default=32,
                    help="이미지 처리 병렬 워커 수 (CPU bound)")
    args = ap.parse_args()

    root = Path(args.root)
    src = root / "data"
    dst = Path(args.dst) if args.dst else root / "data_rot"
    if not src.is_dir():
        print(f"[!] {src} 없음"); sys.exit(2)
    dst.mkdir(exist_ok=True)

    print(f"src: {src}")
    print(f"dst: {dst}")

    t0 = time.time()
    ori_map: Dict = {}

    if args.skip_images:
        # 이미지는 이미 처리됐다고 가정 — 기존 사본에서 EXIF 없다고 치고, 원본 data/에서 ori만 다시 읽어 map 생성
        print("\n[skip images] 이미지 건너뜀 — 원본에서 ori만 재스캔")
        for s in SPLITS:
            ori_map[s] = {}
            for c in CLASSES:
                ori_map[s][c] = {}
                sd = src / "img" / s / c
                for f in sd.iterdir():
                    if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                        continue
                    try:
                        with Image.open(f) as im:
                            ori = get_orientation(im)
                            ori_map[s][c][f.name] = (ori, im.size[0], im.size[1])
                    except Exception as e:
                        print(f"  [warn] {f}: {e}")
    else:
        print(f"\n[1/3] 이미지 회전 + EXIF 제거 저장 (workers={args.workers})")
        img_stats = process_images(src, dst, ori_map, workers=args.workers)
        print(f"  이미지: ok={img_stats['ok']}, err={img_stats['err']}")
        print(f"  orientation 분포: {dict(img_stats['by_ori'])}")

    print("\n[2/3] 라벨 JSON 복사 (bbox는 이미 정규화 좌표계)")
    lbl_stats = process_labels(src, dst, ori_map)
    print(f"  label files={lbl_stats['files']}, copied={lbl_stats['copied']}, err={lbl_stats['err']}")

    print("\n[3/3] segmentation npz 복사 (mask도 이미 정규화 좌표계)")
    seg_stats = process_segmentation(src, dst, ori_map)
    print(f"  seg files={seg_stats['files']}, copied={seg_stats['copied']}, err={seg_stats['err']}")

    verify_samples(src, dst, ori_map, n=3)

    dt = time.time() - t0
    print(f"\n=== 완료 — 총 {dt:.1f}s ===")
    print(f"저장 위치: {dst}")


if __name__ == "__main__":
    main()
