"""데이터 검증 (이미지 + 라벨 + 세그멘테이션 + 매칭).

기대 구조:
  data/
    img/{train,val}/{anger,happy,panic,sadness}/*.jpg
    label/{train,val}/{split}_{cls}.json     (encoding=euc-kr)
    segmentation/{train,val}/{split}_{cls}.npz

기대 수치: train 각 ~1500 / val 각 300, 총 7196장.
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

CLASSES = ["anger", "happy", "panic", "sadness"]
SPLITS = ["train", "val"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_IMG_TOTAL = 7196


def scan_images(img_root: Path):
    per = defaultdict(lambda: defaultdict(list))
    broken, zero_byte = [], []
    for split in SPLITS:
        for cls in CLASSES:
            d = img_root / split / cls
            if not d.is_dir():
                continue
            for f in d.iterdir():
                if f.suffix.lower() not in IMG_EXTS:
                    continue
                if f.stat().st_size == 0:
                    zero_byte.append(f)
                    continue
                per[split][cls].append(f)
    for split in SPLITS:
        for cls in CLASSES:
            for f in per[split][cls]:
                try:
                    with Image.open(f) as im:
                        im.verify()
                except (UnidentifiedImageError, OSError) as e:
                    broken.append((f, str(e)))
    return per, broken, zero_byte


def scan_labels(label_root: Path):
    per = defaultdict(dict)  # per[split][cls] = list of dicts
    for split in SPLITS:
        for cls in CLASSES:
            p = label_root / split / f"{split}_{cls}.json"
            if not p.is_file():
                per[split][cls] = None
                continue
            try:
                with open(p, encoding="euc-kr") as f:
                    per[split][cls] = json.load(f)
            except Exception as e:
                per[split][cls] = f"[ERR] {e}"
    return per


def scan_segmentation(seg_root: Path):
    per = defaultdict(dict)
    for split in SPLITS:
        for cls in CLASSES:
            p = seg_root / split / f"{split}_{cls}.npz"
            if not p.is_file():
                per[split][cls] = None
                continue
            try:
                npz = np.load(p, allow_pickle=True)
                first = npz.files[0] if npz.files else None
                shape = tuple(npz[first].shape) if first else None
                uniq = np.unique(npz[first]).tolist() if first else []
                per[split][cls] = {
                    "count": len(npz.files),
                    "sample_shape": shape,
                    "sample_values": uniq,
                }
            except Exception as e:
                per[split][cls] = f"[ERR] {e}"
    return per


def main():
    default_root = os.environ.get(
        "PROJECT_ROOT",
        str(Path(__file__).resolve().parent.parent),
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(default_root) / "data"))
    args = ap.parse_args()

    root = Path(args.root)
    img_root = root / "img"
    label_root = root / "label"
    seg_root = root / "segmentation"

    print(f"\n=== DATA ROOT: {root} ===")
    print(f"  img : {'OK' if img_root.is_dir() else 'MISSING'}")
    print(f"  label : {'OK' if label_root.is_dir() else 'MISSING'}")
    print(f"  segmentation : {'OK' if seg_root.is_dir() else 'MISSING'}")

    # 1) 이미지
    print("\n── 이미지 ──")
    img_per, broken, zero = scan_images(img_root)
    total = 0
    for split in SPLITS:
        print(f"[{split}]")
        sub = 0
        for cls in CLASSES:
            n = len(img_per[split][cls])
            sub += n
            print(f"  {cls:10s}: {n}")
        print(f"  합계: {sub}")
        total += sub
    print(f"전체: {total}  (기대 {EXPECTED_IMG_TOTAL},  일치: {total == EXPECTED_IMG_TOTAL})")
    print(f"깨진 이미지: {len(broken)}  0바이트: {len(zero)}")

    # 2) 라벨
    print("\n── 라벨 (encoding=euc-kr) ──")
    lbl = scan_labels(label_root)
    lbl_total = 0
    for split in SPLITS:
        print(f"[{split}]")
        for cls in CLASSES:
            v = lbl[split].get(cls)
            if v is None:
                print(f"  {cls:10s}: [파일 없음]")
            elif isinstance(v, str):
                print(f"  {cls:10s}: {v}")
            else:
                n = len(v)
                lbl_total += n
                keys = list(v[0].keys())[:5] if v else []
                has_bbox = v and "annot_A" in v[0]
                print(f"  {cls:10s}: {n} labels   bbox_annot_A={'Y' if has_bbox else 'N'}   top_keys={keys}")
    print(f"전체 라벨: {lbl_total}")

    # 3) 세그멘테이션
    print("\n── 세그멘테이션 (npz) ──")
    seg = scan_segmentation(seg_root)
    seg_total = 0
    for split in SPLITS:
        print(f"[{split}]")
        for cls in CLASSES:
            v = seg[split].get(cls)
            if v is None:
                print(f"  {cls:10s}: [파일 없음]")
            elif isinstance(v, str):
                print(f"  {cls:10s}: {v}")
            else:
                seg_total += v["count"]
                print(f"  {cls:10s}: {v['count']} masks  shape={v['sample_shape']}  vals={v['sample_values']}")
    print(f"전체 마스크: {seg_total}")

    # 4) 매칭 (이미지 vs 라벨 파일명)
    print("\n── 이미지↔라벨 매칭 ──")
    mismatch_total = 0
    for split in SPLITS:
        for cls in CLASSES:
            img_files = {f.name for f in img_per[split][cls]}
            v = lbl[split].get(cls)
            if not isinstance(v, list):
                continue
            label_files = {item.get("filename", "") for item in v if isinstance(item, dict)}
            matched = img_files & label_files
            only_img = img_files - label_files
            only_lbl = label_files - img_files
            mismatch_total += len(only_img) + len(only_lbl)
            print(f"  {split}/{cls:8s}: img={len(img_files)} lbl={len(label_files)} 매칭={len(matched)} img전용={len(only_img)} lbl전용={len(only_lbl)}")
    print(f"매칭 불일치 총 {mismatch_total}")

    # 5) 요약
    print("\n── 요약 ──")
    ok_img = total == EXPECTED_IMG_TOTAL and len(broken) == 0 and len(zero) == 0
    ok_lbl = lbl_total == total  # 라벨 수가 이미지 수와 같아야 함 (일반적으로)
    print(f"이미지 OK: {ok_img}")
    print(f"라벨 수 이미지와 일치: {ok_lbl} (lbl={lbl_total}, img={total})")
    print(f"Overall: {'PASS' if (ok_img and mismatch_total == 0) else 'CHECK'}")

    if broken:
        print("\n깨진 이미지 (최대 10):")
        for f, msg in broken[:10]:
            print(f"  {f}: {msg}")


if __name__ == "__main__":
    main()
