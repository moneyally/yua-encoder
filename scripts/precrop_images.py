"""precrop_images.py — data_rot 의 이미지를 bbox crop + resize 하여 JPEG 로 캐싱.

병목 해결 목적:
  원본 data_rot 의 이미지는 median 3088×2208 고해상도. 학습 루프에서 매 epoch
  PIL.open → convert RGB → bbox crop → resize 가 CPU-bound 병목이 된다.
  이 스크립트는 1회 실행으로 bbox crop + target_size resize 된 JPEG 를 저장하여,
  이후 학습 시 Dataset 이 작은 이미지를 바로 open 만 하면 되게 한다.

산출물:
  data_rot_crop<N>/
    img/{train,val}/<class>/<filename>.jpg    # 각 N×N JPEG (bbox crop 후 resize)
    label/  seg/                               # 원본 그대로 심볼릭 링크 (참고용)

사용:
  python scripts/precrop_images.py --target-size 224  # ViT 용
  python scripts/precrop_images.py --target-size 384  # SigLIP 용
  python scripts/precrop_images.py --target-size 224 --quality 92 --force  # 재생성

학습 스크립트에서:
  python scripts/train_vit.py --data data_rot_crop224 --crop ...
  (단, crop=True 옵션은 bbox crop 이 이미 돼있으니 내부에서 no-op 에 가까움
   → Dataset 의 crop 분기는 bbox 가 전체 이미지 범위로 돼있어야. 아래 주의 참조)

주의:
  - bbox 가 이미 적용된 이미지에 대해 train_vit.py 가 crop 다시 하면 중복.
    따라서 precrop 데이터로 학습 시 **--crop 끄고** 일반 image_dataset 경로로
    가거나, label JSON 의 bbox 를 (0, 0, W, H) 로 재생성해 주어야 한다.
    이 스크립트는 전자를 가정 (crop=False 로 학습) — 더 단순.
  - 원본 라벨 JSON 은 그대로 두고 이미지만 crop. label 파싱이 필요 없는
    폴더 기반 학습 (crop=False) 에서 바로 쓰임.
  - 9건 bbox 음수, 2건 area<=1, 2건 라벨 누락은 skip 또는 전체 이미지.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageOps

# 경로: 환경변수 우선, 없으면 __file__ 기반
PROJECT = Path(os.environ.get("EMOTION_PROJECT_ROOT",
                              str(Path(__file__).resolve().parent.parent)))

CLASSES = ("anger", "happy", "panic", "sadness")
SPLITS = ("train", "val")


def _load_labels(data_root: Path, split: str, cls: str) -> dict[str, dict]:
    """split/cls json 파일 로드. {filename: {bbox_xyxy: (x0,y0,x1,y1)}} 반환.
    라벨 없거나 bbox 없으면 빈 dict.
    """
    out: dict[str, dict] = {}
    p = data_root / "label" / split / f"{split}_{cls}.json"
    if not p.is_file():
        return out
    loaded = None
    for enc in ("euc-kr", "utf-8"):
        try:
            with open(p, encoding=enc) as f:
                loaded = json.load(f)
            break
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    if not loaded:
        return out
    for item in loaded:
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
        out[fname] = {"bbox": (x0, y0, x1, y1)}
    return out


def _process_one(args: tuple) -> tuple[str, bool, str]:
    """한 이미지: PIL open + clip bbox + crop + resize + JPEG save.
    반환: (상태 태그, 성공여부, 메시지)
    """
    (src_path, dst_path, bbox, target_size, quality) = args
    try:
        with Image.open(src_path) as im:
            im.load()
            im = ImageOps.exif_transpose(im)  # data_rot 는 이미 정규화지만 안전
            if im.mode != "RGB":
                im = im.convert("RGB")
            W, H = im.size
            if bbox is not None:
                x0, y0, x1, y1 = bbox
                # clip to [0, W] x [0, H]
                x0 = max(0.0, min(float(W), x0))
                x1 = max(0.0, min(float(W), x1))
                y0 = max(0.0, min(float(H), y0))
                y1 = max(0.0, min(float(H), y1))
                if (x1 - x0) <= 1 or (y1 - y0) <= 1:
                    # bbox 무효 → 전체 이미지 사용
                    cropped = im
                    tag = "no_crop_badbbox"
                else:
                    cropped = im.crop((int(x0), int(y0),
                                       int(round(x1)), int(round(y1))))
                    tag = "cropped"
            else:
                cropped = im
                tag = "no_crop_nolabel"
            # 정사각형으로 resize (학습 기준 이미지)
            resized = cropped.resize((target_size, target_size), Image.BILINEAR)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            resized.save(dst_path, "JPEG", quality=quality, optimize=True)
        return (tag, True, "")
    except Exception as e:
        return ("error", False, str(e))


def main():
    ap = argparse.ArgumentParser(description="bbox crop + resize 된 이미지 JPEG 캐시 생성")
    ap.add_argument("--data-root", default=str(PROJECT / "data_rot"),
                    help="입력 루트 (EXIF 정규화 완료). 기본 $EMOTION_PROJECT_ROOT/data_rot")
    ap.add_argument("--out-root", default=None,
                    help="출력 루트. 기본은 data_root 와 같은 부모 + '_crop<target>' 접미")
    ap.add_argument("--target-size", type=int, required=True,
                    help="정사각형 resize 목표 크기 (예: 224, 384).")
    ap.add_argument("--quality", type=int, default=92,
                    help="JPEG quality (80~95 권장).")
    ap.add_argument("--workers", type=int, default=max(4, (os.cpu_count() or 4) - 2),
                    help="병렬 프로세스 수.")
    ap.add_argument("--force", action="store_true",
                    help="이미 존재해도 덮어쓰기.")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    if not (data_root / "img").is_dir():
        raise SystemExit(f"[!] {data_root}/img 없음. --data-root 확인")

    out_root = Path(args.out_root) if args.out_root else \
        data_root.with_name(f"{data_root.name}_crop{args.target_size}")
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[precrop] src={data_root}  dst={out_root}  size={args.target_size}")

    tasks: list[tuple] = []
    label_stats = {"with_bbox": 0, "no_label": 0}
    for split in args.splits:
        for cls in CLASSES:
            src_cls_dir = data_root / "img" / split / cls
            if not src_cls_dir.is_dir():
                continue
            dst_cls_dir = out_root / "img" / split / cls
            dst_cls_dir.mkdir(parents=True, exist_ok=True)

            labels = _load_labels(data_root, split, cls)
            for img_path in src_cls_dir.iterdir():
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                dst = dst_cls_dir / (img_path.stem + ".jpg")
                if dst.is_file() and not args.force:
                    continue
                lbl = labels.get(img_path.name)
                if lbl is not None:
                    bbox = lbl["bbox"]
                    label_stats["with_bbox"] += 1
                else:
                    bbox = None
                    label_stats["no_label"] += 1
                tasks.append((str(img_path), dst, bbox, args.target_size, args.quality))

    print(f"[precrop] 작업 {len(tasks)}장  "
          f"(with_bbox={label_stats['with_bbox']} no_label={label_stats['no_label']})")
    if not tasks:
        print("[precrop] 할 일 없음 (이미 캐시됐거나 입력 없음). --force 고려.")
        return

    start = time.time()
    done = 0
    errors = 0
    tag_counts: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_process_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), start=1):
            tag, ok, msg = fut.result()
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
            if not ok:
                errors += 1
                if errors <= 5:
                    print(f"  [err] {msg}")
            done = i
            if done % 500 == 0:
                elapsed = time.time() - start
                rate = done / max(1e-6, elapsed)
                eta = (len(tasks) - done) / max(1e-6, rate)
                print(f"  [progress] {done}/{len(tasks)}  "
                      f"{rate:.1f} img/s  ETA {eta:.0f}s")

    elapsed = time.time() - start
    print(f"[precrop] 완료  {done}/{len(tasks)}  elapsed={elapsed:.1f}s  "
          f"rate={done/max(1e-6,elapsed):.1f} img/s  errors={errors}")
    print(f"[precrop] 태그 분포: {tag_counts}")

    # label/, segmentation/ 심볼릭 링크 (참고용, 원본 그대로 재사용 가능)
    for sub in ("label", "segmentation"):
        src_sub = data_root / sub
        dst_sub = out_root / sub
        if src_sub.is_dir() and not dst_sub.exists():
            try:
                dst_sub.symlink_to(src_sub)
                print(f"[precrop] symlink {dst_sub} → {src_sub}")
            except OSError as e:
                print(f"  [warn] symlink 실패 ({e}) — 무시")

    print(f"[precrop] DONE → {out_root}")


if __name__ == "__main__":
    main()
