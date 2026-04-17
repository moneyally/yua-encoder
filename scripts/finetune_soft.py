"""emotion-project / finetune_soft.py — soft-label 미세조정 (ViT 전제).

===============================================================================
설계 (DESIGN)
===============================================================================

목적
----
기존 best ViT checkpoint (예: models/exp05_vit_b16_two_stage.pt) 를 읽어
`results/soft_labels/train_soft.npz` 의 3인 vote soft target 으로 **짧게·낮은 lr** 로
미세조정. 재학습이 아닌 label noise regularization + 3rd-party rater 반영.

핵심 수식 — Soft CrossEntropy
-----------------------------
loss(x, q) = -Σ_c q_c · log softmax(f(x))_c

이는 KL(q || p) + H(q) 과 동치 (H(q) 는 x-independent → gradient 에 무관).
따라서 KL-div 로 학습하는 것과 수학적으로 동일. 구현은 NumPy-stable 을 위해
F.log_softmax 사용.

출력 기준(monitor) 은 soft metric 이 아니라 **hard val accuracy** (기존 E5 와 직접 비교).
soft loss 는 회귀적 지표라 hard task 성능을 보증하지 않음.

입력
----
- --src-model: 기존 ViT ckpt (.pt, train_vit.py 포맷 {model, args, meta})
- --soft-labels: results/soft_labels/train_soft.npz
- --val-dir / --val-label-root: data_rot/img/val, data_rot/label (hard val)

파이프라인
----------
1) ckpt 읽기 → meta['model_name'] 로 timm.create_model(pretrained=False)
   state_dict strict=False load
2) train records: soft npz 에서 filename/gt/soft 읽음.
   img path 는 data_rot/img/train/<CLASSES[gt]>/<filename>
   (없으면 drop + warn)
3) crop 여부는 --crop flag. on 이면 label json 의 annot_A.boxes 로
   build_crop_records 처럼 bbox 적용.
4) val records: train_vit 와 동일하게 build_crop_records(val) or folder_records(val).
5) train_transform / val_transform: make_transforms (train_vit.py 재사용).
6) optimizer AdamW(lr=5e-6 default, wd=1e-5 default), fixed lr (짧은 epoch).
7) loss = soft_ce (hard CE 아님)
8) val = nn.CrossEntropyLoss (hard, 비교용)
9) best val_acc 저장 (prev_val_acc 와 비교 → meta 에 같이 기록)

CLI
---
--src-model (required)
--soft-labels (required)
--name (required)
--val-dir (default: data_rot/img/val)
--val-label-root (default: data_rot/label)
--data-root (default: data_rot)  — train 이미지 경로 만들기 용
--lr (default 5e-6)
--epochs (default 2)
--weight-decay (default 1e-5)
--batch-size (default 32)
--img-size (default 224)
--crop (flag, default off — src 모델이 crop=True 로 학습됐으면 켜야 함)
--augment (flag, default off)
--amp (bf16/fp16/off, default bf16)
--seed (default 42)
--patience (default 1)
--monitor (val_accuracy only — soft label 에선 val_loss/f1 지표 해석 애매)
--num-workers (default 4)
--prefetch-factor (default 2)
--note (default "")
--grad-clip-norm (default 1.0)

산출물
------
- models/<name>.pt  (dict: model, args, meta)
- logs/<name>.csv    (epoch, train_loss, train_soft_argmax_acc, val_loss, val_acc)
- logs/<name>.meta.json
- experiments.md 한 줄 append
===============================================================================
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import timm
from torchvision import transforms as T

# train_vit.py 재사용 — 같은 디렉토리라 sys.path append
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# 재사용: transforms / record builder / seed / worker_init
from train_vit import (  # noqa: E402
    CLASSES, NUM_CLASSES, IMAGENET_MEAN, IMAGENET_STD,
    make_transforms, build_crop_records, build_folder_records,
    load_labels_per_split, set_seed, _make_worker_init,
)

PROJECT = Path(os.environ.get(
    "EMOTION_PROJECT_ROOT",
    str(Path(__file__).resolve().parent.parent),
))
DEFAULT_DATA = PROJECT / "data_rot"


# --------------------------------------------------------------------------
# Dataset — soft target 을 반환
# --------------------------------------------------------------------------
class SoftLabelDataset(Dataset):
    """record = {path, class_idx, bbox_xyxy(optional), soft (np.ndarray[4])}.
    __getitem__ 반환: (image_tensor, soft_tensor[4], hard_label_int)
    hard_label 은 train-loop 에선 안 쓰지만 디버깅용으로 tuple 에 포함.
    """
    def __init__(self, records: list[dict], transform=None):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        img = Image.open(r["path"]).convert("RGB")
        bbox = r.get("bbox_xyxy")
        if bbox is not None:
            W, H = img.size
            x0, y0, x1, y1 = bbox
            x0 = max(0.0, min(float(W), float(x0)))
            x1 = max(0.0, min(float(W), float(x1)))
            y0 = max(0.0, min(float(H), float(y0)))
            y1 = max(0.0, min(float(H), float(y1)))
            if (x1 - x0) > 1 and (y1 - y0) > 1:
                img = img.crop((
                    int(x0), int(y0),
                    int(math.ceil(x1)), int(math.ceil(y1)),
                ))
        if self.transform is not None:
            img = self.transform(img)
        soft = torch.as_tensor(r["soft"], dtype=torch.float32)
        # 방어: sum=1 (부동소수 오차 재보정)
        s = float(soft.sum().item())
        if not math.isfinite(s) or s <= 0:
            # fallback: hard one-hot(gt)
            soft = torch.zeros(NUM_CLASSES, dtype=torch.float32)
            soft[int(r["class_idx"])] = 1.0
        elif abs(s - 1.0) > 1e-4:
            soft = soft / s
        return img, soft, int(r["class_idx"])


class HardLabelDataset(Dataset):
    """val 용. train_vit.EmotionViTDataset 과 동일 (import 로 가져올 수도 있으나
    soft 와 대칭 유지를 위해 여기 복제)."""
    def __init__(self, records: list[dict], transform=None):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        img = Image.open(r["path"]).convert("RGB")
        bbox = r.get("bbox_xyxy")
        if bbox is not None:
            W, H = img.size
            x0, y0, x1, y1 = bbox
            x0 = max(0.0, min(float(W), float(x0)))
            x1 = max(0.0, min(float(W), float(x1)))
            y0 = max(0.0, min(float(H), float(y0)))
            y1 = max(0.0, min(float(H), float(y1)))
            if (x1 - x0) > 1 and (y1 - y0) > 1:
                img = img.crop((
                    int(x0), int(y0),
                    int(math.ceil(x1)), int(math.ceil(y1)),
                ))
        if self.transform is not None:
            img = self.transform(img)
        return img, int(r["class_idx"])


# --------------------------------------------------------------------------
# val records override (--val-dir / --val-label-root 지정 시)
# --------------------------------------------------------------------------
def _build_val_records_override(val_img_root: Path,
                                val_label_root: Path,
                                use_crop: bool) -> list[dict]:
    """--val-dir / --val-label-root override 사용 시의 val records 빌더.
    train_vit.build_crop_records / build_folder_records 와 동일 필터 규칙.
    """
    if not val_img_root.is_dir():
        raise SystemExit(f"[!] val_img_root 없음: {val_img_root}")
    exts = {".jpg", ".jpeg", ".png"}

    if use_crop:
        # bbox 기반 records — label 은 val_label_root/val/val_<cls>.json 기대
        lbl_dir = val_label_root / "val"
        if not lbl_dir.is_dir():
            raise SystemExit(
                f"[!] val label dir 없음: {lbl_dir} "
                f"(expect {{val_label_root}}/val/val_<cls>.json)"
            )
        good, dropped_area, dropped_missing = [], 0, 0
        for c_idx, c in enumerate(CLASSES):
            p = lbl_dir / f"val_{c}.json"
            if not p.is_file():
                continue
            loaded = None
            for enc in ("euc-kr", "utf-8"):
                try:
                    with open(p, encoding=enc) as f:
                        loaded = json.load(f)
                    break
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
            for item in (loaded or []):
                if not isinstance(item, dict):
                    continue
                fname = item.get("filename")
                annot_a = item.get("annot_A")
                box = annot_a.get("boxes") if isinstance(annot_a, dict) else None
                if not fname or not isinstance(box, dict):
                    continue
                try:
                    x0 = float(box["minX"]); y0 = float(box["minY"])
                    x1 = float(box["maxX"]); y1 = float(box["maxY"])
                except (KeyError, TypeError, ValueError):
                    continue
                ip = val_img_root / c / fname
                if not ip.is_file():
                    dropped_missing += 1
                    continue
                try:
                    with Image.open(ip) as im:
                        W, H = im.size
                except Exception:
                    dropped_missing += 1
                    continue
                x0 = max(0.0, min(float(W), x0)); x1 = max(0.0, min(float(W), x1))
                y0 = max(0.0, min(float(H), y0)); y1 = max(0.0, min(float(H), y1))
                if (x1 - x0) <= 1 or (y1 - y0) <= 1:
                    dropped_area += 1
                    continue
                good.append({
                    "path": str(ip),
                    "class_idx": c_idx,
                    "bbox_xyxy": (x0, y0, x1, y1),
                    "img_wh": (float(W), float(H)),
                    "is_crop": True,
                })
        print(f"[val-override/crop] kept={len(good)}  "
              f"dropped_area_le0={dropped_area}  dropped_missing={dropped_missing}")
        return good
    # folder 모드 — bbox 무시, 디렉터리 스캔
    out = []
    for c_idx, c in enumerate(CLASSES):
        cls_dir = val_img_root / c
        if not cls_dir.is_dir():
            continue
        for img_path in cls_dir.iterdir():
            if img_path.suffix.lower() not in exts:
                continue
            out.append({
                "path": str(img_path),
                "class_idx": c_idx,
                "bbox_xyxy": None,
                "is_crop": False,
            })
    print(f"[val-override/folder] kept={len(out)}")
    return out


# --------------------------------------------------------------------------
# records 빌더: soft_labels.npz → train records (bbox 옵션)
# --------------------------------------------------------------------------
def build_soft_train_records(data_root: Path,
                             soft_npz: Path,
                             use_crop: bool) -> list[dict]:
    """soft_labels npz 를 열어 train records 생성.
    파일 존재 확인 후 drop. crop=True 면 label json 에서 annot_A.boxes 재로드.
    """
    if not soft_npz.is_file():
        raise SystemExit(f"[!] soft labels npz 없음: {soft_npz}")
    data = np.load(soft_npz, allow_pickle=True)
    needed = {"filenames", "soft_targets", "class_idx_gt"}
    if not needed.issubset(set(data.files)):
        raise SystemExit(
            f"[!] npz 필수 key 누락: {needed - set(data.files)}  "
            f"(found: {data.files})"
        )
    filenames = data["filenames"]
    soft = data["soft_targets"]
    gt = data["class_idx_gt"]
    if len(filenames) != len(soft) or len(filenames) != len(gt):
        raise SystemExit(
            f"[!] npz 길이 불일치: filenames={len(filenames)}  soft={len(soft)}  "
            f"gt={len(gt)}"
        )
    # shape 검증 — (N, NUM_CLASSES) 강제
    if soft.ndim != 2 or soft.shape[1] != NUM_CLASSES:
        raise SystemExit(
            f"[!] soft_targets shape 이상: {tuple(soft.shape)} "
            f"(expected (N,{NUM_CLASSES}))"
        )
    # class_idx_gt 범위 검증 — [0, NUM_CLASSES)
    gt_min = int(np.min(gt)) if len(gt) else 0
    gt_max = int(np.max(gt)) if len(gt) else 0
    if gt_min < 0 or gt_max >= NUM_CLASSES:
        raise SystemExit(
            f"[!] class_idx_gt 범위 초과: min={gt_min}  max={gt_max}  "
            f"(expected [0,{NUM_CLASSES-1}])"
        )
    # sanity check: soft 각 row sum≈1
    sums = soft.sum(axis=1)
    bad = np.where(np.abs(sums - 1.0) > 1e-3)[0]
    if bad.size > 0:
        print(f"[warn] soft row sum != 1 rows: {bad.size}  (first 5 sums: "
              f"{sums[bad[:5]].tolist()})")
    # bbox 매핑 (crop 사용 시): (class, filename) 키로 bbox 찾기 (클래스 간 동명이인 방어)
    bbox_by_key: dict[tuple[str, str], tuple[float, float, float, float]] = {}
    if use_crop:
        lbls = load_labels_per_split(data_root, "train")
        for lr in lbls:
            bbox_by_key[(lr["class"], lr["filename"])] = lr["bbox_xyxy"]

    img_root = data_root / "img" / "train"
    out: list[dict] = []
    missing, bbox_drop = 0, 0
    for fname, s, g in zip(filenames, soft, gt):
        fname = str(fname)
        gi = int(g)
        path = img_root / CLASSES[gi] / fname
        if not path.is_file():
            missing += 1
            continue
        rec: dict = {
            "path": str(path),
            "class_idx": gi,
            "soft": np.asarray(s, dtype=np.float32),
            "is_crop": bool(use_crop),
            "bbox_xyxy": None,
        }
        if use_crop:
            b = bbox_by_key.get((CLASSES[gi], fname))
            if b is None:
                # bbox 없는 샘플: crop 모드여도 그냥 full image 로 둠 (skip 안 함)
                rec["bbox_xyxy"] = None
            else:
                try:
                    with Image.open(path) as im:
                        W, H = im.size
                except Exception:
                    missing += 1
                    continue
                x0, y0, x1, y1 = b
                x0 = max(0.0, min(float(W), x0))
                x1 = max(0.0, min(float(W), x1))
                y0 = max(0.0, min(float(H), y0))
                y1 = max(0.0, min(float(H), y1))
                if (x1 - x0) <= 1 or (y1 - y0) <= 1:
                    bbox_drop += 1
                    continue
                rec["bbox_xyxy"] = (x0, y0, x1, y1)
        out.append(rec)
    print(f"[soft-train] kept={len(out)}  missing_img={missing}  "
          f"dropped_small_bbox={bbox_drop}")
    return out


# --------------------------------------------------------------------------
# 모델 로드
# --------------------------------------------------------------------------
def load_src_model(src_path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    """ckpt 에서 model_name 뽑아 timm.create_model(pretrained=False) 후 state_dict load.
    반환: (model, src_meta)  — src_meta 는 ckpt['meta'] 에 있던 원본 meta + prev_val_acc 추정치
    """
    if not src_path.is_file():
        raise SystemExit(f"[!] src-model 없음: {src_path}")
    try:
        ckpt = torch.load(str(src_path), map_location="cpu", weights_only=False)
    except TypeError:
        # older torch 호환
        ckpt = torch.load(str(src_path), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise SystemExit(f"[!] ckpt 포맷 비정상: {type(ckpt)}")
    state = ckpt.get("model")
    if not isinstance(state, dict):
        raise SystemExit("[!] ckpt['model'] 에 state_dict 없음")
    meta = ckpt.get("meta", {}) or {}
    args_saved = ckpt.get("args", {}) or {}
    model_name = meta.get("model_name") or args_saved.get("model") \
        or "vit_base_patch16_224"
    print(f"[i] src model_name resolved: {model_name}")

    model = timm.create_model(model_name, pretrained=False, num_classes=NUM_CLASSES)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys (first 5): {list(missing)[:5]}  "
              f"total={len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys (first 5): {list(unexpected)[:5]}  "
              f"total={len(unexpected)}")
    model = model.to(device)

    # prev_val_acc 추출 (stage_b_best → val_accuracy)
    prev_val_acc = None
    sb = meta.get("stage_b_best") or {}
    if isinstance(sb, dict):
        prev_val_acc = sb.get("val_accuracy")
    if prev_val_acc is None:
        sa = meta.get("stage_a_best") or {}
        if isinstance(sa, dict):
            prev_val_acc = sa.get("val_accuracy")

    return model, {
        "model_name": model_name,
        "src_args": args_saved,
        "src_meta": meta,
        "prev_val_acc": prev_val_acc,
    }


# --------------------------------------------------------------------------
# Loss — soft CE
# --------------------------------------------------------------------------
def soft_cross_entropy(logits: torch.Tensor, soft_target: torch.Tensor) -> torch.Tensor:
    """loss = -Σ q_c · log_softmax(logits)_c  (batch mean).

    equivalent to KL(q || p) + H(q), where H(q) is target-only constant.
    soft_target 은 (B, C) 이고 각 row sum=1 가정 (DataSet 에서 재보정 완료).
    """
    # log_softmax 은 numerical stable.
    log_p = F.log_softmax(logits, dim=-1)
    # 각 row loss = -Σ q · log p
    per_sample = -(soft_target * log_p).sum(dim=-1)
    return per_sample.mean()


# --------------------------------------------------------------------------
# train / eval
# --------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device, amp_dtype,
                    grad_clip_norm: float) -> dict:
    model.train()
    total_loss = 0.0
    total_argmax_match = 0
    total_n = 0
    use_amp = (amp_dtype is not None) and (device.type == "cuda")
    scaler = None
    if amp_dtype is torch.float16 and use_amp:
        try:
            scaler = torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler()

    for imgs, soft, hard in loader:
        imgs = imgs.to(device, non_blocking=True)
        soft = soft.to(device, non_blocking=True)
        hard = hard.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(imgs)
                loss = soft_cross_entropy(logits, soft)
            if scaler is not None:
                scaler.scale(loss).backward()
                if grad_clip_norm and grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        (p for p in model.parameters() if p.requires_grad),
                        max_norm=grad_clip_norm,
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip_norm and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        (p for p in model.parameters() if p.requires_grad),
                        max_norm=grad_clip_norm,
                    )
                optimizer.step()
        else:
            logits = model(imgs)
            loss = soft_cross_entropy(logits, soft)
            loss.backward()
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    max_norm=grad_clip_norm,
                )
            optimizer.step()

        bs = imgs.size(0)
        total_loss += float(loss.detach()) * bs
        # 참고용: soft argmax vs hard gt 매칭률 (학습 진행 지표, soft==gt 강제는 아님)
        total_argmax_match += int((logits.argmax(1) == hard).sum().item())
        total_n += bs

    return {
        "loss": total_loss / max(1, total_n),
        "argmax_vs_hard_acc": total_argmax_match / max(1, total_n),
        "n": total_n,
    }


@torch.no_grad()
def evaluate_hard(model, loader, device, amp_dtype) -> dict:
    """hard label 기반 val accuracy / macro-F1."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    all_pred, all_true = [], []
    use_amp = (amp_dtype is not None) and (device.type == "cuda")
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda",
                            dtype=(amp_dtype or torch.float16),
                            enabled=use_amp):
            logits = model(imgs)
            loss = loss_fn(logits, labels)
        bs = imgs.size(0)
        total_loss += float(loss.detach()) * bs
        pred = logits.argmax(1)
        total_correct += int((pred == labels).sum().item())
        total_n += bs
        all_pred.append(pred.cpu().numpy())
        all_true.append(labels.cpu().numpy())

    y_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=np.int64)
    y_true = np.concatenate(all_true) if all_true else np.array([], dtype=np.int64)
    try:
        from sklearn.metrics import f1_score
        macro_f1 = float(f1_score(
            y_true, y_pred, labels=list(range(NUM_CLASSES)),
            average="macro", zero_division=0,
        )) if y_true.size else float("nan")
    except Exception:
        macro_f1 = float("nan")
    return {
        "loss": total_loss / max(1, total_n),
        "acc": total_correct / max(1, total_n),
        "macro_f1": macro_f1,
        "n": total_n,
        "y_pred": y_pred,
        "y_true": y_true,
    }


# --------------------------------------------------------------------------
# 산출물 저장
# --------------------------------------------------------------------------
def save_outputs(args, csv_rows, best_val_meta, src_info, ckpt_path,
                 timing, final_eval):
    logs_dir = PROJECT / "logs"
    logs_dir.mkdir(exist_ok=True)

    # csv
    csv_path = logs_dir / f"{args.name}.csv"
    if csv_rows:
        import csv as _csv
        keys = list(csv_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in csv_rows:
                w.writerow(r)

    # meta json
    meta = {
        "name": args.name,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "src_info": {
            "model_name": src_info.get("model_name"),
            "prev_val_acc": src_info.get("prev_val_acc"),
            "src_args": src_info.get("src_args"),
        },
        "history": csv_rows,
        "best": best_val_meta,
        "final_eval": {
            "val_loss": final_eval.get("loss"),
            "val_acc": final_eval.get("acc"),
            "val_macro_f1": final_eval.get("macro_f1"),
            "n": final_eval.get("n"),
        },
        "timing_sec": timing,
        "model_path": str(ckpt_path),
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    with open(logs_dir / f"{args.name}.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # experiments.md append
    exp_md = PROJECT / "experiments.md"
    best_val_acc = best_val_meta.get("val_acc", 0.0) if best_val_meta else 0.0
    best_val_loss = best_val_meta.get("val_loss", float("inf")) \
        if best_val_meta else float("inf")
    prev = src_info.get("prev_val_acc")
    prev_str = f"{prev:.4f}" if isinstance(prev, (int, float)) else "?"
    note = f"{args.note or '-'} | soft_ce from={Path(args.src_model).name} " \
           f"prev_val_acc={prev_str}"
    line = (
        f"| {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} | {args.name} | "
        f"{src_info.get('model_name','?')} | "
        f"bs={args.batch_size} img={args.img_size} lr={args.lr} "
        f"soft=1 crop={int(args.crop)} aug={int(args.augment)} | "
        f"epochs={len(csv_rows)}/{args.epochs} | val_acc={best_val_acc:.4f} | "
        f"val_loss={best_val_loss:.4f} | final_train_acc="
        f"{(csv_rows[-1].get('train_argmax_vs_hard', float('nan')) if csv_rows else float('nan')):.4f} | "
        f"note={note} |\n"
    )
    header_needed = True
    if exp_md.is_file():
        try:
            with open(exp_md, encoding="utf-8") as f:
                head = f.read(200)
            if "| date |" in head or "| 날짜 |" in head:
                header_needed = False
        except Exception:
            pass
    with open(exp_md, "a", encoding="utf-8") as f:
        if header_needed:
            f.write(
                "| date | name | model | config | epochs | val_acc | val_loss | "
                "train_acc | note |\n"
                "|---|---|---|---|---|---|---|---|---|\n"
            )
        f.write(line)
    return line


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="ViT soft-label fine-tune (low lr, short epochs).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--src-model", required=True,
                    help="pretrained ViT ckpt (.pt, from train_vit.py).")
    ap.add_argument("--soft-labels", required=True,
                    help="train_soft.npz from build_soft_labels.py")
    ap.add_argument("--name", required=True,
                    help="실험 이름 (models/<name>.pt 로 저장)")
    ap.add_argument("--data-root", default=str(DEFAULT_DATA),
                    help="data_rot 루트 (train img 경로 조립용)")
    ap.add_argument("--val-dir", default=None,
                    help="val 이미지 폴더 (default: data_rot/img/val)")
    ap.add_argument("--val-label-root", default=None,
                    help="val label root (default: data_rot/label)")
    ap.add_argument("--lr", type=float, default=5e-6,
                    help="고정 lr. 재학습 방지 위해 낮게 (1e-5 이하 권장).")
    ap.add_argument("--epochs", type=int, default=2,
                    help="짧게 (1~3 권장).")
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--crop", action="store_true",
                    help="bbox crop on (src 모델이 crop=True 로 학습됐으면 반드시 켤 것)")
    ap.add_argument("--augment", action="store_true",
                    help="HFlip/Rot±10/ColorJitter on. soft target 이 이미 regularizer 라 기본 off")
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=1,
                    help="early stop patience (짧은 epoch 라 기본 1).")
    ap.add_argument("--monitor", default="val_accuracy",
                    choices=["val_accuracy"],
                    help="soft label 학습에선 hard val_acc 만 유효. 다른 값은 비활성.")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--prefetch-factor", type=int, default=2)
    ap.add_argument("--note", type=str, default="")
    ap.add_argument("--grad-clip-norm", type=float, default=1.0)
    # aug 세부 (train_vit 시그니처와 호환)
    ap.add_argument("--aug-rot-deg", type=float, default=10.0)
    ap.add_argument("--aug-hflip-prob", type=float, default=0.5)
    ap.add_argument("--aug-jitter-bright", type=float, default=0.2)
    ap.add_argument("--aug-jitter-contrast", type=float, default=0.2)
    ap.add_argument("--aug-jitter-saturation", type=float, default=0.1)
    ap.add_argument("--aug-jitter-hue", type=float, default=0.05)
    ap.add_argument("--aug-rotation-fill", type=int, default=128)
    ap.add_argument("--overwrite-if-worse", action="store_true",
                    help="on: fine-tune 결과가 prev_val_acc 보다 낮아도 ckpt 덮어씀. "
                         "off (기본): 낮으면 저장 안 함(기존 best 유지)")
    args = ap.parse_args()

    # 경계 방어
    if args.lr <= 0 or args.lr > 1e-4:
        raise SystemExit(
            f"[!] --lr 는 (0, 1e-4] 권장. 재학습 방지. (입력 {args.lr})"
        )
    if args.epochs <= 0 or args.epochs > 10:
        raise SystemExit(f"[!] --epochs 는 [1,10] 권장. (입력 {args.epochs})")
    return args


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed, deterministic=False)

    if not torch.cuda.is_available():
        print("[!] CUDA not available — CPU fallback (매우 느림).")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}
    amp_dtype = amp_map[args.amp] if device.type == "cuda" else None
    print(f"[i] device={device}  amp={args.amp}->{amp_dtype}")

    data_root = Path(args.data_root)
    if not (data_root / "img" / "train").is_dir():
        raise SystemExit(f"[!] {data_root}/img/train 없음. --data-root 확인")

    PROJECT.joinpath("models").mkdir(exist_ok=True)
    PROJECT.joinpath("logs").mkdir(exist_ok=True)

    # === 모델 로드 ===
    ckpt_path = PROJECT / "models" / f"{args.name}.pt"
    model, src_info = load_src_model(Path(args.src_model), device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[i] model loaded: total={total_params:,}  trainable={trainable_params:,}")
    prev = src_info.get("prev_val_acc")
    if prev is not None:
        print(f"[i] prev_val_acc (from src ckpt meta): {prev:.4f}")

    # === records ===
    tr_recs = build_soft_train_records(
        data_root=data_root,
        soft_npz=Path(args.soft_labels),
        use_crop=args.crop,
    )
    override_val = (args.val_dir is not None) or (args.val_label_root is not None)
    if override_val:
        v_img = Path(args.val_dir) if args.val_dir else (data_root / "img" / "val")
        v_lbl = Path(args.val_label_root) if args.val_label_root \
            else (data_root / "label")
        print(f"[i] val override: img={v_img}  label={v_lbl}")
        vl_recs = _build_val_records_override(v_img, v_lbl, args.crop)
    else:
        if args.crop:
            vl_recs = build_crop_records(data_root, "val")
        else:
            vl_recs = build_folder_records(data_root, "val")
    if len(tr_recs) == 0:
        raise SystemExit("[!] train records 0 — 경로/soft npz 확인")
    if len(vl_recs) == 0:
        raise SystemExit("[!] val records 0 — data_rot/img/val 확인")

    # === transforms ===
    train_tf, val_tf = make_transforms(
        args.img_size, args.augment,
        crop=args.crop,
        rot_deg=args.aug_rot_deg,
        hflip_prob=args.aug_hflip_prob,
        jitter_bright=args.aug_jitter_bright,
        jitter_contrast=args.aug_jitter_contrast,
        jitter_saturation=args.aug_jitter_saturation,
        jitter_hue=args.aug_jitter_hue,
        rotation_fill=args.aug_rotation_fill,
    )
    train_ds = SoftLabelDataset(tr_recs, transform=train_tf)
    val_ds = HardLabelDataset(vl_recs, transform=val_tf)

    common_loader = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=_make_worker_init(args.seed),
    )
    if args.num_workers > 0:
        common_loader["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(
        train_ds, shuffle=True, drop_last=False,
        generator=torch.Generator().manual_seed(args.seed),
        **common_loader,
    )
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common_loader)

    # === optimizer (고정 lr, no scheduler) ===
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.999), eps=1e-8,
    )

    # === loop ===
    csv_rows: list[dict] = []
    best_val_acc = -float("inf")
    best_state = None
    best_meta: dict = {}
    no_improve = 0

    # 저장 thresh: prev_val_acc 있으면 그걸 기준, 없으면 -inf
    save_threshold = -float("inf")
    if isinstance(prev, (int, float)) and not args.overwrite_if_worse:
        save_threshold = float(prev)
        print(f"[i] save threshold = prev_val_acc = {save_threshold:.4f} "
              f"(미달 시 ckpt 저장 안 함)")

    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        t_ep = time.time()
        tr = train_one_epoch(
            model, train_loader, optimizer, device, amp_dtype,
            grad_clip_norm=args.grad_clip_norm,
        )
        va = evaluate_hard(model, val_loader, device, amp_dtype)
        elapsed = time.time() - t_ep
        improved = va["acc"] > best_val_acc
        mark = "*" if improved else " "
        print(
            f"[ep {ep:02d}/{args.epochs}]  "
            f"train soft_loss={tr['loss']:.4f}  "
            f"argmax_vs_hard={tr['argmax_vs_hard_acc']:.4f}  |  "
            f"val loss={va['loss']:.4f}  acc={va['acc']:.4f}  "
            f"f1={va['macro_f1']:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  t={elapsed:.1f}s  {mark}"
        )
        csv_rows.append({
            "epoch": ep,
            "train_loss": tr["loss"],
            "train_argmax_vs_hard": tr["argmax_vs_hard_acc"],
            "val_loss": va["loss"],
            "val_acc": va["acc"],
            "val_f1": va["macro_f1"],
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": round(elapsed, 2),
        })
        if improved:
            best_val_acc = va["acc"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            best_meta = {
                "epoch": ep,
                "val_loss": va["loss"],
                "val_acc": va["acc"],
                "val_f1": va["macro_f1"],
            }
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"[early stop] no improve for {args.patience} (ep {ep}).")
                break

    # === 저장 결정 ===
    saved = False
    if best_state is None:
        print("[!] best_state 없음 (epoch 0 완주 실패?) — 저장 skip")
    elif best_val_acc < save_threshold:
        print(
            f"[i] best_val_acc={best_val_acc:.4f} < prev={save_threshold:.4f}. "
            f"ckpt 저장 skip (--overwrite-if-worse 주면 강제 저장)."
        )
    else:
        model.load_state_dict(best_state)
        torch.save(
            {
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "args": vars(args),
                "meta": {
                    "model_name": src_info.get("model_name"),
                    "classes": CLASSES,
                    "img_size": args.img_size,
                    "prev_val_acc": prev,
                    "best": best_meta,
                    "soft_label_src": str(args.soft_labels),
                    "src_model": str(args.src_model),
                    "saved_at": dt.datetime.now().isoformat(timespec="seconds"),
                },
            },
            str(ckpt_path),
        )
        saved = True
        print(f"[i] saved: {ckpt_path}  (val_acc={best_val_acc:.4f})")

    # === 최종 평가 (best 가 저장됐든 안 됐든 best_state 로 평가) ===
    if best_state is not None:
        model.load_state_dict(best_state)
    final_eval = evaluate_hard(model, val_loader, device, amp_dtype)

    timing = {"total_sec": round(time.time() - t0, 1), "saved_new_ckpt": saved}
    line = save_outputs(args, csv_rows, best_meta, src_info, ckpt_path,
                        timing, final_eval)

    print("\n" + "=" * 72)
    print(
        f"[DONE] {args.name}  epochs_run={len(csv_rows)}  "
        f"val_acc={final_eval['acc']:.4f}  "
        f"val_f1={final_eval['macro_f1']:.4f}  "
        f"saved_new_ckpt={saved}  "
        f"time={timing['total_sec']}s"
    )
    print("experiments.md line:")
    print(line.rstrip())


if __name__ == "__main__":
    main()
