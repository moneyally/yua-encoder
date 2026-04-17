"""emotion-project / train_vit.py — ViT-B/16 trainer (timm, ImageNet21k pretrained).

===============================================================================
설계 문서 (DESIGN)
===============================================================================

좌표계·데이터 가정 (반드시 준수 — train.py 와 동일)
----------------------------------------------------
- 데이터 루트: /workspace/user4/emotion-project/data_rot  (EXIF 정규화 완료 복사본)
- PIL.Image.open().convert('RGB') 로 로드 — data_rot 는 transpose 후 JPEG 재저장
  이므로 추가 exif_transpose 불필요.
- annot_A.boxes (minX/minY/maxX/maxY) 는 EXIF 정규화 좌표계 = decoded 이미지 픽셀.
  그대로 PIL.Image.crop((x0,y0,x1,y1)) 가능.
- bbox 이슈:
    * 9건 음수 좌표 → [0,W]x[0,H] 클립.
    * 클립 후 area<=1 인 샘플 drop (train.py 기준 동일).
    * 라벨 없는 이미지는 crop 모드에서 자동 skip.

함수 시그니처
--------------
- set_seed(seed: int)
- load_labels_per_split(data_root, split) -> list[dict]
- build_crop_records(data_root, split) -> list[dict]
- build_folder_records(data_root, split) -> list[dict]
- class EmotionViTDataset(records, transform, is_crop)
- make_transforms(img_size, augment) -> (train_tf, val_tf)
- build_model(num_classes) -> nn.Module
- freeze_backbone(model) / unfreeze_all(model)
- train_one_epoch(model, loader, optimizer, scheduler, scaler, loss_fn, device)
- evaluate_one_epoch(model, loader, loss_fn, device) -> dict
- run_stage(stage_name, model, train_loader, val_loader, cfg, shared, csv_rows)
- evaluate_full(model, loader, class_names, out_prefix) -> dict
- save_artifacts(name, args, csv_rows, eval_metrics, ckpt_path, timing)
- main()

데이터 파이프라인
-----------------
crop=False (folder mode, 기본):
  data_rot/img/<split>/<class>/*.jpg → walk → records(list of dict)
  → EmotionViTDataset(records, transform, is_crop=False)

crop=True:
  data_rot/label/<split>/<split>_<class>.json (euc-kr → utf-8 fallback)
  → annot_A.boxes 로딩 → clip + drop → records

DataLoader: num_workers=4, pin_memory=True, shuffle=True(train)/False(val)

모델
----
- timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=4)
- head 는 timm 의 기본 Linear(768, 4) 사용. 필요 시 MLP head 로 교체 가능하지만
  일반적으로 기본 Linear head 가 표준. (현재 스펙 기본 Linear 채택)

두 단계 학습 (--two-stage)
---------------------------
Stage A (backbone freeze, head only, 3~5 epoch, lr 1e-3):
  - backbone requires_grad=False (timm 의 model.blocks / patch_embed / pos_embed / cls_token)
  - head (model.head) 만 학습
  - AdamW + Cosine w/ warmup (또는 OneCycle)
  - best (by monitor) 를 메모리에 hold → ckpt_path.replace('.pt','.stageA.pt') 로 저장
Stage B (backbone unfreeze, 전체 fine-tune, 10~20 epoch, lr 5e-5~1e-4):
  - Stage A best state_dict 로드 후 전체 unfreeze
  - optimizer 새로 만들어 lower lr 적용
  - best (monitor) 를 최종 models/<name>.pt 로 저장

--two-stage 없으면 처음부터 전체 unfreeze 로 Stage B 만 실행 (lr=args.lr).

Loss / Optimizer / Scheduler
----------------------------
- Loss: CrossEntropyLoss(label_smoothing=0.1 default)
- Optimizer: AdamW(lr, weight_decay=1e-4 default)
- Scheduler: CosineAnnealingLR with linear warmup (self-impl, torch.optim.lr_scheduler.LambdaLR)
- AMP: torch.cuda.amp.autocast + GradScaler (bf16 기본, fp16 fallback)

평가
----
- epoch 단위: val_loss / val_acc / val_macro_f1
- 최종: confusion_matrix + classification_report (sklearn)
- 실패 케이스 시각화: top-N 확률 낮은 오분류 샘플 grid → results/<name>_fail_grid.png
  (matplotlib 필요. 실패 시 warn 후 skip.)

산출물 (train.py 와 동일 패턴, 확장자만 .pt)
---------------------------------------------
- models/<name>.pt                     (dict: model, args, epoch, val_metric, stage)
- models/<name>.stageA.pt              (two-stage 일 때만)
- logs/<name>.csv                       (epoch, stage, train_loss, train_acc, val_loss, val_acc, val_f1, lr)
- logs/<name>.meta.json                 (args, csv_rows, eval, timing)
- results/<name>_confmat.npz            (cm, labels, y_true, y_pred)
- results/<name>_report.txt             (sklearn classification_report)
- results/<name>_fail_grid.png          (선택)
- experiments.md 한 줄 append

CLI
---
--name --model --data --epochs --batch-size --img-size --lr --lr-backbone
--warmup-epochs --patience --label-smoothing --weight-decay --monitor
--crop --augment --two-stage --stage-a-epochs --note --seed
--num-workers --amp --limit-train-samples --limit-val-samples
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
from typing import Callable, Optional

import numpy as np

# torch import 는 seed 적용을 위해 위에 둔다. 환경변수 OMP 등은 미설정 (A40 단일 GPU).
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image

# timm (ViT)
import timm

# torchvision transforms (classic, v1) — v2 는 API 변화 많아 stable 우선
from torchvision import transforms as T

# 프로젝트 루트: 환경변수 > __file__ 기반 자동 감지
PROJECT = Path(os.environ.get("EMOTION_PROJECT_ROOT",
                              str(Path(__file__).resolve().parent.parent)))
DEFAULT_DATA = PROJECT / "data_rot"
CLASSES = ["anger", "happy", "panic", "sadness"]   # 고정. 알파벳 순.
NUM_CLASSES = len(CLASSES)
SPLITS = ("train", "val")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ============================================================
# Seed
# ============================================================
def set_seed(seed: int, deterministic: bool = False) -> None:
    """시드 고정. deterministic=True 시 cudnn benchmark off + deterministic on
    (속도 희생하지만 bit-exact 재현성 확보, PyTorch 권장)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # use_deterministic_algorithms 은 일부 op 에러 가능 → warn_only
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as e:
            print(f"[warn] use_deterministic_algorithms 실패: {e}")
    else:
        torch.backends.cudnn.benchmark = True


def _make_worker_init(seed: int):
    """PyTorch 권장 worker_init_fn (docs/stable/notes/randomness.html).
    각 worker 가 torch.initial_seed() 기반으로 numpy/random 동기화."""
    def _fn(worker_id: int):
        worker_seed = (torch.initial_seed() + worker_id) % (2 ** 32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    return _fn


# ============================================================
# 라벨 JSON 로딩 + bbox 유효성 필터 (crop 모드에서만 사용)
# ============================================================
def load_labels_per_split(data_root: Path, split: str) -> list[dict]:
    """split 하위의 4개 class json 을 하나의 리스트로 concat.
    encoding: euc-kr 우선 → 실패 시 utf-8.  (train.py 와 동일 규칙)
    """
    out = []
    lbl_dir = data_root / "label" / split
    for c_idx, c in enumerate(CLASSES):
        p = lbl_dir / f"{split}_{c}.json"
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
            out.append({
                "filename": fname,
                "class_idx": c_idx,
                "class": c,
                "bbox_xyxy": (x0, y0, x1, y1),
            })
    return out


def build_crop_records(data_root: Path, split: str) -> list[dict]:
    """라벨 + 이미지 파일 존재 확인 + bbox 클립/유효성 필터. (train.py 와 동일 로직)"""
    recs = load_labels_per_split(data_root, split)
    good, dropped_area, dropped_missing = [], 0, 0
    img_root = data_root / "img" / split
    for r in recs:
        ip = img_root / r["class"] / r["filename"]
        if not ip.is_file():
            dropped_missing += 1
            continue
        try:
            with Image.open(ip) as im:
                W, H = im.size
        except Exception:
            dropped_missing += 1
            continue
        x0, y0, x1, y1 = r["bbox_xyxy"]
        x0 = max(0.0, min(float(W), x0));  x1 = max(0.0, min(float(W), x1))
        y0 = max(0.0, min(float(H), y0));  y1 = max(0.0, min(float(H), y1))
        if (x1 - x0) <= 1 or (y1 - y0) <= 1:
            dropped_area += 1
            continue
        good.append({
            "path": str(ip),
            "class_idx": r["class_idx"],
            "bbox_xyxy": (x0, y0, x1, y1),
            "img_wh": (float(W), float(H)),
            "is_crop": True,
        })
    print(f"[crop/{split}] kept={len(good)}  dropped_area_le0={dropped_area}  "
          f"dropped_missing={dropped_missing}")
    return good


def build_folder_records(data_root: Path, split: str) -> list[dict]:
    """폴더 기반 (bbox 미사용). data_rot/img/<split>/<class>/*.(jpg|jpeg|png) walk."""
    out = []
    img_root = data_root / "img" / split
    exts = {".jpg", ".jpeg", ".png"}
    for c_idx, c in enumerate(CLASSES):
        d = img_root / c
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() not in exts:
                continue
            out.append({
                "path": str(f),
                "class_idx": c_idx,
                "bbox_xyxy": None,
                "img_wh": None,
                "is_crop": False,
            })
    return out


# ============================================================
# Dataset
# ============================================================
class EmotionViTDataset(Dataset):
    """감정 4분류. record = {path, class_idx, bbox_xyxy, is_crop}.
    bbox_xyxy 가 None 이 아니면 PIL.crop 으로 face crop 적용.
    """
    def __init__(self, records: list[dict], transform: Optional[Callable] = None):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        # data_rot 는 EXIF 정규화 완료 → convert('RGB') 만.
        img = Image.open(r["path"]).convert("RGB")
        bbox = r.get("bbox_xyxy")
        if bbox is not None:
            W, H = img.size
            x0, y0, x1, y1 = bbox
            # 방어적 재-clip (build 단계에서 이미 했지만 edge case 안전빵)
            x0 = max(0.0, min(float(W), float(x0)))
            x1 = max(0.0, min(float(W), float(x1)))
            y0 = max(0.0, min(float(H), float(y0)))
            y1 = max(0.0, min(float(H), float(y1)))
            if (x1 - x0) > 1 and (y1 - y0) > 1:
                img = img.crop((int(x0), int(y0), int(math.ceil(x1)), int(math.ceil(y1))))
            # else: crop 포기하고 원본 유지 (빌드에서 드롭됐어야 함)
        if self.transform is not None:
            img = self.transform(img)
        label = int(r["class_idx"])
        return img, label


def make_transforms(img_size: int, augment: bool,
                    crop: bool = False,
                    rot_deg: float = 10.0,
                    hflip_prob: float = 0.5,
                    jitter_bright: float = 0.2,
                    jitter_contrast: float = 0.2,
                    jitter_saturation: float = 0.1,
                    jitter_hue: float = 0.05,
                    rotation_fill: int = 128):
    """ViT 일반 권장 설정 + 세부 인자화.
    - Resize(256) → CenterCrop(224) → Normalize (val/no-aug).
    - train(augment=True): Resize → RandomCrop + 각 aug layer 는 해당 값 > 0 일 때만.
    """
    from torchvision.transforms import InterpolationMode
    resize_short = max(256, int(round(img_size * (256 / 224))))
    # val/no-aug: bbox crop 된 얼굴엔 Resize(정사각형) 이 안전, 그 외엔 Resize+CenterCrop.
    if crop:
        val_tf = T.Compose([
            T.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        val_tf = T.Compose([
            T.Resize(resize_short),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    if not augment:
        return val_tf, val_tf
    # train augment: face crop 이면 RandomResizedCrop(scale 0.9~1.0)로 살짝 줌 in/out.
    # crop 안 된 원본이면 기존 Resize(256)+RandomCrop(img) 유지 (가장자리 4% 버림).
    if crop:
        layers = [T.RandomResizedCrop(
            img_size, scale=(0.9, 1.0), ratio=(0.95, 1.05),
            interpolation=InterpolationMode.BILINEAR,
        )]
    else:
        layers = [T.Resize(resize_short), T.RandomCrop(img_size)]
    if hflip_prob and hflip_prob > 0:
        layers.append(T.RandomHorizontalFlip(p=hflip_prob))
    if rot_deg and rot_deg > 0:
        # 공식 권장: interpolation=BILINEAR, fill 중간회색 (SigLIP/ImageNet normalize 친화).
        layers.append(T.RandomRotation(
            degrees=rot_deg,
            interpolation=InterpolationMode.BILINEAR,
            fill=int(rotation_fill),
        ))
    if any(v and v > 0 for v in (jitter_bright, jitter_contrast,
                                 jitter_saturation, jitter_hue)):
        layers.append(T.ColorJitter(
            brightness=max(0.0, jitter_bright),
            contrast=max(0.0, jitter_contrast),
            saturation=max(0.0, jitter_saturation),
            hue=max(0.0, jitter_hue),
        ))
    layers += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    train_tf = T.Compose(layers)
    return train_tf, val_tf


# ============================================================
# 모델
# ============================================================
def build_model(model_name: str, num_classes: int = NUM_CLASSES,
                drop_rate: float = 0.0) -> nn.Module:
    """timm ViT 생성. model_name: 'vit_base_patch16_224' (기본값)."""
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    return model


def freeze_backbone(model: nn.Module) -> int:
    """timm ViT: head 외 전체 freeze. 반환: frozen 파라미터 수."""
    # timm ViT 는 model.head (or get_classifier()) 가 분류기.
    classifier = model.get_classifier() if hasattr(model, "get_classifier") else model.head
    # 일단 전부 freeze
    for p in model.parameters():
        p.requires_grad_(False)
    # head 만 unfreeze
    for p in classifier.parameters():
        p.requires_grad_(True)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return frozen


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(True)


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# Scheduler (warmup + cosine)
# ============================================================
def make_warmup_cosine(optimizer, warmup_steps: int, total_steps: int,
                       min_lr_ratio: float = 0.01):
    """linear warmup → cosine decay to min_lr_ratio * base_lr."""
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(1, int(total_steps))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        # cosine
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(min_lr_ratio + (1.0 - min_lr_ratio) * cos)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
# train / eval epoch
# ============================================================
def train_one_epoch(model, loader, optimizer, scheduler, scaler, loss_fn, device,
                    amp_dtype, grad_clip_norm: float = 0.0):
    """amp_dtype 이 None 이면 fp32. fp16 이면 autocast+GradScaler. bf16 이면 autocast 만.

    grad_clip_norm > 0 이면 clip_grad_norm_ 호출. ViT fine-tune 에서 loss spike 방지 관용.
    fp16(scaler 사용) 의 경우 scaler.unscale_ 후 clip → scaler.step 순서 필수.
    """
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0
    use_amp = (amp_dtype is not None) and (device.type == "cuda")
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(imgs)
                loss = loss_fn(logits, labels)
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
            loss = loss_fn(logits, labels)
            loss.backward()
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    max_norm=grad_clip_norm,
                )
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        bs = imgs.size(0)
        total_loss += float(loss.detach()) * bs
        total_correct += int((logits.argmax(1) == labels).sum().item())
        total_n += bs
    return {
        "loss": total_loss / max(1, total_n),
        "acc": total_correct / max(1, total_n),
        "n": total_n,
    }


@torch.no_grad()
def evaluate_one_epoch(model, loader, loss_fn, device, amp_dtype,
                       collect: bool = False):
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    all_pred, all_true, all_prob = [], [], []
    use_amp = (amp_dtype is not None) and (device.type == "cuda")
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        # dtype= 은 enabled=True 일 때만 의미 있음. cpu 모드/fp32 모드에선 autocast 비활성.
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
        if collect:
            all_pred.append(pred.cpu().numpy())
            all_true.append(labels.cpu().numpy())
            all_prob.append(F.softmax(logits.float(), dim=1).cpu().numpy())
    out = {
        "loss": total_loss / max(1, total_n),
        "acc": total_correct / max(1, total_n),
        "n": total_n,
    }
    if collect:
        out["y_pred"] = np.concatenate(all_pred) if all_pred else np.array([])
        out["y_true"] = np.concatenate(all_true) if all_true else np.array([])
        out["y_prob"] = np.concatenate(all_prob) if all_prob else np.zeros((0, NUM_CLASSES))
    return out


# ============================================================
# Stage runner
# ============================================================
def run_stage(stage_name: str,
              model: nn.Module,
              train_loader: DataLoader,
              val_loader: DataLoader,
              *,
              lr: float,
              epochs: int,
              warmup_epochs: int,
              weight_decay: float,
              label_smoothing: float,
              patience: int,
              monitor: str,
              device: torch.device,
              amp_dtype,
              ckpt_path: Path,
              csv_rows: list,
              epoch_offset: int = 0,
              min_lr_ratio: float = 0.01,
              grad_clip_norm: float = 0.0,
              adam_beta1: float = 0.9,
              adam_beta2: float = 0.999,
              adam_eps: float = 1e-8,
              class_weight: Optional[torch.Tensor] = None) -> tuple[dict, int]:
    """한 단계(Stage A or B) 학습. 반환: (best_metric_dict, epochs_run).

    class_weight: (NUM_CLASSES,) torch.Tensor — CrossEntropyLoss weight 로 전달.
    None 이면 균등 가중.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_n = sum(p.numel() for p in trainable)
    print(f"[stage {stage_name}] trainable params = {trainable_n:,}  "
          f"lr={lr:g}  epochs={epochs}  warmup={warmup_epochs}  wd={weight_decay:g}")

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay,
                                  betas=(adam_beta1, adam_beta2), eps=adam_eps)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * max(1, epochs)
    warmup_steps = steps_per_epoch * max(0, warmup_epochs)
    scheduler = make_warmup_cosine(optimizer, warmup_steps, total_steps,
                                   min_lr_ratio=min_lr_ratio)
    # torch 2.x 권장 API: torch.amp.GradScaler("cuda"). fp16 에서만 scaling 필요, bf16 은 불필요.
    if amp_dtype is torch.float16:
        try:
            scaler = torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    loss_fn = nn.CrossEntropyLoss(
        weight=class_weight, label_smoothing=label_smoothing,
    )

    assert monitor in ("val_accuracy", "val_loss", "val_f1")
    # val_accuracy / val_f1 은 max, val_loss 는 min
    if monitor == "val_loss":
        better = lambda a, b: a < b
        best_val = float("inf")
    else:
        better = lambda a, b: a > b
        best_val = -float("inf")
    # Stage A 에서 patience hit 으로 개선 0 회 발생 시 best_state=None 방지 seed.
    # 현재 모델 weights 로 초기화 → 최소한 starting point 는 보장됨.
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_meta = {
        "stage": stage_name, "epoch": 0,
        "val_loss": float("inf"), "val_accuracy": -float("inf"), "val_f1": float("nan"),
        "note": "initial (no improvement yet)",
    }
    no_improve = 0
    epochs_run = 0

    for ep in range(1, epochs + 1):
        t_ep = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, scheduler, scaler,
                             loss_fn, device, amp_dtype,
                             grad_clip_norm=grad_clip_norm)
        # val pass: 굳이 predict 2번 돌릴 필요 없어서 여기서 f1 포함
        va = evaluate_one_epoch(model, val_loader, loss_fn, device, amp_dtype,
                                collect=True)
        # macro f1 (sklearn)
        try:
            from sklearn.metrics import f1_score
            val_f1 = float(f1_score(va["y_true"], va["y_pred"],
                                    labels=list(range(NUM_CLASSES)),
                                    average="macro", zero_division=0))
        except Exception:
            val_f1 = float("nan")
        cur_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t_ep
        if monitor == "val_accuracy":
            val_key = va["acc"]
        elif monitor == "val_loss":
            val_key = va["loss"]
        else:  # val_f1 (macro-F1, NaN 방어: 값이 nan 이면 -inf 처리)
            val_key = val_f1 if not math.isnan(val_f1) else -1.0
        improved = better(val_key, best_val)
        mark = "*" if improved else " "
        print(f"[{stage_name} ep {ep:02d}/{epochs}]  "
              f"train loss={tr['loss']:.4f} acc={tr['acc']:.4f}  |  "
              f"val loss={va['loss']:.4f} acc={va['acc']:.4f} f1={val_f1:.4f}  "
              f"lr={cur_lr:.2e}  t={elapsed:.1f}s  {mark}")

        csv_rows.append({
            "epoch": epoch_offset + ep,
            "stage": stage_name,
            "train_loss": tr["loss"],
            "train_acc": tr["acc"],
            "val_loss": va["loss"],
            "val_acc": va["acc"],
            "val_f1": val_f1,
            "lr": cur_lr,
            "time_sec": round(elapsed, 2),
        })

        epochs_run = ep
        if improved:
            best_val = val_key
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_meta = {
                "stage": stage_name,
                "epoch": ep,
                "val_loss": va["loss"],
                "val_accuracy": va["acc"],
                "val_f1": val_f1,
            }
            # 즉시 디스크 저장 (중간 crash 안전)
            torch.save(
                {"model": best_state, "meta": best_meta},
                str(ckpt_path),
            )
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[{stage_name}] early stop at ep {ep} (no improve for {patience}).")
                break

    # 종료 후 best state 를 model 에 복원
    if best_state is not None:
        model.load_state_dict(best_state)
    return (best_meta or {"stage": stage_name, "epoch": 0,
                          "val_loss": float("inf"), "val_accuracy": 0.0, "val_f1": 0.0},
            epochs_run)


# ============================================================
# 최종 평가 (confusion matrix, report, fail grid)
# ============================================================
def evaluate_full(model, loader, class_names, out_prefix: Path, device, amp_dtype,
                  save_fail_grid: bool = True) -> dict:
    loss_fn = nn.CrossEntropyLoss()
    res = evaluate_one_epoch(model, loader, loss_fn, device, amp_dtype, collect=True)
    y_true = res["y_true"].astype(np.int64)
    y_pred = res["y_pred"].astype(np.int64)
    y_prob = res["y_prob"]

    try:
        from sklearn.metrics import classification_report, confusion_matrix, f1_score
        report_txt = classification_report(
            y_true, y_pred, labels=list(range(len(class_names))),
            target_names=class_names, digits=4, zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        macro_f1 = float(f1_score(y_true, y_pred,
                                  labels=list(range(len(class_names))),
                                  average="macro", zero_division=0))
    except Exception as e:
        report_txt = f"(sklearn 불가: {e})"
        cm = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
        macro_f1 = float("nan")

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    (out_prefix.with_suffix(".txt")).write_text(report_txt, encoding="utf-8")
    np.savez(str(out_prefix.with_suffix(".npz")),
             cm=cm, labels=np.array(class_names), y_true=y_true, y_pred=y_pred,
             y_prob=y_prob)

    # Confusion matrix 시각화 (선택, 실패하면 경고만)
    if save_fail_grid and len(y_true) > 0:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(len(class_names)))
            ax.set_yticks(range(len(class_names)))
            ax.set_xticklabels(class_names, rotation=45, ha="right")
            ax.set_yticklabels(class_names)
            ax.set_xlabel("pred"); ax.set_ylabel("true")
            ax.set_title(f"confusion matrix  (n={len(y_true)})")
            vmax = cm.max() if cm.size else 1
            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            color="black" if cm[i, j] < vmax * 0.6 else "white",
                            fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            # out_prefix.name 은 '<name>_confmat' 이므로 .png 로 저장
            cm_png = out_prefix.with_suffix(".png")
            fig.savefig(str(cm_png), dpi=120)
            plt.close(fig)

            # 실패 케이스 grid: '가장 확신 있게 틀린' top-16 원본 썸네일
            wrong_idx = np.where(y_pred != y_true)[0]
            if wrong_idx.size > 0:
                conf = y_prob[wrong_idx, y_pred[wrong_idx]]
                order = np.argsort(-conf)[:min(16, wrong_idx.size)]
                sel = wrong_idx[order]
                # records 가져오기: dataset 이 wrap 되었을 수 있으니 받은 인자로는 복원 어려움.
                # 여기선 loader.dataset.records 가 EmotionViTDataset 이라 가정하고 접근.
                ds_obj = getattr(loader, "dataset", None)
                records = getattr(ds_obj, "records", None)
                if records is not None:
                    n = len(sel); cols = 4
                    rows = (n + cols - 1) // cols
                    fig2, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
                    axes = np.atleast_2d(axes).reshape(rows, cols)
                    for k, idx in enumerate(sel):
                        r2, c2 = divmod(k, cols)
                        ax2 = axes[r2][c2]
                        rec = records[int(idx)]
                        try:
                            img = Image.open(rec["path"]).convert("RGB")
                            if rec.get("bbox_xyxy") is not None:
                                x0, y0, x1, y1 = rec["bbox_xyxy"]
                                W, H = img.size
                                x0 = max(0, min(W, int(x0))); x1 = max(0, min(W, int(math.ceil(x1))))
                                y0 = max(0, min(H, int(y0))); y1 = max(0, min(H, int(math.ceil(y1))))
                                if (x1 - x0) > 1 and (y1 - y0) > 1:
                                    img = img.crop((x0, y0, x1, y1))
                            ax2.imshow(img)
                        except Exception:
                            pass
                        t = class_names[int(y_true[idx])]; p = class_names[int(y_pred[idx])]
                        ax2.set_title(f"T:{t}\nP:{p} ({conf[order[k]]:.2f})", fontsize=8)
                        ax2.axis("off")
                    for k in range(len(sel), rows * cols):
                        r2, c2 = divmod(k, cols); axes[r2][c2].axis("off")
                    fig2.tight_layout()
                    fail_png = out_prefix.parent / f"{out_prefix.name}_fails.png"
                    fig2.savefig(str(fail_png), dpi=110)
                    plt.close(fig2)
        except Exception as e:
            print(f"[warn] cm/fail png skip: {e}")

    return {
        "val_loss": float(res["loss"]),
        "val_accuracy": float(res["acc"]),
        "val_macro_f1": macro_f1,
        "n_val": int(y_true.shape[0]),
        "cm": cm.tolist(),
        "report": report_txt,
    }


# ============================================================
# 산출물 저장 + experiments.md append
# ============================================================
def save_artifacts(name, args, csv_rows, eval_metrics, ckpt_path, timing,
                   stage_a_meta=None, stage_b_meta=None):
    logs_dir = PROJECT / "logs"
    logs_dir.mkdir(exist_ok=True)

    # CSV 저장
    csv_path = logs_dir / f"{name}.csv"
    if csv_rows:
        import csv as _csv
        keys = list(csv_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in csv_rows:
                w.writerow(r)

    # meta.json
    meta = {
        "name": name,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "epochs_run": len(csv_rows),
        "history": csv_rows,
        "stage_a_best": stage_a_meta,
        "stage_b_best": stage_b_meta,
        "eval": {k: v for k, v in eval_metrics.items() if k != "report"},
        "eval_report": eval_metrics.get("report", ""),
        "timing_sec": timing,
        "model_path": str(ckpt_path),
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    with open(logs_dir / f"{name}.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # experiments.md append
    exp_md = PROJECT / "experiments.md"
    # best 지표 뽑기 (train.py 의 experiments.md 9컬럼 포맷과 정합)
    best_val_acc = max((r["val_acc"] for r in csv_rows), default=0.0)
    best_val_loss = min((r["val_loss"] for r in csv_rows), default=float("inf"))
    best_val_f1 = max((r["val_f1"] for r in csv_rows if not math.isnan(r["val_f1"])),
                      default=float("nan"))
    final_train_acc = csv_rows[-1]["train_acc"] if csv_rows else 0.0
    two_stage = 1 if getattr(args, "two_stage", False) else 0
    # train.py 와 동일한 9컬럼 포맷. val_f1 은 note 에 첨가 (meta.json 에도 보존).
    note_with_f1 = f"{args.note or '-'} | val_f1={best_val_f1:.4f}"
    line = (
        f"| {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} | {name} | {args.model} | "
        f"bs={args.batch_size} img={args.img_size} lr={args.lr} "
        f"lr_bb={args.lr_backbone} two_stage={two_stage} "
        f"crop={int(args.crop)} aug={int(args.augment)} | "
        f"epochs={len(csv_rows)}/{args.epochs + (args.stage_a_epochs if two_stage else 0)} | "
        f"val_acc={best_val_acc:.4f} | val_loss={best_val_loss:.4f} | "
        f"final_train_acc={final_train_acc:.4f} | note={note_with_f1} |\n"
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
                "| date | name | model | config | epochs | val_acc | val_loss | train_acc | note |\n"
                "|---|---|---|---|---|---|---|---|---|\n"
            )
        f.write(line)

    return meta, line


# ============================================================
# CLI
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser(
        description="ViT-B/16 trainer (timm, 2-stage fine-tune)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--name", required=True, help="실험 이름 (파일명 prefix)")
    ap.add_argument("--model", default="vit_base_patch16_224",
                    help="timm 모델명 (예: vit_base_patch16_224, vit_small_patch16_224)")
    ap.add_argument("--data", default=str(DEFAULT_DATA), help="data_rot 루트")
    ap.add_argument("--epochs", type=int, default=15,
                    help="Stage B (또는 단일 stage) epochs")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="Stage A head-only lr, 또는 --two-stage off 시 전체 lr.")
    ap.add_argument("--lr-backbone", type=float, default=5e-5,
                    help="Stage B backbone fine-tune lr.")
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="timm drop_rate.")
    ap.add_argument("--crop", action="store_true", help="bbox crop (annot_A) on")
    ap.add_argument("--augment", action="store_true",
                    help="train 에 HFlip/Rot±10/ColorJitter 적용")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--note", type=str, default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=int, default=1,
                    help="Stage B linear warmup epochs.")
    ap.add_argument("--monitor", default="val_accuracy",
                    choices=["val_accuracy", "val_loss", "val_f1"],
                    help="best ckpt 기준. val_f1 은 macro-F1 (class imbalance 대비)")
    ap.add_argument("--deterministic", action="store_true",
                    help="cudnn benchmark off + deterministic on (속도 ↓, 재현성 ↑)")
    ap.add_argument("--aug-rotation-fill", type=int, default=128,
                    help="RandomRotation 빈 영역 fill 값 (0=검정, 128=중간회색).")
    ap.add_argument("--class-weight", default="none",
                    choices=["none", "auto"],
                    help="auto: inv-freq 로 CrossEntropy weight 자동 계산 (class imbalance)")
    ap.add_argument("--two-stage", action="store_true",
                    help="on: Stage A(freeze, head only) → Stage B(unfreeze all). "
                         "off: 처음부터 전체 unfreeze, lr=--lr.")
    ap.add_argument("--stage-a-epochs", type=int, default=3,
                    help="--two-stage on 일 때 Stage A epochs (기본 3).")
    ap.add_argument("--stage-a-warmup", type=int, default=0,
                    help="Stage A linear warmup epochs.")
    ap.add_argument("--num-workers", type=int, default=8,
                    help="DataLoader worker 수. 4→8 상향 (PIL crop+aug 병목 완화).")
    ap.add_argument("--prefetch-factor", type=int, default=4,
                    help="worker 별 미리 준비할 batch 수 (default 2→4, GPU 놀지 않게).")
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"],
                    help="autocast dtype. A40 은 bf16 지원 안정적.")
    ap.add_argument("--limit-train-samples", type=int, default=0,
                    help="smoke test 용. 0 이면 전부.")
    ap.add_argument("--limit-val-samples", type=int, default=0)
    # --- aug 세부 인자화 (각 값 > 0 일 때만 해당 transform 포함) ---
    ap.add_argument("--aug-rot-deg", type=float, default=10.0,
                    help="RandomRotation ±deg. 0 이면 skip.")
    ap.add_argument("--aug-hflip-prob", type=float, default=0.5,
                    help="RandomHorizontalFlip p. 0 이면 skip.")
    ap.add_argument("--aug-jitter-bright", type=float, default=0.2,
                    help="ColorJitter brightness factor.")
    ap.add_argument("--aug-jitter-contrast", type=float, default=0.2,
                    help="ColorJitter contrast factor.")
    ap.add_argument("--aug-jitter-saturation", type=float, default=0.1,
                    help="ColorJitter saturation factor.")
    ap.add_argument("--aug-jitter-hue", type=float, default=0.05,
                    help="ColorJitter hue factor.")
    # --- scheduler / optimizer hyperparameter ---
    ap.add_argument("--min-lr-ratio", type=float, default=0.01,
                    help="cosine schedule 의 final_lr / initial_lr 비율.")
    ap.add_argument("--grad-clip-norm", type=float, default=1.0,
                    help="clip_grad_norm_ max_norm. 0 이면 clip 안 함. ViT fine-tune 권장 1.0.")
    ap.add_argument("--adam-beta1", type=float, default=0.9,
                    help="AdamW beta1 (기본 0.9).")
    ap.add_argument("--adam-beta2", type=float, default=0.999,
                    help="AdamW beta2 (ViT는 0.95 도 쓰임).")
    ap.add_argument("--adam-eps", type=float, default=1e-8,
                    help="AdamW eps.")
    return ap.parse_args()


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    set_seed(args.seed, deterministic=args.deterministic)

    # device / amp
    if not torch.cuda.is_available():
        print("[!] CUDA not available — CPU fallback. ViT on CPU 는 매우 느림.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}
    amp_dtype = amp_map[args.amp] if device.type == "cuda" else None
    print(f"[i] device={device}  amp={args.amp}->{amp_dtype}")

    data_root = Path(args.data)
    if not (data_root / "img" / "train").is_dir():
        raise SystemExit(f"[!] {data_root}/img/train 없음. --data 경로 확인")

    PROJECT.joinpath("models").mkdir(exist_ok=True)
    PROJECT.joinpath("logs").mkdir(exist_ok=True)
    PROJECT.joinpath("results").mkdir(exist_ok=True)

    print(f"[i] args: {vars(args)}")
    t0 = time.time()

    # ===== records =====
    if args.crop:
        tr_recs = build_crop_records(data_root, "train")
        vl_recs = build_crop_records(data_root, "val")
    else:
        tr_recs = build_folder_records(data_root, "train")
        vl_recs = build_folder_records(data_root, "val")

    # shuffle train records 한 번 (DataLoader 의 shuffle 과 별개, 재현성)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(tr_recs)
    if args.limit_train_samples and args.limit_train_samples < len(tr_recs):
        tr_recs = tr_recs[:args.limit_train_samples]
        print(f"[warn] limit_train_samples={args.limit_train_samples} 적용. "
              f"smoke test 전용 — 실제 분포와 다를 수 있음.")
    if args.limit_val_samples and args.limit_val_samples < len(vl_recs):
        vl_recs = vl_recs[:args.limit_val_samples]

    # class counts
    counts_tr = {c: 0 for c in CLASSES}
    for r in tr_recs:
        counts_tr[CLASSES[r["class_idx"]]] += 1
    counts_vl = {c: 0 for c in CLASSES}
    for r in vl_recs:
        counts_vl[CLASSES[r["class_idx"]]] += 1
    print(f"[i] train n={len(tr_recs)} {counts_tr}")
    print(f"[i] val   n={len(vl_recs)} {counts_vl}")

    # ===== transforms / datasets / loaders =====
    train_tf, val_tf = make_transforms(
        args.img_size, args.augment,
        crop=args.crop,  # face crop 이면 RandomCrop 대신 RandomResizedCrop(0.9~1.0)
        rot_deg=args.aug_rot_deg,
        hflip_prob=args.aug_hflip_prob,
        jitter_bright=args.aug_jitter_bright,
        jitter_contrast=args.aug_jitter_contrast,
        jitter_saturation=args.aug_jitter_saturation,
        jitter_hue=args.aug_jitter_hue,
        rotation_fill=args.aug_rotation_fill,
    )
    train_ds = EmotionViTDataset(tr_recs, transform=train_tf)
    val_ds = EmotionViTDataset(vl_recs, transform=val_tf)

    # DataLoader reproducibility: PyTorch 공식 권장 worker_init_fn.
    # 속도: prefetch_factor 늘려 worker 마다 미리 batch 준비, pin_memory 로 host→device 빠르게.
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

    # ===== 모델 =====
    model = build_model(args.model, NUM_CLASSES, drop_rate=args.dropout)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[i] model={args.model}  total_params={total_params:,}")

    ckpt_path = PROJECT / "models" / f"{args.name}.pt"
    csv_rows: list[dict] = []
    stage_a_meta = None
    stage_b_meta = None

    # ===== class_weight 계산 (옵션) =====
    class_weight_tensor = None
    if args.class_weight == "auto":
        counts = np.bincount(
            [r["class_idx"] for r in tr_recs], minlength=NUM_CLASSES,
        ).astype(np.float64)
        # inv-freq: N_total / (K * counts)
        weights = counts.sum() / (NUM_CLASSES * np.maximum(counts, 1))
        class_weight_tensor = torch.tensor(
            weights, dtype=torch.float32, device=device,
        )
        print(f"[i] class_weight=auto → {weights.round(4).tolist()}")

    # ===== two-stage 분기 방어 =====
    if args.two_stage and args.stage_a_epochs <= 0:
        raise ValueError(
            "--two-stage 가 on 이면 --stage-a-epochs > 0 필수. "
            "Stage A skip 원하면 --two-stage 빼고 --unfreeze-encoder 같은 single-stage 옵션을 명시할 것."
        )

    # ===== Stage A (옵션) =====
    if args.two_stage:
        freeze_backbone(model)
        print(f"[i] Stage A prepared: trainable={count_trainable(model):,} "
              f"(head only)")
        stage_a_ckpt = PROJECT / "models" / f"{args.name}.stageA.pt"
        stage_a_meta, run_a = run_stage(
            stage_name="A",
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=args.lr,
            epochs=args.stage_a_epochs,
            warmup_epochs=args.stage_a_warmup,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            patience=args.patience,
            monitor=args.monitor,
            device=device,
            amp_dtype=amp_dtype,
            ckpt_path=stage_a_ckpt,
            csv_rows=csv_rows,
            epoch_offset=0,
            min_lr_ratio=args.min_lr_ratio,
            grad_clip_norm=args.grad_clip_norm,
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            adam_eps=args.adam_eps,
            class_weight=class_weight_tensor,
        )
        print(f"[i] Stage A done. best={stage_a_meta}  epochs_run={run_a}")
        # unfreeze 전체
        unfreeze_all(model)
        print(f"[i] Stage B prepared: trainable={count_trainable(model):,} (all)")

        stage_b_meta, run_b = run_stage(
            stage_name="B",
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=args.lr_backbone,
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            patience=args.patience,
            monitor=args.monitor,
            device=device,
            amp_dtype=amp_dtype,
            ckpt_path=ckpt_path,
            csv_rows=csv_rows,
            epoch_offset=run_a,
            min_lr_ratio=args.min_lr_ratio,
            grad_clip_norm=args.grad_clip_norm,
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            adam_eps=args.adam_eps,
            class_weight=class_weight_tensor,
        )
    else:
        # 처음부터 전체 fine-tune
        unfreeze_all(model)
        print(f"[i] Single-stage (no freeze). trainable={count_trainable(model):,}")
        stage_b_meta, run_b = run_stage(
            stage_name="B",
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=args.lr,
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            patience=args.patience,
            monitor=args.monitor,
            device=device,
            amp_dtype=amp_dtype,
            ckpt_path=ckpt_path,
            csv_rows=csv_rows,
            epoch_offset=0,
            min_lr_ratio=args.min_lr_ratio,
            grad_clip_norm=args.grad_clip_norm,
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            adam_eps=args.adam_eps,
            class_weight=class_weight_tensor,
        )

    # ===== 최종 평가 (best weights 이미 model 에 복원돼 있음) =====
    # models/<name>.pt 에 final meta 덮어쓰기 (args 포함)
    torch.save(
        {
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "args": vars(args),
            "meta": {
                "stage_a_best": stage_a_meta,
                "stage_b_best": stage_b_meta,
                "classes": CLASSES,
                "img_size": args.img_size,
                "model_name": args.model,
                "saved_at": dt.datetime.now().isoformat(timespec="seconds"),
            },
        },
        str(ckpt_path),
    )

    eval_metrics = evaluate_full(
        model, val_loader, CLASSES,
        PROJECT / "results" / f"{args.name}_confmat",
        device=device, amp_dtype=amp_dtype,
        save_fail_grid=True,
    )

    timing = {"total_sec": round(time.time() - t0, 1)}
    meta, line = save_artifacts(args.name, args, csv_rows, eval_metrics, ckpt_path,
                                timing, stage_a_meta, stage_b_meta)

    print("\n" + "=" * 72)
    print(f"[DONE] {args.name}  epochs_run={len(csv_rows)}  "
          f"val_acc={eval_metrics['val_accuracy']:.4f}  "
          f"val_loss={eval_metrics['val_loss']:.4f}  "
          f"val_f1={eval_metrics['val_macro_f1']:.4f}  "
          f"time={timing['total_sec']}s  model={ckpt_path}")
    print("experiments.md line:")
    print(line.rstrip())


if __name__ == "__main__":
    main()
