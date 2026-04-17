"""emotion-project / train_siglip.py — yua-encoder (SigLIP wrapper) trainer.

===============================================================================
설계 문서 (DESIGN)
===============================================================================

데이터 가정 (train.py / train_vit.py 와 동일)
----------------------------------------------
- 데이터 루트: /workspace/user4/emotion-project/data_rot (EXIF 정규화 complete)
- PIL.Image.open().convert('RGB') 로 로드. 추가 exif_transpose 불필요.
- crop=True 모드: annot_A.boxes (minX/minY/maxX/maxY) → EXIF 정규화 좌표계 = decoded pixel
  · 9 건 음수 좌표 → [0,W]×[0,H] clip
  · clip 후 area ≤ 1 인 샘플 drop
  · 라벨 없는 이미지는 crop 모드에서 자동 skip
- crop=False 모드: 폴더 walk (data_rot/img/<split>/<class>/*.{jpg,jpeg,png})

모델 구조 (yua-encoder wrapper 기반)
------------------------------
Image[B,3,H,W]
  → SiglipImageProcessor 전처리 (collate_fn 에서 수행, 384×384 resize + SigLIP norm)
  → VisionEncoder(config)
     · SigLIP ViT-Base (google/siglip-base-patch16-384)
     · freeze_encoder=True 면 backbone 은 no_grad + eval
     · last_hidden_state [B, 576, 768]
     · PixelShuffle r=4 → [B, 36, 768*16=12288]
     · projection MLP (LayerNorm + Linear(12288→2048) + GELU + Dropout + Linear(2048→2048))
  → Mean Pool over N → [B, 2048]
  → Dropout → Linear(2048, 4) → softmax

학습 대상 (trainable params)
----------------------------
- freeze_encoder=True (기본, --unfreeze-encoder off):
  projection MLP + tile_row_embed/col_embed + classifier head (Linear 2048→4)
  ~ 수천만 단위. linear probe 보다는 조금 더 많음 (projection 2단 MLP).
- --unfreeze-encoder on:
  위 + SigLIP backbone (88M) → lr_backbone (5e-5) 별도 파라미터 그룹.

Loss / Optimizer / Scheduler
-----------------------------
- Loss: CrossEntropyLoss(label_smoothing)
- Optimizer: AdamW (head/projection 은 lr, backbone 은 lr_backbone)
- Scheduler: 자체 LambdaLR (linear warmup → cosine decay to min_lr_ratio)
- AMP: bf16 기본 (A40 안정). fp16 시 GradScaler. off 시 fp32.

주의: encode_image 내부는 config.freeze_encoder 에 따라 no_grad/enable_grad 를
     강제하므로 --unfreeze-encoder 시 config.freeze_encoder=False 로 넘겨야 backbone grad 가 흐른다.

2-stage 옵션 (--two-stage)
---------------------------
- Stage A: freeze_encoder=True 그대로 (= linear probe). head+projection 만 학습.
- Stage B: VisionEncoder 내부 self.siglip params requires_grad=True 로 풀고
           encode_image 가 enable_grad 타도록 config.freeze_encoder 를 False 로 override.
           별도 parameter group (backbone 은 lr_backbone).
--two-stage off 기본: --unfreeze-encoder 플래그에 따라 single stage linear-probe 또는 full-ft.

Logging / Artifacts (train_vit.py 와 동일 패턴)
------------------------------------------------
- models/<name>.pt                (dict: model state_dict + args + meta)
- models/<name>.stageA.pt         (two-stage 시)
- logs/<name>.csv
- logs/<name>.meta.json
- results/<name>_confmat.{npz,txt,png}
- results/<name>_confmat_fails.png
- experiments.md append (9 컬럼)

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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image

# torchvision transforms — PIL-level augmentation only (resize/normalize 는 processor 가 담당)
from torchvision import transforms as T

# 프로젝트 루트를 sys.path 에 추가 → models_custom / src 패키지 import 가능
PROJECT = Path(os.environ.get("EMOTION_PROJECT_ROOT",
                              str(Path(__file__).resolve().parent.parent)))
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

# yua-encoder wrapper (models_custom/vision_encoder.py)
from models_custom.vision_encoder import VisionEncoder, VisionConfig  # noqa: E402

DEFAULT_DATA = PROJECT / "data_rot"
CLASSES = ["anger", "happy", "panic", "sadness"]
NUM_CLASSES = len(CLASSES)
SPLITS = ("train", "val")


# ============================================================
# Seed
# ============================================================
def set_seed(seed: int, deterministic: bool = False) -> None:
    """시드 고정. deterministic=True 시 cudnn benchmark off + deterministic on."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as e:
            print(f"[warn] use_deterministic_algorithms 실패: {e}")
    else:
        torch.backends.cudnn.benchmark = True


def _make_worker_init(seed: int):
    """PyTorch 공식 권장 worker_init_fn."""
    def _fn(worker_id: int):
        worker_seed = (torch.initial_seed() + worker_id) % (2 ** 32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    return _fn


# ============================================================
# 라벨 JSON 로딩 + bbox 유효성 필터 (train_vit.py 와 동일)
# ============================================================
def load_labels_per_split(data_root: Path, split: str) -> list[dict]:
    """split 하위 4개 class json 로드. euc-kr 우선 → utf-8 fallback."""
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
    """bbox clip + area drop + 존재 확인."""
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
    """폴더 walk."""
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
# Dataset — PIL 반환 (collate_fn 에서 processor 통과)
# ============================================================
class EmotionSiglipDataset(Dataset):
    """SigLIP 용. __getitem__ 이 (PIL.Image.Image, int) 반환.

    - data_rot 는 EXIF 정규화 완료 → convert('RGB') 만.
    - bbox 가 있으면 PIL.crop (정수 rounding 후).
    - PIL 단계 aug (HFlip / RandomRotation / ColorJitter) 를 transform 으로 적용.
    - 최종 resize / normalize / ToTensor 는 collate_fn 이 SiglipImageProcessor 로 일괄.
    """
    def __init__(self, records: list[dict], pil_transform: Optional[Callable] = None):
        self.records = records
        self.pil_transform = pil_transform  # PIL → PIL. None 이면 그대로.

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
                img = img.crop((int(x0), int(y0),
                                int(math.ceil(x1)), int(math.ceil(y1))))
        if self.pil_transform is not None:
            img = self.pil_transform(img)
            # transform 이 Tensor 로 바꿔버리는 걸 막기 위해 PIL 보장 Transform 만 쓸 것.
        label = int(r["class_idx"])
        return img, label


def make_pil_transforms(augment: bool,
                        rot_deg: float = 10.0,
                        hflip_prob: float = 0.5,
                        jitter_bright: float = 0.2,
                        jitter_contrast: float = 0.2,
                        jitter_saturation: float = 0.1,
                        jitter_hue: float = 0.05,
                        rotation_fill: int = 128):
    """PIL→PIL 변환만 반환 (ToTensor/Normalize 는 processor 가 담당).

    val 은 None (identity). train(augment=True) 면 HFlip/Rot/ColorJitter 조합.
    processor 가 384×384 로 resize 하므로 여기서 resize 하지 않음 (aug 가 작은 영역에
    focus 되는 걸 피하기 위해 원본 해상도에서 aug 후 processor 에 넘김).
    """
    if not augment:
        return None, None  # train / val both identity (processor only)
    layers = []
    if hflip_prob and hflip_prob > 0:
        layers.append(T.RandomHorizontalFlip(p=hflip_prob))
    if rot_deg and rot_deg > 0:
        from torchvision.transforms import InterpolationMode
        layers.append(T.RandomRotation(
            degrees=rot_deg,
            interpolation=InterpolationMode.BILINEAR,
            fill=int(rotation_fill),  # SigLIP 0.5 mean → 128 중간회색
        ))
    if any(v and v > 0 for v in (jitter_bright, jitter_contrast,
                                 jitter_saturation, jitter_hue)):
        layers.append(T.ColorJitter(
            brightness=max(0.0, jitter_bright),
            contrast=max(0.0, jitter_contrast),
            saturation=max(0.0, jitter_saturation),
            hue=max(0.0, jitter_hue),
        ))
    train_tf = T.Compose(layers) if layers else None
    val_tf = None
    return train_tf, val_tf


def make_collate_fn(processor):
    """list[(PIL, int)] → (pixel_values[B,3,H,W], labels[B]).

    SiglipImageProcessor(images=..., return_tensors='pt') 가 384×384 resize +
    내부 mean/std normalize 까지 한 번에 처리.
    """
    def collate(batch):
        imgs = [b[0] for b in batch]
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        inputs = processor(images=imgs, return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        return pixel_values, labels
    return collate


# ============================================================
# 모델 wrapper
# ============================================================
class SiglipEmotionClassifier(nn.Module):
    """SigLIP + projection MLP + mean pool + Linear classifier.

    forward(pixel_values: [B,3,H,W]) -> logits: [B, num_classes]
    """
    def __init__(self,
                 vision_encoder: VisionEncoder,
                 num_classes: int = NUM_CLASSES,
                 dropout: float = 0.1):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(
            vision_encoder.config.projection_dim, num_classes
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # encode_image: [B, N, projection_dim]  (N = num_image_tokens, pixel_shuffle 시 36)
        embeds = self.vision_encoder.encode_image(pixel_values)
        if embeds.ndim != 3:
            raise RuntimeError(
                f"VisionEncoder.encode_image expected [B,N,D], got {tuple(embeds.shape)}"
            )
        pooled = embeds.mean(dim=1)  # [B, D]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


def build_model_siglip(args, device: torch.device) -> SiglipEmotionClassifier:
    """VisionConfig → VisionEncoder.load_siglip → wrapper 반환."""
    cfg = VisionConfig(
        encoder_name=args.siglip_name,
        image_size=args.img_size,
        freeze_encoder=(not args.unfreeze_encoder),
        use_pixel_shuffle=bool(args.use_pixel_shuffle),
        pixel_shuffle_ratio=args.pixel_shuffle_ratio,
        projection_layers=2,
        projection_dim=args.projection_dim,
        use_anyres=False,            # 감정 classification → 단일 이미지, tiling 불필요
        use_ocr_augment=False,       # 얼굴 이미지 → OCR 꺼둠
        use_tile_position_embedding=False,  # anyres off 면 tile embed 도 off → 불필요 param 제거
    )
    enc = VisionEncoder(cfg)
    enc.load_siglip(device=str(device))  # siglip + processor 둘 다 여기서 준비
    model = SiglipEmotionClassifier(
        vision_encoder=enc,
        num_classes=NUM_CLASSES,
        dropout=args.dropout,
    )
    return model


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_siglip_backbone(model: SiglipEmotionClassifier) -> None:
    """stage A: siglip backbone freeze + eval. projection / head 만 학습."""
    if model.vision_encoder.siglip is not None:
        for p in model.vision_encoder.siglip.parameters():
            p.requires_grad_(False)
        model.vision_encoder.siglip.eval()
    # config.freeze_encoder 도 True 로 맞춤 → encode_image 가 no_grad() 타게
    model.vision_encoder.config.freeze_encoder = True
    # projection / classifier / tile_embed(있다면) 은 기본 requires_grad=True
    for p in model.vision_encoder.projection.parameters():
        p.requires_grad_(True)
    for p in model.classifier.parameters():
        p.requires_grad_(True)


def unfreeze_siglip_backbone(model: SiglipEmotionClassifier) -> None:
    """stage B: siglip backbone unfreeze + train. 전체 fine-tune."""
    if model.vision_encoder.siglip is not None:
        for p in model.vision_encoder.siglip.parameters():
            p.requires_grad_(True)
        model.vision_encoder.siglip.train()
    model.vision_encoder.config.freeze_encoder = False


def split_param_groups(model: SiglipEmotionClassifier,
                       lr_head: float,
                       lr_backbone: float,
                       weight_decay: float) -> list[dict]:
    """backbone (siglip) vs. head+projection+etc 로 param group 2 개 분리.

    backbone 이 전부 freeze 된 경우는 head 그룹만 반환.
    """
    backbone_params = []
    head_params = []
    if model.vision_encoder.siglip is not None:
        for p in model.vision_encoder.siglip.parameters():
            if p.requires_grad:
                backbone_params.append(p)
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # siglip 의 parameter 는 이미 backbone_params 에 포함돼 있으므로 skip
        if n.startswith("vision_encoder.siglip."):
            continue
        head_params.append(p)
    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": lr_head,
                       "weight_decay": weight_decay, "name": "head"})
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone,
                       "weight_decay": weight_decay, "name": "backbone"})
    return groups


# ============================================================
# Scheduler (warmup + cosine)
# ============================================================
def make_warmup_cosine(optimizer, warmup_steps: int, total_steps: int,
                       min_lr_ratio: float = 0.01):
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(1, int(total_steps))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
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
    model.train()
    # freeze 상태 유지 (siglip 이 freeze 면 eval mode 고정 — batch norm/dropout 없지만 관용)
    if (model.vision_encoder.siglip is not None
            and model.vision_encoder.config.freeze_encoder):
        model.vision_encoder.siglip.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    use_amp = (amp_dtype is not None) and (device.type == "cuda")
    for pixel_values, labels in loader:
        pixel_values = pixel_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(pixel_values)
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
            logits = model(pixel_values)
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
        bs = pixel_values.size(0)
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
    for pixel_values, labels in loader:
        pixel_values = pixel_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda",
                            dtype=(amp_dtype or torch.float16),
                            enabled=use_amp):
            logits = model(pixel_values)
            loss = loss_fn(logits, labels)
        bs = pixel_values.size(0)
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
              model: SiglipEmotionClassifier,
              train_loader: DataLoader,
              val_loader: DataLoader,
              *,
              lr_head: float,
              lr_backbone: float,
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
    class_weight: CrossEntropyLoss weight (class imbalance 대비)."""
    groups = split_param_groups(model, lr_head=lr_head,
                                lr_backbone=lr_backbone,
                                weight_decay=weight_decay)
    if not groups:
        raise RuntimeError(f"[stage {stage_name}] trainable parameter group 이 비었다.")
    trainable_n = count_trainable(model)
    group_summary = ", ".join(f"{g['name']}(lr={g['lr']:g}, n={sum(p.numel() for p in g['params']):,})"
                              for g in groups)
    print(f"[stage {stage_name}] trainable={trainable_n:,}  {group_summary}")

    optimizer = torch.optim.AdamW(
        groups,
        betas=(adam_beta1, adam_beta2), eps=adam_eps,
    )
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * max(1, epochs)
    warmup_steps = steps_per_epoch * max(0, warmup_epochs)
    scheduler = make_warmup_cosine(optimizer, warmup_steps, total_steps,
                                   min_lr_ratio=min_lr_ratio)
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
    if monitor == "val_loss":
        better = lambda a, b: a < b
        best_val = float("inf")
    else:
        better = lambda a, b: a > b
        best_val = -float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_meta = {
        "stage": stage_name, "epoch": 0,
        "val_loss": float("inf"), "val_accuracy": -float("inf"),
        "val_f1": float("nan"),
        "note": "initial (no improvement yet)",
    }
    no_improve = 0
    epochs_run = 0

    for ep in range(1, epochs + 1):
        t_ep = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, scheduler, scaler,
                             loss_fn, device, amp_dtype,
                             grad_clip_norm=grad_clip_norm)
        va = evaluate_one_epoch(model, val_loader, loss_fn, device, amp_dtype,
                                collect=True)
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
        else:  # val_f1
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

    if best_state is not None:
        model.load_state_dict(best_state)
    return (best_meta, epochs_run)


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
            cm_png = out_prefix.with_suffix(".png")
            fig.savefig(str(cm_png), dpi=120)
            plt.close(fig)

            wrong_idx = np.where(y_pred != y_true)[0]
            if wrong_idx.size > 0:
                conf = y_prob[wrong_idx, y_pred[wrong_idx]]
                order = np.argsort(-conf)[:min(16, wrong_idx.size)]
                sel = wrong_idx[order]
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
                                x0 = max(0, min(W, int(x0)))
                                x1 = max(0, min(W, int(math.ceil(x1))))
                                y0 = max(0, min(H, int(y0)))
                                y1 = max(0, min(H, int(math.ceil(y1))))
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

    csv_path = logs_dir / f"{name}.csv"
    if csv_rows:
        import csv as _csv
        keys = list(csv_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in csv_rows:
                w.writerow(r)

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

    exp_md = PROJECT / "experiments.md"
    best_val_acc = max((r["val_acc"] for r in csv_rows), default=0.0)
    best_val_loss = min((r["val_loss"] for r in csv_rows), default=float("inf"))
    best_val_f1 = max((r["val_f1"] for r in csv_rows if not math.isnan(r["val_f1"])),
                      default=float("nan"))
    final_train_acc = csv_rows[-1]["train_acc"] if csv_rows else 0.0
    two_stage = 1 if getattr(args, "two_stage", False) else 0
    unfreeze = 1 if getattr(args, "unfreeze_encoder", False) else 0
    note_with_f1 = f"{args.note or '-'} | val_f1={best_val_f1:.4f}"
    model_label = f"siglip_base_p16_{args.img_size}"
    line = (
        f"| {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} | {name} | {model_label} | "
        f"bs={args.batch_size} img={args.img_size} lr={args.lr} "
        f"lr_bb={args.lr_backbone} two_stage={two_stage} unfreeze={unfreeze} "
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
        description="SigLIP (yua-encoder wrapper) trainer — Track C",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--name", default="exp06_siglip_linear_probe",
                    help="실험 이름 (파일명 prefix). 기본: exp06_siglip_linear_probe")
    ap.add_argument("--siglip-name", default="google/siglip-base-patch16-384",
                    help="transformers hub name. SigLIP-Base / SigLIP-Large 등.")
    ap.add_argument("--data", default=str(DEFAULT_DATA), help="data_rot 루트")
    ap.add_argument("--epochs", type=int, default=15,
                    help="Stage B (또는 단일 stage) epochs")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=384,
                    help="VisionConfig.image_size (참고용; 실제 resize 는 SiglipImageProcessor 가 "
                         "--siglip-name 에 따라 자동 수행: siglip-base-patch16-384 면 384, "
                         "siglip-large-patch16-256 이면 256). 이 값은 num_image_tokens 계산 참고용.")
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="head / projection lr (linear probe 시 주 lr).")
    ap.add_argument("--lr-backbone", type=float, default=5e-5,
                    help="SigLIP backbone unfreeze 시 backbone lr.")
    ap.add_argument("--dropout", type=float, default=0.1,
                    help="pooled feature 에 적용되는 Dropout.")
    ap.add_argument("--crop", action="store_true", help="bbox crop (annot_A) on")
    ap.add_argument("--augment", action="store_true",
                    help="train 에 PIL HFlip/Rot±10/ColorJitter 적용")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--note", type=str, default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=int, default=2,
                    help="Stage B linear warmup epochs.")
    ap.add_argument("--monitor", default="val_accuracy",
                    choices=["val_accuracy", "val_loss", "val_f1"],
                    help="best ckpt 기준. val_f1 은 macro-F1 (class imbalance 대비)")
    ap.add_argument("--deterministic", action="store_true",
                    help="cudnn benchmark off + deterministic on (속도 ↓, 재현성 ↑)")
    ap.add_argument("--aug-rotation-fill", type=int, default=128,
                    help="RandomRotation 빈 영역 fill (0=검정, 128=SigLIP 0.5 mean 중간회색)")
    ap.add_argument("--class-weight", default="none",
                    choices=["none", "auto"],
                    help="auto: inv-freq CrossEntropy weight (class imbalance)")
    # 2-stage: linear probe 기본은 off 로 둔다 (1-stage head+projection 학습만으로 충분한 경우가 많음).
    ap.add_argument("--two-stage", action="store_true",
                    help="on: Stage A(siglip freeze) → Stage B(siglip unfreeze). "
                         "off: --unfreeze-encoder 플래그에 따라 single stage linear-probe 또는 full-ft.")
    ap.add_argument("--stage-a-epochs", type=int, default=0,
                    help="--two-stage on 일 때 Stage A epochs.")
    ap.add_argument("--stage-a-warmup", type=int, default=0,
                    help="Stage A linear warmup epochs.")
    ap.add_argument("--unfreeze-encoder", action="store_true",
                    help="single-stage 에서 siglip backbone 전체 unfreeze (full fine-tune).")
    ap.add_argument("--num-workers", type=int, default=8,
                    help="DataLoader worker 수. PIL+SigLIP processor 병목 완화용.")
    ap.add_argument("--prefetch-factor", type=int, default=4,
                    help="worker 별 미리 준비할 batch 수. GPU util 안 놓치게.")
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"],
                    help="autocast dtype. A40 은 bf16 권장.")
    ap.add_argument("--limit-train-samples", type=int, default=0,
                    help="smoke test 용. 0 이면 전부.")
    ap.add_argument("--limit-val-samples", type=int, default=0)
    # aug 세부
    ap.add_argument("--aug-rot-deg", type=float, default=10.0,
                    help="RandomRotation ±deg. 0 이면 skip.")
    ap.add_argument("--aug-hflip-prob", type=float, default=0.5,
                    help="RandomHorizontalFlip p. 0 이면 skip.")
    ap.add_argument("--aug-jitter-bright", type=float, default=0.2)
    ap.add_argument("--aug-jitter-contrast", type=float, default=0.2)
    ap.add_argument("--aug-jitter-saturation", type=float, default=0.1)
    ap.add_argument("--aug-jitter-hue", type=float, default=0.05)
    # optimizer / scheduler
    ap.add_argument("--min-lr-ratio", type=float, default=0.01,
                    help="cosine schedule 의 final_lr / initial_lr 비율.")
    ap.add_argument("--grad-clip-norm", type=float, default=1.0,
                    help="clip_grad_norm_ max_norm. 0 이면 off.")
    ap.add_argument("--adam-beta1", type=float, default=0.9)
    ap.add_argument("--adam-beta2", type=float, default=0.999)
    ap.add_argument("--adam-eps", type=float, default=1e-8)
    # VisionConfig 세부
    ap.add_argument("--projection-dim", type=int, default=2048,
                    help="VisionProjection output dim (classifier input dim).")
    ap.add_argument("--pixel-shuffle-ratio", type=int, default=4,
                    help="PixelShuffle2D ratio. 24→6 (r=4) / 24→12 (r=2).")
    ap.add_argument("--use-pixel-shuffle", action="store_true", default=True,
                    help="pixel shuffle on (기본 on). --no-pixel-shuffle 로 off.")
    ap.add_argument("--no-pixel-shuffle", dest="use_pixel_shuffle",
                    action="store_false",
                    help="pixel shuffle off → N=576 토큰 그대로 mean pool.")
    return ap.parse_args()


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    set_seed(args.seed, deterministic=args.deterministic)

    if not torch.cuda.is_available():
        print("[!] CUDA not available — CPU fallback. SigLIP on CPU 는 매우 느림.")
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

    rng = np.random.default_rng(args.seed)
    rng.shuffle(tr_recs)
    if args.limit_train_samples and args.limit_train_samples < len(tr_recs):
        tr_recs = tr_recs[:args.limit_train_samples]
        print(f"[warn] limit_train_samples={args.limit_train_samples} 적용. "
              f"smoke test 전용 — 실제 분포와 다를 수 있음.")
    if args.limit_val_samples and args.limit_val_samples < len(vl_recs):
        vl_recs = vl_recs[:args.limit_val_samples]

    counts_tr = {c: 0 for c in CLASSES}
    for r in tr_recs:
        counts_tr[CLASSES[r["class_idx"]]] += 1
    counts_vl = {c: 0 for c in CLASSES}
    for r in vl_recs:
        counts_vl[CLASSES[r["class_idx"]]] += 1
    print(f"[i] train n={len(tr_recs)} {counts_tr}")
    print(f"[i] val   n={len(vl_recs)} {counts_vl}")

    # ===== 모델 (processor 도 load) =====
    model = build_model_siglip(args, device)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[i] model=siglip  total_params={total_params:,}  "
          f"projection_dim={args.projection_dim}  "
          f"pixel_shuffle={'on' if args.use_pixel_shuffle else 'off'} "
          f"(r={args.pixel_shuffle_ratio})")
    processor = model.vision_encoder.processor
    if processor is None:
        raise RuntimeError("SiglipImageProcessor load 실패. load_siglip() 결과 확인.")

    # SSOT: processor 의 실제 resize 해상도를 img_size 의 single source of truth 로.
    # --siglip-name 과 --img-size 가 불일치 시 processor 기준으로 강제 정렬 (SSOT).
    # transformers 5.x 는 processor.size 가 SizeDict 객체 (dict 상속). 여러 형태 대응.
    def _extract_proc_size(proc):
        # 1) processor.to_dict()["size"] 로 plain dict 획득 시도
        try:
            pd = proc.to_dict()
            s = pd.get("size")
            if isinstance(s, dict):
                h = s.get("height") or s.get("shortest_edge") or s.get("longest_edge")
                if h is not None:
                    return int(h)
            if isinstance(s, (int, float)):
                return int(s)
        except Exception:
            pass
        # 2) processor.size 직접 접근
        try:
            s = proc.size
            # SizeDict / dict / TypedDict 전부 mapping 인터페이스 지원 가정
            if hasattr(s, "get"):
                h = s.get("height") or s.get("shortest_edge") or s.get("longest_edge")
                if h is not None:
                    return int(h)
            # __getitem__ 시도
            for key in ("height", "shortest_edge", "longest_edge"):
                try:
                    return int(s[key])
                except (KeyError, TypeError, IndexError):
                    continue
            if isinstance(s, (int, float)):
                return int(s)
        except Exception:
            pass
        return None

    proc_h = _extract_proc_size(processor)
    if proc_h is None:
        print(f"[warn] processor.size 확인 실패 — 사용자 지정 img_size={args.img_size} 유지")
    elif proc_h != args.img_size:
        print(f"[warn] SSOT: --img-size={args.img_size} 가 processor.size={proc_h} 와 다름. "
              f"processor 기준 {proc_h} 로 강제 정렬 (VisionConfig.image_size 갱신).")
        args.img_size = proc_h
        model.vision_encoder.config.image_size = proc_h
    else:
        print(f"[i] SSOT 확인: processor.size=args.img_size={proc_h} 일치.")

    # ===== transforms / datasets / loaders =====
    train_pil_tf, val_pil_tf = make_pil_transforms(
        augment=args.augment,
        rot_deg=args.aug_rot_deg,
        hflip_prob=args.aug_hflip_prob,
        jitter_bright=args.aug_jitter_bright,
        jitter_contrast=args.aug_jitter_contrast,
        jitter_saturation=args.aug_jitter_saturation,
        jitter_hue=args.aug_jitter_hue,
        rotation_fill=args.aug_rotation_fill,
    )
    train_ds = EmotionSiglipDataset(tr_recs, pil_transform=train_pil_tf)
    val_ds = EmotionSiglipDataset(vl_recs, pil_transform=val_pil_tf)
    collate = make_collate_fn(processor)

    common_loader = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=_make_worker_init(args.seed),
        collate_fn=collate,
    )
    if args.num_workers > 0:
        common_loader["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(
        train_ds, shuffle=True, drop_last=False,
        generator=torch.Generator().manual_seed(args.seed),
        **common_loader,
    )
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common_loader)

    ckpt_path = PROJECT / "models" / f"{args.name}.pt"
    csv_rows: list[dict] = []
    stage_a_meta = None
    stage_b_meta = None

    # ===== class_weight 계산 (옵션, class imbalance 대응) =====
    class_weight_tensor = None
    if args.class_weight == "auto":
        counts = np.bincount(
            [r["class_idx"] for r in tr_recs], minlength=NUM_CLASSES,
        ).astype(np.float64)
        weights = counts.sum() / (NUM_CLASSES * np.maximum(counts, 1))
        class_weight_tensor = torch.tensor(
            weights, dtype=torch.float32, device=device,
        )
        print(f"[i] class_weight=auto → {weights.round(4).tolist()}")

    # ===== two-stage 분기 방어 ('--stage-a-epochs 0' 조합 시 의도 어긋남) =====
    if args.two_stage and args.stage_a_epochs <= 0:
        raise ValueError(
            "--two-stage 가 on 이면 --stage-a-epochs > 0 필수. "
            "Stage A skip 원하면 --two-stage 제거 + --unfreeze-encoder 로 single-stage unfreeze 실행."
        )

    # ===== Stage A (옵션) =====
    if args.two_stage and args.stage_a_epochs > 0:
        freeze_siglip_backbone(model)
        print(f"[i] Stage A (siglip freeze). trainable={count_trainable(model):,}")
        stage_a_ckpt = PROJECT / "models" / f"{args.name}.stageA.pt"
        stage_a_meta, run_a = run_stage(
            stage_name="A",
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr_head=args.lr,
            lr_backbone=args.lr_backbone,   # A 단계선 backbone freeze 라 사용 안 됨
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

        # Stage B: unfreeze
        unfreeze_siglip_backbone(model)
        print(f"[i] Stage B (siglip unfreeze). trainable={count_trainable(model):,}")

        stage_b_meta, run_b = run_stage(
            stage_name="B",
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr_head=args.lr,
            lr_backbone=args.lr_backbone,
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
        # single-stage: --unfreeze-encoder 에 따라 linear-probe 또는 full-ft
        if args.unfreeze_encoder:
            unfreeze_siglip_backbone(model)
            stage_label = "single-stage full-ft"
        else:
            freeze_siglip_backbone(model)
            stage_label = "single-stage linear-probe (siglip freeze)"
        print(f"[i] {stage_label}. trainable={count_trainable(model):,}")

        stage_b_meta, run_b = run_stage(
            stage_name="B",
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr_head=args.lr,
            lr_backbone=args.lr_backbone,
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

    # ===== 최종 save + 평가 =====
    torch.save(
        {
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "args": vars(args),
            "meta": {
                "stage_a_best": stage_a_meta,
                "stage_b_best": stage_b_meta,
                "classes": CLASSES,
                "img_size": args.img_size,
                "model_name": f"siglip_base_p16_{args.img_size}",
                "siglip_name": args.siglip_name,
                "projection_dim": args.projection_dim,
                "use_pixel_shuffle": bool(args.use_pixel_shuffle),
                "pixel_shuffle_ratio": args.pixel_shuffle_ratio,
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
