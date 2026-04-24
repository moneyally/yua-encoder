#!/usr/bin/env python
"""ViT-B/16 Full Combo = Soft Label (3-rater) + Mixup + SWA + 2-stage fine-tune.

설계 (design doc)
--------------------------------------------------------------------
- Base: timm vit_base_patch16_224, ImageNet21k pretrained
- 입력: 원본 이미지 → annot_A.boxes bbox crop → 224x224 resize → ImageNet normalize
- Label 신호 (2종 동시 사용):
    (a) soft label: build_soft_labels.py 가 만든 npz 의 3-rater 투표 분포
    (b) hard label: 폴더명 기반 one-hot (anger/happy/panic/sadness)
  Loss = α · KL(soft ∥ student) + (1 - α) · KL(hard_onehot ∥ student)
         ^                                     ^
         (soft 신호)                            (hard 신호, CE 와 수학적 동등)
  α = 0.5 (default)

- 2-stage fine-tune:
    Stage A: backbone freeze, head only 학습 (5 ep, lr 1e-3)
    Stage B: backbone 전체 unfreeze, 후반 학습 (15 ep, lr 5e-5)
       ep 5~11 (Stage B 전반): + Mixup α=0.2, prob=0.5 (CutMix 는 skip)
       ep 12~19 (Stage B 후반 50%): + SWA 시작

- Mixup + Soft Label 호환 구현: timm 안 쓰고 수동 구현 (self-consistent)
    lam ~ Beta(α, α)
    x_mix    = lam·x_i + (1-lam)·x_j
    soft_mix = lam·soft_i + (1-lam)·soft_j
    hard_mix = lam·onehot_i + (1-lam)·onehot_j

- SWA: torch.optim.swa_utils.AveragedModel + update_bn (ViT 는 LN 만 있어서 사실상 no-op)

- seed 42 고정 (torch/numpy/random + DataLoader worker init)

출력
--------------------------------------------------------------------
- models/exp10_vit_fullcombo.pt         (best val_acc, swa 전 모델)
- models/exp10_vit_fullcombo.swa.pt     (swa averaged model)
- logs/exp10_vit_fullcombo.csv          (epoch별 지표)
- logs/exp10_vit_fullcombo.meta.json    (args + best 수치)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

PROJECT = Path(os.environ.get(
    "EMOTION_FULLCOMBO_ROOT",
    str(Path(__file__).resolve().parent.parent),
))
CLASSES = ["anger", "happy", "panic", "sadness"]
NUM_CLASSES = 4
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


# --------------------------------------------------------------------
# 재현성
# --------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False    # 속도 위해 permit nondet
    torch.backends.cudnn.benchmark = True


def worker_init(worker_id: int):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


# --------------------------------------------------------------------
# 데이터셋
# --------------------------------------------------------------------
def load_json_eucmix(path: Path):
    for enc in ("euc-kr", "utf-8"):
        try:
            with open(path, encoding=enc) as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        except FileNotFoundError:
            return []
    return []


def build_records(data_root: Path, split: str):
    """폴더 기반 이미지 + annot_A bbox 매핑.
    returns: list[(img_path, bbox_xyxy_or_None, cls_idx)]
    """
    records = []
    img_root = data_root / "img" / split
    lbl_root = data_root / "label" / split

    for cls in CLASSES:
        cls_idx = CLASS_TO_IDX[cls]
        json_path = lbl_root / f"{split}_{cls}.json"
        data = load_json_eucmix(json_path)
        fname_to_bbox = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            fname = item.get("filename")
            if not fname:
                continue
            ann = item.get("annot_A") or {}
            box = ann.get("boxes") if isinstance(ann, dict) else None
            if isinstance(box, dict):
                try:
                    bbox = (float(box["minX"]), float(box["minY"]),
                            float(box["maxX"]), float(box["maxY"]))
                except Exception:
                    bbox = None
            else:
                bbox = None
            fname_to_bbox[fname] = bbox

        img_dir = img_root / cls
        if not img_dir.is_dir():
            continue
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            bbox = fname_to_bbox.get(p.name)
            records.append((str(p), bbox, cls_idx))
    return records


def load_soft_lookup(npz_path: Path) -> dict:
    """filename → soft_target(4,) dict."""
    d = np.load(str(npz_path), allow_pickle=False)
    fnames = [str(x) for x in d["filenames"]]
    soft = d["soft_targets"].astype(np.float32)
    return {f: soft[i] for i, f in enumerate(fnames)}


def clip_bbox(bbox, W, H):
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(W, int(round(x1))))
    y1 = max(0, min(H, int(round(y1))))
    x2 = max(0, min(W, int(round(x2))))
    y2 = max(0, min(H, int(round(y2))))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return (x1, y1, x2, y2)


class EmotionDataset(Dataset):
    """img + bbox crop + soft/hard label. augment 는 transform 에서."""

    def __init__(self, records, soft_lookup: dict, transform, crop: bool,
                 gt_onehot_smooth: float = 0.0):
        self.records = records
        self.soft_lookup = soft_lookup
        self.transform = transform
        self.crop = crop
        self.gt_onehot_smooth = gt_onehot_smooth

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        path, bbox, cls_idx = self.records[i]
        pil = Image.open(path).convert("RGB")
        if self.crop and bbox is not None:
            W, H = pil.size
            clipped = clip_bbox(bbox, W, H)
            if clipped is not None:
                pil = pil.crop(clipped)
        x = self.transform(pil)

        # hard label → one-hot (smooth 지원)
        hard = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        if self.gt_onehot_smooth > 0:
            eps = self.gt_onehot_smooth
            hard.fill_(eps / (NUM_CLASSES - 1))
            hard[cls_idx] = 1.0 - eps
        else:
            hard[cls_idx] = 1.0

        # soft label (raw 한번 더 검증)
        fname = Path(path).name
        soft_np = self.soft_lookup.get(fname)
        if soft_np is None:
            # fallback: hard 와 동일
            soft = hard.clone()
        else:
            soft = torch.from_numpy(soft_np).float()
            s = float(soft.sum())
            if not math.isfinite(s) or abs(s - 1.0) > 1e-4:
                soft = hard.clone()

        return x, soft, hard, cls_idx


# --------------------------------------------------------------------
# 모델 (ViT-B/16 via timm)
# --------------------------------------------------------------------
def build_vit(backbone: str = "vit_base_patch16_224",
              num_classes: int = 4,
              pretrained: bool = True,
              img_size: int = 224):
    """timm backbone fine-tune. DINOv3/DINOv2 등 백본 교체 가능."""
    import timm
    kwargs = {"pretrained": pretrained, "num_classes": num_classes}
    # img_size 조정이 필요한 백본 (DINOv2 등) 대응
    kwargs["img_size"] = img_size
    try:
        model = timm.create_model(backbone, **kwargs)
    except TypeError:
        # img_size 를 지원 안 하는 백본 fallback
        kwargs.pop("img_size", None)
        model = timm.create_model(backbone, **kwargs)
    return model


def set_backbone_trainable(model, trainable: bool):
    """timm 범용: get_classifier() 로 head 동적 감지. head 항상 True, 나머지 trainable."""
    try:
        head = model.get_classifier()
        head_ids = {id(p) for p in head.parameters()}
    except Exception:
        head_ids = set()
    for name, p in model.named_parameters():
        # fallback: head/classifier/fc 이름도 커버
        is_head = (id(p) in head_ids) or any(k in name for k in ("head", "classifier", "fc."))
        p.requires_grad = True if is_head else trainable


# --------------------------------------------------------------------
# Mixup / CutMix (수동, soft label 과 정합)
# --------------------------------------------------------------------
def _mixup_inner(x, soft, hard, alpha: float, rng):
    lam = float(rng.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1.0 - lam) * x[idx]
    soft_mix = lam * soft + (1.0 - lam) * soft[idx]
    hard_mix = lam * hard + (1.0 - lam) * hard[idx]
    return x_mix, soft_mix, hard_mix, lam


def _cutmix_inner(x, soft, hard, alpha: float, rng):
    lam = float(rng.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    B, C, H, W = x.shape
    cut_rat = (1.0 - lam) ** 0.5
    cut_h = int(H * cut_rat)
    cut_w = int(W * cut_rat)
    cy = int(rng.integers(0, H)) if H > 0 else 0
    cx = int(rng.integers(0, W)) if W > 0 else 0
    y1 = max(0, cy - cut_h // 2)
    y2 = min(H, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(W, cx + cut_w // 2)
    if y2 > y1 and x2 > x1:
        x_mix = x.clone()
        x_mix[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
        lam = 1.0 - ((y2 - y1) * (x2 - x1)) / float(H * W)
    else:
        x_mix = x
    soft_mix = lam * soft + (1.0 - lam) * soft[idx]
    hard_mix = lam * hard + (1.0 - lam) * hard[idx]
    return x_mix, soft_mix, hard_mix, lam


def mixup_batch(x, soft, hard, alpha: float, prob: float, rng,
                cutmix_alpha: float = 0.0, switch_prob: float = 0.5):
    """Mixup + CutMix 통합 (기존 시그니처 유지, 추가 인자는 기본값으로 off).

    - alpha: Mixup beta(α, α). <=0 이거나 prob 실패 시 원본 반환.
    - cutmix_alpha >0: prob 성공 시 switch_prob 확률로 CutMix 적용, 아니면 Mixup.
    - cutmix_alpha==0: 기존 Mixup-only 경로 유지 (역호환).
    """
    if rng.random() > prob:
        return x, soft, hard, 1.0
    use_cutmix = (cutmix_alpha > 0) and (rng.random() < switch_prob)
    if use_cutmix:
        return _cutmix_inner(x, soft, hard, cutmix_alpha, rng)
    if alpha <= 0:
        return x, soft, hard, 1.0
    return _mixup_inner(x, soft, hard, alpha, rng)


# --------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------
def combined_loss(logits, soft, hard, alpha_kl=0.5):
    """α · KL(soft ∥ student) + (1-α) · KL(hard ∥ student)
    - soft/hard 둘 다 probabilities (sum=1).
    - KL(p ∥ q) = sum p·log(p/q).  p·log(p) 는 학습에 영향 없어서 cross-entropy 형태로 계산.
    """
    log_p = F.log_softmax(logits, dim=-1)
    kl_soft = -(soft * log_p).sum(dim=-1).mean()
    kl_hard = -(hard * log_p).sum(dim=-1).mean()
    return alpha_kl * kl_soft + (1.0 - alpha_kl) * kl_hard


# --------------------------------------------------------------------
# 학습
# --------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scheduler, scaler,
                    alpha_kl, mixup_alpha, mixup_prob, device, rng,
                    amp_dtype=torch.bfloat16, grad_clip=1.0,
                    cutmix_alpha: float = 0.0, cutmix_switch_prob: float = 0.5):
    model.train()
    total_loss, total_n = 0.0, 0
    for x, soft, hard, _ in loader:
        x = x.to(device, non_blocking=True)
        soft = soft.to(device, non_blocking=True)
        hard = hard.to(device, non_blocking=True)

        x, soft, hard, _ = mixup_batch(x, soft, hard, mixup_alpha, mixup_prob, rng,
                                        cutmix_alpha=cutmix_alpha,
                                        switch_prob=cutmix_switch_prob)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype is not None)):
            logits = model(x)
            loss = combined_loss(logits, soft, hard, alpha_kl=alpha_kl)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += float(loss.detach()) * x.size(0)
        total_n += x.size(0)
    return total_loss / max(1, total_n)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds_all, labels_all, probs_all = [], [], []
    for x, _, _, cls_idx in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = F.softmax(logits.float(), dim=-1).cpu().numpy()
        probs_all.append(probs)
        preds_all.append(np.argmax(probs, axis=1))
        labels_all.append(np.asarray(cls_idx))
    probs = np.concatenate(probs_all, axis=0)
    preds = np.concatenate(preds_all, axis=0)
    labels = np.concatenate(labels_all, axis=0)
    acc = float((preds == labels).mean())
    # macro F1
    f1s = []
    for c in range(NUM_CLASSES):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s))
    eps = 1e-8
    p_true = probs[np.arange(len(labels)), labels]
    nll = float(-np.log(np.clip(p_true, eps, 1.0)).mean())
    return {"acc": acc, "macro_f1": macro_f1, "nll": nll,
            "probs": probs, "labels": labels}


# --------------------------------------------------------------------
# 메인
# --------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="exp10_vit_fullcombo")
    ap.add_argument("--backbone", default="vit_base_patch16_224",
                    help="timm backbone name. 예: vit_base_patch16_dinov3 / vit_base_patch14_dinov2")
    ap.add_argument("--img-size", type=int, default=224,
                    help="입력 해상도. DINOv3 16/224, DINOv2 14/518 권장")
    ap.add_argument("--data-root", default=str(PROJECT / "data_rot"))
    ap.add_argument("--soft-dir", default=str(PROJECT / "results" / "soft_labels"))
    ap.add_argument("--out-models", default=str(PROJECT / "models"))
    ap.add_argument("--out-logs", default=str(PROJECT / "logs"))
    # training
    ap.add_argument("--stage-a-epochs", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20,
                    help="total (Stage A + Stage B). Stage A = --stage-a-epochs, rest = Stage B.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--min-lr-ratio", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    # fullcombo knobs
    ap.add_argument("--alpha-kl", type=float, default=0.5,
                    help="Loss = α·KL(soft) + (1-α)·KL(hard_onehot)")
    ap.add_argument("--mixup-alpha", type=float, default=0.2)
    ap.add_argument("--mixup-prob", type=float, default=0.5)
    ap.add_argument("--cutmix-alpha", type=float, default=0.0,
                    help=">0 이면 Mixup 과 switch-prob 으로 섞어서 적용")
    ap.add_argument("--cutmix-switch-prob", type=float, default=0.5,
                    help="prob 성공 후 Mixup vs CutMix 전환 확률")
    ap.add_argument("--rand-augment", action="store_true", default=False,
                    help="RandAugment 사용 (num-ops, magnitude 하이퍼 적용)")
    ap.add_argument("--ra-num-ops", type=int, default=2)
    ap.add_argument("--ra-magnitude", type=int, default=7)
    ap.add_argument("--mixup-start-ep", type=int, default=5,
                    help="Stage B 시작 epoch (ep>=이 값 부터 mixup 적용)")
    ap.add_argument("--swa-start-ep", type=int, default=12,
                    help="SWA 시작 epoch (Stage B 후반 50%%)")
    # misc
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--amp", choices=["bf16", "none"], default="bf16")
    ap.add_argument("--crop", action="store_true", default=True)
    ap.add_argument("--no-crop", dest="crop", action="store_false")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = Path(args.data_root)
    soft_dir = Path(args.soft_dir)
    out_models = Path(args.out_models); out_models.mkdir(parents=True, exist_ok=True)
    out_logs = Path(args.out_logs); out_logs.mkdir(parents=True, exist_ok=True)

    # 1) 데이터 준비
    print(f"[data] data_root={data_root}")
    train_recs = build_records(data_root, "train")
    val_recs = build_records(data_root, "val")
    train_soft = load_soft_lookup(soft_dir / "train_soft.npz")
    val_soft = load_soft_lookup(soft_dir / "val_soft.npz")
    print(f"       train {len(train_recs)}  val {len(val_recs)}")
    print(f"       soft: train {len(train_soft)}  val {len(val_soft)}")

    img_size = int(args.img_size)
    resize_base = max(img_size + 32, int(round(img_size * 1.15)))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_ops = [
        transforms.Resize((resize_base, resize_base)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
    ]
    if args.rand_augment:
        # RandAugment 는 PIL 입력 필요 → ColorJitter/Rotation 보다 먼저.
        train_ops.append(transforms.RandAugment(
            num_ops=int(args.ra_num_ops),
            magnitude=int(args.ra_magnitude),
        ))
    else:
        train_ops += [
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.1, hue=0.05),
            transforms.RandomRotation(10),
        ]
    train_ops += [transforms.ToTensor(), normalize]
    tf_train = transforms.Compose(train_ops)
    tf_val = transforms.Compose([
        transforms.Resize((resize_base, resize_base)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])

    train_ds = EmotionDataset(train_recs, train_soft, tf_train, crop=args.crop)
    val_ds = EmotionDataset(val_recs, val_soft, tf_val, crop=args.crop)

    gen = torch.Generator()
    gen.manual_seed(args.seed)
    nw = int(args.num_workers)
    loader_kwargs = dict(
        num_workers=nw,
        pin_memory=True,
        persistent_workers=(nw > 0),
        prefetch_factor=(4 if nw > 0 else None),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              worker_init_fn=worker_init, generator=gen,
                              drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            **loader_kwargs)

    # 2) 모델
    print(f"[model] timm {args.backbone} pretrained  img_size={img_size}")
    model = build_vit(backbone=args.backbone, num_classes=NUM_CLASSES,
                      pretrained=True, img_size=img_size).to(device)

    # 3) Stage A: head only, 5 epoch
    set_backbone_trainable(model, trainable=False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr_head,
                                   weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    total_steps_a = steps_per_epoch * args.stage_a_epochs
    warmup_steps = steps_per_epoch * min(args.warmup_epochs, args.stage_a_epochs)

    def make_cosine(warmup, total, min_lr_ratio=0.01):
        def fn(step):
            if step < warmup:
                return (step + 1) / max(1, warmup)
            prog = (step - warmup) / max(1, total - warmup)
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * prog))
        return fn

    sched_a = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=make_cosine(warmup_steps, total_steps_a, args.min_lr_ratio))

    amp_dtype = torch.bfloat16 if args.amp == "bf16" else None
    scaler = None  # bf16 은 scaler 불필요

    rng = np.random.RandomState(args.seed)
    csv_path = out_logs / f"{args.name}.csv"
    meta_path = out_logs / f"{args.name}.meta.json"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,stage,train_loss,val_acc,val_f1,val_nll,phase\n")

    best_acc = -1.0
    best_f1 = -1.0
    best_nll = float("inf")
    best_state = None

    for ep in range(args.stage_a_epochs):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, sched_a, scaler,
            alpha_kl=args.alpha_kl,
            mixup_alpha=0.0, mixup_prob=0.0,   # Stage A 는 mixup off
            device=device, rng=rng, amp_dtype=amp_dtype,
            grad_clip=args.grad_clip)
        val = evaluate(model, val_loader, device)
        dt = time.perf_counter() - t0
        print(f"[A ep{ep+1}/{args.stage_a_epochs}] loss={train_loss:.4f}  "
              f"val_acc={val['acc']:.4f}  f1={val['macro_f1']:.4f}  "
              f"nll={val['nll']:.4f}  ({dt:.0f}s)")
        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(f"{ep+1},A,{train_loss:.6f},{val['acc']:.6f},"
                    f"{val['macro_f1']:.6f},{val['nll']:.6f},stageA\n")
        if val["acc"] > best_acc:
            best_acc = val["acc"]
            best_f1 = val["macro_f1"]
            best_nll = val["nll"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # 4) Stage B: backbone unfreeze, 15 epoch (+Mixup, +SWA 후반)
    set_backbone_trainable(model, trainable=True)
    # group lr: backbone lr-backbone, head lr-head/10 정도 (head 이미 학습됨)
    head_params, backbone_params = [], []
    for n, p in model.named_parameters():
        (head_params if "head" in n else backbone_params).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": args.lr_backbone},
         {"params": head_params, "lr": args.lr_head * 0.1}],
        weight_decay=args.weight_decay)

    stage_b_epochs = args.epochs - args.stage_a_epochs
    total_steps_b = steps_per_epoch * stage_b_epochs
    warmup_b_steps = steps_per_epoch * min(args.warmup_epochs, stage_b_epochs)
    sched_b = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=make_cosine(warmup_b_steps, total_steps_b, args.min_lr_ratio))

    # SWA
    swa_model = AveragedModel(model)
    swa_activated = False

    for local_ep in range(stage_b_epochs):
        global_ep = args.stage_a_epochs + local_ep + 1   # 1-indexed global

        # mixup 적용 여부 (Stage B 전체)
        mix_a = args.mixup_alpha
        mix_p = args.mixup_prob

        t0 = time.perf_counter()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, sched_b, scaler,
            alpha_kl=args.alpha_kl,
            mixup_alpha=mix_a, mixup_prob=mix_p,
            device=device, rng=rng, amp_dtype=amp_dtype,
            grad_clip=args.grad_clip,
            cutmix_alpha=args.cutmix_alpha,
            cutmix_switch_prob=args.cutmix_switch_prob)

        # SWA 활성: global_ep >= swa_start_ep
        if global_ep >= args.swa_start_ep:
            swa_model.update_parameters(model)
            swa_activated = True

        val = evaluate(model, val_loader, device)
        dt = time.perf_counter() - t0
        swa_tag = " swa" if swa_activated else ""
        print(f"[B ep{local_ep+1}/{stage_b_epochs} (g{global_ep})] loss={train_loss:.4f}  "
              f"val_acc={val['acc']:.4f}  f1={val['macro_f1']:.4f}  "
              f"nll={val['nll']:.4f}  ({dt:.0f}s){swa_tag}")
        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(f"{global_ep},B,{train_loss:.6f},{val['acc']:.6f},"
                    f"{val['macro_f1']:.6f},{val['nll']:.6f},stageB\n")

        if val["acc"] > best_acc:
            best_acc = val["acc"]
            best_f1 = val["macro_f1"]
            best_nll = val["nll"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # 5) SWA finalize (update_bn, eval)
    swa_val = None
    if swa_activated:
        print("[swa] update_bn (ViT 는 LN 만 있어 no-op 에 가깝지만 호출)")
        update_bn(train_loader, swa_model, device=device)
        swa_val = evaluate(swa_model, val_loader, device)
        print(f"[swa] val_acc={swa_val['acc']:.4f}  f1={swa_val['macro_f1']:.4f}  "
              f"nll={swa_val['nll']:.4f}")

    # 6) 저장
    ckpt_path = out_models / f"{args.name}.pt"
    torch.save({
        "model": best_state,
        "args": vars(args),
        "classes": CLASSES,
        "best_val_acc": best_acc,
    }, str(ckpt_path))
    print(f"[save] best ckpt → {ckpt_path}  (val_acc={best_acc:.4f})")

    if swa_activated:
        swa_path = out_models / f"{args.name}.swa.pt"
        # AveragedModel 은 내부 module 을 'module.' prefix + n_averaged 스칼라와 함께 저장한다.
        # predict.py / timm loader 가 곧바로 쓸 수 있도록 inner module 의 state_dict 를 꺼낸다.
        inner_sd = swa_model.module.state_dict()
        torch.save({
            "model": {k: v.detach().cpu() for k, v in inner_sd.items()},
            "args": vars(args),
            "classes": CLASSES,
            "swa_val_acc": swa_val["acc"],
            "_swa_stripped": True,
        }, str(swa_path))
        print(f"[save] swa ckpt → {swa_path}  (val_acc={swa_val['acc']:.4f})")

    # 7) meta json
    meta = {
        "name": args.name,
        "model_name": args.backbone,
        "img_size": int(args.img_size),
        "args": vars(args),
        "best_val_acc": best_acc,
        "best_val_f1": best_f1,
        "best_val_nll": best_nll,
        "swa_val_acc": swa_val["acc"] if swa_val else None,
        "swa_val_f1": swa_val["macro_f1"] if swa_val else None,
        "swa_val_nll": swa_val["nll"] if swa_val else None,
        "classes": CLASSES,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[save] meta → {meta_path}")


if __name__ == "__main__":
    main()
