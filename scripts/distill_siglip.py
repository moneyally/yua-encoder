"""distill_siglip.py — SigLIP student KD (Knowledge Distillation).

Teacher: ensemble (via gen_teacher_soft.py 의 npz, 이미 weighted softmax voting / optional TS 적용).
Student: SigLIP + yua-encoder wrapper (E6 ckpt 에서 시작, build_model_siglip 재사용).

Loss (Hinton 2015):
  L = α · T² · KL(log_softmax(s/T), p_t) + (1-α) · CE(s, hard)
  KL direction: PyTorch `F.kl_div(log_input, target_prob, reduction='batchmean')` — 수학적 KL 일치.

Unfreeze TRAP 방어 (vision_encoder.py L488 확인):
  encode_image() 내부가 config.freeze_encoder=True 면 torch.no_grad() 로 감싸므로,
  partial unfreeze 시 반드시:
    1) config.freeze_encoder = False
    2) backbone 전체 requires_grad=False
    3) 마지막 N 블록만 requires_grad=True
  순서로 설정. (no_grad 컨텍스트 자체를 해제하지 않으면 gradient 안 흐름.)

제1헌법: WARN/FAIL/CRITICAL 즉시 중단 (raise SystemExit).
  - NaN loss, shape 불일치, soft npz 깨짐, prev_val_acc 미달 save → stop.
제2헌법: 추측 금지 — KL/T²/unfreeze 경로는 공식 문서 + 소스로 검증 완료.
"""
from __future__ import annotations

import argparse
import csv as _csv
import datetime as dt
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

PROJECT = Path(os.environ.get(
    "EMOTION_PROJECT_ROOT",
    str(Path(__file__).resolve().parent.parent),
))
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# 재사용 — train_siglip.py (모델 빌더 + transforms + scheduler)
from train_siglip import (  # noqa: E402
    CLASSES, NUM_CLASSES,
    SiglipEmotionClassifier, build_model_siglip,
    make_warmup_cosine,
    EmotionSiglipDataset, make_pil_transforms, make_collate_fn,
)
from train_vit import (  # noqa: E402
    build_crop_records, build_folder_records,
    set_seed, _make_worker_init,
)

DEFAULT_DATA = PROJECT / "data_rot"


# ==========================================================================
# Soft target dataset — teacher_probs 를 record 에 lookup 해서 반환
# ==========================================================================
class SoftTargetSiglipDataset(Dataset):
    """train_siglip.EmotionSiglipDataset 와 동일 패턴 — PIL 반환.
    soft target 을 (class, filename) key 로 lookup 후 tuple 반환.

    __getitem__ 반환: (PIL.Image, hard_label:int, soft:np.ndarray[4], weight:float)
    (processor 변환은 make_soft_collate_fn 에서 일괄 수행 — SSOT 원칙)

    records 자동 필터: teacher npz 에 있는 key 만 keep.
    이유: teacher 가 min_conf drop / limit-samples 로 subset 생성 가능.
          0 건이면 FAIL (학습 불가), 일부 drop 은 정상 로그만.
    """
    def __init__(self,
                 records: list[dict],
                 soft_lookup: dict[tuple[str, str], np.ndarray],
                 weight_lookup: dict[tuple[str, str], float],
                 pil_transform: Optional[Callable] = None):
        # 자동 필터링 + 통계 로그
        kept, dropped_by_class = [], {c: 0 for c in CLASSES}
        for r in records:
            cls = CLASSES[int(r["class_idx"])]
            key = (cls, Path(r["path"]).name)
            if key in soft_lookup:
                kept.append(r)
            else:
                dropped_by_class[cls] += 1
        total_dropped = sum(dropped_by_class.values())
        if total_dropped > 0:
            print(f"[records-filter] soft lookup 미포함 drop: {total_dropped} "
                  f"(per-class: {dropped_by_class})")
        if not kept:
            raise SystemExit(
                "[FAIL] records 0 건 — 전부 teacher lookup 에 없음. "
                "teacher npz 규모/키 확인 필요."
            )
        print(f"[records-filter] kept={len(kept)} / input={len(records)}")
        self.records = kept
        self.soft_lookup = soft_lookup
        self.weight_lookup = weight_lookup
        self.pil_transform = pil_transform

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
        hard = int(r["class_idx"])
        key = (CLASSES[hard], Path(r["path"]).name)
        soft = self.soft_lookup[key]  # np.ndarray (4,)
        # 제1헌법: soft 검증 (per-item; npz 전체 검증은 load_teacher_soft 에서 완료)
        if soft.shape[0] != NUM_CLASSES or not np.all(np.isfinite(soft)):
            raise RuntimeError(
                f"[FAIL] soft target 이상 key={key} shape={soft.shape}"
            )
        w = float(self.weight_lookup.get(key, 1.0))
        return img, hard, soft, w


def make_soft_collate_fn(processor):
    """list[(PIL, hard, soft_np, w)] → (pixel_values, hard, soft, weight).
    SSOT: processor 는 model.vision_encoder.processor 에서만 얻음.
    """
    def collate(batch):
        imgs = [b[0] for b in batch]
        hard = torch.tensor([b[1] for b in batch], dtype=torch.long)
        soft = torch.tensor(np.stack([b[2] for b in batch], axis=0),
                            dtype=torch.float32)
        weight = torch.tensor([b[3] for b in batch], dtype=torch.float32)
        inputs = processor(images=imgs, return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        return pixel_values, hard, soft, weight
    return collate


# ==========================================================================
# KD utilities
# ==========================================================================
def kd_loss(student_logits: torch.Tensor,
            teacher_probs: torch.Tensor,
            hard_labels: torch.Tensor,
            T: float,
            alpha: float,
            label_smoothing: float = 0.0,
            sample_weight: Optional[torch.Tensor] = None) -> dict:
    """KD loss (Hinton 2015). 반환: dict{total, soft, hard}.
    sample_weight (B,) 제공 시 per-sample weighted mean (batch 평균 대신).
    """
    if T <= 0:
        raise ValueError(f"[FAIL] T must be > 0: {T}")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"[FAIL] alpha must be [0,1]: {alpha}")

    # Soft (KL) — PyTorch 공식: input=log-prob, target=prob, reduction='batchmean'
    log_p_s = F.log_softmax(student_logits / T, dim=-1)
    if sample_weight is None:
        soft = F.kl_div(log_p_s, teacher_probs, reduction="batchmean") * (T * T)
    else:
        # 수식: Σ_i Σ_c p_t[i,c] * (log p_t[i,c] - log_p_s[i,c]) / B (batchmean)
        # sample weight 주려면 per-sample 계산 후 가중 평균.
        eps = 1e-12
        t_log = torch.log(teacher_probs.clamp(min=eps))
        per_sample = (teacher_probs * (t_log - log_p_s)).sum(dim=-1)  # (B,)
        soft = (per_sample * sample_weight).sum() / sample_weight.sum().clamp(min=eps)
        soft = soft * (T * T)

    # Hard CE
    hard = F.cross_entropy(
        student_logits, hard_labels,
        label_smoothing=label_smoothing, reduction="mean",
    )
    total = alpha * soft + (1.0 - alpha) * hard
    return {"total": total, "soft": soft.detach(), "hard": hard.detach()}


# ==========================================================================
# Model — unfreeze last_n blocks (TRAP 방어)
# ==========================================================================
def configure_unfreeze(model: SiglipEmotionClassifier, unfreeze_last_n: int) -> dict:
    """backbone 의 마지막 n 블록만 unfreeze. projection/classifier 는 항상 train.

    TRAP 방어: config.freeze_encoder=True 면 encode_image 가 torch.no_grad() 로 감싸므로
    parameter requires_grad=True 로 바꿔도 gradient 흐르지 않음. 따라서 config 도 False.
    """
    ve = model.vision_encoder
    total_layers = len(ve.siglip.vision_model.encoder.layers)
    if unfreeze_last_n < 0 or unfreeze_last_n > total_layers:
        raise SystemExit(
            f"[FAIL] unfreeze_last_n 범위: 0..{total_layers} (입력 {unfreeze_last_n})"
        )

    # 1) no_grad 컨텍스트 해제 여부
    if unfreeze_last_n == 0:
        # linear probe — backbone 전부 freeze 유지
        ve.config.freeze_encoder = True
        for p in ve.siglip.parameters():
            p.requires_grad_(False)
        ve.siglip.eval()
    else:
        # partial unfreeze — no_grad 해제 필수
        ve.config.freeze_encoder = False
        # 전부 freeze 후 마지막 n 만 enable
        for p in ve.siglip.parameters():
            p.requires_grad_(False)
        target_layers = ve.siglip.vision_model.encoder.layers[total_layers - unfreeze_last_n:]
        for layer in target_layers:
            for p in layer.parameters():
                p.requires_grad_(True)
        ve.siglip.train()

    # 2) projection / head 항상 train
    for p in ve.projection.parameters():
        p.requires_grad_(True)
    for p in model.classifier.parameters():
        p.requires_grad_(True)

    info = {
        "total_layers": total_layers,
        "unfrozen_layers": unfreeze_last_n,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "backbone_trainable": sum(
            p.numel() for p in ve.siglip.parameters() if p.requires_grad
        ),
        "config_freeze_encoder": ve.config.freeze_encoder,
    }
    print(f"[unfreeze] last_n={unfreeze_last_n}/{total_layers}  "
          f"trainable={info['trainable_params']:,}  "
          f"backbone_trainable={info['backbone_trainable']:,}  "
          f"config.freeze_encoder={info['config_freeze_encoder']}")
    return info


def split_param_groups(model: SiglipEmotionClassifier,
                       lr_head: float,
                       lr_backbone: float,
                       weight_decay: float) -> list[dict]:
    """backbone (siglip trainable) vs head (projection+classifier) 2개 param group."""
    backbone_params, head_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("vision_encoder.siglip."):
            backbone_params.append(p)
        else:
            head_params.append(p)
    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": lr_head,
                       "weight_decay": weight_decay, "name": "head"})
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone,
                       "weight_decay": weight_decay, "name": "backbone"})
    return groups


# ==========================================================================
# Teacher soft load
# ==========================================================================
def load_teacher_soft(npz_path: Path) -> tuple[dict, dict, dict]:
    """(soft_lookup, weight_lookup, meta).
    soft_lookup: {(class, filename): probs (4,)}
    weight_lookup: {(class, filename): weight (float)}
    meta: dict (from .meta.json)
    """
    if not npz_path.is_file():
        raise SystemExit(f"[FAIL] teacher soft npz 없음: {npz_path}")
    data = np.load(str(npz_path), allow_pickle=False)
    needed = {"filenames", "classes", "teacher_probs",
              "class_idx_gt", "sample_weight"}
    missing = needed - set(data.files)
    if missing:
        raise SystemExit(f"[FAIL] npz 필수 key 누락: {missing} (files={data.files})")
    fnames = data["filenames"]
    classes_arr = data["classes"]
    probs = data["teacher_probs"]
    weights = data["sample_weight"]

    if probs.ndim != 2 or probs.shape[1] != NUM_CLASSES:
        raise SystemExit(
            f"[FAIL] teacher_probs shape: {probs.shape} "
            f"(expected (N,{NUM_CLASSES}))"
        )
    if not (len(fnames) == len(classes_arr) == len(probs) == len(weights)):
        raise SystemExit(
            f"[FAIL] 길이 불일치: fnames={len(fnames)} classes={len(classes_arr)} "
            f"probs={len(probs)} weights={len(weights)}"
        )
    if not np.all(np.isfinite(probs)):
        raise SystemExit("[FAIL] teacher_probs NaN/Inf")
    row_sums = probs.sum(axis=1)
    bad = np.where(np.abs(row_sums - 1.0) > 1e-3)[0]
    if bad.size > 0:
        raise SystemExit(
            f"[FAIL] teacher_probs row sum != 1: {bad.size}건 "
            f"(first 3 sums={row_sums[bad[:3]].tolist()})"
        )

    soft_lookup: dict[tuple[str, str], np.ndarray] = {}
    weight_lookup: dict[tuple[str, str], float] = {}
    for i in range(len(fnames)):
        key = (str(classes_arr[i]), str(fnames[i]))
        if key in soft_lookup:
            raise SystemExit(
                f"[FAIL] teacher npz 에 중복 key: {key}"
            )
        soft_lookup[key] = probs[i].astype(np.float32)
        weight_lookup[key] = float(weights[i])

    meta_path = npz_path.with_suffix(".meta.json")
    meta = {}
    if meta_path.is_file():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    print(f"[teacher] loaded {len(soft_lookup)} entries from {npz_path}")
    print(f"[teacher] meta.ensemble: {meta.get('ensemble_json')}")
    print(f"[teacher] meta.apply_ts={meta.get('apply_ts')} "
          f"T*={meta.get('ts_star')}  min_conf={meta.get('min_conf')}")
    return soft_lookup, weight_lookup, meta


# ==========================================================================
# Train / eval
# ==========================================================================
def train_one_epoch(model, loader, optimizer, scheduler, device,
                    amp_dtype, T: float, alpha: float,
                    label_smoothing: float, use_sample_weight: bool,
                    grad_clip_norm: float) -> dict:
    model.train()
    if (model.vision_encoder.siglip is not None
            and model.vision_encoder.config.freeze_encoder):
        model.vision_encoder.siglip.eval()

    total_loss = 0.0
    total_soft = 0.0
    total_hard = 0.0
    total_argmax_match = 0
    total_n = 0

    scaler = None
    use_amp = (amp_dtype is not None) and (device.type == "cuda")
    if use_amp and amp_dtype is torch.float16:
        try:
            scaler = torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler()

    for batch in loader:
        # collate order: (pixel_values, hard, soft, weight)
        pv, hard, soft, sw = batch
        pv = pv.to(device, non_blocking=True)
        hard = hard.to(device, non_blocking=True)
        soft = soft.to(device, non_blocking=True)
        sw = sw.to(device, non_blocking=True).float() if use_sample_weight else None

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(pv)
                ld = kd_loss(logits, soft, hard, T, alpha,
                             label_smoothing=label_smoothing,
                             sample_weight=sw)
                loss = ld["total"]
            # 제1헌법: NaN loss 즉시 중단
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"[FAIL] NaN/Inf loss at train. "
                    f"T={T} α={alpha} batch_size={pv.size(0)}"
                )
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
            logits = model(pv)
            ld = kd_loss(logits, soft, hard, T, alpha,
                         label_smoothing=label_smoothing,
                         sample_weight=sw)
            loss = ld["total"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"[FAIL] NaN/Inf loss (no-AMP)")
            loss.backward()
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    max_norm=grad_clip_norm,
                )
            optimizer.step()

        if scheduler is not None:
            scheduler.step()  # step-based warmup+cosine

        bs = pv.size(0)
        total_loss += float(loss.detach()) * bs
        total_soft += float(ld["soft"]) * bs
        total_hard += float(ld["hard"]) * bs
        total_argmax_match += int((logits.argmax(1) == hard).sum().item())
        total_n += bs

    return {
        "loss": total_loss / max(1, total_n),
        "soft_loss": total_soft / max(1, total_n),
        "hard_loss": total_hard / max(1, total_n),
        "argmax_vs_hard_acc": total_argmax_match / max(1, total_n),
        "n": total_n,
    }


@torch.no_grad()
def evaluate_hard(model, loader, device, amp_dtype) -> dict:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_correct, total_n = 0.0, 0, 0
    all_pred, all_true = [], []
    use_amp = (amp_dtype is not None) and (device.type == "cuda")
    for pv, hard in loader:
        pv = pv.to(device, non_blocking=True)
        hard = hard.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda",
                            dtype=(amp_dtype or torch.float16),
                            enabled=use_amp):
            logits = model(pv)
            loss = loss_fn(logits, hard)
        if not torch.isfinite(loss):
            raise RuntimeError("[FAIL] NaN/Inf val loss")
        bs = pv.size(0)
        total_loss += float(loss.detach()) * bs
        pred = logits.argmax(1)
        total_correct += int((pred == hard).sum().item())
        total_n += bs
        all_pred.append(pred.cpu().numpy())
        all_true.append(hard.cpu().numpy())
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
        "y_pred": y_pred, "y_true": y_true,
    }


# ==========================================================================
# CLI
# ==========================================================================
def parse_args():
    ap = argparse.ArgumentParser(
        description="SigLIP student KD (teacher=앙상블)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--src-ckpt", required=True,
                    help="E6 ckpt (SigLIP linear probe)")
    ap.add_argument("--teacher-soft", required=True,
                    help="gen_teacher_soft.py 출력 npz (train split)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--data-root", default=str(DEFAULT_DATA))
    # KD hyperparams
    ap.add_argument("--temperature", "-T", type=float, default=4.0,
                    help="KD temperature. sweep 권장: {1, 2, 4}")
    ap.add_argument("--alpha", type=float, default=0.7,
                    help="α * soft + (1-α) * hard. sweep 권장: {0.5, 0.7, 0.9}")
    ap.add_argument("--unfreeze-last-n", type=int, default=0,
                    help="backbone 마지막 N 블록 unfreeze. 0=linear probe, total=12")
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="KD 와 중복이라 기본 0. 설계 원칙: off.")
    ap.add_argument("--use-sample-weight", action="store_true",
                    help="teacher soft npz 의 sample_weight 적용 (conf-weight 모드 용)")
    # Optim
    ap.add_argument("--lr", type=float, default=5e-5,
                    help="head lr (KD low-lr).")
    ap.add_argument("--lr-backbone", type=float, default=5e-6,
                    help="backbone lr (layer-wise decay 10x 기본).")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=0,
                    help="step 단위 warmup (0=auto = 10% of total).")
    ap.add_argument("--min-lr-ratio", type=float, default=0.01)
    ap.add_argument("--grad-clip-norm", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--prefetch-factor", type=int, default=2)
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--img-size", type=int, default=384)
    # Data
    ap.add_argument("--crop", action="store_true", default=True)
    ap.add_argument("--no-crop", dest="crop", action="store_false")
    ap.add_argument("--augment", action="store_true", default=True)
    ap.add_argument("--no-augment", dest="augment", action="store_false")
    ap.add_argument("--aug-rot-deg", type=float, default=10.0)
    ap.add_argument("--aug-hflip-prob", type=float, default=0.5)
    ap.add_argument("--aug-jitter-bright", type=float, default=0.2)
    ap.add_argument("--aug-jitter-contrast", type=float, default=0.2)
    ap.add_argument("--aug-jitter-saturation", type=float, default=0.1)
    ap.add_argument("--aug-jitter-hue", type=float, default=0.05)
    ap.add_argument("--aug-rotation-fill", type=int, default=128)
    # Val override
    ap.add_argument("--val-dir", default=None)
    ap.add_argument("--val-label-root", default=None)
    # Save gate
    ap.add_argument("--overwrite-if-worse", action="store_true",
                    help="distilled < prev_val_acc 여도 저장. 기본 off (제1헌법: fail 시 중단).")
    ap.add_argument("--note", type=str, default="")
    args = ap.parse_args()

    # 경계 체크
    if args.lr <= 0 or args.lr > 1e-3:
        raise SystemExit(f"[FAIL] --lr (0, 1e-3]: {args.lr}")
    if args.lr_backbone <= 0 or args.lr_backbone > args.lr:
        raise SystemExit(
            f"[FAIL] --lr-backbone (0, lr]: {args.lr_backbone} (lr={args.lr})"
        )
    if args.epochs <= 0 or args.epochs > 10:
        raise SystemExit(f"[FAIL] --epochs [1,10]: {args.epochs}")
    if args.temperature <= 0 or args.temperature > 20:
        raise SystemExit(f"[FAIL] --temperature (0,20]: {args.temperature}")
    if not (0 <= args.alpha <= 1):
        raise SystemExit(f"[FAIL] --alpha [0,1]: {args.alpha}")
    return args


# Augment 는 train_siglip.make_pil_transforms 재사용 (SSOT).


# ==========================================================================
# main
# ==========================================================================
def main():
    args = parse_args()
    set_seed(args.seed, deterministic=False)

    if not torch.cuda.is_available():
        print("[WARN] CUDA 없음 — CPU fallback 매우 느림")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}
    amp_dtype = amp_map[args.amp] if device.type == "cuda" else None
    print(f"[i] device={device}  amp={args.amp}→{amp_dtype}")

    data_root = Path(args.data_root)
    if not (data_root / "img" / "train").is_dir():
        raise SystemExit(f"[FAIL] data_root/img/train 없음: {data_root}")

    PROJECT.joinpath("models").mkdir(exist_ok=True)
    PROJECT.joinpath("logs").mkdir(exist_ok=True)

    # ===== teacher soft =====
    soft_lookup, weight_lookup, teacher_meta = load_teacher_soft(
        Path(args.teacher_soft),
    )

    # ===== src ckpt =====
    src_path = Path(args.src_ckpt)
    if not src_path.is_file():
        raise SystemExit(f"[FAIL] src-ckpt 없음: {src_path}")
    try:
        ckpt = torch.load(str(src_path), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(src_path), map_location="cpu")
    if not isinstance(ckpt, dict) or "model" not in ckpt or "args" not in ckpt:
        raise SystemExit(f"[FAIL] src ckpt 포맷 비정상: keys={list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")
    src_args_dict = ckpt["args"]
    prev_val_acc = None
    sb = (ckpt.get("meta") or {}).get("stage_b_best") or {}
    if isinstance(sb, dict):
        prev_val_acc = sb.get("val_accuracy")
    if prev_val_acc is None:
        sa = (ckpt.get("meta") or {}).get("stage_a_best") or {}
        if isinstance(sa, dict):
            prev_val_acc = sa.get("val_accuracy")
    # final_eval fallback
    if prev_val_acc is None:
        fe = (ckpt.get("meta") or {}).get("eval") or {}
        prev_val_acc = fe.get("val_accuracy")
    print(f"[src] prev_val_acc = {prev_val_acc}")

    # model build — src args 기준으로 아키 맞추기 (projection_dim/img_size 등)
    # 제3헌법: hardcode 없이 src ckpt args 를 단일 source 로, 필요 override 만 최소.
    # (QA-Critical) 필수 key 존재 검증 — 누락 시 FAIL 즉시 중단
    _required_src_keys = (
        "img_size", "siglip_name", "projection_dim",
        "use_pixel_shuffle", "pixel_shuffle_ratio", "dropout",
    )
    _missing = [k for k in _required_src_keys if k not in src_args_dict]
    if _missing:
        raise SystemExit(
            f"[FAIL] src ckpt args 필수 key 누락: {_missing}. "
            f"E6 (train_siglip) 로 저장된 ckpt 인지 확인."
        )
    import types as _types
    m_args = _types.SimpleNamespace(**src_args_dict)
    m_args.unfreeze_encoder = (args.unfreeze_last_n > 0)
    if args.img_size and args.img_size != src_args_dict["img_size"]:
        print(f"[WARN] img_size override: {src_args_dict['img_size']} → {args.img_size}")
        m_args.img_size = args.img_size

    model = build_model_siglip(m_args, device)
    # state_dict load (strict=False — 혹시라도 추가 layer 있으면 warn)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"[WARN] missing keys: {len(missing)} (first: {list(missing)[:3]})")
    if unexpected:
        print(f"[WARN] unexpected keys: {len(unexpected)} (first: {list(unexpected)[:3]})")
    model = model.to(device)

    # ===== unfreeze 정책 =====
    unfreeze_info = configure_unfreeze(model, args.unfreeze_last_n)

    # ===== processor (SSOT: model.vision_encoder.processor 에서 직접) =====
    processor = model.vision_encoder.processor
    if processor is None:
        raise SystemExit("[FAIL] model.vision_encoder.processor is None — load_siglip 실패")
    # ===== transforms (train_siglip.make_pil_transforms 재사용) =====
    train_tf, _val_tf = make_pil_transforms(
        augment=args.augment,
        rot_deg=args.aug_rot_deg,
        hflip_prob=args.aug_hflip_prob,
        jitter_bright=args.aug_jitter_bright,
        jitter_contrast=args.aug_jitter_contrast,
        jitter_saturation=args.aug_jitter_saturation,
        jitter_hue=args.aug_jitter_hue,
        rotation_fill=args.aug_rotation_fill,
    )

    # ===== records =====
    if args.crop:
        tr_recs = build_crop_records(data_root, "train")
    else:
        tr_recs = build_folder_records(data_root, "train")
    # val override 처리 (간단히 기본 data_root 만 지원 — distill 은 override 미사용 가정)
    if args.crop:
        vl_recs = build_crop_records(data_root, "val")
    else:
        vl_recs = build_folder_records(data_root, "val")
    if len(tr_recs) == 0 or len(vl_recs) == 0:
        raise SystemExit(f"[FAIL] records 0 — train={len(tr_recs)} val={len(vl_recs)}")
    print(f"[records] train={len(tr_recs)}  val={len(vl_recs)}")

    # ===== Dataset =====
    train_ds = SoftTargetSiglipDataset(
        records=tr_recs,
        soft_lookup=soft_lookup,
        weight_lookup=weight_lookup,
        pil_transform=train_tf,
    )
    val_ds = EmotionSiglipDataset(vl_recs, pil_transform=None)

    common_loader = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=_make_worker_init(args.seed),
    )
    if args.num_workers > 0:
        common_loader["prefetch_factor"] = args.prefetch_factor

    # collate_fn — SigLIPImageProcessor 를 batch 단위로 일괄 적용 (SSOT)
    train_collate = make_soft_collate_fn(processor)
    val_collate = make_collate_fn(processor)

    train_loader = DataLoader(
        train_ds, shuffle=True, drop_last=False,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=train_collate,
        **common_loader,
    )
    val_loader = DataLoader(
        val_ds, shuffle=False, drop_last=False,
        collate_fn=val_collate,
        **common_loader,
    )

    # ===== optimizer (layer-wise LR) =====
    param_groups = split_param_groups(
        model, lr_head=args.lr, lr_backbone=args.lr_backbone,
        weight_decay=args.weight_decay,
    )
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    # step-based warmup + cosine
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = args.warmup_steps if args.warmup_steps > 0 else max(1, total_steps // 10)
    scheduler = make_warmup_cosine(
        optimizer, warmup_steps=warmup_steps,
        total_steps=total_steps, min_lr_ratio=args.min_lr_ratio,
    )
    print(f"[sched] steps/epoch={steps_per_epoch}  total_steps={total_steps}  "
          f"warmup_steps={warmup_steps}")

    # ===== train loop =====
    csv_rows: list[dict] = []
    best_val_acc = -float("inf")
    best_state = None
    best_meta = {}
    no_improve = 0
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        t_ep = time.time()
        tr = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, amp_dtype,
            T=args.temperature, alpha=args.alpha,
            label_smoothing=args.label_smoothing,
            use_sample_weight=args.use_sample_weight,
            grad_clip_norm=args.grad_clip_norm,
        )
        va = evaluate_hard(model, val_loader, device, amp_dtype)
        elapsed = time.time() - t_ep
        improved = va["acc"] > best_val_acc
        mark = "*" if improved else " "
        print(
            f"[ep {ep:02d}/{args.epochs}] "
            f"train total={tr['loss']:.4f} soft={tr['soft_loss']:.4f} "
            f"hard={tr['hard_loss']:.4f} argmax_vs_hard={tr['argmax_vs_hard_acc']:.4f} | "
            f"val loss={va['loss']:.4f} acc={va['acc']:.4f} f1={va['macro_f1']:.4f} "
            f"lr_head={optimizer.param_groups[0]['lr']:.2e} t={elapsed:.1f}s {mark}"
        )
        csv_rows.append({
            "epoch": ep,
            "train_loss": tr["loss"],
            "train_soft_loss": tr["soft_loss"],
            "train_hard_loss": tr["hard_loss"],
            "train_argmax_vs_hard": tr["argmax_vs_hard_acc"],
            "val_loss": va["loss"],
            "val_acc": va["acc"],
            "val_f1": va["macro_f1"],
            "lr_head": optimizer.param_groups[0]["lr"],
            "lr_backbone": (optimizer.param_groups[-1]["lr"]
                            if len(optimizer.param_groups) > 1 else 0.0),
            "time_sec": round(elapsed, 2),
        })
        if improved:
            best_val_acc = va["acc"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            best_meta = {
                "epoch": ep,
                "val_loss": va["loss"],
                "val_accuracy": va["acc"],
                "val_macro_f1": va["macro_f1"],
            }
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"[early-stop] ep={ep} no_improve={no_improve}≥patience={args.patience}")
                break

    if best_state is None:
        raise SystemExit("[FAIL] best_state None — 학습 중 improvement 없음")

    # ===== save gate (제1헌법) =====
    ckpt_path = PROJECT / "models" / f"{args.name}.pt"
    gate_pass = True
    if isinstance(prev_val_acc, (int, float)) and not args.overwrite_if_worse:
        if best_val_acc <= float(prev_val_acc):
            gate_pass = False
            print(f"[WARN] distilled {best_val_acc:.4f} ≤ prev {prev_val_acc:.4f} "
                  f"— save gate FAIL (제1헌법 적용)")

    if gate_pass:
        model.load_state_dict(best_state)
        final_eval = evaluate_hard(model, val_loader, device, amp_dtype)
        save_obj = {
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "args": {k: (str(v) if isinstance(v, Path) else v)
                     for k, v in vars(args).items()},
            "meta": {
                "prev_val_acc": prev_val_acc,
                "stage_b_best": best_meta,
                "eval": {
                    "val_loss": final_eval["loss"],
                    "val_accuracy": final_eval["acc"],
                    "val_macro_f1": final_eval["macro_f1"],
                    "n_val": final_eval["n"],
                },
                "unfreeze_info": unfreeze_info,
                "teacher_meta": teacher_meta,
                "distill": {
                    "temperature": args.temperature,
                    "alpha": args.alpha,
                    "label_smoothing": args.label_smoothing,
                    "use_sample_weight": args.use_sample_weight,
                    "lr": args.lr, "lr_backbone": args.lr_backbone,
                },
                "created_at": dt.datetime.now().isoformat(timespec="seconds"),
                "timing_sec": {"total_sec": time.time() - t0},
            },
        }
        torch.save(save_obj, str(ckpt_path))
        print(f"[SAVE] {ckpt_path}  val_acc={best_val_acc:.4f}")
    else:
        # 제1헌법: 실패 — 저장 안 하고 종료 (재시작 원칙)
        print("[EXIT] save gate 실패 — rollback. ckpt 저장 안함.")

    # ===== 로그 저장 (성공/실패 무관) =====
    logs_dir = PROJECT / "logs"
    csv_path = logs_dir / f"{args.name}.csv"
    if csv_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            for r in csv_rows:
                w.writerow(r)
    meta_json = logs_dir / f"{args.name}.meta.json"
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump({
            "name": args.name,
            "args": {k: (str(v) if isinstance(v, Path) else v)
                     for k, v in vars(args).items()},
            "history": csv_rows,
            "best": best_meta,
            "prev_val_acc": prev_val_acc,
            "save_gate_pass": gate_pass,
            "teacher_meta": teacher_meta,
            "unfreeze_info": unfreeze_info,
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        }, f, indent=2, ensure_ascii=False)

    # experiments.md append — 헤더 감지는 전체 파일 검사 (QA-Critical: byte slice 제거)
    exp_md = PROJECT / "experiments.md"
    header_needed = True
    if exp_md.is_file():
        try:
            _existing = exp_md.read_text(encoding="utf-8")
            if "| date |" in _existing or "| 날짜 |" in _existing:
                header_needed = False
        except Exception as _e:
            print(f"[WARN] experiments.md read 실패 ({_e}) — header 추가")
    line = (
        f"| {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} | {args.name} | "
        f"siglip_kd_unfreeze{args.unfreeze_last_n} | "
        f"bs={args.batch_size} img={args.img_size} T={args.temperature} "
        f"α={args.alpha} unfreeze={args.unfreeze_last_n} crop={int(args.crop)} "
        f"aug={int(args.augment)} | "
        f"epochs={len(csv_rows)}/{args.epochs} | val_acc={best_val_acc:.4f} | "
        f"val_loss={best_meta.get('val_loss', float('nan')):.4f} | "
        f"final_train_acc="
        f"{(csv_rows[-1].get('train_argmax_vs_hard', float('nan')) if csv_rows else float('nan')):.4f} | "
        f"note=KD teacher={Path(args.teacher_soft).name} save_gate={'pass' if gate_pass else 'FAIL'} |\n"
    )
    with open(exp_md, "a", encoding="utf-8") as f:
        if header_needed:
            f.write(
                "| date | name | model | config | epochs | val_acc | val_loss | "
                "train_acc | note |\n"
                "|---|---|---|---|---|---|---|---|---|\n"
            )
        f.write(line)

    if not gate_pass:
        # 제1헌법: 비정상 종료 코드로 sweep 스크립트에 신호
        raise SystemExit(2)


if __name__ == "__main__":
    main()
