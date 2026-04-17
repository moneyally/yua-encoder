"""gen_teacher_soft_batch.py — 앙상블 teacher soft (BATCH inference 최적).

기존 gen_teacher_soft.py 가 predict.py 의 단일 PIL API 로 sequential
→ 5996장 × 4모델 = ~90분. GPU util 평균 30% 이하.

BATCH 설계 (제4헌법 — 시간/환경 최적):
  - DataLoader(num_workers=16) 로 PIL 병렬 디코드 + bbox crop (CPU 96코어 활용)
  - 4 모델 native batch forward:
      TF .h5        → model.predict(batch_arr, batch_size=N)
      torch ViT     → timm model(batch_tensor) + autocast bf16
      torch SigLIP  → SiglipEmotionClassifier(batch_tensor) + autocast bf16
  - 각 batch 단위로 weighted softmax voting → 저장
  - OOM 방어: 각 모델 batch_size 인자화 (안전한 32 기본, 탐색 후 확대)

출력 포맷 = gen_teacher_soft.py 와 동일 (npz + meta.json).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROJECT = Path(os.environ.get(
    "EMOTION_PROJECT_ROOT",
    str(Path(__file__).resolve().parent.parent),
))
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# SSOT 재사용
from predict import CLASSES  # noqa: E402
from train_vit import build_crop_records, build_folder_records  # noqa: E402

NUM_CLASSES = len(CLASSES)


# ==========================================================================
# Runner — 모델별 native batch inference
# ==========================================================================

class BaseRunner:
    name: str = "base"
    weight: float = 1.0

    def predict_batch(self, pil_list: list) -> np.ndarray:
        """PIL list → softmax probs (B, NUM_CLASSES)."""
        raise NotImplementedError


class TFKerasRunner(BaseRunner):
    """TF .h5 (ResNet50 / EfficientNet) batch runner."""
    def __init__(self, h5_path: Path, weight: float,
                 family_hint: str | None = None,
                 batch_size: int = 32):
        import tensorflow as tf
        # GPU memory_growth 설정 (공유 방어)
        try:
            for g in tf.config.list_physical_devices("GPU"):
                try:
                    tf.config.experimental.set_memory_growth(g, True)
                except Exception:
                    pass
        except Exception:
            pass
        self._tf = tf
        self.name = f"tf_{h5_path.stem}"
        self.weight = float(weight)
        self.batch_size = int(batch_size)
        print(f"[runner/tf] loading {h5_path} ...")
        self.model = tf.keras.models.load_model(str(h5_path), compile=False)
        # family / preprocess / img_size 추론 — predict.py 로직 SSOT
        from predict import _infer_tf_family, _resolve_tf_preprocess, _read_meta_json
        meta = _read_meta_json(h5_path)
        if family_hint:
            self.family = family_hint
            self.img_size = 224
        else:
            self.family, self.img_size = _infer_tf_family(self.model, meta)
        self.preprocess = _resolve_tf_preprocess(self.family)
        print(f"[runner/tf] {self.name}  family={self.family}  img_size={self.img_size}  bs={self.batch_size}")

    def predict_batch(self, pil_list: list) -> np.ndarray:
        if not pil_list:
            return np.zeros((0, NUM_CLASSES), dtype=np.float32)
        arrs = []
        for p in pil_list:
            img = p.resize((self.img_size, self.img_size), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32)
            if arr.ndim == 2:  # grayscale 방어
                arr = np.stack([arr] * 3, axis=-1)
            arrs.append(arr)
        batch = np.stack(arrs, axis=0).copy()
        batch = self.preprocess(batch)
        out = self.model.predict(batch, batch_size=self.batch_size, verbose=0)
        probs = np.asarray(out, dtype=np.float32)
        if probs.ndim != 2 or probs.shape[1] != NUM_CLASSES:
            raise RuntimeError(
                f"[FAIL] TF out shape {probs.shape} (expected (B,{NUM_CLASSES}))"
            )
        # softmax 재정규화 (Dense softmax 면 이미 합=1, 방어적)
        row_sum = probs.sum(axis=1, keepdims=True)
        bad = (~np.isfinite(row_sum.squeeze())) | (np.abs(row_sum.squeeze() - 1.0) > 1e-3)
        if bad.any():
            # bad rows 에만 softmax 적용
            mx = probs[bad].max(axis=1, keepdims=True)
            ez = np.exp(probs[bad] - mx)
            probs[bad] = ez / ez.sum(axis=1, keepdims=True)
        return probs


class TorchViTRunner(BaseRunner):
    """torch ViT (timm) batch runner."""
    def __init__(self, pt_path: Path, weight: float,
                 device: torch.device, batch_size: int = 32,
                 amp_dtype=torch.bfloat16):
        import timm
        from torchvision import transforms as T
        self.name = f"torch_{pt_path.stem}"
        self.weight = float(weight)
        self.batch_size = int(batch_size)
        self.device = device
        self.amp_dtype = amp_dtype
        print(f"[runner/vit] loading {pt_path} ...")
        try:
            ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(str(pt_path), map_location="cpu")
        if not isinstance(ckpt, dict) or "model" not in ckpt:
            raise SystemExit(f"[FAIL] ViT ckpt 포맷: {pt_path}")
        model_name = (ckpt.get("meta") or {}).get("model_name") \
            or (ckpt.get("args") or {}).get("model") \
            or "vit_base_patch16_224"
        self.model = timm.create_model(
            model_name, pretrained=False, num_classes=NUM_CLASSES,
        )
        missing, unexpected = self.model.load_state_dict(ckpt["model"], strict=False)
        if missing or unexpected:
            print(f"[runner/vit] missing={len(missing)} unexpected={len(unexpected)}")
        self.model.eval().to(device)
        # transforms — train_vit val 경로와 동일 (Resize 256 → CenterCrop 224 → ImageNet norm)
        self.transforms = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print(f"[runner/vit] {self.name}  model={model_name}  bs={self.batch_size}")

    @torch.no_grad()
    def predict_batch(self, pil_list: list) -> np.ndarray:
        if not pil_list:
            return np.zeros((0, NUM_CLASSES), dtype=np.float32)
        tensors = [self.transforms(p.convert("RGB")) for p in pil_list]
        batch = torch.stack(tensors, 0).to(self.device, non_blocking=True)
        # 큰 batch 는 내부 split (OOM 방어)
        probs_list = []
        for i in range(0, batch.size(0), self.batch_size):
            sub = batch[i:i + self.batch_size]
            with torch.autocast(device_type="cuda",
                                dtype=self.amp_dtype,
                                enabled=(self.device.type == "cuda")):
                logits = self.model(sub)
            p = F.softmax(logits.float(), dim=-1).cpu().numpy()
            probs_list.append(p)
        probs = np.concatenate(probs_list, axis=0).astype(np.float32)
        if probs.shape[1] != NUM_CLASSES:
            raise RuntimeError(f"[FAIL] ViT shape: {probs.shape}")
        return probs


class TorchSigLIPRunner(BaseRunner):
    """torch SigLIP + yua-encoder wrapper batch runner."""
    def __init__(self, pt_path: Path, weight: float,
                 device: torch.device, batch_size: int = 16,
                 amp_dtype=torch.bfloat16):
        self.name = f"torch_{pt_path.stem}"
        self.weight = float(weight)
        self.batch_size = int(batch_size)
        self.device = device
        self.amp_dtype = amp_dtype
        print(f"[runner/siglip] loading {pt_path} ...")
        try:
            ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(str(pt_path), map_location="cpu")
        if not isinstance(ckpt, dict) or "model" not in ckpt or "args" not in ckpt:
            raise SystemExit(f"[FAIL] SigLIP ckpt 포맷: {pt_path}")
        src_args = ckpt["args"]
        # build_model_siglip 재사용
        from train_siglip import build_model_siglip
        import types as _types
        m_args = _types.SimpleNamespace(**src_args)
        # inference 전용 — unfreeze_encoder 는 False (로드만, 학습 X)
        m_args.unfreeze_encoder = False
        self.model = build_model_siglip(m_args, device)
        missing, unexpected = self.model.load_state_dict(ckpt["model"], strict=False)
        if missing or unexpected:
            print(f"[runner/siglip] missing={len(missing)} unexpected={len(unexpected)}")
        self.model.eval().to(device)
        # processor — SSOT (model 의 vision_encoder.processor)
        self.processor = self.model.vision_encoder.processor
        if self.processor is None:
            raise SystemExit("[FAIL] SigLIP processor None (load_siglip 실패)")
        print(f"[runner/siglip] {self.name}  siglip={src_args.get('siglip_name')}  "
              f"img_size={src_args.get('img_size')}  bs={self.batch_size}")

    @torch.no_grad()
    def predict_batch(self, pil_list: list) -> np.ndarray:
        if not pil_list:
            return np.zeros((0, NUM_CLASSES), dtype=np.float32)
        # processor 가 list 한 번에 처리 → 큰 batch 는 내부 split
        probs_list = []
        for i in range(0, len(pil_list), self.batch_size):
            sub_pil = [p.convert("RGB") for p in pil_list[i:i + self.batch_size]]
            inputs = self.processor(images=sub_pil, return_tensors="pt")
            pv = inputs["pixel_values"].to(self.device, non_blocking=True)
            with torch.autocast(device_type="cuda",
                                dtype=self.amp_dtype,
                                enabled=(self.device.type == "cuda")):
                logits = self.model(pv)
            p = F.softmax(logits.float(), dim=-1).cpu().numpy()
            probs_list.append(p)
        probs = np.concatenate(probs_list, axis=0).astype(np.float32)
        if probs.shape[1] != NUM_CLASSES:
            raise RuntimeError(f"[FAIL] SigLIP shape: {probs.shape}")
        return probs


# ==========================================================================
# Runner factory — ensemble json 의 model path 를 보고 runner 타입 선택
# ==========================================================================
def build_runner(model_path: Path, weight: float,
                 device: torch.device,
                 tf_batch: int, vit_batch: int, siglip_batch: int) -> BaseRunner:
    """확장자 + 파일명 힌트로 runner 타입 결정."""
    suf = model_path.suffix.lower()
    stem = model_path.stem.lower()
    if suf == ".h5":
        # family hint — exp02 resnet, exp04 efficientnet 식별
        hint = None
        if "resnet" in stem:
            hint = "resnet50"
        elif "effnet" in stem or "efficient" in stem:
            hint = "efficientnet"
        return TFKerasRunner(model_path, weight, family_hint=hint, batch_size=tf_batch)
    if suf == ".pt":
        if "siglip" in stem:
            return TorchSigLIPRunner(model_path, weight, device, batch_size=siglip_batch)
        return TorchViTRunner(model_path, weight, device, batch_size=vit_batch)
    raise SystemExit(f"[FAIL] 지원 안 하는 확장자: {model_path}")


# ==========================================================================
# Dataset — PIL + bbox crop 병렬 로딩
# ==========================================================================
class BboxCropDataset(Dataset):
    """__getitem__: (PIL 이미지, class_idx, filename, idx).
    collate_fn: list → (list[PIL], list[int], list[str], list[int])
    """
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        img = Image.open(r["path"])
        img.load()
        if img.mode != "RGB":
            img = img.convert("RGB")
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
        return img, int(r["class_idx"]), Path(r["path"]).name, idx

    @staticmethod
    def collate(batch):
        pils = [b[0] for b in batch]
        cls = [b[1] for b in batch]
        names = [b[2] for b in batch]
        idxs = [b[3] for b in batch]
        return pils, cls, names, idxs


# ==========================================================================
# TS — val 에서 T* 학습 (batch 경로에서도 동일)
# ==========================================================================
def fit_temperature(probs: np.ndarray, gt: np.ndarray,
                    lo: float = 0.5, hi: float = 10.0) -> float:
    from scipy.optimize import minimize_scalar
    eps = 1e-12
    logits = np.log(np.clip(probs, eps, 1.0))

    def nll(T: float) -> float:
        if T <= 0:
            return float("inf")
        z = logits / float(T)
        z = z - z.max(axis=1, keepdims=True)
        ez = np.exp(z)
        p = ez / ez.sum(axis=1, keepdims=True)
        p_gt = np.clip(p[np.arange(len(gt)), gt], eps, 1.0)
        return float(-np.mean(np.log(p_gt)))

    res = minimize_scalar(nll, bounds=(float(lo), float(hi)), method="bounded",
                          options={"xatol": 1e-4})
    return float(res.x)


def apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    if T is None or abs(T - 1.0) < 1e-6:
        return probs.astype(np.float32)
    eps = 1e-12
    logits = np.log(np.clip(probs, eps, 1.0))
    z = logits / float(T)
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return (ez / ez.sum(axis=1, keepdims=True)).astype(np.float32)


# ==========================================================================
# 검증
# ==========================================================================
def validate_probs(probs: np.ndarray, tag: str = "probs") -> None:
    if probs.ndim != 2 or probs.shape[1] != NUM_CLASSES:
        raise SystemExit(
            f"[{tag}] shape 오류: {tuple(probs.shape)} (expected (N,{NUM_CLASSES}))"
        )
    if not np.all(np.isfinite(probs)):
        raise SystemExit(f"[{tag}] NaN/Inf 포함")
    if not np.all(probs >= 0):
        raise SystemExit(f"[{tag}] 음수 확률 포함: min={probs.min()}")
    sums = probs.sum(axis=1)
    bad = np.where(np.abs(sums - 1.0) > 1e-3)[0]
    if bad.size > 0:
        raise SystemExit(
            f"[{tag}] row sum != 1: {bad.size}건 "
            f"(first 5: {sums[bad[:5]].tolist()})"
        )
    print(f"[{tag}] validate OK — shape={probs.shape}, "
          f"sum∈[{sums.min():.6f},{sums.max():.6f}]")


# ==========================================================================
# Ensemble batch inference
# ==========================================================================
def ensemble_predict_records(runners: list[BaseRunner],
                             records: list[dict],
                             num_workers: int,
                             batch_size: int,
                             prefetch_factor: int = 4) -> np.ndarray:
    """records 전체에 대해 앙상블 weighted softmax voting → (N, 4)."""
    ds = BboxCropDataset(records)
    loader = DataLoader(
        ds, batch_size=batch_size,
        shuffle=False,               # order preserve
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
        collate_fn=BboxCropDataset.collate,
        pin_memory=False,            # PIL 은 pin 불가
    )
    N = len(records)
    probs = np.zeros((N, NUM_CLASSES), dtype=np.float32)
    total_w = sum(r.weight for r in runners)
    if total_w <= 0:
        raise SystemExit(f"[FAIL] runners weight sum {total_w} ≤ 0")

    t0 = time.time()
    processed = 0
    for pil_list, cls_list, names, idxs in loader:
        bsz = len(pil_list)
        ens = np.zeros((bsz, NUM_CLASSES), dtype=np.float32)
        for runner in runners:
            p = runner.predict_batch(pil_list)
            if p.shape != (bsz, NUM_CLASSES):
                raise RuntimeError(
                    f"[FAIL] runner {runner.name} shape {p.shape} (expected ({bsz},{NUM_CLASSES}))"
                )
            if not np.all(np.isfinite(p)):
                raise RuntimeError(f"[FAIL] {runner.name} NaN/Inf")
            ens += runner.weight * p
        # normalize (weight sum !=1 여도 정확한 확률로)
        ens = ens / total_w
        # row sum ≈ 1 assert (weighted softmax 합 = Σwi * 1 / total_w = 1)
        rs = ens.sum(axis=1)
        if np.any(np.abs(rs - 1.0) > 1e-3):
            # 재정규화 (부동소수 오차)
            ens = ens / rs[:, None]
        # records 원래 순서로 돌려놓기 (idxs 그대로 shuffle=False 라 순차)
        probs[processed:processed + bsz] = ens
        processed += bsz
        if processed % 512 == 0 or processed == N:
            el = time.time() - t0
            rate = processed / max(1e-6, el)
            eta = (N - processed) / max(1e-6, rate)
            print(f"  [ensemble] {processed}/{N}  {rate:.1f} img/s  ETA {eta:.0f}s")
    print(f"[ensemble] done. elapsed={time.time()-t0:.1f}s  rate={N/max(1e-6,time.time()-t0):.1f} img/s")
    return probs


# ==========================================================================
# CLI / main
# ==========================================================================
def parse_args():
    ap = argparse.ArgumentParser(
        description="앙상블 teacher soft 생성 (BATCH 최적화 버전)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--ensemble-json", required=True)
    ap.add_argument("--data-root", default=str(PROJECT / "data_rot"))
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--crop", action="store_true", default=True)
    ap.add_argument("--no-crop", dest="crop", action="store_false")
    ap.add_argument("--apply-ts", action="store_true")
    ap.add_argument("--val-split-for-ts", default="val")
    ap.add_argument("--ts-bounds", nargs=2, type=float, default=[0.5, 10.0])
    ap.add_argument("--min-conf", type=float, default=0.0)
    ap.add_argument("--conf-weight", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-samples", type=int, default=0)
    # DataLoader / batch 튜닝
    ap.add_argument("--loader-batch-size", type=int, default=64,
                    help="DataLoader batch 단위 (각 runner 내부에서 더 작게 split 됨)")
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--prefetch-factor", type=int, default=4)
    # 각 runner 의 내부 batch (OOM 방어)
    ap.add_argument("--tf-batch", type=int, default=32)
    ap.add_argument("--vit-batch", type=int, default=64)
    ap.add_argument("--siglip-batch", type=int, default=16)
    args = ap.parse_args()
    if args.min_conf < 0 or args.min_conf > 1:
        raise SystemExit(f"[FAIL] --min-conf [0,1]: {args.min_conf}")
    return args


def main():
    args = parse_args()
    ensemble_json = Path(args.ensemble_json)
    data_root = Path(args.data_root)
    out_path = Path(args.out)
    if not ensemble_json.is_file():
        raise SystemExit(f"[FAIL] ensemble json 없음: {ensemble_json}")
    if not (data_root / "img" / args.split).is_dir():
        raise SystemExit(f"[FAIL] data_root/img/{args.split} 없음: {data_root}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ensemble config load
    with open(ensemble_json, encoding="utf-8") as f:
        cfg = json.load(f)
    sub_models = cfg.get("models") or []
    if not sub_models:
        raise SystemExit(f"[FAIL] ensemble json 'models' 비어있음: {ensemble_json}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] device={device}")
    print(f"[i] ensemble: {ensemble_json}  n_models={len(sub_models)}")

    # runner 생성 (4 모델 동시 GPU 로드)
    runners = []
    for m in sub_models:
        mp = Path(m["path"])
        if not mp.is_absolute():
            mp = PROJECT / mp
        if not mp.is_file():
            raise SystemExit(f"[FAIL] 모델 파일 없음: {mp}")
        r = build_runner(
            mp, float(m["weight"]), device,
            tf_batch=args.tf_batch,
            vit_batch=args.vit_batch,
            siglip_batch=args.siglip_batch,
        )
        runners.append(r)

    total_w = sum(r.weight for r in runners)
    print(f"[i] runners: {[r.name for r in runners]}")
    print(f"[i] weights: {[round(r.weight, 4) for r in runners]}  sum={total_w:.4f}")

    # ===== records =====
    if args.crop:
        records = build_crop_records(data_root, args.split)
    else:
        records = build_folder_records(data_root, args.split)
    if args.limit_samples > 0:
        from collections import defaultdict
        per_class = defaultdict(list)
        for r in records:
            per_class[int(r["class_idx"])].append(r)
        each = max(1, args.limit_samples // NUM_CLASSES)
        records = []
        for c in range(NUM_CLASSES):
            records.extend(per_class.get(c, [])[:each])
        print(f"[limit] stratified {each}/class = {len(records)}")
    print(f"[records] {args.split}: N={len(records)}")
    cls_counts = np.bincount(
        [r["class_idx"] for r in records], minlength=NUM_CLASSES,
    )
    print(f"[records] per-class: {dict(zip(CLASSES, cls_counts.tolist()))}")

    # ===== 앙상블 batch inference =====
    t0 = time.time()
    probs = ensemble_predict_records(
        runners=runners,
        records=records,
        num_workers=args.num_workers,
        batch_size=args.loader_batch_size,
        prefetch_factor=args.prefetch_factor,
    )
    validate_probs(probs, tag=f"teacher_raw[{args.split}]")

    # ===== TS (optional) =====
    T_star = 1.0
    if args.apply_ts:
        print(f"\n[TS] fit on split={args.val_split_for_ts}")
        if args.crop:
            ts_records = build_crop_records(data_root, args.val_split_for_ts)
        else:
            ts_records = build_folder_records(data_root, args.val_split_for_ts)
        ts_probs = ensemble_predict_records(
            runners=runners,
            records=ts_records,
            num_workers=args.num_workers,
            batch_size=args.loader_batch_size,
            prefetch_factor=args.prefetch_factor,
        )
        validate_probs(ts_probs, tag=f"teacher_raw[{args.val_split_for_ts}]")
        ts_gt = np.array([r["class_idx"] for r in ts_records], dtype=np.int64)
        T_star = fit_temperature(
            ts_probs, ts_gt,
            lo=args.ts_bounds[0], hi=args.ts_bounds[1],
        )
        print(f"[TS] T* = {T_star:.4f}")
        probs = apply_temperature(probs, T_star)
        validate_probs(probs, tag=f"teacher_ts[{args.split}]")

    # ===== confidence filter / weight =====
    gt_arr = np.array([r["class_idx"] for r in records], dtype=np.int64)
    top1_conf = probs.max(axis=1)
    drop_mask = np.zeros(len(records), dtype=bool)
    if args.min_conf > 0:
        drop_mask = top1_conf < args.min_conf
        drop_count = int(drop_mask.sum())
        drop_by_class = {
            CLASSES[c]: int(((gt_arr == c) & drop_mask).sum())
            for c in range(NUM_CLASSES)
        }
        print(f"[conf-filter] min_conf={args.min_conf}  drop={drop_count}/{len(records)}")
        print(f"  per-class drop: {drop_by_class}")
    else:
        drop_count = 0
        drop_by_class = {c: 0 for c in CLASSES}

    sample_weight = np.ones(len(records), dtype=np.float32)
    if args.conf_weight:
        sample_weight = top1_conf.astype(np.float32)
        print(f"[conf-weight] mean={sample_weight.mean():.4f} min={sample_weight.min():.4f}")

    keep_mask = ~drop_mask
    filenames_kept, classes_kept = [], []
    for i, r in enumerate(records):
        if keep_mask[i]:
            filenames_kept.append(Path(r["path"]).name)
            classes_kept.append(CLASSES[int(r["class_idx"])])
    probs_kept = probs[keep_mask]
    gt_kept = gt_arr[keep_mask]
    weight_kept = sample_weight[keep_mask]
    conf_kept = top1_conf[keep_mask]

    teacher_argmax = probs_kept.argmax(axis=1)
    mismatch = int((teacher_argmax != gt_kept).sum())
    eps = 1e-12
    entropy = float(-np.sum(
        probs_kept * np.log(np.clip(probs_kept, eps, 1.0)), axis=1,
    ).mean())
    print(f"[mismatch] teacher argmax vs hard gt: {mismatch}/{len(gt_kept)} "
          f"({mismatch/max(1,len(gt_kept))*100:.2f}%)")
    print(f"[entropy] mean={entropy:.4f}  (uniform={math.log(NUM_CLASSES):.4f})")

    # ===== 저장 =====
    np.savez(
        str(out_path),
        filenames=np.array(filenames_kept, dtype=str),
        classes=np.array(classes_kept, dtype=str),
        teacher_probs=probs_kept.astype(np.float32),
        class_idx_gt=gt_kept.astype(np.int64),
        sample_weight=weight_kept.astype(np.float32),
        teacher_conf=conf_kept.astype(np.float32),
        cls_names=np.array(CLASSES, dtype=str),
    )

    meta = {
        "ensemble_json": str(ensemble_json),
        "data_root": str(data_root),
        "split": args.split,
        "crop": bool(args.crop),
        "apply_ts": bool(args.apply_ts),
        "ts_star": float(T_star),
        "ts_bounds": list(args.ts_bounds) if args.apply_ts else None,
        "val_split_for_ts": args.val_split_for_ts if args.apply_ts else None,
        "min_conf": float(args.min_conf),
        "conf_weight": bool(args.conf_weight),
        "limit_samples": int(args.limit_samples),
        "classes": CLASSES,
        "n_total_records": len(records),
        "n_kept": int(keep_mask.sum()),
        "n_dropped": int(drop_count),
        "drop_by_class": drop_by_class,
        "mismatch_teacher_vs_hard": mismatch,
        "entropy_mean": entropy,
        "top1_conf_stats": {
            "mean": float(top1_conf.mean()),
            "min": float(top1_conf.min()),
            "max": float(top1_conf.max()),
            "median": float(np.median(top1_conf)),
        },
        "per_class_kept": {
            CLASSES[c]: int((gt_kept == c).sum()) for c in range(NUM_CLASSES)
        },
        "batch_config": {
            "loader_batch_size": args.loader_batch_size,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "tf_batch": args.tf_batch,
            "vit_batch": args.vit_batch,
            "siglip_batch": args.siglip_batch,
        },
        "total_elapsed_sec": round(time.time() - t0, 1),
    }
    meta_path = out_path.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] npz  → {out_path}")
    print(f"[DONE] meta → {meta_path}")
    print(f"[DONE] total elapsed = {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
