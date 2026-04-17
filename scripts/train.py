"""emotion-project / train.py — 4-class emotion classification trainer.

===============================================================================
설계 문서 (DESIGN)
===============================================================================

좌표계·데이터 가정 (반드시 준수)
--------------------------------
- 데이터 루트: /workspace/user4/emotion-project/data_rot  (EXIF 정규화 완료 복사본)
- 이미지는 이미 PIL.Image.open() 기준 직립 (EXIF transpose 적용 완료).
  → tf.io.read_file + decode_jpeg 로도 같은 회전이 나온다 (data_rot 는
    transpose 된 픽셀을 그대로 JPEG 재저장했기 때문에 EXIF orientation=1).
- JSON 라벨의 bbox (annot_A.boxes: minX/minY/maxX/maxY) 도 EXIF 정규화 좌표계.
  → 추가 회전/스왑 금지. 그냥 이미지 픽셀 좌표로 바로 쓴다.
- seg mask shape == image shape 은 validate_data_rot.py 전수 통과 (쓰진 않음).
- bbox 이슈 (CLAUDE.md / validate_data_rot 결과):
    * 9건 음수 좌표 → [0,W] x [0,H] 클립.
    * 클립 후 area <= 0 (2건 예상) → drop.
    * 라벨 없는 이미지 2건 → crop 모드에서 skip (non-crop 모드는 폴더 기반이라 자동 포함).

함수 시그니처
--------------
- build_dataset(data_root, split, img_size, batch_size, *, crop=False,
                augment=False, model_family='resnet50', shuffle=True,
                limit_samples=None, seed=42) -> (tf.data.Dataset, num_samples, class_counts)
- get_augmentation() -> tf.keras.Sequential   (flip + small rotate + brightness/contrast)
- build_model(name, img_size, num_classes, dropout=0.3) -> (tf.keras.Model, preprocess_fn)
- compile_train(model, train_ds, val_ds, *, lr, epochs, ckpt_path, csv_path,
                patience) -> keras.callbacks.History
- evaluate(model, val_ds, class_names, out_prefix) -> dict
- save_artifacts(name, args, history, eval_metrics, model_path) -> None
- main()

데이터 파이프라인 (flow)
-------------------------
crop=False (E1, 기본):
  tf.keras.utils.image_dataset_from_directory
    └ data_rot/img/<split>/<class>/*.(jpg|jpeg|png)
    └ labels='inferred', label_mode='categorical', class_names=CLASSES (고정 순서)
  → map: preprocess_fn (ResNet/EfficientNet 각자 preprocess_input)
  → (optional) augment: data augmentation layer (train 만)
  → cache / prefetch

crop=True (E2~):
  JSON 라벨 전체 로드 (euc-kr → utf-8 fallback)
  → (filename, class_idx, bbox_xyxy) 튜플 리스트 생성
  → bbox clip & drop
  → tf.data.Dataset.from_tensor_slices(list)
  → map(load_image_and_crop): tf.io.read_file → decode → crop_to_bounding_box
    (bbox 는 이미지 픽셀 좌표 그대로 — EXIF 정규화 됐으므로)
  → resize(img_size,img_size) → preprocess_fn → one-hot
  → (optional) augment

지원 모델
---------
- custom_cnn        : 간단 CNN (비교용)
- vgg16             : VGG16 frozen, Dense head
- resnet50_frozen   : ResNet50 conv frozen, Dense head
- resnet50_ft       : ResNet50 + conv5 (layer name startswith 'conv5') unfreeze
- efficientnet      : EfficientNetB0 frozen, Dense head
- efficientnet_ft   : EfficientNetB0 + block6 (startswith 'block6') unfreeze

평가
----
- model.evaluate(val_ds) → val_loss, val_acc
- 전 val 샘플 예측 → sklearn classification_report + confusion_matrix
- results/<name>_confmat.npz (cm, labels, y_true, y_pred)
- results/<name>_report.txt (classification_report text)

산출물
------
- models/<name>.h5                              (best val_accuracy)
- logs/<name>.csv                               (CSVLogger per-epoch)
- logs/<name>.meta.json                         (args, history, eval, timing)
- experiments.md append                         (한 줄 요약)
- results/<name>_confmat.npz, _report.txt

CLI
---
--name --model --data --epochs --batch-size --img-size --lr --crop --augment
--patience --note --dropout --limit-train-samples --seed
===============================================================================
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# --- GPU 라이브러리 재exec (TF 2.21 이 nvidia-*-cu12 pip 패키지를 LD path 에 안 올리는 경우 대비) ---
# sys.prefix 기반으로 동적 탐색 (팀원 환경 호환). 환경변수로 override 가능.
#   EMOTION_CONDA_NVIDIA_LIB=/path/to/site-packages/nvidia
_PYLIB = os.environ.get(
    "EMOTION_CONDA_NVIDIA_LIB",
    str(Path(sys.prefix) / "lib" /
        f"python{sys.version_info.major}.{sys.version_info.minor}" /
        "site-packages" / "nvidia"),
)
if os.environ.get("EMOTION_TRAIN_RELAUNCHED") != "1" and os.path.isdir(_PYLIB):
    _parts = [f"{_PYLIB}/{d}/lib" for d in
              ("cudnn", "cuda_runtime", "cublas", "cuda_nvrtc", "cufft",
               "curand", "cusolver", "cusparse", "nccl", "nvjitlink", "cuda_cupti")]
    _ldp = ":".join(p for p in _parts if os.path.isdir(p))
    if _ldp:
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = _ldp + ":" + env.get("LD_LIBRARY_PATH", "")
        env["EMOTION_TRAIN_RELAUNCHED"] = "1"
        os.execve(sys.executable, [sys.executable, *sys.argv], env)

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, optimizers

# 프로젝트 루트: 환경변수 > __file__ 기반 자동 감지 (scripts/train.py → 상위 = emotion-project)
PROJECT = Path(os.environ.get("EMOTION_PROJECT_ROOT",
                              str(Path(__file__).resolve().parent.parent)))
DEFAULT_DATA = PROJECT / "data_rot"
CLASSES = ["anger", "happy", "panic", "sadness"]   # 고정. 알파벳 순.
NUM_CLASSES = len(CLASSES)
SPLITS = ("train", "val")


# ============================================================
# 라벨 JSON 로딩 + bbox 유효성 필터 (crop 모드에서만 사용)
# ============================================================
def load_labels_per_split(data_root: Path, split: str) -> list[dict]:
    """split 하위의 4개 class json 을 하나의 리스트로 concat.
    encoding: euc-kr 우선 → 실패 시 utf-8.  (validate_data_rot 에서 확인된 관례)
    반환 항목: {'filename','class_idx','bbox_xyxy','annot_A'} (원본 + class_idx 추가)
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
            box = (item.get("annot_A") or {}).get("boxes") if isinstance(item.get("annot_A"), dict) else None
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
    """라벨 + 이미지 파일 존재 확인 + bbox 클립/유효성 필터.
    bbox 는 EXIF 정규화 좌표계이므로 PIL.size == (W, H) 로 바로 클립한다.
    """
    from PIL import Image
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
        # clip to image bounds  ([0,W]x[0,H])
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
        })
    print(f"[crop/{split}] kept={len(good)}  dropped_area_le0={dropped_area}  dropped_missing={dropped_missing}")
    return good


# ============================================================
# Augmentation
# ============================================================
def get_augmentation(rot_deg: float = 10.0,
                     zoom: float = 0.0,
                     translation: float = 0.0,
                     color: float = 0.1,
                     flip: bool = True) -> tf.keras.Sequential:
    """얼굴 표정 데이터용 증강. 각 layer 는 해당 인자 > 0 일 때만 포함됨.

    인자
      rot_deg: ±rot_deg도 rotation. 0 이면 skip.
      zoom: RandomZoom height/width factor. 0 이면 skip.
      translation: RandomTranslation height/width factor. 0 이면 skip.
      color: RandomBrightness + RandomContrast factor. 0 이면 skip.
      flip: 좌우 flip 여부 (얼굴 좌우대칭 의미 있음).
    preprocess_input 전에 [0,255] float 상태에서 적용하는 것을 전제로 한다.
    """
    layer_list = []
    if flip:
        layer_list.append(layers.RandomFlip("horizontal"))
    if rot_deg and rot_deg > 0:
        layer_list.append(layers.RandomRotation(factor=rot_deg / 360.0))
    if zoom and zoom > 0:
        layer_list.append(layers.RandomZoom(height_factor=zoom, width_factor=zoom))
    if translation and translation > 0:
        layer_list.append(layers.RandomTranslation(
            height_factor=translation, width_factor=translation))
    if color and color > 0:
        layer_list.append(layers.RandomBrightness(
            factor=color, value_range=(0.0, 255.0)))
        layer_list.append(layers.RandomContrast(factor=color))
    if not layer_list:
        # augment 요청했는데 모든 값 0 인 경우 fallback: 최소 flip
        layer_list.append(layers.RandomFlip("horizontal"))
    return tf.keras.Sequential(layer_list, name="augment")


# ============================================================
# Dataset 빌드
# ============================================================
def _resolve_preprocess(model_family: str) -> Callable:
    from tensorflow.keras.applications import resnet50, vgg16, efficientnet
    if model_family.startswith("resnet50"):
        return resnet50.preprocess_input
    if model_family == "vgg16":
        return vgg16.preprocess_input
    if model_family.startswith("efficientnet"):
        return efficientnet.preprocess_input
    # custom_cnn: just rescale to [0,1]
    return lambda x: tf.cast(x, tf.float32) / 255.0


def build_dataset(
    data_root: Path,
    split: str,
    img_size: int,
    batch_size: int,
    *,
    crop: bool = False,
    augment: bool = False,
    aug_rot_deg: float = 10.0,
    aug_zoom: float = 0.0,
    aug_translation: float = 0.0,
    aug_color: float = 0.1,
    aug_flip: bool = True,
    model_family: str = "resnet50",
    shuffle: bool = True,
    limit_samples: Optional[int] = None,
    seed: int = 42,
):
    """데이터셋 빌드. (ds, n_samples, class_counts) 반환."""
    preprocess = _resolve_preprocess(model_family)
    aug_layer = (
        get_augmentation(rot_deg=aug_rot_deg, zoom=aug_zoom,
                         translation=aug_translation, color=aug_color, flip=aug_flip)
        if augment and split == "train" else None
    )
    autotune = tf.data.AUTOTUNE

    if not crop:
        # --- 폴더 기반 (E1 기본) ---
        dir_ = data_root / "img" / split
        ds = tf.keras.utils.image_dataset_from_directory(
            dir_,
            labels="inferred",
            label_mode="categorical",
            class_names=CLASSES,
            image_size=(img_size, img_size),
            batch_size=None,   # 먼저 샘플 단위로 받아서 limit 적용
            shuffle=shuffle,
            seed=seed,
        )
        # 클래스별 카운트 (파일 수 count)
        class_counts = {c: 0 for c in CLASSES}
        for c in CLASSES:
            d = dir_ / c
            if d.is_dir():
                class_counts[c] = sum(
                    1 for f in d.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
                )
        total = sum(class_counts.values())

        if limit_samples and limit_samples < total:
            ds = ds.take(limit_samples)
            total = limit_samples
            print(f"[warn] limit_samples={limit_samples} 적용됨. class_counts 는 디스크 전체 기준 "
                  f"참고값이며 실제 take 된 샘플 분포와 다를 수 있음 (smoke test 전용).")

        # 순서: decoded uint8 cache → shuffle(매 epoch 재shuffle) → augment → preprocess → batch.
        # cache 뒤 shuffle 이 없으면 image_dataset_from_directory 의 1회 셔플 결과가 memoize 돼
        # 모든 epoch 동일 순서로 돌아 수렴이 악화됨. shuffle 요청 시만 cache 뒤 shuffle 삽입.
        ds = ds.cache()
        if shuffle:
            ds = ds.shuffle(min(total, 2048), seed=seed,
                            reshuffle_each_iteration=True)
        if aug_layer is not None:
            ds = ds.map(lambda x, y: (aug_layer(x, training=True), y), num_parallel_calls=autotune)
        ds = ds.map(lambda x, y: (preprocess(tf.cast(x, tf.float32)), y),
                    num_parallel_calls=autotune)
        ds = ds.batch(batch_size).prefetch(autotune)
        return ds, total, class_counts

    # --- crop 모드 ---
    recs = build_crop_records(data_root, split)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(recs)
    if limit_samples and limit_samples < len(recs):
        recs = recs[:limit_samples]
    class_counts = {c: 0 for c in CLASSES}
    for r in recs:
        class_counts[CLASSES[r["class_idx"]]] += 1
    total = len(recs)

    paths = tf.constant([r["path"] for r in recs])
    labels = tf.constant([r["class_idx"] for r in recs], dtype=tf.int32)
    boxes = tf.constant([r["bbox_xyxy"] for r in recs], dtype=tf.float32)  # (N,4) xyxy in px
    whs = tf.constant([r["img_wh"] for r in recs], dtype=tf.float32)       # (N,2) (W,H)

    def _load_and_crop(path, label, box, wh):
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        # box: xyxy in pixels (EXIF 정규화 기준 == decoded image 기준)
        W = wh[0]; H = wh[1]
        # decode_image 실제 shape 로 crop. wh 는 PIL 값(참고용)이라 tf shape 우선.
        shape = tf.shape(img)
        Ht = tf.cast(shape[0], tf.float32)
        Wt = tf.cast(shape[1], tf.float32)
        x0 = tf.clip_by_value(box[0], 0.0, Wt)
        x1 = tf.clip_by_value(box[2], 0.0, Wt)
        y0 = tf.clip_by_value(box[1], 0.0, Ht)
        y1 = tf.clip_by_value(box[3], 0.0, Ht)
        w = tf.maximum(x1 - x0, 1.0)
        h = tf.maximum(y1 - y0, 1.0)
        img = tf.image.crop_to_bounding_box(
            img,
            tf.cast(y0, tf.int32), tf.cast(x0, tf.int32),
            tf.cast(h, tf.int32),  tf.cast(w, tf.int32),
        )
        img = tf.image.resize(img, (img_size, img_size), method="bilinear")
        one_hot = tf.one_hot(label, NUM_CLASSES)
        return img, one_hot

    ds = tf.data.Dataset.from_tensor_slices((paths, labels, boxes, whs))
    # 순서: load+crop → cache (I/O + crop 고비용 작업 memoize) → shuffle(매 epoch 재shuffle)
    #      → augment → preprocess → batch.
    # recs 는 load_labels_per_split 이 class 순서로 반환하므로 앞에서 numpy rng 으로 1회 셔플 완료.
    # cache 뒤 tf.shuffle 이 매 epoch 새 순서로 재배열.
    ds = ds.map(_load_and_crop, num_parallel_calls=autotune)
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(min(total, 2048), seed=seed,
                        reshuffle_each_iteration=True)
    if aug_layer is not None:
        ds = ds.map(lambda x, y: (aug_layer(x, training=True), y), num_parallel_calls=autotune)
    ds = ds.map(lambda x, y: (preprocess(tf.cast(x, tf.float32)), y), num_parallel_calls=autotune)
    ds = ds.batch(batch_size).prefetch(autotune)
    return ds, total, class_counts


# ============================================================
# 모델
# ============================================================
def build_model(name: str, img_size: int, num_classes: int = NUM_CLASSES,
                dropout: float = 0.3):
    """(model, model_family) 반환.  model_family 는 preprocess 선택용."""
    input_shape = (img_size, img_size, 3)
    inputs = tf.keras.Input(shape=input_shape)

    if name == "custom_cnn":
        x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
        x = layers.MaxPool2D()(x)
        x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
        x = layers.MaxPool2D()(x)
        x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(dropout)(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        return models.Model(inputs, outputs, name="custom_cnn"), "custom_cnn"

    if name == "vgg16":
        from tensorflow.keras.applications import VGG16
        base = VGG16(include_top=False, weights="imagenet", input_tensor=inputs, pooling="avg")
        base.trainable = False
        x = layers.Dropout(dropout)(base.output)
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        return models.Model(inputs, outputs, name="vgg16"), "vgg16"

    if name in ("resnet50_frozen", "resnet50_ft"):
        from tensorflow.keras.applications import ResNet50
        base = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs, pooling="avg")
        if name == "resnet50_frozen":
            base.trainable = False
        else:
            # conv5 block 만 학습 (layer 이름 "conv5_" prefix)
            base.trainable = True
            for lyr in base.layers:
                if not lyr.name.startswith("conv5"):
                    lyr.trainable = False
        x = layers.Dropout(dropout)(base.output)
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        return models.Model(inputs, outputs, name=name), "resnet50"

    if name in ("efficientnet", "efficientnet_ft"):
        from tensorflow.keras.applications import EfficientNetB0
        base = EfficientNetB0(include_top=False, weights="imagenet",
                              input_tensor=inputs, pooling="avg")
        if name == "efficientnet":
            base.trainable = False
        else:
            base.trainable = True
            for lyr in base.layers:
                if not lyr.name.startswith("block6"):
                    lyr.trainable = False
        x = layers.Dropout(dropout)(base.output)
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        return models.Model(inputs, outputs, name=name), "efficientnet"

    raise ValueError(f"unknown --model: {name}")


# ============================================================
# 학습 루프
# ============================================================
def compile_train(model, train_ds, val_ds, *, lr, epochs, ckpt_path, csv_path, patience,
                  label_smoothing: float = 0.0,
                  weight_decay: float = 0.0,
                  lr_schedule: str = "plateau",
                  warmup_epochs: int = 0,
                  monitor: str = "val_accuracy",
                  steps_per_epoch: int = 0,
                  cosine_alpha: float = 0.01,
                  plateau_factor: float = 0.5,
                  plateau_min_lr: float = 1e-6):
    """compile + fit. lr_schedule:
      - "plateau" : 고정 lr + ReduceLROnPlateau callback (기본).
      - "cosine"  : CosineDecay(initial_learning_rate=lr, decay_steps=total).
      - "cosine_warmup" : linear warmup → CosineDecay (논문 ViT 표준).
    weight_decay > 0 이면 Adam → AdamW 교체 (decoupled weight decay).
    monitor: val_accuracy (max) 또는 val_loss (min) 로 checkpoint/earlystopping 기준.
    """
    # --- lr schedule 결정 ---
    if lr_schedule in ("cosine", "cosine_warmup"):
        if steps_per_epoch <= 0:
            raise ValueError("cosine schedule 은 steps_per_epoch > 0 필요")
        total_steps = steps_per_epoch * epochs
        if lr_schedule == "cosine_warmup":
            warmup_steps = max(1, steps_per_epoch * max(0, warmup_epochs))
            use_lr = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=lr,
                decay_steps=max(1, total_steps - warmup_steps),
                warmup_target=lr,
                warmup_steps=warmup_steps,
                alpha=cosine_alpha,
            )
        else:
            use_lr = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=lr,
                decay_steps=total_steps,
                alpha=cosine_alpha,
            )
    elif lr_schedule == "plateau":
        use_lr = lr
    else:
        raise ValueError(f"unknown lr_schedule: {lr_schedule}")

    # --- optimizer 결정 ---
    if weight_decay and weight_decay > 0:
        opt = optimizers.AdamW(learning_rate=use_lr, weight_decay=weight_decay)
    else:
        opt = optimizers.Adam(learning_rate=use_lr)

    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=["accuracy"],
    )

    # --- monitor 방향 자동 결정 ---
    if monitor not in ("val_accuracy", "val_loss"):
        raise ValueError(f"monitor 는 val_accuracy|val_loss 여야 함: {monitor}")
    mode = "max" if monitor == "val_accuracy" else "min"

    cbs = [
        callbacks.ModelCheckpoint(
            str(ckpt_path),
            monitor=monitor, mode=mode,
            save_best_only=True, save_weights_only=False,
            verbose=1,
        ),
        callbacks.CSVLogger(str(csv_path)),
        callbacks.EarlyStopping(
            monitor=monitor, mode=mode,
            patience=patience, restore_best_weights=True, verbose=1,
        ),
    ]
    # ReduceLROnPlateau 는 plateau 일 때만 (cosine 은 optimizer 가 schedule 내장)
    if lr_schedule == "plateau":
        cbs.append(callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=plateau_factor,
            patience=max(1, patience // 2),
            min_lr=plateau_min_lr, verbose=1,
        ))

    return model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                     callbacks=cbs, verbose=2)


# ============================================================
# 평가
# ============================================================
def evaluate(model, val_ds, class_names, out_prefix: Path) -> dict:
    # overall metric
    loss, acc = model.evaluate(val_ds, verbose=0)

    # per-sample predictions: 한 번의 predict(val_ds) + label unbatch 수집
    # (이전: for xb,yb 마다 model.predict 는 graph 재구성 + ds 재읽기로 느림)
    y_pred_prob = model.predict(val_ds, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1).astype(np.int64)

    y_true = []
    for _, yb in val_ds.unbatch():
        y_true.append(int(np.argmax(yb.numpy())))
    y_true = np.array(y_true, dtype=np.int64)

    # 길이 불일치 방어 (이론상 없음, unbatch 순서는 val_ds 순서 그대로)
    if y_true.shape[0] != y_pred.shape[0]:
        m = min(y_true.shape[0], y_pred.shape[0])
        print(f"[warn] eval length mismatch y_true={y_true.shape[0]} y_pred={y_pred.shape[0]} → trim to {m}")
        y_true = y_true[:m]; y_pred = y_pred[:m]

    try:
        from sklearn.metrics import classification_report, confusion_matrix
        report_txt = classification_report(
            y_true, y_pred, labels=list(range(len(class_names))),
            target_names=class_names, digits=4, zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    except Exception as e:
        report_txt = f"(sklearn 불가: {e})"
        cm = np.zeros((len(class_names), len(class_names)), dtype=np.int64)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    (out_prefix.with_suffix(".txt")).write_text(report_txt, encoding="utf-8")
    np.savez(str(out_prefix.with_suffix(".npz")),
             cm=cm, labels=np.array(class_names), y_true=y_true, y_pred=y_pred)

    return {
        "val_loss": float(loss),
        "val_accuracy": float(acc),
        "n_val": int(y_true.shape[0]),
        "cm": cm.tolist(),
        "report": report_txt,
    }


# ============================================================
# 산출물 저장 + experiments.md append
# ============================================================
def save_artifacts(name, args, history, eval_metrics, model_path, timing):
    logs_dir = PROJECT / "logs"
    logs_dir.mkdir(exist_ok=True)

    meta = {
        "name": name,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "epochs_run": len(history.history.get("loss", [])),
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
        "eval": {k: v for k, v in eval_metrics.items() if k != "report"},
        "eval_report": eval_metrics.get("report", ""),
        "timing_sec": timing,
        "model_path": str(model_path),
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    with open(logs_dir / f"{name}.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # experiments.md append (한 줄)
    exp_md = PROJECT / "experiments.md"
    best_val_acc = max(history.history.get("val_accuracy", [0.0]) or [0.0])
    best_val_loss = min(history.history.get("val_loss", [float("inf")]) or [float("inf")])
    line = (
        f"| {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} | {name} | {args.model} | "
        f"bs={args.batch_size} img={args.img_size} lr={args.lr} crop={int(args.crop)} "
        f"aug={int(args.augment)} | epochs={meta['epochs_run']}/{args.epochs} | "
        f"val_acc={best_val_acc:.4f} | val_loss={best_val_loss:.4f} | "
        f"final_train_acc={history.history.get('accuracy', [0.0])[-1]:.4f} | "
        f"note={args.note or '-'} |\n"
    )
    # 헤더 없으면 붙이기
    header_needed = True
    if exp_md.is_file():
        try:
            with open(exp_md, encoding="utf-8") as f:
                head = f.read(200)
            if "| date |" in head or "| 날짜 |" in head:
                header_needed = False
        except Exception:
            pass
    mode = "a"
    with open(exp_md, mode, encoding="utf-8") as f:
        if header_needed:
            f.write(
                "| date | name | model | config | epochs | val_acc | val_loss | train_acc | note |\n"
                "|---|---|---|---|---|---|---|---|---|\n"
            )
        f.write(line)

    return meta, line


# ============================================================
# GPU memory growth (fallback: CPU)
# ============================================================
def configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception as e:
            print(f"[warn] memory_growth({g}): {e}")
    print(f"[i] GPUs visible to TF: {gpus}")
    if not gpus:
        print("[i] CPU mode (cuDNN 9 not installed in env — 학습은 CPU 로 진행).")


# ============================================================
# CLI
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser(description="emotion classification trainer")
    ap.add_argument("--name", required=True, help="실험 이름 (파일명 prefix)")
    ap.add_argument("--model", default="resnet50_frozen",
                    choices=["custom_cnn", "vgg16", "resnet50_frozen",
                             "resnet50_ft", "efficientnet", "efficientnet_ft"])
    ap.add_argument("--data", default=str(DEFAULT_DATA), help="data_rot 루트")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32,
                    help="default 32 (소규모 5996장엔 32~64가 적합, 128은 과도).")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--crop", action="store_true", help="bbox crop (annot_A) on")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--note", type=str, default="")
    ap.add_argument("--limit-train-samples", type=int, default=0,
                    help="smoke test 용. 0 이면 전부.")
    ap.add_argument("--limit-val-samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="CategoricalCrossentropy label_smoothing (e.g., 0.1).")
    # --- augmentation hyperparameters (--augment 와 함께 쓸 때만 효과) ---
    ap.add_argument("--aug-rot-deg", type=float, default=10.0,
                    help="RandomRotation 각도 (±deg). 0 이면 skip.")
    ap.add_argument("--aug-zoom", type=float, default=0.0,
                    help="RandomZoom height/width factor. 0 이면 skip. (strong 권장 0.1)")
    ap.add_argument("--aug-translation", type=float, default=0.0,
                    help="RandomTranslation height/width factor. 0 이면 skip. (strong 권장 0.05)")
    ap.add_argument("--aug-color", type=float, default=0.1,
                    help="RandomBrightness/Contrast factor. 0 이면 skip. (strong 권장 0.15)")
    ap.add_argument("--no-aug-flip", action="store_true",
                    help="좌우 flip 끔 (얼굴 데이터는 보통 on 권장).")
    # --- optimizer / lr schedule ---
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="AdamW weight decay (권장 1e-4). 0 이면 Adam 사용.")
    ap.add_argument("--lr-schedule", default="plateau",
                    choices=["plateau", "cosine", "cosine_warmup"],
                    help="plateau=ReduceLROnPlateau(기본), cosine=CosineDecay, "
                         "cosine_warmup=linear warmup + CosineDecay.")
    ap.add_argument("--warmup-epochs", type=int, default=0,
                    help="cosine_warmup 전용. linear warmup epoch 수 (예: 3).")
    ap.add_argument("--cosine-alpha", type=float, default=0.01,
                    help="CosineDecay alpha (final_lr / initial_lr 비율, 기본 0.01).")
    ap.add_argument("--plateau-factor", type=float, default=0.5,
                    help="ReduceLROnPlateau factor (기본 0.5).")
    ap.add_argument("--plateau-min-lr", type=float, default=1e-6,
                    help="ReduceLROnPlateau min_lr (기본 1e-6).")
    ap.add_argument("--monitor", default="val_accuracy",
                    choices=["val_accuracy", "val_loss"],
                    help="best checkpoint / early stopping 기준. overfit 우려 시 val_loss.")
    return ap.parse_args()


def main():
    args = parse_args()

    # seed (재현성)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    configure_gpu()

    data_root = Path(args.data)
    if not (data_root / "img" / "train").is_dir():
        raise SystemExit(f"[!] {data_root}/img/train 없음. --data 경로 확인")

    PROJECT.joinpath("models").mkdir(exist_ok=True)
    PROJECT.joinpath("logs").mkdir(exist_ok=True)
    PROJECT.joinpath("results").mkdir(exist_ok=True)

    print(f"[i] args: {vars(args)}")
    t0 = time.time()

    # 모델 먼저 (preprocess family 결정)
    model, model_family = build_model(
        args.model, args.img_size, NUM_CLASSES, args.dropout
    )
    print(f"[i] model={args.model}  family={model_family}  "
          f"params(total/trainable)={model.count_params()}/"
          f"{sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)}")

    # 데이터셋
    train_ds, n_train, counts_tr = build_dataset(
        data_root, "train", args.img_size, args.batch_size,
        crop=args.crop, augment=args.augment,
        aug_rot_deg=args.aug_rot_deg, aug_zoom=args.aug_zoom,
        aug_translation=args.aug_translation, aug_color=args.aug_color,
        aug_flip=(not args.no_aug_flip),
        model_family=model_family,
        shuffle=True,
        limit_samples=(args.limit_train_samples or None),
        seed=args.seed,
    )
    val_ds, n_val, counts_vl = build_dataset(
        data_root, "val", args.img_size, args.batch_size,
        crop=args.crop, augment=False,
        model_family=model_family,
        shuffle=False,
        limit_samples=(args.limit_val_samples or None),
        seed=args.seed,
    )
    print(f"[i] train n={n_train} {counts_tr}")
    print(f"[i] val   n={n_val} {counts_vl}")

    ckpt_path = PROJECT / "models" / f"{args.name}.h5"
    csv_path = PROJECT / "logs" / f"{args.name}.csv"

    # cosine schedule 용 steps_per_epoch (ceil)
    steps_per_epoch = (n_train + args.batch_size - 1) // args.batch_size

    hist = compile_train(
        model, train_ds, val_ds,
        lr=args.lr, epochs=args.epochs,
        ckpt_path=ckpt_path, csv_path=csv_path,
        patience=args.patience,
        label_smoothing=args.label_smoothing,
        weight_decay=args.weight_decay,
        lr_schedule=args.lr_schedule,
        warmup_epochs=args.warmup_epochs,
        monitor=args.monitor,
        steps_per_epoch=steps_per_epoch,
        cosine_alpha=args.cosine_alpha,
        plateau_factor=args.plateau_factor,
        plateau_min_lr=args.plateau_min_lr,
    )

    # best 체크포인트 강제 load → evaluate (EarlyStopping restore_best_weights 가
    # patience 미발동 + epochs 끝까지 가면 trigger 안 함 → last weights 로 평가되는 버그 방지).
    # compile=False 로 load 한 뒤 evaluate 용으로 재 compile (loss/metric 복원).
    if ckpt_path.is_file():
        try:
            model = tf.keras.models.load_model(str(ckpt_path), compile=False)
            model.compile(
                optimizer=optimizers.Adam(learning_rate=args.lr),
                loss=tf.keras.losses.CategoricalCrossentropy(
                    label_smoothing=args.label_smoothing),
                metrics=["accuracy"],
            )
            print(f"[i] evaluate: best ckpt 강제 load + re-compile → {ckpt_path}")
        except Exception as e:
            print(f"[warn] best ckpt load 실패 ({e}) — 현재 in-memory model 사용")
    eval_metrics = evaluate(
        model, val_ds, CLASSES,
        PROJECT / "results" / f"{args.name}_confmat",
    )

    timing = {"total_sec": round(time.time() - t0, 1)}
    meta, line = save_artifacts(args.name, args, hist, eval_metrics, ckpt_path, timing)

    print("\n" + "=" * 72)
    print(f"[DONE] {args.name}  epochs_run={meta['epochs_run']}/{args.epochs}  "
          f"val_acc={eval_metrics['val_accuracy']:.4f}  "
          f"val_loss={eval_metrics['val_loss']:.4f}  "
          f"time={timing['total_sec']}s  model={ckpt_path}")
    print("experiments.md line:")
    print(line.rstrip())


if __name__ == "__main__":
    main()
