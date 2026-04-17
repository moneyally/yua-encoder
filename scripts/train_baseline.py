"""ResNet50 transfer baseline (감정 4분류).

사용:
  python scripts/train_baseline.py --name exp1_baseline --epochs 10

저장:
  models/<name>.h5        (best by val_acc)
  logs/<name>.csv         (per-epoch loss/acc/val_loss/val_acc)
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

PROJECT = Path(os.environ.get("EMOTION_PROJECT_ROOT",
                              str(Path(__file__).resolve().parent.parent)))
DEFAULT_DATA = PROJECT / "data" / "img"  # validate_data.py 에서 자동감지한 실제 루트
CLASSES = ["anger", "happy", "panic", "sadness"]


def build_datasets(data_root: Path, img_size: int, batch_size: int):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_root / "train",
        labels="inferred",
        label_mode="categorical",
        class_names=CLASSES,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=True,
        seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_root / "val",
        labels="inferred",
        label_mode="categorical",
        class_names=CLASSES,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=autotune).prefetch(autotune)
    val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=autotune).prefetch(autotune)
    return train_ds, val_ds


def build_model(img_size: int, num_classes: int, dropout: float):
    base = ResNet50(include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3), pooling="avg")
    base.trainable = False  # 1차에선 frozen transfer
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="resnet50_baseline")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="실험 이름 (e.g., exp1_baseline)")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    args = ap.parse_args()

    data_root = Path(args.data)
    if not (data_root / "train").is_dir():
        raise SystemExit(f"[!] {data_root}/train 없음. data 경로 확인")

    models_dir = PROJECT / "models"
    logs_dir = PROJECT / "logs"
    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    print(f"[i] GPUs: {gpus}")

    train_ds, val_ds = build_datasets(data_root, args.img_size, args.batch_size)
    model = build_model(args.img_size, len(CLASSES), args.dropout)
    model.compile(
        optimizer=optimizers.Adam(args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    ckpt_path = models_dir / f"{args.name}.h5"
    csv_path = logs_dir / f"{args.name}.csv"

    cbs = [
        callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_accuracy", mode="max", save_best_only=True, save_weights_only=False),
        callbacks.CSVLogger(str(csv_path)),
        callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=5, restore_best_weights=True),
    ]

    hist = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=cbs)

    # 실험 메타 저장
    meta = {
        "name": args.name,
        "epochs_run": len(hist.history.get("loss", [])),
        "best_val_acc": float(max(hist.history.get("val_accuracy", [0]))),
        "final_train_acc": float(hist.history.get("accuracy", [0])[-1]),
        "args": vars(args),
    }
    with open(logs_dir / f"{args.name}.meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[i] best_val_acc={meta['best_val_acc']:.4f}  saved={ckpt_path}")


if __name__ == "__main__":
    main()
