#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the two-hand sequence classifier from per-class CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


DEFAULT_BASE_DIR = Path("model/two_hand_sequence_classifier")
DEFAULT_RAW_DIR = DEFAULT_BASE_DIR / "raw"
DEFAULT_LABEL_MAP_PATH = DEFAULT_BASE_DIR / "label_map.json"
DEFAULT_KERAS_PATH = DEFAULT_BASE_DIR / "sequence_classifier.keras"
DEFAULT_TFLITE_PATH = DEFAULT_BASE_DIR / "sequence_classifier.tflite"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrain sequence two-hand classifier from raw/<CLASS>.csv files."
    )
    parser.add_argument("--raw_dir", type=str, default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument(
        "--feature_dims",
        type=int,
        default=None,
        help="Per-frame feature dimensions. If omitted, inferred from CSV row width and --window.",
    )
    parser.add_argument("--label_map_path", type=str, default=str(DEFAULT_LABEL_MAP_PATH))
    parser.add_argument("--model_save_path", type=str, default=str(DEFAULT_KERAS_PATH))
    parser.add_argument("--tflite_save_path", type=str, default=str(DEFAULT_TFLITE_PATH))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lstm_units", type=int, default=64)
    parser.add_argument("--dense_units", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--min_samples_per_class", type=int, default=5)
    parser.add_argument("--quantize", action="store_true", default=True)
    parser.add_argument("--no-quantize", dest="quantize", action="store_false")
    return parser.parse_args()


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _parse_float_row(row: list[str], csv_path: Path, line_num: int) -> list[float] | None:
    if not row:
        return None
    stripped = [value.strip() for value in row]
    if not any(stripped):
        return None
    try:
        return [float(value) for value in stripped]
    except ValueError as error:
        raise ValueError(
            f"{csv_path}: line {line_num} contains non-numeric data. "
            "Per-class sequence rows must contain only flattened features."
        ) from error


def _load_per_class_rows(raw_dir: Path) -> tuple[list[str], dict[str, list[list[float]]], int]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    class_rows: dict[str, list[list[float]]] = {}
    flattened_width: int | None = None

    for csv_path in sorted(raw_dir.glob("*.csv")):
        class_key = csv_path.stem.strip()
        if not class_key:
            continue

        rows: list[list[float]] = []
        with csv_path.open(newline="") as file:
            reader = csv.reader(file)
            for line_num, row in enumerate(reader, start=1):
                parsed = _parse_float_row(row, csv_path, line_num)
                if parsed is None:
                    continue
                if flattened_width is None:
                    flattened_width = len(parsed)
                elif len(parsed) != flattened_width:
                    raise ValueError(
                        f"Row width mismatch in {csv_path} line {line_num}: "
                        f"expected {flattened_width}, got {len(parsed)}."
                    )
                rows.append(parsed)

        if rows:
            class_rows[class_key] = rows

    if not class_rows:
        raise ValueError(
            f"No valid samples found in {raw_dir}. "
            "Add samples with manage_two_hand_sequence_signs.py add-sign first."
        )
    if flattened_width is None or flattened_width <= 0:
        raise ValueError(f"Could not infer flattened row width from {raw_dir}.")

    classes = sorted(class_rows.keys())
    return classes, class_rows, flattened_width


def _infer_feature_dims(flattened_width: int, window: int, feature_dims: int | None) -> int:
    if window <= 0:
        raise ValueError("--window must be > 0.")

    if feature_dims is not None:
        expected = window * feature_dims
        if expected != flattened_width:
            raise ValueError(
                f"CSV width mismatch: window*feature_dims={expected}, csv_width={flattened_width}."
            )
        return feature_dims

    if flattened_width % window != 0:
        raise ValueError(
            f"Cannot infer feature dims: csv_width={flattened_width} not divisible by window={window}."
        )
    inferred = flattened_width // window
    if inferred <= 0:
        raise ValueError(f"Inferred invalid feature dims: {inferred}.")
    return inferred


def _build_dataset(
    classes: list[str],
    class_rows: dict[str, list[list[float]]],
    window: int,
    feature_dims: int,
) -> tuple[np.ndarray, np.ndarray]:
    samples: list[list[float]] = []
    labels: list[int] = []

    for idx, class_key in enumerate(classes):
        rows = class_rows[class_key]
        samples.extend(rows)
        labels.extend([idx] * len(rows))

    x = np.asarray(samples, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)
    x = x.reshape((-1, window, feature_dims))
    return x, y


def _write_label_map(classes: list[str], window: int, feature_dims: int, label_map_path: Path) -> None:
    label_map = {
        "classes": classes,
        "label_to_idx": {class_key: idx for idx, class_key in enumerate(classes)},
        "idx_to_label": {str(idx): class_key for idx, class_key in enumerate(classes)},
        "input_shape": [window, feature_dims],
    }
    _ensure_parent_dir(label_map_path)
    with label_map_path.open("w", encoding="utf-8") as file:
        json.dump(label_map, file, indent=2, ensure_ascii=True)


def _build_model(
    window: int,
    feature_dims: int,
    num_classes: int,
    lstm_units: int,
    dense_units: int,
    dropout: float,
) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(window, feature_dims)),
            # unroll=True avoids TensorList ops that require Flex delegate in TFLite
            tf.keras.layers.LSTM(lstm_units, unroll=True),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(dense_units, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _split_dataset(
    x: np.ndarray,
    y: np.ndarray,
    train_size: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < train_size < 1.0:
        raise ValueError("--train_size must be between 0 and 1.")
    if len(y) < 2:
        raise ValueError("Need at least 2 samples for train/test split.")

    class_counts = Counter(y.tolist())
    can_stratify = len(class_counts) > 1 and min(class_counts.values()) >= 2
    stratify = y if can_stratify else None

    return train_test_split(
        x,
        y,
        train_size=train_size,
        random_state=random_seed,
        stratify=stratify,
    )


def main() -> None:
    args = parse_args()

    raw_dir = Path(args.raw_dir)
    label_map_path = Path(args.label_map_path)
    model_save_path = Path(args.model_save_path)
    tflite_save_path = Path(args.tflite_save_path)

    classes, class_rows, flattened_width = _load_per_class_rows(raw_dir)
    feature_dims = _infer_feature_dims(flattened_width, args.window, args.feature_dims)

    if len(classes) < 2:
        raise ValueError(
            f"Need at least 2 classes to train a classifier. Found: {classes}."
        )

    low_sample_classes = [
        f"{class_key}({len(class_rows[class_key])})"
        for class_key in classes
        if len(class_rows[class_key]) < args.min_samples_per_class
    ]
    if low_sample_classes:
        raise ValueError(
            "Classes below minimum sample count "
            f"(--min_samples_per_class={args.min_samples_per_class}): "
            + ", ".join(low_sample_classes)
        )

    _write_label_map(classes, args.window, feature_dims, label_map_path)

    x_dataset, y_dataset = _build_dataset(classes, class_rows, args.window, feature_dims)
    x_train, x_test, y_train, y_test = _split_dataset(
        x_dataset, y_dataset, args.train_size, args.random_seed
    )

    model = _build_model(
        window=args.window,
        feature_dims=feature_dims,
        num_classes=len(classes),
        lstm_units=args.lstm_units,
        dense_units=args.dense_units,
        dropout=args.dropout,
    )

    _ensure_parent_dir(model_save_path)
    _ensure_parent_dir(tflite_save_path)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        str(model_save_path),
        verbose=1,
        save_weights_only=False,
        save_best_only=True,
        monitor="val_accuracy",
        mode="max",
    )
    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        patience=args.patience,
        monitor="val_accuracy",
        mode="max",
        restore_best_weights=True,
        verbose=1,
    )

    model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(x_test, y_test),
        callbacks=[checkpoint_cb, early_stop_cb],
    )

    best_model = tf.keras.models.load_model(model_save_path)
    test_loss, test_acc = best_model.evaluate(x_test, y_test, verbose=0)

    converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
    # Keep model portable for app runtime by exporting builtins-only ops.
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    if args.quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    try:
        tflite_model = converter.convert()
    except Exception as error:
        raise RuntimeError(
            "TFLite conversion failed without SELECT_TF_OPS. "
            "Keep fixed input shape and use the default LSTM(unroll=True) settings."
        ) from error

    with tflite_save_path.open("wb") as file:
        file.write(tflite_model)

    print("Retrain complete:")
    print(f"- Classes: {len(classes)}")
    for class_key in classes:
        print(f"  - {class_key}: {len(class_rows[class_key])} samples")
    print(f"- Input shape: ({args.window}, {feature_dims})")
    print(f"- Test accuracy: {test_acc:.4f} | loss: {test_loss:.4f}")
    print(f"- label_map: {label_map_path}")
    print(f"- keras model: {model_save_path}")
    print(f"- tflite model: {tflite_save_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Error: {error}")
        raise SystemExit(1) from error
