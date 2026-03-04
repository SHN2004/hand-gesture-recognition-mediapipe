#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a hand-sign keypoint classifier from a CSV dataset and export a TFLite model.

Works for:
- 1-hand signs: 21 landmarks * (x,y) = 42 features
- 2-hand signs: 2 * 21 landmarks * (x,y) = 84 features
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def _default_paths(num_hands: int):
    if num_hands == 2:
        base_dir = Path("model/two_hand_keypoint_classifier")
        return {
            "dataset": str(base_dir / "keypoint.csv"),
            "model_save_path": str(base_dir / "two_hand_keypoint_classifier.keras"),
            "tflite_save_path": str(base_dir / "two_hand_keypoint_classifier.tflite"),
        }
    base_dir = Path("model/keypoint_classifier")
    return {
        "dataset": str(base_dir / "keypoint.csv"),
        "model_save_path": str(base_dir / "keypoint_classifier.keras"),
        "tflite_save_path": str(base_dir / "keypoint_classifier.tflite"),
    }


def _infer_num_classes(y_dataset: np.ndarray) -> int:
    if y_dataset.size == 0:
        raise ValueError("Dataset is empty.")
    return int(np.max(y_dataset)) + 1


def _build_model(input_size: int, num_classes: int) -> tf.keras.Model:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((input_size,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(20, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 1-hand or 2-hand keypoint classifier and export TFLite.")
    parser.add_argument("--num_hands", type=int, choices=[1, 2], default=1)
    parser.add_argument("--dataset", type=str, default=None, help="CSV path (default depends on --num_hands).")
    parser.add_argument("--num_classes", type=int, default=None, help="If omitted, inferred from dataset labels.")

    parser.add_argument("--model_save_path", type=str, default=None, help="Keras model output path.")
    parser.add_argument("--tflite_save_path", type=str, default=None, help="TFLite output path.")

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--train_size", type=float, default=0.75)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--quantize", action="store_true", default=True, help="Enable DEFAULT TFLite optimizations.")
    parser.add_argument("--no-quantize", dest="quantize", action="store_false")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    defaults = _default_paths(args.num_hands)
    dataset = args.dataset or defaults["dataset"]
    model_save_path = args.model_save_path or defaults["model_save_path"]
    tflite_save_path = args.tflite_save_path or defaults["tflite_save_path"]

    input_size = 21 * 2 * args.num_hands

    if not os.path.isfile(dataset):
        raise FileNotFoundError(f"Dataset not found: {dataset}")

    X_dataset = np.loadtxt(
        dataset,
        delimiter=",",
        dtype="float32",
        usecols=list(range(1, input_size + 1)),
    )
    y_dataset = np.loadtxt(dataset, delimiter=",", dtype="int32", usecols=(0,))

    num_classes = args.num_classes if args.num_classes is not None else _infer_num_classes(y_dataset)

    X_train, X_test, y_train, y_test = train_test_split(
        X_dataset,
        y_dataset,
        train_size=args.train_size,
        random_state=args.random_seed,
        stratify=y_dataset if num_classes > 1 else None,
    )

    model = _build_model(input_size=input_size, num_classes=num_classes)
    model.summary()

    _ensure_parent_dir(model_save_path)
    _ensure_parent_dir(tflite_save_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        model_save_path,
        verbose=1,
        save_weights_only=False,
        save_best_only=True,
        monitor="val_accuracy",
        mode="max",
    )
    es_callback = tf.keras.callbacks.EarlyStopping(
        patience=args.patience,
        verbose=1,
        monitor="val_accuracy",
        mode="max",
    )

    model.fit(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_test, y_test),
        callbacks=[cp_callback, es_callback],
    )

    # Evaluate best-saved model
    best_model = tf.keras.models.load_model(model_save_path)
    loss, acc = best_model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}  loss: {loss:.4f}")

    converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
    if args.quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(tflite_save_path, "wb") as f:
        f.write(tflite_model)

    print(f"Saved Keras model:  {model_save_path}")
    print(f"Saved TFLite model: {tflite_save_path}")


if __name__ == "__main__":
    main()

