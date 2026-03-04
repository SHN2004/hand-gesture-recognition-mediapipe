#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TFLite wrapper for sequence two-hand sign classification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf


DEFAULT_BASE_DIR = Path("model/two_hand_sequence_classifier")
DEFAULT_MODEL_PATH = DEFAULT_BASE_DIR / "sequence_classifier.tflite"
DEFAULT_LABEL_MAP_PATH = DEFAULT_BASE_DIR / "label_map.json"
DEFAULT_CLASS_TO_WORD_PATH = DEFAULT_BASE_DIR / "class_to_word.json"


class TwoHandSequenceClassifier:
    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        label_map_path: str | Path = DEFAULT_LABEL_MAP_PATH,
        class_to_word_path: str | Path = DEFAULT_CLASS_TO_WORD_PATH,
        score_th: float = 0.0,
        invalid_value: int = -1,
        num_threads: int = 1,
    ):
        self.model_path = Path(model_path)
        self.label_map_path = Path(label_map_path)
        self.class_to_word_path = Path(class_to_word_path)
        self.score_th = float(score_th)
        self.invalid_value = int(invalid_value)

        self.interpreter = tf.lite.Interpreter(
            model_path=str(self.model_path),
            num_threads=num_threads,
        )
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.label_map = self._load_json_dict(self.label_map_path)
        self.idx_to_label = self._parse_idx_to_label(self.label_map.get("idx_to_label", {}))
        self.label_to_idx = self._parse_label_to_idx(self.label_map.get("label_to_idx", {}))
        self.class_to_word = self._load_json_dict(self.class_to_word_path)

        self.input_shape = tuple(int(value) for value in self.input_details[0].get("shape", []))
        self.expected_time_steps = None
        self.expected_dims = None
        self.expected_flattened = None
        self._init_expected_shapes()
        self._apply_label_map_shape_hint()

    @staticmethod
    def _load_json_dict(path: Path) -> dict:
        if not path.exists():
            return {}
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _parse_idx_to_label(raw_map) -> dict[int, str]:
        idx_to_label = {}
        if not isinstance(raw_map, dict):
            return idx_to_label
        for key, value in raw_map.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
            idx_to_label[idx] = str(value)
        return idx_to_label

    @staticmethod
    def _parse_label_to_idx(raw_map) -> dict[str, int]:
        label_to_idx = {}
        if not isinstance(raw_map, dict):
            return label_to_idx
        for key, value in raw_map.items():
            try:
                idx = int(value)
            except (TypeError, ValueError):
                continue
            label_to_idx[str(key)] = idx
        return label_to_idx

    def _init_expected_shapes(self) -> None:
        if len(self.input_shape) >= 3:
            if self.input_shape[-2] > 0:
                self.expected_time_steps = int(self.input_shape[-2])
            if self.input_shape[-1] > 0:
                self.expected_dims = int(self.input_shape[-1])
        elif len(self.input_shape) == 2 and self.input_shape[-1] > 0:
            self.expected_flattened = int(self.input_shape[-1])

    def _apply_label_map_shape_hint(self) -> None:
        raw_shape = self.label_map.get("input_shape")
        if not isinstance(raw_shape, list) or len(raw_shape) != 2:
            return
        try:
            hint_time_steps = int(raw_shape[0])
            hint_dims = int(raw_shape[1])
        except (TypeError, ValueError):
            return

        if hint_time_steps > 0 and self.expected_time_steps in (None, hint_time_steps):
            self.expected_time_steps = hint_time_steps
        if hint_dims > 0 and self.expected_dims in (None, hint_dims):
            self.expected_dims = hint_dims

    def _prepare_input(self, sequence) -> np.ndarray:
        array = np.asarray(sequence, dtype=np.float32)

        if len(self.input_shape) >= 3:
            if array.ndim == 1:
                if self.expected_time_steps is None or self.expected_dims is None:
                    raise ValueError("Model input has dynamic shape; provide 2D sequence input.")
                expected_size = self.expected_time_steps * self.expected_dims
                if array.size != expected_size:
                    raise ValueError(f"Expected flattened size {expected_size}, got {array.size}.")
                array = array.reshape((1, self.expected_time_steps, self.expected_dims))
            elif array.ndim == 2:
                if (
                    self.expected_time_steps is not None
                    and self.expected_dims is not None
                    and array.shape != (self.expected_time_steps, self.expected_dims)
                ):
                    raise ValueError(
                        f"Expected sequence shape {(self.expected_time_steps, self.expected_dims)}, got {array.shape}."
                    )
                array = array[np.newaxis, ...]
            elif array.ndim == 3:
                if array.shape[0] != 1:
                    raise ValueError(f"Expected batch size 1, got {array.shape[0]}.")
            else:
                raise ValueError(f"Unsupported sequence rank {array.ndim}.")
            return array

        if len(self.input_shape) == 2:
            if array.ndim == 3:
                if array.shape[0] != 1:
                    raise ValueError(f"Expected batch size 1, got {array.shape[0]}.")
                array = array.reshape(-1)
            elif array.ndim == 2:
                array = array.reshape(-1)
            elif array.ndim != 1:
                raise ValueError(f"Unsupported sequence rank {array.ndim}.")

            if self.expected_flattened is not None and array.size != self.expected_flattened:
                raise ValueError(
                    f"Expected flattened size {self.expected_flattened}, got {array.size}."
                )
            return array[np.newaxis, ...]

        raise ValueError(f"Unsupported input tensor shape {self.input_shape}.")

    def __call__(self, sequence) -> tuple[int, float]:
        input_tensor = self._prepare_input(sequence)

        input_tensor_index = self.input_details[0]["index"]
        self.interpreter.set_tensor(input_tensor_index, input_tensor)
        self.interpreter.invoke()

        output_tensor_index = self.output_details[0]["index"]
        output = np.squeeze(self.interpreter.get_tensor(output_tensor_index))
        output = np.asarray(output, dtype=np.float32)

        if output.ndim != 1:
            raise ValueError(f"Expected 1D class probabilities, got shape {output.shape}.")

        class_id = int(np.argmax(output))
        confidence = float(output[class_id])

        if confidence < self.score_th:
            class_id = self.invalid_value

        return class_id, confidence

    def class_id_to_label(self, class_id: int) -> str:
        if class_id in self.idx_to_label:
            return self.idx_to_label[class_id]
        if class_id == self.invalid_value:
            return "UNKNOWN"
        return f"CLASS_{class_id}"

    def class_key_to_word(self, class_key: str) -> str:
        if class_key in self.class_to_word:
            return str(self.class_to_word[class_key])
        if class_key.startswith("CUSTOM_"):
            text = class_key[len("CUSTOM_") :].replace("_", " ").strip()
            return text if text else class_key
        return class_key.replace("_", " ").strip() or class_key

    def predict(self, sequence) -> dict:
        class_id, confidence = self(sequence)
        class_key = self.class_id_to_label(class_id)
        word = self.class_key_to_word(class_key)
        return {
            "class_id": class_id,
            "class_key": class_key,
            "word": word,
            "confidence": confidence,
        }
