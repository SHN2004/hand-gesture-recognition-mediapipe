#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Management CLI for the sequence two-hand model family."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path("model/two_hand_sequence_classifier")
RAW_DIR = BASE_DIR / "raw"
CLASS_TO_WORD_PATH = BASE_DIR / "class_to_word.json"


def _normalize_word(word: str) -> str:
    return " ".join(word.strip().upper().split())


def _word_to_class_key(word: str) -> str:
    normalized = _normalize_word(word)
    raw = normalized.replace(" ", "_").replace("-", "_")
    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
    key = "".join(ch if ch in allowed else "_" for ch in raw)
    while "__" in key:
        key = key.replace("__", "_")
    key = key.strip("_")
    if not key:
        raise ValueError(f"Cannot build class key from word: {word!r}")
    if not key.startswith("CUSTOM_"):
        key = f"CUSTOM_{key}"
    return key


def _normalize_class_key(class_key: str) -> str:
    normalized = class_key.strip().upper().replace(" ", "_").replace("-", "_")
    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
    key = "".join(ch if ch in allowed else "_" for ch in normalized)
    while "__" in key:
        key = key.replace("__", "_")
    key = key.strip("_")
    if not key:
        raise ValueError(f"Invalid class key: {class_key!r}")
    return key


def _fallback_word_for_class_key(class_key: str) -> str:
    if class_key.startswith("CUSTOM_"):
        return class_key[len("CUSTOM_") :].replace("_", " ").strip() or class_key
    return class_key.replace("_", " ").strip() or class_key


def _read_json(path: Path, default):
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=True)


def _count_samples(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    count = 0
    with csv_path.open(newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue
            if any(cell.strip() for cell in row):
                count += 1
    return count


def _update_class_to_word(class_key: str, word: str | None) -> None:
    mapping = _read_json(CLASS_TO_WORD_PATH, {})
    if not isinstance(mapping, dict):
        mapping = {}
    normalized_word = _normalize_word(word) if word else _fallback_word_for_class_key(class_key)
    mapping[class_key] = normalized_word
    _write_json(CLASS_TO_WORD_PATH, mapping)


def _run_cmd(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from error


def cmd_add_sign(args: argparse.Namespace) -> None:
    if not args.word and not args.class_key:
        raise ValueError("add-sign requires --word or --class-key.")

    if args.class_key:
        class_key = _normalize_class_key(args.class_key)
    else:
        class_key = _word_to_class_key(args.word)

    word = _normalize_word(args.word) if args.word else _fallback_word_for_class_key(class_key)
    _update_class_to_word(class_key, word)

    collector_cmd = [
        sys.executable,
        "collect_two_hand_sequence_data.py",
        "--class_key",
        class_key,
        "--samples_target",
        str(args.samples_target),
        "--window",
        str(args.window),
        "--append_every_n_frames",
        str(args.append_every_n_frames),
        "--device",
        str(args.device),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--num_hands",
        str(args.num_hands),
        "--model_asset_path",
        args.model_asset_path,
        "--min_detection_confidence",
        str(args.min_detection_confidence),
        "--min_tracking_confidence",
        str(args.min_tracking_confidence),
    ]
    collector_cmd.append("--include_presence_mask" if args.include_presence_mask else "--no-include_presence_mask")
    collector_cmd.append("--auto_exit_on_target" if args.auto_exit_on_target else "--no-auto_exit_on_target")

    _run_cmd(collector_cmd)

    class_csv = RAW_DIR / f"{class_key}.csv"
    total = _count_samples(class_csv)
    print(f"add-sign complete: class_key={class_key} word={word!r} samples={total}")
    print("Next step: uv run python manage_two_hand_sequence_signs.py retrain")


def cmd_retrain(args: argparse.Namespace) -> None:
    retrain_cmd = [
        sys.executable,
        "train_two_hand_sequence_classifier.py",
        "--raw_dir",
        str(args.raw_dir),
        "--window",
        str(args.window),
        "--label_map_path",
        str(args.label_map_path),
        "--model_save_path",
        str(args.model_save_path),
        "--tflite_save_path",
        str(args.tflite_save_path),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--train_size",
        str(args.train_size),
        "--random_seed",
        str(args.random_seed),
        "--patience",
        str(args.patience),
        "--lstm_units",
        str(args.lstm_units),
        "--dense_units",
        str(args.dense_units),
        "--dropout",
        str(args.dropout),
        "--min_samples_per_class",
        str(args.min_samples_per_class),
    ]

    if args.feature_dims is not None:
        retrain_cmd.extend(["--feature_dims", str(args.feature_dims)])
    retrain_cmd.append("--quantize" if args.quantize else "--no-quantize")

    _run_cmd(retrain_cmd)


def cmd_list_signs(args: argparse.Namespace) -> None:
    raw_dir = Path(args.raw_dir)
    mapping = _read_json(Path(args.class_to_word_path), {})
    if not isinstance(mapping, dict):
        mapping = {}

    if not raw_dir.exists():
        print(f"No raw directory found: {raw_dir}")
        return

    class_files = sorted(raw_dir.glob("*.csv"))
    if not class_files:
        print(f"No sign CSVs found in {raw_dir}")
        return

    print("Sequence sign inventory:")
    for class_csv in class_files:
        class_key = class_csv.stem
        word = mapping.get(class_key) or _fallback_word_for_class_key(class_key)
        count = _count_samples(class_csv)
        print(f"- {class_key}: {word} ({count} samples)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage sequence two-hand signs (add-sign, retrain, list-signs)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_sign = subparsers.add_parser("add-sign", help="Create/update one class and collect samples.")
    add_sign.add_argument("--word", type=str, default=None, help='Display word, e.g. "thank you".')
    add_sign.add_argument("--class-key", type=str, default=None, help="Optional explicit class key.")
    add_sign.add_argument("--samples_target", type=int, default=120)
    add_sign.add_argument("--window", type=int, default=30)
    add_sign.add_argument("--append_every_n_frames", type=int, default=1)
    add_sign.add_argument("--include_presence_mask", action=argparse.BooleanOptionalAction, default=True)
    add_sign.add_argument("--auto_exit_on_target", action=argparse.BooleanOptionalAction, default=True)
    add_sign.add_argument("--device", type=int, default=0)
    add_sign.add_argument("--width", type=int, default=960)
    add_sign.add_argument("--height", type=int, default=540)
    add_sign.add_argument("--num_hands", type=int, choices=[1, 2], default=2)
    add_sign.add_argument("--model_asset_path", type=str, default="hand_landmarker.task")
    add_sign.add_argument("--min_detection_confidence", type=float, default=0.7)
    add_sign.add_argument("--min_tracking_confidence", type=float, default=0.5)
    add_sign.set_defaults(func=cmd_add_sign)

    retrain = subparsers.add_parser("retrain", help="Rebuild label map and retrain sequence model.")
    retrain.add_argument("--raw_dir", type=Path, default=RAW_DIR)
    retrain.add_argument("--window", type=int, default=30)
    retrain.add_argument("--feature_dims", type=int, default=None)
    retrain.add_argument("--label_map_path", type=Path, default=BASE_DIR / "label_map.json")
    retrain.add_argument("--model_save_path", type=Path, default=BASE_DIR / "sequence_classifier.keras")
    retrain.add_argument("--tflite_save_path", type=Path, default=BASE_DIR / "sequence_classifier.tflite")
    retrain.add_argument("--epochs", type=int, default=120)
    retrain.add_argument("--batch_size", type=int, default=64)
    retrain.add_argument("--train_size", type=float, default=0.8)
    retrain.add_argument("--random_seed", type=int, default=42)
    retrain.add_argument("--patience", type=int, default=20)
    retrain.add_argument("--lstm_units", type=int, default=64)
    retrain.add_argument("--dense_units", type=int, default=32)
    retrain.add_argument("--dropout", type=float, default=0.3)
    retrain.add_argument("--min_samples_per_class", type=int, default=5)
    retrain.add_argument("--quantize", action=argparse.BooleanOptionalAction, default=True)
    retrain.set_defaults(func=cmd_retrain)

    list_signs = subparsers.add_parser("list-signs", help="List available class CSVs and sample counts.")
    list_signs.add_argument("--raw_dir", type=str, default=str(RAW_DIR))
    list_signs.add_argument("--class_to_word_path", type=str, default=str(CLASS_TO_WORD_PATH))
    list_signs.set_defaults(func=cmd_list_signs)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
