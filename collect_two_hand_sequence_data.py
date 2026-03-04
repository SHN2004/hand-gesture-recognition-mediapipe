#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect dynamic two-hand sequence samples into raw/<CLASS>.csv files."""

from __future__ import annotations

import argparse
import csv
import itertools
from collections import deque
from pathlib import Path

import cv2 as cv
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils import CvFpsCalc


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect dynamic two-hand sequence samples for one class key."
    )
    parser.add_argument("--class_key", type=str, required=True, help="Example: CUSTOM_HELLO")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="model/two_hand_sequence_classifier/raw",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Optional explicit output CSV path. Overrides --output_dir.",
    )
    parser.add_argument("--samples_target", type=int, default=120)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--num_hands", type=int, choices=[1, 2], default=2)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument(
        "--append_every_n_frames",
        type=int,
        default=1,
        help="Append one feature vector every N frames (>=1).",
    )
    parser.add_argument(
        "--include_presence_mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append left/right presence bits to each frame feature.",
    )
    parser.add_argument(
        "--auto_exit_on_target",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--model_asset_path",
        type=str,
        default="hand_landmarker.task",
    )
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    return parser.parse_args()


def _sanitize_class_key(class_key: str) -> str:
    cleaned = class_key.strip().upper().replace(" ", "_").replace("-", "_")
    if not cleaned:
        raise ValueError("--class_key cannot be empty.")
    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
    normalized = "".join(ch if ch in allowed else "_" for ch in cleaned)
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    normalized = normalized.strip("_")
    if not normalized:
        raise ValueError(f"Invalid class key: {class_key!r}")
    return normalized


def _resolve_output_csv(args: argparse.Namespace, class_key: str) -> Path:
    if args.output_csv:
        return Path(args.output_csv)
    return Path(args.output_dir) / f"{class_key}.csv"


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def calc_landmark_list(image: np.ndarray, landmarks: list) -> list[list[int]]:
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point: list[list[int]] = []

    landmark_list = landmarks if isinstance(landmarks, list) else landmarks.landmark
    for landmark in landmark_list:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point


def handedness_name(handedness) -> str | None:
    if handedness is None:
        return None
    if isinstance(handedness, list):
        if not handedness:
            return None
        return getattr(handedness[0], "category_name", None)
    try:
        return handedness.classification[0].label
    except Exception:
        return None


def order_hands(hands: list[dict]) -> list[dict]:
    """Return hands in stable order: Left then Right if available, else wrist x-order."""
    if not hands:
        return []

    with_side = []
    for hand in hands:
        side = handedness_name(hand.get("handedness"))
        side_norm = side.lower() if isinstance(side, str) else ""
        with_side.append((side_norm, hand))

    left = [hand for side, hand in with_side if side.startswith("left")]
    right = [hand for side, hand in with_side if side.startswith("right")]
    unknown = [
        hand
        for side, hand in with_side
        if not (side.startswith("left") or side.startswith("right"))
    ]

    ordered = []
    ordered.extend(left)
    ordered.extend(right)
    ordered.extend(unknown)

    if len(ordered) > 1 and not (left and right):
        ordered.sort(key=lambda hand: hand["landmark_list"][0][0])

    return ordered[:2]


def pre_process_landmark(landmark_list: list[list[int]]) -> list[float]:
    temp_landmark_list = [list(point) for point in landmark_list]

    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = landmark_point[0] - base_x
        temp_landmark_list[index][1] = landmark_point[1] - base_y

    flattened = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, flattened))) if flattened else 0
    if max_value == 0:
        return [0.0 for _ in flattened]
    return [value / max_value for value in flattened]


def pre_process_two_hand_landmarks(
    landmark_list_0: list[list[int]],
    landmark_list_1: list[list[int]],
) -> list[float]:
    temp_landmark_list = [list(point) for point in (landmark_list_0 + landmark_list_1)]

    base_x = (landmark_list_0[0][0] + landmark_list_1[0][0]) / 2.0
    base_y = (landmark_list_0[0][1] + landmark_list_1[0][1]) / 2.0

    for index, landmark_point in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] = landmark_point[0] - base_x
        temp_landmark_list[index][1] = landmark_point[1] - base_y

    flattened = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, flattened))) if flattened else 0
    if max_value == 0:
        return [0.0 for _ in flattened]
    return [value / max_value for value in flattened]


def calc_bounding_rect_from_points(landmark_points: list[list[int]]) -> list[int]:
    landmark_array = np.array(landmark_points, dtype=np.int32)
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]


def draw_bounding_rect(image: np.ndarray, rect: list[int], color: tuple[int, int, int]) -> np.ndarray:
    cv.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), color, 2)
    return image


def _wrist_xy(hand: dict | None) -> np.ndarray | None:
    if hand is None:
        return None
    return np.array(hand["landmark_list"][0], dtype=np.float32)


def _dist2(point_0: np.ndarray | None, point_1: np.ndarray | None) -> float:
    if point_0 is None or point_1 is None:
        return float("inf")
    delta = point_0 - point_1
    return float(np.dot(delta, delta))


def assign_hands_to_slots(
    hands: list[dict],
    prev_left_wrist: np.ndarray | None,
    prev_right_wrist: np.ndarray | None,
    frame_width: int,
) -> tuple[dict | None, dict | None]:
    """Assign detections to left/right slots with temporal stability."""
    hands = order_hands(hands)
    if not hands:
        return None, None

    if len(hands) == 1:
        hand = hands[0]
        side = handedness_name(hand.get("handedness"))
        side_norm = side.lower() if isinstance(side, str) else ""
        wrist = _wrist_xy(hand)

        if side_norm.startswith("left"):
            return hand, None
        if side_norm.startswith("right"):
            return None, hand

        if prev_left_wrist is not None and prev_right_wrist is not None:
            if _dist2(wrist, prev_left_wrist) <= _dist2(wrist, prev_right_wrist):
                return hand, None
            return None, hand
        if prev_left_wrist is not None:
            return hand, None
        if prev_right_wrist is not None:
            return None, hand
        if wrist is not None and wrist[0] > (frame_width / 2.0):
            return None, hand
        return hand, None

    hand_0, hand_1 = hands[0], hands[1]
    side_0 = handedness_name(hand_0.get("handedness"))
    side_1 = handedness_name(hand_1.get("handedness"))
    side_0_norm = side_0.lower() if isinstance(side_0, str) else ""
    side_1_norm = side_1.lower() if isinstance(side_1, str) else ""

    if side_0_norm.startswith("left") and side_1_norm.startswith("right"):
        return hand_0, hand_1
    if side_0_norm.startswith("right") and side_1_norm.startswith("left"):
        return hand_1, hand_0

    wrist_0 = _wrist_xy(hand_0)
    wrist_1 = _wrist_xy(hand_1)

    if prev_left_wrist is not None and prev_right_wrist is not None:
        direct_cost = _dist2(wrist_0, prev_left_wrist) + _dist2(wrist_1, prev_right_wrist)
        swapped_cost = _dist2(wrist_1, prev_left_wrist) + _dist2(wrist_0, prev_right_wrist)
        if direct_cost <= swapped_cost:
            return hand_0, hand_1
        return hand_1, hand_0

    if prev_left_wrist is not None:
        if _dist2(wrist_0, prev_left_wrist) <= _dist2(wrist_1, prev_left_wrist):
            return hand_0, hand_1
        return hand_1, hand_0

    if prev_right_wrist is not None:
        if _dist2(wrist_0, prev_right_wrist) <= _dist2(wrist_1, prev_right_wrist):
            return hand_1, hand_0
        return hand_0, hand_1

    if wrist_0 is not None and wrist_1 is not None and wrist_0[0] <= wrist_1[0]:
        return hand_0, hand_1
    return hand_1, hand_0


def frame_feature_vector(
    left_hand: dict | None,
    right_hand: dict | None,
    include_presence_mask: bool,
) -> list[float]:
    zero_hand = [0.0] * 42

    if left_hand is not None and right_hand is not None:
        feature_84 = pre_process_two_hand_landmarks(
            left_hand["landmark_list"],
            right_hand["landmark_list"],
        )
    else:
        left_feature = (
            pre_process_landmark(left_hand["landmark_list"]) if left_hand is not None else zero_hand
        )
        right_feature = (
            pre_process_landmark(right_hand["landmark_list"]) if right_hand is not None else zero_hand
        )
        feature_84 = left_feature + right_feature

    if include_presence_mask:
        return feature_84 + [
            1.0 if left_hand is not None else 0.0,
            1.0 if right_hand is not None else 0.0,
        ]
    return feature_84


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


def append_sample(csv_path: Path, feature_window: list[list[float]]) -> None:
    flattened = list(itertools.chain.from_iterable(feature_window))
    _ensure_parent_dir(csv_path)
    with csv_path.open("a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(flattened)


def draw_overlay(
    image: np.ndarray,
    fps: float,
    class_key: str,
    output_csv: Path,
    buffer_len: int,
    window: int,
    sample_count: int,
    samples_target: int,
    status_text: str,
    status_error: bool,
    hands_detected: int,
    left_present: bool,
    right_present: bool,
) -> np.ndarray:
    text_color = (255, 255, 255)
    error_color = (30, 30, 255)
    ok_color = (80, 220, 120)
    bg_color = (20, 20, 20)

    cv.rectangle(image, (10, 10), (900, 248), bg_color, -1)
    cv.rectangle(image, (10, 10), (900, 248), (90, 90, 90), 1)

    lines = [
        f"FPS: {fps:.1f}",
        f"Class key: {class_key}",
        f"Output: {output_csv}",
        f"Buffer: {buffer_len}/{window} ({(100.0 * buffer_len / max(window, 1)):.0f}%)",
        f"Hands detected: {hands_detected} | Left: {int(left_present)} Right: {int(right_present)}",
        f"Samples saved: {sample_count}/{samples_target if samples_target > 0 else 'inf'}",
        "Keys: [SPACE]=save current window, [c]=clear buffer, [q]/[ESC]=quit",
    ]

    y = 35
    for line in lines:
        cv.putText(
            image,
            line,
            (20, y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            text_color,
            1,
            cv.LINE_AA,
        )
        y += 30

    if status_text:
        cv.putText(
            image,
            status_text,
            (20, 238),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            error_color if status_error else ok_color,
            2,
            cv.LINE_AA,
        )
    return image


def main() -> None:
    args = get_args()
    class_key = _sanitize_class_key(args.class_key)
    output_csv = _resolve_output_csv(args, class_key)

    if args.window <= 0:
        raise ValueError("--window must be > 0")
    if args.append_every_n_frames <= 0:
        raise ValueError("--append_every_n_frames must be >= 1")
    if args.samples_target < 0:
        raise ValueError("--samples_target must be >= 0")

    dims = 84 + (2 if args.include_presence_mask else 0)
    sample_count = _count_samples(output_csv)

    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera device {args.device}")

    base_options = python.BaseOptions(model_asset_path=args.model_asset_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=args.num_hands,
        min_hand_detection_confidence=args.min_detection_confidence,
        min_hand_presence_confidence=args.min_tracking_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    hand_landmarker = vision.HandLandmarker.create_from_options(options)

    fps_calc = CvFpsCalc(buffer_len=10)
    feature_history: deque[list[float]] = deque(maxlen=args.window)
    prev_left_wrist = None
    prev_right_wrist = None
    frame_index = 0
    status_text = ""
    status_error = False

    try:
        while True:
            fps = fps_calc.get()

            key = cv.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("c"), ord("C")):
                feature_history.clear()
                status_text = "Buffer cleared."
                status_error = False
            if key == ord(" "):
                if len(feature_history) < args.window:
                    status_text = f"Buffer not full ({len(feature_history)}/{args.window}); sample not saved."
                    status_error = True
                else:
                    append_sample(output_csv, list(feature_history))
                    sample_count += 1
                    status_text = f"Saved sample {sample_count} for class {class_key}."
                    status_error = False
                    if (
                        args.auto_exit_on_target
                        and args.samples_target > 0
                        and sample_count >= args.samples_target
                    ):
                        break

            ret, image = cap.read()
            if not ret:
                break
            image = cv.flip(image, 1)
            debug_image = image.copy()

            image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            timestamp_ms = int(cap.get(cv.CAP_PROP_POS_MSEC))
            if timestamp_ms == 0:
                timestamp_ms = int(cv.getTickCount() / cv.getTickFrequency() * 1000)

            results = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

            hands: list[dict] = []
            if results.hand_landmarks:
                for index, hand_landmarks in enumerate(results.hand_landmarks):
                    handedness = results.handedness[index] if results.handedness else None
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                    hands.append(
                        {
                            "handedness": handedness,
                            "landmark_list": landmark_list,
                        }
                    )

            left_hand, right_hand = assign_hands_to_slots(
                hands,
                prev_left_wrist,
                prev_right_wrist,
                frame_width=debug_image.shape[1],
            )
            left_present = left_hand is not None
            right_present = right_hand is not None

            if left_present:
                prev_left_wrist = _wrist_xy(left_hand)
                left_rect = calc_bounding_rect_from_points(left_hand["landmark_list"])
                debug_image = draw_bounding_rect(debug_image, left_rect, (70, 170, 240))
                cv.putText(
                    debug_image,
                    "LEFT",
                    (left_rect[0], max(20, left_rect[1] - 8)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (70, 170, 240),
                    2,
                    cv.LINE_AA,
                )
            if right_present:
                prev_right_wrist = _wrist_xy(right_hand)
                right_rect = calc_bounding_rect_from_points(right_hand["landmark_list"])
                debug_image = draw_bounding_rect(debug_image, right_rect, (120, 220, 120))
                cv.putText(
                    debug_image,
                    "RIGHT",
                    (right_rect[0], max(20, right_rect[1] - 8)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (120, 220, 120),
                    2,
                    cv.LINE_AA,
                )

            feature = frame_feature_vector(
                left_hand=left_hand,
                right_hand=right_hand,
                include_presence_mask=args.include_presence_mask,
            )
            if len(feature) != dims:
                raise RuntimeError(f"Feature dimension mismatch: expected {dims}, got {len(feature)}")

            if frame_index % args.append_every_n_frames == 0:
                feature_history.append(feature)
            frame_index += 1

            debug_image = draw_overlay(
                image=debug_image,
                fps=fps,
                class_key=class_key,
                output_csv=output_csv,
                buffer_len=len(feature_history),
                window=args.window,
                sample_count=sample_count,
                samples_target=args.samples_target,
                status_text=status_text,
                status_error=status_error,
                hands_detected=len(hands),
                left_present=left_present,
                right_present=right_present,
            )

            cv.imshow("Two-Hand Sequence Data Collector", debug_image)
    finally:
        try:
            hand_landmarker.close()
        except Exception:
            pass
        cap.release()
        cv.destroyAllWindows()

    print(f"Collection ended. Class={class_key} | samples={sample_count} | output={output_csv}")


if __name__ == "__main__":
    main()
