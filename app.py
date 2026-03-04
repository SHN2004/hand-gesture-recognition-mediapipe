#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
import os
import time
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier
from two_hand_sequence_classifier import TwoHandSequenceClassifier


def safe_label(labels, index, fallback_prefix="Unknown"):
    if 0 <= index < len(labels):
        return labels[index]
    return f"{fallback_prefix}({index})"


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument(
        "--num_hands",
        help="Maximum number of hands to detect (1 or 2).",
        type=int,
        choices=[1, 2],
        default=2,
    )

    parser.add_argument(
        "--capture_countdown_seconds",
        help="Seconds to wait after pressing a class number before recording samples.",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--capture_duration_seconds",
        help="Seconds to record samples after the countdown.",
        type=float,
        default=20.0,
    )

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=float,
                        default=0.5)
    parser.add_argument(
        "--disable_sequence_classifier",
        action="store_true",
        help="Disable optional sequence two-hand classifier overlay.",
    )
    parser.add_argument(
        "--sequence_min_confidence",
        type=float,
        default=0.6,
        help="Minimum confidence threshold for sequence classifier predictions.",
    )
    parser.add_argument(
        "--sequence_infer_interval",
        type=int,
        default=2,
        help="Run sequence inference every N frames when the window buffer is full.",
    )

    args = parser.parse_args()

    return args


def main():
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    num_hands = args.num_hands
    capture_countdown_seconds = args.capture_countdown_seconds
    capture_duration_seconds = args.capture_duration_seconds
    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_tracking_confidence,
        min_tracking_confidence=min_tracking_confidence)
    hand_landmarker = vision.HandLandmarker.create_from_options(options)

    keypoint_classifier = KeyPointClassifier()

    # Optional: if you train a two-hand sign model, put it here and the app will
    # show a combined "Both:..." label when two hands are detected.
    two_hand_keypoint_classifier = None
    two_hand_keypoint_classifier_labels = None
    two_hand_model_path = 'model/two_hand_keypoint_classifier/two_hand_keypoint_classifier.tflite'
    two_hand_label_path = 'model/two_hand_keypoint_classifier/two_hand_keypoint_classifier_label.csv'
    if os.path.isfile(two_hand_model_path):
        try:
            two_hand_keypoint_classifier = KeyPointClassifier(
                model_path=two_hand_model_path
            )
            if os.path.isfile(two_hand_label_path):
                with open(two_hand_label_path, encoding='utf-8-sig') as f:
                    two_hand_keypoint_classifier_labels = csv.reader(f)
                    two_hand_keypoint_classifier_labels = [
                        row[0] for row in two_hand_keypoint_classifier_labels
                    ]
        except Exception:
            # Keep the demo running even if the optional model is incompatible.
            two_hand_keypoint_classifier = None
            two_hand_keypoint_classifier_labels = None

    sequence_classifier = None
    sequence_feature_history = None
    sequence_window = 30
    sequence_feature_dims = 86
    sequence_include_presence_mask = True
    sequence_prev_left_wrist = None
    sequence_prev_right_wrist = None
    sequence_frame_counter = 0
    sequence_class_key = None
    sequence_word = None
    sequence_confidence = 0.0
    sequence_model_path = 'model/two_hand_sequence_classifier/sequence_classifier.tflite'
    sequence_label_map_path = 'model/two_hand_sequence_classifier/label_map.json'
    sequence_class_to_word_path = 'model/two_hand_sequence_classifier/class_to_word.json'
    if (not args.disable_sequence_classifier
            and os.path.isfile(sequence_model_path)
            and os.path.isfile(sequence_label_map_path)):
        try:
            sequence_classifier = TwoHandSequenceClassifier(
                model_path=sequence_model_path,
                label_map_path=sequence_label_map_path,
                class_to_word_path=sequence_class_to_word_path,
                score_th=max(0.0, float(args.sequence_min_confidence)),
            )
            if sequence_classifier.expected_time_steps is not None:
                sequence_window = int(sequence_classifier.expected_time_steps)
            if sequence_classifier.expected_dims is not None:
                sequence_feature_dims = int(sequence_classifier.expected_dims)
            sequence_include_presence_mask = (sequence_feature_dims == 86)
            sequence_feature_history = deque(maxlen=sequence_window)
            print(
                "[sequence] loaded",
                sequence_model_path,
                f"input=({sequence_window},{sequence_feature_dims})",
            )
        except Exception:
            print(
                "[sequence] failed to load model. "
                "Re-export with train_two_hand_sequence_classifier.py (builtins-only TFLite)."
            )
            sequence_classifier = None
            sequence_feature_history = None
    if sequence_classifier is not None and sequence_feature_dims not in (84, 86):
        sequence_classifier = None
        sequence_feature_history = None

    point_history_classifier = PointHistoryClassifier()

    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = 0
    capture_session = None

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        prev_mode = mode
        number, mode = select_mode(key, mode)
        if mode != prev_mode:
            # Avoid mixing datasets if the user switches modes mid-capture.
            capture_session = None

        now = time.monotonic()
        if capture_session is None and (0 <= number <= 9) and mode in (1, 3):
            capture_session = {
                "mode": mode,
                "class_id": number,
                "phase": "countdown",
                "countdown_end": now + capture_countdown_seconds,
                "recording_end": now + capture_countdown_seconds + capture_duration_seconds,
                "frames_logged": 0,
            }

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Get timestamp in milliseconds
        timestamp_ms = int(cap.get(cv.CAP_PROP_POS_MSEC))
        if timestamp_ms == 0:
            timestamp_ms = int(cv.getTickCount() / cv.getTickFrequency() * 1000)

        results = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        #  ####################################################################
        hands = []
        if results.hand_landmarks:
            for idx, hand_landmarks in enumerate(results.hand_landmarks):
                handedness = results.handedness[idx] if results.handedness else None
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                hands.append({
                    "handedness": handedness,
                    "landmark_list": landmark_list,
                })

        hands = order_hands(hands)

        if sequence_classifier is not None and sequence_feature_history is not None:
            sequence_left_hand, sequence_right_hand = assign_hands_to_slots(
                hands,
                sequence_prev_left_wrist,
                sequence_prev_right_wrist,
                frame_width=debug_image.shape[1],
            )
            if sequence_left_hand is not None:
                sequence_prev_left_wrist = wrist_xy(sequence_left_hand)
            if sequence_right_hand is not None:
                sequence_prev_right_wrist = wrist_xy(sequence_right_hand)

            sequence_feature = frame_feature_vector(
                sequence_left_hand,
                sequence_right_hand,
                include_presence_mask=sequence_include_presence_mask,
            )
            if len(sequence_feature) == sequence_feature_dims:
                sequence_feature_history.append(sequence_feature)
                infer_interval = max(1, int(args.sequence_infer_interval))
                if (
                    len(sequence_feature_history) == sequence_window
                    and sequence_frame_counter % infer_interval == 0
                ):
                    try:
                        prediction = sequence_classifier.predict(
                            np.asarray(sequence_feature_history, dtype=np.float32)
                        )
                        sequence_class_key = prediction["class_key"]
                        sequence_word = prediction["word"]
                        sequence_confidence = float(prediction["confidence"])
                    except Exception:
                        sequence_class_key = None
                        sequence_word = None
                        sequence_confidence = 0.0
            sequence_frame_counter += 1

        # Per-hand sign classification (for display + legacy point-history logic)
        for h in hands:
            h["brect"] = calc_bounding_rect_from_points(h["landmark_list"])
            h["pre_processed_landmarks"] = pre_process_landmark(h["landmark_list"])
            h["hand_sign_id"] = keypoint_classifier(h["pre_processed_landmarks"])

        primary_hand = hands[0] if hands else None

        # Update point history using the primary hand (keeps legacy behavior stable)
        if primary_hand is not None:
            if primary_hand["hand_sign_id"] == 2:  # Point gesture
                point_history.append(primary_hand["landmark_list"][8])
            else:
                point_history.append([0, 0])
        else:
            point_history.append([0, 0])

        # Finger gesture classification (uses point_history)
        pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)
        finger_gesture_id = 0
        point_history_len = len(pre_processed_point_history_list)
        if point_history_len == (history_length * 2):
            finger_gesture_id = point_history_classifier(pre_processed_point_history_list)

        finger_gesture_history.append(finger_gesture_id)
        most_common_fg_id = Counter(finger_gesture_history).most_common()

        # Capture session state update / logging
        if capture_session is not None:
            if capture_session["phase"] == "countdown" and now >= capture_session["countdown_end"]:
                capture_session["phase"] = "recording"
            if capture_session["phase"] == "recording" and now >= capture_session["recording_end"]:
                capture_session = None

        # Dataset logging
        # - Mode 1 (single-hand) and mode 3 (two-hand) are captured continuously via capture_session.
        # - Mode 2 (point-history) keeps the legacy "press number to append one sample" behavior.
        if mode == 2 and (0 <= number <= 9):
            logging_csv_point_history(number, pre_processed_point_history_list)

        if capture_session is not None and capture_session["phase"] == "recording":
            if capture_session["mode"] == 1 and primary_hand is not None:
                logging_csv_single_hand(
                    capture_session["class_id"],
                    primary_hand["pre_processed_landmarks"],
                )
                capture_session["frames_logged"] += 1
            if capture_session["mode"] == 3 and len(hands) == 2:
                two_hand_landmarks = pre_process_two_hand_landmarks(
                    hands[0]["landmark_list"], hands[1]["landmark_list"]
                )
                logging_csv_two_hand(capture_session["class_id"], two_hand_landmarks)
                capture_session["frames_logged"] += 1

        # Combined two-hand sign classification (optional model)
        two_hand_sign_id = None
        if two_hand_keypoint_classifier is not None and len(hands) == 2:
            try:
                two_hand_landmarks = pre_process_two_hand_landmarks(
                    hands[0]["landmark_list"], hands[1]["landmark_list"]
                )
                two_hand_sign_id = two_hand_keypoint_classifier(two_hand_landmarks)
            except Exception:
                two_hand_sign_id = None

        # Drawing
        for idx, h in enumerate(hands):
            debug_image = draw_bounding_rect(use_brect, debug_image, h["brect"])
            debug_image = draw_landmarks(debug_image, h["landmark_list"])
            finger_text = ""
            if idx == 0 and most_common_fg_id:
                finger_text = safe_label(point_history_classifier_labels, most_common_fg_id[0][0], "FingerGesture")
            debug_image = draw_info_text(
                debug_image,
                h["brect"],
                h["handedness"],
                safe_label(keypoint_classifier_labels, h["hand_sign_id"], "HandSign"),
                finger_text,
            )

        if two_hand_sign_id is not None:
            all_points = []
            for h in hands:
                all_points.extend(h["landmark_list"])
            both_rect = calc_bounding_rect_from_points(all_points)
            both_label = safe_label(
                two_hand_keypoint_classifier_labels or [], two_hand_sign_id, "BothHands"
            )
            debug_image = draw_two_hand_info_text(debug_image, both_rect, both_label)

        if sequence_classifier is not None and sequence_feature_history is not None:
            debug_image = draw_sequence_info_text(
                debug_image,
                sequence_word,
                sequence_class_key,
                sequence_confidence,
                len(sequence_feature_history),
                sequence_window,
            )

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_capture_status(debug_image, capture_session, now, len(hands))
        debug_image = draw_info(debug_image, fps, mode, number, capture_session)

        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    if key == 98:  # b (both hands)
        mode = 3
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    # Handle both old API (landmarks.landmark) and new API (direct list)
    landmark_list = landmarks if isinstance(landmarks, list) else landmarks.landmark

    for _, landmark in enumerate(landmark_list):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Handle both old API (landmarks.landmark) and new API (direct list)
    landmark_list = landmarks if isinstance(landmarks, list) else landmarks.landmark

    # Keypoint
    for _, landmark in enumerate(landmark_list):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def _ensure_parent_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def logging_csv_single_hand(number, landmark_list):
    csv_path = 'model/keypoint_classifier/keypoint.csv'
    _ensure_parent_dir(csv_path)
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([number, *landmark_list])


def logging_csv_point_history(number, point_history_list):
    csv_path = 'model/point_history_classifier/point_history.csv'
    _ensure_parent_dir(csv_path)
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([number, *point_history_list])


def logging_csv_two_hand(number, landmark_list):
    csv_path = 'model/two_hand_keypoint_classifier/keypoint.csv'
    _ensure_parent_dir(csv_path)
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([number, *landmark_list])


def handedness_name(handedness):
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


def order_hands(hands):
    """Return hands in stable order: Left then Right if available, else left-to-right by wrist x."""
    if not hands:
        return []

    with_side = []
    for h in hands:
        side = handedness_name(h.get("handedness"))
        side_norm = side.lower() if isinstance(side, str) else ""
        with_side.append((side_norm, h))

    left = [h for side, h in with_side if side.startswith("left")]
    right = [h for side, h in with_side if side.startswith("right")]
    unknown = [h for side, h in with_side if not (side.startswith("left") or side.startswith("right"))]

    ordered = []
    ordered.extend(left)
    ordered.extend(right)
    ordered.extend(unknown)

    # If we don't have a clean Left+Right pair, fall back to x-order.
    if len(ordered) > 1 and not (left and right):
        ordered.sort(key=lambda h: h["landmark_list"][0][0])

    return ordered[:2]


def wrist_xy(hand):
    if hand is None:
        return None
    return np.array(hand["landmark_list"][0], dtype=np.float32)


def dist2(point_0, point_1):
    if point_0 is None or point_1 is None:
        return float("inf")
    delta = point_0 - point_1
    return float(np.dot(delta, delta))


def assign_hands_to_slots(hands, prev_left_wrist, prev_right_wrist, frame_width):
    """Assign detections to left/right slots with temporal consistency."""
    hands = order_hands(hands)
    if not hands:
        return None, None

    if len(hands) == 1:
        hand = hands[0]
        side = handedness_name(hand.get("handedness"))
        side_norm = side.lower() if isinstance(side, str) else ""
        wrist = wrist_xy(hand)

        if side_norm.startswith("left"):
            return hand, None
        if side_norm.startswith("right"):
            return None, hand

        if prev_left_wrist is not None and prev_right_wrist is not None:
            if dist2(wrist, prev_left_wrist) <= dist2(wrist, prev_right_wrist):
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

    wrist_0 = wrist_xy(hand_0)
    wrist_1 = wrist_xy(hand_1)

    if prev_left_wrist is not None and prev_right_wrist is not None:
        direct_cost = dist2(wrist_0, prev_left_wrist) + dist2(wrist_1, prev_right_wrist)
        swapped_cost = dist2(wrist_1, prev_left_wrist) + dist2(wrist_0, prev_right_wrist)
        if direct_cost <= swapped_cost:
            return hand_0, hand_1
        return hand_1, hand_0

    if prev_left_wrist is not None:
        if dist2(wrist_0, prev_left_wrist) <= dist2(wrist_1, prev_left_wrist):
            return hand_0, hand_1
        return hand_1, hand_0

    if prev_right_wrist is not None:
        if dist2(wrist_0, prev_right_wrist) <= dist2(wrist_1, prev_right_wrist):
            return hand_1, hand_0
        return hand_0, hand_1

    if wrist_0 is not None and wrist_1 is not None and wrist_0[0] <= wrist_1[0]:
        return hand_0, hand_1
    return hand_1, hand_0


def frame_feature_vector(left_hand, right_hand, include_presence_mask):
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


def calc_bounding_rect_from_points(landmark_points):
    landmark_array = np.array(landmark_points, dtype=np.int32)
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]


def pre_process_two_hand_landmarks(landmark_list_0, landmark_list_1):
    """Preprocess 2 hands together so relative hand-to-hand geometry is preserved."""
    temp_landmark_list = copy.deepcopy(landmark_list_0 + landmark_list_1)

    base_x = (landmark_list_0[0][0] + landmark_list_1[0][0]) / 2.0
    base_y = (landmark_list_0[0][1] + landmark_list_1[0][1]) / 2.0

    for index, landmark_point in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    max_value = max(list(map(abs, temp_landmark_list))) if temp_landmark_list else 0
    if max_value == 0:
        return [0.0 for _ in temp_landmark_list]

    return [n / max_value for n in temp_landmark_list]


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手首1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    # Handle both old API (handedness.classification) and new API (list)
    if handedness is None:
        info_text = "Unknown"
    elif isinstance(handedness, list):
        info_text = handedness[0].category_name if handedness else "Unknown"
    else:
        info_text = handedness.classification[0].label[0:]

    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_two_hand_info_text(image, brect, two_hand_sign_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = f"Both:{two_hand_sign_text}" if two_hand_sign_text else "Both:Unknown"
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image


def draw_sequence_info_text(
    image,
    sequence_word,
    sequence_class_key,
    sequence_confidence,
    sequence_buffer_len,
    sequence_window,
):
    if sequence_word:
        info_text = f"Seq2H:{sequence_word} ({sequence_confidence:.2f})"
        if sequence_class_key:
            info_text = info_text + f" [{sequence_class_key}]"
    else:
        info_text = "Seq2H: waiting..."

    buffer_text = f"SeqBuffer:{sequence_buffer_len}/{sequence_window}"
    cv.putText(image, info_text, (10, 190), cv.FONT_HERSHEY_SIMPLEX,
               0.65, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, info_text, (10, 190), cv.FONT_HERSHEY_SIMPLEX,
               0.65, (80, 220, 120), 2, cv.LINE_AA)
    cv.putText(image, buffer_text, (10, 214), cv.FONT_HERSHEY_SIMPLEX,
               0.6, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, buffer_text, (10, 214), cv.FONT_HERSHEY_SIMPLEX,
               0.6, (255, 255, 255), 2, cv.LINE_AA)
    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_capture_status(image, capture_session, now, hands_count):
    if capture_session is None:
        return image

    class_id = capture_session["class_id"]
    phase = capture_session["phase"]
    mode = capture_session["mode"]

    if phase == "countdown":
        remaining = max(0.0, capture_session["countdown_end"] - now)
        text_1 = f"Get ready: class {class_id}"
        text_2 = f"Recording starts in {remaining:.1f}s"
    else:
        remaining = max(0.0, capture_session["recording_end"] - now)
        text_1 = f"REC class {class_id} ({remaining:.1f}s left) samples:{capture_session['frames_logged']}"
        if mode == 3:
            ok = (hands_count == 2)
            text_2 = "Detecting: OK (2 hands)" if ok else f"Detecting: need 2 hands (now {hands_count})"
        else:
            ok = (hands_count >= 1)
            text_2 = "Detecting: OK (1+ hand)" if ok else "Detecting: need 1 hand"

    cv.putText(image, text_1, (10, 140), cv.FONT_HERSHEY_SIMPLEX,
               0.7, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, text_1, (10, 140), cv.FONT_HERSHEY_SIMPLEX,
               0.7, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(image, text_2, (10, 165), cv.FONT_HERSHEY_SIMPLEX,
               0.7, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, text_2, (10, 165), cv.FONT_HERSHEY_SIMPLEX,
               0.7, (255, 255, 255), 2, cv.LINE_AA)

    return image


def draw_info(image, fps, mode, number, capture_session=None):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History', 'Logging Two-Hand Key Point']
    if 1 <= mode <= 3:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        display_number = capture_session["class_id"] if capture_session is not None else number
        if 0 <= display_number <= 9:
            cv.putText(image, "NUM:" + str(display_number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
