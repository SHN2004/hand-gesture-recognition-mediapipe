# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hand gesture recognition system using MediaPipe for real-time webcam detection. Recognizes two types of gestures:
- **Hand signs** (static poses): Open, Close, Pointer, OK
- **Finger gestures** (dynamic movements): Stop, Clockwise, Counter-Clockwise, Move

## Commands

### Setup with uv (Recommended)
This project uses `uv` for Python package management.

First-time setup (if `pyproject.toml` doesn't exist):
```bash
uv init --no-readme
```

Install dependencies:
```bash
uv add mediapipe opencv-python tensorflow scikit-learn matplotlib
```

The hand landmarker model file (`hand_landmarker.task`) should already be in the project root. If missing, download it:
```bash
curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

### Running the Application
```bash
uv run python app.py
```

Options:
- `--device` - Camera device number (default: 0)
- `--width` / `--height` - Capture dimensions (default: 960x540)
- `--min_detection_confidence` / `--min_tracking_confidence` - Thresholds (default: 0.5)

### Training Models
Install Jupyter if needed: `uv add jupyter`

Open Jupyter notebooks and run all cells:
- `keypoint_classification.ipynb` - Hand sign recognition
- `point_history_classification.ipynb` - Finger gesture recognition

To add new gesture classes: modify `NUM_CLASSES` in the notebook and update the corresponding label CSV.

## Architecture

```
app.py                    # Main entry: webcam capture, MediaPipe detection, inference, visualization
├── model/
│   ├── keypoint_classifier/
│   │   ├── keypoint_classifier.py      # TFLite inference wrapper
│   │   ├── keypoint_classifier.tflite  # Pre-trained model (21 landmarks → class)
│   │   ├── keypoint.csv                # Training data
│   │   └── keypoint_classifier_label.csv
│   └── point_history_classifier/
│       ├── point_history_classifier.py # TFLite inference wrapper
│       ├── point_history_classifier.tflite # Pre-trained model (16-frame history → class)
│       ├── point_history.csv           # Training data
│       └── point_history_classifier_label.csv
└── utils/cvfpscalc.py    # FPS measurement utility
```

### Data Flow
1. MediaPipe detects 21 hand landmarks
2. **Keypoint classifier**: Landmarks normalized relative to wrist, flattened to 42 values → classifies hand sign
3. **Point history classifier**: Last 16 fingertip positions normalized → classifies finger gesture (uses temporal smoothing via most-common-in-window)

### Keyboard Controls (in app.py)
- `ESC` - Exit
- `n` - Normal inference mode
- `k` - Keypoint logging mode (collect hand sign training data)
- `h` - Point history logging mode (collect finger gesture training data)
- `0-9` - Select class ID when logging

## Dependencies

- mediapipe 0.10+ (uses new tasks API instead of deprecated solutions API)
- OpenCV 3.4.2+
- TensorFlow 2.3.0+ (tf-nightly 2.5.0.dev+ for LSTM models)
- scikit-learn 0.23.2+, matplotlib 3.3.2+ (for confusion matrix visualization)

## MediaPipe API Migration (0.8.x → 0.10+)

The original code was written for MediaPipe 0.8.x which used the `mp.solutions.hands` API. This has been migrated to MediaPipe 0.10+ which uses the new `tasks.python.vision` API.

### Key Changes Made

1. **Model Initialization** (app.py:64-73)
   - **Old API:** `mp.solutions.hands.Hands()`
   - **New API:** `vision.HandLandmarker.create_from_options()` with `vision.HandLandmarkerOptions`
   - Requires `hand_landmarker.task` model file (downloaded from Google's model repository)
   - Uses `RunningMode.VIDEO` for webcam processing

2. **Detection Processing** (app.py:123-137)
   - **Old API:** `hands.process(image)` returned `results.multi_hand_landmarks`
   - **New API:** `hand_landmarker.detect_for_video(mp_image, timestamp_ms)` returns `results.hand_landmarks`
   - Requires timestamp in milliseconds for video mode
   - Image must be wrapped in `mp.Image` object

3. **Landmark Structure**
   - **Old API:** `landmarks.landmark` (attribute access)
   - **New API:** Direct list of landmarks
   - Updated `calc_bounding_rect()` and `calc_landmark_list()` to handle both formats

4. **Handedness Detection** (app.py:513-517)
   - **Old API:** `handedness.classification[0].label`
   - **New API:** `handedness[0].category_name`
   - Updated `draw_info_text()` to handle both formats

### Required Files

- `hand_landmarker.task` - MediaPipe hand landmark detection model (7.6 MB)
  - Download: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`
  - Must be in project root directory

### Backward Compatibility

The updated code includes backward compatibility checks using `isinstance()` to handle both old and new API structures, though only the new API is actually used in practice.
