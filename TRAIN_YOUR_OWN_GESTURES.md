# Train Your Own Hand Gestures

This project recognizes two kinds of gestures:

- **Hand signs (static poses)** via `model/keypoint_classifier/*`
- **Finger gestures (motion/trajectories)** via `model/point_history_classifier/*`

Both are trained from CSV data you record with `app.py`.

- Hand-sign (static pose) models can be trained/exported with the provided notebooks or with `train_keypoint_classifier.py`.
- Finger-gesture (motion) models are trained/exported with the provided notebooks.

## 0) Decide your class IDs (important)

During data collection you assign a **class ID** by pressing a number key:

- `0`–`9` map to class IDs `0`–`9` (so you can train up to 10 classes without code changes).
- The app shows the current mode and class ID on-screen as `MODE:...` and `NUM:...`.

The label CSV files must match those IDs:

- `model/keypoint_classifier/keypoint_classifier_label.csv` (hand signs)
- `model/point_history_classifier/point_history_classifier_label.csv` (finger gestures)

Each line is the display name for its class ID, in order (line 1 = ID 0, line 2 = ID 1, etc).

## 1) Collect training data (hand signs)

1. Start the app: `python3 app.py`
2. Press `k` to enter **Logging Key Point** mode.
3. Press the number key `X` once.
4. The app shows a short countdown so you can get into position, then records samples continuously for ~20 seconds to:
   - `model/keypoint_classifier/keypoint.csv`
5. Repeat for every class ID you want to support.

You can adjust the timing via `app.py` flags: `--capture_countdown_seconds` and `--capture_duration_seconds`.

Tips:

- Collect **lots of variety** (distance to camera, slight rotations, left/right hand if you plan to use both, different lighting).
- Keep classes **balanced** (roughly similar number of samples per class).
- If you mess up, delete the bad rows from `model/keypoint_classifier/keypoint.csv` (it’s plain CSV).

## 2) Train the hand sign model

Option A (script, recommended):

1. Train + export:
   ```bash
   python3 train_keypoint_classifier.py --num_hands 1 --num_classes <N>
   ```
   If you omit `--num_classes`, the script infers it as `max(class_id)+1` from the CSV.
2. This produces:
   - `model/keypoint_classifier/keypoint_classifier.keras`
   - `model/keypoint_classifier/keypoint_classifier.tflite`
3. Update `model/keypoint_classifier/keypoint_classifier_label.csv` so its lines match your class order.

Option B (notebook):

1. Open `keypoint_classification.ipynb` (or the English version `keypoint_classification_EN.ipynb`) in Jupyter.
2. Set `NUM_HANDS = 1` and `NUM_CLASSES` to the number of hand sign classes you recorded.
3. Run the notebook top-to-bottom to export `model/keypoint_classifier/keypoint_classifier.tflite`.
4. Update `model/keypoint_classifier/keypoint_classifier_label.csv` so its lines match your class order.

### Pointer class note (affects finger-gesture training)

`app.py` only records point-history motion when the hand sign classifier returns ID `2`:

- `app.py` checks `if hand_sign_id == 2:  # Point gesture`

If you change your hand sign class ordering and “Pointer” is no longer class ID `2`, either:

- Keep “Pointer” as ID `2`, or
- Update that check in `app.py` to the correct ID for your “Pointer” class.

## 2b) Collect training data (two-hand signs)

Two-hand signs are logged as a **single feature vector that includes both hands** (so the model can learn the hand-to-hand relationship).

1. Start the app: `python3 app.py`
2. Press `b` to enter **Logging Two-Hand Key Point** mode.
3. Press the number key `X` once.
4. The app shows a short countdown, then records samples continuously for ~20 seconds to:
   - `model/two_hand_keypoint_classifier/keypoint.csv`

Notes:

- Samples are only written when **two hands are detected**.
- The app orders hands consistently (Left then Right when available) before building the two-hand feature vector.

## 2c) Train the two-hand sign model

Option A (script, recommended):

1. Train + export:
   ```bash
   python3 train_keypoint_classifier.py --num_hands 2 --num_classes <N>
   ```
   If you omit `--num_classes`, the script infers it as `max(class_id)+1` from the CSV.
2. This produces:
   - `model/two_hand_keypoint_classifier/two_hand_keypoint_classifier.keras`
   - `model/two_hand_keypoint_classifier/two_hand_keypoint_classifier.tflite`
3. Update `model/two_hand_keypoint_classifier/two_hand_keypoint_classifier_label.csv`.

Option B (notebook):

1. Open `keypoint_classification.ipynb` (or `keypoint_classification_EN.ipynb`) in Jupyter.
2. Set `NUM_HANDS = 2` and point `dataset` to `model/two_hand_keypoint_classifier/keypoint.csv`.
3. Train/export `model/two_hand_keypoint_classifier/two_hand_keypoint_classifier.tflite`.
4. Update `model/two_hand_keypoint_classifier/two_hand_keypoint_classifier_label.csv`.

## 3) Collect training data (finger gestures / motion)

Finger gestures are learned from the **index fingertip trajectory** over time.

1. Start the app: `python3 app.py`
2. Press `h` to enter **Logging Point History** mode.
3. Make sure the current hand sign is your **Pointer** pose (see note above).
4. Perform the motion for class ID `X` (e.g. swipe left), then press number key `X` while doing it to append samples to:
   - `model/point_history_classifier/point_history.csv`
5. Repeat for each finger gesture class.

Tip: if the app isn’t recognizing your pointer pose reliably, fix that first (more/better hand sign training data), otherwise your motion dataset will be mostly zeros.

## 4) Train the finger gesture model

1. Open `point_history_classification.ipynb` in Jupyter.
2. Set `NUM_CLASSES` to the number of finger gesture classes you recorded.
3. Run the notebook top-to-bottom to produce:
   - `model/point_history_classifier/point_history_classifier.tflite` (used by the app)
4. Update `model/point_history_classifier/point_history_classifier_label.csv` to match your class order.

## 5) Verify in the app

Run `python3 app.py` and confirm:

- The on-screen hand sign text matches your `keypoint_classifier_label.csv`.
- The on-screen finger gesture text matches your `point_history_classifier_label.csv`.

If outputs look “shifted” (wrong names but consistent), it’s almost always a **class-ID / label order mismatch** between:

- what you logged in the CSV,
- what `NUM_CLASSES` expects in the notebook,
- and the order of lines in the label CSV.
