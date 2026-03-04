# Sequence Two-Hand Runbook

This project is now set up for **sequence two-hand signs only**.

Use this file as the quick reference to collect data, train, and run inference later.

## 1) One-Time Setup

From repo root:

```bash
uv sync
```

If you ever need new dependencies:

```bash
uv add <package>
```

## 2) Project Paths Used by This Flow

All sequence-two-hand assets live here:

- `model/two_hand_sequence_classifier/raw/<CLASS>.csv` (training samples per class)
- `model/two_hand_sequence_classifier/class_to_word.json` (class key -> display text)
- `model/two_hand_sequence_classifier/label_map.json` (generated on retrain)
- `model/two_hand_sequence_classifier/sequence_classifier.keras` (generated on retrain)
- `model/two_hand_sequence_classifier/sequence_classifier.tflite` (generated on retrain)

## 3) Add a New Sign (Data Collection)

### Recommended command

```bash
uv run python manage_two_hand_sequence_signs.py add-sign --word "hello" --samples_target 120
```

What this does:

1. Creates class key `CUSTOM_HELLO`
2. Updates `class_to_word.json`
3. Opens webcam collector and appends samples to:
   - `model/two_hand_sequence_classifier/raw/CUSTOM_HELLO.csv`

### Collector controls

- `SPACE`: save current sequence window
- `C`: clear current buffer
- `Q` or `ESC`: quit

Notes:

- Save when the motion is fully visible in the recent window.
- Keep left/right usage consistent while recording.

### Optional: explicit class key

```bash
uv run python manage_two_hand_sequence_signs.py add-sign --class-key CUSTOM_THANK_YOU --word "thank you" --samples_target 150
```

## 4) List Collected Signs

```bash
uv run python manage_two_hand_sequence_signs.py list-signs
```

Shows class keys, resolved words, and sample counts.

## 5) Retrain the Sequence Model

```bash
uv run python manage_two_hand_sequence_signs.py retrain
```

This rebuilds:

1. `label_map.json` from filenames in `raw/*.csv`
2. LSTM model weights
3. `sequence_classifier.keras`
4. `sequence_classifier.tflite`

Important constraints:

- You need at least **2 classes** to train a classifier.
- Default minimum per class is **5** samples (`--min_samples_per_class`).
- Do not mix old merged CSV formats into this flow.

## 6) Run Inference in App

```bash
uv run python app.py
```

The app auto-loads sequence classifier if these files exist:

- `model/two_hand_sequence_classifier/sequence_classifier.tflite`
- `model/two_hand_sequence_classifier/label_map.json`

### Useful app flags

```bash
uv run python app.py --sequence_min_confidence 0.7 --sequence_infer_interval 2
```

- `--sequence_min_confidence`: confidence threshold for accepted predictions
- `--sequence_infer_interval`: run sequence inference every N frames when buffer is full
- `--disable_sequence_classifier`: disable sequence overlay

## 7) Typical Workflow

1. Add first sign:
   ```bash
   uv run python manage_two_hand_sequence_signs.py add-sign --word "hello" --samples_target 120
   ```
2. Add second sign:
   ```bash
   uv run python manage_two_hand_sequence_signs.py add-sign --word "thank you" --samples_target 120
   ```
3. Retrain:
   ```bash
   uv run python manage_two_hand_sequence_signs.py retrain
   ```
4. Test in app:
   ```bash
   uv run python app.py
   ```

## 8) Troubleshooting

- `Raw directory not found`:
  - Run `add-sign` at least once to create samples.
- Predictions always `waiting`:
  - Model or label map not generated yet. Run `retrain`.
- Predictions unstable:
  - Collect cleaner, balanced data per class.
  - Increase `--sequence_min_confidence`.
- Shape mismatch errors after changing window/features:
  - Retrain again so `label_map.json` and model input shape match.

  uv run python app.py --sequence_min_confidence 0.35 --sequence_infer_interval 1

