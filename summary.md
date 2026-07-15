# Project Summary — Audio Event Detection (Edge-Oriented)

Last updated: 2026-07-15

## Progress

| Step | Status |
|------|--------|
| 1. Select ESC-50 classes | ✅ Done (8 classes) |
| 2. YAMNet embeddings + fine-tuned classifier head | ✅ Done (98.1% clip acc, 5-fold CV) |
| 2.5. Custom DS-CNN comparison | ✅ Done (87.5% clip acc, 24k params) |
| 3. Streaming inference with overlapping windows | ✅ Done (8/8 events, 0 FA, 1.5 s latency, 140× RT) |
| 4. TFLite export | ⬜ Not started |
| 5. INT8 quantization | ⬜ Not started |
| 6. Accuracy / latency / RAM / size / FP-rate comparison | ⬜ Not started |
| 7. macOS inference demo | ⬜ Not started |

## Step 1 — Class selection

8 ESC-50 classes chosen for a home/safety-monitoring edge device
(40 clips each, 320 clips total, 5 s per clip):

`car_horn`, `clock_alarm`, `crying_baby`, `dog`, `door_wood_knock`,
`glass_breaking`, `siren`, `vacuum_cleaner`

## Step 2 — YAMNet transfer learning

**Pipeline**: each clip loaded at 16 kHz mono → YAMNet (TF Hub) →
per-frame 1024-d embeddings (0.96 s window, 0.48 s hop, ~10 frames/clip,
3200 frames total) → dense head `1024 → Dense(256, relu) → Dropout(0.4) →
Dense(8)`, Adam 1e-3, early stopping on val loss. Clip prediction =
argmax of mean frame probabilities. YAMNet backbone is frozen (embeddings
precomputed once); only the head is trained.

**Evaluation**: standard ESC-50 5-fold cross-validation (train on 4 folds,
test on the held-out fold — folds are the dataset's official splits, so no
clip ever appears in both train and test).

### Results (5-fold CV)

| Fold | Frame acc | Clip acc |
|------|-----------|----------|
| 1 | 0.812 | 0.969 |
| 2 | 0.842 | 0.969 |
| 3 | 0.811 | 1.000 |
| 4 | 0.831 | 0.969 |
| 5 | 0.864 | 1.000 |
| **Mean** | **0.832** | **0.981 ± 0.017** |

### Per-class (clip-level, aggregated over all 5 test folds, n=40 each)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| clock_alarm | 1.000 | 1.000 | 1.000 |
| crying_baby | 1.000 | 1.000 | 1.000 |
| door_wood_knock | 1.000 | 1.000 | 1.000 |
| vacuum_cleaner | 1.000 | 1.000 | 1.000 |
| glass_breaking | 0.930 | 1.000 | 0.964 |
| car_horn | 0.951 | 0.975 | 0.963 |
| siren | 0.974 | 0.950 | 0.962 |
| dog | 1.000 | 0.925 | 0.961 |

Only 6 of 320 clips misclassified; the small confusions are dog/car_horn/
siren/glass_breaking mix-ups (see `plots/confusion_matrix.png`).

### Qualitative check on test samples

`src/plot_samples.py` plots 3 random fold-5 (test) clips — waveform plus the
per-frame posterior heatmap (`plots/test_samples_posteriors.png`). All 3
correct: vacuum_cleaner (mean p=0.68), car_horn (p=0.39), glass_breaking
(p=0.52). The heatmaps show the frame-vs-clip gap concretely: short events
(car horn ~1.5 s, glass break ~1 s) dominate the posteriors only in the
frames that contain them; trailing silence/reverb frames drift toward other
classes (e.g. glass_breaking at low confidence), so clip-level averaging —
and later, streaming smoothing — is what makes the decision robust.

**Notes**
- Frame-level accuracy (83.2%) is much lower than clip-level (98.1%):
  single 0.96 s windows are often ambiguous (silence/background between
  events), but averaging ~10 frames per clip is very robust. This gap will
  matter for step 3 (streaming), where decisions are made per-window.
- Saved artifacts: `results/classifier_head.keras` (head trained on folds
  1–4, fold 5 held out — the model to carry into TFLite export),
  `results/fold_metrics.csv`, `results/classification_report.txt`,
  `data/yamnet_embeddings.npz` (cached embeddings, 2 MB).

## Step 2.5 — Custom DS-CNN vs YAMNet transfer learning

**DS-CNN**: log-mel features (64 mels, 25 ms window, 10 ms hop) cut into
0.96 s patches with 0.48 s hop — the same framing YAMNet uses, so numbers are
directly comparable. Architecture (keyword-spotting style): Conv2D 64 (10×4,
stride 2×2) → 4× [DepthwiseConv 3×3 → Pointwise 1×1, BN+ReLU] → GAP →
Dropout 0.3 → Dense(8). Trained from scratch per fold, same 5-fold CV,
patches inherit clip labels, clip = mean of patch probabilities.

### Model comparison (ordered by clip accuracy)

| Model | Clip acc (5-fold CV) | Frame acc | Params | ms/frame (CPU) |
|-------|---------------------|-----------|--------|----------------|
| YAMNet (frozen) + dense head | **0.981 ± 0.017** | 0.832 | 4,014,456 | 14.8 |
| DS-CNN (custom, from scratch) | 0.875 ± 0.043 | 0.768 | **24,072** | 13.8 |

DS-CNN per-fold: 0.828 / 0.891 / 0.875 / 0.938 / 0.844 (higher variance than
YAMNet's 0.969–1.000). Weakest class is car_horn (F1 0.686, recall 0.60);
glass_breaking over-triggers (precision 0.741). See
`plots/dscnn_confusion_matrix.png`, `results/dscnn_classification_report.txt`,
`results/model_comparison.csv`.

**Takeaways**
- Transfer learning is worth ~10.6 points of clip accuracy at this data size
  (256 training clips/fold): 98.1% vs 87.5%. AudioSet pretraining is doing a
  lot of work.
- The DS-CNN is **167× smaller** (24k vs 4M params) — the real edge trade-off.
- The ms/frame numbers are nearly equal only because Keras `predict()` call
  overhead (~10 ms) dominates both at batch 1; the true compute gap will only
  show up after TFLite export (steps 4–6), where the proper latency/RAM
  comparison belongs.

## Step 3 — Streaming inference (overlapping windows)

`src/streaming_inference.py` — `StreamingDetector` consumes arbitrary-size
audio chunks (mic-callback style, 1024 samples in the simulation), keeps a
ring buffer, and every **0.48 s hop** runs the newest **0.96 s window**
through YAMNet + the trained head. Decision layer on top of the per-hop
posteriors:

- moving average over **K=3** hops (~1.9 s context)
- fire when smoothed p > **θ=0.5** for **M=2** consecutive hops
- per-class refractory **3.0 s**

**Simulation**: one random fold-5 clip per class (held out from the head)
concatenated with 1 s silence gaps → 49 s stream.

| Metric | Value |
|--------|-------|
| Hits | **8/8** |
| False alarms | **0** |
| Duplicate re-fires (same clip after refractory) | 5 |
| Detection latency from clip onset | mean 1.46 s, worst 1.88 s |
| Per-hop inference (YAMNet + head, CPU) | median 2.5 ms (budget 480 ms) |
| Real-time factor | 0.007 (≈140× faster than real time) |

Timeline: `plots/streaming_timeline.png` — posterior mass sits inside the
true spans; duplicates are re-fires on long continuous sounds (alarm,
vacuum, siren) after the 3 s refractory expires, arguably "event still
ongoing" rather than errors. Latency matches theory:
window (0.96) + M·hop ≈ 1.4–1.9 s; lower θ/M trades latency for FP risk
(step 6 sweeps this). Note the 2.5 ms/hop here vs 14.8 ms in step 2.5's
table — the difference is Keras `predict()` overhead vs direct `model()`
calls; the streaming path uses the direct call.

## Environment notes

- Python 3.12 venv (`python3.12 -m venv .venv`) — TensorFlow does not yet
  support the system Python 3.14.
- `setuptools<81` is pinned: tensorflow_hub still imports `pkg_resources`,
  which setuptools ≥81 removed.
- Embedding extraction for all 320 clips takes ~11 s on this MacBook;
  head training is ~1 s per fold.
