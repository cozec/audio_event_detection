# Audio Event Detection (Edge-Oriented)

Edge-oriented audio event detection built on **YAMNet + ESC-50**. Roadmap
(from `guild_from_chatgpt.md`): select classes → fine-tune a YAMNet
classifier head → compare with a small custom DS-CNN → streaming inference →
TFLite export → INT8 quantization → benchmarks → macOS demo.

## Current status

- **Step 1 — class selection**: 8 ESC-50 classes relevant to a home/safety
  monitoring edge device: `dog`, `siren`, `car_horn`, `glass_breaking`,
  `crying_baby`, `door_wood_knock`, `clock_alarm`, `vacuum_cleaner`
  (40 clips each, 320 clips total).
- **Step 2 — YAMNet transfer learning**: per-frame 1024-d YAMNet embeddings
  (0.96 s window, 0.48 s hop) + a small dense head (1024 → 256 → 8),
  evaluated with the standard ESC-50 5-fold cross-validation. Clip
  predictions average the frame probabilities. **98.1% ± 1.7** clip accuracy.
- **Step 2.5 — custom DS-CNN baseline**: keyword-spotting-style
  depthwise-separable CNN (24k params) trained from scratch on log-mel
  patches with the same framing and CV protocol: **87.5% ± 4.3** clip
  accuracy — transfer learning buys ~10.6 points at 167× the parameter count.
- **Step 3 — streaming inference**: chunk-fed detector (0.96 s window /
  0.48 s hop, K-hop smoothing, threshold + refractory). On a simulated 49 s
  stream of held-out clips: **8/8 detected, 0 false alarms**, mean onset
  latency 1.46 s, 140× faster than real time on CPU.
- **Steps 4–6 — TFLite + INT8 + edge comparison**: all models exported to
  TFLite; head and DS-CNN fully INT8-quantized (3.9× / 2.1× smaller, head
  loses zero accuracy). Full pipeline comparison: YAMNet+head = 100% fold-5
  accuracy at 15–16 MB / ~32 MB RAM; DS-CNN int8 = 77% at **48 KB / ~0 RAM**.
  Open-set false positives are the main weakness (3–19 FP/min on unknown
  sounds depending on threshold) — an explicit background class is the
  next-step fix.
- **Step 7 — macOS live demo**: `mic_demo.py` runs the streaming detector
  on the TFLite models (f32 backbone + int8 head) from the Mac microphone,
  with a live top-3 probability readout and event alerts. Also replays wav
  files (`--file`, `--fast`) for headless testing.

## Results so far

| Model | Clip acc (5-fold CV) | Frame acc | Params | ms/frame (CPU)* |
|-------|---------------------|-----------|--------|-----------------|
| YAMNet (frozen) + dense head | **0.981 ± 0.017** | 0.832 | 4.0M | 14.8 |
| DS-CNN (custom, from scratch) | 0.875 ± 0.043 | 0.768 | **24k** | 13.8 |

\* Batch-1 Keras `predict()` — call overhead dominates both; the honest
latency comparison comes with TFLite export (steps 4–6).

| YAMNet + head | DS-CNN |
|---|---|
| ![YAMNet confusion matrix](plots/confusion_matrix.png) | ![DS-CNN confusion matrix](plots/dscnn_confusion_matrix.png) |

Per-frame posteriors on test clips — why clip-level averaging (and later
streaming smoothing) matters:

![Test samples: waveform and posteriors](plots/test_samples_posteriors.png)

Edge deployment comparison (TFLite, single-threaded CPU; FP rate measured
on 5 min of never-seen ESC-50 classes with the streaming decision layer):

| Pipeline | Fold-5 clip acc | ms/window | RAM (MB) | Size (MB) | FP/min @θ=0.5 → 0.9 |
|----------|----------------|-----------|----------|-----------|---------------------|
| YAMNet f32 + head f32 | **1.000** | 0.63 | 32.8 | 16.09 | 16.4 → 3.0 |
| YAMNet f32 + head int8 | **1.000** | 0.62 | 31.4 | 15.31 | 16.2 → 3.0 |
| DS-CNN f32 | 0.781 | 0.66 | 3.1 | **0.10** | 19.0 → 4.0 |
| DS-CNN int8 | 0.766 | 0.64 | **~0** | **0.05** | 19.0 → 3.4 |

Streaming detection on a simulated 49 s live stream (held-out clips, 1 s
gaps) — 8/8 events, 0 false alarms, mean onset latency 1.46 s:

![Streaming timeline](plots/streaming_timeline.png)

Full details and per-class metrics in [summary.md](summary.md).

## Pipeline

![Training & inference pipeline](plots/pipeline.png)

(Interactive version: https://claude.ai/code/artifact/0eef4fa5-f363-48db-ad40-375cba35cbb2)

## Setup

```bash
python3.12 -m venv .venv          # TensorFlow needs Python <= 3.12/3.13
source .venv/bin/activate
pip install tensorflow tensorflow-hub librosa soundfile scikit-learn pandas matplotlib sounddevice
pip install "setuptools<81"   # tensorflow_hub still imports pkg_resources
```

## Data

```bash
cd data
curl -sL -o esc50.zip https://github.com/karolpiczak/ESC-50/archive/master.zip
unzip -q esc50.zip && rm esc50.zip
```

## Run

```bash
source .venv/bin/activate
cd src
python extract_embeddings.py   # -> data/yamnet_embeddings.npz
python train_classifier.py     # -> results/, plots/confusion_matrix.png
python train_dscnn.py          # step 2.5 -> results/model_comparison.csv
python streaming_inference.py  # step 3   -> results/streaming_report.txt
python export_tflite.py        # steps 4+5 -> results/tflite/*.tflite
python benchmark_tflite.py     # step 6   -> results/edge_comparison.csv
python mic_demo.py             # step 7   live mic demo (or --file x.wav)
```

## Layout

- `src/` — source code (`config.py`, `extract_embeddings.py`,
  `train_classifier.py`, `train_dscnn.py`, `plot_samples.py`)
- `data/` — ESC-50 dataset + cached embeddings (not checked in)
- `results/` — metrics, reports, saved classifier head (not checked in)
- `plots/` — confusion matrix and other figures
- `logs/` — run logs (not checked in)
