"""Step 6: Compare TFLite pipelines — accuracy, latency, RAM, size, FP rate.

Four deployable pipelines:
    1. YAMNet f32 TFLite backbone + head f32 TFLite
    2. YAMNet f32 TFLite backbone + head int8 TFLite
    3. DS-CNN f32 TFLite
    4. DS-CNN int8 TFLite

Metrics:
    accuracy   clip-level on fold 5 (held out from both saved models),
               end-to-end through TFLite interpreters
    latency    median ms per 0.96 s window, single-threaded CPU
    RAM        peak-RSS delta of a fresh subprocess that loads + runs the
               pipeline (approximate but honest)
    size       total .tflite bytes per pipeline
    FP rate    events/min of the step 3 decision layer (K=3, THETA=0.5,
               M=2, refractory 3 s) on a ~5 min distractor stream built
               from fold-5 clips of the 42 NON-target ESC-50 classes

Outputs: results/edge_comparison.csv, results/edge_comparison.txt
"""

import logging
import resource
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import librosa
import tensorflow as tf

from config import (
    ESC50_AUDIO_DIR,
    ESC50_META_CSV,
    LOGS_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    SAMPLE_RATE,
    SELECTED_CLASSES,
)
from train_dscnn import (
    FEATURES_NPZ,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    PATCH_FRAMES,
    PATCH_HOP,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "benchmark_tflite.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

TFLITE_DIR = RESULTS_DIR / "tflite"
WINDOW = int(0.96 * SAMPLE_RATE)
HOP = int(0.48 * SAMPLE_RATE)

# Step 3 decision layer.
K_SMOOTH, THETA, M_CONSECUTIVE, REFRACTORY_S = 3, 0.5, 2, 3.0
HOP_S = 0.48

N_DISTRACTOR_CLIPS = 60


def make_interpreter(name, input_shape=None):
    interp = tf.lite.Interpreter(
        model_path=str(TFLITE_DIR / name), num_threads=1
    )
    if input_shape is not None:
        interp.resize_tensor_input(
            interp.get_input_details()[0]["index"], input_shape
        )
    interp.allocate_tensors()
    return interp


def run_tflite(interp, x):
    """Invoke a single-input/single-output interpreter, handling int8 IO."""
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    if inp["dtype"] == np.int8:
        scale, zp = inp["quantization"]
        x = np.clip(np.round(x / scale + zp), -128, 127).astype(np.int8)
    interp.set_tensor(inp["index"], x)
    interp.invoke()
    y = interp.get_tensor(out["index"])
    if out["dtype"] == np.int8:
        scale, zp = out["quantization"]
        y = (y.astype(np.float32) - zp) * scale
    return y


def yamnet_embed(interp, window):
    """Run the TFLite YAMNet backbone; return the 1024-d embedding(s)."""
    inp = interp.get_input_details()[0]
    interp.set_tensor(inp["index"], window.astype(np.float32))
    interp.invoke()
    for detail in interp.get_output_details():
        if detail["shape"][-1] == 1024:
            return interp.get_tensor(detail["index"])
    raise RuntimeError("no 1024-d embedding output found")


def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def windows_of(waveform):
    """Slice a waveform into 0.96 s windows at 0.48 s hop (>=1 window)."""
    if len(waveform) < WINDOW:
        waveform = np.pad(waveform, (0, WINDOW - len(waveform)))
    return [
        waveform[s:s + WINDOW]
        for s in range(0, len(waveform) - WINDOW + 1, HOP)
    ]


def logmel_patch(window, mean, std):
    """One standardized (96, 64, 1) log-mel patch from a 0.96 s window."""
    mel = librosa.feature.melspectrogram(
        y=window, sr=SAMPLE_RATE, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS,
    )
    logmel = librosa.power_to_db(mel).T[:PATCH_FRAMES].astype(np.float32)
    return ((logmel - mean) / (std + 1e-6))[np.newaxis, ..., np.newaxis]


def logmel_stats():
    """Recompute the training-set scalar mean/std of raw log-mel patches
    (train_dscnn.py standardized before caching, so redo it identically)."""
    meta = pd.read_csv(ESC50_META_CSV)
    meta = meta[meta["category"].isin(SELECTED_CLASSES)].reset_index(drop=True)
    vals = []
    for _, row in meta.iterrows():
        wav, _ = librosa.load(
            ESC50_AUDIO_DIR / row["filename"], sr=SAMPLE_RATE, mono=True
        )
        mel = librosa.feature.melspectrogram(
            y=wav, sr=SAMPLE_RATE, n_fft=N_FFT,
            hop_length=HOP_LENGTH, n_mels=N_MELS,
        )
        logmel = librosa.power_to_db(mel).T.astype(np.float32)
        for s in range(0, len(logmel) - PATCH_FRAMES + 1, PATCH_HOP):
            vals.append(logmel[s:s + PATCH_FRAMES])
    arr = np.stack(vals)
    return float(arr.mean()), float(arr.std())


# ---------------- accuracy ----------------

def fold5_clips():
    meta = pd.read_csv(ESC50_META_CSV)
    meta = meta[(meta["category"].isin(SELECTED_CLASSES)) & (meta["fold"] == 5)]
    class_names = sorted(SELECTED_CLASSES)
    return meta.reset_index(drop=True), class_names


def accuracy_yamnet(head_name):
    """Fold-5 clip accuracy end-to-end: TFLite backbone + TFLite head."""
    backbone = make_interpreter("yamnet_backbone_f32.tflite", [WINDOW])
    head = make_interpreter(head_name)
    meta, class_names = fold5_clips()
    correct = 0
    for _, row in meta.iterrows():
        wav, _ = librosa.load(
            ESC50_AUDIO_DIR / row["filename"], sr=SAMPLE_RATE, mono=True
        )
        probs = [
            softmax(run_tflite(head, yamnet_embed(backbone, w)))[0]
            for w in windows_of(wav)
        ]
        pred = np.mean(probs, axis=0).argmax()
        correct += class_names[pred] == row["category"]
    return correct / len(meta)


def accuracy_dscnn(name, mean, std):
    """Fold-5 clip accuracy: log-mel windows through a DS-CNN interpreter."""
    interp = make_interpreter(name)
    meta, class_names = fold5_clips()
    correct = 0
    for _, row in meta.iterrows():
        wav, _ = librosa.load(
            ESC50_AUDIO_DIR / row["filename"], sr=SAMPLE_RATE, mono=True
        )
        probs = [
            softmax(run_tflite(interp, logmel_patch(w, mean, std)))[0]
            for w in windows_of(wav)
        ]
        pred = np.mean(probs, axis=0).argmax()
        correct += class_names[pred] == row["category"]
    return correct / len(meta)


# ---------------- latency ----------------

def latency_yamnet(head_name, n=100):
    backbone = make_interpreter("yamnet_backbone_f32.tflite", [WINDOW])
    head = make_interpreter(head_name)
    window = np.random.default_rng(0).standard_normal(WINDOW).astype(np.float32) * 0.05
    yamnet_embed(backbone, window)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        emb = yamnet_embed(backbone, window)
        run_tflite(head, emb[:1])
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def latency_dscnn(name, mean, std, n=100):
    interp = make_interpreter(name)
    window = np.random.default_rng(0).standard_normal(WINDOW).astype(np.float32) * 0.05
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        patch = logmel_patch(window, mean, std)  # frontend counted, like YAMNet's
        run_tflite(interp, patch)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


# ---------------- RAM ----------------

RAM_PROBE = r"""
import sys, resource, numpy as np
import tensorflow as tf
mode = sys.argv[1]
if mode != "baseline":
    interps = []
    for path in sys.argv[2:]:
        it = tf.lite.Interpreter(model_path=path, num_threads=1)
        it.allocate_tensors()
        inp = it.get_input_details()[0]
        x = np.zeros(inp["shape"], dtype=inp["dtype"])
        it.set_tensor(inp["index"], x)
        it.invoke()
        interps.append(it)
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
"""


def ram_mb(*model_names):
    """Peak-RSS delta (MB) vs a baseline subprocess that only imports TF.

    Median of 3 probes each; small models disappear into RSS noise
    (a few MB), so the result is floored at 0."""
    def probe(args):
        out = subprocess.run(
            [sys.executable, "-c", RAM_PROBE, *args],
            capture_output=True, text=True, check=True,
        )
        return int(out.stdout.strip().splitlines()[-1])

    baseline = np.median([probe(["baseline"]) for _ in range(3)])
    loaded = np.median(
        [probe(["run"] + [str(TFLITE_DIR / n) for n in model_names])
         for _ in range(3)]
    )
    return max(0.0, (loaded - baseline) / 1e6)  # ru_maxrss is bytes on macOS


# ---------------- false-positive rate ----------------

def build_distractor_stream():
    """~5 min of fold-5 audio from classes NOT in the target set."""
    meta = pd.read_csv(ESC50_META_CSV)
    meta = meta[(~meta["category"].isin(SELECTED_CLASSES)) & (meta["fold"] == 5)]
    picks = meta.sample(N_DISTRACTOR_CLIPS, random_state=RANDOM_SEED)
    waves = [
        librosa.load(ESC50_AUDIO_DIR / f, sr=SAMPLE_RATE, mono=True)[0]
        for f in picks["filename"]
    ]
    return np.concatenate(waves), sorted(picks["category"].unique())


def fp_events_per_min(frame_probs, theta=THETA):
    """Apply the step 3 decision layer; count events fired."""
    from collections import deque

    recent = deque(maxlen=K_SMOOTH)
    above = np.zeros(frame_probs.shape[1], dtype=int)
    last = np.full(frame_probs.shape[1], -np.inf)
    events = 0
    for i, p in enumerate(frame_probs):
        t = i * HOP_S
        recent.append(p)
        smoothed = np.mean(recent, axis=0)
        for c in range(len(p)):
            above[c] = above[c] + 1 if smoothed[c] > theta else 0
            if above[c] >= M_CONSECUTIVE and t - last[c] > REFRACTORY_S:
                last[c] = t
                events += 1
    minutes = len(frame_probs) * HOP_S / 60
    return events / minutes


def main():
    log.info("Recomputing log-mel training stats ...")
    mean, std = logmel_stats()
    log.info("log-mel stats: mean %.2f std %.2f", mean, std)

    log.info("Building distractor stream ...")
    stream, distractor_classes = build_distractor_stream()
    stream_windows = windows_of(stream)
    log.info(
        "Distractor stream: %.1f min, %d windows, classes: %s",
        len(stream) / SAMPLE_RATE / 60, len(stream_windows),
        ", ".join(distractor_classes[:8]) + " ...",
    )

    log.info("Computing distractor posteriors (YAMNet paths) ...")
    backbone = make_interpreter("yamnet_backbone_f32.tflite", [WINDOW])
    embs = np.concatenate([yamnet_embed(backbone, w)[:1] for w in stream_windows])
    fp_probs_yamnet = {}
    for head_name in ["head_f32.tflite", "head_int8.tflite"]:
        head = make_interpreter(head_name)
        fp_probs_yamnet[head_name] = np.stack(
            [softmax(run_tflite(head, e[np.newaxis]))[0] for e in embs]
        )

    log.info("Computing distractor posteriors (DS-CNN paths) ...")
    patches = [logmel_patch(w, mean, std) for w in stream_windows]
    fp_probs_dscnn = {}
    for name in ["dscnn_f32.tflite", "dscnn_int8.tflite"]:
        interp = make_interpreter(name)
        fp_probs_dscnn[name] = np.stack(
            [softmax(run_tflite(interp, p))[0] for p in patches]
        )

    size = lambda *names: sum((TFLITE_DIR / n).stat().st_size for n in names) / 1e6

    log.info("Measuring accuracy, latency, RAM ...")
    rows = [
        {
            "pipeline": "YAMNet f32 + head f32",
            "fold5_clip_acc": accuracy_yamnet("head_f32.tflite"),
            "ms_per_window": latency_yamnet("head_f32.tflite"),
            "ram_mb": ram_mb("yamnet_backbone_f32.tflite", "head_f32.tflite"),
            "size_mb": size("yamnet_backbone_f32.tflite", "head_f32.tflite"),
            "fp_per_min": fp_events_per_min(fp_probs_yamnet["head_f32.tflite"]),
        },
        {
            "pipeline": "YAMNet f32 + head int8",
            "fold5_clip_acc": accuracy_yamnet("head_int8.tflite"),
            "ms_per_window": latency_yamnet("head_int8.tflite"),
            "ram_mb": ram_mb("yamnet_backbone_f32.tflite", "head_int8.tflite"),
            "size_mb": size("yamnet_backbone_f32.tflite", "head_int8.tflite"),
            "fp_per_min": fp_events_per_min(fp_probs_yamnet["head_int8.tflite"]),
        },
        {
            "pipeline": "DS-CNN f32",
            "fold5_clip_acc": accuracy_dscnn("dscnn_f32.tflite", mean, std),
            "ms_per_window": latency_dscnn("dscnn_f32.tflite", mean, std),
            "ram_mb": ram_mb("dscnn_f32.tflite"),
            "size_mb": size("dscnn_f32.tflite"),
            "fp_per_min": fp_events_per_min(fp_probs_dscnn["dscnn_f32.tflite"]),
        },
        {
            "pipeline": "DS-CNN int8",
            "fold5_clip_acc": accuracy_dscnn("dscnn_int8.tflite", mean, std),
            "ms_per_window": latency_dscnn("dscnn_int8.tflite", mean, std),
            "ram_mb": ram_mb("dscnn_int8.tflite"),
            "size_mb": size("dscnn_int8.tflite"),
            "fp_per_min": fp_events_per_min(fp_probs_dscnn["dscnn_int8.tflite"]),
        },
    ]

    df = pd.DataFrame(rows).sort_values("fold5_clip_acc", ascending=False)
    df.to_csv(RESULTS_DIR / "edge_comparison.csv", index=False)

    # FP rate vs decision threshold (same distractor posteriors).
    all_fp_probs = {
        "YAMNet f32 + head f32": fp_probs_yamnet["head_f32.tflite"],
        "YAMNet f32 + head int8": fp_probs_yamnet["head_int8.tflite"],
        "DS-CNN f32": fp_probs_dscnn["dscnn_f32.tflite"],
        "DS-CNN int8": fp_probs_dscnn["dscnn_int8.tflite"],
    }
    sweep = pd.DataFrame(
        [
            {"pipeline": name,
             **{f"fp_per_min@{th}": fp_events_per_min(probs, th)
                for th in (0.5, 0.6, 0.7, 0.8, 0.9)}}
            for name, probs in all_fp_probs.items()
        ]
    )
    sweep.to_csv(RESULTS_DIR / "fp_vs_theta.csv", index=False)
    log.info("FP sweep:\n%s", sweep.to_string(index=False,
             float_format=lambda v: f"{v:.2f}"))
    txt = df.to_string(index=False, float_format=lambda v: f"{v:.3f}")
    (RESULTS_DIR / "edge_comparison.txt").write_text(
        f"Distractor stream: {N_DISTRACTOR_CLIPS} fold-5 clips from "
        f"non-target classes, decision layer K={K_SMOOTH} THETA={THETA} "
        f"M={M_CONSECUTIVE} refractory={REFRACTORY_S}s\n"
        f"Latency: single-threaded TFLite, median over 100 windows "
        f"(DS-CNN includes log-mel frontend)\n\n{txt}\n"
    )
    log.info("Edge comparison:\n%s", txt)


if __name__ == "__main__":
    main()
