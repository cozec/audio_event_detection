"""Step 3: Streaming inference with overlapping audio windows.

StreamingDetector consumes arbitrary-size audio chunks (as a mic
callback would deliver them) and runs YAMNet + the trained dense head
on a 0.96 s window every 0.48 s hop:

    chunks -> ring buffer -> [0.96 s window each 0.48 s] -> YAMNet
    -> head -> posterior -> moving average over K hops
    -> fire event when smoothed p > THETA for M consecutive hops
    -> per-class refractory so one sound emits one event

The simulator concatenates fold-5 clips (held out from the saved head)
with silence gaps into one long stream, feeds it in mic-sized chunks,
and scores detections against the known clip timeline.

Outputs:
    results/streaming_events.csv     detected events with timestamps
    results/streaming_report.txt     hits / misses / false alarms / latency
    plots/streaming_timeline.png     smoothed posteriors + events vs truth
"""

import logging
import sys
import time
from collections import deque

import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    ESC50_AUDIO_DIR,
    ESC50_META_CSV,
    LOGS_DIR,
    PLOTS_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    SAMPLE_RATE,
    SELECTED_CLASSES,
    YAMNET_HANDLE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "streaming_inference.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

WINDOW_S = 0.96
HOP_S = 0.48
WINDOW = int(WINDOW_S * SAMPLE_RATE)   # 15360 samples
HOP = int(HOP_S * SAMPLE_RATE)         # 7680 samples

# Decision-layer knobs (trade latency vs false positives, step 6 tunes them).
K_SMOOTH = 3          # moving average over K hops (~1.9 s context)
THETA = 0.5           # fire threshold on smoothed probability
M_CONSECUTIVE = 2     # hops above THETA required to fire
REFRACTORY_S = 3.0    # per-class dead time after firing

CHUNK = 1024          # simulated mic callback size
GAP_S = 1.0           # silence between clips in the simulated stream
CLIP_S = 5.0


class StreamingDetector:
    """Streams audio chunks through YAMNet + head and emits events."""

    def __init__(self, yamnet, head, class_names):
        self.yamnet = yamnet
        self.head = head
        self.class_names = class_names
        self.buffer = np.zeros(0, dtype=np.float32)
        self.samples_seen = 0
        self.next_hop_end = WINDOW
        self.recent = deque(maxlen=K_SMOOTH)
        self.above = np.zeros(len(class_names), dtype=int)
        self.last_fired = np.full(len(class_names), -np.inf)
        self.posteriors = []   # (t, smoothed probs) per hop, for plotting
        self.infer_ms = []

    def feed(self, chunk):
        """Consume one audio chunk; return list of events fired within it."""
        self.buffer = np.concatenate([self.buffer, chunk.astype(np.float32)])
        self.samples_seen += len(chunk)
        events = []
        while self.samples_seen >= self.next_hop_end:
            # Keep only the newest window worth of samples.
            extra = len(self.buffer) - (self.samples_seen - self.next_hop_end) - WINDOW
            window = self.buffer[extra:extra + WINDOW]
            events += self._process_window(self.next_hop_end / SAMPLE_RATE, window)
            self.next_hop_end += HOP
        # Drop samples no future window can need.
        keep = WINDOW + (self.samples_seen - self.next_hop_end + HOP)
        if len(self.buffer) > keep > 0:
            self.buffer = self.buffer[-keep:]
        return events

    def _process_window(self, t, window):
        t0 = time.perf_counter()
        _, emb, _ = self.yamnet(window)
        logits = self.head(emb.numpy(), training=False)
        probs = tf.nn.softmax(logits, axis=-1).numpy().mean(axis=0)
        self.infer_ms.append((time.perf_counter() - t0) * 1000)

        self.recent.append(probs)
        smoothed = np.mean(self.recent, axis=0)
        self.posteriors.append((t, smoothed))

        events = []
        for c in range(len(self.class_names)):
            if smoothed[c] > THETA:
                self.above[c] += 1
            else:
                self.above[c] = 0
            if (
                self.above[c] >= M_CONSECUTIVE
                and t - self.last_fired[c] > REFRACTORY_S
            ):
                self.last_fired[c] = t
                events.append(
                    {"time_s": t, "class": self.class_names[c],
                     "prob": float(smoothed[c])}
                )
        return events


def build_stream(rng):
    """Concatenate one random fold-5 clip per class with silence gaps."""
    meta = pd.read_csv(ESC50_META_CSV)
    meta = meta[(meta["category"].isin(SELECTED_CLASSES)) & (meta["fold"] == 5)]
    picks = (
        meta.groupby("category")
        .apply(lambda g: g.sample(1, random_state=rng), include_groups=False)
        .reset_index()
    )
    picks = picks.sample(frac=1, random_state=rng).reset_index(drop=True)

    gap = np.zeros(int(GAP_S * SAMPLE_RATE), dtype=np.float32)
    stream, truth = [gap], []
    t = GAP_S
    for _, row in picks.iterrows():
        wav, _ = librosa.load(
            ESC50_AUDIO_DIR / row["filename"], sr=SAMPLE_RATE, mono=True
        )
        stream += [wav, gap]
        truth.append({"start_s": t, "end_s": t + CLIP_S, "class": row["category"],
                      "filename": row["filename"]})
        t += CLIP_S + GAP_S
    return np.concatenate(stream), pd.DataFrame(truth)


def score(events, truth):
    """Match events to ground-truth clip spans; return hits/misses/FAs."""
    hits, latencies = [], []
    matched = set()
    false_alarms = []
    for ev in events:
        m = truth[
            (truth["class"] == ev["class"])
            & (truth["start_s"] <= ev["time_s"])
            & (ev["time_s"] <= truth["end_s"] + HOP_S)
        ]
        if len(m) and m.index[0] not in matched:
            matched.add(m.index[0])
            hits.append(ev)
            latencies.append(ev["time_s"] - m.iloc[0]["start_s"])
        elif len(m):
            pass  # duplicate detection of an already-matched clip
        else:
            false_alarms.append(ev)
    misses = truth[~truth.index.isin(matched)]
    return hits, latencies, false_alarms, misses


def plot_timeline(detector, truth, events, class_names, out_path):
    """Smoothed posterior heatmap with ground truth spans and events."""
    times = np.array([t for t, _ in detector.posteriors])
    probs = np.stack([p for _, p in detector.posteriors])
    fig, ax = plt.subplots(figsize=(14, 4.5))
    edges = np.concatenate([times - HOP_S, [times[-1]]])
    ax.pcolormesh(edges, np.arange(len(class_names) + 1), probs.T,
                  cmap="Blues", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(class_names)) + 0.5, class_names, fontsize=8)
    for _, row in truth.iterrows():
        c = class_names.index(row["class"])
        ax.plot([row["start_s"], row["end_s"]], [c + 0.5] * 2,
                color="#cc3311", linewidth=2, alpha=0.9)
    for ev in events:
        c = class_names.index(ev["class"])
        ax.plot(ev["time_s"], c + 0.5, marker="v", color="#cc3311",
                markersize=9, markeredgecolor="white")
    ax.set_xlabel("Stream time (s)")
    ax.set_title(
        "Streaming detection - smoothed posteriors (blue), ground-truth spans "
        "(red lines), fired events (red markers)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def main():
    """Simulate a live stream from fold-5 clips and score detections."""
    class_names = sorted(SELECTED_CLASSES)
    log.info("Loading YAMNet and classifier head ...")
    yamnet = hub.load(YAMNET_HANDLE)
    head = tf.keras.models.load_model(RESULTS_DIR / "classifier_head.keras")

    stream, truth = build_stream(RANDOM_SEED)
    log.info("Built %.1fs stream with %d events", len(stream) / SAMPLE_RATE,
             len(truth))

    detector = StreamingDetector(yamnet, head, class_names)
    events = []
    t0 = time.time()
    for i in range(0, len(stream), CHUNK):
        events += detector.feed(stream[i:i + CHUNK])
    wall = time.time() - t0

    hits, latencies, false_alarms, misses = score(events, truth)
    rtf = wall / (len(stream) / SAMPLE_RATE)
    report = (
        f"Stream: {len(stream) / SAMPLE_RATE:.1f}s, {len(truth)} true events, "
        f"chunk={CHUNK} samples\n"
        f"Knobs: K={K_SMOOTH} THETA={THETA} M={M_CONSECUTIVE} "
        f"refractory={REFRACTORY_S}s\n\n"
        f"Detected events: {len(events)}\n"
        f"Hits:            {len(hits)}/{len(truth)}\n"
        f"False alarms:    {len(false_alarms)}\n"
        f"Missed:          {len(misses)} "
        f"({', '.join(misses['class']) if len(misses) else '-'})\n"
        f"Detection latency from clip onset: "
        f"mean {np.mean(latencies):.2f}s, worst {np.max(latencies):.2f}s\n\n"
        f"Per-hop inference: median {np.median(detector.infer_ms):.1f} ms "
        f"(budget {HOP_S * 1000:.0f} ms/hop)\n"
        f"Real-time factor: {rtf:.3f} ({1 / rtf:.0f}x faster than real time)\n"
    )
    (RESULTS_DIR / "streaming_report.txt").write_text(report)
    log.info("\n%s", report)

    pd.DataFrame(events).to_csv(RESULTS_DIR / "streaming_events.csv", index=False)
    plot_timeline(detector, truth, events, class_names,
                  PLOTS_DIR / "streaming_timeline.png")
    log.info("Saved events, report, and timeline plot")


if __name__ == "__main__":
    main()
