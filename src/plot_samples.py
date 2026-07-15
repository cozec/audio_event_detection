"""Plot waveform + per-frame posterior probabilities for 3 test-set clips.

Takes 3 clips (distinct classes) from fold 5 — the fold held out when
results/classifier_head.keras was trained — runs their cached YAMNet
frame embeddings through the head, and plots the waveform aligned with
a class x time posterior heatmap.

Output: plots/test_samples_posteriors.png
"""

import logging
import sys

import numpy as np
import librosa
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    EMBEDDINGS_NPZ,
    ESC50_AUDIO_DIR,
    LOGS_DIR,
    PLOTS_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    SAMPLE_RATE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "plot_samples.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# YAMNet frame geometry: 0.96 s window, 0.48 s hop.
FRAME_HOP = 0.48
FRAME_LEN = 0.96
TEST_FOLD = 5
N_SAMPLES = 3


def main():
    """Render waveform + posterior heatmap for 3 fold-5 clips."""
    data = np.load(EMBEDDINGS_NPZ, allow_pickle=False)
    embeddings = data["embeddings"]
    clip_of_frame = data["clip_idx"]
    filenames = data["filenames"]
    labels = data["labels"]
    folds = data["folds"]
    class_names = [str(c) for c in data["class_names"]]

    model = tf.keras.models.load_model(RESULTS_DIR / "classifier_head.keras")

    rng = np.random.default_rng(RANDOM_SEED)
    test_clips = np.where(folds == TEST_FOLD)[0]
    chosen_classes = rng.choice(np.unique(labels[test_clips]), N_SAMPLES, replace=False)
    chosen = [rng.choice(test_clips[labels[test_clips] == c]) for c in chosen_classes]

    fig, axes = plt.subplots(
        2, N_SAMPLES, figsize=(5 * N_SAMPLES, 6),
        gridspec_kw={"height_ratios": [1, 1.3]}, sharex="col",
    )

    for col, cid in enumerate(chosen):
        wav_path = ESC50_AUDIO_DIR / str(filenames[cid])
        waveform, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        t = np.arange(len(waveform)) / SAMPLE_RATE

        frame_emb = embeddings[clip_of_frame == cid]
        probs = tf.nn.softmax(model.predict(frame_emb, verbose=0), axis=-1).numpy()
        clip_pred = probs.mean(axis=0).argmax()
        n_frames = len(probs)
        frame_centers = FRAME_HOP * np.arange(n_frames) + FRAME_LEN / 2

        true_name = class_names[labels[cid]]
        pred_name = class_names[clip_pred]
        log.info(
            "%s: true=%s pred=%s (p=%.2f)",
            filenames[cid], true_name, pred_name, probs.mean(axis=0).max(),
        )

        ax_w = axes[0, col]
        ax_w.plot(t, waveform, color="#4477aa", linewidth=0.4)
        ax_w.set_ylim(-1.05 * np.abs(waveform).max(), 1.05 * np.abs(waveform).max())
        ax_w.set_ylabel("Amplitude" if col == 0 else "")
        ok = "correct" if clip_pred == labels[cid] else "WRONG"
        ax_w.set_title(
            f"{filenames[cid]}\ntrue: {true_name} | pred: {pred_name} ({ok})",
            fontsize=10,
        )
        ax_w.grid(alpha=0.2)

        ax_p = axes[1, col]
        # Column k spans the k-th 0.96 s YAMNet window (0.48 s hop).
        edges = np.concatenate([
            FRAME_HOP * np.arange(n_frames), [FRAME_HOP * (n_frames - 1) + FRAME_LEN]
        ])
        mesh = ax_p.pcolormesh(
            edges, np.arange(len(class_names) + 1), probs.T,
            cmap="Blues", vmin=0, vmax=1,
        )
        if col == 0:
            ax_p.set_yticks(np.arange(len(class_names)) + 0.5, class_names, fontsize=8)
        else:
            ax_p.set_yticks([])
        ax_p.set_xlabel("Time (s)")
        ax_p.set_xlim(0, t[-1])
        # Outline the true-class row.
        ax_p.add_patch(plt.Rectangle(
            (0, labels[cid]), edges[-1], 1,
            fill=False, edgecolor="#cc3311", linewidth=2,
        ))
        for k, p in enumerate(probs.argmax(axis=1)):
            ax_p.plot(frame_centers[k], p + 0.5, marker="o", color="#ee6677",
                      markersize=4, markeredgecolor="white", markeredgewidth=0.8)

    fig.colorbar(mesh, ax=axes[1, :], label="Posterior probability",
                 fraction=0.02, pad=0.01)
    fig.suptitle(
        "Test-set samples (fold 5): waveform and per-frame posteriors "
        "(red dot = per-frame argmax, outlined row = true class)",
        fontsize=12,
    )
    out = PLOTS_DIR / "test_samples_posteriors.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    log.info("Saved %s", out)


if __name__ == "__main__":
    main()
