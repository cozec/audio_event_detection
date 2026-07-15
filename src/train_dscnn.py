"""Step 2.5: Train a small custom DS-CNN and compare with YAMNet + head.

Features: log-mel spectrograms (64 mels, 25 ms window, 10 ms hop) cut
into 0.96 s patches (96 frames) with a 0.48 s hop — the same framing
YAMNet uses internally, so frame- and clip-level numbers are directly
comparable. Patches inherit their clip's label; clip prediction is the
mean of patch probabilities. Same ESC-50 5-fold cross-validation.

Outputs:
    data/logmel_patches.npz          cached features
    results/dscnn_fold_metrics.csv
    results/dscnn_classification_report.txt
    results/dscnn.keras              model trained with fold 5 held out
    results/model_comparison.csv     YAMNet+head vs DS-CNN
    plots/dscnn_confusion_matrix.png
"""

import json
import logging
import sys
import time

import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    DATA_DIR,
    ESC50_AUDIO_DIR,
    ESC50_META_CSV,
    LOGS_DIR,
    PLOTS_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    SAMPLE_RATE,
    SELECTED_CLASSES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "train_dscnn.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

FEATURES_NPZ = DATA_DIR / "logmel_patches.npz"

N_MELS = 64
N_FFT = 400          # 25 ms at 16 kHz
HOP_LENGTH = 160     # 10 ms at 16 kHz
PATCH_FRAMES = 96    # 0.96 s
PATCH_HOP = 48       # 0.48 s


def extract_features():
    """Compute log-mel patches for all selected clips and cache them."""
    meta = pd.read_csv(ESC50_META_CSV)
    meta = meta[meta["category"].isin(SELECTED_CLASSES)].reset_index(drop=True)
    class_names = sorted(SELECTED_CLASSES)
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    patches, clip_idx, labels, folds, filenames = [], [], [], [], []
    t0 = time.time()
    for i, row in meta.iterrows():
        waveform, _ = librosa.load(
            ESC50_AUDIO_DIR / row["filename"], sr=SAMPLE_RATE, mono=True
        )
        mel = librosa.feature.melspectrogram(
            y=waveform, sr=SAMPLE_RATE, n_fft=N_FFT,
            hop_length=HOP_LENGTH, n_mels=N_MELS,
        )
        logmel = librosa.power_to_db(mel).T.astype(np.float32)  # (frames, mels)
        for start in range(0, len(logmel) - PATCH_FRAMES + 1, PATCH_HOP):
            patches.append(logmel[start:start + PATCH_FRAMES])
            clip_idx.append(i)
        filenames.append(row["filename"])
        labels.append(class_to_idx[row["category"]])
        folds.append(int(row["fold"]))

    x = np.stack(patches)
    # Per-dataset standardization (single scalar mean/std, edge-friendly).
    x = (x - x.mean()) / (x.std() + 1e-6)
    log.info(
        "Extracted %d patches of %s from %d clips in %.1fs",
        len(x), x.shape[1:], len(meta), time.time() - t0,
    )
    np.savez_compressed(
        FEATURES_NPZ,
        patches=x,
        clip_idx=np.array(clip_idx, dtype=np.int32),
        labels=np.array(labels, dtype=np.int32),
        folds=np.array(folds, dtype=np.int32),
        filenames=np.array(filenames),
        class_names=np.array(class_names),
    )
    log.info("Cached features to %s", FEATURES_NPZ)


def build_dscnn(n_classes):
    """Small keyword-spotting-style DS-CNN over (96, 64, 1) log-mel patches."""
    inp = tf.keras.layers.Input(shape=(PATCH_FRAMES, N_MELS, 1))
    x = tf.keras.layers.Conv2D(64, (10, 4), strides=(2, 2), padding="same",
                               use_bias=False)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    for _ in range(4):
        x = tf.keras.layers.DepthwiseConv2D((3, 3), padding="same",
                                            use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(64, (1, 1), use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(n_classes, name="logits")(x)

    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def train_one_fold(x, y_patch, clip_of_patch, labels, folds, test_fold, n_classes):
    """Train on all folds except test_fold; return metrics and clip predictions."""
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    clip_is_test = folds == test_fold
    patch_is_test = clip_is_test[clip_of_patch]
    x_train, y_train = x[~patch_is_test], y_patch[~patch_is_test]
    x_test, y_test = x[patch_is_test], y_patch[patch_is_test]

    model = build_dscnn(n_classes)
    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=80,
        batch_size=64,
        verbose=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=12, restore_best_weights=True
            )
        ],
    )

    probs = tf.nn.softmax(model.predict(x_test, verbose=0), axis=-1).numpy()
    frame_acc = float(np.mean(probs.argmax(axis=1) == y_test))

    test_clip_ids = np.where(clip_is_test)[0]
    test_clip_of_patch = clip_of_patch[patch_is_test]
    clip_pred = {}
    for cid in test_clip_ids:
        clip_pred[cid] = probs[test_clip_of_patch == cid].mean(axis=0).argmax()
    clip_acc = float(np.mean([clip_pred[cid] == labels[cid] for cid in test_clip_ids]))
    return frame_acc, clip_acc, clip_pred, model


def benchmark_ms_per_frame(model, x, n_runs=50):
    """Median single-frame inference latency in ms (batch of 1)."""
    sample = x[:1]
    model.predict(sample, verbose=0)  # warmup
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(sample, verbose=0)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def main():
    """Extract features (cached), run 5-fold CV, compare with YAMNet head."""
    if not FEATURES_NPZ.exists():
        extract_features()

    data = np.load(FEATURES_NPZ, allow_pickle=False)
    x = data["patches"][..., np.newaxis]
    clip_of_patch = data["clip_idx"]
    labels = data["labels"]
    folds = data["folds"]
    class_names = [str(c) for c in data["class_names"]]
    n_classes = len(class_names)
    y_patch = labels[clip_of_patch]
    log.info("Loaded %d patches %s from %d clips", len(x), x.shape[1:], len(labels))

    fold_rows, all_true, all_pred = [], [], []
    final_model = None
    for test_fold in sorted(np.unique(folds)):
        t0 = time.time()
        frame_acc, clip_acc, clip_pred, model = train_one_fold(
            x, y_patch, clip_of_patch, labels, folds, test_fold, n_classes
        )
        log.info(
            "Fold %d: frame acc %.3f | clip acc %.3f (%.0fs)",
            test_fold, frame_acc, clip_acc, time.time() - t0,
        )
        fold_rows.append(
            {"test_fold": int(test_fold), "frame_acc": frame_acc, "clip_acc": clip_acc}
        )
        for cid, pred in clip_pred.items():
            all_true.append(labels[cid])
            all_pred.append(pred)
        if test_fold == 5:
            final_model = model

    df = pd.DataFrame(fold_rows)
    df.to_csv(RESULTS_DIR / "dscnn_fold_metrics.csv", index=False)
    mean_clip, std_clip = df["clip_acc"].mean(), df["clip_acc"].std()
    mean_frame = df["frame_acc"].mean()
    log.info(
        "DS-CNN 5-fold CV: clip acc %.3f +/- %.3f | frame acc %.3f",
        mean_clip, std_clip, mean_frame,
    )

    report = classification_report(all_true, all_pred, target_names=class_names,
                                   digits=3)
    (RESULTS_DIR / "dscnn_classification_report.txt").write_text(
        f"5-fold CV clip-level accuracy: {mean_clip:.3f} +/- {std_clip:.3f}\n"
        f"5-fold CV frame-level accuracy: {mean_frame:.3f}\n\n{report}"
    )
    log.info("Classification report:\n%s", report)

    cm = confusion_matrix(all_true, all_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n_classes), class_names, rotation=45, ha="right")
    ax.set_yticks(range(n_classes), class_names)
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"DS-CNN - clip-level 5-fold CV (acc {mean_clip:.1%})")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "dscnn_confusion_matrix.png", dpi=150)

    final_model.save(RESULTS_DIR / "dscnn.keras")

    # ---- comparison with YAMNet + head ----
    dscnn_params = final_model.count_params()
    dscnn_ms = benchmark_ms_per_frame(final_model, x)

    yamnet_row = {}
    head_metrics = pd.read_csv(RESULTS_DIR / "fold_metrics.csv")
    head = tf.keras.models.load_model(RESULTS_DIR / "classifier_head.keras")
    import tensorflow_hub as hub

    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
    wave = np.zeros(int(0.96 * SAMPLE_RATE), dtype=np.float32)
    yamnet(wave)  # warmup
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        _, emb, _ = yamnet(wave)
        head.predict(emb.numpy(), verbose=0)
        times.append((time.perf_counter() - t0) * 1000)
    yamnet_ms = float(np.median(times))
    # YAMNet backbone is ~3.75M params (MobileNet-v1); count head exactly.
    yamnet_row = {
        "model": "YAMNet (frozen) + dense head",
        "clip_acc": head_metrics["clip_acc"].mean(),
        "clip_acc_std": head_metrics["clip_acc"].std(),
        "frame_acc": head_metrics["frame_acc"].mean(),
        "params": 3_750_000 + head.count_params(),
        "ms_per_frame_cpu": yamnet_ms,
    }
    dscnn_row = {
        "model": "DS-CNN (custom, from scratch)",
        "clip_acc": mean_clip,
        "clip_acc_std": std_clip,
        "frame_acc": mean_frame,
        "params": dscnn_params,
        "ms_per_frame_cpu": dscnn_ms,
    }
    comp = pd.DataFrame([yamnet_row, dscnn_row]).sort_values(
        "clip_acc", ascending=False
    )
    comp.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    log.info("Model comparison:\n%s", comp.to_string(index=False))


if __name__ == "__main__":
    main()
