"""Step 2b: Train a classifier head on YAMNet embeddings.

Trains a small dense head on the per-frame 1024-d YAMNet embeddings
(frames inherit their clip's label) and evaluates with the standard
ESC-50 5-fold cross-validation: for each fold f, train on the other
four folds and test on f. Clip-level predictions are the mean of the
clip's frame probabilities.

Outputs:
    results/fold_metrics.csv        per-fold frame/clip accuracy
    results/classification_report.txt
    plots/confusion_matrix.png      aggregated over the 5 test folds
    results/classifier_head.keras   final head trained on folds 1-4
                                    (fold 5 held out), for later TFLite export
"""

import json
import logging
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    EMBEDDINGS_NPZ,
    LOGS_DIR,
    PLOTS_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "train_classifier.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def build_head(n_classes):
    """Build the small dense classifier head over 1024-d YAMNet embeddings."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(1024,), name="yamnet_embedding"),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(n_classes, name="logits"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def train_one_fold(x, y_frame, clip_of_frame, labels, folds, test_fold, n_classes):
    """Train on all folds except test_fold; return metrics and clip predictions."""
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    clip_is_test = folds == test_fold
    frame_is_test = clip_is_test[clip_of_frame]

    x_train, y_train = x[~frame_is_test], y_frame[~frame_is_test]
    x_test, y_test = x[frame_is_test], y_frame[frame_is_test]

    model = build_head(n_classes)
    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=60,
        batch_size=64,
        verbose=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=8, restore_best_weights=True
            )
        ],
    )

    frame_probs = tf.nn.softmax(model.predict(x_test, verbose=0), axis=-1).numpy()
    frame_acc = float(np.mean(frame_probs.argmax(axis=1) == y_test))

    # Clip-level: average frame probabilities per test clip.
    test_clip_ids = np.where(clip_is_test)[0]
    clip_pred = {}
    test_clip_of_frame = clip_of_frame[frame_is_test]
    for cid in test_clip_ids:
        clip_pred[cid] = frame_probs[test_clip_of_frame == cid].mean(axis=0).argmax()
    clip_acc = float(
        np.mean([clip_pred[cid] == labels[cid] for cid in test_clip_ids])
    )

    return frame_acc, clip_acc, clip_pred, model


def main():
    """Run 5-fold CV over the YAMNet embeddings and save results."""
    data = np.load(EMBEDDINGS_NPZ, allow_pickle=False)
    x = data["embeddings"]
    clip_of_frame = data["clip_idx"]
    labels = data["labels"]
    folds = data["folds"]
    class_names = [str(c) for c in data["class_names"]]
    n_classes = len(class_names)
    y_frame = labels[clip_of_frame]

    log.info(
        "Loaded %d frames from %d clips, %d classes: %s",
        len(x), len(labels), n_classes, class_names,
    )

    fold_rows = []
    all_true, all_pred = [], []
    final_model = None
    for test_fold in sorted(np.unique(folds)):
        frame_acc, clip_acc, clip_pred, model = train_one_fold(
            x, y_frame, clip_of_frame, labels, folds, test_fold, n_classes
        )
        log.info(
            "Fold %d: frame acc %.3f | clip acc %.3f", test_fold, frame_acc, clip_acc
        )
        fold_rows.append(
            {"test_fold": int(test_fold), "frame_acc": frame_acc, "clip_acc": clip_acc}
        )
        for cid, pred in clip_pred.items():
            all_true.append(labels[cid])
            all_pred.append(pred)
        if test_fold == 5:
            final_model = model  # head trained on folds 1-4, standard holdout

    df = pd.DataFrame(fold_rows)
    df.to_csv(RESULTS_DIR / "fold_metrics.csv", index=False)
    mean_clip = df["clip_acc"].mean()
    std_clip = df["clip_acc"].std()
    mean_frame = df["frame_acc"].mean()
    log.info(
        "5-fold CV: clip acc %.3f +/- %.3f | frame acc %.3f",
        mean_clip, std_clip, mean_frame,
    )

    report = classification_report(
        all_true, all_pred, target_names=class_names, digits=3
    )
    (RESULTS_DIR / "classification_report.txt").write_text(
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
    ax.set_title(f"YAMNet + dense head - clip-level 5-fold CV (acc {mean_clip:.1%})")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=150)
    log.info("Saved confusion matrix to plots/confusion_matrix.png")

    final_model.save(RESULTS_DIR / "classifier_head.keras")
    with open(RESULTS_DIR / "class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)
    log.info("Saved classifier head (trained on folds 1-4) and class names")


if __name__ == "__main__":
    main()
