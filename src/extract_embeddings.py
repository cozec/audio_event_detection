"""Step 2a: Extract YAMNet embeddings for the selected ESC-50 classes.

Loads each 5 s ESC-50 clip at 16 kHz mono, runs it through YAMNet
(tfhub), and stores the per-frame 1024-d embeddings (one frame per
0.96 s window, 0.48 s hop -> ~10 frames per clip) together with the
clip's label and cross-validation fold.

Output: data/yamnet_embeddings.npz with arrays
    embeddings : (n_frames_total, 1024) float32
    clip_idx   : (n_frames_total,) int    - index into the clip arrays below
    filenames  : (n_clips,) str
    labels     : (n_clips,) int           - index into class_names
    folds      : (n_clips,) int           - ESC-50 fold 1..5
    class_names: (n_classes,) str
"""

import logging
import sys
import time

import numpy as np
import pandas as pd
import librosa
import tensorflow_hub as hub

from config import (
    ESC50_AUDIO_DIR,
    ESC50_META_CSV,
    EMBEDDINGS_NPZ,
    LOGS_DIR,
    SAMPLE_RATE,
    SELECTED_CLASSES,
    YAMNET_HANDLE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "extract_embeddings.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def main():
    """Extract and save YAMNet embeddings for all selected clips."""
    meta = pd.read_csv(ESC50_META_CSV)
    meta = meta[meta["category"].isin(SELECTED_CLASSES)].reset_index(drop=True)
    class_names = sorted(SELECTED_CLASSES)
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    log.info("Selected %d clips across %d classes", len(meta), len(class_names))

    log.info("Loading YAMNet from %s ...", YAMNET_HANDLE)
    yamnet = hub.load(YAMNET_HANDLE)

    all_embeddings = []
    clip_idx = []
    filenames = []
    labels = []
    folds = []

    t0 = time.time()
    for i, row in meta.iterrows():
        wav_path = ESC50_AUDIO_DIR / row["filename"]
        waveform, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        _, embeddings, _ = yamnet(waveform)
        emb = embeddings.numpy().astype(np.float32)

        all_embeddings.append(emb)
        clip_idx.extend([i] * len(emb))
        filenames.append(row["filename"])
        labels.append(class_to_idx[row["category"]])
        folds.append(int(row["fold"]))

        if (i + 1) % 50 == 0:
            log.info("Processed %d/%d clips (%.1fs)", i + 1, len(meta), time.time() - t0)

    embeddings_arr = np.concatenate(all_embeddings, axis=0)
    log.info(
        "Done: %d clips -> %d frames of %d-d embeddings in %.1fs",
        len(meta), len(embeddings_arr), embeddings_arr.shape[1], time.time() - t0,
    )

    np.savez_compressed(
        EMBEDDINGS_NPZ,
        embeddings=embeddings_arr,
        clip_idx=np.array(clip_idx, dtype=np.int32),
        filenames=np.array(filenames),
        labels=np.array(labels, dtype=np.int32),
        folds=np.array(folds, dtype=np.int32),
        class_names=np.array(class_names),
    )
    log.info("Saved embeddings to %s", EMBEDDINGS_NPZ)


if __name__ == "__main__":
    main()
