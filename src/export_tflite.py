"""Steps 4+5: Export models to TFLite and apply INT8 quantization.

Exports:
    results/tflite/yamnet_backbone_f32.tflite   fixed 0.96 s input window
    results/tflite/head_f32.tflite
    results/tflite/head_int8.tflite             full integer, calibrated on
                                                cached fold 1-4 embeddings
    results/tflite/dscnn_f32.tflite
    results/tflite/dscnn_int8.tflite            full integer, calibrated on
                                                fold 1-4 log-mel patches

The YAMNet backbone keeps float32: its FFT/log-mel frontend ops don't
quantize to INT8 (that is exactly the DS-CNN's edge advantage).
"""

import logging
import sys

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from config import LOGS_DIR, RESULTS_DIR, SAMPLE_RATE, YAMNET_HANDLE
from train_dscnn import FEATURES_NPZ

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "export_tflite.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

TFLITE_DIR = RESULTS_DIR / "tflite"
WINDOW = int(0.96 * SAMPLE_RATE)


def save(name, blob):
    path = TFLITE_DIR / name
    path.write_bytes(blob)
    log.info("%-28s %8.1f KB", name, len(blob) / 1024)


def export_yamnet_backbone():
    """Convert YAMNet to TFLite with a fixed one-window input."""
    yamnet = hub.load(YAMNET_HANDLE)
    cf = yamnet.__call__.get_concrete_function(
        tf.TensorSpec([WINDOW], tf.float32)
    )
    converter = tf.lite.TFLiteConverter.from_concrete_functions([cf], yamnet)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    save("yamnet_backbone_f32.tflite", converter.convert())


def export_head():
    """Export the dense head as float32 and full-INT8 TFLite."""
    head = tf.keras.models.load_model(RESULTS_DIR / "classifier_head.keras")
    emb = np.load(RESULTS_DIR.parent / "data" / "yamnet_embeddings.npz")
    x = emb["embeddings"]
    train_frames = x[emb["folds"][emb["clip_idx"]] != 5]

    converter = tf.lite.TFLiteConverter.from_keras_model(head)
    save("head_f32.tflite", converter.convert())

    converter = tf.lite.TFLiteConverter.from_keras_model(head)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: (
        [train_frames[i:i + 1]] for i in range(0, len(train_frames), 7)
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    save("head_int8.tflite", converter.convert())


def export_dscnn():
    """Export the DS-CNN as float32 and full-INT8 TFLite."""
    dscnn = tf.keras.models.load_model(RESULTS_DIR / "dscnn.keras")
    data = np.load(FEATURES_NPZ, allow_pickle=False)
    patches = data["patches"][..., np.newaxis]
    train_patches = patches[data["folds"][data["clip_idx"]] != 5]

    converter = tf.lite.TFLiteConverter.from_keras_model(dscnn)
    save("dscnn_f32.tflite", converter.convert())

    converter = tf.lite.TFLiteConverter.from_keras_model(dscnn)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: (
        [train_patches[i:i + 1].astype(np.float32)]
        for i in range(0, len(train_patches), 7)
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    save("dscnn_int8.tflite", converter.convert())


def main():
    TFLITE_DIR.mkdir(exist_ok=True)
    export_yamnet_backbone()
    export_head()
    export_dscnn()
    log.info("All models exported to %s", TFLITE_DIR)


if __name__ == "__main__":
    main()
