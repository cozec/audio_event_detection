"""Shared configuration for the audio event detection project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PLOTS_DIR = PROJECT_ROOT / "plots"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

ESC50_DIR = DATA_DIR / "ESC-50-master"
ESC50_AUDIO_DIR = ESC50_DIR / "audio"
ESC50_META_CSV = ESC50_DIR / "meta" / "esc50.csv"

EMBEDDINGS_NPZ = DATA_DIR / "yamnet_embeddings.npz"

# Step 1: 8 ESC-50 classes relevant to an edge audio-event-detection device
# (home / safety monitoring scenario).
SELECTED_CLASSES = [
    "dog",
    "siren",
    "car_horn",
    "glass_breaking",
    "crying_baby",
    "door_wood_knock",
    "clock_alarm",
    "vacuum_cleaner",
]

# YAMNet expects 16 kHz mono float32 waveforms.
SAMPLE_RATE = 16000

YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"

RANDOM_SEED = 42
