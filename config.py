"""Shared paths and audio settings for the pipeline scripts in src/."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Input: your own audio files, one sub-folder per genre (data/raw/<GENRE>/*.mp3).
# This folder is git-ignored: audio is never committed.
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

# Generated artefacts (git-ignored).
PROCESSED_DIR = BASE_DIR / "data" / "processed"
EMBEDDINGS_FILE = PROCESSED_DIR / "embeddings_matrix.npy"
SONG_DATA_FILE = PROCESSED_DIR / "song_data.pkl"
HTML_DIR = BASE_DIR / "data" / "html"

# Static plots shown in the README.
PLOTS_DIR = BASE_DIR / "data" / "plots"

# Audio feature extraction.
SAMPLE_RATE = 22050
N_MFCC = 20
