"""
Configuration file for MedCaption-LSTM.
All hyperparameters and paths are defined here.
"""
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR        = os.path.join(BASE_DIR, "data")
MODELS_DIR      = os.path.join(BASE_DIR, "models")

TRAIN_IMAGE_DIR = os.path.join(DATA_DIR, "all_data", "train", "radiology", "images")
TRAIN_CSV       = os.path.join(DATA_DIR, "all_data", "train", "radiology", "traindata.csv")

FEATURES_PATH   = os.path.join(MODELS_DIR, "features.pkl")
TOKENIZER_PATH  = os.path.join(MODELS_DIR, "tokenizer.pkl")
MODEL_PATH      = os.path.join(MODELS_DIR, "best_model.keras")

# ── Model Hyperparameters ──────────────────────────────────────────────────────
EMBEDDING_DIM   = 256
LSTM_UNITS      = 256
DENSE_UNITS     = 256
DROPOUT_RATE    = 0.4
IMAGE_FEAT_DIM  = 4096   # VGG16 penultimate layer output

# ── Training ────────────────────────────────────────────────────────────────────
EPOCHS          = 20
BATCH_SIZE      = 32
VALIDATION_SPLIT = 0.2
TRAIN_SPLIT     = 0.9

# ── Image Preprocessing ────────────────────────────────────────────────────────
IMAGE_SIZE      = (224, 224)   # VGG16 input size
