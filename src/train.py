"""
Training script for MedCaption-LSTM.

Usage:
    python src/train.py \
        --features models/features.pkl \
        --captions data/all_data/train/radiology/traindata.csv \
        --epochs 20 --batch-size 32 \
        --output models/best_model.keras
"""
import argparse
import os
import pickle

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from configs.config import (
    BATCH_SIZE, EPOCHS, TOKENIZER_PATH, MODEL_PATH, MODELS_DIR
)
from src.data_loader import (
    load_captions_from_csv, clean_mapping,
    filter_mapping_by_features, get_all_captions
)
from src.model import build_caption_model
from src.utils import build_tokenizer


def data_generator(keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    """
    Infinite generator that yields batches of (image_feat, seq_input) → next_word.
    """
    X1, X2, y = [], [], []
    n = 0
    while True:
        for key in keys:
            for caption in mapping[key]:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq  = pad_sequences([seq[:i]], maxlen=max_length)[0]
                    out_seq = to_categorical([seq[i]], num_classes=vocab_size)[0]
                    X1.append(features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
            n += 1
            if n == batch_size and len(X1) > 0:
                yield (
                    tf.convert_to_tensor(np.array(X1), dtype=tf.float32),
                    tf.convert_to_tensor(np.array(X2), dtype=tf.float32),
                ), tf.convert_to_tensor(np.array(y), dtype=tf.float32)
                X1, X2, y = [], [], []
                n = 0


def train(features_path, captions_path, epochs, batch_size, output_path):
    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading features...")
    with open(features_path, "rb") as f:
        features = pickle.load(f)

    print("Loading captions...")
    mapping = load_captions_from_csv(captions_path)
    mapping = filter_mapping_by_features(mapping, features)
    clean_mapping(mapping)

    all_captions = get_all_captions(mapping)
    tokenizer    = build_tokenizer(all_captions, save_path=TOKENIZER_PATH)
    vocab_size   = len(tokenizer.word_index) + 1
    max_length   = max(len(c.split()) for c in all_captions)
    print(f"  Vocabulary: {vocab_size} | Max length: {max_length}")

    # ── Split ──────────────────────────────────────────────────────────────
    keys = list(mapping.keys())
    train_keys, val_keys = train_test_split(keys, test_size=0.2, random_state=42)
    print(f"  Train: {len(train_keys)} | Val: {len(val_keys)}")

    # ── Model ──────────────────────────────────────────────────────────────
    model = build_caption_model(vocab_size, max_length)
    model.summary()

    # ── Training ───────────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    steps  = max(1, len(train_keys) // batch_size)

    callbacks = [
        ModelCheckpoint(output_path, save_best_only=True, monitor="loss", verbose=1),
        EarlyStopping(patience=3, restore_best_weights=True, verbose=1),
    ]

    train_gen = data_generator(
        train_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size
    )
    model.fit(
        train_gen,
        epochs=epochs,
        steps_per_epoch=steps,
        callbacks=callbacks,
        verbose=1,
    )
    print(f"Model saved → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MedCaption-LSTM")
    parser.add_argument("--features",   default="models/features.pkl")
    parser.add_argument("--captions",   default="data/all_data/train/radiology/traindata.csv")
    parser.add_argument("--epochs",     type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--output",     default=MODEL_PATH)
    args = parser.parse_args()

    train(args.features, args.captions, args.epochs, args.batch_size, args.output)
