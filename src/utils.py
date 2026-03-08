"""
Utility functions for MedCaption-LSTM:
  - VGG16 feature extraction
  - Tokenizer build/save/load
  - Index-to-word lookup
"""
import os
import pickle

import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer

from configs.config import IMAGE_SIZE, IMAGE_FEAT_DIM


def extract_features(image_dir: str, save_path: str = None) -> dict:
    """
    Run VGG16 over every image in image_dir and return a dict of
    {image_id: feature_vector (1, 4096)}.

    Args:
        image_dir (str): Directory containing .jpg images.
        save_path (str): Optional path to pickle the result.

    Returns:
        dict: Feature vectors keyed by image ID (no extension).
    """
    vgg  = VGG16()
    extractor = Model(inputs=vgg.inputs, outputs=vgg.layers[-2].output)

    features = {}
    for fname in tqdm(os.listdir(image_dir), desc="Extracting features"):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img_path = os.path.join(image_dir, fname)
        img      = load_img(img_path, target_size=IMAGE_SIZE)
        arr      = img_to_array(img)
        arr      = arr.reshape((1, *arr.shape))
        arr      = preprocess_input(arr)
        feat     = extractor.predict(arr, verbose=0)
        image_id = os.path.splitext(fname)[0]
        features[image_id] = feat

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(features, f)
        print(f"Saved features → {save_path}")

    return features


def build_tokenizer(captions: list, save_path: str = None) -> Tokenizer:
    """
    Fit a Keras Tokenizer on the list of caption strings.

    Args:
        captions  (list): All caption strings.
        save_path (str):  Optional path to pickle the tokenizer.

    Returns:
        Tokenizer: Fitted tokenizer.
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(tokenizer, f)
        print(f"Saved tokenizer → {save_path}")

    return tokenizer


def load_tokenizer(path: str) -> Tokenizer:
    with open(path, "rb") as f:
        return pickle.load(f)


def idx_to_word(index: int, tokenizer: Tokenizer):
    """Return the word corresponding to a token index."""
    for word, idx in tokenizer.word_index.items():
        if idx == index:
            return word
    return None


if __name__ == "__main__":
    import argparse
    from configs.config import TRAIN_IMAGE_DIR, FEATURES_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-features", action="store_true")
    parser.add_argument("--image-dir",  default=TRAIN_IMAGE_DIR)
    parser.add_argument("--output",     default=FEATURES_PATH)
    args = parser.parse_args()

    if args.extract_features:
        extract_features(args.image_dir, save_path=args.output)
