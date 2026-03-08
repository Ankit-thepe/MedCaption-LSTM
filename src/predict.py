"""
Inference script — generate a caption for a single image using greedy search.

Usage:
    python src/predict.py \
        --model  models/best_model.keras \
        --image  path/to/xray.jpg \
        --tokenizer models/tokenizer.pkl
"""
import argparse
import os
import pickle

import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.utils import idx_to_word
from configs.config import IMAGE_SIZE, IMAGE_FEAT_DIM


def extract_single_feature(image_path: str) -> np.ndarray:
    """Extract VGG16 features from a single image file."""
    vgg = VGG16()
    extractor = Model(inputs=vgg.inputs, outputs=vgg.layers[-2].output)

    img = load_img(image_path, target_size=IMAGE_SIZE)
    arr = img_to_array(img).reshape((1, 224, 224, 3))
    arr = preprocess_input(arr)
    return extractor.predict(arr, verbose=0)


def predict_caption(model, image_feature: np.ndarray, tokenizer, max_length: int) -> str:
    """
    Greedy-search caption generation.

    Starts from 'startseq', appends predicted words one at a time
    until 'endseq' or max_length is reached.

    Args:
        model         : Trained Keras captioning model.
        image_feature : (1, 4096) VGG16 feature vector.
        tokenizer     : Fitted Keras Tokenizer.
        max_length    : Maximum caption length.

    Returns:
        str: Generated caption (without startseq/endseq tags).
    """
    in_text = "startseq"
    for _ in range(max_length):
        seq  = tokenizer.texts_to_sequences([in_text])[0]
        seq  = pad_sequences([seq], maxlen=max_length)
        yhat = model.predict([image_feature, seq], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break

    # Remove sequence markers
    caption = in_text.replace("startseq", "").replace("endseq", "").strip()
    return caption


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate caption for a medical image")
    parser.add_argument("--model",     required=True, help="Path to .keras model file")
    parser.add_argument("--image",     required=True, help="Path to input image")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.pkl")
    parser.add_argument("--max-len",   type=int, default=38)
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    model = load_model(args.model)

    print(f"Loading tokenizer from {args.tokenizer}...")
    with open(args.tokenizer, "rb") as f:
        tokenizer = pickle.load(f)

    print(f"Extracting features from {args.image}...")
    feat = extract_single_feature(args.image)

    caption = predict_caption(model, feat, tokenizer, args.max_len)
    print(f"\n📋 Generated Caption:\n  {caption}")
