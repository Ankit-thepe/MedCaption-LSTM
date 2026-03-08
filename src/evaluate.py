"""
Evaluation module — compute BLEU-1 and BLEU-2 scores on the test split.

Usage:
    python src/evaluate.py \
        --model    models/best_model.keras \
        --features models/features.pkl \
        --captions data/all_data/train/radiology/traindata.csv
"""
import argparse
import pickle

from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

from src.data_loader import (
    load_captions_from_csv, clean_mapping,
    filter_mapping_by_features
)
from src.predict import predict_caption
from src.utils import load_tokenizer
from configs.config import MODEL_PATH, FEATURES_PATH, TOKENIZER_PATH, TRAIN_CSV


def evaluate(model_path, features_path, captions_path, tokenizer_path, test_split=0.1):
    from tensorflow.keras.models import load_model
    from sklearn.model_selection import train_test_split

    print("Loading artifacts...")
    model     = load_model(model_path)
    tokenizer = load_tokenizer(tokenizer_path)
    max_length = max(
        len(seq)
        for seq in tokenizer.texts_to_sequences(
            [" ".join(tokenizer.word_index.keys())]
        )
    )

    with open(features_path, "rb") as f:
        features = pickle.load(f)

    mapping = load_captions_from_csv(captions_path)
    mapping = filter_mapping_by_features(mapping, features)
    clean_mapping(mapping)

    keys = list(mapping.keys())
    _, test_keys = train_test_split(keys, test_size=test_split, random_state=42)
    print(f"Evaluating on {len(test_keys)} test images...")

    actual, predicted = [], []
    for key in tqdm(test_keys, desc="Generating captions"):
        captions = mapping[key]
        y_pred   = predict_caption(model, features[key], tokenizer, max_length)

        actual.append([cap.split() for cap in captions])
        predicted.append(y_pred.split())

    bleu1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
    bleu2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))

    print(f"\n{'='*40}")
    print(f"  BLEU-1 : {bleu1:.4f}")
    print(f"  BLEU-2 : {bleu2:.4f}")
    print(f"{'='*40}")
    return bleu1, bleu2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default=MODEL_PATH)
    parser.add_argument("--features",  default=FEATURES_PATH)
    parser.add_argument("--captions",  default=TRAIN_CSV)
    parser.add_argument("--tokenizer", default=TOKENIZER_PATH)
    args = parser.parse_args()

    evaluate(args.model, args.features, args.captions, args.tokenizer)
