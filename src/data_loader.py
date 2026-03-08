"""
Data loading and preprocessing for the ROCO radiology dataset.
"""
import csv
import os
import re

from tqdm import tqdm


def load_captions_from_csv(csv_path: str) -> dict:
    """
    Parse the ROCO traindata.csv and return a mapping of
    image_id → list of captions.

    Args:
        csv_path (str): Path to traindata.csv.

    Returns:
        dict: {image_id: [caption1, ...]}
    """
    mapping = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = os.path.splitext(row["name"])[0]
            caption  = row["caption"].strip()
            if image_id not in mapping:
                mapping[image_id] = []
            mapping[image_id].append(caption)
    return mapping


def clean_caption(caption: str) -> str:
    """
    Normalise a single caption:
    - Lowercase
    - Remove non-alphabetic characters
    - Collapse multiple spaces
    - Wrap with startseq / endseq tokens

    Args:
        caption (str): Raw caption string.

    Returns:
        str: Cleaned caption with sequence markers.
    """
    caption = caption.lower()
    caption = re.sub(r"[^a-z\s]", "", caption)
    caption = re.sub(r"\s+", " ", caption).strip()
    tokens  = [w for w in caption.split() if len(w) > 1]
    return "startseq " + " ".join(tokens) + " endseq"


def clean_mapping(mapping: dict) -> None:
    """
    Clean all captions in the mapping in-place.

    Args:
        mapping (dict): {image_id: [raw_captions]}.
    """
    for key, captions in mapping.items():
        mapping[key] = [clean_caption(c) for c in captions]


def filter_mapping_by_features(mapping: dict, features: dict) -> dict:
    """
    Keep only image IDs that have precomputed VGG16 features.

    Args:
        mapping  (dict): Full caption mapping.
        features (dict): Precomputed {image_id: feature_vector}.

    Returns:
        dict: Filtered mapping.
    """
    return {k: v for k, v in mapping.items() if k in features}


def get_all_captions(mapping: dict) -> list:
    """Flatten the mapping into a list of all caption strings."""
    return [cap for captions in mapping.values() for cap in captions]


if __name__ == "__main__":
    import pickle
    from configs.config import TRAIN_CSV, FEATURES_PATH

    print("Loading captions...")
    mapping = load_captions_from_csv(TRAIN_CSV)
    print(f"  Raw: {len(mapping)} images")

    print("Loading features...")
    with open(FEATURES_PATH, "rb") as f:
        features = pickle.load(f)
    print(f"  Features: {len(features)} images")

    mapping = filter_mapping_by_features(mapping, features)
    clean_mapping(mapping)

    all_caps = get_all_captions(mapping)
    print(f"  Filtered: {len(mapping)} images, {len(all_caps)} captions")
    print(f"  Sample: {all_caps[0]}")
