"""
Flask Web Application for MedCaption-LSTM.

Routes:
    GET  /           → Upload form
    POST /predict    → Generate caption for uploaded image

Run:
    cd app && python app.py
    Open http://localhost:5000
"""
import io
import os
import pickle
import sys

import numpy as np
from flask import Flask, jsonify, render_template, request
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import idx_to_word
from configs.config import MODEL_PATH, TOKENIZER_PATH, IMAGE_SIZE

app = Flask(__name__)

# ── Load model & tokenizer at startup ─────────────────────────────────────────
print("Loading VGG16 feature extractor...")
_vgg        = VGG16()
_extractor  = Model(inputs=_vgg.inputs, outputs=_vgg.layers[-2].output)

print(f"Loading caption model from {MODEL_PATH}...")
try:
    _caption_model = load_model(MODEL_PATH)
    _tokenizer_loaded = True
except Exception as e:
    print(f"  WARNING: Could not load model — {e}")
    _caption_model = None
    _tokenizer_loaded = False

print(f"Loading tokenizer from {TOKENIZER_PATH}...")
try:
    with open(TOKENIZER_PATH, "rb") as f:
        _tokenizer = pickle.load(f)
    _max_length = max(len(s.split()) for s in _tokenizer.word_index)
except Exception as e:
    print(f"  WARNING: Could not load tokenizer — {e}")
    _tokenizer  = None
    _max_length = 38


def _extract_feature(image_bytes: bytes) -> np.ndarray:
    img = load_img(io.BytesIO(image_bytes), target_size=IMAGE_SIZE)
    arr = img_to_array(img).reshape((1, 224, 224, 3))
    return _extractor.predict(preprocess_input(arr), verbose=0)


def _generate_caption(feat: np.ndarray) -> str:
    if _caption_model is None or _tokenizer is None:
        return "Model not loaded. Please train and place best_model.keras in models/."
    in_text = "startseq"
    for _ in range(_max_length):
        seq  = _tokenizer.texts_to_sequences([in_text])[0]
        seq  = pad_sequences([seq], maxlen=_max_length)
        yhat = _caption_model.predict([feat, seq], verbose=0)
        word = idx_to_word(int(np.argmax(yhat)), _tokenizer)
        if word is None or word == "endseq":
            break
        in_text += " " + word
    return in_text.replace("startseq", "").strip()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file  = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    image_bytes = file.read()
    feat        = _extract_feature(image_bytes)
    caption     = _generate_caption(feat)
    return jsonify({"caption": caption})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
