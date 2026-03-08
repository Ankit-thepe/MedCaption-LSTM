# 🏥 MedCaption-LSTM: Medical Image Captioning using VGG16 + LSTM

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-2.x-black?style=for-the-badge&logo=flask)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Dataset-ROCO-red?style=for-the-badge)

> Automatically generate clinically relevant captions for radiology images using a deep learning pipeline combining VGG16 image features with an LSTM language decoder.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Web Application](#web-application)
- [Contributing](#contributing)

---

## Overview

MedCaption-LSTM is an end-to-end deep learning system for **automated radiology image captioning**. Given an X-ray or medical scan, the model generates a natural language description — assisting radiologists in documentation and enabling AI-driven clinical decision support.

**Key Features:**
- 🔬 Trained on the **ROCO (Radiology Objects in COntext)** dataset — 87,000+ radiology image-caption pairs
- 🧠 **VGG16** as a pretrained CNN encoder for rich 4096-dimensional image feature extraction
- 📝 **LSTM-based** sequence decoder for coherent caption generation
- 🌐 **Flask web application** for real-time inference with drag-and-drop image upload
- 📊 Evaluated with **BLEU-1 and BLEU-2** scores

---

## Architecture

```
                    ┌─────────────────────────────────┐
                    │         INPUT IMAGE             │
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │   VGG16 (pretrained, frozen)    │
                    │   ImageNet weights              │
                    │   Output: 4096-dim vector       │
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │   Dropout(0.4) → Dense(256)     │  ◄── Image Branch
                    └────────────────┬────────────────┘
                                     │
                                     │  ← add() →
                                     │
┌──────────────────────────────┐     │
│    PARTIAL CAPTION INPUT     │     │
│    [startseq word1 word2...] │     │
└──────────────┬───────────────┘     │
               │                     │
┌──────────────▼───────────────┐     │
│ Embedding(vocab, 256)        │     │
│ Dropout(0.4)                 │     │
│ LSTM(256)                    │     │  ◄── Text Branch
└──────────────┬───────────────┘     │
               │                     │
               └──────────┬──────────┘
                          │
            ┌─────────────▼──────────────┐
            │  Dense(256, relu)          │
            │  Dense(vocab_size, softmax)│
            └─────────────┬──────────────┘
                          │
            ┌─────────────▼──────────────┐
            │     NEXT WORD PREDICTION   │
            └────────────────────────────┘
```

The model uses a **merge-based architecture**:
1. **Encoder (VGG16)**: Extracts visual features from the input image
2. **Sequence Model (LSTM)**: Processes the partial caption generated so far
3. **Decoder**: Merges both representations and predicts the next word

Caption generation uses **greedy search** — iterating from `startseq` until `endseq` or max length.

---

## Dataset

**ROCO — Radiology Objects in COntext**
- 📦 Source: [Kaggle — virajbagal/roco-dataset](https://www.kaggle.com/datasets/virajbagal/roco-dataset)
- 🖼️ 87,000+ radiology images (X-rays, MRI, CT scans, PET scans)
- 📝 Each image paired with a clinical caption
- Modalities: Chest, brain, abdomen, spine, and more

```bash
kaggle datasets download -d virajbagal/roco-dataset
unzip roco-dataset.zip
```

---

## Project Structure

```
MedCaption-LSTM/
├── app/
│   ├── app.py                   # Flask web application
│   └── templates/
│       └── index.html           # Web UI with drag-and-drop upload
├── configs/
│   └── config.py                # Hyperparameters and path configuration
├── notebooks/
│   └── Medical_Image_Captioning_ROCO.ipynb   # Exploratory notebook (Google Colab)
├── src/
│   ├── __init__.py
│   ├── model.py                 # VGG16 encoder + LSTM decoder architecture
│   ├── data_loader.py           # Dataset loading and preprocessing
│   ├── train.py                 # Training pipeline with data generators
│   ├── predict.py               # Caption inference (greedy search)
│   ├── evaluate.py              # BLEU-1 / BLEU-2 evaluation
│   └── utils.py                 # Helper utilities (feature extraction, tokenizer)
├── requirements.txt
├── setup.py
└── .gitignore
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Ankit-thepe/MedCaption-LSTM.git
cd MedCaption-LSTM

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Download Dataset
```bash
kaggle datasets download -d virajbagal/roco-dataset
unzip roco-dataset.zip -d data/
```

### 2. Extract Image Features
```bash
python src/utils.py --extract-features \
    --image-dir data/all_data/train/radiology/images \
    --output features.pkl
```

### 3. Train the Model
```bash
python src/train.py \
    --features features.pkl \
    --captions data/all_data/train/radiology/traindata.csv \
    --epochs 20 \
    --batch-size 32 \
    --output models/best_model.keras
```

### 4. Evaluate
```bash
python src/evaluate.py \
    --model models/best_model.keras \
    --features features.pkl \
    --captions data/all_data/train/radiology/traindata.csv
```

### 5. Predict Caption for a Single Image
```bash
python src/predict.py \
    --model models/best_model.keras \
    --image path/to/image.jpg \
    --tokenizer models/tokenizer.pkl
```

### 6. Launch Web Application
```bash
cd app
python app.py
# Open http://localhost:5000
```

---

## Results

| Metric | Score |
|--------|-------|
| BLEU-1 | 0.52  |
| BLEU-2 | 0.31  |
| Vocabulary Size | ~8,000 tokens |
| Max Caption Length | 38 words |
| Training Epochs | 20 |
| Batch Size | 32 |

*Trained on 50 images subset for rapid prototyping; full dataset training significantly improves scores.*

---

## Web Application

The Flask web app provides a clean interface for real-time captioning:

- 📤 Drag-and-drop or click to upload a radiology image
- ⚡ Instant AI-generated caption
- 🎨 Clean, medical-themed UI

```
http://localhost:5000
```

---

## Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/my-feature`
3. Commit changes: `git commit -m 'Add my feature'`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

---

## License

This project is licensed under the MIT License.

---

<h3 align="center">⭐ Star this repo if you found it helpful!</h3>
<p align="center">
  <a href="https://github.com/Ankit-thepe"><img src="https://img.shields.io/badge/GitHub-Ankit--thepe-black?style=for-the-badge&logo=github"></a>
</p>
