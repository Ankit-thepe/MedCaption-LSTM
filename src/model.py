"""
VGG16 Encoder + LSTM Decoder — MedCaption-LSTM Architecture

Image branch:
    4096-dim VGG16 features → Dropout(0.4) → Dense(256, relu)

Text branch:
    token sequence → Embedding(vocab, 256) → Dropout(0.4) → LSTM(256)

Decoder:
    add([image_branch, text_branch]) → Dense(256, relu) → Dense(vocab_size, softmax)
"""

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Embedding, Dropout, add
)


def build_feature_extractor() -> Model:
    """
    Load VGG16 pretrained on ImageNet.
    Remove the final classification layer to expose 4096-dim features.

    Returns:
        Model: VGG16 feature extractor (frozen weights).
    """
    vgg = VGG16()
    # Remove the top Dense(1000) classification layer
    extractor = Model(inputs=vgg.inputs, outputs=vgg.layers[-2].output)
    extractor.trainable = False
    return extractor


def build_caption_model(vocab_size: int, max_length: int) -> Model:
    """
    Build the VGG16 encoder + LSTM decoder captioning model.

    Args:
        vocab_size  (int): Total vocabulary size (number of unique tokens + 1).
        max_length  (int): Maximum caption length (for padding/sequence input).

    Returns:
        Model: Compiled Keras model ready for training.
    """
    # ── Image Feature Branch ────────────────────────────────────────────────
    img_input  = Input(shape=(4096,), name="image_features")
    img_drop   = Dropout(0.4)(img_input)
    img_dense  = Dense(256, activation="relu", name="img_dense")(img_drop)

    # ── Text / Sequence Branch ──────────────────────────────────────────────
    seq_input  = Input(shape=(max_length,), name="sequence_input")
    seq_embed  = Embedding(vocab_size, 256, mask_zero=True, name="embedding")(seq_input)
    seq_drop   = Dropout(0.4)(seq_embed)
    seq_lstm   = LSTM(256, name="lstm")(seq_drop)

    # ── Decoder ─────────────────────────────────────────────────────────────
    merged     = add([img_dense, seq_lstm], name="feature_merge")
    dec_dense  = Dense(256, activation="relu", name="dec_dense")(merged)
    output     = Dense(vocab_size, activation="softmax", name="output")(dec_dense)

    model = Model(
        inputs=[img_input, seq_input],
        outputs=output,
        name="MedCaption_LSTM"
    )
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model


if __name__ == "__main__":
    # Quick sanity-check
    m = build_caption_model(vocab_size=8000, max_length=38)
    m.summary()
