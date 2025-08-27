# app.py

import streamlit as st
import pickle
import re
import numpy as np
import gdown
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# === Config ===
MAX_LEN = 200
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
MODEL_ID = "1Z2UtnBkNHA_tgsh6YN3Zs8Gmux6tBLhC"
MODEL_PATH = "lstm_model.h5"
DEFAULT_THRESHOLD = 0.5

# === Focal Loss ===
def focal_loss(gamma=2., alpha=.25):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        return -tf.reduce_mean(alpha * tf.pow(1. - pt, gamma) * tf.math.log(pt))
    return loss

# === Download model if needed ===
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            gdown.download(
                url=f"https://drive.google.com/uc?id={MODEL_ID}",
                output=MODEL_PATH,
                quiet=False
            )

# === Load assets ===
def load_assets():
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    model = load_model(MODEL_PATH, custom_objects={"loss": focal_loss()})
    return model, tokenizer

# === Preprocessing ===
def preprocess_text(text):
    import contractions, string, re
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    if not isinstance(text, str): return ""
    text = contractions.fix(text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"\w*\d\w*", '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

# === Prediction ===
def predict_comment(text, model, tokenizer):
    cleaned = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    raw_pred = model.predict(padded)[0]
    final_pred = {
        label: int(raw_pred[i] > DEFAULT_THRESHOLD)
        for i, label in enumerate(LABELS)
    }
    return final_pred

# === Streamlit UI ===
st.title("🧠 Toxic Comment Classifier (LSTM + Focal Loss)")
download_model()
model, tokenizer = load_assets()

user_input = st.text_area("Enter a comment:")
if st.button("Predict"):
    if user_input.strip():
        result = predict_comment(user_input, model, tokenizer)
        st.subheader("Prediction:")
        for label, value in result.items():
            st.write(f"{label}: {'✅' if value else '❌'}")
    else:
        st.warning("Please enter a comment.")
