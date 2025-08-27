# app.py

import streamlit as st
import pickle
import re
import numpy as np
import gdown
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Constants
MAX_LEN = 300
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
MODEL_ID = "1Z2UtnBkNHA_tgsh6YN3Zs8Gmux6tBLhC"
MODEL_PATH = "lstm_model.h5"
DEFAULT_THRESHOLD = 0.5

# Download model from Google Drive if not present
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(id=MODEL_ID, output=MODEL_PATH, quiet=False)

# Load tokenizer
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

# Text cleaning
def clean_text(text):
    return re.sub(r"[^a-zA-Z0-9\s]", "", str(text).lower())

# Prediction function
def predict_comment(text, model, tokenizer):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    raw_pred = model.predict(padded)[0]
    final_pred = {
        label: int(raw_pred[i] > DEFAULT_THRESHOLD)
        for i, label in enumerate(LABELS)
    }
    return final_pred

# App logic
st.title("🧠 Toxic Comment Classifier (LSTM)")
download_model()
model = load_model(MODEL_PATH)
tokenizer = load_tokenizer()

user_input = st.text_area("Enter a comment:")
if st.button("Predict"):
    if user_input.strip():
        result = predict_comment(user_input, model, tokenizer)
        st.subheader("Prediction:")
        for label, value in result.items():
            st.write(f"{label}: {'✅' if value else '❌'}")
    else:
        st.warning("Please enter a comment.")
