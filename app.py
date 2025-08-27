import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import os
import requests

# === Constants ===
MAX_LEN = 300
VOCAB_SIZE = 20000
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
MODEL_PATH = "cnn_bigru_model.h5"
MODEL_URL = "https://raw.githubusercontent.com/alwinshaji/Comment_Toxicity_Detection/main/cnn_bigru_model.h5"

# === Download Model if Missing ===
def download_model():
    if not os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "wb") as f:
            response = requests.get(MODEL_URL)
            f.write(response.content)

download_model()

# === Load Model ===
@st.cache_resource
def load_model_safe():
    return load_model(MODEL_PATH)

model = load_model_safe()

# === Tokenizer Setup ===
@st.cache_resource
def get_tokenizer():
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    fallback_corpus = [
        "You are amazing", "I hate this", "This is awful", "I love it", "Go away", "You're brilliant"
    ]
    tokenizer.fit_on_texts(fallback_corpus)
    return tokenizer

tokenizer = get_tokenizer()

# === Text Cleaning ===
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

# === Preprocessing ===
def preprocess(texts):
    cleaned = [clean_text(t) for t in texts]
    sequences = tokenizer.texts_to_sequences(cleaned)
    if not sequences or all(len(seq) == 0 for seq in sequences):
        return None
    padded = pad_sequences(sequences, maxlen=MAX_LEN)
    return padded

# === Label Thresholds ===
LABEL_THRESHOLDS = {
    'toxic': 0.5,
    'severe_toxic': 0.4,
    'obscene': 0.45,
    'threat': 0.3,
    'insult': 0.5,
    'identity_hate': 0.35
}

def apply_thresholds(pred_df):
    return pred_df.gt(pd.Series(LABEL_THRESHOLDS))

# === Prediction ===
def predict_toxicity(texts):
    try:
        X = preprocess(texts)
        if X is None or X.shape[1] != MAX_LEN:
            return pd.DataFrame(columns=LABELS)
        preds = model.predict(X)
        raw_df = pd.DataFrame(preds, columns=LABELS)
        binarized_df = apply_thresholds(raw_df)
        return raw_df, binarized_df
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return pd.DataFrame(columns=LABELS), pd.DataFrame(columns=LABELS)

# === UI Setup ===
st.set_page_config(page_title="🧠 Toxicity Detection", layout="wide")
st.title("🧠 Comment Toxicity Detection Dashboard")
st.markdown("""
Welcome to your toxicity detection tool powered by a CNN + BiGRU model.  
You can analyze individual comments or upload CSVs for bulk predictions.
""")

# === Sidebar ===
with st.sidebar:
    st.header("📊 Model Performance")
    st.markdown("**Architecture:** CNN + BiGRU")
    st.markdown("**Macro F1 Score:** 0.63")
    st.markdown("**Weighted F1 Score:** 0.76")
    st.markdown("**Micro F1 Score:** 0.76")
    st.markdown("**Label-wise F1 Scores:**")
    st.markdown("""
- Toxic: 0.80  
- Obscene: 0.81  
- Insult: 0.75  
- Severe Toxic: 0.47  
- Threat: 0.44  
- Identity Hate: 0.49  
""")

# === Real-Time Prediction ===
with st.expander("🔍 Real-Time Comment Prediction", expanded=True):
    st.markdown("""
Type a comment below to check for toxicity.  
Examples:
- “You are amazing!”  
- “Go kill yourself.”  
- “I hate everything about this.”
""")
    user_input = st.text_area("Enter a comment to analyze:")
    if st.button("Run Prediction"):
        if user_input.strip():
            raw, binary = predict_toxicity([user_input])
            if raw.empty:
                st.warning("Could not process the input. Try a different comment.")
            else:
                st.success("Prediction complete.")
                st.write("🔢 Raw Scores:")
                st.dataframe(raw.round(3))
                st.write("✅ Predicted Labels:")
                st.dataframe(binary)
        else:
            st.warning("Please enter a comment.")

# === Bulk Prediction ===
with st.expander("📁 Bulk Prediction via CSV"):
    uploaded_file = st.file_uploader("Upload a CSV file with a 'comment_text' column", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'comment_text' in df.columns:
            st.success(f"Loaded {len(df)} comments.")
            raw, binary = predict_toxicity(df['comment_text'].tolist())
            result_df = pd.concat([df, raw.round(3), binary], axis=1)
            st.dataframe(result_df.head(10))
        else:
            st.error("CSV must contain a 'comment_text' column.")
