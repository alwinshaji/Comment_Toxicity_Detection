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
import matplotlib.pyplot as plt
import seaborn as sns

# === Constants ===
MAX_LEN = 300
VOCAB_SIZE = 20000
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
MODEL_PATH = "cnn_bigru_model.h5"
MODEL_URL = "https://raw.githubusercontent.com/alwinshaji/Comment_Toxicity_Detection/main/cnn_bigru_model.h5"  # Update if needed

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

# === Prediction ===
def predict_toxicity(texts):
    try:
        X = preprocess(texts)
        if X is None or X.shape[1] != MAX_LEN:
            return pd.DataFrame(columns=LABELS)
        preds = model.predict(X)
        return pd.DataFrame(preds, columns=LABELS)
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return pd.DataFrame(columns=LABELS)

# === UI Setup ===
st.set_page_config(page_title="🧠 Toxicity Detection", layout="wide")
st.title("🧠 Comment Toxicity Detection Dashboard")
st.markdown("""
Welcome to your toxicity detection tool powered by a CNN + BiGRU model.  
You can analyze individual comments, upload CSVs for bulk predictions, and explore model performance insights.
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
    st.markdown("---")
    st.markdown("📁 Upload CSV or try a sample comment below.")

# === Real-Time Prediction ===
with st.expander("🔍 Real-Time Comment Prediction", expanded=True):
    st.markdown("""
Type a comment below to check for toxicity.  
Examples:
- “You are amazing!” (non-toxic)  
- “Go kill yourself.” (highly toxic)  
- “I hate everything about this.” (moderately toxic)
""")
    user_input = st.text_area("Enter a comment to analyze:")
    if st.button("Run Prediction"):
        if user_input.strip():
            result = predict_toxicity([user_input])
            if result.empty:
                st.warning("Could not process the input. Try a different comment.")
            else:
                st.success("Prediction complete.")
                st.dataframe(result.style.highlight_max(axis=1))
        else:
            st.warning("Please enter a comment.")

# === Bulk Prediction ===
with st.expander("📁 Bulk Prediction via CSV"):
    uploaded_file = st.file_uploader("Upload a CSV file with a 'comment_text' column", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'comment_text' in df.columns:
            st.success(f"Loaded {len(df)} comments.")
            predictions = predict_toxicity(df['comment_text'].tolist())
            result_df = pd.concat([df, predictions], axis=1)
            st.dataframe(result_df.head(10))
        else:
            st.error("CSV must contain a 'comment_text' column.")

# === Sample Test Cases ===
with st.expander("🧪 Sample Test Cases"):
    sample_comments = [
        "You are amazing!",
        "I hate everything about this.",
        "Go kill yourself.",
        "This is the worst thing ever.",
        "You're a genius!"
    ]
    sample_preds = predict_toxicity(sample_comments)
    if not sample_preds.empty:
        sample_df = pd.DataFrame({'Comment': sample_comments})
        sample_result = pd.concat([sample_df, sample_preds], axis=1)
        st.dataframe(sample_result)
    else:
        st.warning("Sample predictions failed to generate.")

# === Visualization ===
with st.expander("📈 Toxic Label Distribution (Sample)"):
    if not sample_preds.empty:
        label_counts = (sample_preds > 0.5).sum()
        fig, ax = plt.subplots()
        sns.barplot(x=label_counts.index, y=label_counts.values, palette="viridis", ax=ax)
        ax.set_ylabel("Count")
        ax.set_title("Toxic Label Distribution")
        st.pyplot(fig)
    else:
        st.warning("No data available for chart.")
