import streamlit as st
import pandas as pd
import numpy as np
import re, string, contractions, os, requests, pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Ensure NLTK resources are available
for resource in ["stopwords", "wordnet"]:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource)

# Download model from Google Drive if missing
def download_if_missing(url, local_path):
    if not os.path.exists(local_path):
        r = requests.get(url)
        with open(local_path, 'wb') as f:
            f.write(r.content)

@st.cache_resource
def load_assets():
    model_url = "https://drive.google.com/uc?id=1Z2UtnBkNHA_tgsh6YN3Zs8Gmux6tBLhC"
    model_path = "lstm_model.h5"
    tokenizer_path = "tokenizer.pkl"

    download_if_missing(model_url, model_path)

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    model = load_model(model_path)
    return tokenizer, model

tokenizer, model = load_assets()

# Preprocessing
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    text = contractions.fix(text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"\w*\d\w*", '', text)
    words = [word for word in text.split() if word not in stop_words]
    lemmatized = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized)

# Prediction
def predict_comment(text, model, tokenizer):
    cleaned = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=100)
    pred = model.predict(padded)[0][0]
    return "Toxic" if pred > 0.5 else "Non-toxic", float(pred)

# UI
st.set_page_config(page_title="Toxicity Detector", layout="centered")
st.title("🧪 Comment Toxicity Classifier")
st.markdown("Enter a comment or upload a CSV to detect toxicity.")

# Text input
user_input = st.text_area("🔍 Enter a comment", placeholder="Type something...")

if st.button("Analyze Text"):
    if user_input.strip():
        label, score = predict_comment(user_input, model, tokenizer)
        st.success(f"Prediction: **{label}** ({score:.2f})")
    else:
        st.warning("Please enter a comment to analyze.")

# CSV upload
st.markdown("---")
st.subheader("📁 Batch Prediction via CSV")
uploaded_file = st.file_uploader("Upload CSV with a 'comment' column", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if "comment" not in df.columns:
            st.error("CSV must contain a 'comment' column.")
        else:
            df["cleaned"] = df["comment"].apply(preprocess_text)
            seqs = tokenizer.texts_to_sequences(df["cleaned"])
            padded = pad_sequences(seqs, maxlen=100)
            preds = model.predict(padded).flatten()
            df["toxicity_score"] = preds
            df["label"] = np.where(preds > 0.5, "Toxic", "Non-toxic")
            st.dataframe(df[["comment", "label", "toxicity_score"]])
    except Exception as e:
        st.error(f"Error processing file: {e}")
