import streamlit as st
import pandas as pd
import numpy as np
import re, string, contractions, os, pickle, requests
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import matplotlib.pyplot as plt

# === NLTK Setup ===
for resource in ["stopwords", "wordnet"]:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# === Focal Loss ===
def focal_loss(gamma=2., alpha=.25):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        return -tf.reduce_mean(alpha * tf.pow(1. - pt, gamma) * tf.math.log(pt))
    return loss

# === Preprocessing ===
def preprocess_text(text):
    if not isinstance(text, str): return ""
    try:
        text = contractions.fix(text)
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)
        text = re.sub(r"\w*\d\w*", '', text)
        words = [word for word in text.split() if word not in stop_words]
        lemmatized = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized)
    except Exception:
        return ""

# === Load Assets ===
@st.cache_resource
def load_assets():
    model_url = "https://drive.google.com/uc?id=1Z2UtnBkNHA_tgsh6YN3Zs8Gmux6tBLhC"
    model_path = "lstm_model.h5"
    tokenizer_path = "tokenizer.pkl"

    if not os.path.exists(model_path):
        r = requests.get(model_url)
        with open(model_path, "wb") as f:
            f.write(r.content)

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    try:
        model = load_model(model_path, custom_objects={"loss": focal_loss()})
    except Exception:
        model = load_model(model_path, compile=False)

    return tokenizer, model

tokenizer, model = load_assets()

# === Prediction ===
def predict_comment(text, model, tokenizer, toxic_threshold=0.6, default_threshold=0.5):
    cleaned = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=200)
    pred = model.predict(padded)[0]
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    scores = dict(zip(labels, pred))
    binary = {}
    for label in labels:
        t = toxic_threshold if label == "toxic" else default_threshold
        binary[label] = int(scores[label] >= t)
    return scores, binary

# === UI ===
st.set_page_config(page_title="Toxicity Detector", layout="centered")
st.title("🧪 Multi-label Toxicity Classifier")

with st.expander("📘 About the Model"):
    st.markdown("""
    This classifier uses an optimized LSTM architecture trained on the Jigsaw dataset.
    It predicts six toxicity categories:
    - **Toxic**
    - **Severe Toxic**
    - **Obscene**
    - **Threat**
    - **Insult**
    - **Identity Hate**
    """)

st.markdown("💡 *Tip: For best results, enter short, informal comments (e.g., social media replies, forum posts).*")

# === Threshold Sliders ===
threshold = st.slider("⚙️ General threshold for all labels", 0.0, 1.0, 0.5, 0.05)
toxic_threshold = st.slider("🧪 Adjust 'toxic' sensitivity", 0.0, 1.0, 0.6, 0.05)

# === Single Comment ===
user_input = st.text_area("🔍 Enter a comment", placeholder="Type something like 'You're such a loser!' or 'I love this!'")

if st.button("Analyze Text"):
    if user_input.strip():
        scores, binary = predict_comment(user_input, model, tokenizer, toxic_threshold, threshold)
        st.subheader("📊 Toxicity Breakdown")

        fig, ax = plt.subplots()
        ax.barh(list(scores.keys()), list(scores.values()), color="salmon")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Toxicity Score")
        ax.set_title("Toxicity Prediction")
        for i, (label, score) in enumerate(scores.items()):
            ax.text(score + 0.02, i, f"{score:.2f}", va='center')
        st.pyplot(fig)

        st.markdown("### ✅ Binary Classification")
        st.write(binary)
    else:
        st.warning("Please enter a comment to analyze.")

# === CSV Upload ===
st.markdown("---")
st.subheader("📁 Batch Prediction via CSV")
uploaded_file = st.file_uploader("Upload CSV with a 'comment_text' column", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        if "comment_text" not in df.columns:
            st.error("CSV must contain a 'comment_text' column.")
        else:
            df["comment_text"] = df["comment_text"].astype(str).fillna("")
            df["cleaned"] = df["comment_text"].apply(preprocess_text)

            valid_df = df[df["cleaned"].str.strip() != ""].copy()
            skipped_count = len(df) - len(valid_df)

            if valid_df.empty:
                st.warning("No valid comments found after preprocessing.")
            else:
                seqs = tokenizer.texts_to_sequences(valid_df["cleaned"])
                padded = pad_sequences(seqs, maxlen=200)
                preds = model.predict(padded)

                labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
                for i, label in enumerate(labels):
                    t = toxic_threshold if label == "toxic" else threshold
                    valid_df[label + "_score"] = preds[:, i]
                    valid_df[label] = (preds[:, i] >= t).astype(int)

                st.success(f"✅ Prediction complete. Skipped {skipped_count} empty or invalid rows.")
                st.dataframe(valid_df[["comment_text"] + [label + "_score" for label in labels] + labels])
    except Exception as e:
        st.error(f"Error processing file: {e}")
