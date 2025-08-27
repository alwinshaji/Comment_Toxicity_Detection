import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import matplotlib.pyplot as plt
import seaborn as sns

# === Constants ===
MAX_LEN = 300
VOCAB_SIZE = 20000
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
MODEL_PATH = 'cnn_bigru_model.h5'

# === Load Model ===
@st.cache_resource
def load_cnn_bigru_model():
    return load_model(MODEL_PATH)

model = load_cnn_bigru_model()

# === Text Cleaning ===
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

# === Tokenizer Setup ===
tokenizer = Tokenizer(num_words=VOCAB_SIZE)
# Load tokenizer from training data or fit on sample corpus
sample_corpus = ["This is a sample comment", "You are horrible", "I love this!"]
tokenizer.fit_on_texts(sample_corpus)

def preprocess(texts):
    cleaned = [clean_text(t) for t in texts]
    sequences = tokenizer.texts_to_sequences(cleaned)
    padded = pad_sequences(sequences, maxlen=MAX_LEN)
    return padded

def predict_toxicity(texts):
    X = preprocess(texts)
    preds = model.predict(X)
    return pd.DataFrame(preds, columns=LABELS)

# === Streamlit UI ===
st.set_page_config(page_title="Comment Toxicity Detection", layout="wide")
st.title("🧠 Comment Toxicity Detection Dashboard")

st.markdown("""
This app uses a fine-tuned CNN-BiGRU model to detect toxic content in user comments.
You can enter a comment below for real-time prediction, or upload a CSV file for bulk analysis.
""")

# === Sidebar ===
st.sidebar.header("📊 Model Insights")
st.sidebar.markdown("**Model:** CNN + BiGRU")
st.sidebar.markdown("**Accuracy:** ~77%")
st.sidebar.markdown("**F1 Score:** ~76%")
st.sidebar.markdown("**Labels:** Toxic, Severe Toxic, Obscene, Threat, Insult, Identity Hate")

# === Real-Time Prediction ===
st.subheader("🔍 Real-Time Comment Prediction")
user_input = st.text_area("Enter a comment:", "")
if st.button("Predict"):
    if user_input.strip():
        result = predict_toxicity([user_input])
        st.write("Prediction:")
        st.dataframe(result.style.highlight_max(axis=1))
    else:
        st.warning("Please enter a comment to analyze.")

# === Bulk Prediction ===
st.subheader("📁 Bulk Prediction via CSV")
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
st.subheader("🧪 Sample Test Cases")
sample_comments = [
    "You are amazing!",
    "I hate everything about this.",
    "Go kill yourself.",
    "This is the worst thing ever.",
    "You're a genius!"
]
sample_preds = predict_toxicity(sample_comments)
sample_df = pd.DataFrame({'Comment': sample_comments})
sample_result = pd.concat([sample_df, sample_preds], axis=1)
st.dataframe(sample_result)

# === Visualization ===
st.subheader("📈 Toxic Label Distribution (Sample)")
label_counts = sample_preds.apply(lambda x: x > 0.5).sum()
fig, ax = plt.subplots()
sns.barplot(x=label_counts.index, y=label_counts.values, palette="viridis", ax=ax)
ax.set_ylabel("Count")
ax.set_title("Toxic Label Distribution")
st.pyplot(fig)
