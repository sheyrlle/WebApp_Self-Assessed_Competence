import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import gdown
from datetime import datetime

# -----------------------------
# File paths & Google Drive IDs
# -----------------------------
VECTOR_PATH = "vectorizer.pkl"
MODEL_PATH = "rf_model.pkl"

VECTOR_FILE_ID = "1PdTdJaCyULawJ_Nvo22Wy2t7hYe5Zmns"
MODEL_FILE_ID = "10YDoNv8PAYoy-Pp5Jp2c4B9653yny3-a"

VECTOR_URL = f"https://drive.google.com/uc?id={VECTOR_FILE_ID}"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"

# -----------------------------
# Download files if missing
# -----------------------------
if not os.path.exists(VECTOR_PATH):
    gdown.download(VECTOR_URL, VECTOR_PATH, quiet=False)

if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# -----------------------------
# Load vectorizer
# -----------------------------
try:
    with open(VECTOR_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    # Check type
    if not hasattr(vectorizer, "transform"):
        st.error("Loaded vectorizer is not a vectorizer object. Check your file.")
        st.stop()
except Exception as e:
    st.error(f"Error loading vectorizer: {e}")
    st.stop()

# -----------------------------
# Load model
# -----------------------------
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    # Check type
    if not hasattr(model, "predict"):
        st.error("Loaded model is not a predictive model. Check your file.")
        st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(page_title="Self-Assessed Competence Analyzer", layout="wide")

# -----------------------------
# History file
# -----------------------------
history_file = "sentiment_history.csv"
if not os.path.exists(history_file):
    df_init = pd.DataFrame(columns=["Date", "Time", "Response", "Classification"])
    df_init.to_csv(history_file, index=False)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.write("Analyze students’ self-assessed competence in Python, Java, and C.")
    page = st.radio("Go to", ["Home Page", "History"])
    
    if st.button("Delete History"):
        if os.path.exists(history_file):
            os.remove(history_file)
            st.success("History deleted successfully.")
        else:
            st.info("No history to delete.")

# -----------------------------
# Home Page
# -----------------------------
if page == "Home Page":
    col1, col2 = st.columns([1, 11])
    with col1:
        st.image("ccitlogo.png", width=110)
    with col2:
        st.title("Self-Assessed Competence Sentiment Analyzer")

    comment = st.text_area("Enter your sentiments about your competence in Python, Java, and C:")

    if st.button("Analyze Sentiment"):
        if comment.strip() == "":
            st.warning("Please enter some text first.")
        else:
            # Transform input
            comment_vector = vectorizer.transform([comment])
            sentiment = model.predict(comment_vector)[0]

            label_map = {0: "Weak Competence", 1: "Normal Competence", 2: "Strong Competence"}
            sentiment_label = label_map.get(sentiment, "Unknown")

            st.success(f"Sentiment Classification: {sentiment_label}")

            # Save history
            now = datetime.now()
            new_row = pd.DataFrame({
                "Date": [now.strftime("%Y-%m-%d")],
                "Time": [now.strftime("%H:%M:%S")],
                "Response": [comment],
                "Classification": [sentiment_label]
            })
            new_row.to_csv(history_file, mode='a', header=False, index=False)

    # Graph summary
    st.write("Sentiment Distribution")
    if os.path.exists(history_file):
        data = pd.read_csv(history_file)
        if not data.empty:
            sentiment_counts = data["Classification"].value_counts()
            fig, ax = plt.subplots(figsize=(2.5, 1.2))
            ax.bar(sentiment_counts.index, sentiment_counts.values, width=0.5)
            ax.set_xlabel("")
            ax.set_ylabel("")
            plt.xticks(rotation=2, fontsize=4)
            plt.yticks(fontsize=5)
            plt.tight_layout(pad=0.2)
            st.pyplot(fig, use_container_width=False)

            total_responses = sentiment_counts.sum()
            summary_text = " | ".join([f"{label}: {count}" for label, count in sentiment_counts.items()])
            st.caption(f"Total: {total_responses} | {summary_text}")
        else:
            st.info("No responses yet.")
    else:
        st.info("No responses yet.")

# -----------------------------
# History Page
# -----------------------------
elif page == "History":
    st.write("History of Students' Responses")
    if os.path.exists(history_file):
        data = pd.read_csv(history_file)
        if not data.empty:
            st.dataframe(data, use_container_width=True)
        else:
            st.info("No sentiment history found yet.")
    else:
        st.info("No sentiment history file found yet.")
