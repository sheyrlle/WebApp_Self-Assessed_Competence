import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import os
import gdown

# gdown
VECTOR_PATH = "C:\\Users\\Sherylle Rose\\Desktop\\rfmodeloct26\\vectorizer.pkl"
MODEL_PATH = "C:\\Users\\Sherylle Rose\\Desktop\\rfmodeloct26\\rf_model.pkl"

VECTOR_FILE_ID = "1PdTdJaCyULawJ_Nvo22Wy2t7hYe5Zmns"
MODEL_FILE_ID = "10YDoNv8PAYoy-Pp5Jp2c4B9653yny3-a"

VECTOR_URL = f"https://drive.google.com/uc?id={VECTOR_FILE_ID}"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"

# download file if not exist
if not os.path.exists(VECTOR_PATH):
    gdown.download(VECTOR_URL, VECTOR_PATH, quiet=False)
if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# load pkl files
try:
    vectorizer = joblib.load(VECTOR_PATH)
    if not hasattr(vectorizer, "transform"):
        raise TypeError(f"Loaded object is not a vectorizer! Found type: {type(vectorizer)}")
except Exception as e:
    st.error(f"Error loading vectorizer: {e}")
    vectorizer = None

try:
    model = joblib.load(MODEL_PATH)
    if not hasattr(model, "predict"):
        raise TypeError(f"Loaded object is not a model! Found type: {type(model)}")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# streamlit
st.set_page_config(page_title="Sentiment Analysis", layout="wide")

# history file
history_file = "sentiment_history.csv"
if not os.path.exists(history_file):
    df_init = pd.DataFrame(columns=["Date", "Time", "Response", "Classification"])
    df_init.to_csv(history_file, index=False)

# sidebar
with st.sidebar:
    st.write("The app analyzes students’ competence in Python, Java, and C based on English-language sentiments.")
    page = st.radio("Go to", ["Home Page", "History"])

    if st.button("Delete History"):
        if os.path.exists(history_file):
            os.remove(history_file)
            st.success("History deleted successfully.")
        else:
            st.info("No history to delete.")

# home page
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
        elif vectorizer is None or model is None:
            st.error("Model or vectorizer not loaded. Check your .pkl files.")
        else:
            try:
                comment_vector = vectorizer.transform([comment])
                sentiment = model.predict(comment_vector)[0]

                label_map = {0: "Weak Competence", 1: "Normal Competence", 2: "Strong Competence"}
                sentiment_label = label_map.get(sentiment, "Unknown")

                st.success(f"Sentiment Classification: {sentiment_label}")

                now = datetime.now()
                new_row = pd.DataFrame({
                    "Date": [now.strftime("%Y-%m-%d")],
                    "Time": [now.strftime("%H:%M:%S")],
                    "Response": [comment],
                    "Classification": [sentiment_label]
                })
                new_row.to_csv(history_file, mode='a', header=False, index=False)

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    # summary
    st.write("Sentiment Distribution")
    if os.path.exists(history_file):
        data = pd.read_csv(history_file)
        if not data.empty:
            sentiment_counts = data["Classification"].value_counts()

            fig, ax = plt.subplots(figsize=(3, 2))
            ax.bar(sentiment_counts.index, sentiment_counts.values, width=0.5)
            ax.set_xlabel("")
            ax.set_ylabel("")
            plt.xticks(rotation=0, fontsize=4)
            plt.yticks(fontsize=8)
            plt.tight_layout(pad=0.2)
            st.pyplot(fig, use_container_width=False)

            total_responses = sentiment_counts.sum()
            summary_text = " | ".join([f"{label}: {count}" for label, count in sentiment_counts.items()])
            st.caption(f"Total: {total_responses} | {summary_text}")
        else:
            st.info("No responses yet.")
    else:
        st.info("No responses yet.")

# history
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


