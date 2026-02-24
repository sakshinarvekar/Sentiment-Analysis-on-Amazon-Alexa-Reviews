import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="Alexa Sentiment Classifier", page_icon="🤖", layout="centered")

st.title("🤖 Alexa Review Sentiment Classifier (No joblib)")
st.write("This app trains the model from `amazon_alexa.tsv` and then predicts sentiment for new reviews.")

@st.cache
def train_model():
    # Load dataset
    df = pd.read_csv("amazon_alexa.tsv", sep="\t")
    df = df.dropna(subset=["verified_reviews", "feedback"]).copy()
    df["verified_reviews"] = df["verified_reviews"].astype(str)

    X = df["verified_reviews"]
    y = df["feedback"]

    # Split (stratified due to imbalance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build pipeline
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    model.fit(X_train, y_train)

    # quick evaluation (optional)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, digits=4, output_dict=False)
    cm = confusion_matrix(y_test, y_pred)

    return model, report, cm, df.shape[0]

with st.spinner("Training model from amazon_alexa.tsv..."):
    model, report, cm, n_rows = train_model()

st.success(f"Model trained successfully on {n_rows} reviews ✅")

with st.expander("See evaluation (test set)"):
    st.text(report)
    st.write("Confusion Matrix (rows=true, cols=pred):")
    st.write(cm)

# User input
text = st.text_area("Enter a review to classify:", height=160)

if st.button("Predict"):
    if not text.strip():
        st.warning("Please type a review first.")
    else:
        pred = model.predict([text])[0]
        proba = model.predict_proba([text])[0][1]  # P(positive)

        if pred == 1:
            st.success("✅ Positive")
        else:
            st.error("❌ Negative")

        st.metric("Confidence (P positive)", f"{proba:.3f}")
