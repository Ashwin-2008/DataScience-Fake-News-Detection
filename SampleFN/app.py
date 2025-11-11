import streamlit as st
import joblib


vectorizer = joblib.load("tfidf_vectorizer.pkl")
ml_model = joblib.load("fake_news_model.pkl")


def ml_predict(news_text):
    """Predicts whether news is REAL or FAKE using ML model."""
    tfidf_features = vectorizer.transform([news_text])
    pred_label = ml_model.predict(tfidf_features)[0]
    return "REAL" if pred_label == 1 else "FAKE"


st.title("Fake News Detection — ML Prediction Only")
st.write("Predicts whether news is REAL or FAKE using a trained ML model.")

news_input = st.text_area(" Enter news headline or paragraph")

if st.button("🔍 Analyze News"):
    if not news_input.strip():
        st.warning("Please enter a news headline or paragraph.")
    else:
        with st.spinner("Predicting..."):
            result = ml_predict(news_input)
            color = "green" if result == "REAL" else "red"
            st.markdown(
                f"###  Prediction: <span style='color:{color}'>{result}</span>",
                unsafe_allow_html=True
            )
