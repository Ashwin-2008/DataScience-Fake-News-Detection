# DataScience-Fake-News-Detection
Project Overview

This project is a machine learning-based fake news detection system that classifies news articles or headlines as REAL or FAKE. It uses TF-IDF vectorization to convert text into numerical features and a Logistic Regression classifier for prediction.

This project is suitable for research demos, learning purposes, and basic AI-powered news verification.

Features

✅ Detects fake or real news based purely on ML (TF-IDF + Logistic Regression)

✅ Easy to use with a simple Streamlit interface

✅ Provides confidence scores for predictions

✅ Lightweight and fast inference

Dataset

The system requires a CSV dataset with the following columns:

text → News content or headline

label → Target label (0 = FAKE, 1 = REAL)

Recommended dataset: Kaggle Fake and Real News Dataset

Installation

Clone the repository:

git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection


Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate


Install required packages:

pip install -r requirements.txt

Usage
1. Train the Model
python train_model.py


This will:

Train the Logistic Regression classifier

Save the model as fake_news_model.pkl

Save the TF-IDF vectorizer as tfidf_vectorizer.pkl

2. Run the Streamlit App
streamlit run app.py


Enter a news headline or paragraph

Click Analyze News

View the ML prediction and confidence score

Project Structure
fake-news-detection/
│
├─ train_model.py          # Script to train ML model
├─ app.py                  # Streamlit app for prediction
├─ fake_news_model.pkl     # Trained ML model (generated)
├─ tfidf_vectorizer.pkl    # TF-IDF vectorizer (generated)
├─ fake_news_dataset.csv   # Sample dataset
├─ requirements.txt        # Python dependencies
└─ README.md               # Project documentation

Dependencies

Python ≥ 3.10

pandas

scikit-learn

joblib

streamlit

sentence-transformers (optional for semantic enhancements)

Install all dependencies using:

pip install -r requirements.tx
