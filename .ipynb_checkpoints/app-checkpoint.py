import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from lime.lime_text import LimeTextExplainer


st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")

# Debug: First check Streamlit is loading
st.write("✅ App started successfully")

# Download stopwords (if not already)
nltk.download('stopwords')

# Load stopwords safely with caching
@st.cache_resource
def get_stopwords():
    return set(stopwords.words('english'))

stop_words = get_stopwords()

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\\s]', '', text)
    tokens = text.split()
    return ' '.join([word for word in tokens if word not in stop_words])

# Debug: Load dataset
try:
    fake = pd.read_csv("Fake.csv")
    real = pd.read_csv("True.csv")
    st.write("✅ Data loaded successfully")
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

# Prepare dataset
fake['label'] = 0
real['label'] = 1
data = pd.concat([fake, real])
data = data.sample(frac=1).reset_index(drop=True)
data['clean_text'] = data['text'].apply(clean_text)

# TF-IDF and model training
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['clean_text'])
y = data['label']

model = RandomForestClassifier()
model.fit(X, y)

# App input
user_input = st.text_area("📋 Paste news article or headline below:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        confidence = model.predict_proba(vector).max()

        label = "🟢 Real News" if prediction == 1 else "🔴 Fake News"
        st.markdown(f"## Prediction: {label}")
        st.markdown(f"**Confidence:** `{confidence:.2f}`")

        # Create pipeline for LIME
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(vectorizer, model)

# LIME Explainer
explainer = LimeTextExplainer(class_names=["Fake", "Real"])

if st.button("Explain with LIME"):
    if user_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        explanation = explainer.explain_instance(
            user_input,
            pipeline.predict_proba,
            num_features=10
        )
        st.subheader("🧠 Explanation")
        st.components.v1.html(explanation.as_html(), height=500, scrolling=True)

