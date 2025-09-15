import streamlit as st
import joblib

# Load model + vectorizer
model = joblib.load("models/model_lr.pkl")
vectorizer = joblib.load("models/vectorizer_tfidf.pkl")

# Streamlit page setup
st.set_page_config(page_title="Fake Job Detector", layout="centered")
st.title("🕵️ Fake Job Post Detector")
st.write("Paste a job ad below to check if it's real or a scam.")

# Text input
user_input = st.text_area("Job description / ad", height=300)

if st.button("🔍 Analyze"):
    if user_input.strip() == "":
        st.warning("Please paste some job text first.")
    else:
        # Transform text
        X_input = vectorizer.transform([user_input])
        pred = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0][pred]

        if pred == 1:
            st.error(f"❌ This looks like a **FAKE** job posting. (Confidence: {proba:.2%})")
        else:
            st.success(f"✅ This looks like a **REAL** job posting. (Confidence: {proba:.2%})")

        # Show full probabilities
        st.write("Prediction Probabilities:")
        st.json({
            "Real (0)": f"{model.predict_proba(X_input)[0][0]:.2%}",
            "Fake (1)": f"{model.predict_proba(X_input)[0][1]:.2%}"
        })
