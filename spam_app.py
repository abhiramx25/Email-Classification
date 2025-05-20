import streamlit as st
import joblib
import sklearn
import pandas as pd

# Load the model and vectorizer
model = joblib.load("spam_model.pkl") # Replace with your actual model filename
vectorizer = joblib.load("vectorizer.pkl") # Replace with your actual vectorizer filename

# Page config
st.set_page_config(page_title="📧 Email Classifier", layout="centered")

# Custom background and style
page_bg = """
<style> 
body { 
    background-color: #f0f2f6; 
    font-family: 'Segoe UI', sans-serif; 
} 
.stApp { 
    background-image: linear-gradient(to right top, #e3f2fd, #e1f5fe); 
    padding: 2rem; 
    border-radius: 1rem; 
} 
h1 { 
    color: #003366; 
} 
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Title and subtitle
st.markdown("<h1>📧 Spam Detector App</h1>", unsafe_allow_html=True)
st.markdown("Type an email message below to check if it's spam or not. 🚀")

# Input box
email_text = st.text_area("✉️ Enter your message:", height=200)

if st.button("🔍 Classify"):
    if email_text.strip() == "":
        st.warning("⚠️ Please enter a message to classify.")
    else:
        # Transform input
        transformed = vectorizer.transform([email_text])
        prediction = model.predict(transformed)[0]
        prob = model.predict_proba(transformed)[0][1] # Probability of being spam

        # Show results with emojis
        if prediction == 1:
            st.error(f"🚫 This is SPAM! (Confidence: {prob*100:.2f}%)")
        else:
            st.success(f"✅ Not Spam. You're safe! (Confidence: {(1-prob)*100:.2f}%)")

        # Optional explanation
        st.caption("🔍 Model uses text vectorization + ML classifier.")
