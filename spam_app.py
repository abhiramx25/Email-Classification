import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer  # Added import

# Page config - should be the first Streamlit command
st.set_page_config(page_title="📧 Email Classifier", layout="centered")

try:
    # Load the model and vectorizer with error handling
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except Exception as e:
    st.error(f"❌ Error loading model files: {e}")
    st.stop()  # Stop the app if files can't be loaded

# Custom CSS styling
st.markdown("""
<style>
    .stApp {
        background-image: linear-gradient(to right top, #e3f2fd, #e1f5fe);
        padding: 2rem;
        border-radius: 1rem;
    }
    h1 {
        color: #003366;
        text-align: center;
    }
    .stTextArea textarea {
        font-size: 16px !important;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown("<h1>📧 Spam Detector App</h1>", unsafe_allow_html=True)
st.markdown("Type an email message below to check if it's spam or not. 🚀")

# Text input area
email_text = st.text_area(
    "✉️ Enter your message:", 
    height=200,
    placeholder="Paste your email content here..."
)

# Classification button
if st.button("🔍 Classify", type="primary"):
    if not email_text.strip():
        st.warning("⚠️ Please enter a message to classify.")
    else:
        try:
            # Transform and predict
            transformed = vectorizer.transform([email_text])
            prediction = model.predict(transformed)[0]
            proba = model.predict_proba(transformed)[0]
            
            # Display results
            if prediction == 1:  # Assuming 1 is spam
                confidence = proba[1] * 100
                st.error(f"🚫 SPAM ALERT! (Confidence: {confidence:.1f}%)")
                st.warning("Be careful with this message!")
            else:
                confidence = proba[0] * 100
                st.success(f"✅ Legitimate Email (Confidence: {confidence:.1f}%)")
                st.balloons()
            
            # Add some explanation
            with st.expander("ℹ️ How this works"):
                st.markdown("""
                - The model analyzes text patterns using machine learning
                - It compares your message against known spam characteristics
                - Confidence score shows how certain the model is
                """)
                
        except Exception as e:
            st.error(f"❌ Error during classification: {e}")
