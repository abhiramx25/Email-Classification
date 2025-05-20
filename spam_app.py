import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("spam_classifier.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📧 Spam Detector")
st.write("Enter a message below to check if it's spam or not.")

input_text = st.text_area("Message:")

if st.button("Predict"):
    if input_text.strip() == "":
        st.warning("Please enter a message.")
    else:
        transformed = vectorizer.transform([input_text])
        prediction = model.predict(transformed)
        if prediction[0] == 1:
            st.error("🚨 Spam Detected!")
        else:
            st.success("✅ This is not spam.")
