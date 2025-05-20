Email Spam Classifier

OVERVIEW
A machine learning system that automatically classifies emails as spam or legitimate (ham) using Natural Language Processing techniques and the Naïve Bayes algorithm. The model achieves 98.38% accuracy and is deployed as an interactive web application.

KEY FEATURES
- Text Processing:
  • Email cleaning/normalization
  • Tokenization & stopword removal
  • TF-IDF vectorization
- ML Model:
  • Multinomial Naïve Bayes
  • Trained on 5,572 emails
- Web Interface:
  • Streamlit app
  • Real-time predictions

DATASET
spam.csv containing:
• Spam: Ads, phishing emails
• Ham: Legitimate emails

HOW TO USE
1. Paste email text
2. Click "Predict"
3. Get spam/ham result

PERFORMANCE
• Accuracy: 98.38%
• Precision: 94.12%
• Recall: 89.47%

Check out the live APP: 
https://email-classification-3p6vmxbnss9pqzivrd7kq6.streamlit.app/
