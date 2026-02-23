import streamlit as st
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Spam Email Detection", layout="centered")
st.title("Spam Email Detection")

# Training Dataset - Expanded slightly for better reliability
emails = [
    "Win a free iPhone now", 
    "Meeting at 11 am tomorrow", 
    "Claim your prize immediately", 
    "Project discussion with team", 
    "Limited offer buy now",
    "Can we reschedule the call?",
    "Get cheap insurance today",
    "The report is attached for your review"
]
# 1 for Spam, 0 for Not Spam
labels = [1, 0, 1, 0, 1, 0, 1, 0] 

# Text Processing
# TF-IDF converts text into numerical "importance" scores
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(emails)

# Model Training
model = LinearSVC(dual=False) # dual=False is better when samples < features
model.fit(X, labels)

st.subheader("Enter Email Message")
message = st.text_area("Email Text", placeholder="Paste email content here...")

if st.button("Check Spam"):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        # Transform the new message using the same vectorizer
        msg_vec = vectorizer.transform([message])
        prediction = model.predict(msg_vec)[0]
        
        if prediction == 1:
            st.error("🚨 This looks like a Spam Email!")
        else:
            st.success("✅ This seems like a Safe Email.")
