import streamlit as st
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Spam Email Detection", layout="centered")
st.title("Spam Email Detection")

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
labels = [1, 0, 1, 0, 1, 0, 1, 0] 

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(emails)

model = LinearSVC(dual=False) 
model.fit(X, labels)

st.subheader("Enter Email Message")
message = st.text_area("Email Text", placeholder="Paste email content here...")

if st.button("Check Spam"):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        msg_vec = vectorizer.transform([message])
        prediction = model.predict(msg_vec)[0]
        
        if prediction == 1:
            st.error("🚨 This looks like a Spam Email!")
        else:
            st.success("✅ This seems like a Safe Email.")
