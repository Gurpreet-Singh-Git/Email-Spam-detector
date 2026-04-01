import streamlit as st
import re
import pickle

model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📧 Spam Email Detector")

user_input = st.text_area("Enter your email here:")

if st.button("Check"):
    mail = user_input.lower()
    mail = re.sub(r'http\S+|www\S+', '', mail)
    mail = re.sub(r'[^\w\s]', '', mail)
    mail = re.sub(r'\d+', '', mail)

    mail = [mail]
    vector_mail = vectorizer.transform(mail)
    output = model.predict(vector_mail)

    if output[0] == 1:
        st.error("🚨 Spam Detected")
    else:
        st.success("✅ Not Spam")