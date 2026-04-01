import streamlit as st
import re
import pickle

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "spam_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))

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
