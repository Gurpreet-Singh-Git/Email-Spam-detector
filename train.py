import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load data
df = pd.read_csv(r"C:\Users\GURPREET\OneDrive\Desktop\combined_data.csv")

# Preprocessing
df['text'] = df['text'].str.lower()
df['text'] = df['text'].str.replace(r'http\S+|www\S+', '', regex=True)
df['text'] = df['text'].str.replace(r'[^\w\s]', '', regex=True)
df['text'] = df['text'].str.replace(r'\d+', '', regex=True)

x = df['text']
y = df['label']

# Vectorization
vectorizer = TfidfVectorizer()
x_vec = vectorizer.fit_transform(x)

# Model training
model = MultinomialNB()
model.fit(x_vec, y)

# Save model + vectorizer
pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained and saved ✅")