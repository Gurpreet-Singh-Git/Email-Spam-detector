import re
import pickle

# Load saved model + vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Input mail
mail = input("Give your email: ")

# Preprocessing (same as training)
mail = mail.lower()
mail = re.sub(r'http\S+|www\S+', '', mail)
mail = re.sub(r'[^\w\s]', '', mail)
mail = re.sub(r'\d+', '', mail)

mail = [mail]

# Prediction
vector_mail = vectorizer.transform(mail)
output = model.predict(vector_mail)

print("spam" if output[0] == 1 else "not spam")