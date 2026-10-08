# 📧 Spam Email Classifier

A machine learning-based **Spam Email Classifier** that uses Natural Language Processing (NLP) to automatically classify emails as **Spam** or **Not Spam**.

The project uses **TF-IDF (Term Frequency–Inverse Document Frequency)** for text feature extraction and **Multinomial Naive Bayes** for classification. A **Streamlit** web application provides an interactive interface for real-time predictions.

---

## 🚀 Features

- 📩 Classifies emails as **Spam** or **Not Spam**
- 🧹 Automated text preprocessing
- 🔤 TF-IDF-based feature extraction
- 🤖 Multinomial Naive Bayes classification
- ⚡ Real-time prediction
- 🌐 Interactive Streamlit web interface
- 💾 Saved model and vectorizer using Pickle
- 🖥️ Supports both command-line and web-based prediction

---

## 🧠 Machine Learning Pipeline

```text
Raw Email
    ↓
Text Preprocessing
    ↓
TF-IDF Vectorization
    ↓
Multinomial Naive Bayes
    ↓
Spam / Not Spam
```

### 1. Text Preprocessing

The input email goes through several preprocessing steps:

- Convert text to lowercase
- Remove URLs and web addresses
- Remove punctuation and special characters
- Remove numerical characters
- Convert cleaned text into a format suitable for ML

### 2. Feature Extraction

The cleaned email text is transformed into numerical features using:

**TF-IDF Vectorization**

TF-IDF assigns higher importance to words that are informative for a particular email while reducing the importance of very common words.

### 3. Classification

The project uses:

**Multinomial Naive Bayes**

This algorithm is particularly suitable for text classification problems because it works effectively with word-frequency-based features such as TF-IDF.

---

## 📊 Model & Project Statistics

| Component | Details |
|---|---|
| Problem Type | Binary Text Classification |
| Task | Spam Detection |
| Input | Email Text |
| Output | Spam / Not Spam |
| Feature Extraction | TF-IDF |
| ML Algorithm | Multinomial Naive Bayes |
| NLP Preprocessing | Lowercasing, URL, punctuation & number removal |
| Model Serialization | Pickle |
| Frontend | Streamlit |
| Programming Language | Python |

### 📈 Evaluation Metrics

The following metrics should be reported after evaluating the model on a held-out test set:

| Metric | Score |
|---|---:|
| Accuracy | **Add value** |
| Precision | **Add value** |
| Recall | **Add value** |
| F1-Score | **Add value** |

> **Note:** These values should be generated from the actual test-set predictions rather than estimated or manually entered.

---

## 🏗️ Project Structure

```text
Email-Spam-detector/
│
├── app.py                 # Streamlit web application
├── train.py               # Model training pipeline
├── predict.py             # Command-line prediction
├── spam_model.pkl         # Trained Naive Bayes model
├── vectorizer.pkl         # Trained TF-IDF vectorizer
├── requirements.txt       # Python dependencies
└── README.md
```

---

## 🛠️ Technologies Used

### Programming
- Python

### Machine Learning
- Scikit-learn
- Multinomial Naive Bayes
- TF-IDF Vectorizer

### Data Processing
- Pandas
- Regular Expressions

### Deployment / Interface
- Streamlit

### Model Persistence
- Pickle

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Gurpreet-Singh-Git/Email-Spam-detector.git
```

Navigate to the project:

```bash
cd Email-Spam-detector
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Streamlit Application

Start the application using:

```bash
streamlit run app.py
```

The application will open in your browser.

Enter an email message and click **Check**.

The classifier will return either:

```text
🚨 Spam Detected
```

or

```text
✅ Not Spam
```

---

## 💻 Command-Line Prediction

You can also test the classifier directly from the terminal:

```bash
python predict.py
```

Enter an email when prompted:

```text
Give your email:
```

The model will return:

```text
spam
```

or

```text
not spam
```

---

## 🔬 Training Process

The model is trained using the following process:

```python
Raw Dataset
     ↓
Text Cleaning
     ↓
TF-IDF Vectorization
     ↓
Multinomial Naive Bayes
     ↓
Trained Model
     ↓
spam_model.pkl
```

The trained TF-IDF vectorizer is also saved separately:

```text
vectorizer.pkl
```

This ensures that new emails are transformed using the **same feature representation** used during training.

---

## 📌 Example

### Input

```text
Congratulations! You have won a $1000 reward.
Click the link below to claim your prize.
```

### Prediction

```text
🚨 Spam Detected
```

### Another Example

```text
Hi, can you send me the project report before tomorrow's meeting?
```

### Prediction

```text
✅ Not Spam
```

---

## 📊 Why TF-IDF + Naive Bayes?

### TF-IDF

TF-IDF converts text into numerical features while considering the importance of words within the dataset.

### Multinomial Naive Bayes

Multinomial Naive Bayes is lightweight, fast, and commonly used for NLP classification tasks.

Together, they provide a simple and efficient approach for building a spam detection system.

---

## 🔮 Future Improvements

- [ ] Add train/test split and automated evaluation
- [ ] Add confusion matrix visualization
- [ ] Compare multiple ML algorithms
- [ ] Perform hyperparameter tuning
- [ ] Add cross-validation
- [ ] Add probability/confidence score
- [ ] Improve text preprocessing with stop-word removal and lemmatization
- [ ] Deploy the application online
- [ ] Add a larger and more diverse email dataset
- [ ] Add email subject and metadata as additional features

---

## 👨‍💻 Author

**Gurpreet Singh**

GitHub:  
https://github.com/Gurpreet-Singh-Git

---

## ⭐ Project

If you find this project useful, consider giving the repository a ⭐.
