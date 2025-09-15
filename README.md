# 🕵️‍♀️ Fake Job Detector

A machine learning project that detects **fake job postings** using **Natural Language Processing (NLP)** techniques and a **Logistic Regression classifier**. It includes a **Streamlit app** for real-time prediction based on job descriptions.

---

## 📂 Project Structure
Fake-Job-Detector/
│
├── app.py # Streamlit web app
├── data_cleaning.ipynb # Data cleaning and EDA
├── train_model.ipynb # TF-IDF vectorization + model training
├── requirements.txt # Dependencies for the project
├── README.md # Project overview
├── fake_job_postings.csv # Original dataset
│
├── models/
│ ├── model_lr.pkl # Trained Logistic Regression model
│ └── vectorizer_tfidf.pkl # TF-IDF vectorizer
│
└── Data/
└── processed/ # Cleaned train, val, test CSVs

## 📊 Dataset

- Source: [Kaggle - Fake Job Postings](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- Features include: `title`, `location`, `description`, `requirements`, `company_profile`, etc.
- Target: `fraudulent` (1 = fake, 0 = real)

## 🧹 Preprocessing Steps

- Dropped irrelevant or missing-value-heavy columns
- Cleaned HTML, punctuation, and stopwords
- Combined important text features into a single corpus
- Split data into train/val/test
- Handled class imbalance

## ⚙️ Model Training

- Text vectorized using **TF-IDF**
- Classifier: **Logistic Regression**
- Evaluation: Confusion matrix, precision, recall, F1-score

## 🖥️ Streamlit App Features

- Paste any job description to check if it's real or fake
- Displays:
  - Prediction result
  - Confidence score
  - Probability of both classes
- Simple, clean interface

## 🚀 How to Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/shatakshii02/Fake-Job-Detector.git
