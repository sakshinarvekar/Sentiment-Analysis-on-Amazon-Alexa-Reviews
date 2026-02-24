# Amazon Alexa Review Sentiment Analysis

This project is a simple NLP web application that classifies Amazon Alexa product reviews as Positive or Negative.

The model is trained using TF-IDF for text vectorization and Logistic Regression for classification. The application is built with Streamlit and allows users to enter a review and receive a sentiment prediction with a confidence score.

---

## Overview

The goal of this project is to build an end-to-end sentiment analysis system:

- Load and preprocess review data
- Train a machine learning model
- Evaluate performance on a test set
- Deploy the model as an interactive web application

The model is trained dynamically from the dataset when the app starts (no pre-saved model file is used).

---
## Live Demo

Hugging Face Space:  
https://huggingface.co/spaces/sakshinarvekar/alexa-sentiment

---
## Dataset

- File: `amazon_alexa.tsv`
- Total reviews: 3,150
- Target column: `feedback`
  - 1 = Positive
  - 0 = Negative

The dataset is slightly imbalanced, with a higher number of positive reviews.

---

## Model Details

- Vectorizer: TF-IDF (unigrams and bigrams)
- Classifier: Logistic Regression
- Class imbalance handled using `class_weight="balanced"`

The model outputs:
- Predicted sentiment (Positive or Negative)
- Probability score for the positive class

---

## Project Structure

Amazon_Alexa.ipynb   → Data analysis and model experimentation  
app.py               → Streamlit web application  
amazon_alexa.tsv     → Dataset  
requirements.txt     → Dependencies  
README.md            → Documentation  

---

## How to Run Locally

1. Clone the repository:

git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git  
cd YOUR_REPO_NAME  

2. Install dependencies:

pip install -r requirements.txt  

3. Run the application:

streamlit run app.py  

Then open your browser and go to:

http://localhost:8501  

---

## Notes

- Because the dataset contains mostly positive reviews, performance on the negative class should be interpreted carefully.
- Short or negation-heavy phrases such as “not good” may sometimes be misclassified.
- This project focuses on simplicity and clarity rather than advanced deep learning techniques.

---

## Author

Sakshi Narvekar