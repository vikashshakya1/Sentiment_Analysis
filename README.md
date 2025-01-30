# Sentiment Analysis Pipeline

## Project Overview
This project implements an end-to-end sentiment analysis pipeline using the **IMDB Movie Reviews Dataset**. It includes:
- Data acquisition and storage in an SQLite database.
- Data preprocessing and exploratory analysis.
- Training a machine learning model to classify reviews as **positive** or **negative**.
- Deploying the trained model via a **Flask API** for real-time predictions.

## Project Setup

### 1. Install Dependencies
Ensure you have Python installed (>=3.8), then run:
```bash
pip install -r requirements.txt
```
This will install required libraries such as `pandas`, `scikit-learn`, `flask`, `datasets`, and `sqlite3`.

### 2. Set Up the SQLite Database
Run the following script to set up the database and insert data:
```bash
python data_setup.py
```
This script will:
- Load the **IMDB dataset** from Hugging Face.
- Create an SQLite database (`imdb_reviews.db`).
- Insert the dataset into the database.

## Data Acquisition

### Downloading the Dataset
The dataset is obtained using **Hugging Face Datasets**:
```python
from datasets import load_dataset
dataset = load_dataset("imdb")
```
Alternatively, you can manually download it from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and load it as a CSV file.

## Run Instructions

### 1. Train the Model
Run the training script to train a **Logistic Regression model**:
```bash
python train_model.py
```
This script will:
- Load and clean the dataset.
- Convert text into **TF-IDF vectors**.
- Train a **Logistic Regression** classifier.
- Save the trained model and vectorizer (`model.pkl`).

### 2. Start the Flask API
Launch the API server:
```bash
python app.py
```
This will start a **Flask server** on `http://127.0.0.1:5000`.

### 3. Test the API Endpoint
Send a test request using **Postman, curl, or Python**:
#### Using curl:
```bash
curl -X POST "http://127.0.0.1:5000/predict" \
     -H "Content-Type: application/json" \
     -d '{"review_text": "This movie was amazing!"}'
```
#### Using Python:
```python
import requests
url = "http://127.0.0.1:5000/predict"
data = {"review_text": "This movie was amazing!"}
response = requests.post(url, json=data)
print(response.json())
```
Expected Output:
```json
{"sentiment_prediction": "positive"}
```

## Model Information
- **Preprocessing**: Lowercasing, punctuation removal, removing HTML tags.
- **Vectorization**: TF-IDF with 5000 features.
- **Model**: Logistic Regression.
- **Evaluation Metrics**:
  - Accuracy: **~87%** on the test set.
  - F1-score: **~0.86** (positive/negative sentiment).
---
**Author:** Vikash Shakya 
**Contact:** vikashhshakya@gmail.com
