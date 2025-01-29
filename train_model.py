import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load data
conn = sqlite3.connect("imdb_reviews.db")
df = pd.read_sql("SELECT * FROM imdb_reviews", conn)
conn.close()

# Preprocessing function
def clean_text(text):
    import re, string
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

df["cleaned_review"] = df["review_text"].apply(clean_text)

# Convert sentiment to binary labels
df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["cleaned_review"])
y = df["sentiment"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump((model, vectorizer), f)
