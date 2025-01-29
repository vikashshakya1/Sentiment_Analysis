import pandas as pd
import sqlite3
from datasets import load_dataset

# Load dataset from Hugging Face
dataset = load_dataset("imdb")

# Convert to Pandas DataFrame
df_train = pd.DataFrame(dataset["train"])
df_test = pd.DataFrame(dataset["test"])

# Combine train and test datasets
df = pd.concat([df_train, df_test], ignore_index=True)

# Convert sentiment labels (0 = negative, 1 = positive) to text
df["sentiment"] = df["label"].map({0: "negative", 1: "positive"})

# Connect to SQLite database
conn = sqlite3.connect("imdb_reviews.db")
cursor = conn.cursor()

# Create table
cursor.execute("""
CREATE TABLE IF NOT EXISTS imdb_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_text TEXT,
    sentiment TEXT
)
""")
conn.commit()

# Insert data into the database
df[["text", "sentiment"]].rename(columns={"text": "review_text"}).to_sql(
    "imdb_reviews", conn, if_exists="replace", index=False
)

print("Database setup complete. Data inserted from Hugging Face.")
conn.close()
