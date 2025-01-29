from flask import Flask, request, jsonify
import pickle

# Load trained model
with open("model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    review_text = data.get("review_text", "")

    # Preprocess input text
    def clean_text(text):
        import re, string
        text = text.lower()
        text = re.sub(r"<br\s*/?>", " ", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text

    cleaned_text = clean_text(review_text)
    transformed_text = vectorizer.transform([cleaned_text])

    # Predict sentiment
    prediction = model.predict(transformed_text)[0]
    sentiment = "positive" if prediction == 1 else "negative"

    return jsonify({"sentiment_prediction": sentiment})

if __name__ == "__main__":
    app.run(debug=True)