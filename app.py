from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Charger dataset
df = pd.read_csv("sms_dataset.csv")

# Prétraitement
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["message"])
y = df["label"]

# Modèle
model = LogisticRegression()
model.fit(X, y)

# API
app = Flask(__name__)

@app.route("/")
def home():
    return "API SMS Classifier is running"

@app.route("/predict", methods=["POST"])
def predict():
    text = request.json["message"]
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)
    return jsonify({"classe": prediction[0]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
