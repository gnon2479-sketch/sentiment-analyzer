from flask import Flask, render_template, request
import pickle
import os
import subprocess

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "sentiment_model.pkl")

model = None
vectorizer = None


def load_or_train_model():
    global model, vectorizer

    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model, vectorizer = pickle.load(f)
        print("Model loaded successfully")
    else:
        print("Model not found. Training model...")
        subprocess.run(["python", "model/train_model.py"], check=True)

        with open(MODEL_PATH, "rb") as f:
            model, vectorizer = pickle.load(f)
        print("Model trained and loaded")


@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None

    if request.method == "POST":
        text = request.form["text"]

        if model is None or vectorizer is None:
            load_or_train_model()

        text_vector = vectorizer.transform([text])
        prediction = model.predict(text_vector)[0]

        sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"

    return render_template("index.html", sentiment=sentiment)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
