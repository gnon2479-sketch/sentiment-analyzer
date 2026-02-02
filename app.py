import os
import pickle
import subprocess
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MODEL_PATH = "model/sentiment_model.pkl"

# Agar model file nahi hai to pehle train karo
if not os.path.exists(MODEL_PATH):
    print("Model not found. Training model...")
    subprocess.run(["python", "model/train_model.py"], check=True)

# Trained model load karo
with open(MODEL_PATH, "rb") as f:
    model, vectorizer = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    if text.strip() == "":
        return jsonify({"sentiment": "Please enter some text 📝"})

    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]

    result = "Positive 😊💖" if prediction == 1 else "Negative 😔💔"
    return jsonify({"sentiment": result})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
