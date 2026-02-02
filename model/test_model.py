import pickle

# Load trained model
with open("sentiment_model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

def predict_sentiment(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]

    return "Positive 😊" if prediction == 1 else "Negative 😠"


# Test cases
print(predict_sentiment("I absolutely loved this movie"))
print(predict_sentiment("This was the worst experience ever"))
