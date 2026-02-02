import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Load dataset
df = pd.read_csv("../dataset/imdb_small.csv")

# 2. Convert labels to numbers
df["sentiment"] = df["sentiment"].map({
    "positive": 1,
    "negative": 0
})

# 3. Split input and output
X = df["review"]
y = df["sentiment"]

# 4. Convert text to numbers
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)
X_tfidf = vectorizer.fit_transform(X)

# 5. Train model
model = LogisticRegression()
model.fit(X_tfidf, y)

# 6. Save model and vectorizer
with open("sentiment_model.pkl", "wb") as f:
    pickle.dump((model, vectorizer), f)

print("Model trained and saved")
