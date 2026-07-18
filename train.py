import sys
import pandas as pd
import sklearn.feature_extraction.text

# Load datasets
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine datasets
df = pd.concat([fake, true], ignore_index=True)

# Shuffle data
df = df.sample(frac=1, random_state=42)

print("Total Rows:", df.shape)
print(df[["title", "label"]].head())
# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())
df["content"] = df["title"] + " " + df["text"]

vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(stop_words="english", max_features=5000)

X = vectorizer.fit_transform(df["content"])
y = df["label"]

print("TF-IDF Shape:", X.shape)
print("Labels Shape:", y.shape)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
def predict_news(news):
    news_vector = vectorizer.transform([news])
    prediction = model.predict(news_vector)

    if prediction[0] == 0:
        return "Fake News"
    else:
        return "Real News"


sample_news = input("Enter a news headline: ")
print("Prediction:", predict_news(sample_news))
import pickle

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model Saved Successfully!")