import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

# Check missing values
print("Missing Values:")
print(df.isnull().sum())

# Create content column
df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

X = vectorizer.fit_transform(df["content"])
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Prediction Function
def predict_news(news):
    news_vector = vectorizer.transform([news])
    prediction = model.predict(news_vector)

    if prediction[0] == 0:
        return "Fake News"
    else:
        return "Real News"

# User Input
sample_news = input("Enter a news headline: ")
print("Prediction:", predict_news(sample_news))

# Save Model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model Saved Successfully!")