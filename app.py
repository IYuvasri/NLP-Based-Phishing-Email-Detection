import pandas as pd
import re
import string
import nltk
import sys
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack

# Ensure proper encoding
sys.stdout.reconfigure(encoding='utf-8')
nltk.download('stopwords')

# Load dataset
file_path = "enron_spam_data.csv"
df = pd.read_csv(file_path)

# Prepare labels
df["Spam/Ham"] = df["Spam/Ham"].astype(str).str.strip().str.capitalize()
df["spam"] = df["Spam/Ham"].map({"Ham": 0, "Spam": 1})
df = df.dropna(subset=["spam"])

# Fill missing fields
df["subject"] = df["Subject"].fillna("")
df["body"] = df["Message"].fillna("")

# Stopwords
stop_words = set(stopwords.words("english"))

# Clean text
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return " ".join([word for word in text.split() if word not in stop_words])

df["clean_subject"] = df["subject"].apply(clean_text)
df["clean_body"] = df["body"].apply(clean_text)
df["text"] = df["clean_subject"] + " " + df["clean_body"]

# Extra feature functions
def count_links(text):
    return len(re.findall(r"http[s]?://", text))

def count_uppercase_words(text):
    return len(re.findall(r"\b[A-Z]{2,}\b", text))

def count_special_chars(text):
    return len(re.findall(r"[^a-zA-Z0-9\s]", text))

def get_email_features(row):
    raw_text = row["subject"] + " " + row["body"]
    return pd.Series({
        "num_links": count_links(raw_text),
        "uppercase_words": count_uppercase_words(raw_text),
        "special_chars": count_special_chars(raw_text),
        "email_length": len(raw_text),
        "word_count": len(raw_text.split())
    })

# Apply extra features
extra_features = df.apply(get_email_features, axis=1)

# Vectorize cleaned text
vectorizer = TfidfVectorizer(max_features=5000)
X_text = vectorizer.fit_transform(df["text"])

# Combine text features + engineered features
X = hstack([X_text, extra_features.values])
y = df["spam"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {100 * accuracy_score(y_test, y_pred):.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Model training complete!")

# Update prediction function with new features
def extract_email_features(subject, body):
    raw = subject + " " + body
    return [
        count_links(raw),
        count_uppercase_words(raw),
        count_special_chars(raw),
        len(raw),
        len(raw.split())
    ]

def predict_email(subject, body):
    email_text = clean_text(subject) + " " + clean_text(body)
    email_vector = vectorizer.transform([email_text])
    extra_feats = np.array(extract_email_features(subject, body)).reshape(1, -1)
    full_vector = hstack([email_vector, extra_feats])
    prediction = model.predict(full_vector)[0]
    result = "Spam" if prediction == 1 else "Not Spam"
    print("\nEmail:")
    print("Subject:", subject)
    print("Body:", body)
    print("Prediction:", result)
    return result

# Sample predictions
print("\n--- Sample 1 ---")
predict_email(
    "Meeting request for project discussion",
    "Dear team, I would like to schedule a meeting to discuss the upcoming project. Please let me know your availability. Best regards, John Doe."
)

print("\n--- Sample 2 ---")
predict_email(
    "Congratulations! You've won a $1000 gift card 🎁",
    "Click here to claim your reward now: http://scamlink.com. This is a limited-time offer! Act fast and don't miss out on your free prize."
)
