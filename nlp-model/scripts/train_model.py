import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Define file paths
DATA_DIR = "C:\\Code\\Fake News Detection System\\nlp-model\\data"
TFIDF_TRAIN_FILE = os.path.join(DATA_DIR, "liar_train_tfidf.pkl")
TFIDF_VECTORIZER_FILE = os.path.join(DATA_DIR, "tfidf_vectorizer.pkl")
CLEANED_TRAIN_FILE = os.path.join(DATA_DIR, "liar_train_clean.csv")
MODEL_FILE = os.path.join(DATA_DIR, "fake_news_model.pkl")

def train_model():
    # Train a Logistic Regression model for fake news detection
    print("Loading dataset and TF-IDF features...")
    
    # Load TF-IDF matrix
    with open(TFIDF_TRAIN_FILE, "rb") as f:
        X_tfidf = pickle.load(f)

    # Load dataset to get labels
    df = pd.read_csv(CLEANED_TRAIN_FILE)
    if "label" not in df.columns:
        raise ValueError("Dataset must contain 'label' column")

    y = df["label"]  # Labels

    # Split into training & test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    # Save trained model
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved as {MODEL_FILE}")

if __name__ == "__main__":
    train_model()
