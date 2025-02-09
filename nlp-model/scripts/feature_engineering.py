import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Define file paths
DATA_DIR = "C:\\Code\\Fake News Detection System\\nlp-model\\data"
CLEANED_TRAIN_FILE = os.path.join(DATA_DIR, "liar_train_clean.csv")
TFIDF_TRAIN_FILE = os.path.join(DATA_DIR, "liar_train_tfidf.pkl")
TFIDF_VECTORIZER_FILE = os.path.join(DATA_DIR, "tfidf_vectorizer.pkl")

def apply_tfidf():
    # Convert cleaned text into TF-IDF vectors and save
    print("Loading preprocessed dataset...")
    df = pd.read_csv(CLEANED_TRAIN_FILE)

    # Ensure required columns exist
    if "clean_text" not in df.columns:
        raise ValueError("Dataset must contain 'clean_text' column")

    print("🔢 Applying TF-IDF transformation...")
    vectorizer = TfidfVectorizer(max_features=5000)  # Use top 5000 words
    X_tfidf = vectorizer.fit_transform(df["clean_text"])  # Transform text into TF-IDF features

    # Save the transformed dataset and vectorizer
    with open(TFIDF_TRAIN_FILE, "wb") as f:
        pickle.dump(X_tfidf, f)

    with open(TFIDF_VECTORIZER_FILE, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"TF-IDF vectors saved as {TFIDF_TRAIN_FILE}")
    print(f"TF-IDF vectorizer saved as {TFIDF_VECTORIZER_FILE}")

if __name__ == "__main__":
    apply_tfidf()
