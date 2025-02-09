import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# Download necessary nltk data
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")


# Define File Path
DATA_DIR="C:\\Code\\Fake News Detection System\\nlp-model\\data"
TRAIN_FILE=os.path.join(DATA_DIR,"liar_train.csv")
CLEANED_TRAIN_FILE=os.path.join(DATA_DIR,"liar_train_clean.csv")


# Initialize nlp tools
stop_words=set(stopwords.words("english"))
lemmatizer=WordNetLemmatizer()


def clean_text(text):
    # Preprocess text by removing punctuation, numbers, and stopwords.
    text=text.lower()
    text=re.sub(r"\d+","",text) # remove number
    text=re.sub(r"[^\w\s]","",text) # remove punctuation
    words=word_tokenize(text) # tokenize text
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatization & remove stopwords
    return " ".join(words)


def preprocess_dataset():
        # Load dataset, clean text, and save preprocessed data.
        print("Loading Dataset")
        df=pd.read_csv(TRAIN_FILE)

        # Ensure 'statement' and 'label' columns exist
        if "statement" not in df.columns or "label" not in df.columns:
              raise ValueError("Dataset must contain 'statement' and 'label' column")
        
        print("Cleaning Text")
        df["clean_text"]=df["statement"].astype(str).apply(clean_text)

        # Save cleaned dataset
        df.to_csv(CLEANED_TRAIN_FILE, index=False)
        print(f"✅ Preprocessed dataset saved as {CLEANED_TRAIN_FILE}")
        print(df.head())  # Show sample output


if __name__ == "__main__":
    preprocess_dataset()