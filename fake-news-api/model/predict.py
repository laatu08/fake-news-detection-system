import sys
import os
import pickle
import pandas as pd


# defile file path
DATA_DIR=os.path.dirname(os.path.abspath(__file__))
MODEL_FILE=os.path.join(DATA_DIR,"fake_news_model.pkl")
VECTORIZER_FILE=os.path.join(DATA_DIR,"tfidf_vectorizer.pkl")


# load the model and vectorizer
with open(MODEL_FILE,"rb") as f:
    model=pickle.load(f)

with open(VECTORIZER_FILE,"rb") as f:
    vectorizer=pickle.load(f)


def predict(text):
    text_vectorized=vectorizer.transform([text])
    prediction=model.predict(text_vectorized)[0]
    confidence=model.predict_proba(text_vectorized).max()
    return prediction,confidence


if __name__=="__main__":
    input_text=sys.argv[1]
    pred_label,pred_confidence=predict(input_text)
    print(f"Prediction: {'Fake' if pred_label==1 else 'Real'}")
    print(f"Confidence: {pred_confidence:.2f}")