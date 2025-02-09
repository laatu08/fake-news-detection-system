import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

# Define file paths
DATA_DIR = "C:\\Code\\Fake News Detection System\\nlp-model\\data"
TFIDF_TRAIN_FILE = os.path.join(DATA_DIR, "liar_train_tfidf.pkl")
CLEANED_TRAIN_FILE = os.path.join(DATA_DIR, "liar_train_clean.csv")
MODEL_FILE = os.path.join(DATA_DIR, "fake_news_model.pkl")

def evaluate_model():
    # Evaluate the trained model using various metrics and visualizations.
    print("Loading dataset and TF-IDF features...")
    
    # Load TF-IDF matrix
    with open(TFIDF_TRAIN_FILE, "rb") as f:
        X_tfidf = pickle.load(f)

    # Load dataset to get labels
    df = pd.read_csv(CLEANED_TRAIN_FILE)
    y = df["label"]  # Labels

    # Split into training & test sets (same as before)
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    print("Loading trained model...")
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)

    print("Generating evaluation report...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)  # Probability scores for ROC curve

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(DATA_DIR, "confusion_matrix.png"))
    print("Confusion matrix saved.")

    # ROC-AUC Score
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr")
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")  # Diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(DATA_DIR, "roc_curve.png"))
    print("ROC curve saved.")

if __name__ == "__main__":
    evaluate_model()
