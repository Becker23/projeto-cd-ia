import os
import re
import glob
import json
import pickle
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

BASE_DIR = Path(__file__).parent.resolve()

df = pd.read_json("dataset_final.json")
# ------------- Train/test split -------------
if len(df) >= 4:
    X_train, X_test, y_train, y_test = train_test_split(
        df["texto"], df["classe"], test_size=0.3, random_state=42, stratify=df["classe"]
    )
else:
    # If dataset is too small for stratified split, fallback
    X_train, X_test, y_train, y_test = (
        df["texto"],
        df["texto"],
        df["classe"],
        df["classe"],
    )

# ------------- Pipeline: TF-IDF + LinearSVC -------------
# Create a TfidfVectorizer and save it
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),  # unigrams + bigrams
    max_df=0.9,
    min_df=1,
    max_features=20000,
)

X_train_tfidf = vectorizer.fit_transform(X_train)

# Save the vectorizer for future use
vectorizer_path = str(BASE_DIR / "vectorizer.pkl")
with open(vectorizer_path, "wb") as f:
    pickle.dump(vectorizer, f)
    print(f"Vectorizer salvo como {vectorizer_path}")

# Creating and fitting the model
model = LinearSVC(C=1.0)
model.fit(X_train_tfidf, y_train)

# Transform the test data using the same vectorizer
X_test_tfidf = vectorizer.transform(X_test)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Metrics and reports
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=4)
cm = confusion_matrix(y_test, y_pred, labels=["humano", "ia"]).tolist()

# Save the model
model_path = str(BASE_DIR / "modelo_tfidf_linearsvc.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)

metrics = {
    "accuracy": float(acc),
    "labels": ["humano", "ia"],
    "confusion_matrix": cm,
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
}
metrics_path = str(BASE_DIR / "metrics_linear.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

# Display a compact summary for the user
print("Dataset (amostra):")
print(df.head(10))

print("Resumo de treinamento simples (TF-IDF + LinearSVC):")
print(f"- Amostras treino: {len(X_train)} | teste: {len(X_test)}")
print(f"- Accuracy teste: {acc:.4f}")
print("Matriz de confusão [rows: humano, ia]:")
print(np.array(cm))
print("\nClassification report:\n", report)

print("\nArquivos gerados:")
# print("Dataset CSV:", dataset_path)
print("Modelo (pickle):", model_path)
print("Métricas JSON:", metrics_path)
print("Vectorizer (pickle):", vectorizer_path)
