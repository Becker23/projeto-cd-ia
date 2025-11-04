import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# Load dataset
df = pd.read_json("dataset_final.json")
# Define pipeline
pipeline = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),  # unigrams + bigrams
                max_df=0.9,
                min_df=1,
                max_features=20000,
            ),
        ),
        ("clf", LinearSVC(C=1.0)),
    ]
)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(
    pipeline, df["texto"], df["classe"], cv=5, scoring="accuracy"
)

# Print cross-validation results
print("\nCross-validation results:")
print(f"Mean accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
print(f"Individual fold scores: {cv_scores}")
