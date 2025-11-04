# controller/classify_controller.py
import models as m
from transformers import TextClassificationPipeline
import numpy as np

def classify_with_bert(text: str):
    tokenizer = m.bert_tokenizer
    model = m.bert_model

    if model is None or tokenizer is None:
        return {"error": "BERT model not loaded"}

    pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)
    results = pipeline(text)[0]
    # results will be something like: [{'label': 'LABEL_0', 'score': 0.65}, {'label': 'LABEL_1', 'score': 0.35}]

    best = max(results, key=lambda x: x['score'])
    label = best["label"]
    score = best["score"]

    # Map labels
    if label == "LABEL_0":
        mapped_label = "humano"
    elif label == "LABEL_1":
        mapped_label = "ia"
    else:
        # fallback in case there are more labels
        mapped_label = label

    return {
        "model": "BERTimbau",
        "prediction": mapped_label,
        "confidence": round(score * 100, 2)
    }



def classify_with_tfidf(text: str):
    model = m.tfidf_model
    vectorizer = m.tfidf_vectorizer

    if model is None or vectorizer is None:
        return {"error": "TF-IDF model or vectorizer not loaded"}

    # Transform text to features
    features = vectorizer.transform([text])  # shape (1, n_features)

    # Prediction
    pred = model.predict(features)[0]

    # Probability or pseudo-probability
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]
        conf = np.max(probs)
    else:
        if hasattr(model, "decision_function"):
            score = model.decision_function(features)[0]
            # Convert decision function value to pseudo-probability
            conf = float(1 / (1 + np.exp(-abs(score))))
        else:
            conf = 1.0

    return {
        "model": "TF-IDF + LinearSVC",
        "prediction": str(pred),
        "confidence": round(conf * 100, 2)
    }


def classify_both(text: str):
    bert_result = classify_with_bert(text)
    tfidf_result = classify_with_tfidf(text)

    return {
        "bert": bert_result,
        "tfidf": tfidf_result
    }