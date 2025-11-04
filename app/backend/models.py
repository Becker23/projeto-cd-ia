import pickle

import joblib
from transformers import BertTokenizer, BertForSequenceClassification

bert_model = None
bert_tokenizer = None
tfidf_model = None
tfidf_vectorizer = None

def load_models():
    global bert_model, bert_tokenizer, tfidf_model, tfidf_vectorizer

    if bert_model is None:
        print("Loading BERT model...")
        bert_tokenizer = BertTokenizer.from_pretrained("./models/modelo_bert")
        bert_model = BertForSequenceClassification.from_pretrained("./models/modelo_bert")
        bert_model.eval()
        print("BERTimbau model loaded!")

    if tfidf_model is None:
        print("Loading TFIDF model")
        with open("./models/vectorizer.pkl", 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        tfidf_model = joblib.load("./models/modelo_tfidf_linearsvc.pkl")
        print(type(tfidf_model))
        print("TFIDF model loaded!")