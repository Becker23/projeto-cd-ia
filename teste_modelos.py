import sys
from pathlib import Path
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pickle

BASE_DIR = Path(__file__).parent.resolve()
BERT_MODEL_DIR = BASE_DIR / "modelo_bert"
TFIDF_PKL = BASE_DIR / "modelo_tfidf_linearsvc.pkl"
MAX_LENGTH = 256

# device selection (falls back to CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BertInference:
    def __init__(self, model_dir: Path = BERT_MODEL_DIR, device: torch.device = device):
        self.device = device
        self.model_dir = Path(model_dir)
        self.tokenizer = BertTokenizer.from_pretrained(str(self.model_dir))
        self.model = BertForSequenceClassification.from_pretrained(
            str(self.model_dir)
        ).to(self.device)
        self.model.eval()
        # mapping used during fine-tuning: 0 -> "humano", 1 -> "ia"
        self.label_map = {0: "humano", 1: "ia"}

    def predict(self, text: str) -> str:
        enc = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            pred = int(torch.argmax(outputs.logits, dim=1).cpu().item())
        return self.label_map.get(pred, str(pred))

    def predict_batch(self, texts, batch_size: int = 8):
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer.batch_encode_plus(
                batch,
                add_special_tokens=True,
                max_length=MAX_LENGTH,
                truncation=True,
                padding="longest",
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
            results.extend([self.label_map.get(p, str(p)) for p in preds])
        return results


class CombinedInference(BertInference):
    def __init__(self, tfidf_path: Path = TFIDF_PKL, **kwargs):
        super().__init__(**kwargs)
        self.tfidf = None
        # try load TF-IDF + LinearSVC pipeline
        try:
            with open(tfidf_path, "rb") as f:
                self.tfidf = pickle.load(f)
        except Exception as e:
            print(f"Warning: failed to load TF-IDF model '{tfidf_path}': {e}")

    def predict_tfidf(self, text: str) -> str:
        if self.tfidf is None:
            raise RuntimeError("TF-IDF model not loaded")
        return self.tfidf.predict([text])[0]

    def predict_both(self, text: str) -> dict:
        res = {}
        try:
            res["tfidf"] = self.predict_tfidf(text)
        except Exception as e:
            res["tfidf"] = f"error: {e}"
        try:
            res["bert"] = self.predict(text)
        except Exception as e:
            res["bert"] = f"error: {e}"
        return res


def main():
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = input("Enter text to classify: ").strip()

    print(f"Using device: {device}")
    infer = CombinedInference()
    try:
        preds = infer.predict_both(text)
        print("Predictions:")
        print(" - TF-IDF + LinearSVC:", preds.get("tfidf"))
        print(" - BERT:", preds.get("bert"))
    except Exception as e:
        print("Error during inference:", e)


if __name__ == "__main__":
    main()
