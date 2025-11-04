import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW  # Changed import to torch.optim
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
BASE_DIR = Path(r"C:\Users\Enzo\Documents\projects\projeto-cd-ia")
MAX_LENGTH = 256  # Reduced from 512
BATCH_SIZE = 4  # Reduced from 8
EPOCHS = 5
LEARNING_RATE = 2e-5


# Custom dataset class
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = 1 if self.labels.iloc[idx] == "ia" else 0

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def print_gpu_memory_stats():
    if torch.cuda.is_available():
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")


def train_model(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx % 10 == 0:  # Print memory stats every 10 batches
            print_gpu_memory_stats()

        try:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            # Explicitly clear some memory
            del outputs
            del loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except RuntimeError as e:
            print(f"Error in batch {batch_idx}: {str(e)}")
            print_gpu_memory_stats()
            raise

    return total_loss


def evaluate_model(model, test_loader, device):
    model.eval()
    predictions = []
    actual_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            _, preds = torch.max(outputs.logits, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())

    return predictions, actual_labels


def main():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print_gpu_memory_stats()

    # Load data
    df = pd.read_json("dataset_final.json")

    # Split data
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        df["texto"], df["classe"], test_size=0.3, random_state=42, stratify=df["classe"]
    )

    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    model = BertForSequenceClassification.from_pretrained(
        "neuralmind/bert-base-portuguese-cased", num_labels=2
    ).to(device)

    # Create datasets
    train_dataset = TextClassificationDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    test_dataset = TextClassificationDataset(X_test, y_test, tokenizer, MAX_LENGTH)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print("Training BERT model...")
    for epoch in range(EPOCHS):
        total_loss = train_model(model, train_loader, optimizer, device)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

    # Evaluation
    print("\nEvaluating model...")
    predictions, actual_labels = evaluate_model(model, test_loader, device)

    # Convert numeric predictions back to labels
    label_map = {0: "humano", 1: "ia"}
    pred_labels = [label_map[p] for p in predictions]
    true_labels = [label_map[l] for l in actual_labels]

    # Calculate metrics
    acc = accuracy_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, digits=4)
    cm = confusion_matrix(true_labels, pred_labels, labels=["humano", "ia"]).tolist()

    # Save metrics
    metrics = {
        "accuracy": float(acc),
        "labels": ["humano", "ia"],
        "confusion_matrix": cm,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    metrics_path = str(BASE_DIR / "metrics_bert.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Save model
    model_path = str(BASE_DIR / "modelo_bert")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    # Display results
    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix [humano, ia]:")
    print(np.array(cm))

    print("\nFiles generated:")
    print(f"BERT Model: {model_path}")
    print(f"Metrics JSON: {metrics_path}")


if __name__ == "__main__":
    main()
