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

    print("\nPerforming 5-fold cross-validation...")
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df), 1):
        print(f"\nFold {fold}/5")

        # Split data for this fold
        X_train = df["texto"].iloc[train_idx]
        y_train = df["classe"].iloc[train_idx]
        X_val = df["texto"].iloc[val_idx]
        y_val = df["classe"].iloc[val_idx]

        # Initialize model and tokenizer
        tokenizer = BertTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
        )
        model = BertForSequenceClassification.from_pretrained(
            "neuralmind/bert-base-portuguese-cased", num_labels=2
        ).to(device)

        # Create datasets
        train_dataset = TextClassificationDataset(
            X_train, y_train, tokenizer, MAX_LENGTH
        )
        val_dataset = TextClassificationDataset(X_val, y_val, tokenizer, MAX_LENGTH)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        # Train
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
        for epoch in range(EPOCHS):
            total_loss = train_model(model, train_loader, optimizer, device)
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

        # Evaluate
        predictions, actual_labels = evaluate_model(model, val_loader, device)

        # Calculate accuracy
        label_map = {0: "humano", 1: "ia"}
        pred_labels = [label_map[p] for p in predictions]
        true_labels = [label_map[l] for l in actual_labels]
        fold_acc = accuracy_score(true_labels, pred_labels)

        cv_scores.append(fold_acc)
        print(f"Fold {fold} accuracy: {fold_acc:.4f}")

        # Clean up memory
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Print cross-validation results
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    print("\nCross-validation results:")
    print(f"Mean accuracy: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
    print(f"Individual fold scores: {cv_scores}")


if __name__ == "__main__":
    main()
