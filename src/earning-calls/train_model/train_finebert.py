import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm

BASE_DIR = Path("./src/earning-calls")
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models" / "finbert_multitask"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH = DATA_DIR / "val.csv"
TEST_PATH = DATA_DIR / "test.csv"

MODEL_NAME = "ProsusAI/finbert" 

MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1

LABEL2ID = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

class EarningsCallsDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        unc_labels: List[str],
        fl_labels: List[str],
        tokenizer,
        max_length: int = 128,
    ):
        self.texts = texts
        self.unc_labels = [LABEL2ID[str(l)] for l in unc_labels]
        self.fl_labels = [LABEL2ID[str(l)] for l in fl_labels]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "uncertainty_label": torch.tensor(self.unc_labels[idx], dtype=torch.long),
            "forward_label": torch.tensor(self.fl_labels[idx], dtype=torch.long),
        }
        return item

class FinBERTMultiTask(nn.Module):
    def __init__(self, base_model_name: str, num_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(0.3)
        self.uncertainty_head = nn.Linear(hidden_size, num_labels)
        self.forward_head = nn.Linear(hidden_size, num_labels)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        unc_labels: torch.Tensor = None,
        forward_labels: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] / pooled representation depending on base model
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0]

        pooled = self.dropout(pooled)

        logits_unc = self.uncertainty_head(pooled)
        logits_forward = self.forward_head(pooled)

        out = {
            "logits_unc": logits_unc,
            "logits_forward": logits_forward,
        }

        if unc_labels is not None and forward_labels is not None:
            loss_unc = self.loss_fn(logits_unc, unc_labels)
            loss_forward = self.loss_fn(logits_forward, forward_labels)
            loss = (loss_unc + loss_forward) / 2.0
            out["loss"] = loss
            out["loss_unc"] = loss_unc
            out["loss_forward"] = loss_forward

        return out

def load_split(path: Path) -> Tuple[List[str], List[str], List[str]]:
    df = pd.read_csv(path)

    if "text" not in df.columns:
        raise ValueError(f"'text' column not found in {path}")
    if "uncertainty_label" not in df.columns:
        raise ValueError(f"'uncertainty_label' column not found in {path}")
    if "forward_looking_label" not in df.columns:
        raise ValueError(f"'forward_looking_label' column not found in {path}")

    df["uncertainty_label"] = df["uncertainty_label"].fillna("none").astype(str)
    df["forward_looking_label"] = df["forward_looking_label"].fillna("none").astype(str)

    texts = df["text"].tolist()
    unc = df["uncertainty_label"].tolist()
    fl = df["forward_looking_label"].tolist()
    return texts, unc, fl


def compute_accuracy(
    preds_unc: np.ndarray,
    labels_unc: np.ndarray,
    preds_fl: np.ndarray,
    labels_fl: np.ndarray,
) -> Dict[str, float]:
    acc_unc = (preds_unc == labels_unc).mean()
    acc_fl = (preds_fl == labels_fl).mean()
    acc_joint = ((preds_unc == labels_unc) & (preds_fl == labels_fl)).mean()
    return {
        "uncertainty_acc": float(acc_unc),
        "forward_acc": float(acc_fl),
        "joint_acc": float(acc_joint),
    }


# -------------------------------------------------------------------
# Training / evaluation loops
# -------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    running_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Training epoch {epoch}", leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        unc_labels = batch["uncertainty_label"].to(device)
        forward_labels = batch["forward_label"].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            unc_labels=unc_labels,
            forward_labels=forward_labels,
        )
        loss = outputs["loss"]
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    return running_loss / len(dataloader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    losses = []
    all_unc_preds = []
    all_unc_labels = []
    all_fl_preds = []
    all_fl_labels = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        unc_labels = batch["uncertainty_label"].to(device)
        forward_labels = batch["forward_label"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            unc_labels=unc_labels,
            forward_labels=forward_labels,
        )

        loss = outputs["loss"]
        losses.append(loss.item())

        logits_unc = outputs["logits_unc"]
        logits_forward = outputs["logits_forward"]

        preds_unc = torch.argmax(logits_unc, dim=1)
        preds_fl = torch.argmax(logits_forward, dim=1)

        all_unc_preds.extend(preds_unc.cpu().numpy())
        all_unc_labels.extend(unc_labels.cpu().numpy())
        all_fl_preds.extend(preds_fl.cpu().numpy())
        all_fl_labels.extend(forward_labels.cpu().numpy())

    all_unc_preds = np.array(all_unc_preds)
    all_unc_labels = np.array(all_unc_labels)
    all_fl_preds = np.array(all_fl_preds)
    all_fl_labels = np.array(all_fl_labels)

    metrics = compute_accuracy(
        all_unc_preds,
        all_unc_labels,
        all_fl_preds,
        all_fl_labels,
    )
    metrics["loss"] = float(np.mean(losses))
    return metrics


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    print("Loading tokenizer and base FinBERT model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data splits...")
    train_texts, train_unc, train_fl = load_split(TRAIN_PATH)
    val_texts, val_unc, val_fl = load_split(VAL_PATH)
    test_texts, test_unc, test_fl = load_split(TEST_PATH)

    print(f"Train size: {len(train_texts)}")
    print(f"Val size:   {len(val_texts)}")
    print(f"Test size:  {len(test_texts)}")

    train_ds = EarningsCallsDataset(
        train_texts, train_unc, train_fl, tokenizer, max_length=MAX_LENGTH
    )
    val_ds = EarningsCallsDataset(
        val_texts, val_unc, val_fl, tokenizer, max_length=MAX_LENGTH
    )
    test_ds = EarningsCallsDataset(
        test_texts, test_unc, test_fl, tokenizer, max_length=MAX_LENGTH
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print("Building multi-task FinBERT...")
    model = FinBERTMultiTask(MODEL_NAME, NUM_LABELS)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_joint_acc = 0.0
    best_model_path = MODEL_DIR / "best_model.pt"

    for epoch in range(1, EPOCHS + 1):
        print(f"\n==== Epoch {epoch}/{EPOCHS} ====")
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )
        print(f"Train loss: {train_loss:.4f}")

        val_metrics = evaluate(model, val_loader, device)
        print(
            f"Val loss: {val_metrics['loss']:.4f} | "
            f"unc_acc: {val_metrics['uncertainty_acc']:.4f} | "
            f"fwd_acc: {val_metrics['forward_acc']:.4f} | "
            f"joint_acc: {val_metrics['joint_acc']:.4f}"
        )

        if val_metrics["joint_acc"] > best_val_joint_acc:
            best_val_joint_acc = val_metrics["joint_acc"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "label2id": LABEL2ID,
                    "id2label": ID2LABEL,
                    "model_name": MODEL_NAME,
                },
                best_model_path,
            )
            print(f"Saved new best model to: {best_model_path}")

    print("\nLoading best model for test evaluation...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(model, test_loader, device)
    print("\n=== Test metrics ===")
    print(
        f"Test loss: {test_metrics['loss']:.4f} | "
        f"unc_acc: {test_metrics['uncertainty_acc']:.4f} | "
        f"fwd_acc: {test_metrics['forward_acc']:.4f} | "
        f"joint_acc: {test_metrics['joint_acc']:.4f}"
    )

    # Save final config/metadata
    meta_path = MODEL_DIR / "training_metadata.json"
    import json

    with open(meta_path, "w") as f:
        json.dump(
            {
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "max_length": MAX_LENGTH,
                "best_val_joint_acc": best_val_joint_acc,
                "test_metrics": test_metrics,
                "label2id": LABEL2ID,
                "id2label": ID2LABEL,
                "model_name": MODEL_NAME,
            },
            f,
            indent=2,
        )
    print(f"\nSaved training metadata to: {meta_path}")


if __name__ == "__main__":
    main()