import argparse
import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline


LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def load_splits(data_dir: str):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    for df in (train, val, test):
        # defensive: ensure required columns exist
        missing = {"text", "label"} - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns {missing} in one of the splits.")
        # cast label to int just in case
        df["label"] = df["label"].astype(int)
        # normalize text
        df["text"] = df["text"].astype(str).str.replace("\u200b", "", regex=False).str.strip()
    return train, val, test


def build_pipeline(max_features=10000):
    """TF-IDF (uni+bi-grams) -> Logistic Regression."""
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words="english",
        lowercase=True,
        strip_accents="unicode",
    )
    clf = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced",   
        n_jobs=None,
        random_state=42,
    )
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def evaluate(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )

    report = classification_report(
        y_true, y_pred, labels=[0, 1, 2],
        target_names=[LABEL_MAP[i] for i in [0, 1, 2]],
        zero_division=0, output_dict=True
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    metrics = {
        "accuracy": acc,
        "per_class": {
            LABEL_MAP[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            } for i in [0, 1, 2]
        },
        "macro_avg": {
            "precision": float(report["macro avg"]["precision"]),
            "recall": float(report["macro avg"]["recall"]),
            "f1": float(report["macro avg"]["f1-score"]),
        },
        "weighted_avg": {
            "precision": float(report["weighted avg"]["precision"]),
            "recall": float(report["weighted avg"]["recall"]),
            "f1": float(report["weighted avg"]["f1-score"]),
        },
        "confusion_matrix": cm.tolist(),
    }

    # include ROC-ish info if proba available (optional)
    if y_proba is not None:
        metrics["note"] = "y_proba contains class probabilities in order [0,1,2]."

    return metrics


def add_predictions(df, y_pred, y_proba=None):
    out = df.copy()
    out["pred_label_id"] = y_pred
    out["pred_label"] = out["pred_label_id"].map(LABEL_MAP)
    if y_proba is not None:
        # store probabilities per class (rounded)
        proba = np.round(y_proba, 6)
        out["proba_negative"] = proba[:, 0]
        out["proba_neutral"]  = proba[:, 1]
        out["proba_positive"] = proba[:, 2]
    return out


def main():
    parser = argparse.ArgumentParser(description="Train TF-IDF + Logistic Regression for sentiment")
    parser.add_argument("--data_dir", default="src/data/sentiment_splits", help="Folder with train/val/test CSVs")
    parser.add_argument("--out_dir", default="src/sentiment-analysis/logistic-regression", help="Where to save artifacts")
    parser.add_argument("--max_features", type=int, default=10000, help="TF-IDF vocab size")
    parser.add_argument("--save_probs", action="store_true", help="Save per-class probabilities to prediction CSVs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading splits from {args.data_dir} ...")
    train, val, test = load_splits(args.data_dir)

    print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")
    print("Label distribution (train):", train["label"].value_counts().sort_index().to_dict())

    pipe = build_pipeline(max_features=args.max_features)

    print("\nFitting Logistic Regression on TF-IDF features ...")
    pipe.fit(train["text"].values, train["label"].values)

    # Validation
    print("\nEvaluating on validation set ...")
    y_val_pred = pipe.predict(val["text"].values)
    y_val_proba = pipe.predict_proba(val["text"].values) if args.save_probs else None
    val_metrics = evaluate(val["label"].values, y_val_pred, y_val_proba)
    val_pred_df = add_predictions(val, y_val_pred, y_val_proba)

    # Test
    print("Evaluating on test set ...")
    y_test_pred = pipe.predict(test["text"].values)
    y_test_proba = pipe.predict_proba(test["text"].values) if args.save_probs else None
    test_metrics = evaluate(test["label"].values, y_test_pred, y_test_proba)
    test_pred_df = add_predictions(test, y_test_pred, y_test_proba)

    # Save artifacts
    model_path = os.path.join(args.out_dir, "model.joblib")
    vect_path = os.path.join(args.out_dir, "vectorizer.joblib")  # kept for clarity even though pipeline holds it
    metrics_path = os.path.join(args.out_dir, "metrics.json")
    val_pred_path = os.path.join(args.out_dir, "val_predictions.csv")
    test_pred_path = os.path.join(args.out_dir, "test_predictions.csv")

    print("\nSaving artifacts ...")
    # Save the full pipeline (vectorizer + classifier)
    joblib.dump(pipe, model_path)
    # Save vectorizer separately 
    joblib.dump(pipe.named_steps["tfidf"], vect_path)

    all_metrics = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": "TF-IDF + LogisticRegression",
        "params": {
            "max_features": args.max_features,
            "class_weight": "balanced",
            "random_state": 42,
        },
        "val": val_metrics,
        "test": test_metrics,
        "label_map": LABEL_MAP,
    }
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    val_pred_df.to_csv(val_pred_path, index=False)
    test_pred_df.to_csv(test_pred_path, index=False)

    print("\n=== DONE ===")
    print(f"Saved:")
    print(f"  - {model_path}")
    print(f"  - {vect_path}")
    print(f"  - {metrics_path}")
    print(f"  - {val_pred_path}")
    print(f"  - {test_pred_path}")

    # Quick console summary
    print("\nQuick metrics (macro F1):")
    print(f"  Val  macro F1: {all_metrics['val']['macro_avg']['f1']:.4f}")
    print(f"  Test macro F1: {all_metrics['test']['macro_avg']['f1']:.4f}")


if __name__ == "__main__":
    main()

