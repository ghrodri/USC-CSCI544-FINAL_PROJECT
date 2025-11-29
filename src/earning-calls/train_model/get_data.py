from pathlib import Path
import hashlib
import re
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

OUTPUT_DIR = Path("./src/earning-calls/data")

def text_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

def clean_text(text: str) -> str:
    text = str(text).strip()
    text = text.replace("\u200b", "")
    text = re.sub(r"\s+", " ", text)
    return text

def load_financial_phrasebank():
    ds = load_dataset(
        "takala/financial_phrasebank",
        "sentences_allagree",
        trust_remote_code=True,
    )
    split = ds["train"]
    label_feat = split.features["label"]
    records = []
    for row in tqdm(split, desc="Loading Financial PhraseBank"):
        sent = clean_text(row["sentence"])
        label_val = row["label"]
        if hasattr(label_feat, "int2str"):
            label = label_feat.int2str(label_val)
        else:
            label = str(label_val)
        records.append(
            {
                "text": sent,
                "label": label,
                "source": "financial_phrasebank",
            }
        )
    return pd.DataFrame(records)

def load_aiera():
    ds = load_dataset("Aiera/aiera-transcript-sentiment")
    split_name = "train" if "train" in ds else "test"
    split = ds[split_name]
    records = []
    for row in tqdm(split, desc="Loading Aiera earnings calls"):
        text = clean_text(row["transcript"])
        label = str(row["sentiment"])
        records.append(
            {
                "text": text,
                "label": label,
                "source": "aiera_transcript",
            }
        )
    return pd.DataFrame(records)

def main():
    print("Loading datasets from HuggingFace...")
    df1 = load_financial_phrasebank()
    df2 = load_aiera()

    print("\nCombining datasets...")
    combined_df = pd.concat([df1, df2], ignore_index=True)
    print(f"Total samples before deduplication: {len(combined_df)}")

    combined_df["text"] = combined_df["text"].astype(str)
    combined_df = combined_df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"Total samples after deduplication: {len(combined_df)}")

    combined_df["id"] = combined_df["text"].apply(text_id)

    dup_ids = combined_df["id"].duplicated(keep=False)
    if dup_ids.any():
        for idx in combined_df[dup_ids].index:
            combined_df.at[idx, "id"] = text_id(
                combined_df.at[idx, "text"] + str(idx)
            )
    print("\nLabel distribution:")
    label_counts = combined_df["label"].value_counts()
    for label, count in label_counts.items():
        pct = count / len(combined_df) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")

    print("\nCreating train/val/test splits (70/15/15)...")
    train_df, temp_df = train_test_split(
        combined_df,
        test_size=0.30,
        stratify=combined_df["label"],
        random_state=42,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=42,
    )

    print(f"  Train size: {len(train_df)}")
    print(f"  Val size:   {len(val_df)}")
    print(f"  Test size:  {len(test_df)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    columns = ["id", "text", "label", "source"]

    train_df[columns].to_csv(OUTPUT_DIR / "train.csv", index=False)
    val_df[columns].to_csv(OUTPUT_DIR / "val.csv", index=False)
    test_df[columns].to_csv(OUTPUT_DIR / "test.csv", index=False)

    print("\nSaved CSV files to:", OUTPUT_DIR.resolve())
    print(f"Total examples in combined dataset: {len(combined_df)}")

if __name__ == "__main__":
    main()
