from pathlib import Path
import json
import pandas as pd
from collections import Counter

DATA_DIR = Path("/Users/ghrodri/Local/Final_Project/src/text-to-sql/data")

def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def check_duplicates(df: pd.DataFrame):
    dup_sql = df[df.duplicated("sql", keep=False)]
    dup_question = df[df.duplicated("question", keep=False)]
    return dup_sql, dup_question

def summarize_intents(df: pd.DataFrame, name: str):
    counts = Counter(df["intent"])
    total = len(df)
    print(f"\n{name.upper()} ({total} examples)")
    for intent, c in sorted(counts.items()):
        print(f"  {intent:<25} {c:5d} ({c/total:.2%})")

def main():
    parts = {}
    for split in ["train", "val", "test"]:
        path = DATA_DIR / f"{split}.jsonl"
        rows = load_jsonl(path)
        parts[split] = pd.DataFrame(rows)
    
    all_df = pd.concat(parts.values(), ignore_index=True)

    # duplicates
    dup_sql, dup_q = check_duplicates(all_df)
    print("=== DUPLICATE CHECKS ===")
    print(f"Total examples: {len(all_df)}")
    print(f"Duplicate SQLs: {len(dup_sql)}")
    print(f"Duplicate questions: {len(dup_q)}")
    if len(dup_sql):
        print("\nSample duplicate SQLs:")
        print(dup_sql[["intent", "question"]].head(5).to_string(index=False))
    if len(dup_q):
        print("\nSample duplicate questions:")
        print(dup_q[["intent", "question"]].head(5).to_string(index=False))
    
    # balance summary
    print("\n=== INTENT DISTRIBUTION ===")
    for split, df in parts.items():
        summarize_intents(df, split)
    print("\nCombined:")
    summarize_intents(all_df, "all")

if __name__ == "__main__":
    main()