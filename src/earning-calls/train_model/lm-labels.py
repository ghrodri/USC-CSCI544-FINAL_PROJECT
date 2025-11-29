import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from MOD_Load_MasterDictionary_v2023 import load_masterdictionary
import re

BASE_DIR = Path("./src/earning-calls/data")
DICT_PATH = BASE_DIR / "Loughran-McDonald_MasterDictionary_1993-2024.csv"

INPUT_FILES = {
    "train": BASE_DIR / "train.csv",
    "val": BASE_DIR / "val.csv",
    "test": BASE_DIR / "test.csv",
}

FORWARD_LOOKING_CATEGORIES = ["strong_modal", "weak_modal"]
UNCERTAINTY_CATEGORY = "uncertainty"


def tokenize(text):
    text = text.upper()
    text = re.sub(r"[^A-Z0-9 ]+", " ", text)
    return text.split()


def build_category_sets(master_dict):
    cats = {
        "uncertainty": set(),
        "strong_modal": set(),
        "weak_modal": set(),
    }
    for word, obj in master_dict.items():
        if obj.uncertainty:
            cats["uncertainty"].add(word)
        if obj.strong_modal:
            cats["strong_modal"].add(word)
        if obj.weak_modal:
            cats["weak_modal"].add(word)
    return cats


def score_text(tokens, cat_sets):
    unc = 0
    sm = 0
    wm = 0
    for t in tokens:
        if t in cat_sets["uncertainty"]:
            unc += 1
        if t in cat_sets["strong_modal"]:
            sm += 1
        if t in cat_sets["weak_modal"]:
            wm += 1
    return unc, sm, wm


def label_from_score(score):
    if score == 0:
        return "none"
    elif score == 1:
        return "low"
    elif 2 <= score <= 3:
        return "medium"
    return "high"


def process_file(path, cat_sets):
    df = pd.read_csv(path)
    u_scores = []
    u_labels = []
    f_scores = []
    f_labels = []

    for text in tqdm(df["text"], desc=f"Processing {path.name}"):
        tokens = tokenize(str(text))
        unc, sm, wm = score_text(tokens, cat_sets)

        forward_score = sm + wm

        u_scores.append(unc)
        u_labels.append(label_from_score(unc))

        f_scores.append(forward_score)
        f_labels.append(label_from_score(forward_score))

    df["uncertainty_score"] = u_scores
    df["uncertainty_label"] = u_labels
    df["forward_looking_score"] = f_scores
    df["forward_looking_label"] = f_labels

    df.to_csv(path, index=False)
    print(f"Saved → {path}")
    return df


def main():
    print(f"Loading Loughran-McDonald dictionary from: {DICT_PATH}")
    master_dict = load_masterdictionary(str(DICT_PATH), print_flag=True, get_other=False)
    print("Dictionary loaded.")

    cat_sets = build_category_sets(master_dict)

    print("\nWords per category:")
    print(f"  uncertainty: {len(cat_sets['uncertainty'])}")
    print(f"  strong_modal: {len(cat_sets['strong_modal'])}")
    print(f"  weak_modal: {len(cat_sets['weak_modal'])}\n")

    for split, path in INPUT_FILES.items():
        print(f"\n=== Processing {split} ===")
        df = process_file(path, cat_sets)

        print("Label distributions:")
        print(df["uncertainty_label"].value_counts(), "\n")
        print(df["forward_looking_label"].value_counts(), "\n")


if __name__ == "__main__":
    main()