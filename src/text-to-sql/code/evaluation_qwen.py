import json
from pathlib import Path
import sys

import duckdb
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

# ============================
# PATHS
# ============================
BASE_MODEL = "Qwen/Qwen2.5-0.5B"
LORA_DIR = Path("src/text-to-sql/model/QWEN/Qwen")
TEST_DIR = Path("src/text-to-sql/ir-data/test")
DB_PATH = "src/db/financial.duckdb"

# Para import correcto
sys.path.append("./src/text-to-sql")
from semantic.ir_to_sql import ir_to_sql


# ============================
# HELPERS
# ============================
def build_prompt(question: str) -> str:
    return f"Question: {question}\nIR:\n"


def extract_json_block(text: str):
    start = None
    depth = 0

    for i, ch in enumerate(text):
        if ch == "{":
            if start is None:
                start = i
            depth += 1
        elif ch == "}":
            if start is not None:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i+1])
                    except json.JSONDecodeError:
                        return None
    return None


def normalize_assets(lst):
    return sorted(list(set(lst or [])))


def try_sql(con, sql: str):
    try:
        df = con.execute(sql).fetchdf()
        return True, df
    except Exception:
        return False, None


# ============================
# LOAD MODEL + DATA
# ============================
def load_model_and_tokenizer():
    print("[LOAD] Loading base model + tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    ).to(device)

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Loading LoRA from {LORA_DIR} ...")

    model = PeftModel.from_pretrained(base, str(LORA_DIR))
    model.to(device)
    model.eval()

    return model, tokenizer, device


def load_test_dataset():
    print("[LOAD] Loading test set...")
    test_files = list(TEST_DIR.glob("*_ir.jsonl"))
    if not test_files:
        raise RuntimeError(f"No *_ir.jsonl found in {TEST_DIR}")

    dset = load_dataset("json", data_files=[str(p) for p in test_files], split="train")
    print(f"[INFO] Test examples: {len(dset)}")

    MAX_EVAL = 1000
    if len(dset) > MAX_EVAL:
        dset = dset.shuffle(seed=42).select(range(MAX_EVAL))
        print(f"[INFO] Using subset of {len(dset)} examples for evaluation")

    return dset

def main():
    model, tokenizer, device = load_model_and_tokenizer()
    dset = load_test_dataset()
    con = duckdb.connect(DB_PATH, read_only=True)

    N = len(dset)

    exact_matches = 0
    intent_correct = 0
    metric_correct = 0
    time_correct = 0

    asset_precisions = []
    asset_recalls = []
    asset_f1s = []

    sql_success = 0
    sql_result_match = 0

    for ex in tqdm(dset, desc="Evaluating", ncols=100):
        question = ex["question"]
        gold_ir = ex["ir"]

        # ===== Generate IR =====
        prompt = build_prompt(question)
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)

        with torch.no_grad():
            out_ids = model.generate(
                **enc,
                max_new_tokens=256,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(out_ids[0], skip_special_tokens=True)

        # Remove the prompt prefix
        tail = decoded[len(prompt):] if decoded.startswith(prompt) else decoded
        tail = tail.strip()

        pred_ir = extract_json_block(tail)
        if pred_ir is None:
            pred_ir = {}

        if pred_ir == gold_ir:
            exact_matches += 1

        if pred_ir.get("intent") == gold_ir.get("intent"):
            intent_correct += 1

        if (pred_ir.get("metric") or {}).get("name") == (gold_ir.get("metric") or {}).get("name"):
            metric_correct += 1

        if pred_ir.get("time") == gold_ir.get("time"):
            time_correct += 1

        gold_assets = normalize_assets((gold_ir.get("entities") or {}).get("assets"))
        pred_assets = normalize_assets((pred_ir.get("entities") or {}).get("assets"))

        all_assets = sorted(list(set(gold_assets + pred_assets)))
        if all_assets:
            y_true = [1 if a in gold_assets else 0 for a in all_assets]
            y_pred = [1 if a in pred_assets else 0 for a in all_assets]

            asset_precisions.append(precision_score(y_true, y_pred, zero_division=0))
            asset_recalls.append(recall_score(y_true, y_pred, zero_division=0))
            asset_f1s.append(f1_score(y_true, y_pred, zero_division=0))

        try:
            pred_sql = ir_to_sql(pred_ir)
        except Exception:
            continue

        ok_pred, df_pred = try_sql(con, pred_sql)
        if ok_pred:
            sql_success += 1

        try:
            gold_sql = ir_to_sql(gold_ir)
            ok_gold, df_gold = try_sql(con, gold_sql)

            if ok_pred and ok_gold and df_pred.equals(df_gold):
                sql_result_match += 1
        except Exception:
            pass

    con.close()
    print("\n=========== FINAL METRICS ===========")
    print(f"Total examples:           {N}")
    print("-------------------------------------")
    print(f"IR Exact Match:           {exact_matches / N:.3f}")
    print(f"Intent Accuracy:          {intent_correct / N:.3f}")
    print(f"Metric Accuracy:          {metric_correct / N:.3f}")
    print(f"Time Accuracy:            {time_correct / N:.3f}")

    if asset_precisions:
        print(f"Assets Precision (avg):   {np.mean(asset_precisions):.3f}")
        print(f"Assets Recall (avg):      {np.mean(asset_recalls):.3f}")
        print(f"Assets F1 (avg):          {np.mean(asset_f1s):.3f}")
    else:
        print("Assets metrics:           (no assets in test set)")

    print("-------------------------------------")
    print(f"SQL Execution Success:    {sql_success / N:.3f}")
    print(f"SQL Result-Set Match:     {sql_result_match / N:.3f}")
    print("=====================================\n")


if __name__ == "__main__":
    main()