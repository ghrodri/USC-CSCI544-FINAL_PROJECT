import os
import re
import json
from pathlib import Path
from typing import Dict, List

import torch
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset, DatasetDict, load_dataset
import duckdb

DATA_DIR = Path("/src/text-to-sql/data")
TRAIN_PATH = DATA_DIR / "train.jsonl"
VAL_PATH   = DATA_DIR / "val.jsonl"
TEST_PATH  = DATA_DIR / "test.jsonl"
OUTPUT_DIR = Path("/src/text-to-sql/model/T5/outputs")
DB_PATH    = Path("/src/db/financial.duckdb")

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

train_rows = read_jsonl(TRAIN_PATH)
val_rows   = read_jsonl(VAL_PATH)
test_rows  = read_jsonl(TEST_PATH)

def rows_to_hfds(rows: List[Dict]) -> Dataset:
    return Dataset.from_list([{"input_text": r["question"], "target_text": r["sql"]} for r in rows])

hfds = DatasetDict({
    "train": rows_to_hfds(train_rows),
    "validation": rows_to_hfds(val_rows),
    "test": rows_to_hfds(test_rows),
})

MODEL_NAME = "t5-small"
tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

INPUT_PREFIX = "translate question to sql: "

MAX_SOURCE_LEN = 128
MAX_TARGET_LEN = 192

def preprocess(batch):
    inputs = [INPUT_PREFIX + q for q in batch["input_text"]]
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_SOURCE_LEN,
        padding="max_length",
        truncation=True,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["target_text"],
            max_length=MAX_TARGET_LEN,
            padding="max_length",
            truncation=True,
        )["input_ids"]
    labels = [[(tok if tok != tokenizer.pad_token_id else -100) for tok in seq] for seq in labels]
    model_inputs["labels"] = labels
    return model_inputs

tokenized = hfds.map(preprocess, batched=True, remove_columns=hfds["train"].column_names)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

duck_con_eval = duckdb.connect(DB_PATH.as_posix(), read_only=True)

def normalize_sql(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace('"', "'")
    return s.lower()

def exec_sql(sql_text: str):
    try:
        df = duck_con_eval.sql(sql_text).df()
        df.columns = [c.lower() for c in df.columns]
        df = df.reindex(sorted(df.columns), axis=1)
        if len(df) > 5000:
            df = df.head(5000)
        df = df.sort_values(by=list(df.columns)).reset_index(drop=True)
        return True, df
    except Exception:
        return False, None

def component_hits(pred: str, gold: str) -> Dict[str, int]:
    pred = normalize_sql(pred)
    gold = normalize_sql(gold)
    keys = ["select", "from", "where", "group by", "order by"]
    out = {}
    for k in keys:
        out[k] = int((k in pred) == (k in gold))
    return out

def compute_text_metrics(pred_texts: List[str], gold_texts: List[str]) -> Dict[str, float]:
    n = len(pred_texts)
    em = ex = valid = 0
    comp_sum = {"select": 0, "from": 0, "where": 0, "group by": 0, "order by": 0}

    gold_cache = {}

    for p, g in zip(pred_texts, gold_texts):
        p_n = normalize_sql(p)
        g_n = normalize_sql(g)

        if p_n == g_n:
            em += 1

        ok_p, df_p = exec_sql(p_n)
        if ok_p:
            valid += 1

        if g_n not in gold_cache:
            ok_g, df_g = exec_sql(g_n)
            gold_cache[g_n] = (ok_g, df_g)
        else:
            ok_g, df_g = gold_cache[g_n]

        if ok_p and ok_g:
            try:
                if list(df_p.columns) == list(df_g.columns) and df_p.shape == df_g.shape and df_p.equals(df_g):
                    ex += 1
            except Exception:
                pass

        ch = component_hits(p, g)
        for k in comp_sum:
            comp_sum[k] += ch[k]

    comp_f1 = sum(comp_sum.values()) / (len(comp_sum) * max(1, n))
    return {
        "em": em / max(1, n),
        "ex": ex / max(1, n),
        "valid_sql": valid / max(1, n),
        "component_f1": comp_f1,
    }

def compute_metrics(eval_pred):
    pred_ids, labels = eval_pred
    labels = [[(tid if tid != -100 else tokenizer.pad_token_id) for tid in seq] for seq in labels]
    pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    gold_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return compute_text_metrics(pred_texts, gold_texts)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

args = TrainingArguments(
    output_dir=OUTPUT_DIR.as_posix(),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=100,
    per_device_train_batch_size=8, 
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,  
    num_train_epochs=10,
    learning_rate=3e-4,
    weight_decay=0.01,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LEN,
    load_best_model_at_end=True,
    metric_for_best_model="em",
    greater_is_better=True,
    save_total_limit=2,
    fp16=False,
    report_to=[],
)

trainer = Trainer(
    model=model.to(device),
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print(f"Device: {device.type.upper()}")
print(f"Data: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
print("Starting fine-tuning T5-small…")

trainer.train()

print("Evaluating validation set…")
val_metrics = trainer.evaluate(eval_dataset=tokenized["validation"])
print({k: round(float(v), 4) for k, v in val_metrics.items() if isinstance(v, (int, float))})

print("Evaluating test set…")
test_gen = trainer.predict(test_dataset=tokenized["test"], metric_key_prefix="test")
test_metrics = test_gen.metrics
print({k: round(float(v), 4) for k, v in test_metrics.items() if isinstance(v, (int, float))})

trainer.save_model(OUTPUT_DIR.as_posix())
tokenizer.save_pretrained(OUTPUT_DIR.as_posix())
print(f"Saved to {OUTPUT_DIR}")