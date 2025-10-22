import os, re, json, math, random, time
from typing import List, Dict, Any, Tuple
from collections import Counter
import numpy as np
import pandas as pd
import duckdb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    get_linear_schedule_with_warmup,
)


# hyperparameters

EPOCHS= 40 # number of epochs
BATCH_SIZE= 8 # batch size (seq2seq models are heavier)
LEARNING_RATE= 3e-4 # learning rate
WEIGHT_DECAY= 0.05  # strong L2 regularization
DROPOUT  = 0.2 # T5 dropout
LABEL_SMOOTHING  = 0.1   # label smoothing for seq loss
WARMUP_FRAC= 0.06  # fraction of steps for warmup
MAX_INPUT_LEN= 256   # max source tokens
MAX_TARGET_LEN= 196   # max target tokens
BEAM_SIZE= 4 # beam size for decoding
EARLY_STOP_PAT= 5 # early stopping patience (epochs without val_exec improvement)

SEED= 42 # random seed
MAX_VAL_STEPS= None # limit val batches
MAX_TEST_STEPS= None # limit test batches

# paths (keep *_DATA style)
MODEL_NAME= "t5-small"
TRAIN_DATA= "./src/text-to-sql/data/train.jsonl"
VAL_DATA = "./src/text-to-sql/data/val.jsonl"
TEST_DATA= "./src/text-to-sql/data/test.jsonl"
META_DATA= "./src/text-to-sql/data/dataset_metadata.json"
DB_DATA  = "./src/db/financial.duckdb"
CHECKP   = "./src/text-to-sql/model/T5/ckpt/"
EVAL_ROOT= "./src/evl/T5/"
PLOTS_DIR= "./src/evl/T5/plots"
REPORTS_DIR= "./src/evl/T5/reports"
os.makedirs(CHECKP, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def read_jsonl_file(p: str):
    xs = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                xs.append(json.loads(line))
    return xs

def normalize_sql(s: str) -> str:
    if not s:
        return s
    s = s.strip().rstrip(";")
    s = re.sub(r"\s+", " ", s)
    s = s.replace("`", '"').lower()
    m = re.search(r"(select\s+)([^;]+?)(\s+from\s+)", s)
    if m:
        head = m.group(1)
        cols = m.group(2)
        tail = m.group(3)
        parts = cols.split(",")
        clean = []
        for i in range(len(parts)):
            clean.append(parts[i].strip())
        clean.sort()
        s = s[:m.start()] + head + ", ".join(clean) + tail + s[m.end():]
    return s.strip()

def try_execute_sql(con, s: str):
    try:
        df = con.execute(s).df()
        return True, df, ""
    except Exception as e:
        return False, pd.DataFrame(), str(e)

def dataframe_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    try:
        if set(a.columns) != set(b.columns):
            return False
        cols = list(sorted(a.columns))
        a2 = a[cols].sort_values(by=cols).reset_index(drop=True)
        b2 = b[cols].sort_values(by=cols).reset_index(drop=True)
        return a2.equals(b2)
    except Exception:
        return False

class Q2SQLDataset(Dataset):
    def __init__(self, rows, tokenizer: T5TokenizerFast, max_in=256, max_tg=196):
        self.rows = rows
        self.tok = tokenizer
        self.max_in = max_in
        self.max_tg = max_tg
        self.src = []
        self.tgt = []
        self.intent = []
        for i in range(len(rows)):
            r = rows[i]
            q = str(r.get("question", "")).strip()
            s = normalize_sql(str(r.get("sql", "")).strip())
            it = str(r.get("intent", "")).strip()
            src = f"translate to sql: {q}" 
            self.src.append(src)
            self.tgt.append(s)
            self.intent.append(it)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return {
            "src": self.src[i],
            "tgt": self.tgt[i],
            "intent": self.intent[i],
            "q": self.rows[i].get("question", "")
        }

def collate_batch(batch, tok: T5TokenizerFast, max_in: int, max_tg: int):
    # encode inputs
    src_texts = [b["src"] for b in batch]
    enc = tok(
        src_texts,
        padding=True,
        truncation=True,
        max_length=max_in,
        return_tensors="pt"
    )
    tgt_texts = [b["tgt"] for b in batch]
    tgt = tok(
        text_target=tgt_texts,
        padding=True,
        truncation=True,
        max_length=max_tg,
        return_tensors="pt"
    )
    labels = tgt["input_ids"]
    labels[labels == tok.pad_token_id] = -100

    qs = [b["q"] for b in batch]
    gsql = [b["tgt"] for b in batch]
    its = [b["intent"] for b in batch]

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": labels,
        "q": qs,
        "sql": gsql,
        "intent": its
    }

# evaluation
def evaluate_model(m, tok, dl, con, dev="cpu", max_steps=None, beams=4, max_len=196):
    m.eval()
    em_hit = 0
    em_tot = 0
    ex_hit = 0
    ex_tot = 0
    fails = []
    per_intent_fail = Counter()

    step = 0
    with torch.no_grad():
        bar = tqdm(dl, desc="Validation", leave=False)
        for b in bar:
            ids = b["input_ids"].to(dev)
            att = b["attention_mask"].to(dev)

            out = m.generate(
                input_ids=ids,
                attention_mask=att,
                max_length=max_len,
                num_beams=beams,
                early_stopping=True
            )
            preds = tok.batch_decode(out, skip_special_tokens=True)

            for j in range(len(preds)):
                ps = normalize_sql(preds[j])
                gs = normalize_sql(b["sql"][j])

                em_tot += 1
                if ps == gs:
                    em_hit += 1

                ok_g, df_g, _ = try_execute_sql(con, gs)
                ok_p, df_p, err_p = try_execute_sql(con, ps)
                if ok_g:
                    ex_tot += 1
                    if ok_p and dataframe_equal(df_g, df_p):
                        ex_hit += 1
                    else:
                        fails.append({
                            "question": b["q"][j],
                            "intent_gold": b["intent"][j],
                            "sql_gold": gs,
                            "sql_pred": ps,
                            "exec_err_pred": err_p
                        })
                        per_intent_fail[b["intent"][j]] += 1

            step += 1
            if (max_steps is not None) and (step >= max_steps):
                break

    em = em_hit / max(1, em_tot)
    ex = ex_hit / max(1, ex_tot)
    cov = ex_tot / max(1, em_tot)
    return em, ex, cov, fails, per_intent_fail

# training
def smooth_ce_loss(logits, labels, eps, ignore_index=-100):
    mask = (labels != ignore_index)
    if mask.sum() == 0:
        return logits.new_tensor(0.0)

    B, T, V = logits.size()
    flat_logits = logits.view(B*T, V)
    flat_labels = labels.view(B*T)

    valid = (flat_labels != ignore_index)
    flat_logits = flat_logits[valid]
    flat_labels = flat_labels[valid]

    logp = torch.log_softmax(flat_logits, dim=-1)
    nll = -logp.gather(1, flat_labels.unsqueeze(1)).squeeze(1)
    smooth = -logp.mean(dim=-1)
    loss = (1.0 - eps) * nll + eps * smooth
    return loss.mean()

def train_model(m, tok, dl_tr, dl_va, con, dev="cpu", epochs=1, lr=3e-4):
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    total_steps = len(dl_tr) * max(1, epochs)
    warmup = int(WARMUP_FRAC * total_steps)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=total_steps)

    loss_hist = []
    val_em_hist = []
    val_exec_hist = []
    cov_hist = []
    lr_hist = []

    best_val_exec = -1.0
    no_improve = 0

    for ep in range(1, epochs+1):
        m.train()
        losses = []
        bar = tqdm(dl_tr, desc="Training", leave=False)
        for b in bar:
            ids = b["input_ids"].to(dev)
            att = b["attention_mask"].to(dev)
            y = b["labels"].to(dev)

            # teacher forcing: provide decoder_input_ids explicitly to avoid error
            dec_inp = m.prepare_decoder_input_ids_from_labels(labels=y)

            out = m(input_ids=ids, attention_mask=att, decoder_input_ids=dec_inp)
            logits = out.logits
            loss = smooth_ce_loss(logits, y, LABEL_SMOOTHING, ignore_index=-100)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            sched.step()

            losses.append(loss.item())
            lr_hist.append(sched.get_last_lr()[0])

        em, ex, cov, _, _ = evaluate_model(
            m, tok, dl_va, con, dev=dev, max_steps=MAX_VAL_STEPS, beams=BEAM_SIZE, max_len=MAX_TARGET_LEN
        )

        torch.save(m.state_dict(), os.path.join(CHECKP, "last.pt"))

        if ex > best_val_exec:
            best_val_exec = ex
            no_improve = 0
            torch.save(m.state_dict(), os.path.join(CHECKP, "best.pt"))
        else:
            no_improve += 1

        loss_hist.append(float(np.mean(losses)))
        val_em_hist.append(em)
        val_exec_hist.append(ex)
        cov_hist.append(cov)

        print(f"Epoch {ep}: loss={loss_hist[-1]:.4f} val_em={em:.3f} val_exec={ex:.3f} gold_exec_cov={cov:.2f}")

        with open(os.path.join(REPORTS_DIR, "metrics_per_epoch.txt"), "a", encoding="utf-8") as f:
            f.write(f"Epoch {ep}: loss={loss_hist[-1]:.6f} val_em={em:.6f} val_exec={ex:.6f} gold_exec_cov={cov:.6f}\n")

        if no_improve >= EARLY_STOP_PAT:
            print(f"Early stopping triggered at Epoch {ep} (no improvement for {EARLY_STOP_PAT} epochs).")
            break

    return {
        "loss": loss_hist,
        "val_em": val_em_hist,
        "val_exec": val_exec_hist,
        "cov": cov_hist,
        "lr": lr_hist
    }

# plots
def plot_line(xs, ys, title, xlabel, ylabel, out_path):
    plt.figure(figsize=(6,4))
    plt.plot(xs, ys, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_multi(xs, ys_list, labels, title, xlabel, ylabel, out_path):
    plt.figure(figsize=(6,4))
    for i in range(len(ys_list)):
        plt.plot(xs, ys_list[i], marker="o", label=labels[i])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_bar(labels, vals, title, xlabel, ylabel, out_path):
    plt.figure(figsize=(6,4))
    pos = np.arange(len(labels))
    plt.bar(pos, vals)
    plt.xticks(pos, labels, rotation=30, ha="right")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# load metadata
db_path = DB_DATA
try:
    meta = json.load(open(META_DATA, "r", encoding="utf-8"))
    if isinstance(meta, dict) and ("db_path" in meta):
        db_path = meta["db_path"]
except Exception:
    pass

# load data
tr_rows = read_jsonl_file(TRAIN_DATA) if os.path.exists(TRAIN_DATA) else []
va_rows = read_jsonl_file(VAL_DATA) if os.path.exists(VAL_DATA) else []
te_rows = read_jsonl_file(TEST_DATA) if os.path.exists(TEST_DATA) else []
if (not tr_rows) and (not va_rows):
    raise RuntimeError("Missing dataset under ./src/text-to-sql/data/")


# db schema connection
con = duckdb.connect(db_path, read_only=True)

# tokenizer and model
tok = T5TokenizerFast.from_pretrained(MODEL_NAME)
mdl = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# increase dropout for regularization
mdl.config.dropout_rate = DROPOUT
mdl.config.attention_dropout_rate = DROPOUT
mdl.to(DEVICE)

# dataloaders
ds_tr = Q2SQLDataset(tr_rows, tok, max_in=MAX_INPUT_LEN, max_tg=MAX_TARGET_LEN)
ds_va = Q2SQLDataset(va_rows if va_rows else tr_rows, tok, max_in=MAX_INPUT_LEN, max_tg=MAX_TARGET_LEN)
ds_te = Q2SQLDataset(te_rows if te_rows else va_rows, tok, max_in=MAX_INPUT_LEN, max_tg=MAX_TARGET_LEN)

def collate_fn(batch):
    return collate_batch(batch, tok, MAX_INPUT_LEN, MAX_TARGET_LEN)

dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# train
logs = train_model(mdl, tok, dl_tr, dl_va, con, dev=DEVICE, epochs=EPOCHS, lr=LEARNING_RATE)

# load best and test
best_path = os.path.join(CHECKP, "best.pt")
if os.path.exists(best_path):
    mdl.load_state_dict(torch.load(best_path, map_location=DEVICE))

em, ex, cov, fails, fail_by_intent = evaluate_model(
    mdl, tok, dl_te, con, dev=DEVICE, max_steps=MAX_TEST_STEPS, beams=BEAM_SIZE, max_len=MAX_TARGET_LEN
)
print(f"Testing: em={em:.3f} exec={ex:.3f} gold_exec_cov={cov:.2f}")

with open(os.path.join(REPORTS_DIR, "test_metrics.txt"), "w", encoding="utf-8") as f:
    f.write(f"em={em:.6f} exec={ex:.6f} gold_exec_cov={cov:.6f}\n")

with open(os.path.join(REPORTS_DIR, "failures.txt"), "w", encoding="utf-8") as f:
    for i in range(len(fails)):
        exi = fails[i]
        f.write("Q: " + exi["question"] + "\n")
        f.write("GOLD_INTENT: " + exi["intent_gold"] + "\n")
        f.write("GOLD_SQL: " + exi["sql_gold"] + "\n")
        f.write("PRED_SQL: " + exi["sql_pred"] + "\n")
        if exi.get("exec_err_pred"):
            f.write("PRED_ERR: " + exi["exec_err_pred"] + "\n")
        f.write("---\n")

# plots
xs = list(range(1, len(logs["loss"])+1))
plot_line(xs, logs["loss"], "Training Loss", "Epoch", "Loss", os.path.join(PLOTS_DIR, "loss.png"))
plot_line(xs, logs["val_em"], "Validation EM", "Epoch", "EM", os.path.join(PLOTS_DIR, "val_em.png"))
plot_line(xs, logs["val_exec"], "Validation ExecAcc", "Epoch", "ExecAcc", os.path.join(PLOTS_DIR, "val_exec.png"))

ys_list = [logs["val_em"], logs["val_exec"]]
labels = ["EM", "ExecAcc"]
plot_multi(xs, ys_list, labels, "Seq2Seq Metrics", "Epoch", "Score", os.path.join(PLOTS_DIR, "head_accuracies.png"))

if len(logs.get("lr", [])) > 0:
    lx = list(range(1, len(logs["lr"])+1))
    plot_line(lx, logs["lr"], "Learning Rate", "Step", "LR", os.path.join(PLOTS_DIR, "lr.png"))