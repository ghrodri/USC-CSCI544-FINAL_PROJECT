import json
import random
import re
from pathlib import Path

import duckdb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

DATA_DIR = Path("/src/text-to-sql/data")
TRAIN_PATH = DATA_DIR / "train.jsonl"
VAL_PATH   = DATA_DIR / "val.jsonl"
TEST_PATH  = DATA_DIR / "test.jsonl"
DB_PATH    = Path("/src/db/financial.duckdb")
CKPT_DIR   = Path("/src/text-to-sql/model/LSTM/checkpoints")

EPOCHS = 20
BATCH_SIZE = 32
EMBEDDING_DIM = 192
HIDDEN_DIM = 384
LEARNING_RATE = 3e-4
TEACHER_FORCING_RATIO = 0.45
MAX_DECODE_LENGTH = 192
RANDOM_SEED = 123
NUM_WORKERS = 0

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def tokenize(text: str):
    text = text.strip()
    text = re.sub(r"([(),.;=<>*/+\-])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text)
    return text.split()


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


class Vocab:
    def __init__(self, texts, min_freq=1, max_size=None):
        counts = {}
        for t in texts:
            for tok in tokenize(t):
                counts[tok] = counts.get(tok, 0) + 1

        special = ["<pad>", "<unk>", "<s>", "</s>"]
        tokens = [tok for tok, c in counts.items() if c >= min_freq]
        tokens.sort(key=lambda x: (-counts[x], x))
        if max_size:
            tokens = tokens[: max(0, max_size - len(special))]

        self.index_to_token = special + tokens
        self.token_to_index = {w: i for i, w in enumerate(self.index_to_token)}

    def encode(self, text, add_start_end=False):
        toks = tokenize(text)
        if add_start_end:
            toks = ["<s>"] + toks + ["</s>"]
        unk = self.token_to_index["<unk>"]
        return [self.token_to_index.get(t, unk) for t in toks]

    def decode_ids(self, ids):
        return [self.index_to_token[i] for i in ids]

    def pad_id(self): return self.token_to_index["<pad>"]
    def sos_id(self): return self.token_to_index["<s>"]
    def eos_id(self): return self.token_to_index["</s>"]
    def __len__(self): return len(self.index_to_token)


class Text2SQLDataset(Dataset):
    def __init__(self, rows, src_vocab: Vocab, tgt_vocab: Vocab):
        self.rows = rows
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        q = self.rows[idx]["question"]
        s = self.rows[idx]["sql"]
        src = torch.tensor(self.src_vocab.encode(q, add_start_end=False), dtype=torch.long)
        tgt = torch.tensor(self.tgt_vocab.encode(s, add_start_end=True), dtype=torch.long)
        return src, tgt


def collate_pad(batch, pad_src, pad_tgt):
    srcs, tgts = zip(*batch)
    max_s = max(len(x) for x in srcs)
    max_t = max(len(y) for y in tgts)
    ps = torch.full((len(batch), max_s), pad_src, dtype=torch.long)
    pt = torch.full((len(batch), max_t), pad_tgt, dtype=torch.long)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        ps[i, : len(s)] = s
        pt[i, : len(t)] = t
    return ps, pt


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb, hid):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, batch_first=True)

    def forward(self, x):
        emb = self.embedding(x)
        out, (h, c) = self.lstm(emb)
        return out, (h, c)


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb, hid):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, batch_first=True)
        self.proj = nn.Linear(hid, vocab_size)

    def forward(self, inp, state):
        emb = self.embedding(inp)
        out, new_state = self.lstm(emb, state)
        logits = self.proj(out)
        return logits, new_state


class Seq2Seq(nn.Module):
    def __init__(self, enc: Encoder, dec: Decoder):
        super().__init__()
        self.enc = enc
        self.dec = dec

    def forward(self, src, tgt, teacher_forcing_ratio=0.0, start_id=2):
        _, state = self.enc(src)
        B, T = tgt.size()
        V = self.dec.proj.out_features
        logits_all = torch.zeros(B, T, V, device=src.device)
        dec_in = torch.full((B, 1), start_id, dtype=torch.long, device=src.device)
        for t in range(1, T):
            logits, state = self.dec(dec_in, state)
            logits_all[:, t, :] = logits[:, -1, :]
            gold = tgt[:, t].unsqueeze(1)
            pred = logits[:, -1, :].argmax(-1, keepdim=True)
            use_tf = torch.rand(B, device=src.device) < teacher_forcing_ratio
            dec_in = torch.where(use_tf, gold, pred)
        return logits_all


def greedy_decode(model: Seq2Seq, src, tgt_vocab: Vocab, max_len=128):
    model.eval()
    with torch.no_grad():
        _, state = model.enc(src)
        dec_in = torch.full((src.size(0), 1), tgt_vocab.sos_id(), dtype=torch.long, device=src.device)
        outs = []
        for _ in range(max_len):
            logits, state = model.dec(dec_in, state)
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)
            outs.append(nxt)
            dec_in = nxt
        ids = torch.cat(outs, dim=1)
    return ids


def ids_to_text(ids_tensor, vocab: Vocab):
    texts = []
    for seq in ids_tensor.tolist():
        tokens = []
        for t in seq:
            tok = vocab.index_to_token[t]
            if tok == "</s>":
                break
            tokens.append(tok)
        texts.append(" ".join(tokens))
    return texts


def norm_sql(s: str):
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace('"', "'")
    return s.lower()


def exec_sql(con, sql: str):
    try:
        df = con.sql(sql).df()
        cols = [c.lower() for c in df.columns]
        df.columns = cols
        df = df.reindex(sorted(df.columns), axis=1)
        if len(df) > 5000:
            df = df.head(5000)
        df = df.sort_values(by=list(df.columns)).reset_index(drop=True)
        return True, df
    except Exception:
        return False, None


def metrics_on_predictions(pred_texts, gold_texts, con):
    assert len(pred_texts) == len(gold_texts)
    n = len(pred_texts)
    em = 0
    ex = 0
    val = 0
    comp_hits = {"select": 0, "from": 0, "where": 0, "group_by": 0, "order_by": 0}

    def clauses(x):
        x = norm_sql(x)
        return {
            "select": "select" in x,
            "from": "from" in x,
            "where": "where" in x,
            "group_by": "group by" in x,
            "order_by": "order by" in x,
        }

    gold_exec_cache = {}

    for i in range(n):
        p = norm_sql(pred_texts[i])
        g = norm_sql(gold_texts[i])

        if p == g:
            em += 1

        ok_pred, pred_df = exec_sql(con, p)
        if ok_pred:
            val += 1

        if g not in gold_exec_cache:
            ok_gold, gold_df = exec_sql(con, g)
            gold_exec_cache[g] = (ok_gold, gold_df)
        else:
            ok_gold, gold_df = gold_exec_cache[g]

        if ok_pred and ok_gold:
            try:
                same_cols = list(pred_df.columns) == list(gold_df.columns)
                same_shape = pred_df.shape == gold_df.shape
                same_vals = pred_df.equals(gold_df)
                if same_cols and same_shape and same_vals:
                    ex += 1
            except Exception:
                pass

        pc = clauses(p)
        gc = clauses(g)
        for k in comp_hits.keys():
            if pc[k] == gc[k]:
                comp_hits[k] += 1

    em_rate = em / max(1, n)
    ex_rate = ex / max(1, n)
    val_rate = val / max(1, n)
    comp_f1 = sum(comp_hits.values()) / (len(comp_hits) * max(1, n))
    return em_rate, ex_rate, val_rate, comp_f1, comp_hits


def save_checkpoint(path: Path, model: nn.Module, src_vocab: Vocab, tgt_vocab: Vocab, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "model_state": model.state_dict(), 
        "src_vocab": src_vocab.index_to_token,
        "tgt_vocab": tgt_vocab.index_to_token,
        "meta": meta,
    }
    torch.save(payload, path.as_posix())


def load_state_dict_flexible(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            return ckpt_obj["model"]
        if "model_state" in ckpt_obj and isinstance(ckpt_obj["model_state"], dict):
            return ckpt_obj["model_state"]
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
        if all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return ckpt_obj
    raise KeyError("No model state dict found in checkpoint")


def train_lstm():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if device.type == "mps":
        torch.mps.manual_seed(RANDOM_SEED)

    train_rows = read_jsonl(TRAIN_PATH)
    val_rows   = read_jsonl(VAL_PATH)
    test_rows  = read_jsonl(TEST_PATH)

    print(f"Device: {device.type.upper()}")
    print(f"Data: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

    src_texts = [r["question"] for r in (train_rows + val_rows + test_rows)]
    tgt_texts = [r["sql"] for r in (train_rows + val_rows + test_rows)]
    src_vocab = Vocab(src_texts, min_freq=1, max_size=None)
    tgt_vocab = Vocab(tgt_texts, min_freq=1, max_size=None)

    print(f"Vocab: src={len(src_vocab)} tgt={len(tgt_vocab)}")

    train_ds = Text2SQLDataset(train_rows, src_vocab, tgt_vocab)
    val_ds   = Text2SQLDataset(val_rows,   src_vocab, tgt_vocab)
    test_ds  = Text2SQLDataset(test_rows,  src_vocab, tgt_vocab)

    collate = lambda b: collate_pad(b, src_vocab.pad_id(), tgt_vocab.pad_id())
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=NUM_WORKERS)

    enc = Encoder(len(src_vocab), EMBEDDING_DIM, HIDDEN_DIM)
    dec = Decoder(len(tgt_vocab), EMBEDDING_DIM, HIDDEN_DIM)
    model = Seq2Seq(enc, dec).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {total_params:,}")
    print("Training start")

    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_id())

    best_em = -1.0
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_path = CKPT_DIR / "lstm_best.pt"
    last_path = CKPT_DIR / "lstm_last.pt"

    con_val = duckdb.connect(DB_PATH.as_posix(), read_only=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for src_b, tgt_b in tqdm(train_loader, desc=f"epoch {epoch:02d} training", leave=False):
            src_b = src_b.to(device)
            tgt_b = tgt_b.to(device)

            opt.zero_grad()
            logits = model(src_b, tgt_b, teacher_forcing_ratio=TEACHER_FORCING_RATIO, start_id=tgt_vocab.sos_id())

            gold = tgt_b[:, 1:].contiguous()
            pred = logits[:, 1:, :].contiguous()
            loss = loss_fn(pred.view(-1, pred.size(-1)), gold.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        pred_texts, gold_texts = [], []
        with torch.no_grad():
            for src_b, tgt_b in tqdm(val_loader, desc=f"epoch {epoch:02d} validating", leave=False):
                src_b = src_b.to(device)
                ids = greedy_decode(model, src_b, tgt_vocab, max_len=MAX_DECODE_LENGTH)
                pred_texts += ids_to_text(ids, tgt_vocab)
                gold_texts += ids_to_text(tgt_b[:, 1:].to(device), tgt_vocab)

        val_em, val_ex, val_valid, val_f1, _ = metrics_on_predictions(pred_texts, gold_texts, con_val)

        save_checkpoint(
            last_path, model, src_vocab, tgt_vocab,
            {"epoch": epoch, "val_em": val_em, "val_ex": val_ex, "val_valid": val_valid, "val_f1": val_f1}
        )

        if val_em > best_em:
            best_em = val_em
            save_checkpoint(
                best_path, model, src_vocab, tgt_vocab,
                {"epoch": epoch, "val_em": val_em, "val_ex": val_ex, "val_valid": val_valid, "val_f1": val_f1}
            )

        print(f"epoch={epoch:02d} val_em={val_em:.4f} val_ex={val_ex:.4f} val_valid={val_valid:.4f} val_f1={val_f1:.4f} best_em={best_em:.4f}")
        
    if best_path.exists():
        ckpt = torch.load(best_path.as_posix(), map_location=device)
        try:
            state = load_state_dict_flexible(ckpt)
            model.load_state_dict(state)
        except Exception as e:
            print(f"Warning: could not load best checkpoint ({e}); continuing with last epoch weights")

    con_test = duckdb.connect(DB_PATH.as_posix(), read_only=True)
    model.eval()
    pred_texts, gold_texts = [], []
    with torch.no_grad():
        for src_b, tgt_b in tqdm(test_loader, desc="testing", leave=False):
            src_b = src_b.to(device)
            ids = greedy_decode(model, src_b, tgt_vocab, max_len=MAX_DECODE_LENGTH)
            pred_texts += ids_to_text(ids, tgt_vocab)
            gold_texts += ids_to_text(tgt_b[:, 1:].to(device), tgt_vocab)

    test_em, test_ex, test_valid, test_f1, comp = metrics_on_predictions(pred_texts, gold_texts, con_test)
    print(f"test_em={test_em:.4f} test_ex={test_ex:.4f} test_valid={test_valid:.4f} test_f1={test_f1:.4f}")
    print(f"components={comp}")


if __name__ == "__main__":
    train_lstm()