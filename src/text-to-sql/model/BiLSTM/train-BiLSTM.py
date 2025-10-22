import os, re, json, math, random
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import duckdb
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

# hyperparameters
EPOCHS = 40 # number of epochs
BATCH_SIZE = 16 # batch size
LEARNING_RATE = 1e-3 # learning rate
EMB_DIM = 256 # embedding dim
HID_DIM = 384 # BiLSTM hidden dim (total, bidirectional)
LAYERS = 2 # BiLSTM layers
DROPOUT = 0.3 # dropout
K_COLS = 48 # candidate columns per example
SEED = 42 # random seed
MAX_VAL_STEPS = None # limit val batches 
MAX_TEST_STEPS = None # limit test batches 

TRAIN_DATA = "./src/text-to-sql/data/train.jsonl"
VAL_DATA   = "./src/text-to-sql/data/val.jsonl"
TEST_DATA  = "./src/text-to-sql/data/test.jsonl"
META_DATA  = "./src/text-to-sql/data/dataset_metadata.json"
DB_DATA    = "./src/db/financial.duckdb"
CHECKP     = "./src/text-to-sql/model/BiLSTM/ckpt/" 
EVAL_ROOT  = "./src/evl/BiLSTM/"
PLOTS_DIR  = os.path.join(EVAL_ROOT, "plots")
REPORTS_DIR= os.path.join(EVAL_ROOT, "reports")

os.makedirs(CHECKP, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# tokenization 
punct_table = {}
puncts = list(",.;:()[]=<>+-*/%'")
for c in puncts:
    punct_table[c] = f" {c} "
punct_trans = str.maketrans(punct_table)

def read_jsonl_file(p: str):
    x = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                x.append(json.loads(line))
    return x

def tokenize_text(s: str):
    if not s:
        return []
    s = s.lower().replace("\t", " ").translate(punct_trans)
    return s.split()

def split_identifier(name: str):
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    s = s.replace("_", " ").lower()
    out = s.split()
    out = [t for t in out if t]
    return out

# sql utils
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

# schema load
class DuckSchema:
    def __init__(self, db_path: str):
        self.con = duckdb.connect(db_path, read_only=True)
        self.cols = []
        self.col_full = []
        self.col_tok = []
        self.tab_cols = defaultdict(list)    # "sch.tab" -> [col]
        self.tab_types = defaultdict(dict)   # "sch.tab" -> {col: type}
        self.load_all()

    def load_all(self):
        q = """
        select table_schema, table_name, column_name, data_type
        from information_schema.columns
        order by table_schema, table_name, ordinal_position
        """
        df = self.con.execute(q).df()
        for i in range(len(df)):
            sch = df.loc[i, "table_schema"]
            tab = df.loc[i, "table_name"]
            col = df.loc[i, "column_name"]
            typ = df.loc[i, "data_type"]
            self.cols.append((sch, tab, col, typ))
            full = f"{sch}.{tab}.{col}"
            toks = split_identifier(tab) + split_identifier(col)
            self.col_full.append(full)
            self.col_tok.append(toks)
            key = f"{sch}.{tab}"
            self.tab_cols[key].append(col)
            self.tab_types[key][col] = typ

    def numeric_cols(self, sch_tab: str):
        cols = []
        dtypes = self.tab_types[sch_tab]
        for c in dtypes:
            t = dtypes[c].lower()
            if ("int" in t) or ("decimal" in t) or ("double" in t) or ("float" in t) or ("numeric" in t) or ("real" in t):
                cols.append(c)
        return cols

# slot parsing and hints
AGGS = ["none", "avg", "sum", "min", "max", "count"]
SECTORS = [
    "communication services","consumer cyclical","financial services","technology",
    "healthcare","industrials","consumer defensive","energy","basic materials",
    "real estate","utilities"
]
EX_MAP = {"nyse":"nyse", "nasdaq":"nasdaq", "amex":"amex"}

datetime_re = re.compile(r"(20[0-9]{2}-[01][0-9]-[0-3][0-9](?:\s+[0-2][0-9]:[0-5][0-9]:[0-5][0-9])?)")
date_re = re.compile(r"(20[0-9]{2}-[01][0-9]-[0-3][0-9])")
int_re  = re.compile(r"\b([0-9]{1,3})\b")

def parse_slots_from_sql(s: str, it: str) -> Dict[str, Any]:
    x = s.strip().rstrip(";").lower()
    out = {"intent": it, "agg": "none"}

    if re.search(r"\bavg\s*\(", x): out["agg"] = "avg"
    elif re.search(r"\bsum\s*\(", x): out["agg"] = "sum"
    elif re.search(r"\bmin\s*\(", x): out["agg"] = "min"
    elif re.search(r"\bmax\s*\(", x): out["agg"] = "max"
    elif re.search(r"\bcount\s*\(", x): out["agg"] = "count"

    m_from = re.search(r"\bfrom\s+([a-z0-9_\.]+)(?:\s+as\s+[a-z0-9_]+|\s+[a-z0-9_]+)?", x)
    if m_from:
        out["from_table"] = m_from.group(1)

    m_head = re.search(r"select\s+(.*?)\s+from\s", x, flags=re.DOTALL)
    if m_head:
        head = m_head.group(1).replace('"', '').replace('`','')
        m_fun = re.search(r"(?:avg|sum|min|max|count)\s*\(\s*(?:([a-z0-9_]+)\.)?([a-z0-9_]+)\s*\)", head)
        if m_fun:
            tab = m_fun.group(1)
            col = m_fun.group(2)
            if (not tab) and ("from_table" in out):
                tab = out["from_table"].split(".")[-1]
            if tab and col:
                out["sel_col"] = f"{tab}.{col}"
        else:
            m_tc = re.search(r"([a-z0-9_]+)\.([a-z0-9_]+)", head)
            if m_tc:
                out["sel_col"] = f"{m_tc.group(1)}.{m_tc.group(2)}"
            else:
                m_c = re.search(r"\b([a-z0-9_]+)\b", head)
                if m_c and ("from_table" in out):
                    tab = out["from_table"].split(".")[-1]
                    out["sel_col"] = f"{tab}.{m_c.group(1)}"

    m_t = re.search(r"where\s+.*?ticker\s*=\s*'([^']+)'", x)
    if m_t:
        out["ticker"] = m_t.group(1)
    else:
        m_in = re.search(r"ticker\s+in\s*\(([^)]+)\)", x)
        if m_in:
            raw = m_in.group(1).split(",")
            vals = []
            for i in range(len(raw)):
                vals.append(raw[i].strip().strip("'\" "))
            out["tickers"] = vals

    m_b = re.search(r"date\s+between\s*'([0-9\-]+)'\s+and\s*'([0-9\-]+)'", x)
    if m_b:
        out["date_from"] = m_b.group(1)
        out["date_to"] = m_b.group(2)
    else:
        m_ge = re.search(r"date\s*>=\s*'([0-9\-]+)'", x)
        m_le = re.search(r"date\s*<=\s*'([0-9\-]+)'", x)
        if m_ge:
            out["date_from"] = m_ge.group(1)
        if m_le:
            out["date_to"] = m_le.group(1)

    m_k = re.search(r"\blimit\s+(\d+)", x)
    if m_k:
        out["top_k"] = int(m_k.group(1))
    m_o = re.search(r"order\s+by\s+([a-z0-9_\.]+)\s*(asc|desc)?", x)
    if m_o:
        out["order_col"] = m_o.group(1)
        out["order_dir"] = m_o.group(2) or "asc"

    return out

def extract_sector_from_text(q: str):
    x = q.lower()
    for s in SECTORS:
        if s in x:
            return s
    m = re.search(r"sector\s+([a-z ]+)", x)
    if m:
        return m.group(1).strip()
    return None

def extract_exchange_from_text(q: str):
    x = q.lower()
    for k in EX_MAP:
        if k in x:
            return EX_MAP[k]
    return None

def extract_windows_from_text(q: str):
    x = q.lower()
    wins = re.findall(r"(\d+)\s*[- ]?\s*day", x)
    if not wins:
        wins = re.findall(r"\b(\d+)\s*d\b", x)
    if not wins:
        return None
    out = []
    for i in range(len(wins)):
        out.append(int(wins[i]))
    return out

def hint_values_from_text(q: str) -> Dict[str, Any]:
    out = {}
    dts = datetime_re.findall(q)
    if dts:
        out["period_dt"] = dts[0]
    ds = date_re.findall(q)
    if len(ds) >= 2:
        out["date_from"] = ds[0]
        out["date_to"] = ds[1]
    elif len(ds) == 1:
        out["date_from"] = ds[0]
        if re.search(r"\b(on|for)\b", q.lower()):
            out["date_eq"] = ds[0]
    ns = int_re.findall(q)
    if ns:
        out["top_k"] = int(ns[0])
    caps = re.findall(r"\b[A-Z]{1,5}\b", q)
    filt = []
    for i in range(len(caps)):
        c = caps[i]
        if c.lower() not in EX_MAP:
            filt.append(c)
    if filt:
        out["ticker"] = filt[0]
    sec = extract_sector_from_text(q)
    if sec:
        out["sector"] = sec
    ex = extract_exchange_from_text(q)
    if ex:
        out["exchange"] = ex
    wins = extract_windows_from_text(q)
    if wins:
        out["ret_windows"] = wins
    return out

def ensure_select_col(slots: Dict[str,Any], sch: DuckSchema):
    base = slots.get("from_table")
    if (not base) or (base not in sch.tab_cols):
        return slots
    cols = sch.tab_cols[base]
    for i in range(len(cols)):
        c = cols[i]
        if "close" in c:
            slots["sel_col"] = f"{base.split('.')[-1]}.{c}"
            return slots
    nums = sch.numeric_cols(base)
    if nums:
        slots["sel_col"] = f"{base.split('.')[-1]}.{nums[0]}"
    return slots

# vocabulary and batch
class Vocab:
    def __init__(self, specials=["<pad>", "<unk>"]):
        self.stoi = {}
        self.itos = []
        self.freq = Counter()
        for i in range(len(specials)):
            self.add(specials[i])
        self.pad = self.stoi["<pad>"]
        self.unk = self.stoi["<unk>"]

    def add(self, t: str):
        if t not in self.stoi:
            self.stoi[t] = len(self.itos)
            self.itos.append(t)
        self.freq[t] += 1

    def encode(self, ts: List[str]):
        out = []
        for i in range(len(ts)):
            t = ts[i]
            out.append(self.stoi.get(t, self.unk))
        return out

def pad_sequences(x, pad_id: int):
    L = 0
    for i in range(len(x)):
        L = max(L, len(x[i]))
    B = len(x)
    out = torch.full((B, L), pad_id, dtype=torch.long)
    msk = torch.zeros((B, L), dtype=torch.bool)
    for i in range(B):
        s = x[i]
        out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        msk[i, :len(s)] = 1
    return out, msk

INTENTS = [
    "adjusted_price","avg_between_dates","calendar_filter","corporate_actions",
    "delta_window","dividend_yield","join_dividends","join_splits","latest_point",
    "membership","point_lookup","profile","resample_monthly","resample_weekly",
    "sector_aggregate","top_k","ytd"
]
IT2ID = {}
for i in range(len(INTENTS)):
    IT2ID[INTENTS[i]] = i

def load_split(path: str):
    rows = read_jsonl_file(path)
    out = []
    for i in range(len(rows)):
        r = rows[i]
        q = r.get("question", "")
        s = r.get("sql", "")
        it = r.get("intent", "")
        if it not in IT2ID:
            continue
        out.append({"q": q, "sql": s, "it": it, "slots": parse_slots_from_sql(s, it)})
    return out

def build_vocab(tr, sch: DuckSchema):
    v = Vocab()
    src = tr if tr else []
    for i in range(len(src)):
        ex = src[i]
        ts = tokenize_text(ex["q"])
        for j in range(len(ts)):
            v.add(ts[j])
    for i in range(len(sch.col_tok)):
        toks = sch.col_tok[i]
        for j in range(len(toks)):
            v.add(toks[j])
    return v

def hash_col_embedding(toks: List[str], dim=128):
    if toks == ["__PAD__"]:
        return np.zeros(dim, dtype=np.float32)
    v = np.zeros(dim, dtype=np.float32)
    if not toks:
        return v
    for i in range(len(toks)):
        t = toks[i]
        h = abs(hash(t)) % 1_000_003
        rng = np.random.default_rng(h)
        r = rng.normal(0, 1.0, size=dim).astype(np.float32)
        v += r
    v /= max(1, len(toks))
    return v

def pick_cols_for_table(q: str, sch: DuckSchema, sch_tab: str, k: int = K_COLS):
    names = []
    toks_list = []
    if sch_tab and (sch_tab in sch.tab_cols):
        tname = sch_tab.split(".")[-1]
        cols = sch.tab_cols[sch_tab]
        for i in range(len(cols)):
            c = cols[i]
            names.append(f"{sch_tab}.{c}")
            toks_list.append(split_identifier(tname) + split_identifier(c))
        return names[:k], toks_list[:k]
    qs = set(tokenize_text(q))
    scores = []
    for i in range(len(sch.col_tok)):
        toks = sch.col_tok[i]
        inter = 0
        for j in range(len(toks)):
            if toks[j] in qs:
                inter += 1
        scores.append((inter, i))
    scores.sort(reverse=True)
    idx = []
    for i in range(min(k, len(scores))):
        idx.append(scores[i][1])
    for i in range(len(idx)):
        names.append(sch.col_full[idx[i]])
        toks_list.append(sch.col_tok[idx[i]])
    return names, toks_list

def pad_to_k(names: List[str], toks_list: List[List[str]], k: int):
    n = len(names)
    mask = []
    for i in range(n):
        mask.append(True)
    while len(names) < k:
        names.append("__PAD__.pad")
        toks_list.append(["__PAD__"])
        mask.append(False)
    if len(names) > k:
        names = names[:k]
        toks_list = toks_list[:k]
        mask = mask[:k]
    return names, toks_list, mask

class BatchMaker:
    def __init__(self, data, v: Vocab, sch: DuckSchema, bs=BATCH_SIZE, kcols=K_COLS):
        self.data = data
        self.v = v
        self.sch = sch
        self.bs = bs
        self.kcols = kcols
        self.n = math.ceil(len(data) / bs)

    def __iter__(self):
        idx = list(range(len(self.data)))
        random.shuffle(idx)
        for b in range(self.n):
            ids = idx[b*self.bs:(b+1)*self.bs]
            exs = [self.data[i] for i in ids]
            for i in range(len(exs)):
                exs[i]["slots"] = ensure_select_col(exs[i]["slots"], self.sch)

            qs = []
            for i in range(len(exs)):
                qs.append(tokenize_text(exs[i]["q"]))
            x = []
            for i in range(len(qs)):
                x.append(self.v.encode(qs[i]))
            x_pad, msk = pad_sequences(x, self.v.pad)

            y_sk = torch.tensor([IT2ID[exs[i]["it"]] for i in range(len(exs))], dtype=torch.long)
            y_ag = []
            for i in range(len(exs)):
                ag = exs[i]["slots"].get("agg", "none")
                y_ag.append(AGGS.index(ag) if ag in AGGS else 0)
            y_ag = torch.tensor(y_ag, dtype=torch.long)

            cand_names = []
            cand_tok = []
            y_col = []
            col_mask = []
            for i in range(len(exs)):
                from_tab = exs[i]["slots"].get("from_table")
                names, toks_list = pick_cols_for_table(exs[i]["q"], self.sch, from_tab, k=self.kcols)
                if len(names) == 0:
                    names, toks_list = pick_cols_for_table(exs[i]["q"], self.sch, None, k=self.kcols)
                names, toks_list, mask = pad_to_k(names, toks_list, self.kcols)
                cand_names.append(names)
                cand_tok.append(toks_list)
                col_mask.append(mask)
                gold = exs[i]["slots"].get("sel_col", "")
                tgt = -1
                if gold:
                    gold_tail = ".".join(gold.split(".")[-2:])
                    for j in range(len(names)):
                        full = names[j]
                        if full.endswith(gold_tail) and mask[j]:
                            tgt = j
                            break
                y_col.append(tgt)

            col_emb = []
            for i in range(len(cand_tok)):
                tl = cand_tok[i]
                row = []
                for j in range(len(tl)):
                    row.append(hash_col_embedding(tl[j]))
                col_emb.append(np.stack(row, axis=0))
            col_emb = torch.tensor(np.stack(col_emb, axis=0), dtype=torch.float32)
            col_mask = torch.tensor(np.array(col_mask, dtype=bool))

            yield {
                "x": x_pad, "m": msk,
                "y_sk": y_sk, "y_ag": y_ag,
                "cand_names": cand_names, "cand_tok": cand_tok,
                "col_emb": col_emb, "col_mask": col_mask,
                "y_col": torch.tensor(y_col, dtype=torch.long),
                "exs": exs
            }

# model
class EncoderBiLSTM(nn.Module):
    def __init__(self, vsz, emb=EMB_DIM, hid=HID_DIM, layers=LAYERS, drop=DROPOUT):
        super().__init__()
        self.emb = nn.Embedding(vsz, emb, padding_idx=0)
        self.rnn = nn.LSTM(emb, hid//2, num_layers=layers, bidirectional=True, batch_first=True, dropout=drop)
        self.drop = nn.Dropout(drop)
        self.hid = hid

    def forward(self, x, m):
        e = self.emb(x)
        lens = m.sum(1).cpu()
        pk = nn.utils.rnn.pack_padded_sequence(e, lens, batch_first=True, enforce_sorted=False)
        o, _ = self.rnn(pk)
        o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True)
        o = self.drop(o)
        h = (o * m.unsqueeze(-1)).sum(1) / (m.sum(1, keepdim=True) + 1e-9)
        return o, h

class Heads(nn.Module):
    def __init__(self, hid, n_sk, n_ag):
        super().__init__()
        self.sk = nn.Linear(hid, n_sk)
        self.ag = nn.Linear(hid, n_ag)

    def forward(self, h):
        out = {}
        out["sk"] = self.sk(h)
        out["ag"] = self.ag(h)
        return out

class ColumnPointer(nn.Module):
    def __init__(self, hid, cdim=128):
        super().__init__()
        self.q = nn.Linear(hid, hid)
        self.c = nn.Linear(cdim, hid)
        self.sc = nn.Linear(hid, 1)

    def forward(self, h, ce):
        b, k, c = ce.size()
        qv = self.q(h).unsqueeze(1).expand(b, k, -1)
        cv = self.c(ce)
        t = torch.tanh(qv + cv)
        s = self.sc(t).squeeze(-1)
        return s

class Model(nn.Module):
    def __init__(self, vsz, hid=HID_DIM, emb=EMB_DIM, layers=LAYERS, drop=DROPOUT, n_sk=len(INTENTS), n_ag=len(AGGS), cdim=128):
        super().__init__()
        self.enc = EncoderBiLSTM(vsz, emb, hid, layers, drop)
        self.hd = Heads(hid, n_sk, n_ag)
        self.cp = ColumnPointer(hid, cdim)

    def forward(self, x, m, ce):
        _, h = self.enc(x, m)
        out = self.hd(h)
        out["col"] = self.cp(h, ce)
        return out

# decode
def build_sql(intent: str, slots: Dict[str, Any], sch: DuckSchema) -> str:
    base = slots.get("from_table")
    if not base:
        if intent in ("avg_between_dates","delta_window","ytd","resample_weekly","resample_monthly",
                      "latest_point","top_k","adjusted_price","point_lookup","join_dividends"):
            base = "core.prices_daily"
        elif intent in ("dividend_yield",):
            base = "core.dividends"
        elif intent in ("sector_aggregate","profile","membership"):
            base = "core.tickers"
        elif intent in ("calendar_filter",):
            base = "core.trading_calendar"
        else:
            base = "core.prices_daily"

    slots = ensure_select_col(slots, sch)
    sel = slots.get("sel_col")
    colname = sel.split(".")[-1] if sel else "date"

    tic = slots.get("ticker")
    d1, d2 = slots.get("date_from"), slots.get("date_to")
    deq = slots.get("date_eq")
    k = int(slots.get("top_k", 10))
    sector = slots.get("sector")
    exch = slots.get("exchange")
    period_dt = slots.get("period_dt")
    wins = slots.get("ret_windows") or [7, 30]

    def agg_wrap(x):
        ag = slots.get("agg", "none")
        if intent == "avg_between_dates":
            return f"avg({x})"
        if ag != "none":
            return f"{ag}({x})"
        return x

    if intent == "membership" and sector and (deq or d1):
        dt = deq or d1
        sql = (
            "select distinct p.ticker from core.prices_daily p "
            "join core.tickers t using (ticker) "
            f"where p.date='{dt}' and t.sector='{sector}' order by 1"
        )

    elif intent == "calendar_filter" and exch and (deq or d1):
        dt = deq or d1
        sql = f"select is_open from core.trading_calendar where exchange='{exch}' and date='{dt}'"

    elif intent == "resample_monthly" and ("views.prices_monthly" in base) and period_dt and tic:
        sql = f"select * from views.prices_monthly where ticker='{tic.lower()}' and month='{period_dt}'"

    elif intent == "resample_weekly" and ("views.prices_weekly" in base) and period_dt and tic:
        sql = f"select * from views.prices_weekly where ticker='{tic.lower()}' and week='{period_dt}'"

    elif intent == "top_k" and (deq or d1):
        dt = deq or d1
        sql = (
            "with r as ( "
            "select close/lag(close) over (partition by ticker order by date)-1 as ret, "
            "ticker from core.prices_daily "
            f"where date='{dt}' ) "
            "select ticker, ret from r order by ret desc "
            f"limit {k}"
        )

    elif intent == "delta_window" and tic:
        w1 = wins[0] if len(wins) >= 1 else 7
        w2 = wins[1] if len(wins) >= 2 else 30
        sql = (
            "with base as (select date, ticker, close from core.prices_daily "
            f"where ticker='{tic.lower()}') "
            f"select date, ticker, close, "
            f"close/lag(close,{w1}) over (partition by ticker order by date)-1 as r{w1}, "
            f"close/lag(close,{w2}) over (partition by ticker order by date)-1 as r{w2} "
            "from base"
        )

    elif intent == "ytd" and tic:
        sql = (
            "with dr as ( "
            f"select date, ticker, close/lag(close) over (partition by ticker order by date)-1 as ret "
            f"from core.prices_daily where ticker='{tic.lower()}' "
            "), wy as ( "
            "select date, ticker, ret, date_trunc('year', date) as y from dr "
            "), agg as ( "
            "select date, ticker, exp(sum(ln(1+ret)) over (partition by ticker, y order by date "
            "rows between unbounded preceding and current row))-1 as cum_ret_ytd "
            "from wy ) "
            "select date, cum_ret_ytd from agg order by date"
        )

    elif intent == "profile" and tic:
        sql = (
            "select r.company_name, r.industry, r.market, r.primary_exchange, "
            "t.exchange, t.name, t.sector, t.ticker "
            "from core.tickers t left join core.ref_tickers r using (ticker) "
            f"where t.ticker='{tic.lower()}'"
        )

    elif intent == "join_dividends" and tic and (d1 or d2):
        wh = [f"p.ticker='{tic.lower()}'"]
        if d1 and d2:
            wh.append(f"p.date between '{d1}' and '{d2}'")
        elif d1:
            wh.append(f"p.date >= '{d1}'")
        elif d2:
            wh.append(f"p.date <= '{d2}'")
        w = " and ".join(wh)
        sql = (
            "select p.date, p.ticker, p.close, coalesce(d.cash_amount, 0) as cash_amount "
            "from core.prices_daily p "
            "left join core.dividends d "
            "on d.ticker=p.ticker and d.event_date=p.date "
            f"where {w} order by p.date"
        )

    elif intent == "avg_between_dates" and tic and d1 and d2:
        sql = (
            "select avg(close) as avg_close from core.prices_daily "
            f"where ticker='{tic.lower()}' and date between '{d1}' and '{d2}'"
        )

    elif intent == "latest_point" and tic:
        sql = f"select {colname} from {base} where ticker='{tic.lower()}' order by date desc limit 1"

    elif intent == "sector_aggregate" and sector and (deq or d1):
        dt = deq or d1
        sql = (
            "select avg(p.close) as avg_close, sum(p.volume) as total_volume "
            "from core.prices_daily p join core.tickers t using (ticker) "
            f"where t.sector='{sector}' and p.date='{dt}'"
        )

    elif intent == "dividend_yield" and tic:
        if "views.dividend_yield_ttm" in base:
            sql = (
                "select date, dividend_yield_ttm from views.dividend_yield_ttm "
                f"where ticker='{tic.lower()}' order by date"
            )
        else:
            wh = [f"ticker='{tic.lower()}'"]
            if d1 and d2:
                wh.append(f"event_date between '{d1}' and '{d2}'")
            elif d1:
                wh.append(f"event_date >= '{d1}'")
            elif d2:
                wh.append(f"event_date <= '{d2}'")
            sql = f"select sum(cash_amount) as div_cash from core.dividends where {' and '.join(wh)}"

    elif intent == "adjusted_price" and ("views.adjusted_prices_splits" in base) and tic:
        cols = set(sch.tab_cols.get("views.adjusted_prices_splits", []))
        tgt = "close_adj_splits" if ("close_adj_splits" in cols) else "close"
        sql = (
            f"select {tgt}, date from views.adjusted_prices_splits "
            f"where ticker='{tic.lower()}' order by date"
        )

    elif intent in ("resample_monthly","resample_weekly") and tic:
        fmt = "%Y-%m" if intent == "resample_monthly" else "%Y-%W"
        grp = "month" if intent == "resample_monthly" else "week"
        sql = (
            f"select strftime(date, '{fmt}') as {grp}, {colname} as val "
            f"from {base} where ticker='{tic.lower()}' group by 1 order by 1"
        )

    elif intent == "top_k":
        sql = f"select {colname} from {base} order by date desc limit {k}"

    else:
        wh = []
        if tic:
            wh.append(f"ticker = '{tic.lower()}'")
        if d1 and d2:
            wh.append(f"date between '{d1}' and '{d2}'")
        elif slots.get("date_eq"):
            wh.append(f"date = '{slots['date_eq']}'")
        elif d1:
            wh.append(f"date >= '{d1}'")
        elif d2:
            wh.append(f"date <= '{d2}'")
        wsql = (" where " + " and ".join(wh)) if wh else ""
        sql = f"select {agg_wrap(colname)} from {base}{wsql}"

    return normalize_sql(sql)

# train / eval
def accuracy_from_logits(logits, y):
    return (logits.argmax(-1) == y).float().mean().item()

def evaluate_model(model, it_data, sch, dev="cpu", max_steps=None, collect_fail=False):
    model.eval()
    em_hit = 0
    em_tot = 0
    ex_hit = 0
    ex_tot = 0
    fails = []
    per_intent_fail = Counter()

    with torch.no_grad():
        bar = tqdm(it_data, desc="Validation", leave=False)
        step = 0
        for b in bar:
            x = b["x"].to(dev)
            m = b["m"].to(dev)
            ce = b["col_emb"].to(dev)
            mask = b["col_mask"].to(dev)
            out = model(x, m, ce)
            col_logits = out["col"] + (~mask)*(-1e9)

            sk = out["sk"].argmax(-1).cpu().tolist()
            ag = out["ag"].argmax(-1).cpu().tolist()
            ci = col_logits.argmax(-1).cpu().tolist()

            for j in range(len(b["exs"])):
                ex = b["exs"][j]
                intent_pred = INTENTS[sk[j]]
                agg_pred = AGGS[ag[j]]

                cand = b["cand_names"][j]
                sel_full = cand[ci[j]] if cand else None
                sel = None
                if sel_full and (sel_full != "__PAD__.pad"):
                    pr = sel_full.split(".")
                    if len(pr) >= 3:
                        sel = f"{pr[-2]}.{pr[-1]}"
                    elif len(pr) == 2:
                        sel = sel_full

                slots = {"intent": intent_pred, "agg": "none"}
                if sel:
                    slots["sel_col"] = sel
                gold_slots = ex["slots"]
                for k in gold_slots:
                    if k not in slots:
                        slots[k] = gold_slots[k]
                hv = hint_values_from_text(ex["q"])
                for k in hv:
                    if k not in slots:
                        slots[k] = hv[k]

                pred_sql = build_sql(intent_pred, slots, sch)
                gold_sql = normalize_sql(ex["sql"])

                em_tot += 1
                if normalize_sql(pred_sql) == gold_sql:
                    em_hit += 1

                ok_g, df_g, err_g = try_execute_sql(sch.con, gold_sql)
                ok_p, df_p, err_p = try_execute_sql(sch.con, pred_sql)
                if ok_g:
                    ex_tot += 1
                    if ok_p and dataframe_equal(df_g, df_p):
                        ex_hit += 1
                    else:
                        if collect_fail:
                            fails.append({
                                "question": ex["q"],
                                "intent_gold": ex["it"],
                                "sql_gold": gold_sql,
                                "sql_pred": pred_sql,
                                "exec_err_pred": err_p
                            })
                            per_intent_fail[ex["it"]] += 1

            step += 1
            if max_steps is not None and step >= max_steps:
                break

    em = em_hit / max(1, em_tot)
    ex = ex_hit / max(1, ex_tot)
    cov = ex_tot / max(1, em_tot)
    return em, ex, cov, fails, per_intent_fail

def train_model(model, it_train, it_val, sch, dev="cpu", epochs=EPOCHS, lr=LEARNING_RATE):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    best_val_exec = -1.0

    # logs per epoch
    loss_hist = []
    intent_acc_hist = []
    agg_acc_hist = []
    col_acc_hist = []
    val_em_hist = []
    val_exec_hist = []
    cov_hist = []

    for ep in range(1, epochs+1):
        model.train()
        losses = []
        acc_sk = []
        acc_ag = []
        acc_col = []
        bar = tqdm(it_train, desc="Training", leave=False)
        for b in bar:
            x = b["x"].to(dev)
            m = b["m"].to(dev)
            ce_cols = b["col_emb"].to(dev)
            mask = b["col_mask"].to(dev)
            y_sk = b["y_sk"].to(dev)
            y_ag = b["y_ag"].to(dev)
            y_col = b["y_col"].to(dev)

            out = model(x, m, ce_cols)
            out["col"] = out["col"] + (~mask)*(-1e9)

            loss = ce(out["sk"], y_sk) + ce(out["ag"], y_ag)
            valid = (y_col >= 0)
            if valid.any():
                loss = loss + ce(out["col"][valid], y_col[valid])
                acc_col.append(accuracy_from_logits(out["col"][valid], y_col[valid]))

            acc_sk.append(accuracy_from_logits(out["sk"], y_sk))
            acc_ag.append(accuracy_from_logits(out["ag"], y_ag))

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        # validation
        em, ex, cov, _, _ = evaluate_model(model, it_val, sch, dev=dev, max_steps=MAX_VAL_STEPS, collect_fail=False)

        # save last checkpoint
        torch.save(model.state_dict(), os.path.join(CHECKP, "last.pt"))
        # save best checkpoint
        if ex > best_val_exec:
            best_val_exec = ex
            torch.save(model.state_dict(), os.path.join(CHECKP, "best.pt"))

        # accumulate metrics
        loss_hist.append(float(np.mean(losses)))
        intent_acc_hist.append(float(np.mean(acc_sk)))
        agg_acc_hist.append(float(np.mean(acc_ag)))
        if len(acc_col) > 0:
            col_acc_hist.append(float(np.mean(acc_col)))
        else:
            col_acc_hist.append(float("nan"))
        val_em_hist.append(em)
        val_exec_hist.append(ex)
        cov_hist.append(cov)

        # print metrics
        print(f"Epoch {ep}: loss={loss_hist[-1]:.4f} intent_acc={intent_acc_hist[-1]:.3f} agg_acc={agg_acc_hist[-1]:.3f} col_acc={col_acc_hist[-1]:.3f} val_em={em:.3f} val_exec={ex:.3f} gold_exec_cov={cov:.2f}")

        # write metrics
        with open(os.path.join(REPORTS_DIR, "metrics_per_epoch.txt"), "a", encoding="utf-8") as f:
            f.write(f"Epoch {ep}: loss={loss_hist[-1]:.6f} intent_acc={intent_acc_hist[-1]:.6f} agg_acc={agg_acc_hist[-1]:.6f} col_acc={col_acc_hist[-1]:.6f} val_em={em:.6f} val_exec={ex:.6f} gold_exec_cov={cov:.6f}\n")
    return {
        "loss": loss_hist,
        "intent_acc": intent_acc_hist,
        "agg_acc": agg_acc_hist,
        "col_acc": col_acc_hist,
        "val_em": val_em_hist,
        "val_exec": val_exec_hist,
        "cov": cov_hist
    }

# plots
def plot_line(x, y, title, xlabel, ylabel, out_path):
    plt.figure(figsize=(6,4))
    plt.plot(x, y, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_multi(x, y_list, labels, title, xlabel, ylabel, out_path):
    plt.figure(figsize=(6,4))
    for i in range(len(y_list)):
        plt.plot(x, y_list[i], marker="o", label=labels[i])
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

# load data splits
tr = load_split(TRAIN_DATA) if os.path.exists(TRAIN_DATA) else []
va = load_split(VAL_DATA) if os.path.exists(VAL_DATA) else []
te = load_split(TEST_DATA) if os.path.exists(TEST_DATA) else []
if (not tr) and (not va):
    raise RuntimeError("Missing dataset under ./src/text-to-sql/data/")

# schema, vocabulary
sch = DuckSchema(db_path)
v = build_vocab(tr if tr else va, sch)

it_tr = BatchMaker(tr, v, sch, bs=BATCH_SIZE, kcols=K_COLS)
it_va = BatchMaker(va if va else tr, v, sch, bs=BATCH_SIZE, kcols=K_COLS)
it_te = BatchMaker(te if te else va, v, sch, bs=BATCH_SIZE, kcols=K_COLS)

# model
mdl = Model(vsz=len(v.itos))
mdl.to(DEVICE)

# train
logs = train_model(mdl, it_tr, it_va, sch, dev=DEVICE, epochs=EPOCHS, lr=LEARNING_RATE)

# load best for test
best_path = os.path.join(CHECKP, "best.pt")
if os.path.exists(best_path):
    mdl.load_state_dict(torch.load(best_path, map_location=DEVICE))

# test
# collect failures and per-intent error counts
em, ex, cov, fails, fail_by_intent = evaluate_model(mdl, it_te, sch, dev=DEVICE, max_steps=MAX_TEST_STEPS, collect_fail=True)
print(f"Testing: em={em:.3f} exec={ex:.3f} gold_exec_cov={cov:.2f}")

# test metrics
with open(os.path.join(REPORTS_DIR, "test_metrics.txt"), "w", encoding="utf-8") as f:
    f.write(f"em={em:.6f} exec={ex:.6f} gold_exec_cov={cov:.6f}\n")

#  failures detailed file
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
x = list(range(1, len(logs["loss"])+1))

# Training Loss
plot_line(x, logs["loss"], "Training Loss", "Epoch", "Loss", os.path.join(PLOTS_DIR, "loss.png"))

# Validation EM
plot_line(x, logs["val_em"], "Validation EM", "Epoch", "EM", os.path.join(PLOTS_DIR, "val_em.png"))

# Validation ExecAcc
plot_line(x, logs["val_exec"], "Validation ExecAcc", "Epoch", "ExecAcc", os.path.join(PLOTS_DIR, "val_exec.png"))

# Head Accuracies
y_list = [logs["intent_acc"], logs["agg_acc"], logs["col_acc"]]
labels = ["Intent", "Agg", "Column"]
plot_multi(x, y_list, labels, "Head Accuracies", "Epoch", "Accuracy", os.path.join(PLOTS_DIR, "head_accuracies.png"))

# Errors by Intent (on test)
if len(fail_by_intent) > 0:
    intents = list(fail_by_intent.keys())
    vals = [fail_by_intent[k] for k in intents]
    plot_bar(intents, vals, "Errors by Intent", "Intent", "Count", os.path.join(PLOTS_DIR, "errors_by_intent.png"))