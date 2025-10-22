# ./src/text-to-sql/tools/audit_gold.py
# Audit/fix gold SQL in dataset splits against DuckDB

import re, json, argparse
from collections import Counter
import duckdb
import pandas as pd

DB_PATH = "./src/db/financial.duckdb"
SPLIT2PATH = {
    "train": "./src/text-to-sql/data/train.jsonl",
    "val":   "./src/text-to-sql/data/val.jsonl",
    "test":  "./src/text-to-sql/data/test.jsonl",
}

DATE_RE = re.compile(r"(20[0-9]{2}-[01][0-9]-[0-3][0-9])")
TICK_RE = re.compile(r"\b[A-Z]{1,5}\b")

def read_jsonl(p):
    xs = []
    with open(p, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                xs.append(json.loads(ln))
    return xs

def try_exec(con, sql):
    try:
        con.execute(sql).df()
        return True, ""
    except Exception as e:
        return False, str(e)

def parse_hints_from_q(q):
    # best-effort: ticker y fechas desde la pregunta
    tics = [t for t in TICK_RE.findall(q) if t not in ("NYSE","NASDAQ","AMEX")]
    ds = DATE_RE.findall(q)
    out = {}
    if tics: out["ticker"] = tics[0]
    if len(ds) >= 2:
        out["date_from"], out["date_to"] = ds[0], ds[1]
    elif len(ds) == 1:
        out["date_eq"] = ds[0]
    return out

def fix_join_dividends(q, old_sql):
    # Construye un SQL válido coherente con lo que tu gold intenta hacer
    h = parse_hints_from_q(q)
    tic = h.get("ticker")
    d1, d2 = h.get("date_from"), h.get("date_to")
    if not (tic and (d1 or d2)):
        return None  # no se puede arreglar de forma robusta
    wh = [f"p.ticker='{tic.lower()}'"]
    if d1 and d2: wh.append(f"p.date between '{d1}' and '{d2}'")
    elif d1: wh.append(f"p.date >= '{d1}'")
    elif d2: wh.append(f"p.date <= '{d2}'")
    w = " and ".join(wh)
    sql = (
        "select p.date, p.ticker, p.close, coalesce(d.cash_amount, 0) as cash_amount "
        "from core.prices_daily p "
        "left join core.dividends d on d.ticker=p.ticker and d.event_date=p.date "
        f"where {w} order by p.date"
    )
    return sql

def audit(split, write_fixed=False, fix_only_join_div=False):
    path = SPLIT2PATH[split]
    data = read_jsonl(path)
    con = duckdb.connect(DB_PATH, read_only=True)

    bad = []
    per_int = Counter()
    fixed_count = 0

    for i, ex in enumerate(data):
        sql = ex.get("sql","")
        ok, err = try_exec(con, sql)
        if not ok:
            per_int[ex.get("intent","<none>")] += 1
            # intento de fix para join_dividends
            if write_fixed and ex.get("intent") == "join_dividends":
                new_sql = fix_join_dividends(ex.get("question",""), sql)
                if new_sql:
                    ok2, _ = try_exec(con, new_sql)
                    if ok2:
                        ex["sql"] = new_sql
                        fixed_count += 1
                        continue
            bad.append((i, ex.get("intent"), err))

    if write_fixed and fixed_count:
        outp = path.replace(".jsonl", ".fixed.jsonl")
        with open(outp, "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"wrote: {outp}  (fixed join_dividends={fixed_count})")

    total = len(data)
    print(f"split={split}  total={total}")
    print(f"gold_exec_cov={(total-len(bad))/max(1,total):.3f}  broken={len(bad)}")
    if per_int:
        print("broken_by_intent:", dict(per_int))
    if bad and not write_fixed:
        print("\n--- examples ---")
        for i,(idx,intent,err) in enumerate(bad[:10]):
            print(f"#{i} idx={idx} intent={intent} err={err}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train","val","test"], default="val")
    ap.add_argument("--write_fixed", action="store_true", help="write *.fixed.jsonl with fixes")
    args = ap.parse_args()
    audit(args.split, write_fixed=args.write_fixed)
