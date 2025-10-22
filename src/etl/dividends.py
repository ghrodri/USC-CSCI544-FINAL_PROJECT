import os
import time
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
BASE_URL = "https://api.polygon.io"
SLEEP = 12.5  # seconds between calls
CATALOG = "./src/data/catalog/sp500_companies.csv"
DIVIDENDS_DIR_DATA = "./src/data/dividends"

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def parquet_exists(folder, ticker):
    f = os.path.join(folder, f"ticker={ticker}", "part-0.parquet")
    if (not os.path.exists(f)) or (os.path.getsize(f) == 0):
        return False
    try:
        df = pd.read_parquet(f)
        return (df is not None) and (not df.empty)
    except Exception:
        return False

def save_parquet(df, base, ticker):
    if df is None or df.empty:
        return
    d = os.path.join(base, f"ticker={ticker}")
    ensure_dir(d)
    fp = os.path.join(d, "part-0.parquet")
    try:
        os.remove(fp)
    except FileNotFoundError:
        pass
    df.to_parquet(fp, index=False)


# catalog
def get_tickers():
    df = pd.read_csv(CATALOG)
    if "Symbol" in df.columns:
        col = "Symbol"
    elif "ticker" in df.columns:
        col = "ticker"
    else:
        raise RuntimeError("Catalog missing 'Symbol' or 'ticker' column")
    vals = df[col].astype(str).str.strip().str.upper().tolist()
    return vals


# df cleanup
def drop_non_scalar(df):
    cols = list(df.columns)
    rem = []
    for i in range(len(cols)):
        c = cols[i]
        if df[c].dtype == "object":
            head = df[c].dropna().head(5)
            has_complex = False
            for j in range(len(head)):
                x = head.iloc[j]
                if isinstance(x, dict) or isinstance(x, list):
                    has_complex = True
                    break
            if has_complex:
                rem.append(c)
    if len(rem) > 0:
        return df.drop(columns=rem)
    return df

def ensure_columns(df, schema):
    keys = list(schema.keys())
    for i in range(len(keys)):
        col = keys[i]
        kind = schema[col]
        if col not in df.columns:
            if kind == "date":
                df[col] = pd.NaT
            elif kind == "float":
                df[col] = np.nan
            elif kind == "int":
                df[col] = pd.Series(dtype="Int64")
            else:
                df[col] = pd.NA

    for i in range(len(keys)):
        col = keys[i]
        kind = schema[col]
        if kind == "date":
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
        elif kind == "float":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif kind == "int":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        else:
            df[col] = df[col].astype("string")

    return df[keys]


# retry function
def request_retry(url, params, timeout=30, retries=3):
    for i in range(retries):
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code != 429:
            r.raise_for_status()
            return r
        retry = r.headers.get("Retry-After")
        if retry is None:
            retry_sec = SLEEP
        else:
            try:
                retry_sec = float(retry)
            except Exception:
                retry_sec = SLEEP
        time.sleep(retry_sec)
    r.raise_for_status()
    return r


# polygon call dividends
def get_dividends(ticker):
    url = f"{BASE_URL}/v3/reference/dividends"
    params = {
        "ticker": ticker,
        "limit": 1000,
        "order": "asc",
        "sort": "ex_dividend_date",
        "apiKey": POLYGON_API_KEY
    }
    r = request_retry(url, params)
    js = r.json()
    rows = js.get("results", [])
    if (rows is None) or (len(rows) == 0):
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["ticker"] = ticker

    schema = {
        "ticker": "str",
        "declaration_date": "date",
        "ex_dividend_date": "date",
        "pay_date": "date",
        "record_date": "date",
        "cash_amount": "float",
    }

    df = drop_non_scalar(df)
    df = ensure_columns(df, schema)

    ev = df["pay_date"].copy()
    ev = ev.fillna(df["ex_dividend_date"])
    ev = ev.fillna(df["record_date"])
    ev = ev.fillna(df["declaration_date"])
    df["event_date"] = ev

    df = df[~df["event_date"].isna()]

    cols = ["ticker", "event_date", "declaration_date", "ex_dividend_date", "pay_date", "record_date", "cash_amount"]
    return df[cols]

def build():
    # missing API KEY use: export POLYGON_API_KEY="your_polygon_key"
    if POLYGON_API_KEY == "":
        raise RuntimeError("missing POLYGON_API_KEY")

    ensure_dir(DIVIDENDS_DIR_DATA)

    tickers = get_tickers()
    n = len(tickers)
    created = 0
    skipped = 0
    failed = 0

    for t in tqdm(tickers, desc="processing dividends"):
        if parquet_exists(DIVIDENDS_DIR_DATA, t):
            skipped += 1
            continue

        ok = False
        for i in range(2):
            try:
                df = get_dividends(t)
                if (df is not None) and (not df.empty):
                    save_parquet(df, DIVIDENDS_DIR_DATA, t)
                if parquet_exists(DIVIDENDS_DIR_DATA, t):
                    ok = True
                    created += 1
                    break
            except Exception:
                time.sleep(SLEEP)

        if not ok:
            failed += 1

        time.sleep(SLEEP)

    print(f"\nSummary Dividends: {created} created, {skipped} skipped, {failed} failed of {n} total tickers.")

if __name__ == "__main__":
    build()