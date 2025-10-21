import os
import time
from pathlib import Path
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
BASE_URL = "https://api.polygon.io"
SLEEP = 12.5

CATALOG_CSV = Path("/src/data/catalog/SP500_companies.csv")
DATA_DIR = Path("/src/data")
SPLITS_DIR = DATA_DIR / "splits"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def parquet_exists(folder: Path, ticker: str) -> bool:
    f = folder / f"ticker={ticker}/part-0.parquet"
    if not f.exists() or f.stat().st_size == 0:
        return False
    try:
        df = pd.read_parquet(f)
        return not df.empty
    except Exception:
        return False

def save(df: pd.DataFrame, base: Path, ticker: str):
    if df.empty:
        return
    d = base / f"ticker={ticker}"
    ensure_dir(d)
    (d / "part-0.parquet").unlink(missing_ok=True)
    df.to_parquet(d / "part-0.parquet", index=False)

def get_tickers() -> list[str]:
    df = pd.read_csv(CATALOG_CSV)
    if "Symbol" not in df.columns:
        raise RuntimeError("Catalog missing 'Symbol' column")
    return df["Symbol"].astype(str).str.strip().str.upper().tolist()

def drop_non_scalar(df: pd.DataFrame) -> pd.DataFrame:
    drop = []
    for c in df.columns:
        if df[c].dtype == "object":
            s = df[c].dropna().head(5)
            if s.apply(lambda x: isinstance(x, (dict, list))).any():
                drop.append(c)
    return df.drop(columns=drop) if drop else df

def ensure_columns(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    for col, kind in schema.items():
        if col not in df.columns:
            if kind == "date": df[col] = pd.NaT
            elif kind == "float": df[col] = np.nan
            elif kind == "int": df[col] = pd.Series(dtype="Int64")
            else: df[col] = pd.NA
    for col, kind in schema.items():
        if kind == "date":
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
        elif kind == "float":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif kind == "int":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        else:
            df[col] = df[col].astype("string")
    return df[list(schema.keys())]

def request_retry(url: str, params: dict, timeout: int = 30, retries: int = 3) -> requests.Response:
    for i in range(retries):
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code != 429:
            r.raise_for_status()
            return r
        retry = int(r.headers.get("Retry-After", SLEEP))
        time.sleep(retry)
    r.raise_for_status()
    return r

def get_splits(ticker: str) -> pd.DataFrame:
    url = f"{BASE_URL}/v3/reference/splits"
    params = {"ticker": ticker, "limit": 1000, "order": "asc", "sort": "execution_date", "apiKey": POLYGON_API_KEY}
    r = request_retry(url, params)
    rows = r.json().get("results", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ticker"] = ticker
    schema = {"ticker": "str", "execution_date": "date", "split_from": "float", "split_to": "float"}
    df = drop_non_scalar(df)
    df = ensure_columns(df, schema)
    df["event_date"] = df["execution_date"]
    df = df[~df["event_date"].isna()]
    return df[["ticker","event_date","execution_date","split_from","split_to"]]

def build():
    if not POLYGON_API_KEY:
        raise RuntimeError("missing POLYGON_API_KEY")

    ensure_dir(DATA_DIR)
    ensure_dir(SPLITS_DIR)

    tickers = get_tickers()
    total = len(tickers)
    created, skipped, failed = 0, 0, 0

    for t in tqdm(tickers, desc="processing splits"):
        if parquet_exists(SPLITS_DIR, t):
            skipped += 1
            continue
        success = False
        for _ in range(2):
            try:
                df = get_splits(t)
                if not df.empty:
                    save(df, SPLITS_DIR, t)
                if parquet_exists(SPLITS_DIR, t):
                    success = True
                    created += 1
                    break
            except Exception:
                time.sleep(SLEEP)
        if not success:
            failed += 1
        time.sleep(SLEEP)

    print(f"\nSummary Splits: {created} created, {skipped} skipped, {failed} failed of {total} total tickers.")

if __name__ == "__main__":
    build()