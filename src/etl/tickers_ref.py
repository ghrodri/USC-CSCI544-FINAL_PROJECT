import os
import time
import pandas as pd
import requests
from tqdm import tqdm

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
BASE_URL = "https://api.polygon.io"
SLEEP = 12.5  # seconds between calls
CATALOG = "./src/data/catalog/sp500_companies.csv"
OUT_DIR = "./src/data/ref_tickers"

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

def save_parquet(df, folder, ticker):
    if df is None or df.empty:
        return
    d = os.path.join(folder, f"ticker={ticker}")
    ensure_dir(d)
    fp = os.path.join(d, "part-0.parquet")
    try:
        os.remove(fp)
    except FileNotFoundError:
        pass
    df.to_parquet(fp, index=False)

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
                if isinstance(x, (dict, list)):
                    has_complex = True
                    break
            if has_complex:
                rem.append(c)
    if len(rem) > 0:
        return df.drop(columns=rem)
    return df

def request_retry(url, params, timeout=30, retries=3):
    last = None
    for attempt in range(1, retries + 1):
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code != 429:
            r.raise_for_status()
            return r
        retry_after = r.headers.get("Retry-After")
        sleep_s = float(retry_after) if (retry_after and retry_after.isdigit()) else SLEEP * attempt
        time.sleep(sleep_s)
        last = r
    if last is None:
        last = requests.get(url, params=params, timeout=timeout)
    last.raise_for_status()
    return last

# polygon ref ticker
def get_ref_ticker(ticker):
    url = f"{BASE_URL}/v3/reference/tickers/{ticker}"
    params = {"apiKey": POLYGON_API_KEY}
    r = request_retry(url, params)
    js = r.json().get("results", {})
    if not js:
        return pd.DataFrame()
    df = pd.DataFrame([js])
    df["ticker"] = ticker
    df = drop_non_scalar(df)
    return df

def build():
    # export POLYGON_API_KEY="your_polygon_key"
    if POLYGON_API_KEY == "":
        raise RuntimeError("missing POLYGON_API_KEY")

    ensure_dir(OUT_DIR)

    tickers = get_tickers()
    n = len(tickers)
    created = 0
    skipped = 0
    failed = 0

    for t in tqdm(tickers, desc="processing ref_tickers"):
        if parquet_exists(OUT_DIR, t):
            skipped += 1
            continue

        ok = False
        for _ in range(2):
            try:
                df = get_ref_ticker(t)
                if (df is not None) and (not df.empty):
                    save_parquet(df, OUT_DIR, t)
                if parquet_exists(OUT_DIR, t):
                    ok = True
                    created += 1
                    break
            except requests.exceptions.HTTPError as e:
                code = getattr(e.response, "status_code", None)
                if code in (400, 404):
                    break
                time.sleep(SLEEP)
            except Exception:
                time.sleep(SLEEP)

        if not ok:
            failed += 1

        time.sleep(SLEEP)

    print(f"\nSummary Ref Tickers: {created} created, {skipped} skipped, {failed} failed of {n} tickers.")

if __name__ == "__main__":
    build()