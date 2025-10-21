import os
import time
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
BASE_URL = "https://api.polygon.io"
SLEEP = 12.5

CSV_PATH = Path("/src/data/catalog/SP500_companies.csv")
OUT_DIR = Path("/src/data")
PRICES_DIR = OUT_DIR / "prices_daily"

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

def save_parquet(df: pd.DataFrame, folder: Path, ticker: str):
    if df.empty:
        return
    d = folder / f"ticker={ticker}"
    ensure_dir(d)
    (d / "part-0.parquet").unlink(missing_ok=True)
    df.to_parquet(d / "part-0.parquet", index=False)

def get_tickers() -> list[str]:
    df = pd.read_csv(CSV_PATH)
    if "Symbol" not in df.columns:
        raise RuntimeError("Catalog missing 'Symbol' column")
    return df["Symbol"].astype(str).str.strip().str.upper().tolist()

def request_retry(url: str, params: dict, timeout: int = 30, retries: int = 3) -> requests.Response:
    last = None
    for attempt in range(1, retries + 1):
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code != 429:
            r.raise_for_status()
            return r
        retry_after = r.headers.get("Retry-After")
        sleep_s = int(retry_after) if retry_after and retry_after.isdigit() else SLEEP * attempt
        time.sleep(sleep_s)
        last = r
    if last is None:
        last = requests.get(url, params=params, timeout=timeout)
    last.raise_for_status()
    return last

def get_prices_daily(ticker: str, start: str, end: str) -> pd.DataFrame:
    url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": POLYGON_API_KEY}
    r = request_retry(url, params)
    js = r.json()
    rows = js.get("results", [])
    if not rows:
        return pd.DataFrame(columns=["date","ticker","open","high","low","close","volume","vwap","transactions"])
    df = pd.DataFrame(rows).rename(columns={
        "t":"ts","o":"open","h":"high","l":"low","c":"close","v":"volume","vw":"vwap","n":"transactions"
    })
    df["date"] = pd.to_datetime(df["ts"], unit="ms").dt.date
    df["ticker"] = ticker
    return df[["date","ticker","open","high","low","close","volume","vwap","transactions"]]

def build(years: int = 2):
    if not POLYGON_API_KEY:
        raise RuntimeError("missing POLYGON_API_KEY environment variable")

    ensure_dir(PRICES_DIR)

    tickers = get_tickers()
    total = len(tickers)
    created, skipped, failed = 0, 0, 0

    end = date.today()
    start = end - timedelta(days=365 * years)

    for t in tqdm(tickers, desc="processing prices"):
        if parquet_exists(PRICES_DIR, t):
            skipped += 1
            continue

        success = False
        for _ in range(2):
            try:
                df = get_prices_daily(t, start.isoformat(), end.isoformat())
                if not df.empty:
                    save_parquet(df, PRICES_DIR, t)
                if parquet_exists(PRICES_DIR, t):
                    created += 1
                    success = True
                    break
            except requests.exceptions.HTTPError as e:
                code = getattr(e.response, "status_code", None)
                if code in (400, 404):
                    break
                time.sleep(SLEEP)
            except Exception:
                time.sleep(SLEEP)
        if not success:
            failed += 1

        time.sleep(SLEEP)

    print(f"\nSummary Price Daily: {created} created, {skipped} skipped, {failed} failed of {total} tickers.")

if __name__ == "__main__":
    build(years=2)