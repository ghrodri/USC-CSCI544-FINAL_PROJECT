import os
from pathlib import Path
from datetime import date, datetime, timedelta, time
import pandas as pd
import pytz
import requests
import pandas_market_calendars as mcal

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
BASE_URL = "https://api.polygon.io"
OUT_DIR = Path("/src/data/trading_calendar")

NY_TZ = pytz.timezone("America/New_York")
UTC_TZ = pytz.UTC

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

#Build future holidays and events (API)
def polygon_trading_calendar(exchanges=("nyse", "nasdaq")) -> pd.DataFrame:
    url = f"{BASE_URL}/v1/marketstatus/upcoming"
    params = {"apiKey": POLYGON_API_KEY}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    if not isinstance(js, list):
        return pd.DataFrame(columns=["exchange", "name", "date", "status", "open", "close"])
    rows = []
    for item in js:
        ex = str(item.get("exchange", "")).lower()
        if ex in exchanges:
            rows.append({
                "exchange": ex,
                "name": item.get("name"),
                "date": item.get("date"),
                "status": item.get("status"),
                "open": item.get("open"),
                "close": item.get("close"),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df

# Build historical calendar
def build_historic_calendar(start_years=3):
    start = date.today() - timedelta(days=365 * start_years)
    end = date.today()
    calendars = {}
    for ex_code, label in {"NASDAQ": "nasdaq", "NYSE": "nyse"}.items():
        cal = mcal.get_calendar(ex_code)
        sched = cal.schedule(start_date=start, end_date=end)
        if sched.empty:
            continue
        sched = sched.reset_index().rename(columns={
            "market_open": "market_open_ny",
            "market_close": "market_close_ny",
        })
        sched["exchange"] = label
        calendars[label] = sched
    if not calendars:
        return pd.DataFrame()
    return pd.concat(calendars.values(), ignore_index=True)

#Merge historical and future calendars
def combine_calendars(hist_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
    if hist_df.empty and future_df.empty:
        return pd.DataFrame()
    hist_df["date"] = pd.to_datetime(hist_df["index"]).dt.date if "index" in hist_df.columns else pd.to_datetime(hist_df["date"]).dt.date
    hist_df["is_open"] = True
    future_df = future_df.copy()
    future_df["is_open"] = future_df["status"].apply(lambda s: s != "closed")
    future_df = future_df.rename(columns={"open": "market_open_utc", "close": "market_close_utc"})
    for col in ["market_open_utc", "market_close_utc"]:
        if col in future_df.columns:
            future_df[col] = pd.to_datetime(future_df[col], errors="coerce", utc=True)
    hist_df["market_open_utc"] = hist_df["market_open_ny"].dt.tz_convert("UTC")
    hist_df["market_close_utc"] = hist_df["market_close_ny"].dt.tz_convert("UTC")
    cols = ["exchange", "date", "is_open", "market_open_utc", "market_close_utc"]
    all_df = pd.concat([
        hist_df[cols],
        future_df[cols]
    ], ignore_index=True).drop_duplicates(subset=["exchange", "date"]).sort_values(["exchange", "date"])
    all_df["market_open_ny"] = all_df["market_open_utc"].dt.tz_convert(NY_TZ)
    all_df["market_close_ny"] = all_df["market_close_utc"].dt.tz_convert(NY_TZ)
    return all_df

#Main Build Function
def build(years_back=3):
    if not POLYGON_API_KEY:
        raise RuntimeError("missing POLYGON_API_KEY environment variable")
    ensure_dir(OUT_DIR)
    print("Fetching historic calendar...")
    hist_df = build_historic_calendar(start_years=years_back)
    print("Fetching future holidays from Polygon...")
    future_df = polygon_trading_calendar()
    combined = combine_calendars(hist_df, future_df)
    if combined.empty:
        print("No data combined.")
        return
    fname = OUT_DIR / "trading_calendar.parquet"
    combined.to_parquet(fname, index=False)
    min_d, max_d = combined["date"].min(), combined["date"].max()
    print("Done Trading calendar")

if __name__ == "__main__":
    build(years_back=3)