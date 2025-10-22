import os
from datetime import date, timedelta
import pandas as pd
import pytz
import requests
import pandas_market_calendars as mcal

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
BASE_URL = "https://api.polygon.io"
OUT_DIR = "./src/data/trading_calendar"
OUT_FILE = "./src/data/trading_calendar/trading_calendar.parquet"
NY_TZ = pytz.timezone("America/New_York")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def request_json(url, params, timeout=30):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def to_utc_series(s):
    # robustly UTC tz-aware datetimes
    try:
        tz = getattr(s.dt, "tz", None)
        if tz is None:
            return s.dt.tz_localize("UTC", nonexistent="NaT", ambiguous="NaT")
        return s.dt.tz_convert("UTC")
    except Exception:
        return pd.to_datetime(s, errors="coerce", utc=True)


def to_ny_series_from_utc(s):
    # assumes s is tz-aware (UTC); convert to New York tz
    try:
        return s.dt.tz_convert(NY_TZ)
    except Exception:
        s2 = pd.to_datetime(s, errors="coerce", utc=True)
        return s2.dt.tz_convert(NY_TZ)


#  future holidays & events with Polygon
def polygon_trading_calendar(exchanges=("nyse", "nasdaq")):
    url = f"{BASE_URL}/v1/marketstatus/upcoming"
    params = {"apiKey": POLYGON_API_KEY}
    js = request_json(url, params)
    if not isinstance(js, list):
        return pd.DataFrame(columns=["exchange", "date", "status", "market_open_utc", "market_close_utc"])

    rows = []
    for i in range(len(js)):
        item = js[i]
        ex = str(item.get("exchange", "")).lower()
        if ex in exchanges:
            rows.append({
                "exchange": ex,
                "date": item.get("date"),
                "status": item.get("status"),
                "open": item.get("open"),
                "close": item.get("close"),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["market_open_utc"] = pd.to_datetime(df["open"], errors="coerce", utc=True)
    df["market_close_utc"] = pd.to_datetime(df["close"], errors="coerce", utc=True)
    df["is_open"] = df["status"].astype(str).str.lower().ne("closed")
    return df[["exchange", "date", "is_open", "market_open_utc", "market_close_utc"]]


#  historical calendar with pandas-market-calendars
def build_historic_calendar(years_back=3):
    start = date.today() - timedelta(days=365 * years_back)
    end = date.today()

    out = []
    # "NYSE", "NASDAQ"
    ex_map = [("nyse", "NYSE"), ("nasdaq", "NASDAQ")]
    for i in range(len(ex_map)):
        label, code = ex_map[i][0], ex_map[i][1]
        cal = mcal.get_calendar(code)
        sched = cal.schedule(start_date=start, end_date=end)
        if sched.empty:
            continue
        sched = sched.reset_index()  # index column is the session date
        sched = sched.rename(columns={"index": "date"})
        # pandas_market_calendars returns UTC by default
        sched["market_open_utc"] = to_utc_series(sched["market_open"])
        sched["market_close_utc"] = to_utc_series(sched["market_close"])
        sched["exchange"] = label
        sched["date"] = pd.to_datetime(sched["date"], errors="coerce").dt.date
        sched["is_open"] = True
        out.append(sched[["exchange", "date", "is_open", "market_open_utc", "market_close_utc"]])

    if len(out) == 0:
        return pd.DataFrame(columns=["exchange", "date", "is_open", "market_open_utc", "market_close_utc"])

    return pd.concat(out, ignore_index=True)


# merge historical and future 
def combine_calendars(hist_df, future_df):
    if hist_df.empty and future_df.empty:
        return pd.DataFrame()

    frames = []
    if not hist_df.empty:
        frames.append(hist_df[["exchange", "date", "is_open", "market_open_utc", "market_close_utc"]])
    if not future_df.empty:
        frames.append(future_df[["exchange", "date", "is_open", "market_open_utc", "market_close_utc"]])

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.drop_duplicates(subset=["exchange", "date"]).sort_values(["exchange", "date"])

    # add NY times derived from UTC
    all_df["market_open_ny"] = to_ny_series_from_utc(all_df["market_open_utc"])
    all_df["market_close_ny"] = to_ny_series_from_utc(all_df["market_close_utc"])
    return all_df


def build(years_back=3):
    # export POLYGON_API_KEY="your_polygon_key"
    if POLYGON_API_KEY == "":
        raise RuntimeError("missing POLYGON_API_KEY environment variable")

    ensure_dir(OUT_DIR)

    print("Fetching historic calendar...")
    hist_df = build_historic_calendar(years_back=years_back)

    print("Fetching future holidays from Polygon...")
    fut_df = polygon_trading_calendar()

    df = combine_calendars(hist_df, fut_df)
    if df.empty:
        print("No calendar data found.")
        return

    df.to_parquet(OUT_FILE, index=False)
    dmin = df["date"].min()
    dmax = df["date"].max()
    print(f"Trading calendar saved: {OUT_FILE}  range=[{dmin} .. {dmax}]")

if __name__ == "__main__":
    build(years_back=3)