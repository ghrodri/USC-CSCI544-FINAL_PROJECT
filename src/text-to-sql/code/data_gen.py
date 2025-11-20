from pathlib import Path
import duckdb
import pandas as pd
import random
import json
from tqdm import tqdm

DB_DATA = "./src/db/financial.duckdb"
OUT_DIR_DATA = "./src/text-to-sql/data"
META_DATA = "./src/text-to-sql/data/dataset_metadata.json"
RANDOM_SEED = 321
PER_INTENT = 750
ALLOW_EMPTY_FOR = {"calendar_filter"}

def write_jsonl(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def pick(x):
    return random.choice(x) if x else None

def sample_n(x, k):
    k = min(k, len(x))
    return random.sample(x, k) if x else []

# validation
def sql_is_valid(con, sql: str, intent: str):
    try:
        df = con.sql(sql).df()
        if intent in ALLOW_EMPTY_FOR:
            return True, False
        return (len(df) > 0), (len(df) == 0)
    except Exception:
        return False, False

random.seed(RANDOM_SEED)
con = duckdb.connect(DB_DATA)

tickers = con.sql("select distinct ticker from core.tickers order by ticker").df()["ticker"].tolist()
sectors = con.sql("select distinct sector from core.tickers where sector is not null order by sector").df()["sector"].tolist()
dates_all = con.sql("select distinct date from core.prices_daily order by date").df()["date"].astype(str).tolist()
weeks_all = con.sql("select distinct week from views.prices_weekly order by week").df()["week"].astype(str).tolist()
months_all = con.sql("select distinct month from views.prices_monthly order by month").df()["month"].astype(str).tolist()

tickers_with_div = con.sql("select distinct ticker from core.dividends").df()["ticker"].tolist()
tickers_with_spl = con.sql("select distinct ticker from core.splits").df()["ticker"].tolist()
tickers_with_actions = con.sql("""
    select distinct ticker from core.splits
    union
    select distinct ticker from core.dividends
""").df()["ticker"].tolist()

# sector/date pairs from base tables
sector_dates_df = con.sql("""
  select distinct t.sector as sector, p.date as date
  from core.prices_daily p
  join core.tickers t using (ticker)
  where t.sector is not null
  order by 1,2
""").df()
sector_dates = list(zip(sector_dates_df["sector"].tolist(),
                        sector_dates_df["date"].astype(str).tolist()))

weekly_pairs = con.sql("select distinct ticker, week from views.prices_weekly").df().to_records(index=False).tolist()
monthly_pairs = con.sql("select distinct ticker, month from views.prices_monthly").df().to_records(index=False).tolist()

def choose_date_range():
    if len(dates_all) < 40:
        d1 = pick(dates_all); d2 = pick(dates_all)
        if d1 and d2 and d1 > d2:
            d1, d2 = d2, d1
        return d1, d2
    i1 = random.randint(0, len(dates_all) - 20)
    i2 = random.randint(i1 + 5, min(len(dates_all) - 1, i1 + random.randint(10, 90)))
    return dates_all[i1], dates_all[i2]

# generators
def g_point_ohlcv_first_day():
    t = pick(tickers)
    q = f"Show OHLCV for {t} on its first trading day."
    s = f"""
        with d as (select min(date) as d0 from core.prices_daily where ticker='{t}')
        select p.*
        from core.prices_daily p, d
        where p.ticker='{t}' and p.date=d.d0;
    """
    return {"question": q, "sql": s, "intent": "point_lookup"}

def g_latest_close_volume():
    t = pick(tickers)
    q = f"Latest close and volume for {t}."
    s = f"""
        with mx as (select max(date) as md from core.prices_daily where ticker='{t}')
        select date, close, volume
        from core.prices_daily, mx
        where ticker='{t}' and date=mx.md;
    """
    return {"question": q, "sql": s, "intent": "latest_point"}

def g_returns_window_7_30():
    t = pick(tickers)
    q = f"7-day and 30-day returns for {t}."
    s = f"""
        with base as (
          select date, ticker, close
          from core.prices_daily
          where ticker='{t}'
        )
        select
          date, ticker, close,
          close/lag(close,7) over (partition by ticker order by date)-1 as r7,
          close/lag(close,30) over (partition by ticker order by date)-1 as r30
        from base;
    """
    return {"question": q, "sql": s, "intent": "delta_window"}

def g_sector_agg_on_date():
    sd = pick(sector_dates) or (pick(sectors), pick(dates_all))
    sct, d = sd
    q = f"Average close and total volume for sector {sct} on {d}."
    s = f"""
        select avg(p.close) as avg_close, sum(p.volume) as total_volume
        from core.prices_daily p
        join core.tickers t using (ticker)
        where t.sector='{sct}' and p.date='{d}';
    """
    return {"question": q, "sql": s, "intent": "sector_aggregate"}

def g_weekly_bar():
    tw = pick(weekly_pairs) or (pick(tickers), pick(weeks_all))
    t, w = tw
    q = f"Weekly OHLCV for {t} in week {w}."
    s = f"""
        select *
        from views.prices_weekly
        where ticker='{t}' and week='{w}';
    """
    return {"question": q, "sql": s, "intent": "resample_weekly"}

def g_monthly_bar():
    tm = pick(monthly_pairs) or (pick(tickers), pick(months_all))
    t, m = tm
    q = f"Monthly OHLCV for {t} in {m}."
    s = f"""
        select *
        from views.prices_monthly
        where ticker='{t}' and month='{m}';
    """
    return {"question": q, "sql": s, "intent": "resample_monthly"}

def g_calendar_open():
    d = pick(dates_all)
    q = f"Was NYSE open on {d}?"
    s = f"""
        select is_open
        from core.trading_calendar
        where exchange='nyse' and date='{d}';
    """
    return {"question": q, "sql": s, "intent": "calendar_filter"}

def g_dividend_yield_ttm_series():
    t = pick(tickers)
    q = f"Dividend yield TTM time series for {t}."
    s = f"""
        select date, dividend_yield_ttm
        from views.dividend_yield_ttm
        where ticker='{t}' order by date;
    """
    return {"question": q, "sql": s, "intent": "dividend_yield"}

def g_corporate_actions():
    t = pick(tickers_with_actions) or pick(tickers)
    q = f"Corporate actions for {t} ordered by date."
    s = f"""
        select *
        from views.corporate_actions
        where ticker='{t}'
        order by event_date;
    """
    return {"question": q, "sql": s, "intent": "corporate_actions"}

def g_adjusted_split_close_series():
    t = pick(tickers)
    q = f"Split-adjusted close series for {t}."
    s = f"""
        select date, close_adj_splits
        from views.adjusted_prices_splits
        where ticker='{t}' order by date;
    """
    return {"question": q, "sql": s, "intent": "adjusted_price"}

def g_top_movers_on_day():
    d = pick(dates_all)
    q = f"Top 10 gainers by daily return on {d}."
    s = f"""
        with r as (
          select
            ticker,
            close/lag(close) over (partition by ticker order by date)-1 as ret
          from core.prices_daily
          where date='{d}'
        )
        select ticker, ret
        from r
        order by ret desc
        limit 10;
    """
    return {"question": q, "sql": s, "intent": "top_k"}

def g_between_dates_avg_close_by_ticker():
    t = pick(tickers)
    d1, d2 = choose_date_range()
    q = f"Average close for {t} from {d1} to {d2}."
    s = f"""
        select avg(close) as avg_close
        from core.prices_daily
        where ticker='{t}' and date between '{d1}' and '{d2}';
    """
    return {"question": q, "sql": s, "intent": "avg_between_dates"}

def g_join_prices_dividends_window():
    t = pick(tickers_with_div) or pick(tickers)
    d1, d2 = choose_date_range()
    q = f"Close price and dividend cash events for {t} between {d1} and {d2}."
    s = f"""
        select p.date, p.close, coalesce(d.cash_amount, 0) as cash_amount
        from core.prices_daily p
        left join core.dividends d
          on d.ticker=p.ticker and d.event_date=p.date
        where p.ticker='{t}' and p.date between '{d1}' and '{d2}'
        order by p.date;
    """
    return {"question": q, "sql": s, "intent": "join_dividends"}

def g_join_prices_splits_window():
    t = pick(tickers_with_spl) or pick(tickers)
    d1, d2 = choose_date_range()
    q = f"Close price and split factors for {t} between {d1} and {d2}."
    s = f"""
        select
          p.date, p.close,
          case when s.split_from>0 and s.split_to>0
               then s.split_from/s.split_to else null end as split_factor
        from core.prices_daily p
        left join core.splits s
          on s.ticker=p.ticker and s.event_date=p.date
        where p.ticker='{t}' and p.date between '{d1}' and '{d2}'
        order by p.date;
    """
    return {"question": q, "sql": s, "intent": "join_splits"}

def g_ytd_cum_return():
    t = pick(tickers)
    q = f"Year-to-date cumulative return for {t}."
    s = f"""
        with dr as (
          select date, ticker,
                 close/lag(close) over (partition by ticker order by date)-1 as ret
          from core.prices_daily
          where ticker='{t}'
        ),
        with_y as (
          select date, ticker, ret, date_trunc('year', date) as y
          from dr
        ),
        agg as (
          select date, ticker,
                 exp(sum(ln(1+ret)) over (partition by ticker, y
                    order by date rows between unbounded preceding and current row))-1 as cum_ret_ytd
          from with_y
        )
        select date, cum_ret_ytd
        from agg
        order by date;
    """
    return {"question": q, "sql": s, "intent": "ytd"}

def g_company_profile():
    t = pick(tickers)
    q = f"Company profile for {t} including sector and exchange."
    s = f"""
        select t.ticker, t.name, t.sector, t.exchange,
               r.company_name, r.industry, r.market, r.primary_exchange
        from core.tickers t
        left join core.ref_tickers r using (ticker)
        where t.ticker='{t}';
    """
    return {"question": q, "sql": s, "intent": "profile"}

def g_sector_members_on_date():
    sd = pick(sector_dates) or (pick(sectors), pick(dates_all))
    sct, d = sd
    q = f"Tickers in sector {sct} that traded on {d}."
    s = f"""
        select distinct p.ticker
        from core.prices_daily p
        join core.tickers t using (ticker)
        where p.date='{d}' and t.sector='{sct}'
        order by 1;
    """
    return {"question": q, "sql": s, "intent": "membership"}

generators = [
    g_point_ohlcv_first_day,
    g_latest_close_volume,
    g_returns_window_7_30,
    g_sector_agg_on_date,
    g_weekly_bar,
    g_monthly_bar,
    g_calendar_open,
    g_dividend_yield_ttm_series,
    g_corporate_actions,
    g_adjusted_split_close_series,
    g_top_movers_on_day,
    g_between_dates_avg_close_by_ticker,
    g_join_prices_dividends_window,
    g_join_prices_splits_window,
    g_ytd_cum_return,
    g_company_profile,
    g_sector_members_on_date,
]

# generate raw candidates
raw_examples = []
for gen in tqdm(generators, desc="generate intents"):
    for _ in range(PER_INTENT):
        raw_examples.append(gen())

# validate by execution
valid_examples = []
failed_runtime = 0
failed_empty = 0

for ex in tqdm(raw_examples, desc="validate sql"):
    sql_clean = " ".join(ex["sql"].split())
    ok, was_empty = sql_is_valid(con, sql_clean, ex["intent"])
    if ok:
        valid_examples.append({"question": ex["question"], "sql": sql_clean, "intent": ex["intent"]})
    else:
        if was_empty:
            failed_empty += 1
        else:
            failed_runtime += 1

seen_sql = set()
dedup_examples = []
for r in valid_examples:
    if r["sql"] not in seen_sql:
        dedup_examples.append(r)
        seen_sql.add(r["sql"])

# split data
random.shuffle(dedup_examples)
total = len(dedup_examples)
n_train = int(total * 0.8)
n_val = int(total * 0.1)

train_rows = dedup_examples[:n_train]
val_rows = dedup_examples[n_train:n_train + n_val]
test_rows = dedup_examples[n_train + n_val:]

# write outputs
out_dir = Path(OUT_DIR_DATA)
out_dir.mkdir(parents=True, exist_ok=True)

write_jsonl(out_dir / "train.jsonl", train_rows)
write_jsonl(out_dir / "val.jsonl", val_rows)
write_jsonl(out_dir / "test.jsonl", test_rows)

# write metadata
meta = {
    "db_path": DB_DATA,
    "counts": {"total": total, "train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
    "intents": sorted(list({r["intent"] for r in dedup_examples})),
    "seed": RANDOM_SEED,
    "per_intent_requested": PER_INTENT,
    "validation_stats": {
        "raw": len(raw_examples),
        "kept": total,
        "failed_runtime": failed_runtime,
        "failed_empty_or_disallowed": failed_empty
    }
}
Path(META_DATA).write_text(json.dumps(meta, indent=2), encoding="utf-8")

print(
    f"Done | total={total} train={len(train_rows)} val={len(val_rows)} test={len(test_rows)} "
    f"intents={len(meta['intents'])} | raw={len(raw_examples)} kept={total} "
    f"fail_runtime={failed_runtime} fail_empty={failed_empty}"
)