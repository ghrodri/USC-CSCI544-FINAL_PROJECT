import duckdb
from pathlib import Path

ROOT = Path("")
DATA = ROOT / "data"
DB_PATH = ROOT / "db/financial.duckdb"

CATALOG = DATA / "catalog/sp500_catalog.parquet"
PRICES_GLOB = DATA / "prices_daily" / "ticker=*" / "part-*.parquet"
DIVIDENDS_GLOB = DATA / "dividends" / "ticker=*" / "part-*.parquet"
SPLITS_GLOB = DATA / "splits" / "ticker=*" / "part-*.parquet"
REF_GLOB = DATA / "ref_tickers" / "ticker=*" / "part-*.parquet"
CALENDAR_PARQUET = DATA / "trading_calendar" / "trading_calendar.parquet"

con = duckdb.connect(DB_PATH.as_posix())

# Schemas
con.execute("create schema if not exists ext;")
con.execute("create schema if not exists core;")
con.execute("create schema if not exists views;")

# External sources
con.execute(f"""
create or replace view ext.catalog as
select * from read_parquet('{CATALOG.as_posix()}');
""")

con.execute(f"""
create or replace view ext.prices_daily as
select * from read_parquet('{PRICES_GLOB.as_posix()}');
""")

con.execute(f"""
create or replace view ext.dividends as
select * from read_parquet('{DIVIDENDS_GLOB.as_posix()}', union_by_name=true);
""")

con.execute(f"""
create or replace view ext.splits as
select * from read_parquet('{SPLITS_GLOB.as_posix()}', union_by_name=true);
""")

con.execute(f"""
create or replace view ext.ref_tickers as
select * from read_parquet('{REF_GLOB.as_posix()}', union_by_name=true);
""")

con.execute(f"""
create or replace view ext.trading_calendar as
select * from read_parquet('{CALENDAR_PARQUET.as_posix()}');
""")

# Core tables
con.execute("""
create or replace view core.tickers as
select
  upper(ticker)::varchar as ticker,
  shortname::varchar as name,
  sector::varchar as sector,
  exchange::varchar as exchange,
  longname::varchar as longname,
  industry::varchar as industry,
  city::varchar as city,
  state::varchar as state,
  country::varchar as country
from ext.catalog;
""")

con.execute("""
create or replace view core.prices_daily as
select
  cast(date as date) as date,
  upper(ticker) as ticker,
  cast(open as double) as open,
  cast(high as double) as high,
  cast(low as double) as low,
  cast(close as double) as close,
  cast(volume as bigint) as volume,
  cast(vwap as double) as vwap,
  cast(transactions as bigint) as transactions
from ext.prices_daily;
""")

con.execute("""
create or replace view core.dividends as
select
  upper(ticker) as ticker,
  cast(coalesce(event_date, pay_date, ex_dividend_date, record_date, declaration_date) as date) as event_date,
  cast(ex_dividend_date as date) as ex_dividend_date,
  cast(pay_date as date) as pay_date,
  cast(record_date as date) as record_date,
  cast(cash_amount as double) as cash_amount
from ext.dividends;
""")

con.execute("""
create or replace view core.splits as
select
  upper(ticker) as ticker,
  cast(coalesce(event_date, execution_date) as date) as event_date,
  cast(execution_date as date) as execution_date,
  cast(split_from as double) as split_from,
  cast(split_to as double) as split_to
from ext.splits;
""")

# Reference tickers (dynamic detection)
ref_columns = con.execute("describe select * from ext.ref_tickers").df()["column_name"].str.lower().tolist()
def has_column(c): return c.lower() in ref_columns

company_col = "company_name" if has_column("company_name") else ("name" if has_column("name") else None)
industry_col = "industry" if has_column("industry") else None
sector_col = "sector" if has_column("sector") else None
locale_col = "locale" if has_column("locale") else None
market_col = "market" if has_column("market") else None
primary_ex_col = "primary_exchange" if has_column("primary_exchange") else None

ref_select = [
    "upper(r.ticker) as ticker",
    f"cast(r.{company_col} as varchar) as company_name" if company_col else "cast(NULL as varchar) as company_name",
    f"cast(r.{industry_col} as varchar) as industry" if industry_col else "cast(NULL as varchar) as industry",
    f"cast(r.{sector_col} as varchar) as sector" if sector_col else "cast(NULL as varchar) as sector",
    f"cast(r.{locale_col} as varchar) as locale" if locale_col else "cast(NULL as varchar) as locale",
    f"cast(r.{market_col} as varchar) as market" if market_col else "cast(NULL as varchar) as market",
    f"cast(r.{primary_ex_col} as varchar) as primary_exchange" if primary_ex_col else "cast(NULL as varchar) as primary_exchange",
]

con.execute(f"""
create or replace view core.ref_tickers as
select
  {", ".join(ref_select)}
from ext.ref_tickers r;
""")

con.execute("""
create or replace view core.trading_calendar as
select
  lower(exchange) as exchange,
  cast(date as date) as date,
  cast(is_open as boolean) as is_open,
  cast(market_open_utc as timestamp with time zone) as market_open_utc,
  cast(market_close_utc as timestamp with time zone) as market_close_utc,
  cast(market_open_ny as timestamp with time zone) as market_open_ny,
  cast(market_close_ny as timestamp with time zone) as market_close_ny
from ext.trading_calendar;
""")

# Views
con.execute("""
create or replace view views.prices_enriched as
select
  p.date, p.ticker, p.open, p.high, p.low, p.close, p.volume, p.vwap, p.transactions,
  t.name, t.sector, t.exchange
from core.prices_daily p
left join core.tickers t using (ticker);
""")

con.execute("""
create or replace view views.daily_returns as
select
  date,
  ticker,
  close / lag(close) over (partition by ticker order by date) - 1.0 as ret
from core.prices_daily;
""")

con.execute("""
create or replace view views.daily_returns_open_days as
select r.*
from views.daily_returns r
join core.trading_calendar c
  on c.exchange = 'nyse' and c.date = r.date and c.is_open = true;
""")

con.execute("""
create or replace view views.prices_weekly as
select
  ticker,
  date_trunc('week', date) as week,
  first(open) as open,
  max(high) as high,
  min(low) as low,
  last(close) as close,
  sum(volume) as volume,
  last(vwap) as vwap
from core.prices_daily
group by ticker, date_trunc('week', date)
order by ticker, week;
""")

con.execute("""
create or replace view views.prices_monthly as
select
  ticker,
  date_trunc('month', date) as month,
  first(open) as open,
  max(high) as high,
  min(low) as low,
  last(close) as close,
  sum(volume) as volume,
  last(vwap) as vwap
from core.prices_daily
group by ticker, date_trunc('month', date)
order by ticker, month;
""")

con.execute("""
create or replace view views.corporate_actions as
select 'dividend' as action_type, ticker, event_date,
       cash_amount, null::double as split_from, null::double as split_to
from core.dividends
union all
select 'split' as action_type, ticker, event_date,
       null::double as cash_amount, split_from, split_to
from core.splits;
""")

con.execute("""
create or replace view views.adjusted_prices_splits as
with dates as (
  select ticker, date, close from core.prices_daily
),
sf as (
  select
    ticker,
    event_date as date,
    case when coalesce(split_from,0) > 0 and coalesce(split_to,0) > 0
         then split_from / split_to else 1.0 end as adj_factor
  from core.splits
),
merged as (
  select d.ticker, d.date, d.close, coalesce(sf.adj_factor, 1.0) as adj_factor
  from dates d
  left join sf on sf.ticker = d.ticker and sf.date = d.date
),
cum as (
  select
    ticker, date, close,
    exp(sum(case when adj_factor>0 then log(adj_factor) else 0 end)
        over (partition by ticker order by date
              rows between unbounded preceding and current row)) as cum_adj_factor
  from merged
)
select
  ticker, date, close,
  close * cum_adj_factor as close_adj_splits,
  cum_adj_factor
from cum;
""")

con.execute("""
create or replace view views.dividend_yield_ttm as
with all_dates as (
  select p.date, p.ticker, p.close from core.prices_daily p
),
dv as (
  select ticker, event_date as date, cash_amount from core.dividends
),
joined as (
  select a.date, a.ticker, a.close, coalesce(dv.cash_amount, 0.0) as cash_amount
  from all_dates a left join dv on dv.ticker = a.ticker and dv.date = a.date
),
agg as (
  select
    date, ticker, close,
    sum(cash_amount) over (
      partition by ticker
      order by date::timestamp
      range between interval 365 days preceding and current row
    ) as cash_ttm
  from joined
)
select
  date, ticker, close, cash_ttm,
  case when close > 0 then cash_ttm / close else null end as dividend_yield_ttm
from agg;
""")
print("Schema initialized", DB_PATH.as_posix())