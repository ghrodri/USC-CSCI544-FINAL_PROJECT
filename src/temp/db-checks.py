from pathlib import Path
import sys
import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "db" / "financial.duckdb"
if not DB_PATH.exists():
    print(f"[error] DuckDB not found at: {DB_PATH}")
    sys.exit(1)

con = duckdb.connect(DB_PATH.as_posix())

tables = con.sql("""
  select table_schema, table_name
  from information_schema.tables
  where table_schema in ('core','views')
  order by table_schema, table_name
""").df()
print("Schemas and tables:")
print(tables.to_string(index=False))

print("\nRow counts:")
for _, r in tables.iterrows():
    name = f'{r["table_schema"]}.{r["table_name"]}'
    try:
        n = con.sql(f"select count(*) as c from {name}").fetchone()[0]
        print(f"  {name:<35} {n}")
    except Exception as e:
        print(f"  {name:<35} ERR: {e}")

print("\nDate ranges:")
checks = [
    ("core.prices_daily", "date"),
    ("views.daily_returns", "date"),
    ("views.prices_weekly", "week"),
    ("views.prices_monthly", "month"),
    ("core.trading_calendar", "date"),
]
for rel, col in checks:
    try:
        df = con.sql(f"select min({col}) as min_d, max({col}) as max_d from {rel}").df()
        print(f"  {rel:<28} {df.loc[0,'min_d']} → {df.loc[0,'max_d']}")
    except Exception as e:
        print(f"  {rel:<28} ERR: {e}")

print("\nSmoke test on sensitive views:")
for rel in ["views.adjusted_prices_splits", "views.dividend_yield_ttm"]:
    try:
        df = con.sql(f"select * from {rel} limit 5").df()
        print(f"  {rel}: ok ({len(df)} rows)")
    except Exception as e:
        print(f"  {rel}: ERR: {e}")

print(f"\nOK — inspected {DB_PATH}")