from pathlib import Path
import pandas as pd
import re
csv_path = Path("/src/data/catalog/sp500_companies.csv")
out_dir = Path("/src/data/catalog")
catalog_out = out_dir / "sp500_catalog.parquet"
df = pd.read_csv(csv_path)
df = df.rename(columns={
    "Exchange": "exchange",
    "Symbol": "ticker",
    "Shortname": "shortname",
    "Longname": "longname",
    "Sector": "sector",
    "Industry": "industry",
    "City": "city",
    "State": "state",
    "Country": "country"
})
for col in df.columns:
    if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
        df[col] = df[col].astype(str).str.strip()
if "ticker" in df.columns:
    df["ticker"] = df["ticker"].str.upper()
if "exchange" in df.columns:
    df["exchange"] = df["exchange"].str.upper()
out_dir.mkdir(parents=True, exist_ok=True)
df.to_parquet(catalog_out, index=False, compression="snappy")
if all(col in df.columns for col in ["ticker", "shortname", "sector"]):
    symbols_df = df[["ticker", "shortname", "sector"]].rename(columns={"shortname": "name"})
print("Done SP500 Catalog")