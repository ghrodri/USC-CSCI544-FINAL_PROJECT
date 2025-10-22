import os
import pandas as pd
CSV_DATA = "./src/data/catalog/sp500_companies.csv"
OUT_DIR_DATA = "./src/data/catalog"
CATALOG_OUT_DATA = "./src/data/catalog/sp500_catalog.parquet"
df = pd.read_csv(CSV_DATA)
# rename columns 
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
# normalize string fields
cols = list(df.columns)
for i in range(len(cols)):
    c = cols[i]
    if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
        df[c] = df[c].astype(str).str.strip()

if "ticker" in df.columns:
    df["ticker"] = df["ticker"].str.upper()
if "exchange" in df.columns:
    df["exchange"] = df["exchange"].str.upper()

# check output dir
os.makedirs(OUT_DIR_DATA, exist_ok=True)

# write parquet
df.to_parquet(CATALOG_OUT_DATA, index=False, compression="snappy")

print("Done SP500 Catalog")