
import pandas as pd
def load_sp500_catalog(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.copy()
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    return df


def build_catalog_maps(df: pd.DataFrame):
    name_to_ticker = {row["name"].lower(): row["ticker"] for _, row in df.iterrows()}
    ticker_to_name = {row["ticker"]: row["name"] for _, row in df.iterrows()}
    ticker_to_sector = {row["ticker"]: row["sector"] for _, row in df.iterrows()}

    sector_to_tickers = {}
    for t, s in ticker_to_sector.items():
        if s not in sector_to_tickers:
            sector_to_tickers[s] = []
        sector_to_tickers[s].append(t)

    return {
        "name_to_ticker": name_to_ticker,
        "ticker_to_name": ticker_to_name,
        "ticker_to_sector": ticker_to_sector,
        "sector_to_tickers": sector_to_tickers,
    }