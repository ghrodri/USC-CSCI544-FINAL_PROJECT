from typing import Dict, List
import pandas as pd
COLUMN_SYNONYMS: Dict[str, List[str]] = {
    # prices_daily
    "ticker": [
        "ticker",
        "symbol",
        "stock symbol",
        "stock ticker",
        "company ticker",
        "company symbol",
    ],
    "date": [
        "date",
        "trading date",
        "session date",
        "calendar date",
        "day",
    ],
    "open": [
        "open",
        "opening price",
        "first price",
        "start price",
        "price at open",
    ],
    "high": [
        "high",
        "daily high",
        "highest price",
        "intraday high",
    ],
    "low": [
        "low",
        "daily low",
        "lowest price",
        "intraday low",
    ],
    "close": [
        "close",
        "closing price",
        "last price",
        "last close",
        "end of day price",
        "final price",
        "recent close",
        "latest close",
    ],
    "volume": [
        "volume",
        "shares traded",
        "trading volume",
        "daily volume",
    ],

    # dividends
    "dividend": [
        "dividend",
        "dividend payout",
        "cash dividend",
        "distribution",
        "dividend per share",
    ],

    # splits
    "split_ratio": [
        "split ratio",
        "split",
        "split factor",
        "stock split",
    ],

    # catalog
    "name": [
        "company name",
        "name",
        "full name",
        "issuer name",
    ],
    "sector": [
        "sector",
        "business sector",
        "market sector",
        "industry sector",
    ],
    "industry": [
        "industry",
        "industry group",
        "business group",
    ],

    # portfolio
    "shares": [
        "shares",
        "position",
        "size",
        "quantity",
        "number of shares",
        "units",
    ],
    "avg_cost": [
        "average cost",
        "avg cost",
        "cost basis",
        "average price",
        "purchase price",
    ],

    # trading_calendar
    "is_open": [
        "is open",
        "market open",
        "was open",
        "open flag",
        "open or closed",
        "trading day",
    ],
    "market_open_utc": [
        "market open utc",
        "open time utc",
        "opening time utc",
    ],
    "market_close_utc": [
        "market close utc",
        "close time utc",
        "closing time utc",
    ],
    "market_open_ny": [
        "market open ny",
        "open time ny",
        "opening time ny",
        "ny open",
    ],
    "market_close_ny": [
        "market close ny",
        "close time ny",
        "closing time ny",
        "ny close",
    ],
}

METRIC_SYNONYMS: Dict[str, List[str]] = {
    "daily_return": [
        "daily return",
        "day return",
        "one day return",
        "percentage change for the day",
        "percent move today",
        "day over day return",
    ],
    "ytd_return": [
        "ytd return",
        "year to date return",
        "return this year",
        "performance this year",
        "year to date performance",
    ],
    "total_return": [
        "total return",
        "overall return",
        "cumulative return",
        "return over the period",
    ],
    "volatility": [
        "volatility",
        "price volatility",
        "return volatility",
        "risk",
    ],
    "portfolio_value": [
        "portfolio value",
        "total portfolio value",
        "portfolio market value",
        "value of my portfolio",
    ],
    "unrealized_pnl": [
        "unrealized pnl",
        "unrealized profit",
        "unrealized loss",
        "paper gains",
        "paper losses",
    ],
}

TIME_SYNONYMS: Dict[str, List[str]] = {
    "today": [
        "today",
        "as of today",
        "current day",
        "right now",
    ],
    "yesterday": [
        "yesterday",
        "previous day",
        "last day",
    ],
    "last_week": [
        "last week",
        "past week",
        "previous week",
        "over the last week",
    ],
    "last_month": [
        "last month",
        "past month",
        "previous month",
        "over the last month",
        "in the last 30 days",
    ],
    "last_year": [
        "last year",
        "past year",
        "previous year",
        "over the last year",
        "in the last 12 months",
    ],
    "ytd": [
        "year to date",
        "ytd",
        "so far this year",
        "this year",
        "current year",
    ],
    "recent": [
        "recent",
        "lately",
        "most recent",
        "recent days",
    ],
}
def build_ticker_aliases(catalog_df: pd.DataFrame) -> Dict[str, list]:
    aliases: Dict[str, list] = {}

    df = catalog_df.copy()
    df.columns = df.columns.str.lower()

    if "ticker" not in df.columns or "name" not in df.columns:
        raise ValueError("Catalog dataframe must contain 'ticker' and 'name' columns.")

    for _, row in df.iterrows():
        ticker = str(row["ticker"]).upper()
        name = str(row["name"]).strip()

        base_aliases = set()

        base_aliases.add(ticker)
        base_aliases.add(ticker.lower())

        lower_name = name.lower()
        base_aliases.add(lower_name)

        for suffix in [" inc.", " inc", " corporation", " corp.", " corp", ", inc.", ", inc"]:
            if lower_name.endswith(suffix):
                base_aliases.add(lower_name.replace(suffix, "").strip())

        aliases[ticker] = sorted(base_aliases)

    return aliases

def invert_column_synonyms() -> Dict[str, str]:
    reverse: Dict[str, str] = {}

    for canonical, variants in COLUMN_SYNONYMS.items():
        reverse[canonical] = canonical 
        for v in variants:
            reverse[v.lower()] = canonical

    return reverse

def invert_metric_synonyms() -> Dict[str, str]:
    reverse: Dict[str, str] = {}

    for canonical, variants in METRIC_SYNONYMS.items():
        reverse[canonical] = canonical
        for v in variants:
            reverse[v.lower()] = canonical

    return reverse

def invert_time_synonyms() -> Dict[str, str]:
    reverse: Dict[str, str] = {}

    for canonical, variants in TIME_SYNONYMS.items():
        reverse[canonical] = canonical
        for v in variants:
            reverse[v.lower()] = canonical

    return reverse