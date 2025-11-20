import json
import math
import os
import random
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import duckdb
import ollama
import pandas as pd
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

DB_DATA = "./src/db/financial.duckdb"

OUT_DIR_DATA = "./src/text-to-sql/data_new"
OUT_DIR_TRAIN = os.path.join(OUT_DIR_DATA, "train")
OUT_DIR_VAL = os.path.join(OUT_DIR_DATA, "val")
OUT_DIR_TEST = os.path.join(OUT_DIR_DATA, "test")
META_DATA = os.path.join(OUT_DIR_DATA, "dataset_metadata.json")

SP500_CATALOG_PATH = "./src/data/catalog/sp500_catalog.parquet"

# Total targets (80 / 15 / 5 split)
TRAIN_TARGET = 200_000
VAL_TARGET = 37_500
TEST_TARGET = 12_500
TOTAL_TARGET = TRAIN_TARGET + VAL_TARGET + TEST_TARGET  # 250_000

# Train is sharded for easier handling
TRAIN_SHARD_SIZE = 10_000
NUM_TRAIN_SHARDS = TRAIN_TARGET // TRAIN_SHARD_SIZE  # 20 shards of 10k examples

# Synthetic vs LLM mix for train
TRAIN_LLM_FRACTION = 0.30  # 30% of train examples via LLM
TRAIN_SYNTH_FRACTION = 0.70

# Model used by Ollama (make sure it is installed and pulled)
OLLAMA_MODEL = "llama3.1:8b"

# Random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def open_connection(db_path: str) -> duckdb.DuckDBPyConnection:
    """
    Open a DuckDB connection to the given database file.
    """
    con = duckdb.connect(db_path, read_only=False)
    return con


def load_available_dates(con: duckdb.DuckDBPyConnection) -> List[str]:
    """
    Load the list of distinct available dates in core.prices_daily ordered ascending.
    This is used to ensure we only generate SQL with valid dates.
    """
    sql = "SELECT DISTINCT date FROM core.prices_daily ORDER BY date;"
    rows = con.execute(sql).fetchall()
    return [str(r[0]) for r in rows]


def load_sp500_catalog(path: str) -> pd.DataFrame:
    """
    Load the SP500 catalog from a parquet file and normalize column names.

    Expected columns (or similar):
        - ticker
        - name
        - sector
        - industry
    """
    df = pd.read_parquet(path)
    df = df.copy()
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    return df


def build_catalog_maps(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Build helper mappings from the SP500 catalog:
        - name_to_ticker
        - ticker_to_name
        - ticker_to_sector
        - sector_to_tickers
    """
    name_to_ticker = {}
    ticker_to_name = {}
    ticker_to_sector = {}
    sector_to_tickers: Dict[str, List[str]] = {}

    for _, row in df.iterrows():
        ticker = str(row["ticker"])
        name = str(row.get("name", ticker))
        sector = str(row.get("sector", "Unknown"))

        name_to_ticker[name.lower()] = ticker
        ticker_to_name[ticker] = name
        ticker_to_sector[ticker] = sector

        if sector not in sector_to_tickers:
            sector_to_tickers[sector] = []
        sector_to_tickers[sector].append(ticker)

    return {
        "name_to_ticker": name_to_ticker,
        "ticker_to_name": ticker_to_name,
        "ticker_to_sector": ticker_to_sector,
        "sector_to_tickers": sector_to_tickers,
    }

def build_synthetic_portfolio(catalog_df: pd.DataFrame, size: int = 30) -> pd.DataFrame:
    """
    Build a synthetic portfolio with a random subset of SP500 tickers.
    The schema is:

        portfolio(
            ticker TEXT,
            shares DOUBLE,
            avg_cost DOUBLE
        )
    """
    df = catalog_df.copy()
    if len(df) == 0:
        raise ValueError("SP500 catalog is empty; cannot build synthetic portfolio.")

    # Sample unique tickers
    sample_size = min(size, len(df))
    sample_df = df.sample(sample_size, random_state=RANDOM_SEED).reset_index(drop=True)

    # Synthetic shares and avg_cost
    rng = random.Random(RANDOM_SEED + 123)
    shares = [rng.randint(5, 500) for _ in range(sample_size)]
    # We do not know real cost basis, but we can use a plausible range
    avg_cost = [rng.uniform(20.0, 400.0) for _ in range(sample_size)]

    portfolio_df = pd.DataFrame(
        {
            "ticker": sample_df["ticker"].astype(str).tolist(),
            "shares": shares,
            "avg_cost": avg_cost,
        }
    )

    return portfolio_df


def register_portfolio_table(con, catalog_df):
    """
    Registers a synthetic portfolio using the SP500 catalog.
    User-provided portfolio is NOT used during dataset generation.
    """
    synthetic = build_synthetic_portfolio(catalog_df, size=30)

    con.register("portfolio_temp", synthetic)

    con.execute("""
        CREATE OR REPLACE VIEW portfolio AS
        SELECT * FROM portfolio_temp;
    """)

def random_ticker(catalog_df: pd.DataFrame) -> str:
    """
    Pick a random ticker from the catalog.
    """
    idx = random.randint(0, len(catalog_df) - 1)
    return str(catalog_df.iloc[idx]["ticker"])


def random_sector(catalog_maps: Dict[str, Dict]) -> str:
    """
    Pick a random sector from the catalog maps.
    """
    sectors = list(catalog_maps["sector_to_tickers"].keys())
    return random.choice(sectors)


def random_date(available_dates: List[str]) -> str:
    """
    Pick a random valid date from prices_daily.
    """
    return random.choice(available_dates)


def random_date_range(available_dates: List[str]) -> Tuple[str, str]:
    """
    Pick a random valid date range (d1, d2) such that d1 <= d2.
    """
    d1 = random_date(available_dates)
    d2 = random_date(available_dates)
    return (min(d1, d2), max(d1, d2))


def random_recent_date(available_dates: List[str], window_days: int = 60) -> str:
    """
    Pick a date biased towards the most recent window_days.
    """
    if not available_dates:
        raise ValueError("No available dates loaded.")

    end_idx = len(available_dates) - 1
    start_idx = max(0, end_idx - window_days)
    idx = random.randint(start_idx, end_idx)
    return available_dates[idx]

def validate_sql(con: duckdb.DuckDBPyConnection, sql: str) -> bool:
    """
    Validate that the SQL is executable and returns at least one row.
    """
    try:
        res = con.execute(sql)
        rows = res.fetchmany(1)
        if not rows:
            return False
        return True
    except Exception:
        return False

def synth_latest_price(ticker: str, as_of_date: str) -> Dict:
    """
    Latest price for a ticker as of a given date (or the latest available).
    """
    question = f"What was the latest closing price of {ticker} as of {as_of_date}?"
    sql = f"""
        SELECT ticker, date, close
        FROM core.prices_daily
        WHERE ticker = '{ticker}'
          AND date <= '{as_of_date}'
        ORDER BY date DESC
        LIMIT 1;
    """
    return {
        "question": question,
        "sql": sql.strip(),
        "intents": ["latest_price"],
        "tables": ["core.prices_daily"],
    }


def synth_daily_return(ticker: str, day: str) -> Dict:
    """
    Daily return for a ticker on a specific date.
    """
    question = f"What was the daily return of {ticker} on {day}?"
    sql = f"""
        SELECT
            ticker,
            date,
            close / LAG(close) OVER (PARTITION BY ticker ORDER BY date) - 1 AS daily_return
        FROM core.prices_daily
        WHERE ticker = '{ticker}'
          AND date <= '{day}'
        QUALIFY date = '{day}';
    """
    return {
        "question": question,
        "sql": sql.strip(),
        "intents": ["daily_return"],
        "tables": ["core.prices_daily"],
    }


def synth_range_return(ticker: str, d1: str, d2: str) -> Dict:
    """
    Return between two dates for a ticker.
    """
    question = f"What was the return of {ticker} between {d1} and {d2}?"
    sql = f"""
        WITH prices AS (
            SELECT
                ticker,
                date,
                close,
                FIRST_VALUE(close) OVER (PARTITION BY ticker ORDER BY date) AS first_close,
                LAST_VALUE(close) OVER (PARTITION BY ticker ORDER BY date
                    RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS last_close
            FROM core.prices_daily
            WHERE ticker = '{ticker}'
              AND date BETWEEN '{d1}' AND '{d2}'
        )
        SELECT
            ticker,
            MIN(date) AS start_date,
            MAX(date) AS end_date,
            (MAX(close) / MIN(close)) - 1 AS period_return
        FROM prices
        GROUP BY ticker;
    """
    return {
        "question": question,
        "sql": sql.strip(),
        "intents": ["range_return"],
        "tables": ["core.prices_daily"],
    }


def synth_ytd_return(ticker: str, year: int) -> Dict:
    """
    Year-to-date return for a ticker.
    """
    question = f"What is the year-to-date return of {ticker} for {year}?"
    sql = f"""
        WITH prices AS (
            SELECT
                ticker,
                date,
                close,
                FIRST_VALUE(close) OVER (
                    PARTITION BY ticker
                    ORDER BY date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS first_close_ytd
            FROM core.prices_daily
            WHERE ticker = '{ticker}'
              AND EXTRACT(year FROM date) = {year}
        )
        SELECT
            ticker,
            MAX(date) AS as_of_date,
            MAX(close / first_close_ytd - 1) AS ytd_return
        FROM prices
        GROUP BY ticker;
    """
    return {
        "question": question,
        "sql": sql.strip(),
        "intents": ["ytd_return"],
        "tables": ["core.prices_daily"],
    }


def synth_top_k_gainers(day: str, k: int) -> Dict:
    """
    Top K gainers by daily return on a given date.
    """
    question = f"Top {k} gainers by daily return on {day}."
    sql = f"""
        WITH returns AS (
            SELECT
                ticker,
                date,
                close / LAG(close) OVER (PARTITION BY ticker ORDER BY date) - 1 AS daily_return
            FROM core.prices_daily
            WHERE date <= '{day}'
        )
        SELECT ticker, date, daily_return
        FROM returns
        WHERE date = '{day}'
        ORDER BY daily_return DESC
        LIMIT {k};
    """
    return {
        "question": question,
        "sql": sql.strip(),
        "intents": ["top_k_gainers"],
        "tables": ["core.prices_daily"],
    }


def synth_sector_aggregate(sector: str, d1: str, d2: str) -> Dict:
    """
    Average sector return between two dates.
    """
    question = f"What was the average return for the {sector} sector between {d1} and {d2}?"
    sql = f"""
        WITH sector_prices AS (
            SELECT
                p.ticker,
                p.date,
                p.close
            FROM core.prices_daily p
            JOIN core.catalog c
              ON p.ticker = c.ticker
            WHERE c.sector = '{sector}'
              AND p.date BETWEEN '{d1}' AND '{d2}'
        ),
        returns AS (
            SELECT
                ticker,
                date,
                close / LAG(close) OVER (PARTITION BY ticker ORDER BY date) - 1 AS daily_return
            FROM sector_prices
        )
        SELECT
            '{sector}' AS sector,
            AVG(daily_return) AS avg_daily_return
        FROM returns
        WHERE daily_return IS NOT NULL;
    """
    return {
        "question": question,
        "sql": sql.strip(),
        "intents": ["sector_aggregate"],
        "tables": ["core.prices_daily", "core.catalog"],
    }


def synth_dividends_history(ticker: str, d1: str, d2: str) -> Dict:
    """
    Dividend history for a ticker between two dates.
    """
    question = f"Show the dividend history for {ticker} between {d1} and {d2}."
    sql = f"""
        SELECT
            ticker,
            ex_date,
            amount
        FROM core.dividends
        WHERE ticker = '{ticker}'
          AND ex_date BETWEEN '{d1}' AND '{d2}'
        ORDER BY ex_date;
    """
    return {
        "question": question,
        "sql": sql.strip(),
        "intents": ["dividends_history"],
        "tables": ["core.dividends"],
    }


def synth_splits_history(ticker: str, d1: str, d2: str) -> Dict:
    """
    Split history for a ticker between two dates.
    """
    question = f"Show the split history for {ticker} between {d1} and {d2}."
    sql = f"""
        SELECT
            ticker,
            date,
            ratio
        FROM core.splits
        WHERE ticker = '{ticker}'
          AND date BETWEEN '{d1}' AND '{d2}'
        ORDER BY date;
    """
    return {
        "question": question,
        "sql": sql.strip(),
        "intents": ["splits_history"],
        "tables": ["core.splits"],
    }


def synth_portfolio_value_latest(as_of_date: str) -> Dict:
    """
    Total portfolio value as of a given date using the synthetic portfolio.
    """
    question = f"What is the total value of my portfolio as of {as_of_date}?"
    sql = f"""
        WITH latest_prices AS (
            SELECT
                ticker,
                close AS price
            FROM core.prices_daily
            WHERE date = (
                SELECT MAX(date)
                FROM core.prices_daily
                WHERE date <= '{as_of_date}'
            )
        )
        SELECT
            SUM(p.shares * lp.price) AS portfolio_value
        FROM portfolio p
        JOIN latest_prices lp
          ON p.ticker = lp.ticker;
    """
    return {
        "question": question,
        "sql": sql.strip(),
        "intents": ["portfolio_value_latest"],
        "tables": ["portfolio", "core.prices_daily"],
    }


def synth_portfolio_top_k_positions(as_of_date: str, k: int) -> Dict:
    """
    Top K positions in the portfolio by market value as of a given date.
    """
    question = f"Show the top {k} positions in my portfolio by market value as of {as_of_date}."
    sql = f"""
        WITH latest_prices AS (
            SELECT
                ticker,
                close AS price
            FROM core.prices_daily
            WHERE date = (
                SELECT MAX(date)
                FROM core.prices_daily
                WHERE date <= '{as_of_date}'
            )
        ),
        positions AS (
            SELECT
                p.ticker,
                p.shares,
                lp.price,
                p.shares * lp.price AS position_value
            FROM portfolio p
            JOIN latest_prices lp
              ON p.ticker = lp.ticker
        )
        SELECT *
        FROM positions
        ORDER BY position_value DESC
        LIMIT {k};
    """
    return {
        "question": question,
        "sql": sql.strip(),
        "intents": ["portfolio_top_k"],
        "tables": ["portfolio", "core.prices_daily"],
    }


def synth_portfolio_sector_breakdown(as_of_date: str) -> Dict:
    """
    Portfolio sector breakdown by market value as of a given date.
    """
    question = f"What is my portfolio sector allocation by value as of {as_of_date}?"
    sql = f"""
        WITH latest_prices AS (
            SELECT
                ticker,
                close AS price
            FROM core.prices_daily
            WHERE date = (
                SELECT MAX(date)
                FROM core.prices_daily
                WHERE date <= '{as_of_date}'
            )
        ),
        positions AS (
            SELECT
                p.ticker,
                p.shares,
                lp.price,
                p.shares * lp.price AS position_value
            FROM portfolio p
            JOIN latest_prices lp
              ON p.ticker = lp.ticker
        )
        SELECT
            c.sector,
            SUM(position_value) AS sector_value
        FROM positions pos
        JOIN core.catalog c
          ON pos.ticker = c.ticker
        GROUP BY c.sector
        ORDER BY sector_value DESC;
    """
    return {
        "question": question,
        "sql": sql.strip(),
        "intents": ["portfolio_sector_breakdown"],
        "tables": ["portfolio", "core.prices_daily", "core.catalog"],
    }


def synth_multi_intent_latest_price_and_ytd(ticker: str, year: int) -> Dict:
    """
    Multi-intent: latest price + YTD return for a ticker.
    """
    question = f"What is the latest price and the year-to-date return of {ticker} for {year}?"
    sql = f"""
        WITH ytd AS (
            SELECT
                ticker,
                date,
                close,
                FIRST_VALUE(close) OVER (
                    PARTITION BY ticker
                    ORDER BY date
                ) AS first_close
            FROM core.prices_daily
            WHERE ticker = '{ticker}'
              AND EXTRACT(year FROM date) = {year}
        ),
        ytd_agg AS (
            SELECT
                ticker,
                MAX(date) AS as_of_date,
                MAX(close / first_close - 1) AS ytd_return
            FROM ytd
            GROUP BY ticker
        ),
        latest_price AS (
            SELECT
                ticker,
                date AS latest_date,
                close AS latest_price
            FROM core.prices_daily
            WHERE ticker = '{ticker}'
            ORDER BY date DESC
            LIMIT 1
        )
        SELECT
            lp.ticker,
            lp.latest_date,
            lp.latest_price,
            ya.ytd_return
        FROM latest_price lp
        LEFT JOIN ytd_agg ya
          ON lp.ticker = ya.ticker;
    """
    return {
        "question": question,
        "sql": sql.strip(),
        "intents": ["latest_price", "ytd_return", "multi_intent"],
        "tables": ["core.prices_daily"],
    }

def generate_synthetic_example(
    catalog_df: pd.DataFrame,
    catalog_maps: Dict[str, Dict],
    available_dates: List[str],
) -> Dict:
    """
    Generate a single synthetic example by randomly choosing an intent.
    """
    # All base intents and their generator functions
    intents = [
        "latest_price",
        "daily_return",
        "range_return",
        "ytd_return",
        "top_k_gainers",
        "sector_aggregate",
        "dividends_history",
        "splits_history",
        "portfolio_value_latest",
        "portfolio_top_k",
        "portfolio_sector_breakdown",
        "multi_intent_latest_price_and_ytd",
    ]

    intent = random.choice(intents)

    if intent == "latest_price":
        ticker = random_ticker(catalog_df)
        as_of = random_recent_date(available_dates)
        return synth_latest_price(ticker, as_of)

    if intent == "daily_return":
        ticker = random_ticker(catalog_df)
        day = random_recent_date(available_dates)
        return synth_daily_return(ticker, day)

    if intent == "range_return":
        ticker = random_ticker(catalog_df)
        d1, d2 = random_date_range(available_dates)
        return synth_range_return(ticker, d1, d2)

    if intent == "ytd_return":
        ticker = random_ticker(catalog_df)
        # Choose a plausible year from the available_dates
        sample_date = random_date(available_dates)
        year = int(sample_date[:4])
        return synth_ytd_return(ticker, year)

    if intent == "top_k_gainers":
        day = random_recent_date(available_dates)
        k = random.choice([5, 10, 20])
        return synth_top_k_gainers(day, k)

    if intent == "sector_aggregate":
        sector = random_sector(catalog_maps)
        d1, d2 = random_date_range(available_dates)
        return synth_sector_aggregate(sector, d1, d2)

    if intent == "dividends_history":
        ticker = random_ticker(catalog_df)
        d1, d2 = random_date_range(available_dates)
        return synth_dividends_history(ticker, d1, d2)

    if intent == "splits_history":
        ticker = random_ticker(catalog_df)
        d1, d2 = random_date_range(available_dates)
        return synth_splits_history(ticker, d1, d2)

    if intent == "portfolio_value_latest":
        as_of = random_recent_date(available_dates)
        return synth_portfolio_value_latest(as_of)

    if intent == "portfolio_top_k":
        as_of = random_recent_date(available_dates)
        k = random.choice([3, 5, 10])
        return synth_portfolio_top_k_positions(as_of, k)

    if intent == "portfolio_sector_breakdown":
        as_of = random_recent_date(available_dates)
        return synth_portfolio_sector_breakdown(as_of)

    if intent == "multi_intent_latest_price_and_ytd":
        ticker = random_ticker(catalog_df)
        sample_date = random_date(available_dates)
        year = int(sample_date[:4])
        return synth_multi_intent_latest_price_and_ytd(ticker, year)

    return synth_latest_price(random_ticker(catalog_df), random_recent_date(available_dates))


# LLM-based example generation via Ollama

def build_schema_prompt() -> str:
    """
    Build a schema description prompt for the LLM.
    """
    schema = """
    You are generating SQL queries for DuckDB over the following schema:

    Table core.prices_daily:
        - ticker TEXT
        - date   DATE
        - open   DOUBLE
        - high   DOUBLE
        - low    DOUBLE
        - close  DOUBLE
        - volume BIGINT

    Table core.dividends:
        - ticker  TEXT
        - ex_date DATE
        - amount  DOUBLE

    Table core.splits:
        - ticker TEXT
        - date   DATE
        - ratio  DOUBLE

    Table core.catalog:
        - ticker  TEXT
        - name    TEXT
        - sector  TEXT
        - industry TEXT

    Table portfolio:
        - ticker   TEXT
        - shares   DOUBLE
        - avg_cost DOUBLE

    Rules:
    - Use only SELECT queries (no INSERT, UPDATE, DELETE, CREATE, DROP).
    - Use DuckDB-compatible SQL.
    - Prefer window functions and CTEs when useful.
    - When referencing the portfolio, assume it contains the user's holdings.
    """
    return textwrap.dedent(schema).strip()


def call_ollama_for_example(task_hint: str) -> Optional[Dict]:
    """
    Call the Ollama model to generate a question + SQL pair.

    The model is instructed to return a small JSON object:

        {
          "question": "...",
          "sql": "...",
          "intents": [...],
          "tables": [...]
        }

    If parsing fails, returns None.
    """
    system_prompt = build_schema_prompt()
    user_prompt = f"""
    Generate one realistic user question about:

        {task_hint}

    Return a JSON object with the following keys:

        - "question": the natural language question
        - "sql": a single DuckDB-compatible SELECT statement
        - "intents": a short list of high-level intents, e.g. ["latest_price", "ytd_return"]
        - "tables": the list of tables used, e.g. ["core.prices_daily", "portfolio"]

    The response MUST be valid JSON, with no additional commentary.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": textwrap.dedent(user_prompt).strip()},
    ]

    try:
        res = ollama.chat(model=OLLAMA_MODEL, messages=messages)
        content = res["message"]["content"].strip()
    except Exception:
        return None

    # Try to parse JSON from the content
    try:
        # Sometimes models wrap JSON in markdown fences; remove them if present.
        if content.startswith("```"):
            content = content.strip("`")
            # Remove possible "json" or "JSON"
            if content.lower().startswith("json"):
                content = content[4:].strip()

        data = json.loads(content)
        if "question" not in data or "sql" not in data:
            return None
        # Ensure required keys
        data.setdefault("intents", [])
        data.setdefault("tables", [])
        return data
    except Exception:
        return None


def generate_llm_example(
    catalog_df: pd.DataFrame,
    available_dates: List[str],
) -> Optional[Dict]:
    """
    Generate an example via LLM using random hints derived from the catalog
    and date range.

    If generation or parsing fails, returns None.
    """
    # Build a simple task hint with random parameters
    ticker = random_ticker(catalog_df)
    d1, d2 = random_date_range(available_dates)
    year = int(random_date(available_dates)[:4])
    k = random.choice([3, 5, 10])

    task_templates = [
        f"daily performance of {ticker} between {d1} and {d2}",
        f"top {k} gainers or losers around {d2}",
        f"YTD performance for {ticker} in {year}",
        f"portfolio value and sector breakdown using the portfolio table",
        f"comparison between multiple tickers like {ticker} over a time range",
        f"multi-intent query combining latest price and YTD return for tickers like {ticker}",
        f"dividend history of {ticker} between {d1} and {d2}",
    ]
    task_hint = random.choice(task_templates)

    data = call_ollama_for_example(task_hint)
    return data

def ensure_output_dirs() -> None:
    """
    Ensure the output directory structure exists.
    """
    Path(OUT_DIR_TRAIN).mkdir(parents=True, exist_ok=True)
    Path(OUT_DIR_VAL).mkdir(parents=True, exist_ok=True)
    Path(OUT_DIR_TEST).mkdir(parents=True, exist_ok=True)


def write_jsonl(path: str, rows: List[Dict]) -> None:
    """
    Write a list of dicts to path in JSONL format.
    """
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_train_shard(
    shard_idx: int,
    con: duckdb.DuckDBPyConnection,
    catalog_df: pd.DataFrame,
    catalog_maps: Dict[str, Dict],
    available_dates: List[str],
) -> None:
    shard_path = os.path.join(OUT_DIR_TRAIN, f"train_{shard_idx:02d}.jsonl")

    target = TRAIN_SHARD_SIZE
    llm_target = int(target * TRAIN_LLM_FRACTION)
    synth_target = target - llm_target

    rows: List[Dict] = []
    n_synth = 0
    n_llm = 0
    llm_fail_streak = 0
    max_llm_fail_streak = 15

    pbar = tqdm(total=target, desc=f"train shard {shard_idx:02d}", leave=False)

    while len(rows) < target:
        use_llm = (n_llm < llm_target) and (llm_fail_streak < max_llm_fail_streak)

        if use_llm:
            example = generate_llm_example(catalog_df, available_dates)
            if example is None or not validate_sql(con, example["sql"]):
                llm_fail_streak += 1
                continue
            llm_fail_streak = 0
            n_llm += 1
        else:
            example = generate_synthetic_example(catalog_df, catalog_maps, available_dates)
            if not validate_sql(con, example["sql"]):
                continue
            n_synth += 1

        example_id = f"train_{shard_idx:02d}_{len(rows):06d}"
        row = {
            "id": example_id,
            "split": "train",
            "question": example["question"],
            "sql": example["sql"],
            "intents": example.get("intents", []),
            "tables": example.get("tables", []),
        }
        rows.append(row)
        pbar.update(1)

    pbar.close()
    write_jsonl(shard_path, rows)


def generate_eval_split(
    split_name: str,
    target: int,
    con: duckdb.DuckDBPyConnection,
    catalog_df: pd.DataFrame,
    catalog_maps: Dict[str, Dict],
    available_dates: List[str],
) -> None:
    """
    Generate a synthetic-only split (val or test).
    These are kept purely synthetic for reproducibility and stability.
    """
    if split_name == "val":
        out_path = os.path.join(OUT_DIR_VAL, "val.jsonl")
    elif split_name == "test":
        out_path = os.path.join(OUT_DIR_TEST, "test.jsonl")
    else:
        raise ValueError(f"Unknown split: {split_name}")

    rows: List[Dict] = []

    pbar = tqdm(total=target, desc=f"{split_name} split", leave=False)

    while len(rows) < target:
        example = generate_synthetic_example(catalog_df, catalog_maps, available_dates)
        if not validate_sql(con, example["sql"]):
            continue

        example_id = f"{split_name}_{len(rows):06d}"
        row = {
            "id": example_id,
            "split": split_name,
            "question": example["question"],
            "sql": example["sql"],
            "intents": example.get("intents", []),
            "tables": example.get("tables", []),
        }
        rows.append(row)
        pbar.update(1)

    pbar.close()
    write_jsonl(out_path, rows)

def build_metadata(
    train_count: int,
    val_count: int,
    test_count: int,
) -> Dict:
    """
    Build a simple metadata dictionary describing the dataset.
    """
    meta = {
        "total_examples": train_count + val_count + test_count,
        "splits": {
            "train": train_count,
            "val": val_count,
            "test": test_count,
        },
        "schema": {
            "core.prices_daily": {
                "columns": [
                    "ticker",
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]
            },
            "core.dividends": {
                "columns": [
                    "ticker",
                    "ex_date",
                    "amount",
                ]
            },
            "core.splits": {
                "columns": [
                    "ticker",
                    "date",
                    "ratio",
                ]
            },
            "core.catalog": {
                "columns": [
                    "ticker",
                    "name",
                    "sector",
                    "industry",
                ]
            },
            "portfolio": {
                "columns": [
                    "ticker",
                    "shares",
                    "avg_cost",
                ],
                "description": "Dynamic user portfolio table, synthetic during training, populated at runtime in production.",
            },
        },
        "generation": {
            "train_llm_fraction": TRAIN_LLM_FRACTION,
            "train_synthetic_fraction": TRAIN_SYNTH_FRACTION,
            "random_seed": RANDOM_SEED,
            "num_train_shards": NUM_TRAIN_SHARDS,
            "train_shard_size": TRAIN_SHARD_SIZE,
            "ollama_model": OLLAMA_MODEL,
        },
    }
    return meta


def write_metadata(meta: Dict) -> None:
    """
    Write dataset metadata to META_DATA path.
    """
    with open(META_DATA, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def main() -> None:
    """
    Entry point for dataset generation.

    This assumes the script is executed from the project root:
        USC-CSCI544-FINAL_PROJECT/
    """
    ensure_output_dirs()

    # Open DuckDB connection
    con = open_connection(DB_DATA)

    # Load base data
    available_dates = load_available_dates(con)
    if not available_dates:
        raise RuntimeError("No available dates in core.prices_daily. Check your DB.")

    catalog_df = load_sp500_catalog(SP500_CATALOG_PATH)
    catalog_maps = build_catalog_maps(catalog_df)

    # Build and register synthetic portfolio
    portfolio_df = build_synthetic_portfolio(catalog_df, size=30)
    register_portfolio_table(con, catalog_df)

    # Train shards
    print("Generating train shards...")
    for shard_idx in range(NUM_TRAIN_SHARDS):
        generate_train_shard(shard_idx, con, catalog_df, catalog_maps, available_dates)

    # Validation and test splits
    print("Generating validation split...")
    generate_eval_split("val", VAL_TARGET, con, catalog_df, catalog_maps, available_dates)

    print("Generating test split...")
    generate_eval_split("test", TEST_TARGET, con, catalog_df, catalog_maps, available_dates)

    # Build and write metadata
    meta = build_metadata(TRAIN_TARGET, VAL_TARGET, TEST_TARGET)
    write_metadata(meta)

    print("Done.")
    print(f"Train examples: {TRAIN_TARGET}")
    print(f"Val examples:   {VAL_TARGET}")
    print(f"Test examples:  {TEST_TARGET}")


if __name__ == "__main__":
    main()