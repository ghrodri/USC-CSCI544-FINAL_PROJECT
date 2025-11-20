import re
import json
from typing import Any, Dict, List


TICKER_EQ_RE = re.compile(r"ticker\s*=\s*'([^']+)'", re.IGNORECASE)
TICKER_IN_RE = re.compile(r"ticker\s+IN\s*\(([^)]+)\)", re.IGNORECASE)
YEAR_EXTRACT_RE = re.compile(r"EXTRACT\s*\(\s*year\s+FROM\s+([^)]+)\)\s*=\s*(\d{4})", re.IGNORECASE)
BETWEEN_RE = re.compile(r"date\s+BETWEEN\s+DATE?\s*'([^']+)'\s+AND\s+DATE?\s*'([^']+)'", re.IGNORECASE)
WINDOW_ROWS_RE = re.compile(r"ROWS\s+BETWEEN\s+(\d+)\s+PRECEDING", re.IGNORECASE)
FROM_TABLE_RE = re.compile(r"\bFROM\s+([a-zA-Z0-9_\.]+)", re.IGNORECASE)


def normalize_sql(sql: str) -> str:
    return " ".join(sql.strip().split())


def detect_assets(sql: str) -> List[str]:
    m = TICKER_EQ_RE.search(sql)
    assets: List[str] = []
    if m:
        assets.append(m.group(1))
    m2 = TICKER_IN_RE.search(sql)
    if m2:
        inside = m2.group(1)
        parts = [p.strip() for p in inside.split(",")]
        for p in parts:
            if p.startswith("'") and p.endswith("'"):
                assets.append(p[1:-1])
    return list(dict.fromkeys(assets))


def detect_year(sql: str) -> int | None:
    m = YEAR_EXTRACT_RE.search(sql)
    if not m:
        return None
    year_str = m.group(2)
    try:
        return int(year_str)
    except ValueError:
        return None


def detect_between_dates(sql: str) -> Dict[str, str] | None:
    m = BETWEEN_RE.search(sql)
    if not m:
        return None
    start, end = m.group(1), m.group(2)
    return {"start": start, "end": end}


def detect_window(sql: str) -> int | None:
    m = WINDOW_ROWS_RE.search(sql)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def detect_base_table(sql: str) -> str | None:
    matches = FROM_TABLE_RE.findall(sql)
    if not matches:
        return None
    for t in matches:
        if t.startswith("core.") or t.startswith("views.") or t.startswith("ext.") or t == "portfolio":
            return t
    return matches[0]


def metric_name_from_sql(sql_lower: str) -> str:
    if "ytd_return" in sql_lower or "first_close_ytd" in sql_lower:
        return "ytd_return"
    if "mtd_return" in sql_lower or "first_close_mtd" in sql_lower:
        return "mtd_return"
    if "rolling_vol" in sql_lower or "rolling_volatility" in sql_lower:
        return "rolling_volatility"
    if "rolling_ret" in sql_lower or "rolling_return" in sql_lower:
        return "rolling_return"
    if "dividend_yield_ttm" in sql_lower:
        return "dividend_yield_ttm"
    if "portfolio" in sql_lower and "unrealized_pnl" in sql_lower:
        return "portfolio_value"
    if "views.daily_returns" in sql_lower or "daily_returns" in sql_lower:
        return "daily_return"
    return "close"


def metric_type_from_name(name: str) -> str:
    if name in {"close", "open", "high", "low", "volume", "vwap"}:
        return "price"
    if name in {"daily_return", "ytd_return", "mtd_return", "period_return", "rolling_return"}:
        return "return"
    if name in {"rolling_volatility", "volatility_annualized", "drawdown", "max_drawdown"}:
        return "risk"
    if name in {"dividend_yield_ttm", "dividend_amount", "total_return_price_plus_dividends"}:
        return "corporate_action"
    if name.startswith("portfolio") or name in {"contribution_to_return", "contribution_to_risk"}:
        return "portfolio"
    if name in {"correlation", "correlation_matrix", "covariance"}:
        return "correlation"
    return "descriptive"


def intent_from_metric_and_table(metric_name: str, base_table: str | None) -> str:
    if metric_name == "ytd_return":
        return "get_ytd_return"
    if metric_name == "mtd_return":
        return "get_mtd_return"
    if metric_name in {"rolling_volatility", "rolling_return"}:
        return "rolling_stat"
    if metric_name in {"dividend_yield_ttm", "dividend_amount"}:
        return "get_dividends"
    if metric_name == "portfolio_value":
        return "evaluate_portfolio"
    if metric_name == "daily_return":
        return "get_return"
    if base_table and "corporate_actions" in base_table:
        return "get_corporate_actions"
    if base_table and "prices" in base_table:
        return "get_price"
    return "compute_metric"


def time_from_sql(sql: str) -> Dict[str, Any]:
    year = detect_year(sql)
    between = detect_between_dates(sql)
    if year is not None:
        return {"range": {"type": "year", "value": year}}
    if between is not None:
        return {
            "range": {
                "type": "between_dates",
                "start": between["start"],
                "end": between["end"],
            }
        }
    return {"range": {"type": "all"}}


def metric_arguments_from_sql(metric_name: str, sql: str) -> Dict[str, Any]:
    args: Dict[str, Any] = {}
    year = detect_year(sql)
    between = detect_between_dates(sql)
    window = detect_window(sql)
    if metric_name in {"ytd_return", "mtd_return"} and year is not None:
        args["year"] = year
    if metric_name in {"period_return", "rolling_return", "rolling_volatility", "total_return_price_plus_dividends"}:
        if between is not None:
            args["start_date"] = between["start"]
            args["end_date"] = between["end"]
    if metric_name in {"rolling_return", "rolling_volatility"} and window is not None:
        args["window"] = window
        args["window_unit"] = "trading_days"
    return args


def convert_sql_to_ir(question: str, sql: str) -> Dict[str, Any]:
    sql_norm = normalize_sql(sql)
    sql_lower = sql_norm.lower()
    assets = detect_assets(sql_norm)
    base_table = detect_base_table(sql_norm)
    metric_name = metric_name_from_sql(sql_lower)
    metric_type = metric_type_from_name(metric_name)
    intent = intent_from_metric_and_table(metric_name, base_table)
    metric_args = metric_arguments_from_sql(metric_name, sql_norm)
    time_obj = time_from_sql(sql_norm)
    ir: Dict[str, Any] = {
        "intent": intent,
        "entities": {"assets": assets},
        "metric": {"name": metric_name, "type": metric_type, "arguments": metric_args},
        "time": time_obj,
        "filters": [],
        "output": {"format": "table"},
    }
    return ir


if __name__ == "__main__":
    example_sql = """
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
        WHERE ticker = 'DG'
          AND EXTRACT(year FROM date) = 2024
    )
    SELECT
        ticker,
        MAX(date) AS as_of_date,
        MAX(close / first_close_ytd - 1) AS ytd_return
    FROM prices
    GROUP BY ticker;
    """
    ir = convert_sql_to_ir("What is the year-to-date return of DG for 2024?", example_sql)
    print(json.dumps(ir, indent=2))
