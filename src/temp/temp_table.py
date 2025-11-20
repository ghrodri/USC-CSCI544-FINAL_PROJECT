import os
import sys
import duckdb
import pandas as pd
DB_PATH = "./src/db/financial.duckdb"
DATA_DIR = "./src/data/portfolio"
USER_PORTFOLIO_PARQUET = os.path.join(DATA_DIR, "user_portfolio.parquet")

def load_portfolio_from_csv(csv_path: str) -> pd.DataFrame:
    """
    Load a user portfolio from either a CSV or Parquet file.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Portfolio file not found: {csv_path}")

    _, ext = os.path.splitext(csv_path.lower())
    if ext == ".parquet":
        df = pd.read_parquet(csv_path)
    else:
        df = pd.read_csv(csv_path)

    required_cols = {"ticker", "shares", "avg_cost"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df[["ticker", "shares", "avg_cost"]]


def create_internal_portfolio_table(csv_path: str) -> None:
    os.makedirs("./src/db", exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    df = load_portfolio_from_csv(csv_path)
    df.to_parquet(USER_PORTFOLIO_PARQUET, index=False)
    print(f"Saved user portfolio parquet: {USER_PORTFOLIO_PARQUET}")

    con = duckdb.connect(DB_PATH)
    for stmt in [
        "DROP VIEW IF EXISTS portfolio;",
        "DROP TABLE IF EXISTS portfolio;",
        "DROP MACRO IF EXISTS portfolio;",
        "DROP SEQUENCE IF EXISTS portfolio;",
    ]:
        try:
            con.execute(stmt)
        except duckdb.CatalogException:
            pass

    con.execute(
        f"""
        CREATE TABLE portfolio AS
        SELECT ticker, shares, avg_cost
        FROM read_parquet('{USER_PORTFOLIO_PARQUET}');
        """
    )

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py path_to_csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    create_internal_portfolio_table(csv_path)
    print("\nDONE Internal user portfolio.")