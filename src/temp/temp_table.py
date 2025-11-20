import os
import sys
import duckdb
import pandas as pd
DB_PATH = "./src/db/financial.duckdb"
DATA_DIR = "./src/data/portfolio"
USER_PORTFOLIO_PARQUET = os.path.join(DATA_DIR, "user_portfolio.parquet")

def load_portfolio_from_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

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
    con.execute("DROP VIEW IF EXISTS portfolio;")
    con.execute("DROP TABLE IF EXISTS portfolio;")
    con.execute("DROP MACRO IF EXISTS portfolio;")
    con.execute("DROP SEQUENCE IF EXISTS portfolio;")

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