import pandas as pd
def normalize_portfolio_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {
        "symbol": "ticker",
        "asset": "ticker",
        "stock": "ticker",
        "ticker": "ticker",

        "shares": "shares",
        "quantity": "shares",
        "qty": "shares",
        "position": "shares",
        "position_size": "shares",

        "avgcost": "avg_cost",
        "average_cost": "avg_cost",
        "costbasis": "avg_cost",
        "averageprice": "avg_cost",
        "avg_price": "avg_cost",
    }

    df_norm = df.copy()
    df_norm.columns = df_norm.columns.str.lower().str.replace(" ", "_")

    new_cols = {}
    for col in df_norm.columns:
        if col in col_map:
            new_cols[col] = col_map[col]

    df_norm = df_norm.rename(columns=new_cols)

    required = ["ticker", "shares", "avg_cost"]
    for col in required:
        if col not in df_norm.columns:
            raise ValueError(f"Missing required column: {col}")

    return df_norm[required]