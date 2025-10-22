import pandas as pd
from pathlib import Path

# path del parquet
calendar_path = Path("/Users/ghrodri/Local/Final_Project/src/data/trading_calendar/calendar_2025-10-20.parquet")

# carga el parquet
df = pd.read_parquet(calendar_path)

# muestra las primeras 10 filas
print(df.head(500).to_string(index=False))