import os
import sys
import time
import subprocess
from pathlib import Path
ROOT_DATA = Path(".").resolve()
ETL_SCRIPTS_DATA = [
    ROOT_DATA / "etl" / "build_schema.py",
    ROOT_DATA / "etl" / "catalog.py",
    ROOT_DATA / "etl" / "dividends.py",
    ROOT_DATA / "etl" / "price_daily.py",
    ROOT_DATA / "etl" / "splits.py",
    ROOT_DATA / "etl" / "tickers_ref.py",
    ROOT_DATA / "etl" / "trading_calendar.py",
]
DATA_GEN_DATA = ROOT_DATA / "text-to-sql" / "code" / "data_gen.py"
LOGS_DIR_DATA = ROOT_DATA / "src" / "logs" / "etl"

def ensure_dir(p: Path):
    os.makedirs(p, exist_ok=True)

def run_script(p: Path) -> float:
    t0 = time.time()
    r = subprocess.run([sys.executable, p.as_posix()], capture_output=True, text=True)
    dt = time.time() - t0

    log_fp = LOGS_DIR_DATA / f"{p.stem}.log"
    with log_fp.open("w", encoding="utf-8") as f:
        f.write(r.stdout or "")
        f.write(r.stderr or "")
        f.write(f"\nDuration: {dt:.2f}s\n")

    if r.returncode != 0:
        print(f"FAILED: {p.name} (code {r.returncode})")
        print(f"See log: {log_fp.as_posix()}")
        sys.exit(r.returncode)

    print(f"OK: {p.name}  {dt:.2f}s")
    return dt


if __name__ == "__main__":
    ensure_dir(LOGS_DIR_DATA)

    t_all = time.time()
    for s in ETL_SCRIPTS_DATA:
        run_script(s)

    run_script(DATA_GEN_DATA)
    print(f"\nPipeline completed successfully in {time.time() - t_all:.2f}s")