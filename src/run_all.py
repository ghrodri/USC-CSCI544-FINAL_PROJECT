import subprocess
from pathlib import Path
import sys
import time
import subprocess
from pathlib import Path
import sys
import time

# Base path
ROOT = Path("")

ETL_SCRIPTS = [
    ROOT / "etl" / "build_schema.py",
    ROOT / "etl" / "catalog.py",
    ROOT / "etl" / "dividends.py",
    ROOT / "etl" / "price_daily.py",
    ROOT / "etl" / "splits.py",
    ROOT / "etl" / "tickers_ref.py",
    ROOT / "etl" / "trading_calendar.py",
]
# Data gen
DATA_GEN = ROOT / "text-to-sql" / "code" / "data_gen.py"

def run_script(script_path: Path):
    start = time.time()
    result = subprocess.run([sys.executable, script_path.as_posix()],
                            capture_output=True, text=True)
    duration = time.time() - start

    if result.returncode != 0:
        sys.stderr.write(f"Error in {script_path.name}\n")
        sys.stderr.write(result.stderr + "\n")
        sys.exit(1)

    log_file = script_path.with_suffix(".log")
    with log_file.open("w", encoding="utf-8") as f:
        f.write(result.stdout)
        f.write(result.stderr)
        f.write(f"\nDuration: {duration:.2f}s\n")
if __name__ == "__main__":
    for script in ETL_SCRIPTS:
        run_script(script)
    run_script(DATA_GEN)
    print("\n Pipeline completed successfully")