import json
import sys
from pathlib import Path
import duckdb
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Resolve important paths relative to this file so it works both from CLI and backend imports.
_THIS_FILE = Path(__file__).resolve()
_TEXT_TO_SQL_ROOT = _THIS_FILE.parents[1]  
_SRC_ROOT = _THIS_FILE.parents[2]        

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
LORA_DIR = _TEXT_TO_SQL_ROOT / "model" / "QWEN" / "Qwen"
DB_PATH = str(_SRC_ROOT / "db" / "financial.duckdb")

if str(_TEXT_TO_SQL_ROOT) not in sys.path:
    sys.path.append(str(_TEXT_TO_SQL_ROOT))

from semantic.ir_to_sql import ir_to_sql  # noqa: E402

def lookup_company_name(ticker: str) -> str:
    ticker = str(ticker).upper()

    try:
        con = duckdb.connect(DB_PATH, read_only=True)

        row = con.execute(
            "SELECT longname FROM ext.catalog WHERE ticker = ? LIMIT 1",
            [ticker],
        ).fetchone()
        if row and row[0]:
            return row[0]

        row = con.execute(
            "SELECT name FROM core.tickers WHERE ticker = ? LIMIT 1",
            [ticker],
        ).fetchone()
        if row and row[0]:
            return row[0]

    except Exception:
        pass
    finally:
        try:
            con.close()
        except Exception:
            pass

    return ticker

METRIC_NAMES = {
    "ytd_return": "year-to-date return",
    "daily_return": "daily return",
    "close": "closing price",
    "dividend_yield_ttm": "dividend yield (TTM)",
}

def format_metric_name(col: str) -> str:
    return METRIC_NAMES.get(col, col.replace("_", " "))


def format_value(val, metric_name="", metric_type=""):
    """Reglas:
    - Return metrics → porcentaje
    - Dividend yield → porcentaje
    - Todo lo demás → número normal
    """
    if not isinstance(val, (int, float)):
        return str(val)

    name = metric_name.lower()

    if metric_type == "return" or "return" in name:
        return f"{val * 100:.2f}%"

    if "yield" in name:
        return f"{val * 100:.2f}%"

    return f"{val:.4f}"

def build_prompt(question: str) -> str:
    return f"Question: {question}\nIR:\n"


def extract_json_block(text: str):
    start = None
    depth = 0

    for i, ch in enumerate(text):
        if ch == "{":
            if start is None:
                start = i
            depth += 1

        elif ch == "}":
            if start is not None:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i+1])
                    except json.JSONDecodeError:
                        return None
    return None

def natural_answer(ir: dict, df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "There is no data available for this query."

    metric_name = (ir.get("metric") or {}).get("name", "")
    metric_type = (ir.get("metric") or {}).get("type", "")

    metric_cols = [c for c in df.columns if c not in ("ticker", "date", "as_of_date")]

    if not metric_cols:
        return f"Here are the results:\n{df.to_string(index=False)}"

    metric_col = metric_name if metric_name in df.columns else metric_cols[0]
    pretty = format_metric_name(metric_col)

    if "ticker" in df.columns and len(df) == 1:
        row = df.iloc[0]
        company = lookup_company_name(row["ticker"])
        value = row[metric_col]

        return (
            f"The {pretty} for {company} is "
            f"{format_value(value, metric_col, metric_type)}."
        )

    if "ticker" in df.columns:
        lines = []
        for _, row in df.iterrows():
            company = lookup_company_name(row["ticker"])
            value = row[metric_col]
            lines.append(
                f"- {company}: {format_value(value, metric_col, metric_type)}"
            )
        return f"The {pretty} for your portfolio are:\n" + "\n".join(lines)

    lines = [
        f"- {format_value(row[metric_col], metric_col, metric_type)}"
        for _, row in df.iterrows()
    ]
    return f"The {pretty} values are:\n" + "\n".join(lines)

_model = None
_tokenizer = None
_device = None

def load_model():
    global _model, _tokenizer, _device

    if _model is not None:
        return _model, _tokenizer, _device

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    _device = device

    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
    _tokenizer.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        dtype=torch.float16 if device != "cpu" else torch.float32,
    ).to(device)

    _model = PeftModel.from_pretrained(base, str(LORA_DIR))
    _model.to(device)
    _model.eval()

    return _model, _tokenizer, _device

def generate_ir(model, tokenizer, device, question: str):
    prompt = build_prompt(question)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)

    with torch.no_grad():
        out_ids = model.generate(
            **enc,
            max_new_tokens=256,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(out_ids[0], skip_special_tokens=True)

    tail = decoded[len(prompt):] if decoded.startswith(prompt) else decoded
    tail = tail.strip()

    ir = extract_json_block(tail) or {}

    return ir

def run_financial_query(question: str) -> dict:
    model, tokenizer, device = load_model()

    ir = generate_ir(model, tokenizer, device, question)
    sql = ir_to_sql(ir)

    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        df = con.execute(sql).fetchdf()
    finally:
        con.close()

    answer = natural_answer(ir, df)

    return {
        "ir": ir,
        "sql": sql,
        "result": df,
        "natural_answer": answer,
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline_api.py \"your question here\"")
        sys.exit(1)

    question = sys.argv[1]
    out = run_financial_query(question)

    print("\nSQL:\n", out["sql"])
    print("\nRESULT:\n", out["result"])
    print("\nANSWER:\n", out["natural_answer"])