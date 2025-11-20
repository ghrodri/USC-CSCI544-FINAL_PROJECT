import sys
import json
import re
from pathlib import Path

import torch
import duckdb
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-0.5B" 
LORA_DIR = Path("./src/text-to-sql/model/QWEN/Qwen")
DB_PATH = "./src/db/financial.duckdb"

sys.path.append("./src/text-to-sql")
from semantic.ir_to_sql import ir_to_sql

def build_prompt(question: str) -> str:
    return f"Question: {question}\nIR:\n"

def extract_json_block(text: str) -> dict:
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
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        pass

    for m in re.findall(r"\{.*?\}", text, flags=re.DOTALL):
        try:
            return json.loads(m)
        except json.JSONDecodeError:
            continue

    raise ValueError("No valid JSON object found in model output.")


def generate_ir(model, tokenizer, device: str, question: str):
    prompt = build_prompt(question)

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=256,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)

    print("\n[DEBUG] RAW MODEL OUTPUT:")
    print(decoded)

    if decoded.startswith(prompt):
        tail = decoded[len(prompt):]
    else:
        idx = decoded.rfind("IR:")
        tail = decoded[idx + 3 :] if idx != -1 else decoded

    tail = tail.strip()
    ir = extract_json_block(tail)
    return ir, decoded

def run_sql(sql: str):
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        df = con.execute(sql).fetchdf()
    finally:
        con.close()
    return df

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/text-to-sql/code/run_qwen_pipeline.py \"<question>\"")
        sys.exit(1)

    question = sys.argv[1]

    print("[STEP 1] Loading base model + tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        dtype=torch.float16 if device != "cpu" else torch.float32,
    ).to(device)

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Loading LoRA from {LORA_DIR} ...")
    model = PeftModel.from_pretrained(base_model, str(LORA_DIR))
    model.to(device)
    model.eval()

    print("\n[STEP 2] Generating IR...")
    ir, _ = generate_ir(model, tokenizer, device, question)

    print("\n[STEP 3] Parsed IR:")
    print(json.dumps(ir, indent=2))

    print("\n[STEP 4] IR -> SQL...")
    sql = ir_to_sql(ir)
    print("\nGenerated SQL:")
    print(sql)

    print("\n[STEP 5] Executing SQL on DuckDB...")
    try:
        df = run_sql(sql)
        print("\nResult:")
        print(df)
    except Exception as e:
        print("[SQL ERROR]", e)


if __name__ == "__main__":
    main()