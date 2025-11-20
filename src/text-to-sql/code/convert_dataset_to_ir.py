import json
import glob
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm
import multiprocessing as mp
import yaml
from convert_sql_to_ir import convert_sql_to_ir

INPUT_DATA_DIR = Path("./src/text-to-sql/data_new")
OUTPUT_IR_DIR = Path("./src/text-to-sql/ir-data")
SEMANTIC_DIR = Path("./src/text-to-sql/semantic")

IR_SCHEMA_PATH = SEMANTIC_DIR / "ir_schema.json"
SEMANTIC_LAYER_PATH = SEMANTIC_DIR / "semantic_layer.yaml"


def load_ir_schema() -> Dict[str, Any]:
    with IR_SCHEMA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_semantic_layer() -> Dict[str, Any]:
    with SEMANTIC_LAYER_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


IR_SCHEMA = load_ir_schema()
SEMANTIC = load_semantic_layer()


def process_jsonl_file(path_in: Path, path_out: Path):
    total_lines = sum(1 for _ in path_in.open("r", encoding="utf-8"))

    with path_in.open("r", encoding="utf-8") as f_in, \
         path_out.open("w", encoding="utf-8") as f_out:

        for line in tqdm(f_in, total=total_lines, desc=f"Processing {path_in.name}"):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            question = obj.get("question", "")
            sql = obj.get("sql", "")

            ir_obj = convert_sql_to_ir(question, sql)

            out = {
                "question": question,
                "ir": ir_obj
            }

            f_out.write(json.dumps(out) + "\n")


def process_file_worker(args):
    path_in, path_out = args
    process_jsonl_file(path_in, path_out)
    return str(path_out)


def auto_workers(num_tasks: int) -> int:
    try:
        cpu = mp.cpu_count()
    except NotImplementedError:
        cpu = 1
    if cpu <= 1 or num_tasks <= 1:
        return 1
    return max(1, min(cpu - 1, num_tasks))


def convert_split(split_name: str):
    in_dir = INPUT_DATA_DIR / split_name
    out_dir = OUTPUT_IR_DIR / split_name

    if not in_dir.exists():
        print(f"Input directory not found: {in_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(glob.glob(str(in_dir / "*.jsonl")))

    if not input_files:
        print(f"No JSONL files found in {in_dir}")
        return

    print(f"Found {len(input_files)} shard(s) in {in_dir}")

    tasks = []
    for in_file in input_files:
        p_in = Path(in_file)
        p_out = out_dir / (p_in.stem + "_ir.jsonl")
        tasks.append((p_in, p_out))

    workers = auto_workers(len(tasks))
    print(f"Using {workers} worker(s) for split '{split_name}'")

    if workers > 1:
        mp.set_start_method("spawn", force=True)
        with mp.Pool(processes=workers) as pool:
            list(
                tqdm(
                    pool.imap(process_file_worker, tasks),
                    total=len(tasks),
                    desc=f"Converting split {split_name} (parallel)"
                )
            )
    else:
        for t in tqdm(tasks, desc=f"Converting split {split_name} (single process)"):
            process_file_worker(t)

    print(f"Completed IR generation for split '{split_name}'")

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        convert_split(split)
