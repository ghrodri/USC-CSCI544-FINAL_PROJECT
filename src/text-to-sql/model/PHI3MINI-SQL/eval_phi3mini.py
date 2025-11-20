import os, json, random, sqlparse, torch, duckdb
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_DIR = os.path.abspath(".")
TRAIN_DIR = os.path.join(BASE_DIR, "src/text-to-sql/data_new/train")
TEST_DIR = os.path.join(BASE_DIR, "src/text-to-sql/data_new/test")
MODEL_DIR = os.path.join(BASE_DIR, "src/text-to-sql/model/PHI3MINI/out")
DB_PATH = os.path.join(BASE_DIR, "src/db/financial.duckdb")
REPORT_DIR = os.path.join(BASE_DIR, "src/evl/PHI3MINI/reports")
os.makedirs(REPORT_DIR, exist_ok=True)

TRAIN_SUBSAMPLE = 1000
TEST_SUBSAMPLE = 500
random.seed(42)

if torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE = torch.bfloat16
else:
    DEVICE = "cpu"
    DTYPE = torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    dtype=DTYPE,
    attn_implementation="sdpa"
).to(DEVICE)

model = PeftModel.from_pretrained(base, MODEL_DIR).to(DEVICE)
model.eval()

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)

def load_jsonl(path, limit):
    files = sorted([x for x in os.listdir(path) if x.endswith(".jsonl")])
    rows = []
    for fn in files:
        with open(os.path.join(path, fn), "r") as f:
            for line in f:
                rows.append(json.loads(line))
    random.shuffle(rows)
    return rows[:limit]

def generate_sql(q):
    prompt = f"### Question:\n{q}\n\n### SQL:\n"
    inp = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    out = model.generate(**inp, max_new_tokens=120, do_sample=False)
    dec = tokenizer.decode(out[0], skip_special_tokens=True)
    return dec.split("### SQL:")[-1].strip()

def exact_match(a, b):
    return int(a.strip().lower() == b.strip().lower())

def normalize(sql):
    return sqlparse.format(sql, keyword_case="upper", identifier_case="lower", strip_comments=True).strip()

def structural_match(a, b):
    return int(normalize(a) == normalize(b))

def semantic_sim(a, b):
    e1 = embedder.encode(a, convert_to_tensor=True)
    e2 = embedder.encode(b, convert_to_tensor=True)
    return float(util.cos_sim(e1, e2).item())

def exec_query(sql, conn):
    try: return conn.execute(sql).fetchall(), None
    except Exception as e: return None, str(e)

def exec_and_partial(pred, gold, conn):
    g, ge = exec_query(gold, conn)
    p, pe = exec_query(pred, conn)
    if ge or pe: return 0, 0, True, ge, pe
    ex = int(p == g)
    try: pa = int(len(set(g).intersection(set(p))) > 0)
    except: pa = 0
    return ex, pa, False, None, None

def evaluate(rows, conn, split):
    err_path = os.path.join(REPORT_DIR, f"{split}_errors.jsonl")
    met_path = os.path.join(REPORT_DIR, f"{split}_metrics.txt")
    if os.path.exists(err_path): os.remove(err_path)
    EM, SM, EX, PA, SI = [], [], [], [], []
    with open(err_path, "a") as ef:
        for ex in tqdm(rows, desc=f"Evaluating {split}"):
            q = ex["question"]
            gold = ex["sql"]
            pred = generate_sql(q)
            EM.append(exact_match(pred, gold))
            SM.append(structural_match(pred, gold))
            SI.append(semantic_sim(pred, gold))
            exa, pa, err, ge, pe = exec_and_partial(pred, gold, conn)
            EX.append(exa)
            PA.append(pa)
            if err: ef.write(json.dumps({"question": q,"gold": gold,"pred": pred,"gold_error": ge,"pred_error": pe})+"\n")
    n = len(rows)
    with open(met_path, "w") as f:
        f.write(f"Split:{split}\nExamples:{n}\n\nExactMatch:{sum(EM)/n:.4f}\nStructuralMatch:{sum(SM)/n:.4f}\nExecAcc:{sum(EX)/n:.4f}\nPartialExec:{sum(PA)/n:.4f}\nSemanticSim:{sum(SI)/n:.4f}\n")

def main():
    conn = duckdb.connect(DB_PATH)
    train = load_jsonl(TRAIN_DIR, TRAIN_SUBSAMPLE)
    test = load_jsonl(TEST_DIR, TEST_SUBSAMPLE)
    evaluate(train, conn, "train")
    evaluate(test, conn, "test")
    conn.close()

if __name__ == "__main__":
    main()