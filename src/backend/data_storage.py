import os
import hashlib
import argparse
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from datasets import load_dataset
from backend.pdf_utilities import clean_text, get_pdf_text

load_dotenv()
index_path = os.path.join(os.path.dirname(__file__), "faiss_index")
FINANCEBENCH_PDF_DIR = os.path.join(os.path.dirname(__file__), "pdfs_financebench")
SKIPPED_DOCS = {"adobe_2022_10k"}

def get_text_chunks(text, chunk_size=1500, chunk_overlap=500):
    # Use recursive splitter with multiple fallback separators to avoid giant chunks
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    print(f"[INFO] Created {len(chunks)} text chunks")
    return chunks


def get_or_load_vectorstore(text_chunks, path="faiss_index"):
    # Reduce batch size to avoid 300k token/request cap
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", chunk_size=64)
    print("[INFO] Using OpenAI embeddings: text-embedding-3-large (batch size=64)")
    if os.path.exists(path):
        print(f"[INFO] Loading existing vectorstore from '{path}'...")
        print("[WARN] Ensure this index was built with the same embedding model; otherwise run with --rebuild-index or a different --index-path.")
        vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        if text_chunks:
            print("[INFO] Adding new chunks to existing vectorstore...")
            vectorstore.add_texts(text_chunks)
            vectorstore.save_local(path)
            print(f"[INFO] Updated vectorstore saved to '{path}'")
        return vectorstore

    print("[INFO] Creating new vectorstore...")
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    vectorstore.save_local(path)
    print(f"[INFO] Vectorstore saved to '{path}'")
    return vectorstore

# Fallback LLM-only response
def generate_general_response(question, history_messages):
    """LLM-only response (no retrieval), acts as a fallback."""
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    messages = [
        SystemMessage(content="You are a helpful, concise assistant."),
        *history_messages,
        HumanMessage(content=question),
    ]
    resp = llm.invoke(messages)
    answer = (getattr(resp, "content", "") or "").strip()
    if not answer:
        # Single retry without history if blank
        resp = llm.invoke([SystemMessage(content="You are a helpful, concise assistant."), HumanMessage(content=question)])
        answer = (getattr(resp, "content", "") or "").strip()
    return answer

# In case of duplicate docs
def _dedup_docs(docs):
    seen, out = set(), []
    for d in docs:
        content = (getattr(d, "page_content", "") or "")
        digest = hashlib.md5(content.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        out.append(d)
    return out

def format_history_from_db(chat_messages_db):
    """Convert DB rows to LangChain messages, skipping blanks."""
    msgs = []
    for msg in chat_messages_db:
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if msg.get("sender") == "user":
            msgs.append(HumanMessage(content=content))
        elif msg.get("sender") == "bot":
            msgs.append(AIMessage(content=content))
    return msgs

def generate_rag_response(vectorstore, question, history_messages, k=8, return_docs=False):
    """Retrieve -> prompt with context -> LLM, with safe fallback."""
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    # Prefer high-recall similarity first; fetch more, then keep top-k after dedup
    try:
        sim_docs = vectorstore.similarity_search(question, k=k)
    except Exception as e:
        print(f"[WARN] similarity_search failed: {e}")
        sim_docs = []

    merged = _dedup_docs(sim_docs)[:k]
    print(f"[INFO] Retrieved {len(sim_docs)} sim, merged {len(merged)}")

    if not merged:
        print("[INFO] No docs retrieved; falling back to general response.")
        fallback = generate_general_response(question, history_messages)
        return {"answer": fallback, "documents": []} if return_docs else fallback

    context_text = "\n\n---\n\n".join(getattr(d, "page_content", "") for d in merged)
    system_prompt = (
        "You are a financial analysis expert. Carefully analyze the provided context "
        "from financial documents. For numerical questions, cite specific figures. "
        "Be precise and concise.\n\n"
        f"Context:\n{context_text}"
    )
    messages = [
        SystemMessage(content=system_prompt),
        *history_messages,
        HumanMessage(content=question),
    ]
    resp = llm.invoke(messages)
    # print(f"[CHAT] Retrieved response from RAG chain: {resp}")
    answer = (getattr(resp, "content", "") or "").strip()

    if not answer:
        print("[INFO] Empty RAG answer; retrying without context...")
        answer = generate_general_response(question, history_messages)

    return {"answer": answer, "documents": merged} if return_docs else answer


# Local run function
def run_conversational_agent(pdf_files):
    print("[INFO] Checking for existing FAISS vectorstore...")
    if not os.path.exists(index_path):
        print("[INFO] No vectorstore found. Extracting from PDFs...")
        raw_text = get_pdf_text(pdf_files)
        if not raw_text.strip():
            print("[ERROR] No text extracted. Exiting.")
            return
        cleaned_text = clean_text(raw_text)
        text_chunks = get_text_chunks(cleaned_text)
    else:
        text_chunks = []

    vectorstore = get_or_load_vectorstore(text_chunks, path=index_path)

    print("\n[READY] Ask questions about the document. Type 'exit' to quit.\n")

    # Maintain chat history as LangChain messages for follow-ups
    history_msgs = []

    while True:
        query = input("You: ").strip()
        if query.lower() == "exit":
            print("[INFO] Exiting conversation.")
            break

        # RAG response using the latest pipeline
        answer = generate_rag_response(vectorstore, query, history_messages=history_msgs, k=5)
        print("Bot:", answer)

        # Update history for follow-up questions
        history_msgs.append(HumanMessage(content=query))
        history_msgs.append(AIMessage(content=answer))


def _safe_first(row: dict, keys: list[str]):
    for k in keys:
        v = row.get(k)
        if v is not None and str(v).strip():
            return v
    return None

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _download_pdf(url: str, dest_path: str):
    _ensure_dir(os.path.dirname(dest_path))
    if os.path.exists(dest_path):
        return dest_path
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return dest_path
    except Exception as e:
        print(f"[WARN] Failed to download PDF from {url}: {e}")
        return None

def prepare_financebench(limit: int = 20):
    """
    Load first `limit` rows of FinanceBench, download PDFs, and map Q&A pairs.
    Returns (pdf_paths, dataset)
    """
    ds = load_dataset("PatronusAI/financebench", split="train")
    rows = ds.select(range(min(limit, len(ds))))
    pdf_paths = []
    qa_dataset = []
    seen_docs = set()
    for row in rows:
        # Determine doc_name early and skip if in blocklist
        doc_name = _safe_first(row, ["doc_name", "doc", "document", "document_name"])
        base_doc = os.path.splitext(doc_name)[0].lower() if doc_name else None
        if base_doc and base_doc in SKIPPED_DOCS:
            print(f"[INFO] Skipping blocked dataset row for doc: {doc_name}")
            continue

        # Determine PDF URL
        pdf_url = _safe_first(row, ["pdf_url", "url", "document_url"])
        if not pdf_url:
            if not doc_name:
                print("[WARN] Row missing both pdf_url and doc_name; skipping.")
                continue
            fname = doc_name if doc_name.lower().endswith(".pdf") else f"{doc_name}.pdf"
            pdf_url = f"https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs/{fname}"
        else:
            # Try to derive a clean filename and doc_name if missing
            base = os.path.basename(pdf_url.split("?")[0])
            fname = base if base.lower().endswith(".pdf") else (base + ".pdf")
            if not doc_name:
                doc_name = os.path.splitext(fname)[0]
                base_doc = doc_name.lower()

        # Skip after deriving doc_name
        if base_doc and base_doc in SKIPPED_DOCS:
            print(f"[INFO] Skipping blocked PDF download for: {doc_name}")
            continue

        # Download PDF once per doc
        if doc_name in seen_docs:
            pass
        else:
            dest = os.path.join(FINANCEBENCH_PDF_DIR, f"{doc_name}.pdf")
            saved = _download_pdf(pdf_url, dest)
            if saved:
                pdf_paths.append(os.path.abspath(saved))
                seen_docs.add(doc_name)

        # Map Q&A
        q = _safe_first(row, ["question", "prompt", "query"])
        a = _safe_first(row, ["answer", "final_answer", "ground_truth", "reference_answer", "label"])
        if q and a and (not base_doc or base_doc not in SKIPPED_DOCS):
            qa_dataset.append({"inputs": {"question": q}, "outputs": {"answer": a}})
        else:
            print("[WARN] Skipping Q&A row due to missing fields or blocked doc.")
    if not qa_dataset:
        print("[WARN] No usable FinanceBench Q&A rows collected.")
    if not pdf_paths:
        print("[WARN] No FinanceBench PDFs downloaded.")
    return pdf_paths, qa_dataset

# Local runner
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG pipeline with optional evaluation.")
    parser.add_argument("--eval", action="store_true", help="Run evaluation instead of interactive chat")
    parser.add_argument("--pdfs", "--pdf", nargs="+", dest="pdfs", default=["Apple_Q3.pdf", "Tesla_Q3.pdf"], help="List of PDF files or paths")
    parser.add_argument("--k", type=int, default=15, help="Top-k documents for retrieval during evaluation")
    parser.add_argument("--index-path", dest="idx_path", default=index_path, help="Path to FAISS index directory")
    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuild vectorstore from PDFs")
    # FinanceBench options
    parser.add_argument("--financebench", action="store_true", help="Use FinanceBench dataset and PDFs (first N rows)")
    parser.add_argument("--financebench-rows", type=int, default=20, help="Number of FinanceBench rows to ingest")
    # Baseline flag to disable retrieval during evaluation
    parser.add_argument("--no-rag-baseline", action="store_true", help="Evaluate LLM-only baseline (no retrieval)")
    args = parser.parse_args()

    index_path = args.idx_path

    # Decide data source
    dataset_for_eval = None
    pdf_list = args.pdfs
    if args.financebench:
        print(f"[INFO] Using FinanceBench (first {args.financebench_rows} rows)")
        fb_pdfs, fb_dataset = prepare_financebench(limit=args.financebench_rows)
        # Use FinanceBench PDFs for indexing; fall back to provided pdfs if none downloaded
        if fb_pdfs:
            pdf_list = fb_pdfs
        dataset_for_eval = fb_dataset

    # Build or load vectorstore only if needed:
    # - Always needed for interactive chat (not args.eval)
    # - Needed for evaluation when NOT running baseline
    vs = None
    if (not args.eval) or (args.eval and not args.no_rag_baseline):
        if args.rebuild_index or not os.path.exists(index_path):
            print("[INFO] Building vectorstore from PDFs...")
            raw_text = get_pdf_text(pdf_list)
            if not raw_text.strip():
                print("[ERROR] No text extracted. Exiting.")
                raise SystemExit(1)
            cleaned_text = clean_text(raw_text)
            text_chunks = get_text_chunks(cleaned_text)
        else:
            print("[INFO] Using existing vectorstore at:", index_path)
            print("[WARN] If this index was built with a different embedding model, pass --rebuild-index.")
            text_chunks = []
        vs = get_or_load_vectorstore(text_chunks, path=index_path)

    run_conversational_agent(pdf_list)
