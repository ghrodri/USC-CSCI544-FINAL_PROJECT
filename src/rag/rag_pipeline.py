import os
import hashlib
import argparse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pdf_utilities import clean_text, get_pdf_text
from rag_evaluation import run_evaluation, build_sample_dataset

load_dotenv()
index_path = os.path.join(os.path.dirname(__file__), "faiss_index")

def get_text_chunks(text, chunk_size=1000, chunk_overlap=300):
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


def get_or_load_vectorstore(text_chunks, path="faiss_index", model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    if os.path.exists(path):
        print(f"[INFO] Loading existing vectorstore from '{path}'...")
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
    llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)
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

def generate_rag_response(vectorstore, question, history_messages, k=8, return_docs=False):
    """Retrieve -> prompt with context -> LLM, with safe fallback."""
    llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)

    # Prefer high-recall similarity first; fetch more, then keep top-k after dedup
    try:
        fetch_k = max(15, k * 3)
        sim_docs = vectorstore.similarity_search(question, k=fetch_k)
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
        "You are a helpful assistant. Use the provided context to answer the user's question. "
        "If the context is insufficient, say you don't know. Be concise.\n\n"
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


# Local runner
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG pipeline with optional evaluation.")
    parser.add_argument("--eval", action="store_true", help="Run evaluation instead of interactive chat")
    parser.add_argument("--pdfs", "--pdf", nargs="+", dest="pdfs", default=["Apple_Q3.pdf", "Tesla_Q3.pdf"], help="List of PDF files or paths")
    parser.add_argument("--k", type=int, default=5, help="Top-k documents for retrieval during evaluation")
    parser.add_argument("--index-path", dest="idx_path", default=index_path, help="Path to FAISS index directory")
    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuild vectorstore from PDFs")
    args = parser.parse_args()

    index_path = args.idx_path

    # Build or load vectorstore
    if args.rebuild_index or not os.path.exists(index_path):
        print("[INFO] Building vectorstore from PDFs...")
        raw_text = get_pdf_text(args.pdfs)
        if not raw_text.strip():
            print("[ERROR] No text extracted. Exiting.")
            raise SystemExit(1)
        cleaned_text = clean_text(raw_text)
        text_chunks = get_text_chunks(cleaned_text)
    else:
        print("[INFO] Using existing vectorstore at:", index_path)
        text_chunks = []
    vs = get_or_load_vectorstore(text_chunks, path=index_path)

    if args.eval:
        print("[INFO] Evaluation mode enabled.")
        # Wrap generator to avoid circular imports
        def generate_fn(q: str):
            return generate_rag_response(vs, q, history_messages=[], k=args.k, return_docs=True)
        run_evaluation(generate_fn=generate_fn, dataset=build_sample_dataset())
    else:
        run_conversational_agent(args.pdfs)