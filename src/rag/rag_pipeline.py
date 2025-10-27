import os
import hashlib
import argparse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing_extensions import Annotated, TypedDict
from pdf_utilities import clean_text, get_pdf_text

index_path = os.path.join(os.path.dirname(__file__), "faiss_index")

load_dotenv()

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


# -------------------------
# Evaluation utilities (LLM-as-judge)
# -------------------------

# Typed schemas for structured grading
class CorrectnessGrade(TypedDict):
    explanation: Annotated[str, ..., "Reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer matches the reference"]

class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Reasoning for the score"]
    relevant: Annotated[bool, ..., "True if the answer addresses the question"]

class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Reasoning for the score"]
    grounded: Annotated[bool, ..., "True if the answer is supported by the docs"]

class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Reasoning for the score"]
    relevant: Annotated[bool, ..., "True if retrieved docs relate to the question"]

# Grader prompts
correctness_instructions = """
    You are grading a QA task.
    You get a QUESTION, a REFERENCE (ground truth) ANSWER, and a MODEL ANSWER.
    Score True only if the MODEL ANSWER is factually consistent with the REFERENCE.
    It's fine if the model adds extra details as long as they don't contradict the reference.
    Explain briefly, then output the boolean.
"""
relevance_instructions = """
    You are grading if a MODEL ANSWER addresses a QUESTION.
    Score True if the answer is concise and helps answer the question. Explain briefly, then output the boolean.
"""
grounded_instructions = """
    You are grading if a MODEL ANSWER is supported by the provided FACTS (retrieved docs).
    Score True if the answer stays within the facts and does not hallucinate. Explain briefly, then output the boolean.
"""
retrieval_relevance_instructions = """
    You are grading if the retrieved FACTS are relevant to the QUESTION.
    Score True if the facts contain any keywords or semantic meaning related to the question, even if some content is extra. Explain briefly, then output the boolean.
"""

# Create graders
def _make_llm_structured(schema):
    base = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)
    return base.with_structured_output(schema, method="json_schema", strict=True)

_grader_correctness = _make_llm_structured(CorrectnessGrade)
_grader_relevance = _make_llm_structured(RelevanceGrade)
_grader_grounded = _make_llm_structured(GroundedGrade)
_grader_retrieval = _make_llm_structured(RetrievalRelevanceGrade)

def eval_correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    payload = f"QUESTION: {inputs['question']}\nREFERENCE ANSWER: {reference_outputs['answer']}\nMODEL ANSWER: {outputs['answer']}"
    grade = _grader_correctness.invoke([
        {"role": "system", "content": correctness_instructions},
        {"role": "user", "content": payload},
    ])
    return bool(grade["correct"])

def eval_relevance(inputs: dict, outputs: dict) -> bool:
    payload = f"QUESTION: {inputs['question']}\nMODEL ANSWER: {outputs['answer']}"
    grade = _grader_relevance.invoke([
        {"role": "system", "content": relevance_instructions},
        {"role": "user", "content": payload},
    ])
    return bool(grade["relevant"])

def eval_groundedness(inputs: dict, outputs: dict) -> bool:
    docs = outputs.get("documents") or []
    doc_string = "\n\n".join(getattr(d, "page_content", "") for d in docs)
    payload = f"FACTS:\n{doc_string}\n\nMODEL ANSWER: {outputs['answer']}"
    grade = _grader_grounded.invoke([
        {"role": "system", "content": grounded_instructions},
        {"role": "user", "content": payload},
    ])
    return bool(grade["grounded"])

def eval_retrieval_relevance(inputs: dict, outputs: dict) -> bool:
    docs = outputs.get("documents") or []
    doc_string = "\n\n".join(getattr(d, "page_content", "") for d in docs)
    payload = f"FACTS:\n{doc_string}\n\nQUESTION: {inputs['question']}"
    grade = _grader_retrieval.invoke([
        {"role": "system", "content": retrieval_relevance_instructions},
        {"role": "user", "content": payload},
    ])
    return bool(grade["relevant"])

# Benchmark datasets
def build_sample_dataset():
    return [
        {
            "inputs": {"question": "What were the sales for the iPad in 2025?"},
            "outputs": {"answer": "The iPad net sales for the three months ended June 28, 2025, were $6,581 million, and for the nine months ended June 28, 2025, were $21,071 million."},
        },
        {
            "inputs": {"question": "What was Tesla's gross profit in 2025 compared to 2024?"},
            "outputs": {"answer": "Tesla's total gross profit in Q3 2025 was $5,054 million, which is a 1% increase compared to $4,997 million in Q3 2024."},
        },
        {
            "inputs": {"question": "What are the total net sales of Apple in 2025 so far?"},
            "outputs": {"answer": "The total net sales of Apple in 2025 so far (for the nine months ended June 28, 2025) are $313,695 million."},
        },
        {
            "inputs": {"question": "What was the operating income of Apple in Japan in Q3 of 2025?"},
            "outputs": {"answer": "The operating income of Apple in Japan for Q3 of 2025 (three months ended June 28, 2025) was $2,872 million."},
        },
        {
            "inputs": {"question": "What are the basic and diluted earnings per share for Apple so far in 2025?"},
            "outputs": {"answer": "For the nine months ended June 28, 2025, Apple's basic earnings per share were $5.64, and the diluted earnings per share were $5.62."},
        },
        {
            "inputs": {"question": "What was the driving factor of a change in net sales for Apple in Europe in 2025?"},
            "outputs": {"answer": "The increase in net sales for Apple in Europe during 2025 was primarily due to higher net sales of Services and iPhone."},
        },
        {
            "inputs": {"question": "Which Apple product has contributed the most to sales in 2025?"},
            "outputs": {"answer": "The iPhone contributed the most to Apple's sales in 2025, with net sales of $44,582 million for the third quarter and $160,561 million for the nine months ended June 28, 2025."}
        }
    ]

def run_evaluation(vectorstore, dataset=None, k=5):
    dataset = dataset or build_sample_dataset()
    results = []
    print("[EVAL] Running RAG evaluation on", len(dataset), "examples")
    for i, ex in enumerate(dataset, 1):
        q = ex["inputs"]["question"]
        ref = ex["outputs"]
        # Run RAG with docs
        out = generate_rag_response(vectorstore, q, history_messages=[], k=k, return_docs=True)
        # Apply evaluators
        correctness = eval_correctness(ex["inputs"], out, ref)
        grounded = eval_groundedness(ex["inputs"], out)
        relevance = eval_relevance(ex["inputs"], out)
        retr_rel = eval_retrieval_relevance(ex["inputs"], out)
        results.append({
            "question": q,
            "answer": out["answer"],
            "correctness": correctness,
            "groundedness": grounded,
            "relevance": relevance,
            "retrieval_relevance": retr_rel,
        })
        print(f"[EVAL] {i}. corr={correctness} grounded={grounded} rel={relevance} retr_rel={retr_rel}")

    # Summary
    def _avg(key): return sum(1 for r in results if r[key]) / max(1, len(results))
    print("\n[EVAL][SUMMARY]")
    print("Correctness:", _avg("correctness"))
    print("Groundedness:", _avg("groundedness"))
    print("Relevance:", _avg("relevance"))
    print("Retrieval relevance:", _avg("retrieval_relevance"))
    return results

# Local runner
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG pipeline with optional evaluation.")
    parser.add_argument("--eval", action="store_true", help="Run evaluation instead of interactive chat")
    parser.add_argument("--pdfs", "--pdf", nargs="+", dest="pdfs", default=["Apple_Q3.pdf", "Tesla_Q3.pdf"], help="List of PDF files or paths")
    parser.add_argument("--k", type=int, default=5, help="Top-k documents for retrieval during evaluation")
    parser.add_argument("--index-path", dest="idx_path", default=index_path, help="Path to FAISS index directory")
    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuild vectorstore from PDFs")
    args = parser.parse_args()

    # Respect CLI index path across helpers
    index_path = args.idx_path  # override module-level default

    eval_mode = args.eval

    # Prepare vectorstore andrebuild when requested or missing
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

    if eval_mode:
        print("[INFO] Evaluation mode enabled.")
        run_evaluation(vs, dataset=build_sample_dataset(), k=args.k)
    else:
        # Run interactive chat
        run_conversational_agent(args.pdfs)