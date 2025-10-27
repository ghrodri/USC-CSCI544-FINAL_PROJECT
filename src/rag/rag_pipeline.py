import os
import re
import pytesseract
import hashlib
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()
index_path = os.path.join(os.path.dirname(__file__), "faiss_index")

PDF_DIR = "pdfs/"

def get_pdf_text(pdf_docs):
    text = ""
    for pdf_path in pdf_docs:
        full_path = pdf_path
        if not os.path.isabs(full_path) and not os.path.exists(full_path):
            full_path = os.path.join(PDF_DIR, pdf_path) # For local run

        if not os.path.exists(full_path):
            print("[WARN] File not found:", full_path)
            continue

        print("[INFO] Reading:", full_path)
        extracted_text = ""
        try:
            pdf_reader = PdfReader(full_path)
            for i, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                except Exception as e:
                    print(f"[WARN] Page {i+1} read error: {e}")
                    page_text = None
                if page_text and page_text.strip():
                    extracted_text += page_text + "\n"
                else:
                    print(f"[INFO] Using OCR for page {i+1}...")
                    extracted_text += extract_text_ocr(full_path, i)
        except Exception as e:
            print(f"[WARN] PyPDF2 failed for '{full_path}': {e}")
            print("[INFO] Running full OCR for file...")
            extracted_text = extract_text_ocr(full_path)
        text += extracted_text + "\n"
    return text.strip()


def extract_text_ocr(pdf_path, page_index=None):
    text = ""
    try:
        pages = convert_from_path(pdf_path)
        if page_index is not None:
            pages = [pages[page_index]]
        for img in pages:
            text += pytesseract.image_to_string(img, lang="eng") + "\n"
    except Exception as e:
        print(f"[ERROR] OCR failed for '{pdf_path}': {e}")
    return text


def clean_text(text):
    """
    Normalize newlines while preserving both single and double line breaks.
    - Convert CRLF to LF.
    - Collapse 3+ newlines to exactly two.
    - Trim trailing spaces and collapse multiple spaces.
    """
    text = text.replace("\r\n", "\n")
    # Preserve single newlines; only collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


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

# RAG helper functions
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

def generate_rag_response(vectorstore, question, history_messages, k=8):
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
        return generate_general_response(question, history_messages)

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
    print(f"[CHAT] Retrieved response from RAG chain: {resp}")
    answer = (getattr(resp, "content", "") or "").strip()

    if not answer:
        print("[INFO] Empty RAG answer; retrying without context...")
        answer = generate_general_response(question, history_messages)

    return answer


# Local test run function
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


if __name__ == "__main__":
    pdf_files = ["Apple_Q3.pdf", "Tesla_Q3.pdf"]  # Example PDF files
    run_conversational_agent(pdf_files)