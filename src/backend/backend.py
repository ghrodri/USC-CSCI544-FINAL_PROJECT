import os
import sys
import re
import tempfile
import traceback
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from backend.data_storage import (
    get_pdf_text, get_text_chunks, get_or_load_vectorstore, clean_text,
    format_history_from_db, generate_rag_response, generate_general_response
)
from backend.database import init_database, add_pdf, get_all_pdfs, delete_pdf, create_chat, get_all_chats, get_chat_messages, add_message, update_chat_title, delete_chat

TEXT_TO_SQL_CODE_DIR = os.path.join(os.path.dirname(__file__), "..", "text-to-sql", "code")
if TEXT_TO_SQL_CODE_DIR not in sys.path:
    sys.path.append(TEXT_TO_SQL_CODE_DIR)

try:
    from pipeline_api import run_financial_query  # type: ignore
except Exception as e:  # pragma: no cover - best-effort import
    run_financial_query = None  # type: ignore
    print(f"[STARTUP] Warning: text-to-SQL pipeline not available: {e}")

index_path = os.path.join(os.path.dirname(__file__), "faiss_index")

app = FastAPI(title="Chatbot API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global vectorstore (shared across all chats)
vectorstore = None

FINANCE_KEYWORDS = {
    "ytd",
    "year-to-date",
    "return",
    "performance",
    "price",
    "closing price",
    "close price",
    "portfolio",
    "holdings",
    "positions",
    "pnl",
    "p&l",
    "volatility",
    "sharpe",
    "beta",
    "dividend",
    "dividends",
    "dividend yield",
    "yield",
}


def is_financial_query(text: str) -> bool:
    """
    Heuristic router: decide if a message should go to the text-to-SQL engine.

    - Looks for finance/metric keywords.
    - Boosts if combined with what looks like a ticker symbol.
    """
    t = text.lower()

    if any(kw in t for kw in FINANCE_KEYWORDS):
        return True

    # Rough ticker pattern: 1–5 uppercase letters (e.g., AAPL, NTRS)
    has_ticker_pattern = bool(re.search(r"\b[A-Z]{1,5}\b", text))
    if has_ticker_pattern and any(kw in t for kw in {"return", "price", "portfolio", "yield", "dividend"}):
        return True

    return False


class ChatMessage(BaseModel):
    message: str
    chat_id: Optional[int] = None

class ChatResponse(BaseModel):
    response: str
    chat_id: Optional[int] = None

class CreateChatRequest(BaseModel):
    title: str = "New Chat"

class ChatListResponse(BaseModel):
    chats: list

@app.on_event("startup")
async def startup_event():
    """Initialize the app on startup"""
    global vectorstore
    # Initialize database
    init_database()
    
    # Try to load existing vectorstore if it exists
    try:
        if os.path.exists(index_path):
            print("[STARTUP] Loading existing vectorstore...")
            vectorstore = get_or_load_vectorstore([], path=index_path)  # Load without adding chunks
            print("[STARTUP] Loaded existing vectorstore")
        else:
            print("[STARTUP] No existing vectorstore found")
    except Exception as e:
        print(f"[STARTUP] Error loading vectorstore: {e}")
        vectorstore = None
    
    print("Backend started successfully!")

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF file"""
    global vectorstore
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
            file_size = len(content)
        
        # Process the PDF
        print(f"Processing PDF: {file.filename}")
        raw_text = get_pdf_text([tmp_file_path])
        
        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")
        
        # Clean and chunk the text
        cleaned_text = clean_text(raw_text)
        text_chunks = get_text_chunks(cleaned_text)
        
        # Add to database
        pdf_id = add_pdf(file.filename, file_size)
        
        # Create or update vectorstore (this will add to existing if it exists)
        if vectorstore is None:
            # First time: create vectorstore
            vectorstore = get_or_load_vectorstore(text_chunks, path=index_path)
        else:
            if text_chunks:
                vectorstore.add_texts(text_chunks)
                vectorstore.save_local(index_path)
                print(f"[INFO] Added {len(text_chunks)} chunks to existing vectorstore")
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return {"message": f"PDF '{file.filename}' processed successfully!", "filename": file.filename, "id": pdf_id}
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Handle chat messages"""
    global vectorstore
    
    try:
        print(f"[CHAT] Processing message: {message.message[:10]}...")
        
        # Create new chat if no chat_id provided
        if message.chat_id is None:
            chat_id = create_chat("New Chat")
            print(f"[CHAT] Created new chat: {chat_id}")
        else:
            chat_id = message.chat_id
        
        # Load history from DB and convert to LC messages
        chat_messages_db = get_chat_messages(chat_id)
        history_msgs = format_history_from_db(chat_messages_db)

        bot_response: Optional[str] = None

        # 1) Try text-to-SQL when it looks like a financial data question
        used_sql = False
        print(
            "[CHAT DEBUG] run_financial_query is None?:",
            run_financial_query is None,
            "| is_financial_query?:",
            is_financial_query(message.message),
        )

        if run_financial_query is not None and is_financial_query(message.message):
            try:
                print("[CHAT] Routing to text-to-SQL engine (Qwen)...")
                sql_out = run_financial_query(message.message)
                bot_response = sql_out.get("natural_answer") or ""

                # Optionally append a small hint that this came from structured data
                if bot_response:
                    bot_response += "\n\n_(Answer computed from your financial database via text-to-SQL.)_"

                used_sql = True
            except Exception as e:
                print(f"[CHAT] text-to-SQL engine failed, falling back to RAG/LLM: {e}")
                traceback.print_exc()
                bot_response = None  # ensure we fall through

        # 2) If we didn't use SQL (or it failed), fall back to RAG/general llm
        if not used_sql or not bot_response:
            if vectorstore is None:
                print("[CHAT] No vectorstore available, using general LLM")
                bot_response = generate_general_response(message.message, history_msgs)
            else:
                print("[CHAT] Using explicit RAG flow")
                bot_response = generate_rag_response(vectorstore, message.message, history_msgs, k=5)

        if not bot_response:
            bot_response = "Sorry, I couldn’t generate a response."

        # Limit response length
        if len(bot_response) > 2000:
            bot_response = bot_response[:2000] + "... [Response truncated]"

        # Save user message first, then bot response
        add_message(chat_id, message.message, "user")
        add_message(chat_id, bot_response, "bot")

        # Update chat title if this was the first message exchange
        messages_count = len(get_chat_messages(chat_id))
        if messages_count == 2:  # first user + bot pair
            title = message.message[:30] + "..." if len(message.message) > 30 else message.message
            update_chat_title(chat_id, title)
        
        return ChatResponse(response=bot_response, chat_id=chat_id)
        
    except Exception as e:
        print(f"[ERROR] Chat error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing your message: {str(e)}")

@app.get("/pdfs")
async def get_pdfs():
    """Get all uploaded PDFs"""
    try:
        pdfs = get_all_pdfs()
        return {"pdfs": pdfs}
    except Exception as e:
        print(f"Error getting PDFs: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving PDFs")

@app.delete("/pdfs/{pdf_id}")
async def delete_pdf_endpoint(pdf_id: int):
    """Delete a PDF"""
    try:
        delete_pdf(pdf_id)
        return {"message": "PDF deleted successfully"}
    except Exception as e:
        print(f"Error deleting PDF: {str(e)}")
        raise HTTPException(status_code=500, detail="Error deleting PDF")

@app.get("/debug/vectorstore")
async def debug_vectorstore():
    """Debug endpoint to test vectorstore"""
    global vectorstore
    
    if vectorstore is None:
        return {"status": "no_vectorstore", "message": "No vectorstore loaded"}
    
    try:
        test_query = "What is this document about?"
        answer = generate_rag_response(vectorstore, test_query, history_messages=[], k=3)
        return {
            "status": "success",
            "test_query": test_query,
            "response": (answer[:200] + "...") if answer else "",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Chat endpoints
@app.post("/chats", response_model=dict)
async def create_new_chat(request: CreateChatRequest):
    """Create a new chat"""
    try:
        chat_id = create_chat(request.title)
        return {"chat_id": chat_id, "message": "Chat created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating chat: {str(e)}")

@app.get("/chats", response_model=ChatListResponse)
async def get_chats():
    """Get all chats"""
    try:
        chats = get_all_chats()
        return ChatListResponse(chats=chats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving chats: {str(e)}")

@app.get("/chats/{chat_id}/messages")
async def get_chat_messages_endpoint(chat_id: int):
    """Get messages for a specific chat"""
    try:
        messages = get_chat_messages(chat_id)
        return {"messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving messages: {str(e)}")

@app.delete("/chats/{chat_id}")
async def delete_chat_endpoint(chat_id: int):
    """Delete a chat"""
    try:
        delete_chat(chat_id)
        return {"message": "Chat deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting chat: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)