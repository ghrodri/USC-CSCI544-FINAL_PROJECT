import sqlite3
import os

DATABASE_PATH = os.path.join(os.path.dirname(__file__), "chatbot.db")

def init_database():
    """Initialize the database with required tables"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Create PDFs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pdfs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            file_size INTEGER,
            status TEXT DEFAULT 'processed'
        )
    ''')
    
    # Create chats table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create messages table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            sender TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chat_id) REFERENCES chats (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def add_pdf(filename, file_size):
    """Add a new PDF to the database"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO pdfs (filename, file_size)
        VALUES (?, ?)
    ''', (filename, file_size))
    
    pdf_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return pdf_id

def get_all_pdfs():
    """Get all uploaded PDFs"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, filename, upload_date, file_size, status
        FROM pdfs
        ORDER BY upload_date DESC
    ''')
    
    pdfs = cursor.fetchall()
    conn.close()
    
    return [
        {
            "id": pdf[0],
            "filename": pdf[1],
            "upload_date": pdf[2],
            "file_size": pdf[3],
            "status": pdf[4]
        }
        for pdf in pdfs
    ]

def delete_pdf(pdf_id):
    """Delete a PDF from the database"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM pdfs WHERE id = ?', (pdf_id,))
    
    conn.commit()
    conn.close()

# Chat functions
def create_chat(title="New Chat"):
    """Create a new chat"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('INSERT INTO chats (title) VALUES (?)', (title,))
    chat_id = cursor.lastrowid
    
    conn.commit()
    conn.close()
    
    return chat_id

def get_all_chats():
    """Get all chats"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, title, created_date, last_updated
        FROM chats
        ORDER BY last_updated DESC
    ''')
    
    chats = cursor.fetchall()
    conn.close()
    
    return [
        {
            "id": chat[0],
            "title": chat[1],
            "created_date": chat[2],
            "last_updated": chat[3]
        }
        for chat in chats
    ]

def get_chat_messages(chat_id):
    """Get all messages for a chat"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, content, sender, timestamp
        FROM messages
        WHERE chat_id = ?
        ORDER BY timestamp ASC
    ''', (chat_id,))
    
    messages = cursor.fetchall()
    conn.close()
    
    return [
        {
            "id": msg[0],
            "content": msg[1],
            "sender": msg[2],
            "timestamp": msg[3]
        }
        for msg in messages
    ]

def add_message(chat_id, content, sender):
    """Add a message to a chat"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO messages (chat_id, content, sender)
        VALUES (?, ?, ?)
    ''', (chat_id, content, sender))
    
    # Update chat's last_updated timestamp
    cursor.execute('''
        UPDATE chats SET last_updated = CURRENT_TIMESTAMP
        WHERE id = ?
    ''', (chat_id,))
    
    message_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return message_id

def update_chat_title(chat_id, title):
    """Update chat title"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('UPDATE chats SET title = ? WHERE id = ?', (title, chat_id))
    
    conn.commit()
    conn.close()

def delete_chat(chat_id):
    """Delete a chat and all its messages"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM messages WHERE chat_id = ?', (chat_id,))
    cursor.execute('DELETE FROM chats WHERE id = ?', (chat_id,))
    
    conn.commit()
    conn.close()