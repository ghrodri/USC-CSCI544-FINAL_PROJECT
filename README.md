# Financial Portfolio Assistant

## Project Overview

The Financial Portfolio Assistant integrates three core NLP components to make financial information more accessible and interpretable for non-expert investors:

1. **Text-to-SQL** – Converts natural language queries into SQL for querying personal financial databases.
2. **Retrieval-Augmented Generation (RAG)** – Provides grounded explanations of financial metrics and concepts.
3. **Earnings Call Analysis** – Summarizes corporate transcripts and detects linguistic signals such as sentiment, uncertainty, and forward-looking tone.

These modules together enable clearer insights into financial data, improving decision-making and transparency in portfolio analysis.

---

## App

### Architecture

- **Frontend**: React.js with modern UI components
- **Backend**: FastAPI with Python
- **RAG Pipeline**: LangChain + OpenAI + FAISS
- **Vector Store**: FAISS for efficient similarity search

### Setup Instructions

### 1. Backend Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Set up environment variables
# Create a .env file with your OpenAI API key:
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### 2. Frontend Setup

The frontend shell script file takes care of any frontend-related setup steps.

### 3. Running the Application

**Terminal 1 - Start Backend:**

```bash
python start_backend.py
```

Backend will be available at: http://localhost:8000

**Terminal 2 - Start Frontend:**

```bash
./start_frontend.sh
```

Frontend will be available at: http://localhost:3000
