#!/usr/bin/env python3
"""
Start the FastAPI backend server
"""
import uvicorn

if __name__ == "__main__":
    print("Starting FastAPI backend server...")
    print("Backend will be available at: http://localhost:8000")
    print("API docs will be available at: http://localhost:8000/docs")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    uvicorn.run(
        "backend.backend:app",
        host="127.0.0.1",
        port=8000,
        reload=True,  # keep dev reload
        reload_dirs=["backend"],  # only watch backend code
        reload_includes=["*.py"],
        reload_excludes=[
            "venv/*",
            ".venv/*",
            "frontend/*",
            "node_modules/*",
            "backend/faiss_index/*",
            "**/__pycache__/*",
            "*.db",
            "*.log",
            "*.pyc",
        ],
    )