#!/usr/bin/env python3
"""
Reset the chatbot app to a fresh state
This will delete the database, vectorstore, and any cached data
"""
import os
import shutil

def reset_app():
    """Reset the app to fresh state"""
    print("🔄 Resetting chatbot app...")
    
    # Files/directories to remove
    items_to_remove = [
        "backend/chatbot.db",
        "backend/faiss_index/",
        "backend/__pycache__/",
        "frontend/node_modules/",
        "frontend/build/"
    ]
    
    removed_count = 0
    
    for item in items_to_remove:
        if os.path.exists(item):
            try:
                if os.path.isdir(item):
                    shutil.rmtree(item, ignore_errors=False)
                    print(f"✅ Removed directory: {item}")
                else:
                    os.remove(item)
                    print(f"✅ Removed file: {item}")
                removed_count += 1
            except Exception as e:
                print(f"❌ Error removing {item}: {e}")
        else:
            print(f"ℹ️  {item} not found (already clean)")
    
    print(f"\n🎉 Reset complete! Removed {removed_count} items.")
    print("\n📝 Next steps:")
    print("1. Start backend: python start_backend.py")
    print("2. Start frontend: ./start_frontend.sh")
    print("3. Upload PDFs and start chatting!")

if __name__ == "__main__":
    reset_app()
