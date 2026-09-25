import os
import sys
import webbrowser
import time
import uvicorn

def main():
    print("="*65)
    print("  HemaVision AI: Leukemia Detection & Clinical Decision Support")
    print("="*65)
    print("[*] Initializing Cytological AI Engine...")
    print("[*] Server Address: http://127.0.0.1:8000")
    print("[*] API Documentation: http://127.0.0.1:8000/docs")
    print("[*] Press Ctrl+C to terminate.")
    print("="*65)
    
    # Run uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    main()
