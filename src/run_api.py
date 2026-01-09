#!/usr/bin/env python
"""
API Server entry point - bypasses interactive menu
"""
import sys
from pathlib import Path
import os

project_root = str(Path(__file__).resolve().parent.parent)
src_dir = str(Path(__file__).resolve().parent)
os.environ['PYTHONPATH'] = src_dir
os.environ['DEBUG_MODE'] = 'true'

if __name__ == "__main__":
    print("🌐 Starting API Server...")
    print("   Server will be available at: http://0.0.0.0:8000")
    print("   API Documentation: http://0.0.0.0:8000/docs")
    print("   PYTHONPATH:", src_dir)
    print("\n   Use './start_server.sh' for auto-reload mode")

    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)