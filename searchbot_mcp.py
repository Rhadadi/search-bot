#!/usr/bin/env python3
"""Standalone entry for MCP clients that scrub PYTHONPATH/cwd (this script
makes itself self-contained). Run: .venv/bin/python searchbot_mcp.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from searchbot.mcp_server import serve

if __name__ == "__main__":
    serve()
