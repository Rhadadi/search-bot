#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

"""Standalone entry for MCP clients that scrub PYTHONPATH/cwd (this script
makes itself self-contained). Run: .venv/bin/python searchbot_mcp.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from searchbot.mcp_server import serve

if __name__ == "__main__":
    serve()
