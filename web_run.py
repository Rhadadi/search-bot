#!/usr/bin/env python3
"""Start the search-bot web UI on http://127.0.0.1:8181"""
import sys
sys.path.insert(0, ".")
from searchbot import webserver
webserver.main(int(sys.argv[1]) if len(sys.argv) > 1 else 8181)
