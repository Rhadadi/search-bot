#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

"""Start the search-bot web UI on http://127.0.0.1:8181"""
import sys
sys.path.insert(0, ".")
from searchbot import webserver
webserver.main(int(sys.argv[1]) if len(sys.argv) > 1 else 8181)
