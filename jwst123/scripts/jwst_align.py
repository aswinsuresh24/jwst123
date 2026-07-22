#!/usr/bin/env python3
"""Align JWST visits with JHAT / Gaia (group-level pipeline)."""

from __future__ import annotations

import sys
from pathlib import Path

# Repo root is two levels above jwst123/scripts/<this file>
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jwst123.align import main

if __name__ == "__main__":
    raise SystemExit(main())
