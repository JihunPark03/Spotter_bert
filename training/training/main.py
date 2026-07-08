#!/usr/bin/env python3
"""Compatibility wrapper for the new training entrypoint."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    entrypoint = Path(__file__).resolve().parents[1] / "train_modernbert.py"
    runpy.run_path(str(entrypoint), run_name="__main__")
