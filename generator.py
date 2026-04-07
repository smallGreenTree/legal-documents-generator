#!/usr/bin/env python3
"""Synthetic legal document generator entrypoint."""

from pathlib import Path

from src.synthetic_ner.cli import main

if __name__ == "__main__":
    main(project_root=Path(__file__).resolve().parent)
