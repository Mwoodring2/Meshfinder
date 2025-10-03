#!/usr/bin/env python3
"""
ModelFinder Indexer Runner
A minimal wrapper script for running the ModelFinder indexer.
"""

import argparse
from pathlib import Path
import subprocess
import sys

def main():
    ap = argparse.ArgumentParser(description="Run ModelFinder indexer")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--db", type=Path, default=Path("db/modelfinder.db"))
    ap.add_argument("--faiss", type=Path, default=Path("db/faiss.index"))
    ap.add_argument("--posters", action="store_true")
    args = ap.parse_args()

    cmd = [
        sys.executable, "src/indexer/modelfinder_indexer.py",
        "--root", str(args.root), "--out", str(args.out),
        "--db", str(args.db), "--faiss", str(args.faiss),
    ]
    if args.posters:
        cmd.append("--posters")

    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()














