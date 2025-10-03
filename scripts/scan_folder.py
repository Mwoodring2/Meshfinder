#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Thin CLI wrapper around src/indexer/modelfinder_indexer.py
with auto-detect defaults for --fast and --workers.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from indexer.modelfinder_indexer import scan_and_index  # noqa: E402

def _cpu_default_workers() -> int:
    try:
        cpus = os.cpu_count() or 4
    except Exception:
        cpus = 4
    # Keep UI responsive and avoid oversubscription on Windows
    return max(2, min(8, cpus - 1))

def _has_scipy() -> bool:
    try:
        import scipy  # noqa: F401
        return True
    except Exception:
        return False

def _free_ram_gb() -> float:
    try:
        import psutil  # optional
        return psutil.virtual_memory().available / (1024**3)
    except Exception:
        # Fallback: unknown; return a high value so it doesn't trigger fast mode
        return 999.0

def _count_supported(root: Path) -> tuple[int, int]:
    # Mirror supported extensions from indexer (keep in sync if you change there)
    SUPPORTED_EXTS = {".stl", ".obj", ".ztl"}
    total = 0
    supported = 0
    for dp, _, fns in os.walk(root):
        for fn in fns:
            total += 1
            if Path(fn).suffix.lower() in SUPPORTED_EXTS:
                supported += 1
    return total, supported

def _posters_available() -> bool:
    try:
        import pyrender  # noqa: F401
        import OpenGL  # noqa: F401
        return True
    except Exception:
        return False

def main():
    ap = argparse.ArgumentParser(description="ModelFinder Scan Folder")
    ap.add_argument('--root', type=Path, required=True, help='Root folder to scan')
    ap.add_argument('--out', dest='out_dir', type=Path, default=Path('data'), help='Output folder for GLB/posters')
    ap.add_argument('--db', dest='db_path', type=Path, default=Path('db/modelfinder.db'), help='SQLite database path')
    ap.add_argument('--faiss', dest='faiss_path', type=Path, default=Path('db/faiss.index'), help='FAISS index path')

    # Tri-state fast flag: --fast / --no-fast / auto (default)
    fast_grp = ap.add_mutually_exclusive_group()
    fast_grp.add_argument('--fast', dest='fast', action='store_true', help='Skip expensive metrics like exact volume')
    fast_grp.add_argument('--no-fast', dest='fast', action='store_false', help='Force full metrics mode')
    ap.set_defaults(fast=None)  # None = auto-detect

    ap.add_argument('--posters', action='store_true', help='Attempt poster rendering (requires pyrender + OpenGL)')
    ap.add_argument('--workers', type=int, default=None, help='Parallel workers (default: auto)')

    args = ap.parse_args()

    # Auto workers
    workers = args.workers if args.workers and args.workers > 0 else _cpu_default_workers()

    # Quick pre-scan counts
    total, supported = _count_supported(args.root)

    # Auto fast decision only if user didn’t specify
    fast = args.fast
    if fast is None:
        reasons = []
        if not _has_scipy():
            reasons.append("no-scipy")
        if supported > 1000:
            reasons.append(f"many-files({supported})")
        if _free_ram_gb() < 3.0:
            reasons.append("low-ram")
        fast = len(reasons) > 0
        if fast:
            print(f"[AUTO] fast=True (reason: {', '.join(reasons)})")
        else:
            print("[AUTO] fast=False (full metrics)")

    # Posters availability warning
    if args.posters and not _posters_available():
        print("[WARN] Posters requested but pyrender/OpenGL not available; posters will be skipped.")

    # Summary
    print(
        f"[INFO] Starting scan\n"
        f"  root      : {args.root}\n"
        f"  out_dir   : {args.out_dir}\n"
        f"  db_path   : {args.db_path}\n"
        f"  faiss     : {args.faiss_path}\n"
        f"  posters   : {args.posters}\n"
        f"  workers   : {workers}\n"
        f"  fast      : {fast}\n"
        f"  files     : total={total}, supported={supported}"
    )

    t0 = time.time()
    scan_and_index(
        root=args.root,
        out_dir=args.out_dir,
        db_path=args.db_path,
        faiss_path=args.faiss_path,
        posters=args.posters,
        workers=workers,
        fast=fast,
    )
    dt = time.time() - t0
    print(f"[DONE] Total time: {dt:.1f}s")

if __name__ == '__main__':
    main()
