#!/usr/bin/env python3
"""
ModelFinder - 3D Model Search CLI with FAISS Vector Similarity
Advanced search capabilities using vector embeddings and similarity search.
"""

import argparse
import sqlite3
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import faiss  # pip install faiss-cpu

def load_assets_and_vectors(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    # Join embeddings with assets so we have metadata alongside vectors
    cur.execute("""
        SELECT a.id, a.path, a.ext, a.tri_count, a.bbox_x, a.bbox_y, a.bbox_z,
               a.is_watertight, e.dim, e.vec
        FROM embeddings e
        JOIN assets a ON a.id = e.asset_id
        ORDER BY a.id ASC
    """)
    rows = cur.fetchall()
    if not rows:
        raise SystemExit("No embeddings found. Run the indexer first.")

    ids, meta, vecs = [], [], []
    for r in rows:
        ids.append(r["id"])
        meta.append({
            "id": r["id"],
            "path": r["path"],
            "ext": r["ext"],
            "tri_count": r["tri_count"],
            "bbox": (r["bbox_x"], r["bbox_y"], r["bbox_z"]),
            "watertight": bool(r["is_watertight"]) if r["is_watertight"] is not None else None,
        })
        v = np.frombuffer(r["vec"], dtype=np.float32)
        vecs.append(v)
    X = np.vstack(vecs).astype(np.float32)
    # Cosine similarity via inner product: normalize rows
    faiss.normalize_L2(X)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    return conn, np.array(ids, dtype=np.int64), meta, X, index

def find_query_vector(conn, ids, X, like_id=None, like_path=None, manual_vec=None):
    if manual_vec is not None:
        q = np.asarray(manual_vec, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(q)
        return q, "manual vector"
    cur = conn.cursor()
    if like_id is not None:
        cur.execute("SELECT asset_id, dim, vec FROM embeddings WHERE asset_id=?", (like_id,))
        row = cur.fetchone()
        if row is None:
            raise SystemExit(f"No embedding found for asset id {like_id}")
        v = np.frombuffer(row[2], dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(v)
        return v, f"asset_id={like_id}"
    if like_path is not None:
        cur.execute("SELECT id FROM assets WHERE path=?", (like_path,))
        row = cur.fetchone()
        if row is None:
            raise SystemExit(f"No asset found at path: {like_path}")
        return find_query_vector(conn, ids, X, like_id=row[0])
    raise SystemExit("Provide --like-id, --like-path, or --vec")

def passes_filters(m, ext, max_tris, watertight):
    if ext and (m["ext"] or "").lower() != ext.lower():
        return False
    if max_tris is not None and m["tri_count"] is not None:
        if m["tri_count"] > max_tris:
            return False
    if watertight is not None and m["watertight"] is not None:
        if bool(m["watertight"]) != bool(watertight):
            return False
    return True

def reveal_in_explorer(path: str):
    # Show file highlighted in Explorer (Windows only)
    norm = os.path.normpath(path)
    if os.name == "nt":
        subprocess.run(["explorer", "/select,", norm])
    else:
        print("Reveal is only supported on Windows.")

def open_with_default_app(path: str):
    norm = os.path.normpath(path)
    if os.name == "nt":
        os.startfile(norm)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.run(["open", norm])
    else:
        subprocess.run(["xdg-open", norm])

def main():
    ap = argparse.ArgumentParser(description="Similarity search over ModelFinder (SQLite + FAISS)")
    ap.add_argument("--db", type=Path, default=Path("db/modelfinder.db"), help="SQLite DB path")
    ap.add_argument("--topk", type=int, default=10, help="Top-K results")
    ap.add_argument("--like-id", type=int, help="Find items similar to this asset id")
    ap.add_argument("--like-path", type=str, help=r"Find items similar to this file path")
    ap.add_argument("--vec", type=str, help="Comma-separated 10D vector (advanced/manual)")

    # Filters
    ap.add_argument("--ext", type=str, help="Filter by extension (e.g., .obj or .stl)")
    ap.add_argument("--max-tris", type=int, help="Filter by triangle count upper bound")
    ap.add_argument("--watertight", type=str, choices=["true","false"], help="Filter by watertight flag")

    # Actions on results
    ap.add_argument("--reveal", type=int, help="Reveal result N (1-based) in Explorer")
    ap.add_argument("--reveal-all", action="store_true", help="Reveal all shown results in Explorer")
    ap.add_argument("--open", type=int, help="Open result N (1-based) with default app")
    ap.add_argument("--open-all", action="store_true", help="Open all shown results with default app")

    args = ap.parse_args()

    watertight_flag = None
    if args.watertight is not None:
        watertight_flag = (args.watertight.lower() == "true")

    manual_vec = None
    if args.vec:
        manual_vec = [float(x.strip()) for x in args.vec.split(",")]
    
    conn, ids, meta, X, index = load_assets_and_vectors(args.db)
    q, qdesc = find_query_vector(conn, ids, X, like_id=args.like_id, like_path=args.like_path, manual_vec=manual_vec)

    # Search more than topk so we can drop filtered-out entries
    search_k = max(args.topk * 5, args.topk + 20)
    D, I = index.search(q, search_k)
    scores = D[0]
    idxs = I[0]

    results = []
    for sc, ix in zip(scores, idxs):
        if ix < 0:  # FAISS returns -1 for empty slots
            continue
        m = meta[ix]
        if passes_filters(m, args.ext, args.max_tris, watertight_flag):
            results.append((sc, m))
        if len(results) >= args.topk:
            break

    print(f"\nQuery = {qdesc}")
    print(f"DB     = {args.db}")
    print(f"TopK   = {args.topk}\n")
    if not results:
        print("No results (try relaxing filters).")
        return

    # Pretty print table with row numbers for --reveal/--open
    colw = min(max((len(m['path']) for _, m in results), default=20), 100)
    print(f"{'#':>2}  {'Score':>7}  {'Ext':<5}  {'Tris':>10}  {'Watertight':<11}  {'Path'}")
    print("-"*2 + "  " + "-"*7 + "  " + "-"*5 + "  " + "-"*10 + "  " + "-"*11 + "  " + "-"*max(colw, 20))
    for i, (sc, m) in enumerate(results, start=1):
        tri = m['tri_count'] if m['tri_count'] is not None else -1
        wt = "yes" if m['watertight'] else "no"
        pth = m['path'][:colw] + ("…" if len(m['path']) > colw else "")
        print(f"{i:2d}  {sc:7.3f}  {m['ext']:<5}  {tri:10d}  {wt:<11}  {pth}")

    # Handle Explorer actions
    def get_paths():
        return [m['path'] for _, m in results]

    if args.reveal is not None:
        n = args.reveal
        if 1 <= n <= len(results):
            reveal_in_explorer(results[n-1][1]['path'])
        else:
            print(f"--reveal {n} is out of range (1..{len(results)})")

    if args.reveal_all:
        for p in get_paths():
            reveal_in_explorer(p)

    if args.open is not None:
        n = args.open
        if 1 <= n <= len(results):
            open_with_default_app(results[n-1][1]['path'])
        else:
            print(f"--open {n} is out of range (1..{len(results)})")

    if args.open_all:
        for p in get_paths():
            open_with_default_app(p)

if __name__ == "__main__":
    main()