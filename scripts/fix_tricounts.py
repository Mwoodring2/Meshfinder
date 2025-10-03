#!/usr/bin/env python
import argparse, os, sqlite3, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import trimesh
import numpy as np

def compute_metrics(path: str):
    try:
        m = trimesh.load(path, force='mesh')
        if isinstance(m, trimesh.Scene):
            if not m.geometry:
                return None
            m = trimesh.util.concatenate(tuple(m.geometry.values()))
        faces = 0 if (getattr(m, "faces", None) is None) else len(m.faces)
        if faces <= 0:
            return (0, 0.0, 0.0, 0.0)
        bounds = getattr(m, "bounds", None)
        if bounds is None or not isinstance(bounds, np.ndarray) or bounds.shape != (2, 3):
            bbox = (0.0, 0.0, 0.0)
        else:
            mins, maxs = bounds
            ex, ey, ez = (float(maxs[0]-mins[0]), float(maxs[1]-mins[1]), float(maxs[2]-mins[2]))
            bbox = (ex, ey, ez)
        return (int(faces), *bbox)
    except Exception:
        return None

def worker(row):
    rid, path = row
    if not path or not os.path.exists(path):
        return (rid, None, 0.0, 0.0, 0.0)
    res = compute_metrics(path)
    if res is None:
        return (rid, None, 0.0, 0.0, 0.0)
    tc, bx, by, bz = res
    return (rid, tc, bx, by, bz)

def main():
    ap = argparse.ArgumentParser(description="Recompute tri_count and bbox in-place for selected rows.")
    ap.add_argument("--db", required=True, help="Path to modelfinder.db")
    ap.add_argument("--where", default="tri_count = 0", help='SQL WHERE to select rows (e.g. "path LIKE ''%\\\\Marvel\\\\%'' AND tri_count = 0")')
    ap.add_argument("--limit", type=int, default=None, help="Limit number of rows to process")
    ap.add_argument("--workers", type=int, default=8, help="Parallel workers")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    cols = {r["name"] for r in conn.execute("PRAGMA table_info(assets)")}
    need = {"id","path","tri_count","bbox_x","bbox_y","bbox_z"}
    missing = need - cols
    if missing:
        print(f"[ERR] assets table missing columns: {missing}", file=sys.stderr)
        sys.exit(2)

    sql = f"SELECT id, path FROM assets WHERE {args.where}"
    if args.limit:
        sql += f" LIMIT {int(args.limit)}"
    rows = conn.execute(sql).fetchall()
    total = len(rows)
    if total == 0:
        print("[INFO] No rows match the WHERE clause.")
        return

    print(f"[INFO] Recomputing metrics for {total} rows (workers={args.workers})")
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(worker, (r["id"], r["path"])): r["id"] for r in rows}
        for fut in as_completed(futs):
            rid, tc, bx, by, bz = fut.result()
            if tc is None:
                fail += 1
                continue
            conn.execute(
                "UPDATE assets SET tri_count = ?, bbox_x = ?, bbox_y = ?, bbox_z = ? WHERE id = ?",
                (tc, bx, by, bz, rid),
            )
            ok += 1
            if (ok + fail) % 200 == 0:
                conn.commit()
                print(f"[PROGRESS] ok={ok} fail={fail}/{total}")
    conn.commit()
    print(f"[DONE] updated={ok} failed={fail} total={total}")

if __name__ == "__main__":
    main()
