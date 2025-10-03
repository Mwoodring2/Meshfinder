#!/usr/bin/env python3
import argparse, sqlite3, sys, os, math
from pathlib import Path

def connect(db_path: str):
    if not os.path.exists(db_path):
        print(f"[ERR] DB not found: {db_path}", file=sys.stderr)
        sys.exit(1)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def clause_where(where: str | None) -> str:
    return f"WHERE {where}" if where else ""

def fmt_int(n):
    return f"{n:,}" if isinstance(n, int) and n is not None else str(n)

def fmt_float(n, places=2):
    return f"{n:,.{places}f}" if isinstance(n, (float, int)) and n is not None else str(n)

def print_rows(rows, cols=None, float_places=2):
    if not rows:
        print("(no rows)")
        return
    if cols is None:
        cols = list(rows[0].keys())
    # compute widths
    widths = {c: len(c) for c in cols}
    def cell(c, v):
        if isinstance(v, float):
            # avoid printing -0.00
            v = 0.0 if abs(v) < 5e-13 else v
            s = f"{v:.{float_places}f}"
        elif isinstance(v, int):
            s = fmt_int(v)
        else:
            s = str(v)
        widths[c] = max(widths[c], len(s))
        return s
    table = [[cell(c, r[c]) for c in cols] for r in rows]
    # header
    line = " | ".join(c.ljust(widths[c]) for c in cols)
    sep  = "-+-".join("-"*widths[c] for c in cols)
    print(line)
    print(sep)
    for row in table:
        print(" | ".join(s.ljust(widths[c]) for s, c in zip(row, cols)))

def cmd_stats(conn, where=None):
    # overview
    sql_overview = f"""
        SELECT COUNT(*) AS total_assets,
               SUM(tri_count) AS sum_tris,
               AVG(NULLIF(tri_count,0)) AS avg_tris,
               MIN(tri_count) AS min_tris,
               MAX(tri_count) AS max_tris
        FROM assets
        {clause_where(where)};
    """
    row = conn.execute(sql_overview).fetchone()

    print("[OVERVIEW]")
    print("total_assets | sum_tris      | avg_tris  | min_tris | max_tris")
    print("-------------+---------------+-----------+----------+------------")
    print(f"{fmt_int(row['total_assets']):<13} | "
          f"{fmt_int(row['sum_tris']) or 'None':<13} | "
          f"{fmt_float(row['avg_tris']) or 'None':<9} | "
          f"{fmt_int(row['min_tris']) or 'None':<8} | "
          f"{fmt_int(row['max_tris']) or 'None':<10}")

    # by extension
    sql_ext = f"""
        SELECT ext,
               COUNT(*) AS n,
               SUM(tri_count) AS tris,
               AVG(NULLIF(tri_count,0)) AS avg_tris
        FROM assets
        {clause_where(where)}
        GROUP BY ext
        ORDER BY n DESC;
    """
    rows = conn.execute(sql_ext).fetchall()
    print("\n[COUNT BY EXT]")
    print("ext  | n     | tris          | avg_tris")
    print("-----+-------+---------------+----------")
    for r in rows:
        print(f"{r['ext']:<4} | {fmt_int(r['n']):<5} | "
              f"{fmt_int(r['tris']) or 'None':<13} | "
              f"{fmt_float(r['avg_tris']) or 'None':<8}")

def cmd_top_tris(conn, limit, where=None):
    sql = f"""
        SELECT id, ext, tri_count, path
        FROM assets
        {clause_where(where)}
        ORDER BY tri_count DESC
        LIMIT ?;
    """
    rows = conn.execute(sql, (limit,)).fetchall()
    print("[TOP BY TRI COUNT]")
    print("id    | ext  | tri_count   | path")
    print("------+----- +-------------+---------------------------------------------")
    for r in rows:
        print(f"{fmt_int(r['id']):>5} | {r['ext']:<4} | {fmt_int(r['tri_count']):>11} | {r['path']}")

def cmd_top_volume(conn, limit, where=None):
    sql = f"""
        SELECT id, ext,
               (bbox_x*bbox_y*bbox_z) AS volume,
               tri_count, bbox_x, bbox_y, bbox_z, path
        FROM assets
        {clause_where(where)}
        ORDER BY volume DESC
        LIMIT ?;
    """
    rows = conn.execute(sql, (limit,)).fetchall()
    print("[TOP BY BBOX VOLUME (bbox_x*bbox_y*bbox_z)]")
    print("id    | ext  | volume           | tri_count | bbox_x  | bbox_y  | bbox_z  | path")
    print("------+----- +------------------+-----------+---------+---------+---------+---------------------------------------------")
    for r in rows:
        print(f"{fmt_int(r['id']):>5} | {r['ext']:<4} | {fmt_float(r['volume'],2):>16} | "
              f"{fmt_int(r['tri_count']):>9} | {fmt_float(r['bbox_x']):>7} | {fmt_float(r['bbox_y']):>7} | "
              f"{fmt_float(r['bbox_z']):>7} | {r['path']}")

def cmd_ext_counts(conn, where=None):
    sql = f"""
        SELECT ext, COUNT(*) AS n
        FROM assets
        {clause_where(where)}
        GROUP BY ext
        ORDER BY n DESC;
    """
    rows = conn.execute(sql).fetchall()
    print("[ASSETS PER EXTENSION]")
    print("ext  | n")
    print("-----+----")
    for r in rows:
        print(f"{r['ext']:<4} | {fmt_int(r['n'])}")

def cmd_sample(conn, limit, where=None, order=None, desc=False):
    order_clause = ""
    if order:
        order_clause = f" ORDER BY {order} {'DESC' if desc else 'ASC'}"
    sql = f"""
        SELECT id, ext, tri_count, bbox_x, bbox_y, bbox_z, path
        FROM assets
        {clause_where(where)}
        {order_clause}
        LIMIT ?;
    """
    rows = conn.execute(sql, (limit,)).fetchall()
    print(f"[SAMPLE rows={limit} where={where or 'None'}]")
    print("id | ext  | tri_count | bbox_x | bbox_y | bbox_z | path")
    print("---+------+-----------+--------+--------+--------+-----------------------------------")
    for r in rows:
        print(f"{fmt_int(r['id']):>2} | {r['ext']:<4} | {fmt_int(r['tri_count']):>9} | "
              f"{fmt_float(r['bbox_x']):>6} | {fmt_float(r['bbox_y']):>6} | {fmt_float(r['bbox_z']):>6} | {r['path']}")

def main():
    ap = argparse.ArgumentParser(description="Quick queries for modelfinder.db")
    ap.add_argument("--db", default="db/modelfinder.db", help="Path to SQLite DB")
    subparsers = ap.add_subparsers(dest="cmd", required=True)

    p_stats = subparsers.add_parser("stats", help="Overview + counts per ext")
    p_stats.add_argument("--where", type=str, help="Optional SQL WHERE clause")

    # top-tris
    p_tris = subparsers.add_parser("top-tris")
    p_tris.add_argument("-n", "--limit", type=int, default=20)
    p_tris.add_argument("--where", type=str, help="Optional SQL WHERE clause")

    # top-volume
    p_vol = subparsers.add_parser("top-volume")
    p_vol.add_argument("-n", "--limit", type=int, default=20)
    p_vol.add_argument("--where", type=str, help="Optional SQL WHERE clause")

    p_ext = subparsers.add_parser("ext-counts", help="How many assets per extension")
    p_ext.add_argument("--where", type=str, help="Optional SQL WHERE clause")

    p_sample = subparsers.add_parser("sample", help="Sample rows (optionally with a WHERE)")
    p_sample.add_argument("-n","--limit", type=int, default=10)
    p_sample.add_argument("--where", type=str, help="SQL WHERE clause, e.g. \"ext='stl' AND tri_count>1e6\"")
    p_sample.add_argument("--order", type=str, help="Column to order by (e.g., 'tri_count', 'bbox_x')")
    p_sample.add_argument("--desc", action="store_true", help="Sort in descending order")

    args = ap.parse_args()
    conn = connect(args.db)
    try:
        if args.cmd == "stats":
            cmd_stats(conn, args.where)
        elif args.cmd == "top-tris":
            cmd_top_tris(conn, args.limit, args.where)
        elif args.cmd == "top-volume":
            cmd_top_volume(conn, args.limit, args.where)
        elif args.cmd == "ext-counts":
            cmd_ext_counts(conn, args.where)
        elif args.cmd == "sample":
            cmd_sample(conn, args.limit, args.where, args.order, args.desc)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
