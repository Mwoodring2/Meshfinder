import sqlite3, json, time, os, sys

def ensure_cols(cur, table, cols):
    cur.execute(f"PRAGMA table_info({table})")
    existing = {r[1] for r in cur.fetchall()}
    for col, decl in cols.items():
        if col not in existing:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")

def main(db_path="db\\modelfinder.db"):
    if not os.path.exists(db_path):
        print(f"DB not found: {db_path}"); sys.exit(1)
    con = sqlite3.connect(db_path); cur = con.cursor()

    # 1) files table new fields
    ensure_cols(cur, "files", {
        "project_number": "TEXT",
        "project_name":   "TEXT",
        "part_name":      "TEXT",
        "type_conf":      "REAL",
        "project_conf":   "REAL",
        "sha256":         "TEXT",
        "status":         "TEXT"   # e.g., discovered|staged|migrated|quarantined
    })

    # 2) ops_log table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS ops_log(
      id INTEGER PRIMARY KEY,
      op TEXT,               -- scan|propose|rename|move|rollback
      before_path TEXT,
      after_path  TEXT,
      sha256 TEXT,
      user TEXT,
      detail_json TEXT,
      ts_utc TEXT
    )""")

    # 3) backfill defaults
    cur.execute("UPDATE files SET status = COALESCE(status,'discovered')")
    con.commit(); con.close()
    print("Migration complete.")

if __name__ == "__main__": main()

