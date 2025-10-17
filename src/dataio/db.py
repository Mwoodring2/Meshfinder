# app/dataio/db.py  (add these functions + call ensure_user_corrections() at startup)

import sqlite3, datetime, os
from typing import Dict, Any, Iterable, List, Tuple

DB_PATH = "db/modelfinder.db"

def _con():
    return sqlite3.connect(DB_PATH)

def ensure_user_corrections():
    con = _con(); cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_corrections(
      id INTEGER PRIMARY KEY,
      file_path TEXT,
      old_name TEXT,
      new_name TEXT,
      project_number TEXT,
      part_type TEXT,
      laterality TEXT,
      confidence REAL,
      corrected_utc TEXT
    )""")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_uc_path ON user_corrections(file_path)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_uc_time ON user_corrections(corrected_utc)")
    con.commit(); con.close()

def add_user_correction(
    file_path: str,
    old_name: str | None,
    new_name: str,
    project_number: str | None,
    part_type: str | None,
    laterality: str | None,
    confidence: float | None
) -> None:
    con = _con(); cur = con.cursor()
    cur.execute("""
      INSERT INTO user_corrections(file_path, old_name, new_name, project_number, part_type, laterality, confidence, corrected_utc)
      VALUES(?,?,?,?,?,?,?,datetime('now'))
    """, (file_path, old_name, new_name, project_number, part_type, laterality, confidence))
    con.commit(); con.close()

def get_user_corrections(limit: int | None = None) -> List[Dict[str, Any]]:
    con = _con(); cur = con.cursor()
    q = "SELECT id,file_path,old_name,new_name,project_number,part_type,laterality,confidence,corrected_utc FROM user_corrections ORDER BY id ASC"
    if limit: q += f" LIMIT {int(limit)}"
    rows = cur.execute(q).fetchall()
    con.close()
    keys = ["id","file_path","old_name","new_name","project_number","part_type","laterality","confidence","corrected_utc"]
    return [dict(zip(keys, r)) for r in rows]

def clear_user_corrections(ids: Iterable[int]) -> int:
    ids = list(ids)
    if not ids: return 0
    con = _con(); cur = con.cursor()
    qmarks = ",".join("?" * len(ids))
    cur.execute(f"DELETE FROM user_corrections WHERE id IN ({qmarks})", ids)
    n = cur.rowcount
    con.commit(); con.close()
    return n