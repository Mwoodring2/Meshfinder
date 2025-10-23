# app/dataio/db.py  (add these functions + call ensure_user_corrections() at startup)

import sqlite3, datetime, os
from typing import Dict, Any, Iterable, List, Tuple

DB_PATH = "db/modelfinder.db"

def _con():
    return sqlite3.connect(DB_PATH)

def ensure_user_corrections():
    con = _con(); cur = con.cursor()
    
    # Drop and recreate table to ensure correct schema
    cur.execute("DROP TABLE IF EXISTS user_corrections")
    cur.execute("""
    CREATE TABLE user_corrections(
      id INTEGER PRIMARY KEY,
      file_path TEXT,
      old_name TEXT,
      new_name TEXT,
      project_number TEXT,
      part_type TEXT,
      laterality TEXT,
      confidence REAL,
      corrected_utc TEXT DEFAULT CURRENT_TIMESTAMP,
      used_for_training INTEGER DEFAULT 0
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

def update_file_record(old_path: str, new_path: str):
    """Update file record in database after migration/rename"""
    con = _con(); cur = con.cursor()
    cur.execute("""
        UPDATE files 
        SET path = ?, migration_dest = ?, migration_status = 'migrated'
        WHERE path = ?
    """, (new_path, new_path, old_path))
    con.commit(); con.close()

def update_proposal(src_path: str, fields: dict) -> None:
    """Update file record with proposal data."""
    if not src_path:
        return
    
    con = _con()
    cur = con.cursor()
    
    try:
        cur.execute("""
            UPDATE files
            SET project_number = ?,
                project_name   = ?,
                part_name      = ?,
                type_conf      = ?,
                proposed_name  = ?,
                status         = COALESCE(status, 'discovered')
            WHERE path = ?
        """, (
            fields.get("project_number"),
            fields.get("project_name"),
            fields.get("part_name"),
            float(fields.get("conf", 0.0)),
            fields.get("proposed_name"),
            src_path
        ))
        con.commit()
    finally:
        con.close()

def batch_update_proposals(proposals: List[Dict[str, Any]]) -> int:
    """Batch update proposals in database"""
    if not proposals:
        return 0
    
    con = _con(); cur = con.cursor()
    updated = 0
    
    for proposal in proposals:
        try:
            cur.execute("""
                UPDATE files 
                SET project_number = ?, project_name = ?, part_name = ?, 
                    proposed_name = ?, type_conf = ?
                WHERE path = ?
            """, (
                proposal.get('project_number'),
                proposal.get('project_name'),
                proposal.get('part_name'),
                proposal.get('proposed_name'),
                proposal.get('conf'),
                proposal.get('from')
            ))
            updated += cur.rowcount
        except Exception as e:
            print(f"Failed to update proposal for {proposal.get('from')}: {e}")
    
    con.commit(); con.close()
    return updated

def log_op(operation: str, source_path: str, dest_path: str = None, 
           details: str = "", user: str = None) -> None:
    """Log a file operation to the database"""
    con = _con(); cur = con.cursor()
    
    # Create operations log table if it doesn't exist
    cur.execute("""
        CREATE TABLE IF NOT EXISTS operations_log(
            id INTEGER PRIMARY KEY,
            operation TEXT,
            source_path TEXT,
            dest_path TEXT,
            details TEXT,
            user TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cur.execute("""
        INSERT INTO operations_log(operation, source_path, dest_path, details, user)
        VALUES(?, ?, ?, ?, ?)
    """, (operation, source_path, dest_path or "", details, user or ""))
    
    con.commit(); con.close()

def get_file_records(status: str = None, limit: int = None) -> List[Dict[str, Any]]:
    """Get file records from database, optionally filtered by status"""
    con = _con(); cur = con.cursor()
    
    if status:
        query = "SELECT * FROM files WHERE migration_status = ? ORDER BY path"
        params = [status]
        if limit:
            query += f" LIMIT {int(limit)}"
        rows = cur.execute(query, params).fetchall()
    else:
        query = "SELECT * FROM files ORDER BY path"
        if limit:
            query += f" LIMIT {int(limit)}"
        rows = cur.execute(query).fetchall()
    
    # Get column names
    cols = [desc[0] for desc in cur.description]
    con.close()
    
    return [dict(zip(cols, row)) for row in rows]

# Ensure user_corrections table exists when module is imported
ensure_user_corrections()