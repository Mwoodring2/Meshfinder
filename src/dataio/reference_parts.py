"""
Data access layer for project reference parts.

Provides functions to load and query reference parts from the database
for auto-suggestion and matching during file migration.
"""
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def load_reference_parts(db_path: str, project_number: str) -> Tuple[str, Dict[str, str]]:
    """
    Load reference parts for a given project.
    
    Args:
        db_path: Path to SQLite database
        project_number: Project number to look up
    
    Returns:
        Tuple of (project_name, parts_map) where:
        - project_name: The project name string
        - parts_map: Dictionary mapping normalized_part_name -> original_label
    
    Raises:
        FileNotFoundError: If database file doesn't exist
    
    Example:
        >>> project_name, parts = load_reference_parts("db/modelfinder.db", "300868")
        >>> project_name
        'superman_pf'
        >>> parts
        {'head': 'Head', 'left_arm': 'Left Arm', 'torso': 'Torso'}
    """
    if not db_path or not Path(db_path).exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    con = sqlite3.connect(db_path)
    cur = con.cursor()
    
    # Ensure table exists with simplified schema
    cur.execute("""
        CREATE TABLE IF NOT EXISTS project_reference_parts(
            id INTEGER PRIMARY KEY,
            project_number TEXT,
            project_name   TEXT,
            part_name      TEXT,
            original_label TEXT,
            UNIQUE(project_number, project_name, part_name) ON CONFLICT IGNORE
        )
    """)
    
    # Query parts for this project
    cur.execute("""
        SELECT project_name, part_name, original_label
        FROM project_reference_parts
        WHERE project_number = ?
    """, (project_number,))
    
    rows = cur.fetchall()
    con.close()

    if not rows:
        return "", {}

    # Assume same project_name for a single project_number
    project_name = rows[0][0] or ""
    
    # Build parts map: normalized part_name -> original_label
    parts_map = {
        (rows[i][1] or "").strip().lower(): rows[i][2] 
        for i in range(len(rows))
    }
    
    return project_name, parts_map


def get_all_projects(db_path: str) -> List[Dict[str, str]]:
    """
    Get list of all projects in reference table.
    
    Args:
        db_path: Path to SQLite database
    
    Returns:
        List of dictionaries with project_number and project_name
    """
    if not db_path or not Path(db_path).exists():
        return []
    
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    
    cur.execute("""
        SELECT DISTINCT project_number, project_name
        FROM project_reference_parts
        WHERE project_number IS NOT NULL
        ORDER BY project_number
    """)
    
    rows = cur.fetchall()
    con.close()
    
    return [
        {"project_number": row[0], "project_name": row[1] or ""}
        for row in rows
    ]


def search_part_names(db_path: str, project_number: str, query: str, limit: int = 10) -> List[Dict[str, str]]:
    """
    Search for part names within a project.
    
    Args:
        db_path: Path to SQLite database
        project_number: Project number to search within
        query: Search query (case-insensitive)
        limit: Maximum results to return
    
    Returns:
        List of dictionaries with part_name and original_label
    """
    if not db_path or not Path(db_path).exists():
        return []
    
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    
    # Search both normalized and original labels
    cur.execute("""
        SELECT part_name, original_label
        FROM project_reference_parts
        WHERE project_number = ? 
        AND (
            LOWER(part_name) LIKE ? 
            OR LOWER(original_label) LIKE ?
        )
        ORDER BY 
            CASE 
                WHEN LOWER(part_name) = LOWER(?) THEN 0
                WHEN LOWER(part_name) LIKE ? THEN 1
                ELSE 2
            END,
            part_name
        LIMIT ?
    """, (
        project_number,
        f"%{query.lower()}%",
        f"%{query.lower()}%",
        query.lower(),
        f"{query.lower()}%",
        limit
    ))
    
    rows = cur.fetchall()
    con.close()
    
    return [
        {"part_name": row[0], "original_label": row[1]}
        for row in rows
    ]


def insert_reference_part(
    db_path: str,
    project_number: str,
    project_name: str,
    part_name: str,
    original_label: str
) -> bool:
    """
    Insert a single reference part.
    
    Args:
        db_path: Path to SQLite database
        project_number: Project number
        project_name: Project name (slugified)
        part_name: Part name (normalized/slugified)
        original_label: Original part label from source
    
    Returns:
        True if inserted, False if already exists (UNIQUE constraint)
    """
    db_path_obj = Path(db_path)
    if not db_path_obj.exists():
        db_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    
    # Ensure table exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS project_reference_parts(
            id INTEGER PRIMARY KEY,
            project_number TEXT,
            project_name   TEXT,
            part_name      TEXT,
            original_label TEXT,
            UNIQUE(project_number, project_name, part_name) ON CONFLICT IGNORE
        )
    """)
    
    # Insert (will ignore if duplicate due to UNIQUE constraint)
    cur.execute("""
        INSERT INTO project_reference_parts
        (project_number, project_name, part_name, original_label)
        VALUES (?, ?, ?, ?)
    """, (project_number, project_name, part_name, original_label))
    
    rows_affected = cur.rowcount
    con.commit()
    con.close()
    
    return rows_affected > 0


def delete_project_references(db_path: str, project_number: str) -> int:
    """
    Delete all reference parts for a project.
    
    Args:
        db_path: Path to SQLite database
        project_number: Project number to delete
    
    Returns:
        Number of rows deleted
    """
    if not db_path or not Path(db_path).exists():
        return 0
    
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    
    cur.execute("""
        DELETE FROM project_reference_parts
        WHERE project_number = ?
    """, (project_number,))
    
    rows_deleted = cur.rowcount
    con.commit()
    con.close()
    
    return rows_deleted


def get_reference_stats(db_path: str) -> Dict[str, int]:
    """
    Get statistics about reference parts.
    
    Args:
        db_path: Path to SQLite database
    
    Returns:
        Dictionary with statistics
    """
    if not db_path or not Path(db_path).exists():
        return {"total_projects": 0, "total_parts": 0}
    
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    
    # Count unique projects
    cur.execute("SELECT COUNT(DISTINCT project_number) FROM project_reference_parts")
    total_projects = cur.fetchone()[0]
    
    # Count total parts
    cur.execute("SELECT COUNT(*) FROM project_reference_parts")
    total_parts = cur.fetchone()[0]
    
    con.close()
    
    return {
        "total_projects": total_projects,
        "total_parts": total_parts,
        "avg_parts_per_project": total_parts / total_projects if total_projects > 0 else 0
    }

