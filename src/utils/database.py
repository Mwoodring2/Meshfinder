"""
Database utilities for ModelFinder with advanced filtering capabilities.
"""

import sqlite3
import csv
from dataclasses import dataclass
from typing import Optional, List, Any
from pathlib import Path


@dataclass
class Filters:
    """Filter parameters for database queries."""
    exts: Optional[List[str]] = None
    name_like: Optional[str] = None
    franchises: Optional[List[str]] = None  # tags contains any
    tris_min: Optional[int] = None
    tris_max: Optional[int] = None
    bbox_min_mm: Optional[float] = None
    sql_where: Optional[str] = None  # custom SQL WHERE clause (must be validated)
    page_size: int = 50


def fetch_rows(filters: Filters, page: int, page_size: int, db_path: Path) -> List[tuple]:
    """
    Fetch rows from the database with advanced filtering and pagination.
    
    Args:
        filters: Filter parameters
        page: Page number (1-based)
        page_size: Number of items per page
        db_path: Path to the SQLite database
        
    Returns:
        List of tuples containing (path, name, ext, size, mtime, tags, tris, dim_x, dim_y, dim_z, volume, watertight)
    """
    sql = "SELECT path,name,ext,size,mtime,tags,tri_count,bbox_x,bbox_y,bbox_z,volume,is_watertight FROM assets WHERE 1=1"
    args = []
    
    if filters.exts: 
        sql += f" AND ext IN ({','.join(['?']*len(filters.exts))})"
        args += [e.lower() for e in filters.exts]
    
    if filters.name_like:
        sql += " AND LOWER(name) LIKE ?"
        args += [f"%{filters.name_like.lower()}%"]
    
    if filters.franchises:  # tags contains any
        ors = " OR ".join(["LOWER(tags) LIKE ?"] * len(filters.franchises))
        sql += f" AND ({ors})"
        args += [f"%{f.lower()}%" for f in filters.franchises]
    
    if filters.tris_min is not None:
        sql += " AND COALESCE(tri_count,0) >= ?"
        args += [filters.tris_min]
    
    if filters.tris_max is not None:
        sql += " AND COALESCE(tri_count,0) <= ?"
        args += [filters.tris_max]
    
    if filters.bbox_min_mm is not None:
        sql += " AND (COALESCE(bbox_x,0)>=? AND COALESCE(bbox_y,0)>=? AND COALESCE(bbox_z,0)>=?)"
        args += [filters.bbox_min_mm]*3
    
    if filters.sql_where:
        sql += f" AND ({filters.sql_where})"  # only if validated first
    
    sql += " ORDER BY mtime DESC LIMIT ? OFFSET ?"
    args += [page_size, (page-1)*page_size]
    
    # Execute query
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    try:
        cur.execute(sql, args)
        rows = cur.fetchall()
        return rows
    finally:
        con.close()


def fetch_rows_simple(filters: Filters, page: int, page_size: int, db_path: Path) -> List[tuple]:
    """
    Simplified version that works with the basic 'files' table schema from main.py.
    
    Args:
        filters: Filter parameters
        page: Page number (1-based)
        page_size: Number of items per page
        db_path: Path to the SQLite database
        
    Returns:
        List of tuples containing (path, name, ext, size, mtime, tags, tris, dim_x, dim_y, dim_z, volume, watertight)
        Note: tris, dim_x, dim_y, dim_z, volume, watertight will be None for the simple schema
    """
    sql = "SELECT path,name,ext,size,mtime,tags FROM files WHERE 1=1"
    args = []
    
    if filters.exts: 
        sql += f" AND ext IN ({','.join(['?']*len(filters.exts))})"
        args += [e.lower() for e in filters.exts]
    
    if filters.name_like:
        sql += " AND LOWER(name) LIKE ?"
        args += [f"%{filters.name_like.lower()}%"]
    
    if filters.franchises:  # tags contains any
        ors = " OR ".join(["LOWER(tags) LIKE ?"] * len(filters.franchises))
        sql += f" AND ({ors})"
        args += [f"%{f.lower()}%" for f in filters.franchises]
    
    if filters.sql_where:
        sql += f" AND ({filters.sql_where})"  # only if validated first
    
    sql += " ORDER BY mtime DESC LIMIT ? OFFSET ?"
    args += [page_size, (page-1)*page_size]
    
    # Execute query
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    try:
        cur.execute(sql, args)
        rows = cur.fetchall()
        # Pad rows with None values for missing columns to match expected format
        padded_rows = []
        for row in rows:
            padded_row = row + (None, None, None, None, None, None)  # tris, dim_x, dim_y, dim_z, volume, watertight
            padded_rows.append(padded_row)
        return padded_rows
    finally:
        con.close()


def get_total_count(filters: Filters, db_path: Path, use_assets_table: bool = False) -> int:
    """
    Get total count of rows matching the filters.
    
    Args:
        filters: Filter parameters
        db_path: Path to the SQLite database
        use_assets_table: Whether to use the 'assets' table or 'files' table
        
    Returns:
        Total count of matching rows
    """
    table_name = "assets" if use_assets_table else "files"
    sql = f"SELECT COUNT(*) FROM {table_name} WHERE 1=1"
    args = []
    
    if filters.exts: 
        sql += f" AND ext IN ({','.join(['?']*len(filters.exts))})"
        args += [e.lower() for e in filters.exts]
    
    if filters.name_like:
        sql += " AND LOWER(name) LIKE ?"
        args += [f"%{filters.name_like.lower()}%"]
    
    if filters.franchises:  # tags contains any
        ors = " OR ".join(["LOWER(tags) LIKE ?"] * len(filters.franchises))
        sql += f" AND ({ors})"
        args += [f"%{f.lower()}%" for f in filters.franchises]
    
    if use_assets_table:
        if filters.tris_min is not None:
            sql += " AND COALESCE(tri_count,0) >= ?"
            args += [filters.tris_min]
        
        if filters.tris_max is not None:
            sql += " AND COALESCE(tri_count,0) <= ?"
            args += [filters.tris_max]
        
        if filters.bbox_min_mm is not None:
            sql += " AND (COALESCE(bbox_x,0)>=? AND COALESCE(bbox_y,0)>=? AND COALESCE(bbox_z,0)>=?)"
            args += [filters.bbox_min_mm]*3
    
    if filters.sql_where:
        sql += f" AND ({filters.sql_where})"
    
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    try:
        cur.execute(sql, args)
        return cur.fetchone()[0]
    finally:
        con.close()


def detect_schema_type(db_path: Path) -> str:
    """
    Detect whether the database uses the 'assets' table (advanced) or 'files' table (basic).
    
    Args:
        db_path: Path to the SQLite database
        
    Returns:
        'assets' if assets table exists, 'files' otherwise
    """
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    try:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='assets'")
        if cur.fetchone():
            return 'assets'
        else:
            return 'files'
    finally:
        con.close()


def export_to_csv(filters: Filters, output_path: Path, db_path: Path, use_assets_table: bool = False) -> int:
    """
    Export filtered data to CSV file.
    
    Args:
        filters: Filter parameters
        output_path: Path for the CSV output file
        db_path: Path to the SQLite database
        use_assets_table: Whether to use the 'assets' table or 'files' table
        
    Returns:
        Number of rows exported
    """
    # Get all matching rows (no pagination for export)
    if use_assets_table:
        rows = fetch_rows(filters, page=1, page_size=1000000, db_path=db_path)  # Large page size
    else:
        rows = fetch_rows_simple(filters, page=1, page_size=1000000, db_path=db_path)
    
    # CSV headers
    headers = ["Path", "Name", "Ext", "SizeMB", "Modified", "Tags", "Tris", "DimX", "DimY", "DimZ", "Vol", "Watertight"]
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for row in rows:
            # Extract data from row
            path = row[0] if len(row) > 0 else ""
            name = row[1] if len(row) > 1 else ""
            ext = row[2] if len(row) > 2 else ""
            size_bytes = row[3] if len(row) > 3 else 0
            mtime = row[4] if len(row) > 4 else 0
            tags = row[5] if len(row) > 5 else ""
            tris = row[6] if len(row) > 6 else None
            dim_x = row[7] if len(row) > 7 else None
            dim_y = row[8] if len(row) > 8 else None
            dim_z = row[9] if len(row) > 9 else None
            volume = row[10] if len(row) > 10 else None
            watertight = row[11] if len(row) > 11 else None
            
            # Convert size to MB
            size_mb = size_bytes / (1024 * 1024) if size_bytes else 0
            
            # Format modified time
            import time
            modified_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime)) if mtime else ""
            
            # Format watertight as Yes/No
            watertight_str = "Yes" if watertight else "No" if watertight is False else ""
            
            # Write row
            csv_row = [
                path,
                name,
                ext,
                f"{size_mb:.2f}",
                modified_str,
                tags,
                tris if tris is not None else "",
                f"{dim_x:.2f}" if dim_x is not None else "",
                f"{dim_y:.2f}" if dim_y is not None else "",
                f"{dim_z:.2f}" if dim_z is not None else "",
                f"{volume:.2f}" if volume is not None else "",
                watertight_str
            ]
            writer.writerow(csv_row)
    
    return len(rows)
