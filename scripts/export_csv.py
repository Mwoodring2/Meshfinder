#!/usr/bin/env python3
"""
CSV Export Script for ModelFinder
Exports database data to CSV format with the exact structure you specified.
"""

import sys
import csv
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.database import Filters, fetch_rows_simple, fetch_rows, detect_schema_type
from main import DB_PATH

def export_current_rows_to_csv():
    """
    Export current rows to CSV using your exact format.
    This demonstrates the code snippet you provided.
    """
    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}")
        print("Please run the main application first to create the database.")
        return
    
    # Get current rows (using no filters to get all data)
    filters = Filters()
    schema_type = detect_schema_type(DB_PATH)
    
    if schema_type == 'assets':
        current_rows = fetch_rows(filters, page=1, page_size=1000000, db_path=DB_PATH)
    else:
        current_rows = fetch_rows_simple(filters, page=1, page_size=1000000, db_path=DB_PATH)
    
    # Your exact CSV export code
    with open("export.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Path","Name","Ext","SizeMB","Modified","Tags","Tris","DimX","DimY","DimZ","Vol","Watertight"])
        
        for r in current_rows:
            # Extract data from row
            path = r[0] if len(r) > 0 else ""
            name = r[1] if len(r) > 1 else ""
            ext = r[2] if len(r) > 2 else ""
            size_bytes = r[3] if len(r) > 3 else 0
            mtime = r[4] if len(r) > 4 else 0
            tags = r[5] if len(r) > 5 else ""
            tris = r[6] if len(r) > 6 else ""
            dim_x = r[7] if len(r) > 7 else ""
            dim_y = r[8] if len(r) > 8 else ""
            dim_z = r[9] if len(r) > 9 else ""
            volume = r[10] if len(r) > 10 else ""
            watertight = r[11] if len(r) > 11 else ""
            
            # Convert size to MB
            size_mb = size_bytes / (1024 * 1024) if size_bytes else 0
            
            # Format modified time
            modified_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime)) if mtime else ""
            
            # Format watertight
            watertight_str = "Yes" if watertight else "No" if watertight is False else ""
            
            # Write row using your exact format
            w.writerow([
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
            ])
    
    print(f"Exported {len(current_rows)} rows to export.csv")


def export_with_filters():
    """Export with specific filters applied."""
    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}")
        return
    
    # Example: Export only .stl files with more than 1000 triangles
    filters = Filters(
        exts=['.stl'],
        tris_min=1000
    )
    
    schema_type = detect_schema_type(DB_PATH)
    
    if schema_type == 'assets':
        current_rows = fetch_rows(filters, page=1, page_size=1000000, db_path=DB_PATH)
    else:
        current_rows = fetch_rows_simple(filters, page=1, page_size=1000000, db_path=DB_PATH)
    
    print(f"Found {len(current_rows)} .stl files with 1000+ triangles")
    
    # Export with your exact format
    with open("filtered_export.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Path","Name","Ext","SizeMB","Modified","Tags","Tris","DimX","DimY","DimZ","Vol","Watertight"])
        
        for r in current_rows:
            # Your exact row processing code
            w.writerow([
                r[0] if len(r) > 0 else "",  # Path
                r[1] if len(r) > 1 else "",  # Name
                r[2] if len(r) > 2 else "",  # Ext
                f"{(r[3] / (1024 * 1024)):.2f}" if len(r) > 3 and r[3] else "0.00",  # SizeMB
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r[4])) if len(r) > 4 and r[4] else "",  # Modified
                r[5] if len(r) > 5 else "",  # Tags
                r[6] if len(r) > 6 else "",  # Tris
                f"{r[7]:.2f}" if len(r) > 7 and r[7] is not None else "",  # DimX
                f"{r[8]:.2f}" if len(r) > 8 and r[8] is not None else "",  # DimY
                f"{r[9]:.2f}" if len(r) > 9 and r[9] is not None else "",  # DimZ
                f"{r[10]:.2f}" if len(r) > 10 and r[10] is not None else "",  # Vol
                "Yes" if len(r) > 11 and r[11] else "No" if len(r) > 11 and r[11] is False else ""  # Watertight
            ])
    
    print(f"Exported {len(current_rows)} filtered rows to filtered_export.csv")


def main():
    """Main function demonstrating CSV export."""
    print("ModelFinder CSV Export")
    print("=" * 30)
    
    try:
        print("\n1. Exporting all data...")
        export_current_rows_to_csv()
        
        print("\n2. Exporting filtered data (.stl files with 1000+ triangles)...")
        export_with_filters()
        
        print("\nExport completed successfully!")
        
    except Exception as e:
        print(f"Export failed: {e}")


if __name__ == "__main__":
    main()
