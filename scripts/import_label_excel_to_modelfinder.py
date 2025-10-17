"""
Import Excel label files to ModelFinder CSV format.

Converts project Excel files with part labels into CSV format compatible with ModelFinder.
Extracts project number, part names, and metadata for bulk import.

Usage:
    python scripts/import_label_excel_to_modelfinder.py --in "E:/path/300868 Superman PF.xlsx" --out "E:/path/300868_parts.csv"
"""
import argparse
import sys
from pathlib import Path

# Try to import required libraries
try:
    import pandas as pd
except ImportError:
    print("Error: pandas not installed. Run: pip install pandas openpyxl")
    sys.exit(1)

try:
    import openpyxl
except ImportError:
    print("Error: openpyxl not installed. Run: pip install openpyxl")
    sys.exit(1)

# Import ModelFinder utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.naming import extract_project_number, slug


def extract_project_info(filename: str) -> dict:
    """
    Extract project number and name from filename.
    
    Args:
        filename: Excel filename (e.g., "300868 Superman PF.xlsx")
    
    Returns:
        Dict with project_number and project_name
    """
    stem = Path(filename).stem
    
    # Extract project number
    project_num = extract_project_number(stem)
    
    # Get project name (everything after project number)
    if project_num:
        # Remove project number from stem to get name
        name_part = stem.replace(project_num, "").strip()
    else:
        name_part = stem
        project_num = "unknown"
    
    return {
        "project_number": project_num,
        "project_name": name_part
    }


def read_excel_labels(excel_path: str) -> list[dict]:
    """
    Read part labels from Excel file.
    
    Supports various Excel formats:
    - Single column with part names
    - Multi-column with headers (Part, Description, Quantity, etc.)
    
    Args:
        excel_path: Path to Excel file
    
    Returns:
        List of dictionaries with part data
    """
    try:
        # Try to read Excel file
        df = pd.read_excel(excel_path, sheet_name=0)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        sys.exit(1)
    
    parts = []
    
    # Try to identify part name column
    part_col = None
    desc_col = None
    qty_col = None
    
    # Look for common column names (case-insensitive)
    for col in df.columns:
        col_lower = str(col).lower()
        if any(x in col_lower for x in ['part', 'name', 'component']):
            part_col = col
        elif any(x in col_lower for x in ['desc', 'description', 'note']):
            desc_col = col
        elif any(x in col_lower for x in ['qty', 'quantity', 'count', 'number']):
            qty_col = col
    
    # If no part column found, use first column
    if part_col is None:
        part_col = df.columns[0]
    
    print(f"Using column '{part_col}' as part name")
    if desc_col:
        print(f"Using column '{desc_col}' as description")
    if qty_col:
        print(f"Using column '{qty_col}' as quantity")
    
    # Extract parts
    for idx, row in df.iterrows():
        part_name = row[part_col]
        
        # Skip empty rows
        if pd.isna(part_name) or str(part_name).strip() == "":
            continue
        
        # Clean up part name
        part_name = str(part_name).strip()
        
        # Get optional fields
        description = str(row[desc_col]).strip() if desc_col and not pd.isna(row.get(desc_col)) else ""
        quantity = int(row[qty_col]) if qty_col and not pd.isna(row.get(qty_col)) else 1
        
        parts.append({
            "part_name": part_name,
            "description": description,
            "quantity": quantity,
            "row_num": idx + 2  # +2 for Excel row (1-indexed + header)
        })
    
    return parts


def export_to_csv(parts: list[dict], project_info: dict, output_path: str):
    """
    Export parts to ModelFinder-compatible CSV.
    
    Args:
        parts: List of part dictionaries
        project_info: Project metadata
        output_path: Output CSV path
    """
    rows = []
    
    for part in parts:
        # Generate slugified part name
        part_slug = slug(part["part_name"])
        
        # Build tags from description
        tags = []
        if part["description"]:
            tags.append(part["description"])
        if part["quantity"] > 1:
            tags.append(f"qty_{part['quantity']}")
        
        rows.append({
            "project_number": project_info["project_number"],
            "project_name": slug(project_info["project_name"]),
            "part_name": part_slug,
            "part_name_original": part["part_name"],
            "description": part["description"],
            "quantity": part["quantity"],
            "tags": ", ".join(tags),
            "status": "labeled",
            "source_row": part["row_num"]
        })
    
    # Create DataFrame and export
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8")
    
    print(f"\nExported {len(rows)} parts to: {output_path}")
    print(f"Project: {project_info['project_number']} - {project_info['project_name']}")


def upsert_to_db(parts: list[dict], project_info: dict, db_path: str, table_name: str = "project_reference_parts"):
    """
    Upsert parts to SQLite database for auto-suggestion.
    
    Args:
        parts: List of part dictionaries
        project_info: Project metadata
        db_path: Path to SQLite database
        table_name: Table name for reference parts
    """
    import sqlite3
    
    db_path_obj = Path(db_path)
    if not db_path_obj.exists():
        print(f"Creating new database: {db_path}")
        db_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    
    # Create table if not exists
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_number TEXT NOT NULL,
            project_name TEXT,
            part_name TEXT NOT NULL,
            part_name_original TEXT,
            description TEXT,
            quantity INTEGER DEFAULT 1,
            tags TEXT,
            source_file TEXT,
            source_row INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(project_number, part_name)
        )
    """)
    
    # Create index for faster lookups
    cur.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{table_name}_project 
        ON {table_name}(project_number)
    """)
    
    # Insert/update parts
    inserted = 0
    updated = 0
    
    for part in parts:
        part_slug = slug(part["part_name"])
        tags = []
        if part["description"]:
            tags.append(part["description"])
        if part["quantity"] > 1:
            tags.append(f"qty_{part['quantity']}")
        
        # Check if exists
        cur.execute(
            f"SELECT id FROM {table_name} WHERE project_number=? AND part_name=?",
            (project_info["project_number"], part_slug)
        )
        exists = cur.fetchone()
        
        if exists:
            # Update
            cur.execute(f"""
                UPDATE {table_name} 
                SET part_name_original=?, description=?, quantity=?, tags=?, source_row=?
                WHERE project_number=? AND part_name=?
            """, (
                part["part_name"],
                part["description"],
                part["quantity"],
                ", ".join(tags),
                part["row_num"],
                project_info["project_number"],
                part_slug
            ))
            updated += 1
        else:
            # Insert
            cur.execute(f"""
                INSERT INTO {table_name} 
                (project_number, project_name, part_name, part_name_original, 
                 description, quantity, tags, source_row)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                project_info["project_number"],
                slug(project_info["project_name"]),
                part_slug,
                part["part_name"],
                part["description"],
                part["quantity"],
                ", ".join(tags),
                part["row_num"]
            ))
            inserted += 1
    
    con.commit()
    con.close()
    
    print(f"\nDatabase updated: {db_path}")
    print(f"  Inserted: {inserted} new parts")
    print(f"  Updated: {updated} existing parts")
    print(f"  Table: {table_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Import Excel label files to ModelFinder CSV format"
    )
    parser.add_argument(
        "--in", 
        dest="input_file",
        required=True,
        help="Input Excel file path (e.g., '300868 Superman PF.xlsx')"
    )
    parser.add_argument(
        "--out",
        dest="output_file",
        required=False,
        help="Output CSV file path (e.g., '300868_parts.csv')"
    )
    parser.add_argument(
        "--db",
        dest="db_path",
        help="SQLite database path for direct upsert"
    )
    parser.add_argument(
        "--table",
        dest="table_name",
        default="project_reference_parts",
        help="Database table name (default: project_reference_parts)"
    )
    parser.add_argument(
        "--project-num",
        dest="project_num",
        help="Override project number (auto-detected from filename if not provided)"
    )
    parser.add_argument(
        "--project-name",
        dest="project_name",
        help="Override project name (auto-detected from filename if not provided)"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview data without writing to file"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Extract project info
    project_info = extract_project_info(input_path.name)
    
    # Override if provided
    if args.project_num:
        project_info["project_number"] = args.project_num
    if args.project_name:
        project_info["project_name"] = args.project_name
    
    print(f"Processing: {input_path.name}")
    print(f"Project Number: {project_info['project_number']}")
    print(f"Project Name: {project_info['project_name']}")
    print()
    
    # Read Excel labels
    parts = read_excel_labels(str(input_path))
    
    if not parts:
        print("Error: No parts found in Excel file")
        sys.exit(1)
    
    # Preview mode
    if args.preview:
        print("\n=== PREVIEW ===")
        print(f"Found {len(parts)} parts:")
        for i, part in enumerate(parts[:10], 1):  # Show first 10
            print(f"  {i}. {part['part_name']} -> {slug(part['part_name'])}")
            if part['description']:
                print(f"     Description: {part['description']}")
            if part['quantity'] > 1:
                print(f"     Quantity: {part['quantity']}")
        if len(parts) > 10:
            print(f"  ... and {len(parts) - 10} more")
        print("\nUse without --preview to export to CSV or database")
        sys.exit(0)
    
    # Export to CSV (if output file specified)
    if args.output_file:
        export_to_csv(parts, project_info, args.output_file)
    
    # Upsert to database (if db path specified)
    if args.db_path:
        upsert_to_db(parts, project_info, args.db_path, args.table_name)
    
    # Error if neither output nor db specified
    if not args.output_file and not args.db_path:
        print("\nError: Must specify either --out (CSV) or --db (database) or both")
        sys.exit(1)
    
    # Show statistics
    print("\n=== Statistics ===")
    print(f"Total parts: {len(parts)}")
    print(f"Parts with descriptions: {sum(1 for p in parts if p['description'])}")
    print(f"Parts with qty > 1: {sum(1 for p in parts if p['quantity'] > 1)}")
    print(f"Total quantity: {sum(p['quantity'] for p in parts)}")


if __name__ == "__main__":
    main()
