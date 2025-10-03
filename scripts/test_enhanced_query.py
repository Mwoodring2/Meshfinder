#!/usr/bin/env python3
"""
Test script for the enhanced database query functionality.
Demonstrates the new filtering capabilities.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.database import Filters, fetch_rows, fetch_rows_simple, detect_schema_type, get_total_count
from main import DB_PATH

def test_enhanced_queries():
    """Test the enhanced query functionality."""
    print("ModelFinder Enhanced Query Test")
    print("=" * 50)
    
    # Check if database exists
    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}")
        print("Please run the main application first to create the database.")
        return
    
    # Detect schema type
    schema_type = detect_schema_type(DB_PATH)
    print(f"Database schema type: {schema_type}")
    print()
    
    # Test 1: Basic query
    print("Test 1: Basic query (first 10 results)")
    filters = Filters()
    try:
        if schema_type == 'assets':
            rows = fetch_rows(filters, page=1, page_size=10, db_path=DB_PATH)
        else:
            rows = fetch_rows_simple(filters, page=1, page_size=10, db_path=DB_PATH)
        
        print(f"Found {len(rows)} results:")
        for i, row in enumerate(rows, 1):
            path, name, ext, size, mtime, tags = row[:6]
            print(f"  {i}. {name} ({ext}) - {size/1024/1024:.1f} MB")
            if len(row) > 6:
                tris = row[6]
                if tris:
                    print(f"     Triangles: {tris:,}")
    except Exception as e:
        print(f"Error: {e}")
    
    print()
    
    # Test 2: Filter by extension
    print("Test 2: Filter by .stl files")
    filters = Filters(exts=['.stl'])
    try:
        if schema_type == 'assets':
            rows = fetch_rows(filters, page=1, page_size=5, db_path=DB_PATH)
        else:
            rows = fetch_rows_simple(filters, page=1, page_size=5, db_path=DB_PATH)
        
        print(f"Found {len(rows)} .stl files:")
        for i, row in enumerate(rows, 1):
            path, name, ext, size, mtime, tags = row[:6]
            print(f"  {i}. {name} - {size/1024/1024:.1f} MB")
    except Exception as e:
        print(f"Error: {e}")
    
    print()
    
    # Test 3: Filter by name
    print("Test 3: Filter by name containing 'test'")
    filters = Filters(name_like='test')
    try:
        if schema_type == 'assets':
            rows = fetch_rows(filters, page=1, page_size=5, db_path=DB_PATH)
        else:
            rows = fetch_rows_simple(filters, page=1, page_size=5, db_path=DB_PATH)
        
        print(f"Found {len(rows)} files with 'test' in name:")
        for i, row in enumerate(rows, 1):
            path, name, ext, size, mtime, tags = row[:6]
            print(f"  {i}. {name} ({ext})")
    except Exception as e:
        print(f"Error: {e}")
    
    print()
    
    # Test 4: Triangle count filter (only works with assets table)
    if schema_type == 'assets':
        print("Test 4: Filter by triangle count (min 1000 triangles)")
        filters = Filters(tris_min=1000)
        try:
            rows = fetch_rows(filters, page=1, page_size=5, db_path=DB_PATH)
            print(f"Found {len(rows)} files with >= 1000 triangles:")
            for i, row in enumerate(rows, 1):
                path, name, ext, size, mtime, tags, tris = row[:7]
                if tris:
                    print(f"  {i}. {name} - {tris:,} triangles")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Test 4: Triangle count filtering requires 'assets' table schema")
    
    print()
    
    # Test 5: Total count
    print("Test 5: Total count of all files")
    filters = Filters()
    try:
        total = get_total_count(filters, db_path=DB_PATH, use_assets_table=(schema_type == 'assets'))
        print(f"Total files in database: {total}")
    except Exception as e:
        print(f"Error: {e}")
    
    print()
    print("Enhanced query test completed!")

if __name__ == "__main__":
    test_enhanced_queries()
