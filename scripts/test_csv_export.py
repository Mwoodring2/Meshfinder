#!/usr/bin/env python3
"""
Test script for CSV export functionality.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.database import Filters, export_to_csv, detect_schema_type
from main import DB_PATH

def test_csv_export():
    """Test the CSV export functionality."""
    print("Testing CSV Export Functionality")
    print("=" * 40)
    
    # Check if database exists
    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}")
        print("Please run the main application first to create the database.")
        return False
    
    # Detect schema type
    schema_type = detect_schema_type(DB_PATH)
    print(f"Database schema type: {schema_type}")
    
    # Test 1: Export all data
    print("\n1. Testing export of all data...")
    try:
        filters = Filters()
        output_path = Path("test_export_all.csv")
        row_count = export_to_csv(filters, output_path, DB_PATH, use_assets_table=(schema_type == 'assets'))
        print(f"   ✓ Exported {row_count} rows to {output_path}")
        
        # Check if file was created and has content
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"   ✓ File created successfully ({output_path.stat().st_size} bytes)")
        else:
            print("   ✗ File was not created or is empty")
            return False
            
    except Exception as e:
        print(f"   ✗ Export failed: {e}")
        return False
    
    # Test 2: Export with filters
    print("\n2. Testing export with filters (.stl files only)...")
    try:
        filters = Filters(exts=['.stl'])
        output_path = Path("test_export_stl.csv")
        row_count = export_to_csv(filters, output_path, DB_PATH, use_assets_table=(schema_type == 'assets'))
        print(f"   ✓ Exported {row_count} .stl files to {output_path}")
        
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"   ✓ Filtered file created successfully ({output_path.stat().st_size} bytes)")
        else:
            print("   ✗ Filtered file was not created or is empty")
            return False
            
    except Exception as e:
        print(f"   ✗ Filtered export failed: {e}")
        return False
    
    # Test 3: Export with name filter
    print("\n3. Testing export with name filter...")
    try:
        filters = Filters(name_like='test')
        output_path = Path("test_export_name.csv")
        row_count = export_to_csv(filters, output_path, DB_PATH, use_assets_table=(schema_type == 'assets'))
        print(f"   ✓ Exported {row_count} files with 'test' in name to {output_path}")
        
    except Exception as e:
        print(f"   ✗ Name filter export failed: {e}")
        return False
    
    # Test 4: Check CSV format
    print("\n4. Checking CSV format...")
    try:
        with open("test_export_all.csv", "r", encoding="utf-8") as f:
            lines = f.readlines()
            if len(lines) > 0:
                header = lines[0].strip()
                expected_header = "Path,Name,Ext,SizeMB,Modified,Tags,Tris,DimX,DimY,DimZ,Vol,Watertight"
                if header == expected_header:
                    print(f"   ✓ CSV header is correct: {header}")
                else:
                    print(f"   ✗ CSV header mismatch. Expected: {expected_header}")
                    print(f"   Got: {header}")
                    return False
                
                if len(lines) > 1:
                    print(f"   ✓ CSV has {len(lines)-1} data rows")
                else:
                    print("   ⚠ CSV has no data rows (only header)")
            else:
                print("   ✗ CSV file is empty")
                return False
                
    except Exception as e:
        print(f"   ✗ CSV format check failed: {e}")
        return False
    
    print("\n✅ All CSV export tests passed!")
    return True

def cleanup_test_files():
    """Clean up test CSV files."""
    test_files = [
        "test_export_all.csv",
        "test_export_stl.csv", 
        "test_export_name.csv"
    ]
    
    for file_path in test_files:
        path = Path(file_path)
        if path.exists():
            path.unlink()
            print(f"Cleaned up {file_path}")

if __name__ == "__main__":
    success = test_csv_export()
    
    if success:
        print("\nCleaning up test files...")
        cleanup_test_files()
        print("Test completed successfully!")
    else:
        print("\nSome tests failed. Check the output above.")
        sys.exit(1)
