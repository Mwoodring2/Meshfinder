#!/usr/bin/env python3
"""
Example usage of the enhanced fetch_rows function.
This demonstrates the exact function signature you provided.
"""

import sys
from pathlib import Path
from typing import List

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.database import Filters

def fetch_rows(filters: Filters, page: int, page_size: int) -> List[tuple]:
    """
    Example implementation of your fetch_rows function.
    
    Args:
        filters: Filter parameters object
        page: Page number (1-based)
        page_size: Number of items per page
        
    Returns:
        List of tuples containing (path, name, ext, size, mtime, tags, tris, dim_x, dim_y, dim_z, volume, watertight)
    """
    # This is a simplified version - in practice you'd connect to your actual database
    # For demonstration, we'll return some sample data
    
    # Your original SQL construction logic:
    sql = "SELECT path,name,ext,size,mtime,tags,tris,dim_x,dim_y,dim_z,volume,watertight FROM files WHERE 1=1"
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
        sql += " AND COALESCE(tris,0) >= ?"
        args += [filters.tris_min]
    
    if filters.tris_max is not None:
        sql += " AND COALESCE(tris,0) <= ?"
        args += [filters.tris_max]
    
    if filters.bbox_min_mm is not None:
        sql += " AND (COALESCE(dim_x,0)>=? AND COALESCE(dim_y,0)>=? AND COALESCE(dim_z,0)>=?)"
        args += [filters.bbox_min_mm]*3
    
    if filters.sql_where:
        sql += f" AND ({filters.sql_where})"  # only if validated first
    
    sql += " ORDER BY mtime DESC LIMIT ? OFFSET ?"
    args += [page_size, (page-1)*page_size]
    
    # In a real implementation, you would:
    # 1. Connect to your database
    # 2. Execute the SQL query with the args
    # 3. Return the fetched rows
    
    print(f"Generated SQL: {sql}")
    print(f"Generated args: {args}")
    
    # Return sample data for demonstration
    sample_data = [
        ("/path/to/model1.stl", "model1", ".stl", 1024000, 1234567890.0, "character,game", 5000, 100.0, 50.0, 25.0, 125000.0, True),
        ("/path/to/model2.obj", "model2", ".obj", 2048000, 1234567891.0, "furniture,modern", 8000, 200.0, 100.0, 50.0, 1000000.0, False),
    ]
    
    return sample_data


def example_usage():
    """Demonstrate various ways to use the fetch_rows function."""
    print("Enhanced fetch_rows Function Examples")
    print("=" * 50)
    
    # Example 1: Basic query with pagination
    print("\n1. Basic query - first page, 10 items per page:")
    filters = Filters(page_size=10)
    results = fetch_rows(filters, page=1, page_size=10)
    print(f"   Results: {len(results)} items")
    
    # Example 2: Filter by file extensions
    print("\n2. Filter by extensions (.stl and .obj):")
    filters = Filters(exts=['.stl', '.obj'], page_size=20)
    results = fetch_rows(filters, page=1, page_size=20)
    print(f"   Results: {len(results)} items")
    
    # Example 3: Search by name
    print("\n3. Search by name containing 'model':")
    filters = Filters(name_like='model', page_size=15)
    results = fetch_rows(filters, page=1, page_size=15)
    print(f"   Results: {len(results)} items")
    
    # Example 4: Filter by tags/franchises
    print("\n4. Filter by tags containing 'character' or 'game':")
    filters = Filters(franchises=['character', 'game'], page_size=25)
    results = fetch_rows(filters, page=1, page_size=25)
    print(f"   Results: {len(results)} items")
    
    # Example 5: Triangle count filtering
    print("\n5. Filter by triangle count (min 1000, max 10000):")
    filters = Filters(tris_min=1000, tris_max=10000, page_size=30)
    results = fetch_rows(filters, page=1, page_size=30)
    print(f"   Results: {len(results)} items")
    
    # Example 6: Bounding box filtering
    print("\n6. Filter by minimum bounding box size (10mm):")
    filters = Filters(bbox_min_mm=10.0, page_size=40)
    results = fetch_rows(filters, page=1, page_size=40)
    print(f"   Results: {len(results)} items")
    
    # Example 7: Custom SQL WHERE clause
    print("\n7. Custom SQL WHERE clause:")
    filters = Filters(sql_where="size > 1000000 AND watertight = 1", page_size=50)
    results = fetch_rows(filters, page=1, page_size=50)
    print(f"   Results: {len(results)} items")
    
    # Example 8: Complex combined filters
    print("\n8. Complex combined filters:")
    filters = Filters(
        exts=['.stl'],
        name_like='character',
        franchises=['game'],
        tris_min=5000,
        tris_max=50000,
        bbox_min_mm=5.0,
        page_size=100
    )
    results = fetch_rows(filters, page=2, page_size=100)  # Second page
    print(f"   Results: {len(results)} items (page 2)")
    
    # Example 9: Pagination demonstration
    print("\n9. Pagination example (pages 1-3):")
    filters = Filters(page_size=5)
    for page_num in range(1, 4):
        results = fetch_rows(filters, page=page_num, page_size=5)
        print(f"   Page {page_num}: {len(results)} items")
    
    print("\nExample usage completed!")


if __name__ == "__main__":
    example_usage()
