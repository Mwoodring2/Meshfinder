"""
Enhanced proposal generation with reference part lookup.

Integrates with project_reference_parts table to auto-suggest metadata
based on previously imported Excel labels.
"""
import sqlite3
from pathlib import Path
from typing import Optional, Dict, List
from ..utils.naming import canonical_name, extract_project_number, slug
from ..ml.project_extractor import propose_fields
from ..ml import DEFAULT_THRESHOLD


def get_reference_parts(db_path: str, project_number: str, table_name: str = "project_reference_parts") -> List[Dict]:
    """
    Get reference parts for a project from database.
    
    Args:
        db_path: Path to SQLite database
        project_number: Project number to look up
        table_name: Table name for reference parts
    
    Returns:
        List of reference part dictionaries
    """
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        
        cur.execute(f"""
            SELECT part_name, part_name_original, description, quantity, tags
            FROM {table_name}
            WHERE project_number = ?
            ORDER BY part_name
        """, (project_number,))
        
        rows = cur.fetchall()
        con.close()
        
        return [
            {
                "part_name": row[0],
                "part_name_original": row[1],
                "description": row[2] or "",
                "quantity": row[3] or 1,
                "tags": row[4] or ""
            }
            for row in rows
        ]
    except Exception as e:
        print(f"Warning: Could not read reference parts: {e}")
        return []


def get_project_info(db_path: str, project_number: str, table_name: str = "project_reference_parts") -> Optional[Dict]:
    """
    Get project info from reference table.
    
    Args:
        db_path: Path to SQLite database
        project_number: Project number to look up
        table_name: Table name for reference parts
    
    Returns:
        Dict with project_number and project_name, or None if not found
    """
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        
        cur.execute(f"""
            SELECT DISTINCT project_number, project_name
            FROM {table_name}
            WHERE project_number = ?
            LIMIT 1
        """, (project_number,))
        
        row = cur.fetchone()
        con.close()
        
        if row:
            return {
                "project_number": row[0],
                "project_name": row[1] or "unknown"
            }
        return None
    except Exception as e:
        print(f"Warning: Could not read project info: {e}")
        return None


def match_part_name(filename: str, reference_parts: List[Dict], threshold: float = 0.7) -> Optional[Dict]:
    """
    Match filename to reference part using fuzzy matching.
    
    Args:
        filename: File name (stem) to match
        reference_parts: List of reference part dicts
        threshold: Similarity threshold (0-1)
    
    Returns:
        Best matching reference part dict, or None
    """
    if not reference_parts:
        return None
    
    filename_lower = filename.lower()
    filename_slug = slug(filename)
    
    best_match = None
    best_score = 0
    
    for ref_part in reference_parts:
        part_name = ref_part["part_name"]
        part_orig = ref_part["part_name_original"]
        
        # Exact match (slugified)
        if part_name == filename_slug:
            return ref_part
        
        # Contains match
        if part_name in filename_slug or filename_slug in part_name:
            score = 0.9
            if score > best_score:
                best_score = score
                best_match = ref_part
        
        # Word overlap (simple fuzzy)
        filename_words = set(filename_slug.split("_"))
        part_words = set(part_name.split("_"))
        overlap = len(filename_words & part_words)
        total = len(filename_words | part_words)
        
        if total > 0:
            score = overlap / total
            if score > best_score and score >= threshold:
                best_score = score
                best_match = ref_part
    
    return best_match if best_score >= threshold else None


def build_proposals_enhanced(
    rows: list,
    dest_root: str,
    db_path: Optional[str] = None,
    table_name: str = "project_reference_parts"
) -> list[dict]:
    """
    Build migration proposals with reference part lookup.
    
    Args:
        rows: Iterable of dictionaries with file metadata
        dest_root: Destination root directory for migration
        db_path: Optional path to database with reference parts
        table_name: Table name for reference parts
    
    Returns:
        List of proposal dictionaries with enhanced metadata
    
    Example:
        >>> rows = [{
        ...     "path": "E:/Models/300868/head.stl",
        ...     "name": "head.stl",
        ...     "ext": ".stl",
        ...     "mesh_stats": {},
        ...     "tags": ""
        ... }]
        >>> plans = build_proposals_enhanced(
        ...     rows,
        ...     dest_root="E:/Organized",
        ...     db_path="E:/db/modelfinder.db"
        ... )
        >>> plans[0]['to']
        'E:/Organized/300868/300868_superman_pf_head.stl'
        >>> plans[0]['fields']['reference_match']
        True
    """
    plans = []
    
    # Cache for reference parts by project
    reference_cache = {}
    project_info_cache = {}
    
    for r in rows:
        # Build context from name, tags, and parent folder
        path_obj = Path(r["path"])
        ctx = " ".join([
            r["name"],
            r.get("tags", ""),
            path_obj.parent.name
        ])
        
        # Extract project number from path/context
        project_num = extract_project_number(ctx)
        
        # Try to get project info from reference table
        if db_path and project_num and project_num not in project_info_cache:
            project_info = get_project_info(db_path, project_num, table_name)
            if project_info:
                project_info_cache[project_num] = project_info
        
        # Get reference parts for this project
        reference_parts = []
        if db_path and project_num:
            if project_num not in reference_cache:
                reference_cache[project_num] = get_reference_parts(db_path, project_num, table_name)
            reference_parts = reference_cache[project_num]
        
        # Use ML/rules to propose fields
        fields = propose_fields(
            r["path"],
            r.get("mesh_stats", {}),
            ctx
        )
        
        # Override with reference data if available
        reference_match = None
        if project_num and project_num in project_info_cache:
            # Use reference project info
            proj_info = project_info_cache[project_num]
            fields["project_number"] = proj_info["project_number"]
            fields["project_name"] = proj_info["project_name"]
            
            # Try to match part name
            filename_stem = Path(r["name"]).stem
            reference_match = match_part_name(filename_stem, reference_parts)
            
            if reference_match:
                # Use reference part name
                fields["part_name"] = reference_match["part_name"]
                fields["type_conf"] = 0.95  # High confidence for reference match
                fields["needs_review"] = False
                fields["reference_match"] = True
                fields["reference_source"] = "excel_import"
            else:
                # No match found - needs review
                fields["needs_review"] = True
                fields["reference_match"] = False
                fields["reference_available"] = len(reference_parts)
        
        # Generate canonical filename
        ext = r["ext"].lstrip(".")
        new_name = canonical_name(
            fields["project_number"],
            fields["project_name"],
            fields["part_name"],
            ext
        )
        
        # Determine target directory (group by project number)
        target_dir = Path(dest_root) / fields["project_number"]
        
        # Build plan
        plan = {
            "from": r["path"],
            "to": str(target_dir / new_name),
            "fields": fields,
            "needs_review": fields.get("needs_review", False),
            "reference_match": reference_match is not None
        }
        
        # Add reference info if matched
        if reference_match:
            plan["reference_info"] = {
                "original_name": reference_match["part_name_original"],
                "description": reference_match["description"],
                "quantity": reference_match["quantity"],
                "tags": reference_match["tags"]
            }
        
        plans.append(plan)
    
    return plans


def suggest_part_names(
    filename: str,
    project_number: str,
    db_path: str,
    table_name: str = "project_reference_parts",
    limit: int = 5
) -> List[Dict]:
    """
    Get part name suggestions for a file.
    
    Args:
        filename: Filename to match
        project_number: Project number
        db_path: Path to database
        table_name: Table name for reference parts
        limit: Maximum suggestions to return
    
    Returns:
        List of suggested parts with scores
    
    Example:
        >>> suggestions = suggest_part_names("head_v2.stl", "300868", "db/modelfinder.db")
        >>> suggestions[0]
        {'part_name': 'head', 'score': 0.9, 'original': 'Head', 'description': '...'}
    """
    reference_parts = get_reference_parts(db_path, project_number, table_name)
    
    if not reference_parts:
        return []
    
    filename_slug = slug(Path(filename).stem)
    suggestions = []
    
    for ref_part in reference_parts:
        part_name = ref_part["part_name"]
        
        # Calculate similarity score
        if part_name == filename_slug:
            score = 1.0
        elif part_name in filename_slug:
            score = 0.9
        elif filename_slug in part_name:
            score = 0.85
        else:
            # Word overlap
            filename_words = set(filename_slug.split("_"))
            part_words = set(part_name.split("_"))
            overlap = len(filename_words & part_words)
            total = len(filename_words | part_words)
            score = overlap / total if total > 0 else 0
        
        if score > 0:
            suggestions.append({
                "part_name": part_name,
                "score": score,
                "original": ref_part["part_name_original"],
                "description": ref_part["description"],
                "tags": ref_part["tags"]
            })
    
    # Sort by score and limit
    suggestions.sort(key=lambda x: x["score"], reverse=True)
    return suggestions[:limit]

