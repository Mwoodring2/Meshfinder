"""
Proposal generation for file migration.

Builds migration plans with canonical naming and metadata extraction.
"""
from pathlib import Path
from ..utils.naming import canonical_name
from ..ml.project_extractor import propose_fields


def build_proposals(rows, dest_root: str) -> list[dict]:
    """
    Build migration proposals for a set of files.
    
    Args:
        rows: Iterable of dictionaries with file metadata:
            - path: File path
            - name: File name
            - ext: File extension
            - mesh_stats: Mesh metadata dict (optional)
            - tags: User tags (optional)
        dest_root: Destination root directory for migration
    
    Returns:
        List of proposal dictionaries:
        - from: Source path
        - to: Destination path (with canonical naming)
        - fields: Extracted metadata fields
        - needs_review: Boolean flag for manual review
    
    Example:
        >>> rows = [{
        ...     "path": "E:/Models/ABC-1234/file.stl",
        ...     "name": "file.stl",
        ...     "ext": ".stl",
        ...     "mesh_stats": {"triangle_count": 50000},
        ...     "tags": "star wars"
        ... }]
        >>> plans = build_proposals(rows, "E:/Organized")
        >>> plans[0]['to']
        'E:/Organized/ABC-1234/abc_1234_star_wars_weapon_blade.stl'
    """
    plans = []
    
    for r in rows:
        # Build context from name, tags, and parent folder
        path_obj = Path(r["path"])
        ctx = " ".join([
            r["name"],
            r.get("tags", ""),
            path_obj.parent.name
        ])
        
        # Extract fields using ML/rules
        fields = propose_fields(
            r["path"],
            r.get("mesh_stats", {}),
            ctx
        )
        
        # Generate canonical filename
        ext = r["ext"].lstrip(".")  # Remove leading dot if present
        new_name = canonical_name(
            fields["project_number"],
            fields["project_name"],
            fields["part_name"],
            ext
        )
        
        # Determine target directory (group by project number)
        target_dir = Path(dest_root) / fields["project_number"]
        
        plans.append({
            "from": r["path"],
            "to": str(target_dir / new_name),
            "fields": fields,
            "needs_review": fields.get("needs_review", False)
        })
    
    return plans


def filter_proposals(plans: list[dict], needs_review_only: bool = False) -> list[dict]:
    """
    Filter proposals based on criteria.
    
    Args:
        plans: List of proposal dictionaries
        needs_review_only: If True, return only proposals needing review
    
    Returns:
        Filtered list of proposals
    """
    if needs_review_only:
        return [p for p in plans if p.get("needs_review", False)]
    return plans


def summary_stats(plans: list[dict]) -> dict:
    """
    Generate summary statistics for proposals.
    
    Args:
        plans: List of proposal dictionaries
    
    Returns:
        Dictionary with summary stats
    """
    total = len(plans)
    needs_review = sum(1 for p in plans if p.get("needs_review", False))
    
    # Count by project
    projects = {}
    for p in plans:
        proj = p["fields"].get("project_number", "unknown")
        projects[proj] = projects.get(proj, 0) + 1
    
    # Average confidence
    confidences = [p["fields"].get("type_conf", 0) for p in plans]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    
    return {
        "total_files": total,
        "needs_review": needs_review,
        "auto_migrate": total - needs_review,
        "unique_projects": len(projects),
        "projects": projects,
        "avg_confidence": round(avg_conf, 3)
    }

