"""
Project field extraction from file paths and mesh data.

Combines type classification, naming rules, and heuristics to propose
structured metadata for discovered files.
"""
from . import type_classifier as tc
from ..utils.naming import extract_project_number, slug


def propose_fields(path_str: str, mesh_stats: dict, context_text: str = "") -> dict:
    """
    Propose metadata fields for a 3D model file.
    
    Args:
        path_str: Full file path
        mesh_stats: Dictionary with mesh metadata (triangle_count, dimensions_xyz, etc.)
        context_text: Additional context (filename, tags, parent folders)
    
    Returns:
        Dictionary with proposed fields:
        - project_number: Extracted project ID or "unknown"
        - project_name: Slugified project name from path
        - part_name: Predicted part name (slugified)
        - type_conf: Confidence score for part type prediction
        - needs_review: Boolean flag if confidence is low or project unknown
    
    Example:
        >>> stats = {"triangle_count": 50000, "dimensions_xyz": (200, 5, 20)}
        >>> result = propose_fields(
        ...     "E:/Models/ABC-1234/Star Wars/lightsaber_blade.stl",
        ...     stats,
        ...     "ABC-1234/Star Wars/lightsaber_blade.stl"
        ... )
        >>> result['project_number']
        'ABC-1234'
        >>> result['part_name']
        'weapon_blade'
        >>> result['needs_review']
        False
    """
    # Extract project number from context
    pn = extract_project_number(context_text) or "unknown"
    
    # Predict part type using ML (or rule-based fallback)
    tp = tc.predict_part_name(mesh_stats, context_text)
    
    # Use predicted part name if confident, otherwise generic "part"
    part = slug(tp.part_name) if tp.confidence else "part"
    
    # Extract project name from path hierarchy
    # Try to get parent folder name (assumes path structure like: .../project_folder/file.stl)
    if "/" in context_text:
        path_parts = context_text.split("/")
        # Get second-to-last part (parent folder of the file)
        project_folder = path_parts[-2] if len(path_parts) >= 2 else "unknown"
    elif "\\" in context_text:
        path_parts = context_text.split("\\")
        project_folder = path_parts[-2] if len(path_parts) >= 2 else "unknown"
    else:
        project_folder = "unknown"
    
    return {
        "project_number": pn,
        "project_name": slug(project_folder),
        "part_name": part,
        "type_conf": tp.confidence,
        "needs_review": tp.confidence < tc.DEFAULT_THRESHOLD or pn == "unknown"
    }

