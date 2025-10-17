"""
Helper functions for review queue and batch operations.
"""
from pathlib import Path
from ..utils.naming import extract_project_number, canonical_name
from ..ml import DEFAULT_THRESHOLD


def needs_review_filter(row: dict, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """
    Check if a row needs manual review.
    
    Args:
        row: Dictionary with file metadata
        threshold: Confidence threshold (default: 0.92)
    
    Returns:
        True if the file needs review
    """
    # Needs review if:
    # 1. No project number or unknown
    # 2. Confidence below threshold
    # 3. Status is quarantined
    
    project_num = row.get("project_number")
    confidence = row.get("type_conf", 0)
    status = row.get("status", "discovered")
    
    if not project_num or project_num == "unknown":
        return True
    
    if confidence < threshold:
        return True
    
    if status == "quarantined":
        return True
    
    return False


def extract_project_from_folder(path: str) -> str:
    """
    Extract project number from folder structure.
    
    Args:
        path: File path
    
    Returns:
        Project number or "unknown"
    """
    p = Path(path)
    
    # Check current folder and parents
    for part in [p.parent.name, p.parent.parent.name if p.parent.parent else ""] + list(p.parts):
        proj = extract_project_number(part)
        if proj:
            return proj
    
    return "unknown"


def batch_fill_project_name(rows: list[dict]) -> dict:
    """
    Fill project name from parent folder for multiple rows.
    
    Args:
        rows: List of row dictionaries
    
    Returns:
        Dictionary mapping row indices to project names
    """
    updates = {}
    
    for idx, row in enumerate(rows):
        path = row.get("path", "")
        if path:
            parent = Path(path).parent.name
            if parent and parent not in [".", "..", ""]:
                # Clean up parent folder name for use as project name
                updates[idx] = parent
    
    return updates


def copy_proposed_name(row: dict, dest_root: str = None) -> str:
    """
    Generate proposed canonical name for a row.
    
    Args:
        row: Row dictionary
        dest_root: Optional destination root (for full path)
    
    Returns:
        Proposed filename or full path
    """
    project_num = row.get("project_number", "unknown")
    project_name = row.get("project_name", "unknown")
    part_name = row.get("part_name", "part")
    ext = row.get("ext", "").lstrip(".")
    
    proposed = canonical_name(project_num, project_name, part_name, ext)
    
    if dest_root:
        target_dir = Path(dest_root) / project_num
        return str(target_dir / proposed)
    
    return proposed


def get_review_queue_filter() -> str:
    """
    Get SQL filter for review queue.
    
    Returns:
        SQL WHERE clause for filtering review queue items
    """
    return f"""
        (project_number IS NULL OR project_number = 'unknown' OR 
         type_conf < {DEFAULT_THRESHOLD} OR 
         status = 'quarantined')
    """


def validate_fields(project_number: str, project_name: str, part_name: str) -> tuple[bool, str]:
    """
    Validate field values before applying.
    
    Args:
        project_number: Project number
        project_name: Project name
        part_name: Part name
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not project_number or not project_number.strip():
        return False, "Project number is required"
    
    if not project_name or not project_name.strip():
        return False, "Project name is required"
    
    if not part_name or not part_name.strip():
        return False, "Part name is required"
    
    if project_number.strip().lower() == "unknown":
        return False, "Project number cannot be 'unknown'"
    
    return True, ""


class ReviewQueueStats:
    """Statistics for review queue."""
    
    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.total = len(rows)
        self.needs_review = sum(1 for r in rows if needs_review_filter(r))
        self.by_status = {}
        self.by_confidence = {"high": 0, "medium": 0, "low": 0}
        
        for row in rows:
            # Count by status
            status = row.get("status", "discovered")
            self.by_status[status] = self.by_status.get(status, 0) + 1
            
            # Count by confidence
            conf = row.get("type_conf", 0)
            if conf >= DEFAULT_THRESHOLD:
                self.by_confidence["high"] += 1
            elif conf >= 0.7:
                self.by_confidence["medium"] += 1
            else:
                self.by_confidence["low"] += 1
    
    def summary(self) -> str:
        """Get summary text."""
        return (
            f"Total: {self.total} files\n"
            f"Needs Review: {self.needs_review}\n"
            f"Ready: {self.total - self.needs_review}\n\n"
            f"By Status:\n" +
            "\n".join(f"  {k}: {v}" for k, v in self.by_status.items()) +
            f"\n\nBy Confidence:\n"
            f"  High (≥92%): {self.by_confidence['high']}\n"
            f"  Medium (70-92%): {self.by_confidence['medium']}\n"
            f"  Low (<70%): {self.by_confidence['low']}"
        )

