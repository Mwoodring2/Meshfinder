"""
Naming utilities for ModelFinder
Enforces canonical naming: project_number_projectname_partname.ext
"""
import re
import unicodedata
import hashlib
from pathlib import Path

PROJECT_NUM_RE = re.compile(r'([A-Z]{2,5}-?\d{2,4}(?:-\d{2,3})?|PRJ\d+|\d{3,})', re.I)

def _ascii(s: str) -> str:
    """Convert Unicode to ASCII, removing accents and special chars."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

def slug(s: str) -> str:
    """Convert string to lowercase slug (alphanumeric + underscores only)."""
    s = _ascii(s).lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")

def extract_project_number(text: str) -> str | None:
    """
    Extract project number from text using pattern matching.
    
    Supported formats:
    - ABC-1234 (letter prefix with dash)
    - ABCDE1234 (letter prefix without dash)
    - PRJ123 (PRJ prefix)
    - 1234 (numeric only, 3+ digits)
    
    Returns:
        Project number string or None if not found
    """
    m = PROJECT_NUM_RE.search(text)
    return m.group(1) if m else None

def canonical_name(project_number: str, project_name: str, part_name: str, ext: str) -> str:
    """
    Generate canonical filename: project_number_projectname_partname.ext
    
    Args:
        project_number: Raw project number (will be preserved as-is)
        project_name: Project name (will be slugified)
        part_name: Part name (will be slugified)
        ext: File extension (will be lowercased)
    
    Returns:
        Canonical filename string
    
    Example:
        >>> canonical_name("ABC-1234", "Star Wars Droid", "R2-D2 Body", "stl")
        'ABC-1234_star_wars_droid_r2_d2_body.stl'
    """
    return f"{project_number.strip()}_{slug(project_name)}_{slug(part_name)}.{ext.lower()}"

def hash8(path: Path) -> str:
    """
    Compute first 8 characters of SHA256 hash for file.
    
    Args:
        path: Path to file
    
    Returns:
        8-character hex string (short hash)
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:8]

