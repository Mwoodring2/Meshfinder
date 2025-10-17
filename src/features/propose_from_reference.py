"""
Core proposal logic using reference parts + ML geometric classification.

Generates migration proposals by:
1. Extracting geometric features from mesh
2. Classifying part type using ML (if model available)
3. Matching filenames against reference parts (text matching)
4. Combining geometric and text confidence scores
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Dict, Any, List, Optional

from ..dataio.reference_parts import load_reference_parts
from ..utils.naming import canonical_name
from ..utils.normalize import ascii_slug, guess_part_from_filename

# Try to import ML components
try:
    from ..ml.geometry_features import extract_geometric_features, detect_laterality
    from ..ml.part_classifier import PartClassifier
    _ML_AVAILABLE = True
except ImportError:
    _ML_AVAILABLE = False

# Conservative threshold: below this score requires manual review
DEFAULT_ACCEPT_SCORE = 0.66

# Confidence weighting (if ML available)
GEOMETRY_WEIGHT = 0.6  # 60% weight to geometric classification
TEXT_WEIGHT = 0.4      # 40% weight to text matching


@dataclass
class RowMeta:
    """Minimal file metadata for proposal generation."""
    path: str
    name: str
    ext: str
    tags: Optional[str] = None


def _classify_with_ml(file_path: str, db_path: str) -> Optional[Dict[str, Any]]:
    """
    Classify part type using ML geometric features.
    
    Returns:
        Dict with part_type, confidence, laterality, or None if ML unavailable
    """
    if not _ML_AVAILABLE:
        return None
    
    try:
        # Extract geometric features
        features = extract_geometric_features(file_path)
        if not features:
            return None
        
        # Load trained classifier
        classifier = PartClassifier(db_path)
        if not classifier.load('models/part_classifier.pkl'):
            return None
        
        # Predict
        part_type, part_conf, laterality, lat_conf = classifier.predict(features)
        
        return {
            'part_type': part_type,
            'confidence': part_conf,
            'laterality': laterality,
            'laterality_confidence': lat_conf
        }
        
    except Exception as e:
        print(f"ML classification failed: {e}")
        return None


def propose_for_rows(
    rows: Iterable[RowMeta],
    db_path: str,
    project_number: str
) -> List[Dict[str, Any]]:
    """
    Generate migration proposals for a batch of files using ML + text matching.
    
    Args:
        rows: Iterable of RowMeta objects with file information
        db_path: Path to SQLite database with reference parts
        project_number: Project number to look up reference parts
    
    Returns:
        List of proposal dictionaries containing:
        - from: Source file path
        - project_number: Project number
        - project_name: Project name from reference (or "unknown")
        - part_name: Matched part name (or "part" if no match)
        - conf: Combined confidence score (0.0 to 1.0)
        - geo_conf: Geometric classification confidence (if ML available)
        - text_conf: Text matching confidence
        - laterality: left/right/center (if ML available)
        - proposed_name: Canonical filename
        - needs_review: True if confidence below threshold
    
    Example:
        >>> rows = [
        ...     RowMeta(path="E:/Raw/head.stl", name="head.stl", ext=".stl"),
        ...     RowMeta(path="E:/Raw/arm_left.obj", name="arm_left.obj", ext=".obj")
        ... ]
        >>> proposals = propose_for_rows(rows, "db/modelfinder.db", "300868")
        >>> proposals[0]
        {
            'from': 'E:/Raw/head.stl',
            'project_number': '300868',
            'project_name': 'superman_pf',
            'part_name': 'head',
            'conf': 0.92,
            'geo_conf': 0.95,
            'text_conf': 0.87,
            'proposed_name': '300868_superman_pf_head.stl',
            'needs_review': False
        }
    """
    # Load reference parts for this project
    project_name, parts_map = load_reference_parts(db_path, project_number)
    parts_vocab = set(parts_map.keys())
    
    proposals: List[Dict[str, Any]] = []

    for r in rows:
        # 1) Text-based matching (filename → reference parts)
        text_guess, text_score = guess_part_from_filename(r.name, parts_vocab)
        
        # 2) ML-based classification (geometry → part type)
        ml_result = _classify_with_ml(r.path, db_path)
        
        # 3) Combine results
        if ml_result and ml_result['confidence'] > 0.3:  # ML available and confident
            # Prefer ML classification
            part = ml_result['part_type']
            laterality = ml_result['laterality']
            
            # Combined confidence: 60% geometry + 40% text
            geo_conf = ml_result['confidence']
            
            # If text matching agrees, boost confidence
            if text_guess == part:
                combined_conf = (geo_conf * GEOMETRY_WEIGHT) + (text_score * TEXT_WEIGHT)
            else:
                # Text disagrees - use geometric confidence but penalize
                combined_conf = geo_conf * GEOMETRY_WEIGHT + 0.2 * TEXT_WEIGHT
            
            # Add laterality prefix if detected and not center
            if laterality and laterality != 'center':
                part = f"{laterality}_{part}"
        
        else:
            # Fallback to text-only matching
            part = text_guess if text_guess else "part"
            combined_conf = float(text_score)
            geo_conf = 0.0
            laterality = None

        # 4) Generate canonical filename
        proposed = canonical_name(
            project_number,
            project_name or "unknown",
            part,
            r.ext.lstrip(".")  # Remove leading dot if present
        )

        # 5) Build proposal
        proposal = {
            "from": r.path,
            "project_number": project_number,
            "project_name": project_name or "unknown",
            "part_name": part,
            "conf": combined_conf,
            "text_conf": float(text_score),
            "proposed_name": proposed,
            "needs_review": combined_conf < DEFAULT_ACCEPT_SCORE,
            "original_label": parts_map.get(part, "") if part in parts_map else ""
        }
        
        # Add ML-specific fields if available
        if ml_result:
            proposal['geo_conf'] = ml_result['confidence']
            proposal['laterality'] = laterality
            proposal['laterality_conf'] = ml_result['laterality_confidence']
        else:
            proposal['geo_conf'] = 0.0
            proposal['laterality'] = None
        
        proposals.append(proposal)
    
    return proposals


def propose_for_single_file(
    file_path: str,
    filename: str,
    ext: str,
    db_path: str,
    project_number: str
) -> Dict[str, Any]:
    """
    Generate proposal for a single file.
    
    Args:
        file_path: Full file path
        filename: Filename (with extension)
        ext: File extension
        db_path: Path to SQLite database
        project_number: Project number
    
    Returns:
        Proposal dictionary
    """
    row = RowMeta(path=file_path, name=filename, ext=ext)
    proposals = propose_for_rows([row], db_path, project_number)
    return proposals[0] if proposals else {}


def batch_propose_by_project(
    rows: Iterable[RowMeta],
    db_path: str,
    project_extractor=None
) -> List[Dict[str, Any]]:
    """
    Generate proposals for files from multiple projects.
    
    Automatically groups files by project number and generates proposals
    for each group.
    
    Args:
        rows: Iterable of RowMeta objects
        db_path: Path to SQLite database
        project_extractor: Optional function to extract project number from path
                          Default uses naming.extract_project_number
    
    Returns:
        List of all proposals across all projects
    """
    if project_extractor is None:
        from ..utils.naming import extract_project_number
        project_extractor = extract_project_number
    
    # Group rows by project number
    by_project: Dict[str, List[RowMeta]] = {}
    
    for row in rows:
        # Try to extract project number from path
        project_num = project_extractor(row.path)
        if not project_num:
            project_num = "unknown"
        
        if project_num not in by_project:
            by_project[project_num] = []
        by_project[project_num].append(row)
    
    # Generate proposals for each project
    all_proposals = []
    
    for project_num, project_rows in by_project.items():
        if project_num == "unknown":
            # For unknown projects, create basic proposals
            for row in project_rows:
                all_proposals.append({
                    "from": row.path,
                    "project_number": "unknown",
                    "project_name": "unknown",
                    "part_name": "part",
                    "conf": 0.0,
                    "proposed_name": f"unknown_unknown_part{row.ext}",
                    "needs_review": True,
                    "original_label": ""
                })
        else:
            # Use reference-based proposal
            proposals = propose_for_rows(project_rows, db_path, project_num)
            all_proposals.extend(proposals)
    
    return all_proposals


def filter_proposals_by_confidence(
    proposals: List[Dict[str, Any]],
    min_confidence: float = DEFAULT_ACCEPT_SCORE
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split proposals into auto-accept and needs-review groups.
    
    Args:
        proposals: List of proposal dictionaries
        min_confidence: Minimum confidence for auto-accept (default: 0.66)
    
    Returns:
        Tuple of (auto_accept_list, needs_review_list)
    """
    auto_accept = []
    needs_review = []
    
    for proposal in proposals:
        if proposal.get("conf", 0.0) >= min_confidence:
            auto_accept.append(proposal)
        else:
            needs_review.append(proposal)
    
    return auto_accept, needs_review


def summary_stats(proposals: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate summary statistics for proposals.
    
    Args:
        proposals: List of proposal dictionaries
    
    Returns:
        Dictionary with summary statistics
    """
    total = len(proposals)
    
    if total == 0:
        return {
            "total": 0,
            "auto_accept": 0,
            "needs_review": 0,
            "avg_confidence": 0.0,
            "by_project": {}
        }
    
    auto_accept = sum(1 for p in proposals if not p.get("needs_review", False))
    needs_review = total - auto_accept
    
    # Average confidence
    confidences = [p.get("conf", 0.0) for p in proposals]
    avg_conf = sum(confidences) / len(confidences)
    
    # Group by project
    by_project: Dict[str, int] = {}
    for p in proposals:
        proj = p.get("project_number", "unknown")
        by_project[proj] = by_project.get(proj, 0) + 1
    
    return {
        "total": total,
        "auto_accept": auto_accept,
        "needs_review": needs_review,
        "avg_confidence": round(avg_conf, 3),
        "by_project": by_project
    }

