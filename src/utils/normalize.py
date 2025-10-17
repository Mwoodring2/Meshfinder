"""
Normalization and fuzzy matching utilities.

Provides text normalization (ASCII slugification) and part name matching
using token-based Jaccard similarity.
"""
import re
import unicodedata
from typing import Tuple, Set, List, Dict


def ascii_slug(text: str) -> str:
    """
    Convert text to ASCII slug format.
    
    - Normalizes Unicode to ASCII (removing accents)
    - Converts to lowercase
    - Replaces non-alphanumeric with underscores
    - Removes duplicate underscores and trims
    
    Args:
        text: Input text to slugify
    
    Returns:
        Slugified ASCII string
    
    Example:
        >>> ascii_slug("Café-Racer™ 2.0")
        'cafe_racer_2_0'
        >>> ascii_slug("Left Arm (v3)")
        'left_arm_v3'
        >>> ascii_slug(None)
        ''
    """
    if text is None:
        return ""
    
    # Normalize Unicode to ASCII
    s = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    
    # Convert to lowercase
    s = s.lower()
    
    # Replace non-alphanumeric with underscores
    s = re.sub(r"[^a-z0-9]+", "_", s)
    
    # Remove duplicate underscores and trim
    s = re.sub(r"_+", "_", s).strip("_")
    
    return s


def guess_part_from_filename(name: str, parts_vocab: Set[str]) -> Tuple[str, float]:
    """
    Match filename to vocabulary parts using token-overlap scoring.
    
    Uses Jaccard similarity on word tokens to find best match.
    
    Args:
        name: Filename or text to match
        parts_vocab: Set of normalized part names (vocabulary)
    
    Returns:
        Tuple of (best_matching_part, similarity_score) where:
        - best_matching_part: The closest match from vocabulary (or "" if none)
        - similarity_score: Jaccard score 0.0 to 1.0
    
    Algorithm:
        For each vocab part:
        1. Tokenize filename and part name by underscores
        2. Calculate Jaccard: |intersection| / |union|
        3. Return part with highest score
    
    Example:
        >>> vocab = {"head", "left_arm", "right_arm", "torso"}
        >>> guess_part_from_filename("head_v2.stl", vocab)
        ('head', 0.5)
        >>> guess_part_from_filename("left_hand.obj", vocab)
        ('left_arm', 0.5)
        >>> guess_part_from_filename("unknown.stl", vocab)
        ('', 0.0)
    """
    # Normalize filename
    base = ascii_slug(name)
    
    if not base or not parts_vocab:
        return "", 0.0
    
    # Tokenize filename
    tokens = set(base.split("_"))
    
    best = ""
    best_score = 0.0
    
    for part in parts_vocab:
        # Tokenize vocab part
        part_tokens = set(part.split("_"))
        
        # Calculate Jaccard similarity
        intersection = len(tokens & part_tokens)
        union = max(1, len(tokens | part_tokens))
        score = intersection / union
        
        if score > best_score:
            best = part
            best_score = score
    
    return best, best_score


def batch_guess_parts(
    filenames: List[str],
    parts_vocab: Set[str],
    min_score: float = 0.3
) -> Dict[str, Tuple[str, float]]:
    """
    Match multiple filenames to vocabulary parts.
    
    Args:
        filenames: List of filenames to match
        parts_vocab: Set of normalized part names
        min_score: Minimum score threshold (default: 0.3)
    
    Returns:
        Dictionary mapping filename -> (matched_part, score)
        Only includes matches above min_score threshold
    
    Example:
        >>> vocab = {"head", "left_arm", "torso"}
        >>> files = ["head_v2.stl", "torso.obj", "unknown.fbx"]
        >>> batch_guess_parts(files, vocab, min_score=0.4)
        {'head_v2.stl': ('head', 0.5), 'torso.obj': ('torso', 1.0)}
    """
    results = {}
    
    for filename in filenames:
        part, score = guess_part_from_filename(filename, parts_vocab)
        if score >= min_score:
            results[filename] = (part, score)
    
    return results


def normalize_part_list(parts: List[str]) -> Dict[str, str]:
    """
    Normalize a list of part names.
    
    Args:
        parts: List of original part names
    
    Returns:
        Dictionary mapping normalized_name -> original_name
    
    Example:
        >>> parts = ["Head", "Left Arm", "Right-Arm", "Torso (Main)"]
        >>> normalize_part_list(parts)
        {
            'head': 'Head',
            'left_arm': 'Left Arm',
            'right_arm': 'Right-Arm',
            'torso_main': 'Torso (Main)'
        }
    """
    normalized = {}
    
    for original in parts:
        slug = ascii_slug(original)
        if slug:  # Skip empty slugs
            normalized[slug] = original
    
    return normalized


def extract_version_suffix(text: str) -> Tuple[str, str]:
    """
    Extract version suffix from text.
    
    Args:
        text: Text potentially containing version suffix
    
    Returns:
        Tuple of (base_text, version_suffix)
    
    Example:
        >>> extract_version_suffix("head_v2")
        ('head', 'v2')
        >>> extract_version_suffix("arm_left_v3.1")
        ('arm_left', 'v3_1')
        >>> extract_version_suffix("torso")
        ('torso', '')
    """
    # Pattern for version suffixes: v1, v2.0, ver3, version_4, etc.
    pattern = r'[_\-]?(v|ver|version)[_\-]?(\d+(?:[._]\d+)?)[_\-]?$'
    
    slug = ascii_slug(text)
    match = re.search(pattern, slug, re.IGNORECASE)
    
    if match:
        base = slug[:match.start()].rstrip("_")
        version = match.group(1).lower() + match.group(2).replace(".", "_")
        return base, version
    
    return slug, ""


def similarity_score(text1: str, text2: str) -> float:
    """
    Calculate Jaccard similarity between two texts.
    
    Args:
        text1: First text
        text2: Second text
    
    Returns:
        Similarity score from 0.0 to 1.0
    
    Example:
        >>> similarity_score("left arm", "left_hand")
        0.5
        >>> similarity_score("head", "head")
        1.0
    """
    slug1 = ascii_slug(text1)
    slug2 = ascii_slug(text2)
    
    if not slug1 or not slug2:
        return 0.0
    
    tokens1 = set(slug1.split("_"))
    tokens2 = set(slug2.split("_"))
    
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    
    return intersection / union if union > 0 else 0.0

