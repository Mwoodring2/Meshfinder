"""
ML-based part type classifier (stub + thresholding)

Minimal head now; upgrade later to OpenCLIP + small classifier.
Currently uses rule-based heuristics for demonstration.
"""
from dataclasses import dataclass

@dataclass
class TypePrediction:
    """Result of part type classification."""
    part_name: str
    confidence: float

# Controlled vocabulary for part types
CONTROLLED_PARTS = [
    "head",
    "left_arm",
    "right_arm",
    "torso",
    "base",
    "weapon_blade",
    "cape",
    "key",
    "pin"
]

# Conservative threshold for auto-acceptance
DEFAULT_THRESHOLD = 0.92


def predict_part_name(mesh_stats: dict, text_context: str = "") -> TypePrediction:
    """
    Predict part type from mesh statistics and text context.
    
    Args:
        mesh_stats: Dictionary containing mesh metadata
            - triangle_count: Number of triangles
            - dimensions_xyz: Tuple of (x, y, z) dimensions in mm
            - volume: Volume in mm³
            - watertight: Boolean watertightness
        text_context: Additional context from filename/tags (optional)
    
    Returns:
        TypePrediction with part_name and confidence score
    
    TODO: Replace with OpenCLIP + small classifier head for production use.
    Current implementation uses simple rule-based heuristics.
    
    Example:
        >>> stats = {"triangle_count": 50000, "dimensions_xyz": (200, 5, 20)}
        >>> pred = predict_part_name(stats)
        >>> pred.part_name
        'weapon_blade'
        >>> pred.confidence
        0.65
    """
    # Extract mesh features
    tri = mesh_stats.get("triangle_count", 0)
    dims = mesh_stats.get("dimensions_xyz", (0, 0, 0))
    volume = mesh_stats.get("volume", 0)
    
    # Calculate derived features
    if isinstance(dims, (list, tuple)) and len(dims) >= 3:
        long_axis = max(dims)
        short_axis = min(dims)
        aspect_ratio = long_axis / short_axis if short_axis > 0 else 0
    else:
        long_axis = 0
        aspect_ratio = 0
    
    # Rule-based classification (placeholder for ML model)
    guess = "base"
    confidence = 0.5  # Conservative default
    
    # Example heuristics:
    # 1. Very thin, long objects -> weapon_blade
    if long_axis > 150 and tri < 200_000 and aspect_ratio > 10:
        guess = "weapon_blade"
        confidence = 0.65
    
    # 2. Roughly spherical, moderate size -> head
    elif aspect_ratio < 2 and 50 < long_axis < 150 and tri > 10_000:
        guess = "head"
        confidence = 0.60
    
    # 3. Large, complex geometry -> torso
    elif tri > 500_000 and volume > 100_000:
        guess = "torso"
        confidence = 0.55
    
    # 4. Small, simple -> pin or key
    elif tri < 5_000 and long_axis < 30:
        # Check text context for hints
        if text_context and "pin" in text_context.lower():
            guess = "pin"
            confidence = 0.70
        elif text_context and "key" in text_context.lower():
            guess = "key"
            confidence = 0.70
        else:
            guess = "pin"
            confidence = 0.50
    
    # 5. Thin, wide -> cape
    elif aspect_ratio > 5 and tri > 20_000:
        guess = "cape"
        confidence = 0.58
    
    # Text context boost (simple keyword matching)
    if text_context:
        text_lower = text_context.lower()
        for part in CONTROLLED_PARTS:
            if part.replace("_", " ") in text_lower or part in text_lower:
                guess = part
                confidence = min(0.85, confidence + 0.20)  # Boost but stay conservative
                break
    
    return TypePrediction(part_name=guess, confidence=confidence)


def is_confident_prediction(prediction: TypePrediction, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """
    Check if prediction confidence exceeds threshold for auto-acceptance.
    
    Args:
        prediction: TypePrediction instance
        threshold: Minimum confidence (default: 0.92)
    
    Returns:
        True if prediction is confident enough for auto-acceptance
    """
    return prediction.confidence >= threshold


# Placeholder for future ML model upgrade
class TypeClassifierModel:
    """
    Placeholder for future OpenCLIP + classifier head.
    
    Future implementation will:
    1. Load OpenCLIP model (local ONNX or PyTorch)
    2. Extract visual + text embeddings
    3. Feed to small classifier head
    4. Return probabilities for all CONTROLLED_PARTS
    """
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.loaded = False
    
    def predict(self, mesh_stats: dict, text_context: str = "") -> TypePrediction:
        """Future: ML-based prediction. Currently falls back to rules."""
        return predict_part_name(mesh_stats, text_context)

