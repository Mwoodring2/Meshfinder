"""
Geometric Feature Extraction for 3D Mesh Classification

Extracts geometric features from 3D meshes to train ML models
that can recognize part types (foot, hand, head, etc.) from shape alone.
"""
import numpy as np
from pathlib import Path
from typing import Dict, Optional

try:
    import trimesh
    _TRIMESH_AVAILABLE = True
except ImportError:
    _TRIMESH_AVAILABLE = False


def extract_geometric_features(mesh_path: str) -> Optional[Dict[str, float]]:
    """
    Extract geometric features from a 3D mesh file.
    
    Features include:
    - Bounding box dimensions and ratios
    - Volume and surface area
    - Compactness (sphere-likeness)
    - Principal axes (orientation)
    - Convexity
    - Centroid position
    
    Args:
        mesh_path: Path to 3D mesh file
        
    Returns:
        Dictionary of feature names to values, or None if extraction fails
    """
    if not _TRIMESH_AVAILABLE:
        print("Warning: trimesh not available for geometric feature extraction")
        return None
    
    try:
        # Load mesh
        mesh = trimesh.load(mesh_path, force='mesh', skip_materials=True)
        
        if mesh is None or not hasattr(mesh, 'vertices'):
            return None
        
        # Bounding box
        bbox = mesh.bounding_box.extents
        bbox_x, bbox_y, bbox_z = bbox[0], bbox[1], bbox[2]
        
        # Prevent division by zero
        bbox_x = max(bbox_x, 1e-6)
        bbox_y = max(bbox_y, 1e-6)
        bbox_z = max(bbox_z, 1e-6)
        
        # Volume and area
        volume = max(mesh.volume, 1e-6) if hasattr(mesh, 'volume') else 1e-6
        area = max(mesh.area, 1e-6) if hasattr(mesh, 'area') else 1e-6
        
        # Principal Component Analysis for orientation
        vertices = np.asarray(mesh.vertices)
        centered = vertices - vertices.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        eigenvalues = np.abs(eigenvalues)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order
        
        # Normalize eigenvalues
        eigenvalues = eigenvalues / (eigenvalues.sum() + 1e-6)
        
        features = {
            # Bounding box dimensions
            'bbox_x': float(bbox_x),
            'bbox_y': float(bbox_y),
            'bbox_z': float(bbox_z),
            
            # Aspect ratios (shape indicators)
            'aspect_xy': float(bbox_x / bbox_y),
            'aspect_xz': float(bbox_x / bbox_z),
            'aspect_yz': float(bbox_y / bbox_z),
            
            # Size indicators
            'volume': float(volume),
            'surface_area': float(area),
            'bbox_volume': float(bbox_x * bbox_y * bbox_z),
            
            # Shape compactness (1.0 = perfect sphere)
            'compactness': float(volume / (area ** 1.5) if area > 0 else 0),
            
            # Convexity (1.0 = perfectly convex)
            'convexity': float(volume / mesh.convex_hull.volume if hasattr(mesh, 'convex_hull') else 1.0),
            
            # Topology
            'tri_count': int(len(mesh.faces)) if hasattr(mesh, 'faces') else 0,
            'vertex_count': int(len(mesh.vertices)),
            'is_watertight': int(mesh.is_watertight) if hasattr(mesh, 'is_watertight') else 0,
            
            # Principal axes (orientation indicators)
            'principal_axis_1': float(eigenvalues[0]),
            'principal_axis_2': float(eigenvalues[1]),
            'principal_axis_3': float(eigenvalues[2]),
            
            # Centroid position (height/vertical position indicator)
            'centroid_x': float(mesh.centroid[0]),
            'centroid_y': float(mesh.centroid[1]),
            'centroid_z': float(mesh.centroid[2]),
            
            # Elongation indicator
            'elongation': float(max(bbox_x, bbox_y, bbox_z) / min(bbox_x, bbox_y, bbox_z)),
            
            # Flatness indicator
            'flatness': float(min(bbox_x, bbox_y, bbox_z) / max(bbox_x, bbox_y, bbox_z)),
        }
        
        return features
        
    except Exception as e:
        print(f"Error extracting features from {mesh_path}: {e}")
        return None


def detect_laterality(mesh_path: str) -> Optional[str]:
    """
    Attempt to detect if a mesh is left or right sided.
    
    Uses centroid position and geometry asymmetry.
    
    Returns:
        'left', 'right', 'center', or None
    """
    if not _TRIMESH_AVAILABLE:
        return None
    
    try:
        mesh = trimesh.load(mesh_path, force='mesh', skip_materials=True)
        
        if mesh is None:
            return None
        
        # Check if mesh is symmetric (likely center piece)
        centroid = mesh.centroid
        vertices = np.asarray(mesh.vertices)
        
        # Check X-axis symmetry
        left_verts = vertices[vertices[:, 0] < centroid[0]]
        right_verts = vertices[vertices[:, 0] >= centroid[0]]
        
        if len(left_verts) == 0 or len(right_verts) == 0:
            return 'center'
        
        left_ratio = len(left_verts) / len(vertices)
        
        # If roughly balanced, it's centered
        if 0.4 < left_ratio < 0.6:
            return 'center'
        elif left_ratio > 0.6:
            return 'left'
        else:
            return 'right'
            
    except Exception:
        return None


def features_to_array(features: Dict[str, float]) -> np.ndarray:
    """Convert feature dict to numpy array for ML"""
    # Fixed order of features for consistency
    feature_keys = [
        'bbox_x', 'bbox_y', 'bbox_z',
        'aspect_xy', 'aspect_xz', 'aspect_yz',
        'volume', 'surface_area', 'bbox_volume',
        'compactness', 'convexity',
        'tri_count', 'vertex_count', 'is_watertight',
        'principal_axis_1', 'principal_axis_2', 'principal_axis_3',
        'centroid_x', 'centroid_y', 'centroid_z',
        'elongation', 'flatness'
    ]
    
    return np.array([features.get(k, 0.0) for k in feature_keys])


def get_feature_names() -> list:
    """Get ordered list of feature names"""
    return [
        'bbox_x', 'bbox_y', 'bbox_z',
        'aspect_xy', 'aspect_xz', 'aspect_yz',
        'volume', 'surface_area', 'bbox_volume',
        'compactness', 'convexity',
        'tri_count', 'vertex_count', 'is_watertight',
        'principal_axis_1', 'principal_axis_2', 'principal_axis_3',
        'centroid_x', 'centroid_y', 'centroid_z',
        'elongation', 'flatness'
    ]


if __name__ == "__main__":
    # Test feature extraction
    import sys
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        print(f"Extracting features from: {test_file}")
        features = extract_geometric_features(test_file)
        
        if features:
            print("\nGeometric Features:")
            for key, value in features.items():
                print(f"  {key:20s}: {value:.4f}")
            
            laterality = detect_laterality(test_file)
            print(f"\nDetected laterality: {laterality}")
        else:
            print("Failed to extract features")

