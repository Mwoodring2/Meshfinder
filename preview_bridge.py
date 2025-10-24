# preview_bridge.py — streamlined bridge for efficient 3D previews

from pathlib import Path
from typing import Tuple

def pick_max_faces(file_size_mb: float) -> int:
    """Global face limit based on file size to prevent memory/CPU thrash"""
    if file_size_mb > 300:  # ~6M tris
        return 40_000
    if file_size_mb > 200:  # ~4M tris
        return 60_000
    if file_size_mb > 120:  # ~2.4M tris
        return 90_000
    return 120_000

def render_preview_qpixmap(file_path: str, size: Tuple[int, int] = (384, 384), debug_force_hull: bool = False):
    """
    Efficient preview generation for large STL/OBJ files.
    Returns (QPixmap | None, stats: dict).
    """
    import numpy as np
    from PySide6 import QtGui
    from PIL.ImageQt import ImageQt
    from solid_renderer import render_mesh_to_image
    import trimesh

    stats = {"faces": None, "vertices": None, "extents": None, "watertight": None, "notes": []}
    path = Path(file_path)

    # Fast mesh loading with fallback
    mesh = None
    for process in (False, True):
        try:
            m = trimesh.load(str(path), force='mesh', process=process, maintain_order=True)
            if isinstance(m, trimesh.Scene):
                m = trimesh.util.concatenate(m.dump())
            if isinstance(m, trimesh.points.PointCloud):
                m = m.convex_hull
            mesh = m
            break
        except Exception:
            continue

    if mesh is None:
        return None, stats

    # Extract stats efficiently
    try:
        stats["vertices"] = int(mesh.vertices.shape[0])
        stats["faces"] = int(mesh.faces.shape[0])
        stats["extents"] = tuple(map(float, mesh.extents))
        stats["watertight"] = bool(mesh.is_watertight)
    except Exception:
        pass

    # Use global face limit system
    file_size_mb = path.stat().st_size / (1024 * 1024)
    max_faces = pick_max_faces(file_size_mb)
    
    # Hard kill-switch before rendering for extremely large meshes
    if hasattr(mesh, 'faces') and mesh.faces.shape[0] > 3_000_000:
        mesh = mesh.convex_hull
        stats["notes"].append(f"Mesh too large ({mesh.faces.shape[0]:,} faces): using convex hull")

    # Render with optional debug mode
    try:
        if debug_force_hull:
            mesh = mesh.convex_hull
            img = render_mesh_to_image(mesh, size=size, max_faces=max_faces)
        else:
            img = render_mesh_to_image(file_path, size=size, max_faces=max_faces)
        
        qimage = ImageQt(img)
        pixmap = QtGui.QPixmap.fromImage(qimage)
        return pixmap, stats
    except Exception as e:
        stats["notes"].append(f"render failure: {e}")
        return None, stats
