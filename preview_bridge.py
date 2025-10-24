# preview_bridge.py — streamlined bridge for efficient 3D previews

from pathlib import Path
from typing import Tuple

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

    # Adaptive face limits based on file size
    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb > 200:
        max_faces = 90_000
    elif file_size_mb > 100:
        max_faces = 120_000
    else:
        max_faces = 150_000

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
