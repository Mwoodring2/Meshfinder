# preview_bridge.py — tiny bridge from file path → QPixmap (Qt) or PNG (headless)

from pathlib import Path
from typing import Optional, Tuple

def render_preview_qpixmap(file_path: str, size: Tuple[int, int] = (384, 384), debug_force_hull: bool = False):
    """
    Safe preview entrypoint for huge STL/OBJ.
    Returns (QPixmap | None, stats: dict).
    
    Args:
        file_path: Path to 3D file
        size: Preview dimensions
        debug_force_hull: If True, force convex hull to detect degenerate/self-intersecting source
    """
    import json
    import numpy as np
    from PySide6 import QtGui
    from PIL.ImageQt import ImageQt

    # Lazy import to avoid heavy deps at module import
    from solid_renderer import render_mesh_to_image
    import trimesh

    stats = {"faces": None, "vertices": None, "extents": None, "watertight": None, "notes": []}
    path = Path(file_path)

    # ---- 1) Inspect quickly and defensively
    mesh = None
    for process in (False, True):
        try:
            m = trimesh.load(str(path), force='mesh', process=process, maintain_order=True)
            if isinstance(m, trimesh.Scene):
                m = trimesh.util.concatenate(m.dump())
            if isinstance(m, trimesh.points.PointCloud):
                m = m.convex_hull
            mesh = m
            stats["notes"].append(f"loaded process={process}")
            break
        except Exception as e:
            stats["notes"].append(f"load failed process={process}: {e}")

    if mesh is None:
        stats["notes"].append("fallback: unable to load mesh")
        return None, stats

    try:
        V = int(getattr(mesh, "vertices", np.empty((0, 3))).shape[0])
        F = int(getattr(mesh, "faces", np.empty((0, 3))).shape[0])
        stats["vertices"] = V
        stats["faces"] = F
        stats["extents"] = tuple(map(float, getattr(mesh, "extents", [0, 0, 0])))
        stats["watertight"] = bool(getattr(mesh, "is_watertight", False))
    except Exception as e:
        stats["notes"].append(f"stat failure: {e}")

    # ---- 2) Render shaded isometric via PIL (robust to big/dirty meshes)
    try:
        # Determine face cap based on file size to prevent memory/CPU thrash
        file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
        if file_size_mb > 200:  # Very large files (200MB+) - very conservative
            max_faces = 90_000
            stats["notes"].append(f"Large file ({file_size_mb:.1f}MB): using conservative {max_faces:,} face limit")
        elif file_size_mb > 100:  # Large files (100-200MB)
            max_faces = 120_000
        else:  # Normal files
            max_faces = 150_000
        
        if debug_force_hull:
            # Debug mode: Force convex hull to detect degenerate/self-intersecting source
            stats["notes"].append("DEBUG: Forcing convex hull")
            mesh = mesh.convex_hull
            img = render_mesh_to_image(mesh, size=size, max_faces=max_faces)
        else:
            # Normal mode: Use file path directly
            img = render_mesh_to_image(file_path, size=size, max_faces=max_faces)
        
        qimage = ImageQt(img)   # PIL → QImage
        pixmap = QtGui.QPixmap.fromImage(qimage)
        return pixmap, stats
    except Exception as e:
        stats["notes"].append(f"render failure: {e}")
        return None, stats
