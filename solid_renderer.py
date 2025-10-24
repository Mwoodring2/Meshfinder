# solid_renderer.py
# Clean, shaded 3D previews using PIL + NumPy + trimesh (no OpenGL).
# Usage:
#   from solid_renderer import render_mesh_to_image
#   img = render_mesh_to_image("E:/models/thing.stl", size=(512,512))
#   img.save("preview.png")

from __future__ import annotations
import math
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw
import trimesh


# --------------------------- projection helpers ---------------------------

def _iso_matrix() -> np.ndarray:
    """
    Classic isometric: rotate around Z by 45°, then around X by ~35.264° (arctan(sin/sqrt(2))).
    Returns 3x3 rotation matrix.
    """
    rz = math.radians(45.0)
    rx = math.radians(35.26438968)
    Rz = np.array([
        [ math.cos(rz), -math.sin(rz), 0.0],
        [ math.sin(rz),  math.cos(rz), 0.0],
        [          0.0,           0.0, 1.0],
    ], dtype=np.float32)
    Rx = np.array([
        [1.0,          0.0,           0.0],
        [0.0, math.cos(rx), -math.sin(rx)],
        [0.0, math.sin(rx),  math.cos(rx)],
    ], dtype=np.float32)
    return Rx @ Rz


def _fit_to_canvas(verts: np.ndarray, size: Tuple[int, int], margin_px: int = 16) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Normalizes and scales 3D verts (already rotated) to fit the target image size.
    Returns (verts2d_px, scale, offset_px).
    """
    w, h = size
    # Project to XY plane for fit
    xy = verts[:, :2]
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    span = (maxs - mins).max()
    if span <= 0:
        span = 1.0

    usable = np.array([w - 2 * margin_px, h - 2 * margin_px], dtype=np.float32)
    scale = float(min(usable[0], usable[1]) / span)

    # Scale + center
    xy_scaled = (xy - mins[None, :]) * scale + margin_px
    bbox = np.array([xy_scaled.min(axis=0), xy_scaled.max(axis=0)])
    center_offset = (np.array([w, h], dtype=np.float32) - (bbox[0] + bbox[1])) * 0.5
    xy_final = xy_scaled + center_offset
    return xy_final, scale, center_offset


# --------------------------- mesh prep ------------------------------------

def _ensure_trimesh(mesh_or_scene) -> trimesh.Trimesh:
    """
    Guarantees a Trimesh. Scenes are concatenated; point clouds become convex hulls.
    """
    m = mesh_or_scene
    if isinstance(m, trimesh.Scene):
        # dump returns (geometry, transforms); concatenate to a single mesh
        m = trimesh.util.concatenate(m.dump())
    if isinstance(m, trimesh.points.PointCloud):
        # fall back to convex hull if only points provided
        m = m.convex_hull
    if not isinstance(m, trimesh.Trimesh):
        # try to coerce
        try:
            m = trimesh.Trimesh(vertices=np.array(m.vertices), faces=np.array(m.faces))
        except Exception:
            # last resort: build convex hull from vertices if present
            v = np.array(getattr(m, "vertices", []))
            m = trimesh.Trimesh(vertices=v, faces=[]) if v.size else trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    return m


def _load_mesh(path: Path) -> trimesh.Trimesh:
    m = trimesh.load(str(path), force='mesh', process=True)
    m = _ensure_trimesh(m)

    # Cleanup to avoid degenerate/empty faces
    with np.errstate(all='ignore'):
        try: m.remove_duplicate_faces()
        except Exception: pass
        try: m.remove_degenerate_faces()
        except Exception: pass
        try: m.fill_holes()
        except Exception: pass

    # If still no faces, try convex hull
    if getattr(m, "faces", np.empty((0, 3))).shape[0] == 0:
        try:
            m = m.convex_hull
        except Exception:
            # fallback to tiny icosphere to avoid crashes
            m = trimesh.creation.icosphere(subdivisions=1, radius=1.0)

    # Ensure normals
    try:
        if not m.has_vertex_normals:
            m.fix_normals()
    except Exception:
        pass

    # Recenter to origin so iso fit looks good
    try:
        m.rezero()
    except Exception:
        pass

    return m


# --------------------------- shading helpers ------------------------------

def _lambert_face_colors(mesh: trimesh.Trimesh,
                         base_rgb=(220, 220, 240),
                         light_dir=(0.577, 0.577, 0.577),
                         ambient=0.20) -> np.ndarray:
    """
    Simple Lambert shading per face normal.
    Returns uint8 Nx3 color array.
    """
    if not hasattr(mesh, "face_normals") or mesh.face_normals is None or len(mesh.face_normals) == 0:
        mesh.rezero()
        mesh.fix_normals()

    n = mesh.face_normals.astype(np.float32)
    ld = np.asarray(light_dir, dtype=np.float32)
    ld /= (np.linalg.norm(ld) + 1e-9)

    dots = np.clip((n @ ld), 0.0, 1.0)
    intens = np.clip(ambient + (1.0 - ambient) * dots, ambient, 1.0)  # ambient floor

    base = np.asarray(base_rgb, dtype=np.float32)[None, :]
    cols = (base * intens[:, None]).astype(np.uint8)
    return cols


# --------------------------- rasterization ---------------------------------

def _project_isometric(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply iso rotation to vertices.
    Returns (rotated_vertices, rotation_matrix).
    """
    R = _iso_matrix().astype(np.float32)
    v = vertices.astype(np.float32) @ R.T
    return v, R


def _depth_sort_indices(mesh: trimesh.Trimesh, rotated_vertices: np.ndarray) -> np.ndarray:
    """
    Sort faces back-to-front using mean Z in rotated space.
    """
    tris = rotated_vertices[mesh.faces]  # (F,3,3)
    z_mean = tris[:, :, 2].mean(axis=1)
    return np.argsort(z_mean)[::-1]  # far → near


def _to_int_points(tri2d: np.ndarray) -> list[tuple[int, int]]:
    return [(int(round(tri2d[0, 0])), int(round(tri2d[0, 1]))),
            (int(round(tri2d[1, 0])), int(round(tri2d[1, 1]))),
            (int(round(tri2d[2, 0])), int(round(tri2d[2, 1])))]


# --------------------------- public API ------------------------------------

def render_mesh_to_image(
    file_path: str | Path,
    size: Tuple[int, int] = (512, 512),
    bg_rgba: Tuple[int, int, int, int] = (24, 24, 28, 0),
    face_rgb: Tuple[int, int, int] = (220, 220, 240),
    outline_rgb: Tuple[int, int, int] = (50, 50, 60),
    outline_width: int = 1,
    max_faces: int = 250_000,
    draw_edges: bool = True
) -> Image.Image:
    """
    Render a shaded isometric preview using PIL. No OpenGL required.

    - Caps faces for performance (max_faces).
    - Guarantees triangle fill; if mesh is faceless, uses convex hull.
    - Depth-sorts faces; applies simple Lambert shading; optional edge outlines.

    Returns a Pillow Image (RGBA).
    """
    path = Path(file_path)
    mesh = _load_mesh(path)

    # Cap huge meshes for speed (uniform face subsample)
    F = mesh.faces.shape[0]
    if F > max_faces:
        idx = np.linspace(0, F - 1, num=max_faces, dtype=np.int64)
        mesh = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces[idx].copy(), process=False)

    # Rotate to iso & fit
    verts_rot, _ = _project_isometric(mesh.vertices)
    verts2d_px, _, _ = _fit_to_canvas(verts_rot, size=size, margin_px=16)

    # Shading colors per-face
    face_cols = _lambert_face_colors(mesh, base_rgb=face_rgb)

    # Prepare canvas
    W, H = size
    img = Image.new("RGBA", (W, H), bg_rgba)
    draw = ImageDraw.Draw(img, "RGBA")

    # Depth sort faces
    order = _depth_sort_indices(mesh, verts_rot)

    # Draw filled triangles back-to-front
    tris2d = verts2d_px[mesh.faces]  # (F,3,2)
    for i in order:
        tri = tris2d[i]
        # Skip degenerate
        if np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])) < 1e-3:
            continue
        color = tuple(int(c) for c in face_cols[i])
        draw.polygon(_to_int_points(tri), fill=color)

    # Optional subtle edge outlines for definition
    if draw_edges:
        edge_color = outline_rgb + (220,)  # slightly translucent
        # unique edges
        f = mesh.faces[np.newaxis, :, :]
        e01 = mesh.faces[:, [0, 1]]
        e12 = mesh.faces[:, [1, 2]]
        e20 = mesh.faces[:, [2, 0]]
        edges = np.vstack((e01, e12, e20))
        # sort rows so undirected edges collapse on unique
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0)
        v2 = verts2d_px
        for a, b in edges:
            p0 = (int(round(v2[a, 0])), int(round(v2[a, 1])))
            p1 = (int(round(v2[b, 0])), int(round(v2[b, 1])))
            draw.line([p0, p1], fill=edge_color, width=outline_width)

    return img


# --------------------------- CLI (optional) --------------------------------

def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="Render shaded isometric preview (PIL)")
    ap.add_argument("input", help="Path to .stl/.obj/etc")
    ap.add_argument("-o", "--out", help="Output image (PNG)", default=None)
    ap.add_argument("--size", type=int, nargs=2, default=(512, 512), help="Width Height")
    args = ap.parse_args()

    im = render_mesh_to_image(args.input, size=tuple(args.size))
    out = args.out or (str(Path(args.input).with_suffix("")) + "_preview.png")
    im.save(out)
    print(f"Saved {out}")

if __name__ == "__main__":
    _cli()
