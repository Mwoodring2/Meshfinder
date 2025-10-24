# solid_renderer.py (v2) – resilient PIL renderer for big/dirty meshes
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional
import math, numpy as np
from PIL import Image, ImageDraw
import trimesh

# ---------------- basics ----------------

def _iso_matrix() -> np.ndarray:
    rz = math.radians(45.0)
    rx = math.radians(35.26438968)
    Rz = np.array([[math.cos(rz),-math.sin(rz),0],
                   [math.sin(rz), math.cos(rz),0],
                   [0,0,1]], dtype=np.float32)
    Rx = np.array([[1,0,0],
                   [0, math.cos(rx),-math.sin(rx)],
                   [0, math.sin(rx), math.cos(rx)]], dtype=np.float32)
    return Rx @ Rz

def _project_iso(verts: np.ndarray) -> np.ndarray:
    R = _iso_matrix()
    return (verts.astype(np.float32) @ R.T)

def _fit_percentile(rot_verts: np.ndarray, size: Tuple[int,int],
                    margin: int = 16, lo: float = 1.0, hi: float = 99.0):
    """Fit to canvas using percentile bbox (robust to stray vertices)."""
    W,H = size
    xy = rot_verts[:, :2]
    lo_xy = np.percentile(xy, lo, axis=0)
    hi_xy = np.percentile(xy, hi, axis=0)
    span = max(hi_xy - lo_xy).item()
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    usable = np.array([W - 2*margin, H - 2*margin], dtype=np.float32)
    scale = float(min(usable) / span)
    xy_scaled = (xy - lo_xy[None,:]) * scale + margin
    bb0 = xy_scaled.min(axis=0); bb1 = xy_scaled.max(axis=0)
    center_offset = (np.array([W,H], np.float32) - (bb0 + bb1))*0.5
    xy_final = xy_scaled + center_offset
    return xy_final, scale, center_offset

# ------------- robust loading ------------

def _ensure_trimesh(m) -> trimesh.Trimesh:
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(m.dump())
    if isinstance(m, trimesh.points.PointCloud):
        # point cloud → hull
        try: m = m.convex_hull
        except Exception: pass
    if not isinstance(m, trimesh.Trimesh):
        try:
            m = trimesh.Trimesh(vertices=np.asarray(m.vertices), faces=np.asarray(m.faces))
        except Exception:
            v = np.asarray(getattr(m, "vertices", []))
            if v.size:
                try: m = trimesh.Trimesh(vertices=v)
                except Exception: pass
    if not isinstance(m, trimesh.Trimesh):
        m = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    return m

def _load_large_tolerant(path: Path) -> trimesh.Trimesh:
    # fast path: skip heavy 'process' work first
    try:
        m = trimesh.load(str(path), force='mesh', process=False, maintain_order=True)
    except Exception:
        m = trimesh.load(str(path), force='mesh', process=True)
    m = _ensure_trimesh(m)

    # Quick clean; avoid expensive repairs on huge meshes
    try: m.remove_degenerate_faces()
    except Exception: pass
    try: m.remove_duplicate_faces()
    except Exception: pass

    if getattr(m, "faces", np.empty((0,3))).shape[0] == 0:
        try: m = m.convex_hull
        except Exception:
            m = trimesh.creation.icosphere(subdivisions=1, radius=1.0)

    # Normalize dtype/material
    try:
        m.vertices = m.vertices.astype(np.float32, copy=False)
    except Exception: pass

    # Recenter
    try: m.rezero()
    except Exception: pass

    return m

# ------------- shading/draw --------------

def _per_face_colors(tri_normals: np.ndarray,
                     base_rgb=(220,220,240),
                     light_dir=(0.577,0.577,0.577),
                     ambient=0.20) -> np.ndarray:
    ld = np.asarray(light_dir, np.float32)
    ld /= (np.linalg.norm(ld)+1e-9)
    n = tri_normals.astype(np.float32)
    dots = np.clip(n @ ld, 0.0, 1.0)
    intens = np.clip(ambient + (1.0-ambient)*dots, ambient, 1.0)
    base = np.asarray(base_rgb, np.float32)[None,:]
    return (base * intens[:,None]).astype(np.uint8)

def _triangle_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = verts[faces]                      # (F,3,3)
    n = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0])
    norm = np.linalg.norm(n, axis=1) + 1e-9
    n /= norm[:,None]
    return n

def _depth_order(rot_verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    z = rot_verts[faces][:,:,2].mean(axis=1)
    return np.argsort(z)[::-1]  # far→near

def _to_int_pts(tri2d):  # tri2d: (3,2)
    return [tuple(map(lambda x:int(round(x)), tri2d[i])) for i in range(3)]

# ------------- public API ----------------

def render_mesh_to_image(
    file_path: str | Path,
    size: Tuple[int,int]=(512,512),
    bg_rgba=(24,24,28,0),
    face_rgb=(220,220,240),
    outline_rgb=(60,60,70),
    outline_width=1,
    max_faces=250_000,
    draw_edges=True,
    debug_no_downsample=False,
    debug_force_hull=False
) -> Image.Image:
    path = Path(file_path)
    mesh = _load_large_tolerant(path)

    # Debug option: Force convex hull to rule out degenerate geometry
    if debug_force_hull:
        print(f"DEBUG: Forcing convex hull for {path.name}")
        mesh = mesh.convex_hull

    F = int(mesh.faces.shape[0])
    if F == 0:
        # silhouette fallback from hull
        mesh = mesh.convex_hull
        F = int(mesh.faces.shape[0])

    # Debug option: Disable downsampling to confirm correctness (may be slow on huge files)
    if debug_no_downsample:
        print(f"DEBUG: No downsampling enabled for {path.name} (F={F})")
    elif F > max_faces:
        print(f"DEBUG: Downsampling {F} faces to {max_faces} for {path.name}")
        sel = np.random.default_rng(42).choice(F, size=max_faces, replace=False)
        mesh = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces[sel].copy(), process=False)

    # rotate & fit
    v_rot = _project_iso(mesh.vertices)
    v2d, _, _ = _fit_percentile(v_rot, size=size, margin=16, lo=1.0, hi=99.0)

    # per-face normals (no dependency on precomputed vertex normals)
    triN = _triangle_normals(v_rot, mesh.faces)
    cols = _per_face_colors(triN, base_rgb=face_rgb)

    # canvas
    W,H = size
    img = Image.new("RGBA", (W,H), bg_rgba)
    draw = ImageDraw.Draw(img, "RGBA")

    order = _depth_order(v_rot, mesh.faces)
    tris2d = v2d[mesh.faces]  # (F,3,2)

    # draw filled faces
    for i in order:
        tri = tris2d[i]
        # skip degenerate
        area2 = np.cross(tri[1]-tri[0], tri[2]-tri[0])
        if abs(area2) < 0.5:  # tiny
            continue
        draw.polygon(_to_int_pts(tri), fill=tuple(int(c) for c in cols[i]))

    # optional outline
    if draw_edges:
        # unique edges
        e = np.vstack((mesh.faces[:,[0,1]], mesh.faces[:,[1,2]], mesh.faces[:,[2,0]]))
        e.sort(axis=1)
        e = np.unique(e, axis=0)
        c = outline_rgb + (220,)
        for a,b in e:
            p0 = (int(round(v2d[a,0])), int(round(v2d[a,1])))
            p1 = (int(round(v2d[b,0])), int(round(v2d[b,1])))
            draw.line([p0,p1], fill=c, width=outline_width)

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
