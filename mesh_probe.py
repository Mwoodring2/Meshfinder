#!/usr/bin/env python3
"""
mesh_probe.py — Generate a lightweight debug report for large STL/OBJ meshes,
with an optional shaded isometric thumbnail (no OpenGL).

Usage:
  python mesh_probe.py --in "E:\path\big_model.stl" --out report.json --thumb preview.png --max-faces 150000

Requires: numpy, pillow, trimesh
"""
import argparse, json, math, sys, traceback
from pathlib import Path
import numpy as np

try:
    import trimesh
except Exception as e:
    print("ERROR: trimesh not available:", e, file=sys.stderr); sys.exit(2)
try:
    from PIL import Image, ImageDraw
except Exception:
    Image = None; ImageDraw = None

# ---------- isometric projection helpers ----------
def _iso_matrix():
    rz = math.radians(45.0); rx = math.radians(35.26438968)
    c,s = math.cos, math.sin
    Rz = np.array([[c(rz),-s(rz),0],[s(rz),c(rz),0],[0,0,1]], dtype=np.float32)
    Rx = np.array([[1,0,0],[0,c(rx),-s(rx)],[0,s(rx),c(rx)]], dtype=np.float32)
    return Rx @ Rz
def _project_iso(verts): return (verts.astype(np.float32) @ _iso_matrix().T)
def _fit_percentile(rot_verts, size=(512,512), margin=16, lo=1.0, hi=99.0):
    W,H = size; xy = rot_verts[:, :2]
    lo_xy = np.percentile(xy, lo, axis=0); hi_xy = np.percentile(xy, hi, axis=0)
    span = max(hi_xy - lo_xy).item() or 1.0
    scale = float(min(W-2*margin, H-2*margin) / span)
    xy_scaled = (xy - lo_xy[None,:]) * scale + margin
    bb0 = xy_scaled.min(axis=0); bb1 = xy_scaled.max(axis=0)
    center = (np.array([W,H], np.float32) - (bb0 + bb1)) * 0.5
    return xy_scaled + center, scale, center

# ---------- robust loader ----------
def _ensure_trimesh(m):
    if isinstance(m, trimesh.Scene): m = trimesh.util.concatenate(m.dump())
    if isinstance(m, trimesh.points.PointCloud):
        try: m = m.convex_hull
        except Exception: pass
    if not isinstance(m, trimesh.Trimesh):
        try: m = trimesh.Trimesh(vertices=np.asarray(m.vertices), faces=np.asarray(m.faces))
        except Exception: pass
    if not isinstance(m, trimesh.Trimesh):
        m = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    return m

def load_large_tolerant(path: Path):
    info = {"tried": [], "mode": None, "warnings": []}
    mesh = None
    for process in (False, True):
        try:
            info["tried"].append(f"process={process}")
            m = trimesh.load(str(path), force='mesh', process=process, maintain_order=True)
            mesh = _ensure_trimesh(m); info["mode"] = f"process={process}"
            break
        except Exception as e:
            info["warnings"].append(f"load failure (process={process}): {repr(e)}")
    if mesh is None:
        raise RuntimeError("Unable to load mesh in any mode")
    # quick cleanups
    for fn in ("remove_degenerate_faces","remove_duplicate_faces","rezero"):
        try: getattr(mesh, fn)()
        except Exception: pass
    # ensure float32 verts
    try: mesh.vertices = mesh.vertices.astype(np.float32, copy=False)
    except Exception: pass
    return mesh, info

# ---------- shading & raster ----------
def tri_normals(rot_verts, faces):
    tri = rot_verts[faces]
    n = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0])
    n /= (np.linalg.norm(n, axis=1) + 1e-9)[:,None]
    return n
def depth_order(rot_verts, faces):
    z = rot_verts[faces][:,:,2].mean(axis=1)
    return np.argsort(z)[::-1]
def to_int_pts(tri2d): return [tuple(map(lambda x:int(round(x)), tri2d[i])) for i in range(3)]

def render_thumbnail(mesh, out_png, size=(512,512), max_faces=150_000,
                     bg=(24,24,28,0), base=(220,220,240), edge=(60,60,70,220)):
    if Image is None: return "PIL not available; skipping thumbnail"
    F = int(getattr(mesh, "faces", np.empty((0,3))).shape[0])
    if F == 0:
        try: mesh = mesh.convex_hull; F = int(mesh.faces.shape[0])
        except Exception: return "No faces to render"
    if F > max_faces:
        sel = np.random.default_rng(42).choice(F, size=max_faces, replace=False)
        mesh = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces[sel].copy(), process=False)
    v_rot = _project_iso(mesh.vertices)
    v2d, _, _ = _fit_percentile(v_rot, size=size)
    n = tri_normals(v_rot, mesh.faces)
    ld = np.array([0.577,0.577,0.577], np.float32); ld/= (np.linalg.norm(ld)+1e-9)
    dots = np.clip(n @ ld, 0.0, 1.0); intens = np.clip(0.2 + 0.8*dots, 0.2, 1.0)
    cols = (np.array(base, np.float32)[None,:] * intens[:,None]).astype(np.uint8)

    img = Image.new("RGBA", size, bg); draw = ImageDraw.Draw(img, "RGBA")
    order = depth_order(v_rot, mesh.faces); tris2d = v2d[mesh.faces]
    for i in order:
        tri = tris2d[i]
        area2 = np.cross(tri[1]-tri[0], tri[2]-tri[0])
        if abs(area2) < 0.5: continue
        draw.polygon(to_int_pts(tri), fill=tuple(int(c) for c in cols[i]))
    # edges
    e = np.vstack((mesh.faces[:,[0,1]], mesh.faces[:,[1,2]], mesh.faces[:,[2,0]]))
    e.sort(axis=1); e = np.unique(e, axis=0)
    for a,b in e:
        p0 = (int(round(v2d[a,0])), int(round(v2d[a,1])))
        p1 = (int(round(v2d[b,0])), int(round(v2d[b,1])))
        draw.line([p0,p1], fill=edge, width=1)
    img.save(out_png); return f"Saved {out_png}"

# ---------- probe ----------
def probe_mesh(path: Path, sample_limit=2_000_000):
    rep = {"file": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else None,
           "loader": {}, "mesh": {}, "components": [], "warnings": [], "errors": []}
    if not path.exists(): rep["errors"].append("File does not exist"); return rep
    try:
        mesh, loader_info = load_large_tolerant(path); rep["loader"] = loader_info
    except Exception as e:
        rep["errors"].append(f"load failure: {repr(e)}"); return rep
    V = int(getattr(mesh, "vertices", np.empty((0,3))).shape[0])
    F = int(getattr(mesh, "faces", np.empty((0,3))).shape[0])
    ext = list(map(float, getattr(mesh, "extents", [0,0,0]))) if hasattr(mesh,"extents") else None
    rep["mesh"].update({"vertices": V, "faces": F, "extents": ext, "watertight": bool(getattr(mesh,"is_watertight", False))})
    try:
        faces_to_check = mesh.faces
        if F > sample_limit:
            idx = np.random.default_rng(0).choice(F, size=sample_limit, replace=False)
            faces_to_check = faces_to_check[idx]
        tri = mesh.vertices[faces_to_check]
        edges = np.stack([tri[:,1]-tri[:,0], tri[:,2]-tri[:,1], tri[:,0]-tri[:,2]], axis=1)
        lengths = np.linalg.norm(edges, axis=2)
        degenerate = (lengths < 1e-9).any(axis=1)
        rep["mesh"]["degenerate_faces_percent"] = float(degenerate.mean()*100.0)
    except Exception as e:
        rep["warnings"].append(f"degeneracy check failed: {repr(e)}")
    try:
        comps = mesh.split(only_watertight=False)
        comps = sorted(comps, key=lambda c: getattr(c,"faces",np.empty((0,3))).shape[0], reverse=True)[:5]
        for c in comps:
            rep["components"].append({"faces": int(c.faces.shape[0]),
                                      "vertices": int(c.vertices.shape[0]),
                                      "extents": list(map(float, getattr(c,"extents",[0,0,0])))})
    except Exception as e:
        rep["warnings"].append(f"component split failed: {repr(e)}")
    return rep

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Path to STL/OBJ")
    ap.add_argument("--out", dest="out_json", default="report.json", help="Write JSON report here")
    ap.add_argument("--thumb", dest="thumb", default=None, help="Optional PNG thumbnail path")
    ap.add_argument("--max-faces", dest="max_faces", type=int, default=150000, help="Face cap for thumbnail")
    args = ap.parse_args()
    path = Path(args.inp)
    rep = probe_mesh(path)
    Path(args.out_json).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"Report written: {args.out_json}")
    if args.thumb:
        try:
            mesh, _ = load_large_tolerant(path)
            print(render_thumbnail(mesh, args.thumb, size=(512,512), max_faces=args.max_faces))
        except Exception as e:
            print("Thumbnail generation failed:", e)
            traceback.print_exc()

if __name__ == "__main__":
    main()
