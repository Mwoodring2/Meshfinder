# hybrid_preview_renderer.py
# GPU-first shaded preview with ModernGL; falls back to PIL solid_renderer.
# Usage:
#   from hybrid_preview_renderer import render_preview_auto
#   img = render_preview_auto("E:/models/part.stl", size=(384,384))
#   # returns a Pillow Image (RGBA)

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional
import math
import numpy as np

# --- optional GPU deps ---
try:
    import moderngl  # pip install moderngl
    _HAS_MGL = True
except Exception:
    moderngl = None
    _HAS_MGL = False

# --- trimesh & PIL are required (PIL via solid_renderer fallback anyway) ---
import trimesh
from PIL import Image

# --- CPU fallback (your module) ---
from solid_renderer import render_mesh_to_image as cpu_render  # keep v2

# ===================== math helpers =====================

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

def _safe_load_mesh(path: Path) -> trimesh.Trimesh:
    # fast path first
    for process in (False, True):
        try:
            m = trimesh.load(str(path), force='mesh', process=process, maintain_order=True)
            if isinstance(m, trimesh.Scene):
                m = trimesh.util.concatenate(m.dump())
            if isinstance(m, trimesh.points.PointCloud):
                m = m.convex_hull
            return m
        except Exception:
            continue
    # ultimate fallback
    return trimesh.creation.icosphere(subdivisions=1, radius=1.0)

# ===================== GPU renderer =====================

def _gpu_render(file_path: str | Path,
                size: Tuple[int,int] = (512,512),
                max_faces: int = 150_000,
                msaa_samples: int = 4,
                bg_rgba = (18,18,22,0)) -> Image.Image:
    """
    GPU path: ModernGL offscreen framebuffer, flat shaded tris with z-buffer + MSAA.
    Returns Pillow Image. Raises on any GPU failure.
    """
    if not _HAS_MGL:
        raise RuntimeError("ModernGL not available")

    W, H = size
    path = Path(file_path)
    mesh = _safe_load_mesh(path)

    F = int(getattr(mesh, "faces", np.empty((0,3))).shape[0])
    if F == 0:
        mesh = mesh.convex_hull
        F = int(mesh.faces.shape[0])

    # Face cap for preview memory safety
    if F > max_faces:
        rng = np.random.default_rng(42)
        sel = rng.choice(F, size=max_faces, replace=False)
        mesh = trimesh.Trimesh(vertices=mesh.vertices.copy(),
                               faces=mesh.faces[sel].copy(),
                               process=False)
        F = int(mesh.faces.shape[0])

    # Transform to isometric view on CPU (keeps shader simple)
    verts = mesh.vertices.astype(np.float32, copy=False)
    R = _iso_matrix()
    verts_iso = verts @ R.T

    # Normalize to NDC [-1,1] with small margin
    mins = np.percentile(verts_iso, 1.0, axis=0)
    maxs = np.percentile(verts_iso, 99.0, axis=0)
    span = (maxs - mins).max().astype(np.float32)
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    scale = 1.8 / span  # leave ~10% border
    verts_ndc = (verts_iso - mins[None,:]) * scale - 0.9

    # ---- SMOOTH SHADING (Windows preset) ----
    # ensure vertex normals
    try:
        if not hasattr(mesh, "vertex_normals") or mesh.vertex_normals is None \
           or len(mesh.vertex_normals) != len(mesh.vertices):
            mesh.fix_normals()  # computes per-vertex normals
    except Exception:
        pass

    positions = verts_ndc.astype('f4', copy=False)
    normals   = getattr(mesh, "vertex_normals", None)
    if normals is None or len(normals) != len(positions):
        # fallback: use face normals expanded (still looks ok)
        tri = verts_ndc[mesh.faces]
        fn  = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]).astype('f4')
        fn /= (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-9)
        positions = tri.reshape(-1,3).astype('f4')
        normals   = np.repeat(fn, 3, axis=0).astype('f4')
        indices   = None
    else:
        indices = mesh.faces.astype('i4', copy=False).ravel()

    # Context + FBO with improved MSAA
    ctx = moderngl.create_standalone_context(require=330)
    samples = 8 if msaa_samples >= 8 else 4
    fbo = ctx.simple_framebuffer((W, H), components=4, samples=samples)
    fbo.use()
    ctx.viewport = (0, 0, W, H)
    ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
    ctx.front_face = 'ccw'
    ctx.cull_face  = 'back'
    fbo.clear(bg_rgba[0]/255.0, bg_rgba[1]/255.0, bg_rgba[2]/255.0, bg_rgba[3]/255.0)

    # build buffers / vao
    prog = ctx.program(vertex_shader="""
        #version 330
        in vec3 in_pos;
        in vec3 in_nrm;
        out vec3 v_nrm;
        void main() {
            gl_Position = vec4(in_pos, 1.0);
            v_nrm = normalize(in_nrm);
        }
    """, fragment_shader=fs)

    if indices is not None:
        vbo = ctx.buffer(np.hstack([positions, normals]).astype('f4').tobytes())
        ibo = ctx.buffer(indices.tobytes())
        vao = ctx.vertex_array(prog, [(vbo, '3f 3f', 'in_pos', 'in_nrm')], index_buffer=ibo)
    else:
        vbo = ctx.buffer(np.hstack([positions, normals]).astype('f4').tobytes())
        vao = ctx.vertex_array(prog, [(vbo, '3f 3f', 'in_pos', 'in_nrm')])

    # Draw (indexed or non-indexed triangles)
    vao.render(moderngl.TRIANGLES)

    # Resolve MSAA and read pixels
    img_data = fbo.read(components=4, alignment=1)  # bottom-up
    img = Image.frombytes('RGBA', (W, H), img_data).transpose(Image.FLIP_TOP_BOTTOM)

    # Cleanup
    vao.release(); vbo.release(); prog.release(); fbo.release(); ctx.release()
    return img

# ===================== Public API =====================

def render_preview_auto(file_path: str | Path,
                        size: Tuple[int,int]=(512,512),
                        max_faces_gpu: int = 150_000,
                        max_faces_cpu: int = 120_000,
                        prefer_gpu: bool = True) -> Image.Image:
    """
    Try GPU (ModernGL) first for crisp, anti-aliased, z-buffered shading.
    On any failure or missing driver, fall back to CPU PIL renderer (solid_renderer).
    """
    path = Path(file_path)

    if prefer_gpu and _HAS_MGL:
        try:
            return _gpu_render(path, size=size, max_faces=max_faces_gpu)
        except Exception as e:
            # Fall through to CPU
            # print(f"[GPU preview failed] {e}")  # optional logging
            pass

    # CPU fallback (robust to broken meshes; cross-platform)
    return cpu_render(path, size=size, max_faces=max_faces_cpu)
