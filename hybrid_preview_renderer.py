# hybrid_preview_renderer.py
# GPU-first (ModernGL) shaded preview that mimics Windows Explorer.
# Falls back to robust CPU renderer (solid_renderer) when GPU is unavailable.
#
# Usage:
#   from hybrid_preview_renderer import render_preview_auto
#   img = render_preview_auto("E:/models/part.stl", size=(384,384))
#   # returns a Pillow Image (RGBA)

from __future__ import annotations
from pathlib import Path
from typing import Tuple
import math
import numpy as np
import trimesh
from PIL import Image

# ---- optional GPU dep ----
try:
    import moderngl
    _HAS_MGL = True
except Exception:
    moderngl = None
    _HAS_MGL = False

# ---- CPU fallback renderer (your module) ----
from solid_renderer import render_mesh_to_image as cpu_render

# ---------------- Windows-style preset ----------------
WINDOWS_BG_RGBA   = (18, 18, 22, 0)         # subtle dark gray bg
WINDOWS_BASE_RGB  = (200, 200, 210)         # neutral gray (Explorer-ish)
WINDOWS_AMBIENT   = 0.22                    # soft ambient
WINDOWS_LIGHT_DIR = (0.50, 0.55, 0.70)      # key light

# ---------------- math helpers ----------------
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
    return trimesh.creation.icosphere(subdivisions=1, radius=1.0)

def _project_iso_and_fit(verts: np.ndarray, size: Tuple[int,int], margin: int = 16):
    """Rotate to isometric and fit to clip-like [-1,1] via percentile bounding (robust to outliers).
       Returns positions already in NDC-like space suitable for a pass-through shader."""
    W, H = size
    R = _iso_matrix()
    v = verts.astype(np.float32) @ R.T
    # percentile bbox on XY for robustness
    lo = np.percentile(v[:, :2], 1.0, axis=0)
    hi = np.percentile(v[:, :2], 99.0, axis=0)
    span = float(max(hi - lo))
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    # scale to ~(-0.9..0.9) box, preserve aspect by using span
    scale = 1.8 / span
    xy = (v[:, :2] - lo[None, :]) * scale - 0.9
    # center within aspect if needed
    # map Z similarly (keep relative depth)
    z = v[:, 2]
    z = (z - np.percentile(z, 50)) / (np.percentile(z, 99) - np.percentile(z, 1) + 1e-6)
    z = np.clip(z, -1.0, 1.0).astype(np.float32)
    return np.column_stack([xy.astype(np.float32), z])

# ---------------- GPU path ----------------
def _gpu_render(file_path: Path,
                size: Tuple[int,int] = (512,512),
                max_faces: int = 150_000,
                msaa_samples: int = 8) -> Image.Image:
    if not _HAS_MGL:
        raise RuntimeError("ModernGL not available")

    W, H = size
    mesh = _safe_load_mesh(file_path)

    # Face cap (memory & speed safety)
    F = int(getattr(mesh, "faces", np.empty((0,3))).shape[0])
    if F == 0:
        mesh = mesh.convex_hull
        F = int(mesh.faces.shape[0])
    if F > max_faces:
        rng = np.random.default_rng(42)
        sel = rng.choice(F, size=max_faces, replace=False)
        mesh = trimesh.Trimesh(vertices=mesh.vertices.copy(),
                               faces=mesh.faces[sel].copy(),
                               process=False)

    # Ensure vertex normals (for smooth shading)
    try:
        need_vn = not hasattr(mesh, "vertex_normals") or mesh.vertex_normals is None \
                  or len(mesh.vertex_normals) != len(mesh.vertices)
        if need_vn:
            mesh.fix_normals()
    except Exception:
        pass

    # Pre-transform to NDC-like clip space on CPU (keeps shader simple)
    verts_ndc = _project_iso_and_fit(mesh.vertices, size)
    positions = verts_ndc.astype('f4', copy=False)
    normals   = getattr(mesh, "vertex_normals", None)
    if normals is None or len(normals) != len(mesh.vertices):
        # fallback to face normals expanded (rare)
        tri = positions[mesh.faces]
        fn  = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]).astype('f4')
        fn /= (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-9)
        positions = tri.reshape(-1,3).astype('f4')
        normals   = np.repeat(fn, 3, axis=0).astype('f4')
        indices   = None
    else:
        indices = mesh.faces.astype('i4', copy=False).ravel()
        # rotate normals to iso orient (ignore scale; already in object space)
        R = _iso_matrix().astype('f4')
        normals = (normals.astype('f4') @ R.T)

    # Build ModernGL context & FBO (8-bit RGBA + MSAA)
    ctx = moderngl.create_standalone_context(require=330)
    samples = 8 if msaa_samples >= 8 else 4
    fbo = ctx.simple_framebuffer((W, H), components=4, samples=samples)
    fbo.use()
    ctx.viewport = (0, 0, W, H)
    ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
    ctx.front_face = 'ccw'
    ctx.cull_face  = 'back'
    fbo.clear(WINDOWS_BG_RGBA[0]/255.0, WINDOWS_BG_RGBA[1]/255.0,
              WINDOWS_BG_RGBA[2]/255.0, WINDOWS_BG_RGBA[3]/255.0)

    # Windows-style smooth shading shader (Gouraud-like)
    fs = """
        #version 330
        in vec3 v_nrm;
        out vec4 f_color;
        uniform vec3 u_light = vec3(%f, %f, %f);
        uniform vec3 u_base  = vec3(%f, %f, %f);
        uniform float u_ambient = %f;
        void main() {
            vec3 n = normalize(v_nrm);
            float ndl = max(dot(n, normalize(u_light)), 0.0);
            float shade = clamp(u_ambient + (1.0 - u_ambient) * ndl, u_ambient, 1.0);
            vec3 rgb = clamp(u_base * shade + pow(ndl, 64.0) * 0.06, 0.0, 1.0);
            rgb = pow(rgb, vec3(1.0/2.2));  // sRGB-ish gamma
            f_color = vec4(rgb, 1.0);
        }
    """ % (
        WINDOWS_LIGHT_DIR[0], WINDOWS_LIGHT_DIR[1], WINDOWS_LIGHT_DIR[2],
        WINDOWS_BASE_RGB[0]/255.0, WINDOWS_BASE_RGB[1]/255.0, WINDOWS_BASE_RGB[2]/255.0,
        WINDOWS_AMBIENT
    )

    prog = ctx.program(
        vertex_shader="""
            #version 330
            in vec3 in_pos;
            in vec3 in_nrm;
            out vec3 v_nrm;
            void main() {
                gl_Position = vec4(in_pos, 1.0);   // already in clip-ish space
                v_nrm = normalize(in_nrm);
            }
        """,
        fragment_shader=fs
    )

    # Buffers + draw
    if indices is not None:
        vbo = ctx.buffer(np.hstack([positions, normals]).astype('f4').tobytes())
        ibo = ctx.buffer(indices.tobytes())
        vao = ctx.vertex_array(prog, [(vbo, '3f 3f', 'in_pos', 'in_nrm')], index_buffer=ibo)
        vao.render(moderngl.TRIANGLES)
    else:
        vbo = ctx.buffer(np.hstack([positions, normals]).astype('f4').tobytes())
        vao = ctx.vertex_array(prog, [(vbo, '3f 3f', 'in_pos', 'in_nrm')])
        vao.render(moderngl.TRIANGLES)

    # Readback RGBA8
    pixels = fbo.read(components=4, alignment=1)
    img = Image.frombytes('RGBA', (W, H), pixels).transpose(Image.FLIP_TOP_BOTTOM)

    # cleanup
    vao.release(); vbo.release()
    if indices is not None:
        ibo.release()
    prog.release(); fbo.release(); ctx.release()

    return img

# ---------------- Public API ----------------
def render_preview_auto(file_path: str | Path,
                        size: Tuple[int,int]=(512,512),
                        max_faces_gpu: int = 150_000,
                        max_faces_cpu: int = 120_000,
                        prefer_gpu: bool = True) -> Image.Image:
    """
    Try GPU (ModernGL) with Windows-style shading.
    On any failure / missing driver, fall back to CPU PIL renderer.
    Returns a Pillow Image (RGBA).
    """
    path = Path(file_path)
    if prefer_gpu and _HAS_MGL:
        try:
            return _gpu_render(path, size=size, max_faces=max_faces_gpu, msaa_samples=8)
        except Exception:
            # fall through to CPU
            pass
    return cpu_render(path, size=size, face_rgb=WINDOWS_BASE_RGB, bg_rgba=WINDOWS_BG_RGBA, max_faces=max_faces_cpu)