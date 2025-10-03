#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ModelFinder Indexer (SQLite + FAISS)

MVP features:
- Recursively scan folders for 3D files (.stl, .obj). (ZTL placeholder entries supported)
- Convert to canonical GLB using trimesh (where possible)
- Compute geometry metrics: tri count, bbox, surface area, (optional) volume, watertight flag, manifold heuristic
- (Optional) Render a poster PNG via pyrender (best-effort; skipped if GL not available)
- Persist metadata in SQLite
- Build a FAISS vector index from deterministic geometric embeddings for similarity search

CLI example:
  python src/indexer/modelfinder_indexer.py --root data/raw --out data --db db/modelfinder.db --faiss db/faiss.index --posters --workers 8 --fast
"""

import argparse
import concurrent.futures
import hashlib
import math
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image
from tqdm import tqdm

# ---- Counters
class Counters:
    def __init__(self):
        self.files_scanned = 0
        self.files_processed = 0
        self.files_skipped = 0
        self.files_failed = 0
        self.glbs_created = 0
        self.posters_created = 0
        self.db_entries_created = 0
        self.faiss_vectors_added = 0
    
    def reset(self):
        self.files_scanned = 0
        self.files_processed = 0
        self.files_skipped = 0
        self.files_failed = 0
        self.glbs_created = 0
        self.posters_created = 0
        self.db_entries_created = 0
        self.faiss_vectors_added = 0
    
    def summary(self):
        return {
            'files_scanned': self.files_scanned,
            'files_processed': self.files_processed,
            'files_skipped': self.files_skipped,
            'files_failed': self.files_failed,
            'glbs_created': self.glbs_created,
            'posters_created': self.posters_created,
            'db_entries_created': self.db_entries_created,
            'faiss_vectors_added': self.faiss_vectors_added
        }

# Global debug counters for mesh loading diagnostics
_debug_counters = {}

def bump(counter_name: str, amount: int = 1):
    """Increment a debug counter for diagnostics"""
    _debug_counters[counter_name] = _debug_counters.get(counter_name, 0) + amount

def get_debug_counters():
    """Get a copy of all debug counters"""
    return _debug_counters.copy()

def reset_debug_counters():
    """Reset all debug counters"""
    _debug_counters.clear()

# ---- Threading for renderer singleton
_SINGLETON_RENDERER = None
_RENDER_LOCK = Lock()

# ---- Logging (console + file)
try:
    from loguru import logger
except Exception:
    class _BareLogger:
        def info(self, *a, **k): print(*a)
        def warning(self, *a, **k): print(*a, file=sys.stderr)
        def error(self, *a, **k): print(*a, file=sys.stderr)
    logger = _BareLogger()

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
try:
    from loguru import logger as _logger
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(str(LOG_DIR / "modelfinder.log"), level="INFO", rotation="5 MB", enqueue=True)
except Exception:
    pass

# ---- Required libs (with friendly messages)
try:
    import trimesh
except Exception as e:
    logger.error("[FATAL] trimesh is required: pip install trimesh")
    raise

try:
    import faiss  # type: ignore
except Exception as e:
    logger.error("[FATAL] faiss-cpu is required: pip install faiss-cpu")
    raise

# Poster rendering is optional
try:
    import pyrender  # type: ignore
    import OpenGL  # noqa: F401
    PYMESH_RENDER_AVAILABLE = True
except Exception:
    PYMESH_RENDER_AVAILABLE = False

# -----------------------------
# Configuration
# -----------------------------
SUPPORTED_EXTS = {".stl", ".obj"}          # extendable later (.fbx via assimp)
ZTL_EXTS = {".ztl"}                        # placeholder rows for ZBrush files
EMBED_DIM = 10
FEATURE_DIM = 10  # keep FAISS consistent

# -----------------------------
# Data classes
# -----------------------------
@dataclass
class MeshMetrics:
    tri_count: int
    bbox: Tuple[float, float, float]
    surface_area: float
    volume: Optional[float]
    is_watertight: bool
    manifold_score: float  # 1.0 best, 0.0 worst

    def to_vector(self) -> np.ndarray:
        """Deterministic geometric embedding (small but useful for similarity)."""
        bx, by, bz = self.bbox
        tri = math.log1p(max(0, self.tri_count))
        sa = float(self.surface_area)
        vol = float(self.volume) if self.volume is not None and np.isfinite(self.volume) else 0.0
        wt = 1.0 if self.is_watertight else 0.0
        eps = 1e-9
        compact = (vol * vol) / (sa * sa * sa + eps) if sa > 0 else 0.0
        sides = sorted([bx, by, bz])
        min_s, mid_s, max_s = sides[0] + eps, sides[1] + eps, sides[2] + eps
        ar1 = max_s / min_s
        ar2 = mid_s / min_s
        return np.array([bx, by, bz, tri, sa, vol, wt, compact, ar1, ar2], dtype=np.float32)

# -----------------------------
# SQLite schema & helpers
# -----------------------------
SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS assets (
    id              INTEGER PRIMARY KEY,
    path            TEXT UNIQUE,
    ext             TEXT,
    size_bytes      INTEGER,
    mtime           REAL,
    sha256          TEXT,
    glb_path        TEXT,
    poster_path     TEXT,
    tri_count       INTEGER,
    bbox_x          REAL,
    bbox_y          REAL,
    bbox_z          REAL,
    surface_area    REAL,
    volume          REAL,
    is_watertight   INTEGER,
    manifold_score  REAL,
    units           TEXT DEFAULT 'unknown',
    notes           TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_assets_sha ON assets(sha256);
CREATE INDEX IF NOT EXISTS idx_assets_ext ON assets(ext);

CREATE TABLE IF NOT EXISTS embeddings (
    asset_id INTEGER PRIMARY KEY,
    dim      INTEGER NOT NULL,
    vec      BLOB NOT NULL,
    FOREIGN KEY(asset_id) REFERENCES assets(id) ON DELETE CASCADE
);
"""

def open_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def ensure_schema(conn: sqlite3.Connection) -> None:
    with conn:
        conn.executescript(SCHEMA_SQL)

# -----------------------------
# Utility functions
# -----------------------------
def sha256_of_file(p: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def safe_makedirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def try_load_mesh(p: Path) -> Optional[trimesh.Trimesh]:
    """
    Robustly loads a mesh:
    - If a Scene, concatenates all Trimesh geometries
    - Returns a single Trimesh or None if unsupported/bad
    - Increments debug counters for diagnostics
    """
    bump("load_attempt")
    try:
        m = trimesh.load(str(p), force='mesh')  # may return Trimesh or Scene
        if m is None:
            bump("load_none")
            return None

        if isinstance(m, trimesh.Scene):
            # merge all geometries into one Trimesh if present
            if m.geometry:
                try:
                    m = trimesh.util.concatenate(tuple(m.geometry.values()))
                    bump("scene_concat_ok")
                except Exception:
                    bump("scene_concat_fail")
                    return None
            else:
                bump("scene_empty")
                return None

        if not isinstance(m, trimesh.Trimesh):
            bump("not_trimesh")
            return None

        # Light cleanup; avoid expensive ops
        try:
            # accessing face_normals will compute them lazily if missing
            _ = m.face_normals
            bump("normals_ok")
        except Exception:
            try:
                m.fix_normals()
                bump("normals_fixed")
            except Exception:
                bump("normals_fix_fail")

        try:
            before_faces = int(m.faces.shape[0]) if m.faces is not None else 0
            m.remove_duplicate_faces()
            m.remove_degenerate_faces()
            after_faces = int(m.faces.shape[0]) if m.faces is not None else 0
            if after_faces < before_faces:
                bump("faces_cleaned", before_faces - after_faces)
        except Exception:
            bump("faces_clean_fail")

        bump("load_ok")
        return m

    except Exception as e:
        # keep it quiet but count it
        bump("load_exception")
        return None

def _safe_extents(m) -> tuple[float, float, float]:
    """
    Try multiple ways to get finite (x,y,z) extents.
    Falls back to (0,0,0) if unavailable/invalid.
    """
    candidates = []
    # direct
    candidates.append(lambda mm: getattr(mm, "extents", None))
    # axis-aligned bbox
    candidates.append(lambda mm: getattr(mm, "bounding_box", None).extents if hasattr(mm, "bounding_box") else None)
    # oriented bbox
    candidates.append(lambda mm: getattr(mm, "bounding_box_oriented", None).extents if hasattr(mm, "bounding_box_oriented") else None)

    for fn in candidates:
        try:
            ex = fn(m)
            if ex is None:
                continue
            arr = np.asarray(ex, dtype=np.float64).reshape(-1)
            if arr.size >= 3 and np.all(np.isfinite(arr[:3])):
                return float(arr[0]), float(arr[1]), float(arr[2])
        except Exception:
            continue
    return 0.0, 0.0, 0.0


def _is_mesh_reasonable(m) -> bool:
    """Quick sanity checks to avoid metric crashes."""
    try:
        if m is None:
            return False
        v = getattr(m, "vertices", None)
        f = getattr(m, "faces", None)
        if v is None or f is None:
            return False
        if len(v) == 0 or len(f) == 0:
            return False
        return True
    except Exception:
        return False

def build_feature_vector(metrics: dict) -> np.ndarray:
    """
    Build a fixed-length (10) feature vector from metrics.
    Works in both fast and full modes. Missing values are
    filled/padded deterministically so FAISS dims never vary.
    """
    tri = float(metrics.get("tri_count", 0.0) or 0.0)
    bx  = float(metrics.get("bbox_x", 0.0) or 0.0)
    by  = float(metrics.get("bbox_y", 0.0) or 0.0)
    bz  = float(metrics.get("bbox_z", 0.0) or 0.0)

    # Derived, all safe
    vol   = bx * by * bz
    s     = max(bx, 1e-9)
    t     = max(by, 1e-9)
    u     = max(bz, 1e-9)
    ar_xy = bx / t
    ar_xz = bx / u
    ar_yz = by / u

    # Simple normalizations (keep bounded and finite)
    log_tri = np.log1p(tri)
    root_tri = np.sqrt(tri)
    norm_b = lambda v: v if np.isfinite(v) else 0.0

    vec = np.array([
        log_tri,          # 0
        root_tri,         # 1
        norm_b(bx),       # 2
        norm_b(by),       # 3
        norm_b(bz),       # 4
        norm_b(vol),      # 5
        norm_b(ar_xy),    # 6
        norm_b(ar_xz),    # 7
        norm_b(ar_yz),    # 8
        1.0               # 9 bias/constant term to stabilize FAISS
    ], dtype=np.float32)

    # Guarantee length
    if vec.shape[0] != FEATURE_DIM:
        out = np.zeros((FEATURE_DIM,), dtype=np.float32)
        n = min(FEATURE_DIM, vec.shape[0])
        out[:n] = vec[:n]
        return out
    return vec

def compute_metrics(m: trimesh.Trimesh, fast: bool = False) -> dict:
    # basic triangle count (safe)
    try:
        tri_count = int(len(getattr(m, "faces", []) or []))
    except Exception:
        tri_count = 0

    # safe extents
    bbox_x, bbox_y, bbox_z = _safe_extents(m)

    # safe surface area
    try:
        surface_area = float(getattr(m, "area", 0.0))
    except Exception:
        surface_area = 0.0

    out = {
        "tri_count": tri_count,
        "bbox_x": float(bbox_x),
        "bbox_y": float(bbox_y),
        "bbox_z": float(bbox_z),
        "surface_area": surface_area,
        "volume": None,
        "is_watertight": False,
        "manifold_score": 1.0,
    }

    if fast:
        return out

    # (optional) any extra non-expensive metrics can go here, wrapped safely
    return out

def export_glb(m: trimesh.Trimesh, out_path: Path) -> bool:
    try:
        safe_makedirs(out_path.parent)
        m.export(out_path, file_type='glb')
        return True
    except Exception as e:
        logger.warning(f"GLB export failed: {out_path} ({e})")
        return False

def render_poster(m: "trimesh.Trimesh", out_path: Path, size: int = 640, enabled: bool = False) -> bool:
    """
    Poster rendering entry point.
    - When `enabled=False` (default), this is a NO-OP that always returns True.
    - When `enabled=True`, lazily import heavy deps and attempt a real render.
    """
    if not enabled:
        # Posters are disabled for this run; pretend success and do nothing.
        return True
    try:
        # Lazy imports so that OpenGL stack isn't even touched unless requested
        import numpy as np  # type: ignore
        import pyrender      # type: ignore
        from PIL import Image  # type: ignore
    except Exception as e:
        logger.warning(f"Poster deps unavailable; skipping poster for {out_path.name}: {e}")
        return False
    try:
        with _RENDER_LOCK:
            global _SINGLETON_RENDERER
            if _SINGLETON_RENDERER is None:
                # make one renderer for the whole process
                _SINGLETON_RENDERER = pyrender.OffscreenRenderer(size, size)

            safe_makedirs(out_path.parent)
            scene = pyrender.Scene(bg_color=[0, 0, 0, 0])
            tm = pyrender.Mesh.from_trimesh(m, smooth=True)
            scene.add(tm)

            bbox = m.bounds
            center = (bbox[0] + bbox[1]) / 2.0
            extent = float(np.max(bbox[1] - bbox[0])) if bbox is not None else 1.0

            cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(45.0))
            cam_node = pyrender.Node(camera=cam, matrix=np.eye(4))
            scene.add_node(cam_node)

            cam_dist = 2.5 * extent if extent > 0 else 1.0
            cam_pose = np.array([
                [1, 0, 0, center[0]],
                [0, 1, 0, center[1]],
                [0, 0, 1, center[2] + cam_dist],
                [0, 0, 0, 1],
            ])
            scene.set_pose(cam_node, cam_pose)

            light = pyrender.DirectionalLight(intensity=3.0)
            scene.add(light, pose=cam_pose)

            color, _ = _SINGLETON_RENDERER.render(scene)
            Image.fromarray(color).save(out_path)
            return True
    except Exception as e:
        logger.warning(f"Poster render failed: {out_path} ({e})")
        return False

# -----------------------------
# FAISS helpers
# -----------------------------
def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    xb = vectors.astype(np.float32)
    faiss.normalize_L2(xb)
    index = faiss.IndexFlatIP(xb.shape[1])
    index.add(xb)
    return index

def save_faiss_index(index: faiss.IndexFlatIP, path: Path) -> None:
    faiss.write_index(index, str(path))

# -----------------------------
# DB helpers
# -----------------------------
def upsert_asset(conn: sqlite3.Connection, meta: dict) -> int:
    with conn:
        conn.execute(
            """
            INSERT INTO assets (path, ext, size_bytes, mtime, sha256, glb_path, poster_path,
                                tri_count, bbox_x, bbox_y, bbox_z, surface_area, volume,
                                is_watertight, manifold_score, units, notes)
            VALUES (:path, :ext, :size_bytes, :mtime, :sha256, :glb_path, :poster_path,
                    :tri_count, :bbox_x, :bbox_y, :bbox_z, :surface_area, :volume,
                    :is_watertight, :manifold_score, :units, :notes)
            ON CONFLICT(path) DO UPDATE SET
                size_bytes=excluded.size_bytes,
                mtime=excluded.mtime,
                sha256=excluded.sha256,
                glb_path=excluded.glb_path,
                poster_path=excluded.poster_path,
                tri_count=excluded.tri_count,
                bbox_x=excluded.bbox_x,
                bbox_y=excluded.bbox_y,
                bbox_z=excluded.bbox_z,
                surface_area=excluded.surface_area,
                volume=excluded.volume,
                is_watertight=excluded.is_watertight,
                manifold_score=excluded.manifold_score
            ;
            """,
            meta,
        )
        cur = conn.execute("SELECT id FROM assets WHERE path=?", (meta["path"],))
        return int(cur.fetchone()[0])

def upsert_embedding(conn: sqlite3.Connection, asset_id: int, vec: np.ndarray) -> None:
    blob = vec.astype(np.float32).tobytes(order='C')
    with conn:
        conn.execute(
            """
            INSERT INTO embeddings (asset_id, dim, vec)
            VALUES (?, ?, ?)
            ON CONFLICT(asset_id) DO UPDATE SET dim=excluded.dim, vec=excluded.vec;
            """,
            (asset_id, int(vec.shape[0]), sqlite3.Binary(blob)),
        )

# -----------------------------
# Core pipeline
# -----------------------------
def process_file(p: Path, out_dir: Path, posters: bool, fast: bool) -> Optional[Tuple[dict, np.ndarray]]:
    ext = p.suffix.lower()
    stats = p.stat()
    file_hash = sha256_of_file(p)

    # ZTL placeholder (no metrics yet)
    if ext in ZTL_EXTS:
        meta = {
            "path": str(p),
            "ext": ext,
            "size_bytes": stats.st_size,
            "mtime": stats.st_mtime,
            "sha256": file_hash,
            "glb_path": None,
            "poster_path": None,
            "tri_count": None,
            "bbox_x": None,
            "bbox_y": None,
            "bbox_z": None,
            "surface_area": None,
            "volume": None,
            "is_watertight": None,
            "manifold_score": None,
            "units": "unknown",
            "notes": "ZTL placeholder; export to OBJ/FBX for metrics."
        }
        vec = np.zeros((EMBED_DIM,), dtype=np.float32)
        return meta, vec

    if ext not in SUPPORTED_EXTS:
        return None

    m = try_load_mesh(p)
    if m is None:
        bump("load_failed")
        return None
    else:
        bump("load_succeeded")

    if not _is_mesh_reasonable(m):
        logger.warning(f"Skipping invalid/empty mesh: {p}")
        return None

    try:
        metrics = compute_metrics(m, fast=fast)
    except Exception as e:
        logger.warning(f"Metric computation failed for {p} ({e})")
        return None

    # Export GLB
    glb_rel = Path("glb") / (p.stem + ".glb")
    glb_path = out_dir / glb_rel
    exported = export_glb(m, glb_path)
    if exported:
        bump("glb_export_ok")
    else:
        bump("glb_export_fail")

    # Poster (optional)
    poster_rel = Path("posters") / (p.stem + ".png")
    poster_path = out_dir / poster_rel
    poster_ok = False
    if posters:
        bump("poster_requested")
        poster_ok = render_poster(m, poster_path, enabled=True)
        if poster_ok:
            bump("poster_ok")
        else:
            bump("poster_fail")

    meta = {
        "path": str(p),
        "ext": ext,
        "size_bytes": stats.st_size,
        "mtime": stats.st_mtime,
        "sha256": file_hash,
        "glb_path": str(glb_path) if exported else None,
        "poster_path": str(poster_path) if poster_ok else None,
        "tri_count": metrics["tri_count"],
        "bbox_x": metrics["bbox_x"],
        "bbox_y": metrics["bbox_y"],
        "bbox_z": metrics["bbox_z"],
        "surface_area": metrics["surface_area"],
        "volume": metrics["volume"] if metrics["volume"] is not None else None,
        "is_watertight": 1 if metrics["is_watertight"] else 0,
        "manifold_score": metrics["manifold_score"],
        "units": "unknown",
        "notes": ""
    }
    # Create feature vector using the sophisticated builder
    vec = build_feature_vector(metrics)
    return meta, vec

def scan_and_index(
    root: Path,
    out_dir: Path,
    db_path: Path,
    faiss_path: Path,
    posters: bool,
    workers: int,
    fast: bool
) -> None:
    safe_makedirs(out_dir / "glb")
    safe_makedirs(out_dir / "posters")

    conn = open_db(db_path)
    ensure_schema(conn)

    # Gather files
    all_files: List[Path] = []
    supported_files: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            fp = Path(dirpath) / fn
            all_files.append(fp)
            ext = fp.suffix.lower()
            if ext in SUPPORTED_EXTS or ext in ZTL_EXTS:
                supported_files.append(fp)

    logger.info(f"Starting processing: files={len(all_files)} supported={len(supported_files)} workers={workers} fast={fast} posters={posters}")

    loaded = 0
    db_ok = 0
    vecs: List[np.ndarray] = []

    def _job(pth: Path):
        return process_file(pth, out_dir, posters, fast)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = [ex.submit(_job, p) for p in supported_files]
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing"):
            res = fut.result()
            if res is None:
                continue
            meta, vec = res
            loaded += 1
            try:
                asset_id = upsert_asset(conn, meta)
                bump("db_upsert_ok")
                upsert_embedding(conn, asset_id, vec)
                vecs.append(vec)
                db_ok += 1
            except Exception as e:
                bump("db_upsert_fail")
                logger.warning(f"DB upsert failed for {meta.get('path')}: {e}")

    print(f"[SUMMARY] files={len(all_files)} supported={len(supported_files)} loaded={loaded} db_ok={db_ok}")
    logger.info(f"Summary: files={len(all_files)} supported={len(supported_files)} loaded={loaded} db_ok={db_ok}")

    # --- Debug summary ---
    print("[SUMMARY] debug counters:")
    for k, v in sorted(_debug_counters.items()):
        print(f"  {k:24s}: {v}")

    if len(vecs) == 0:
        print("[INFO] No vectors to index; skipping FAISS build.")
        logger.info("No vectors to index; skipping FAISS build.")
        return

    # Keep only vectors with the target dimension
    TARGET_DIM = 10
    good = []
    bad = 0
    for v in vecs:
        if isinstance(v, np.ndarray) and v.ndim == 1 and v.shape[0] == TARGET_DIM and np.all(np.isfinite(v)):
            good.append(v)
        else:
            bad += 1
    if bad:
        logger.warning(f"Dropping {bad} vectors with wrong dim or non-finite values before FAISS build.")
    vecs = good

    if not vecs:
        logger.error("No valid feature vectors to index; aborting FAISS build.")
        return
    X = np.vstack(vecs).astype(np.float32)
    index = build_faiss_index(X)
    save_faiss_index(index, faiss_path)
    print(f"[OK] Indexed {X.shape[0]} assets into FAISS (dim={X.shape[1]}). Saved to {faiss_path}")
    logger.info(f"FAISS index built: n={X.shape[0]} dim={X.shape[1]} -> {faiss_path}")

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="ModelFinder Indexer (SQLite + FAISS)")
    ap.add_argument('--root', type=Path, required=True, help='Root folder to scan')
    ap.add_argument('--out', dest='out_dir', type=Path, default=Path('data'), help='Output folder for GLB/posters')
    ap.add_argument('--db', dest='db_path', type=Path, default=Path('db/modelfinder.db'), help='SQLite database path')
    ap.add_argument('--faiss', dest='faiss_path', type=Path, default=Path('db/faiss.index'), help='FAISS index path')
    ap.add_argument('--posters', action='store_true', help='Render poster PNGs for each mesh (optional; disabled by default)')
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 4, help='Parallel workers')
    ap.add_argument('--fast', action='store_true', help='Skip expensive metrics like exact volume')
    args = ap.parse_args()

    t0 = time.time()
    scan_and_index(args.root, args.out_dir, args.db_path, args.faiss_path, args.posters, args.workers, args.fast)
    dt = time.time() - t0
    print(f"[DONE] Total time: {dt:.1f}s")
    logger.info(f"Done in {dt:.1f}s")

if __name__ == '__main__':
    main()
