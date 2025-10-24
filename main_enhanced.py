"""
ModelFinder — Enhanced UI Version
Professional 3-panel layout with dark theme, advanced filters, and preview window
"""
from __future__ import annotations
import os
import sys
import time
import sqlite3
import csv
import ctypes
import string
import threading
import hashlib
import shutil
from datetime import datetime
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

# Optional PIL for thumbnail generation
try:
    from PIL import Image, ImageQt  # type: ignore
    _PIL_AVAILABLE = True
except Exception:
    _PIL_AVAILABLE = False

# Optional trimesh for 3D geometry processing
try:
    import trimesh  # type: ignore
    _GEO_AVAILABLE = True
except Exception:
    _GEO_AVAILABLE = False

# Import core database functions (always needed)
from src.dataio.db import ensure_user_corrections, add_user_correction, batch_update_proposals

# Try to import proposal system components
try:
    from src.features.propose_from_reference import propose_for_rows, RowMeta
    from src.features.migrate_flow import MigrationGuardrails
    from src.ui.workers import ProposeWorker
    from src.ml.active_learning import retrain_from_corrections
    from src.utils.mesh_validation import MeshValidator
    _PROPOSAL_AVAILABLE = True
except ImportError:
    _PROPOSAL_AVAILABLE = False
    # Create placeholder classes if imports fail
    class RowMeta:
        def __init__(self, path, name, ext, tags=""):
            self.path = path
            self.name = name
            self.ext = ext
            self.tags = tags
    
    class ProposeWorker:
        def __init__(self, *args, **kwargs):
            pass
    
    class MigrationGuardrails:
        def __init__(self, *args, **kwargs):
            pass
    
    
    class MeshValidator:
        def __init__(self, *args, **kwargs):
            pass
    
    def propose_for_rows(*args, **kwargs):
        return []
    
    def batch_update_proposals(*args, **kwargs):
        pass

# Optional deps for geometry metadata & thumbnails
try:
    import trimesh
    _GEO_AVAILABLE = True
except Exception:
    _GEO_AVAILABLE = False

# Import 3D viewer
try:
    from src.ui.gl_viewer import GLViewer
    _GL_VIEWER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: 3D viewer not available: {e}")
    _GL_VIEWER_AVAILABLE = False

# --- Similarity search (TF‑IDF + FAISS) imports ---
try:
    import numpy as np
    import faiss  # type: ignore
    import joblib  # type: ignore
    from sklearn.preprocessing import normalize as _sk_normalize  # type: ignore
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    _SIM_AVAILABLE = True
except Exception:
    _SIM_AVAILABLE = False

# --- NEW: Proposal system imports ---
try:
    from src.features.propose_from_reference import propose_for_rows, RowMeta
    from src.ui.workers import ProposeWorker
    from src.dataio.db import update_proposal, batch_update_proposals
    from src.dataio.reference_parts import load_reference_parts, get_all_projects
    _PROPOSAL_AVAILABLE = True
except Exception as e:
    _PROPOSAL_AVAILABLE = False
    print(f"Warning: Proposal system not available: {e}")

APP_NAME = "ModelFinder"
DB_DIR = Path(os.environ.get("APPDATA", str(Path.home() / ".model_finder"))) / APP_NAME
DB_PATH = DB_DIR / "index.db"
SUPPORTED_EXTS = {".obj", ".fbx", ".stl", ".ma", ".mb"}

# -----------------------------
# Enhanced Table Model
# -----------------------------

class EnhancedFileTableModel(QtCore.QAbstractTableModel):
    """Enhanced table model with more columns like Torrblar"""
    
    headers = [
        "Name", "Extension", "Size (MB)", "Modified", "Tags", "Path",
        "Project #", "Project Name", "Part Name", "Proposed Name", "Conf.", "Needs Review"
    ]
    
    def __init__(self):
        super().__init__()
        self.rows: list[tuple] = []  # Currently displayed rows (filtered)
        self.all_rows: list[tuple] = []  # All loaded rows (unfiltered)

    def set_rows(self, rows: list[tuple]):
        self.beginResetModel()
        self.rows = rows
        self.all_rows = rows.copy()  # Keep a copy of all data
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.rows)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self.headers)

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        
        row = self.rows[index.row()]
        col = index.column()
        
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            # Handle different column types
            if col == 0:  # Name
                return row[1] if len(row) > 1 else ""
            elif col == 1:  # Extension
                return row[2] if len(row) > 2 else ""
            elif col == 2:  # Size (MB)
                size = row[3] if len(row) > 3 else 0
                return f"{size / (1024*1024):.2f}" if size else "0.00"
            elif col == 3:  # Modified
                mtime = row[4] if len(row) > 4 else 0
                return time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime)) if mtime else ""
            elif col == 4:  # Tags
                return row[5] if len(row) > 5 else ""
            elif col == 5:  # Path
                return row[0] if len(row) > 0 else ""
            elif col == 6:  # Project #
                return row[11] if len(row) > 11 else ""
            elif col == 7:  # Project Name
                return row[12] if len(row) > 12 else ""
            elif col == 8:  # Part Name
                return row[13] if len(row) > 13 else ""
            elif col == 9:  # Proposed Name
                return row[14] if len(row) > 14 else ""
            elif col == 10:  # Conf.
                conf = row[15] if len(row) > 15 and row[15] else None
                return f"{conf:.2f}" if conf else "0.00"
            elif col == 11:  # Needs Review
                # Simple heuristic: needs review if confidence is low or fields are missing
                conf = row[15] if len(row) > 15 and row[15] else None
                has_project = row[11] if len(row) > 11 else ""
                return "Yes" if (conf and conf < 0.66) or not has_project else "No"
        
        # Confidence-based row coloring
        elif role == QtCore.Qt.ItemDataRole.BackgroundRole:
            conf = row[15] if len(row) > 15 and row[15] else None
            has_project = row[11] if len(row) > 11 else ""
            
            if conf is not None and has_project:
                if conf >= 0.85:  # High confidence - green tint
                    return QtGui.QColor(200, 255, 200)  # Light green
                elif conf >= 0.66:  # Medium confidence - yellow tint
                    return QtGui.QColor(255, 255, 200)  # Light yellow
                else:  # Low confidence - red tint (needs review)
                    return QtGui.QColor(255, 220, 220)  # Light red
        
        return None

    def headerData(self, section, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if role == QtCore.Qt.ItemDataRole.DisplayRole and orientation == QtCore.Qt.Orientation.Horizontal:
            return self.headers[section]
        return None
    
    def flags(self, index):
        """Return item flags"""
        flags = super().flags(index)
        # Enable editing for Tags column (column 4)
        if index.column() == 4:
            flags |= QtCore.Qt.ItemFlag.ItemIsEditable
        return flags
    
    def setData(self, index, value, role=QtCore.Qt.ItemDataRole.EditRole):
        """Set data for editable items"""
        if role == QtCore.Qt.ItemDataRole.EditRole and index.column() == 4:  # Tags column
            # Update the tags in the database
            row = index.row()
            if 0 <= row < len(self.rows):
                file_path = self.rows[row][0]  # Get file path from first column
                self._update_tags_in_db(file_path, str(value))
                # Update the local data
                row_data = list(self.rows[row])
                row_data[5] = str(value)  # Update tags column (index 5 in tuple)
                self.rows[row] = tuple(row_data)
                self.dataChanged.emit(index, index, [role])
                return True
        return False
    
    def _update_tags_in_db(self, file_path, tags):
        """Update tags in database"""
        try:
            con = sqlite3.connect(DB_PATH)
            cur = con.cursor()
            cur.execute("UPDATE files SET tags = ? WHERE path = ?", (tags, file_path))
            con.commit()
            con.close()
        except Exception as e:
            print(f"Error updating tags: {e}")

# -----------------------------
# Database Functions
# -----------------------------

def ensure_db():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY,
            path TEXT UNIQUE,
            name TEXT,
            ext TEXT,
            size INTEGER,
            mtime REAL,
            tags TEXT DEFAULT '',
            tris INTEGER,
            dim_x REAL,
            dim_y REAL,
            dim_z REAL,
            volume REAL,
            watertight INTEGER,
            project_number TEXT,
            project_name TEXT,
            part_name TEXT,
            type_conf REAL,
            project_conf REAL,
            sha256 TEXT,
            status TEXT
        );
        """
    )
    # Indices
    cur.execute("CREATE INDEX IF NOT EXISTS idx_name ON files(name);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ext ON files(ext);")
    
    # Reference parts table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS project_reference_parts(
            id INTEGER PRIMARY KEY,
            project_number TEXT,
            project_name TEXT,
            part_name TEXT,
            original_label TEXT,
            description TEXT,
            quantity INTEGER DEFAULT 1,
            UNIQUE(project_number, project_name, part_name) ON CONFLICT IGNORE
        )
    """)
    
    con.commit()
    con.close()

def migrate_db_for_proposals():
    """Add proposal-related columns to the database"""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    
    # Migrate files table
    for ddl in [
        "ALTER TABLE files ADD COLUMN project_number TEXT",
        "ALTER TABLE files ADD COLUMN project_name TEXT", 
        "ALTER TABLE files ADD COLUMN part_name TEXT",
        "ALTER TABLE files ADD COLUMN proposed_name TEXT",
        "ALTER TABLE files ADD COLUMN type_conf REAL",
        "ALTER TABLE files ADD COLUMN status TEXT",
        "ALTER TABLE files ADD COLUMN license_type TEXT DEFAULT 'unknown'",
        "ALTER TABLE files ADD COLUMN asset_category TEXT DEFAULT 'unknown'",
        "ALTER TABLE files ADD COLUMN migration_dest TEXT",
        "ALTER TABLE files ADD COLUMN migration_status TEXT"
    ]:
        try:
            cur.execute(ddl)
        except Exception:
            pass  # Column already exists
    
    # Migrate project_reference_parts table (for Excel imports)
    for ddl in [
        "ALTER TABLE project_reference_parts ADD COLUMN description TEXT",
        "ALTER TABLE project_reference_parts ADD COLUMN quantity INTEGER DEFAULT 1"
    ]:
        try:
            cur.execute(ddl)
        except Exception:
            pass  # Column already exists or table doesn't exist yet
    
    con.commit()
    con.close()

def query_files(term: str, exts: set[str] | None = None, limit: int = 5000):
    term = term.strip()
    sql = "SELECT path,name,ext,size,mtime,tags,tris,dim_x,dim_y,dim_z,volume,project_number,project_name,part_name,proposed_name,type_conf,status FROM files"
    args = []
    clauses = []
    if term:
        like = f"%{term.lower()}%"
        clauses.append("(LOWER(name) LIKE ? OR LOWER(tags) LIKE ?)")
        args.extend([like, like])
    if exts:
        placeholders = ",".join(["?"] * len(exts))
        clauses.append(f"ext IN ({placeholders})")
        args.extend([e.lower() for e in exts])
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY mtime DESC LIMIT ?"
    args.append(limit)
    
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(sql, args)
    rows = cur.fetchall()
    con.close()
    return rows

def upsert_file(path: Path):
    """Add or update a file in the database"""
    try:
        stat = path.stat()
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO files (path, name, ext, size, mtime)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(path),
                path.name,
                path.suffix.lower(),
                stat.st_size,
                stat.st_mtime,
            ),
        )
        con.commit()
        con.close()
        return True
    except Exception:
        return False

def list_fixed_drives():
    """List all fixed drives on Windows"""
    drives = []
    for letter in string.ascii_uppercase:
        drive = Path(f"{letter}:")
        if drive.exists():
            try:
                # Check if it's a fixed drive (not removable)
                if ctypes.windll.kernel32.GetDriveTypeW(str(drive)) == 3:  # DRIVE_FIXED
                    drives.append(drive)
            except Exception:
                pass
    return drives

# -----------------------------
# Similarity Search (TF-IDF + FAISS)
# -----------------------------

SIM_DIR = DB_DIR / "db"
SIM_DIR.mkdir(parents=True, exist_ok=True)
FAISS_INDEX = SIM_DIR / "faiss_tfidf.index"
VECTORIZER_PKL = SIM_DIR / "tfidf_vectorizer.joblib"

def _fetch_text_corpus(limit: int = 500000):
    """Return [(id, path, text_for_index)]"""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT rowid, path, COALESCE(name,'') || ' ' || COALESCE(tags,'') FROM files ORDER BY mtime DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    con.close()
    return rows

def build_similarity_index():
    """Build TF-IDF embeddings + FAISS index on (name + tags)."""
    if not _SIM_AVAILABLE:
        return False, "Similarity search requires: numpy, faiss, sklearn, joblib"
    
    try:
        import numpy as np
        import joblib  # type: ignore
        import faiss  # type: ignore
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.preprocessing import normalize as _sk_normalize  # type: ignore

        corpus = _fetch_text_corpus()
        if not corpus:
            return False, "No rows to index."

        ids = np.array([r[0] for r in corpus], dtype="int64")
        texts = [r[2] or "" for r in corpus]

        vec = TfidfVectorizer(max_features=200000, ngram_range=(1,2), lowercase=True)
        X = vec.fit_transform(texts)            # sparse
        X = X.astype("float32")
        X = _sk_normalize(X, norm="l2", copy=False)

        xb = X.toarray()                         # small/medium sets OK
        dim = xb.shape[1]
        index = faiss.IndexFlatIP(dim)           # cosine via l2-normed vectors → inner product
        index.add(xb)

        # persist
        faiss.write_index(index, str(FAISS_INDEX))
        joblib.dump({"vectorizer": vec, "ids": ids}, VECTORIZER_PKL)
        return True, f"Indexed {len(corpus)} items."
    except Exception as e:
        return False, f"{e}"

def find_similar_by_row(main_window, row_idx: int, k: int = 30):
    """Return [(score, id, name, path, tags)] for the selected table row."""
    if not _SIM_AVAILABLE:
        return []
    
    try:
        import numpy as np
        import joblib  # type: ignore
        import faiss  # type: ignore
        from sklearn.preprocessing import normalize as _sk_normalize  # type: ignore
        
        if not FAISS_INDEX.exists() or not VECTORIZER_PKL.exists():
            return []

        md = joblib.load(VECTORIZER_PKL)
        vec, ids = md["vectorizer"], md["ids"]
        index = faiss.read_index(str(FAISS_INDEX))

        # Build a small text query from the selected row (name + tags)
        try:
            row = main_window.model.rows[row_idx]
        except Exception:
            return []

        name = row[1] if len(row) > 1 else ""
        tags = row[5] if len(row) > 5 else ""
        qtext = f"{name} {tags}".strip()
        if not qtext:
            return []

        q = vec.transform([qtext]).astype("float32")
        q = _sk_normalize(q, norm="l2", copy=False).toarray()
        D, I = index.search(q, k)
        idxs = I[0]; scores = D[0]

        # Map IDs → file rows
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        out = []
        for score, rid in zip(scores, idxs):
            try:
                file_id = int(ids[rid])
                cur.execute("SELECT rowid, name, path, tags FROM files WHERE rowid=?", (file_id,))
                r = cur.fetchone()
                if r:
                    _, nm, pth, tg = r
                    out.append((float(score), file_id, nm or "", pth or "", tg or ""))
            except Exception:
                continue
        con.close()
        
        # Filter out the current file itself
        sel_path = row[0] if row else ""
        out = [t for t in out if t[3] != sel_path]
        return out
    except Exception as e:
        print(f"Error in find_similar_by_row: {e}")
        return []

# -----------------------------
# Background Indexer
# -----------------------------

class Indexer(QtCore.QThread):
    """Background thread for indexing files"""
    progress = QtCore.Signal(int, int)  # found, scanned
    indexed = QtCore.Signal(str)        # path
    finished_ok = QtCore.Signal()
    error = QtCore.Signal(str)

    def __init__(self, roots: list[Path], excludes: list[str] | None = None):
        super().__init__()
        self.roots = roots
        self.excludes = excludes or []
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def _is_excluded(self, path: Path) -> bool:
        """Check if path should be excluded"""
        path_str = str(path).lower()
        return any(excl.lower() in path_str for excl in self.excludes)

    def run(self):
        try:
            found = 0
            scanned = 0
            
            for root in self.roots:
                if self._stop.is_set():
                    break
                    
                for dirpath, dirnames, filenames in os.walk(root, topdown=True):
                    # Prune excluded directories
                    dirnames[:] = [d for d in dirnames if not self._is_excluded(Path(dirpath) / d)]
                    
                    if self._stop.is_set():
                        break
                    
                    scanned += 1
                    
                    for fn in filenames:
                        if self._stop.is_set():
                            break
                            
                        ext = Path(fn).suffix.lower()
                        if ext in SUPPORTED_EXTS:
                            file_path = Path(dirpath) / fn
                            if upsert_file(file_path):
                                found += 1
                                self.indexed.emit(str(file_path))
                    
                    # Update progress every 100 directories
                    if scanned % 100 == 0:
                        self.progress.emit(found, scanned)
            
            self.progress.emit(found, scanned)
            self.finished_ok.emit()
            
        except Exception as e:
            self.error.emit(str(e))

# -----------------------------
# Inline Edit Delegate
# -----------------------------

class InlineEditDelegate(QtWidgets.QStyledItemDelegate):
    """Delegate for inline editing of tags"""
    
    def createEditor(self, parent, option, index):
        """Create editor widget"""
        editor = QtWidgets.QLineEdit(parent)
        return editor
    
    def setEditorData(self, editor, index):
        """Set initial data in editor"""
        value = index.model().data(index, QtCore.Qt.ItemDataRole.DisplayRole)
        editor.setText(str(value) if value else "")
    
    def setModelData(self, editor, model, index):
        """Save data from editor to model"""
        value = editor.text()
        model.setData(index, value, QtCore.Qt.ItemDataRole.EditRole)
    
    def updateEditorGeometry(self, editor, option, index):
        """Update editor geometry"""
        editor.setGeometry(option.rect)

# -----------------------------
# Thumbnail Cache
# -----------------------------

class ThumbnailCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}
        self.max_memory_cache = 50  # LRU (simple)

    def get_cache_path(self, file_path: str) -> Path | None:
        try:
            stat = Path(file_path).stat()
            digest = hashlib.md5(f"{file_path}_{stat.st_mtime}_{stat.st_size}".encode()).hexdigest()
            return self.cache_dir / f"{digest}.png"
        except Exception:
            return None

    def get_thumbnail(self, file_path: str) -> QtGui.QPixmap | None:
        if file_path in self.memory_cache:
            return self.memory_cache[file_path]
        cp = self.get_cache_path(file_path)
        if cp and cp.exists():
            pm = QtGui.QPixmap(str(cp))
            self._add_to_memory(file_path, pm)
            return pm
        return None

    def _add_to_memory(self, file_path: str, pm: QtGui.QPixmap):
        if len(self.memory_cache) >= self.max_memory_cache:
            # naive LRU eviction
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
        self.memory_cache[file_path] = pm

    def save_thumbnail(self, file_path: str, pm: QtGui.QPixmap):
        cp = self.get_cache_path(file_path)
        if cp:
            pm.save(str(cp), "PNG")
            self._add_to_memory(file_path, pm)

# -----------------------------
# Thumbnail Generation Worker
# -----------------------------

class ThumbnailGenWorker(QtCore.QRunnable):
    """Background worker for generating 3D thumbnails"""
    def __init__(self, file_path: str, cache: ThumbnailCache, size: int = 256):
        super().__init__()
        self.file_path = file_path
        self.cache = cache
        self.size = size

    def run(self):
        """Generate thumbnail using solid_renderer with fallback methods"""
        if not _GEO_AVAILABLE or not _PIL_AVAILABLE:
            return
        
        try:
            # Use the new solid_renderer for clean, professional 3D previews
            from solid_renderer import render_mesh_to_image
            from PIL.ImageQt import ImageQt
            
            # Generate high-quality isometric preview using solid_renderer
            # Use conservative face limits for very large files (200-300MB+)
            # Check file size to determine appropriate face limit
            file_size_mb = Path(self.file_path).stat().st_size / (1024 * 1024)
            if file_size_mb > 200:  # Very large files (200MB+)
                max_faces = 30000 if self.size <= 256 else 50000
            elif file_size_mb > 100:  # Large files (100-200MB)
                max_faces = 50000 if self.size <= 256 else 100000
            else:  # Normal files
                max_faces = 100000 if self.size <= 256 else 150000
            
            img = render_mesh_to_image(
                file_path=self.file_path, 
                size=(self.size, self.size),
                bg_rgba=(245, 245, 245, 255),  # Clean white background
                face_rgb=(220, 220, 240),      # Light blue-gray for mesh
                outline_rgb=(160, 160, 180),   # Subtle edge outlines
                outline_width=1,
                max_faces=max_faces,           # Adaptive limit based on size
                draw_edges=True
            )
            
            # Convert PIL Image to QPixmap using ImageQt
            qimage = ImageQt(img)
            pm = QtGui.QPixmap.fromImage(qimage)
            
            if not pm.isNull():
                self.cache.save_thumbnail(self.file_path, pm)
                # Show file size and face limit in status for large files
                if file_size_mb > 100:
                    print(f"Generated thumbnail for {Path(self.file_path).name}: {file_size_mb:.1f}MB, max_faces={max_faces}")
                
        except ImportError:
            # Fallback to original method if solid_renderer not available
            self._fallback_render()
        except Exception as e:
            # Silently fail - thumbnail generation is optional
            print(f"Thumbnail generation failed for {self.file_path}: {e}")
            pass
    
    def _fallback_render(self):
        """Fallback rendering method using the original implementation"""
        try:
            import trimesh
            import numpy as np
            from PIL import Image, ImageDraw
            
            # Load mesh - Force Trimesh to return a triangular mesh
            m = trimesh.load(self.file_path, force='mesh', process=True)
            if m is None:
                return
            
            # Ensure we have a proper Trimesh object, not a scene or point cloud
            if not isinstance(m, trimesh.Trimesh):
                # Convert scenes into one Trimesh
                m = m.dump(concatenate=True)
            
            # Validate we have a proper mesh
            if not isinstance(m, trimesh.Trimesh):
                print(f"Warning: {self.file_path} could not be loaded as a proper mesh")
                return
            
            # Ensure faces exist - if no faces, try to reconstruct from convex hull
            if m.faces.shape[0] == 0:
                print(f"Warning: {self.file_path} has no faces, attempting to reconstruct from convex hull")
                m = m.convex_hull
                
            # Final validation
            if m.faces.shape[0] == 0:
                print(f"Warning: {self.file_path} still has no faces after convex hull reconstruction")
                return
            
            # Compute vertex normals if absent - Lambert shading depends on normals
            if not hasattr(m, 'vertex_normals') or m.vertex_normals is None or len(m.vertex_normals) == 0:
                print(f"Computing normals for {self.file_path}")
                m.fix_normals()
            
            # Auto-correct bad meshes for better rendering (with error handling)
            try:
                print(f"Auto-correcting mesh for {self.file_path}")
                m.remove_duplicate_faces()
                m.remove_degenerate_faces()
                m.fill_holes()
            except Exception as e:
                print(f"Warning: Mesh auto-correction failed for {self.file_path}: {e}")
                # Continue with rendering even if auto-correction fails
            
            # Create a clean, solid mesh preview like Windows File Explorer
            img = Image.new('RGB', (self.size, self.size), color=(245, 245, 245))  # Clean white background
            draw = ImageDraw.Draw(img)
            
            try:
                # Fix 1: Ensure we have proper mesh with vertices and faces
                if hasattr(m, 'vertices') and hasattr(m, 'faces'):
                    vertices = np.array(m.vertices)
                    faces = np.array(m.faces)
                else:
                    # Fallback: check if it's a point cloud
                    if hasattr(m, 'vertices') and not hasattr(m, 'faces'):
                        print(f"Warning: {self.file_path} is a point cloud, not a mesh")
                        return
                    else:
                        print(f"Warning: {self.file_path} has no vertices")
                        return
                
                # Validate mesh data
                if len(vertices) == 0 or len(faces) == 0:
                    print(f"Warning: {self.file_path} has empty vertices or faces")
                    return
                
                # Ensure faces are triangular
                faces = faces[:, :3] if faces.shape[1] > 3 else faces
                
                # Calculate mesh bounds and center
                min_coords = vertices.min(axis=0)
                max_coords = vertices.max(axis=0)
                center = (min_coords + max_coords) / 2
                
                # Normalize vertices to fit in preview
                max_dim = np.max(max_coords - min_coords)
                if max_dim > 0:
                    vertices_normalized = (vertices - center) / max_dim * (self.size * 0.6)
                    
                    # Simple isometric projection for clean 3D look
                    vertices_2d = np.zeros((len(vertices_normalized), 2))
                    # Isometric projection: x' = x - z, y' = y + (x + z)/2
                    vertices_2d[:, 0] = (vertices_normalized[:, 0] - vertices_normalized[:, 2]) * 0.866 + self.size/2
                    vertices_2d[:, 1] = (vertices_normalized[:, 1] + (vertices_normalized[:, 0] + vertices_normalized[:, 2]) * 0.5) * 0.866 + self.size/2
                    
                    # Depth-sort faces correctly using face centers
                    faces_z = vertices_normalized[faces].mean(axis=1)[:, 2]
                    sorted_idx = np.argsort(faces_z)[::-1]  # far → near
                    
                    # Simple Lambert lighting setup
                    light_dir = np.array([0.577, 0.577, 0.577])
                    light_dir /= np.linalg.norm(light_dir)
                    normals = m.face_normals
                    intensity = np.clip(normals @ light_dir, 0.2, 1.0)
                    colors = (intensity[:, None] * np.array([220, 220, 240])).astype(np.uint8)
                    
                    # Render faces with proper depth sorting and lighting
                    for i in sorted_idx:
                        face = faces[i]
                        if len(face) >= 3:
                            try:
                                # Get face vertices in 2D
                                face_2d = vertices_2d[face[:3]]
                                
                                # Get pre-calculated lighting color for this face
                                face_color = tuple(colors[i])
                                
                                # Create polygon points
                                polygon_points = [(int(p[0]), int(p[1])) for p in face_2d]
                                
                                # Draw filled triangle
                                draw.polygon(polygon_points, fill=face_color)
                                
                                # Add subtle edge for definition
                                for j in range(3):
                                    start = face_2d[j]
                                    end = face_2d[(j + 1) % 3]
                                    if (0 <= start[0] < self.size and 0 <= start[1] < self.size and
                                        0 <= end[0] < self.size and 0 <= end[1] < self.size):
                                        draw.line([(int(start[0]), int(start[1])), (int(end[0]), int(end[1]))], 
                                                 fill=(160, 160, 160), width=1)
                                
                            except (IndexError, ValueError):
                                continue
                    
                    # Add clean border
                    padding = 3
                    draw.rectangle([padding, padding, self.size - padding, self.size - padding],
                                 outline=(180, 180, 180), width=1)
                    
                else:
                    # No valid geometry - show clean placeholder
                    center_x, center_y = self.size // 2, self.size // 2
                    draw.ellipse([center_x - 20, center_y - 20, center_x + 20, center_y + 20],
                               fill=(230, 230, 230), outline=(180, 180, 180), width=2)
                    draw.text((center_x - 15, center_y - 8), "3D", fill=(120, 120, 120))
                
            except Exception as render_error:
                # Clean fallback
                center_x, center_y = self.size // 2, self.size // 2
                draw.rectangle([center_x - 15, center_y - 15, center_x + 15, center_y + 15],
                             fill=(230, 230, 230), outline=(180, 180, 180), width=2)
                draw.text((center_x - 10, center_y - 8), "3D", fill=(120, 120, 120))
            
            # Convert PIL image to QPixmap
            import io
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            
            pm = QtGui.QPixmap()
            pm.loadFromData(buffer.read(), "PNG")
            
            if not pm.isNull():
                self.cache.save_thumbnail(self.file_path, pm)
                
        except Exception as e:
            # Silently fail - thumbnail generation is optional
            print(f"Fallback thumbnail generation failed for {self.file_path}: {e}")
            pass

# -----------------------------
# Geometry Computation Worker
# -----------------------------

class GeometryWorker(QtCore.QRunnable):
    """Background worker for computing geometry metadata (tris + dimensions)"""
    def __init__(self, paths: list[str]):
        super().__init__()
        self.paths = paths

    def run(self):
        """Compute geometry in background thread"""
        if not _GEO_AVAILABLE:
            return
        
        try:
            import trimesh
            con = sqlite3.connect(DB_PATH)
            cur = con.cursor()
            
            for p in self.paths:
                try:
                    m = trimesh.load(p, force='mesh', skip_materials=True)
                    if m is None or (hasattr(m, 'is_empty') and m.is_empty):
                        continue
                    
                    tris = int(m.faces.shape[0]) if hasattr(m, 'faces') else None
                    bounds = m.bounds if hasattr(m, 'bounds') else None
                    dx = dy = dz = None
                    
                    if bounds is not None:
                        (minx, miny, minz), (maxx, maxy, maxz) = bounds
                        dx, dy, dz = float(maxx - minx), float(maxy - miny), float(maxz - minz)
                    
                    cur.execute(
                        """UPDATE files SET tris=?, dim_x=?, dim_y=?, dim_z=? WHERE path=?""",
                        (tris, dx, dy, dz, p)
                    )
                    con.commit()
                except Exception as e:
                    print(f"Geometry computation failed for {p}: {e}")
                    continue
            
            con.close()
        except Exception as e:
            print(f"GeometryWorker error: {e}")

# -----------------------------
# Dialog Classes
# -----------------------------

class FindSimilarResultsDialog(QtWidgets.QDialog):
    """Dialog showing similarity search results"""
    def __init__(self, parent, results: list):
        super().__init__(parent)
        self.setWindowTitle("Find Similar — Results")
        self.resize(820, 520)
        self.setModal(True)
        self.results = results
        
        layout = QtWidgets.QVBoxLayout(self)
        
        # Header
        lbl_header = QtWidgets.QLabel("Top matches (double-click to open file)")
        lbl_header.setStyleSheet("font-weight:600;")
        layout.addWidget(lbl_header)
        
        # Results tree
        self.tree_results = QtWidgets.QTreeWidget()
        self.tree_results.setHeaderLabels(["Score", "ID", "Name", "Path", "Tags"])
        self.tree_results.setColumnWidth(2, 280)
        self.tree_results.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.tree_results)
        
        # Empty state label
        self.lbl_empty = QtWidgets.QLabel("No similar items found. Try adding tags or rebuilding the index.")
        self.lbl_empty.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_empty.setStyleSheet("color:#888;")
        self.lbl_empty.setVisible(False)
        layout.addWidget(self.lbl_empty)
        
        # Button box
        btnbox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        btnbox.rejected.connect(self.reject)
        layout.addWidget(btnbox)
        
        self._populate_results()
    
    def _populate_results(self):
        """Populate tree with results"""
        if not self.results:
            self.tree_results.setVisible(False)
            self.lbl_empty.setVisible(True)
            return
        
        for score, file_id, name, path, tags in self.results:
            item = QtWidgets.QTreeWidgetItem([
                f"{score:.3f}",
                str(file_id),
                name,
                path,
                tags or ""
            ])
            self.tree_results.addTopLevelItem(item)
    
    def _on_item_double_clicked(self, item, column):
        """Open file on double-click"""
        path = item.text(3)
        if path and Path(path).exists():
            os.startfile(path)


class ExportCSVDialog(QtWidgets.QDialog):
    """Dialog for configuring CSV export options"""
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Export Options")
        self.resize(460, 520)
        self.setModal(True)
        
        layout = QtWidgets.QVBoxLayout(self)
        
        # Columns group
        grp_columns = QtWidgets.QGroupBox("Columns")
        col_layout = QtWidgets.QVBoxLayout(grp_columns)
        
        self.col_checks = {}
        for col in ["Name", "Extension", "Size (MB)", "Modified", "Tags", "Path", "Parent Folder", "Drive"]:
            cb = QtWidgets.QCheckBox(col)
            cb.setChecked(True)
            self.col_checks[col] = cb
            col_layout.addWidget(cb)
        
        layout.addWidget(grp_columns)
        
        # Filters group
        grp_filters = QtWidgets.QGroupBox("Filters")
        filter_layout = QtWidgets.QVBoxLayout(grp_filters)
        
        # Path contains
        filter_layout.addWidget(QtWidgets.QLabel("Path contains:"))
        self.ed_path_contains = QtWidgets.QLineEdit()
        self.ed_path_contains.setPlaceholderText("Substring (optional)")
        filter_layout.addWidget(self.ed_path_contains)
        
        # Extensions
        ext_group = QtWidgets.QGroupBox("Extensions")
        ext_layout = QtWidgets.QVBoxLayout(ext_group)
        
        self.ext_checks = {}
        for ext in SUPPORTED_EXTS:
            cb = QtWidgets.QCheckBox(ext)
            cb.setChecked(False)
            self.ext_checks[ext] = cb
            ext_layout.addWidget(cb)
        
        filter_layout.addWidget(ext_group)
        
        # Limit
        limit_layout = QtWidgets.QHBoxLayout()
        limit_layout.addWidget(QtWidgets.QLabel("Row limit:"))
        self.sp_limit = QtWidgets.QSpinBox()
        self.sp_limit.setRange(0, 10000000)
        self.sp_limit.setSpecialValueText("No limit")
        self.sp_limit.setValue(5000)
        limit_layout.addWidget(self.sp_limit)
        limit_layout.addStretch()
        filter_layout.addLayout(limit_layout)
        
        # Header row
        self.cb_header = QtWidgets.QCheckBox("Include header row")
        self.cb_header.setChecked(True)
        filter_layout.addWidget(self.cb_header)
        
        layout.addWidget(grp_filters)
        
        # Button box
        btnbox = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | 
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btnbox.accepted.connect(self.accept)
        btnbox.rejected.connect(self.reject)
        layout.addWidget(btnbox)
    
    def get_options(self):
        """Return export options as dict"""
        return {
            'columns': [col for col, cb in self.col_checks.items() if cb.isChecked()],
            'path_contains': self.ed_path_contains.text(),
            'exts': [ext for ext, cb in self.ext_checks.items() if cb.isChecked()],
            'limit': self.sp_limit.value(),
            'include_header': self.cb_header.isChecked()
        }


class DatabaseManagerDialog(QtWidgets.QDialog):
    """Dialog for database maintenance"""
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Database Manager")
        self.resize(520, 380)
        self.setModal(True)
        
        layout = QtWidgets.QVBoxLayout(self)
        
        # DB path
        self.lbl_dbpath = QtWidgets.QLabel(f"DB Path: {DB_PATH}")
        self.lbl_dbpath.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_dbpath.setWordWrap(True)
        layout.addWidget(self.lbl_dbpath)
        
        # Stats grid
        stats_widget = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(stats_widget)
        
        self.lbl_rows = QtWidgets.QLabel("Rows: …")
        self.lbl_size = QtWidgets.QLabel("File size: …")
        grid.addWidget(self.lbl_rows, 0, 0)
        grid.addWidget(self.lbl_size, 0, 1)
        
        layout.addWidget(stats_widget)
        
        # Actions row
        actions_layout = QtWidgets.QHBoxLayout()
        
        btn_analyze = QtWidgets.QPushButton("Analyze")
        btn_analyze.clicked.connect(self._analyze)
        actions_layout.addWidget(btn_analyze)
        
        btn_vacuum = QtWidgets.QPushButton("Vacuum")
        btn_vacuum.clicked.connect(self._vacuum)
        actions_layout.addWidget(btn_vacuum)
        
        btn_backup = QtWidgets.QPushButton("Backup…")
        btn_backup.clicked.connect(self._backup)
        actions_layout.addWidget(btn_backup)
        
        actions_layout.addStretch()
        layout.addLayout(actions_layout)
        
        layout.addStretch()
        
        # Close button
        btnbox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        btnbox.rejected.connect(self.reject)
        layout.addWidget(btnbox)
        
        self._refresh_stats()
    
    def _refresh_stats(self):
        """Refresh database statistics"""
        try:
            con = sqlite3.connect(DB_PATH)
            cur = con.cursor()
            cur.execute("SELECT COUNT(*) FROM files")
            count = cur.fetchone()[0]
            self.lbl_rows.setText(f"Rows: {count:,}")
            con.close()
            
            size = DB_PATH.stat().st_size / (1024 * 1024)
            self.lbl_size.setText(f"File size: {size:.2f} MB")
        except Exception as e:
            self.lbl_rows.setText(f"Rows: Error")
            self.lbl_size.setText(f"File size: Error")
    
    def _analyze(self):
        """Run ANALYZE on database"""
        try:
            con = sqlite3.connect(DB_PATH)
            cur = con.cursor()
            cur.execute("ANALYZE")
            con.commit()
            con.close()
            QtWidgets.QMessageBox.information(self, "Analyze Complete", "Database analyzed successfully.")
            self._refresh_stats()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Analyze Failed", f"Failed to analyze database:\n{e}")
    
    def _vacuum(self):
        """Run VACUUM on database"""
        try:
            con = sqlite3.connect(DB_PATH)
            cur = con.cursor()
            cur.execute("VACUUM")
            con.commit()
            con.close()
            QtWidgets.QMessageBox.information(self, "Vacuum Complete", "Database vacuumed successfully.")
            self._refresh_stats()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Vacuum Failed", f"Failed to vacuum database:\n{e}")
    
    def _backup(self):
        """Backup database to file"""
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Database Backup",
            str(Path.home() / f"modelfinder_backup_{time.strftime('%Y%m%d_%H%M%S')}.db"),
            "Database Files (*.db);;All Files (*)"
        )
        if fn:
            try:
                import shutil
                shutil.copy2(DB_PATH, fn)
                QtWidgets.QMessageBox.information(self, "Backup Complete", f"Database backed up to:\n{fn}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Backup Failed", f"Failed to backup database:\n{e}")


class TrainingDialog(QtWidgets.QDialog):
    """Dialog for training ML model from well-organized archive folders"""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Train ML Model from Archive")
        self.resize(700, 600)
        self.setModal(True)
        
        self.training_folders = []
        self.trainer = None
        
        self._build_ui()
    
    def _build_ui(self):
        """Build the training dialog UI"""
        layout = QtWidgets.QVBoxLayout(self)
        
        # Title
        title = QtWidgets.QLabel("🎓 ML Model Training")
        title.setStyleSheet("font-weight: bold; font-size: 16px;")
        layout.addWidget(title)
        
        desc = QtWidgets.QLabel(
            "Train the ML model to recognize part types from mesh geometry.\n"
            "Select 3-5 well-organized project folders for best results."
        )
        desc.setStyleSheet("color: gray; margin-bottom: 10px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # Training folders section
        folders_group = QtWidgets.QGroupBox("Training Folders")
        folders_layout = QtWidgets.QVBoxLayout(folders_group)
        
        self.folders_list = QtWidgets.QListWidget()
        self.folders_list.setAlternatingRowColors(True)
        folders_layout.addWidget(self.folders_list)
        
        btn_layout = QtWidgets.QHBoxLayout()
        btn_add_folder = QtWidgets.QPushButton("➕ Add Folder")
        btn_add_folder.clicked.connect(self._add_training_folder)
        btn_layout.addWidget(btn_add_folder)
        
        btn_remove_folder = QtWidgets.QPushButton("➖ Remove Selected")
        btn_remove_folder.clicked.connect(self._remove_training_folder)
        btn_layout.addWidget(btn_remove_folder)
        
        btn_layout.addStretch()
        folders_layout.addLayout(btn_layout)
        
        layout.addWidget(folders_group)
        
        # Progress and status
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        self.status_label = QtWidgets.QLabel("Ready to scan folders")
        layout.addWidget(self.status_label)
        
        # Statistics display
        stats_group = QtWidgets.QGroupBox("Training Statistics")
        stats_layout = QtWidgets.QFormLayout(stats_group)
        
        self.lbl_samples = QtWidgets.QLabel("0")
        self.lbl_part_types = QtWidgets.QLabel("0")
        self.lbl_projects = QtWidgets.QLabel("0")
        self.lbl_accuracy = QtWidgets.QLabel("Not trained")
        
        stats_layout.addRow("Training Samples:", self.lbl_samples)
        stats_layout.addRow("Part Types Found:", self.lbl_part_types)
        stats_layout.addRow("Projects Found:", self.lbl_projects)
        stats_layout.addRow("Model Accuracy:", self.lbl_accuracy)
        
        layout.addWidget(stats_group)
        
        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        
        self.btn_scan = QtWidgets.QPushButton("🔍 Scan Folders")
        self.btn_scan.setToolTip("Scan selected folders to build training dataset")
        self.btn_scan.clicked.connect(self._scan_folders)
        self.btn_scan.setEnabled(False)
        btn_row.addWidget(self.btn_scan)
        
        self.btn_train = QtWidgets.QPushButton("🎓 Train Model")
        self.btn_train.setToolTip("Train ML model on scanned data")
        self.btn_train.clicked.connect(self._train_model)
        self.btn_train.setEnabled(False)
        btn_row.addWidget(self.btn_train)
        
        btn_row.addStretch()
        
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        
        layout.addLayout(btn_row)
    
    def _add_training_folder(self):
        """Add folder for training with validation"""
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Well-Organized Project Folder",
            str(Path.home())
        )
        
        if folder:
            # Validate folder
            if folder in self.training_folders:
                QtWidgets.QMessageBox.warning(self, "Duplicate Folder", 
                    "This folder is already in the training list.")
                return
            
            # Check if folder contains 3D files
            try:
                from pathlib import Path
                folder_path = Path(folder)
                if not folder_path.exists():
                    QtWidgets.QMessageBox.warning(self, "Invalid Folder", 
                        "Selected folder does not exist.")
                    return
                
                # Count 3D files in folder
                extensions = {'.stl', '.obj', '.ply', '.fbx', '.glb'}
                file_count = sum(1 for f in folder_path.rglob('*') 
                               if f.suffix.lower() in extensions)
                
                if file_count == 0:
                    reply = QtWidgets.QMessageBox.question(
                        self, "No 3D Files Found", 
                        f"No 3D files found in {folder_path.name}.\n"
                        f"Do you want to add it anyway?",
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                    )
                    if reply != QtWidgets.QMessageBox.Yes:
                        return
                else:
                    QtWidgets.QMessageBox.information(
                        self, "Folder Added", 
                        f"Added {folder_path.name} with {file_count} 3D files."
                    )
                
                self.training_folders.append(folder)
                self.folders_list.addItem(folder)
                self.btn_scan.setEnabled(True)
                self.status_label.setText(f"{len(self.training_folders)} folder(s) selected")
                
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", 
                    f"Error accessing folder: {e}")
    
    def _remove_training_folder(self):
        """Remove selected folder from list"""
        current_row = self.folders_list.currentRow()
        if current_row >= 0:
            self.folders_list.takeItem(current_row)
            self.training_folders.pop(current_row)
            self.btn_scan.setEnabled(len(self.training_folders) > 0)
    
    def _scan_folders(self):
        """Scan folders to build training dataset"""
        try:
            from src.ml.archive_trainer import ArchiveTrainer
            
            self.btn_scan.setEnabled(False)
            self.status_label.setText("Scanning folders...")
            
            # Create trainer
            self.trainer = ArchiveTrainer(str(DB_PATH))
            
            # Scan with progress callback
            def progress_callback(current, total, message):
                self.progress_bar.setMaximum(total)
                self.progress_bar.setValue(current)
                self.status_label.setText(message)
                QtWidgets.QApplication.processEvents()  # Update UI
            
            stats = self.trainer.scan_training_folders(
                self.training_folders, 
                progress_callback
            )
            
            # Save training data to database
            self.trainer.save_training_data()
            
            # Update statistics
            training_stats = self.trainer.get_training_statistics()
            
            self.lbl_samples.setText(str(training_stats['total_samples']))
            self.lbl_part_types.setText(str(len(training_stats['part_types'])))
            self.lbl_projects.setText(str(len(training_stats['projects'])))
            
            self.status_label.setText(
                f"Scanned {stats['total_files']} files, "
                f"extracted {stats['features_extracted']} features"
            )
            
            # Show part type distribution
            if training_stats['part_types']:
                msg = "Part types found:\n\n"
                for part, count in sorted(training_stats['part_types'].items(), 
                                         key=lambda x: x[1], reverse=True)[:10]:
                    msg += f"  {part}: {count}\n"
                
                QtWidgets.QMessageBox.information(
                    self, "Scan Complete", msg
                )
            
            # Enable training
            if training_stats['total_samples'] >= 10:
                self.btn_train.setEnabled(True)
            else:
                QtWidgets.QMessageBox.warning(
                    self, "Insufficient Data",
                    f"Only {training_stats['total_samples']} samples found.\n"
                    f"Need at least 10 samples to train.\n\n"
                    f"Add more well-organized folders."
                )
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Scan Error",
                f"Failed to scan folders:\n{e}"
            )
            import traceback
            traceback.print_exc()
        finally:
            self.btn_scan.setEnabled(True)
            self.progress_bar.setValue(0)
    
    def _train_model(self):
        """Train the ML model"""
        try:
            from src.ml.part_classifier import PartClassifier
            
            self.btn_train.setEnabled(False)
            self.status_label.setText("Training model...")
            QtWidgets.QApplication.processEvents()
            
            # Create and train classifier
            classifier = PartClassifier(str(DB_PATH))
            
            stats = classifier.train()
            
            # Save trained model
            classifier.save('models/part_classifier.pkl')
            
            # Update UI with results
            part_stats = stats['part_type']
            self.lbl_accuracy.setText(
                f"{part_stats['test_accuracy']:.1%} "
                f"(CV: {part_stats['cv_mean']:.1%} ± {part_stats['cv_std']:.1%})"
            )
            
            # Show detailed results
            msg = "Model Training Complete!\n\n"
            msg += f"Part Type Classifier:\n"
            msg += f"  Test Accuracy: {part_stats['test_accuracy']:.1%}\n"
            msg += f"  Cross-validation: {part_stats['cv_mean']:.1%} ± {part_stats['cv_std']:.1%}\n"
            msg += f"  Classes: {part_stats['n_classes']}\n\n"
            
            if stats['laterality']['n_classes'] > 1:
                lat_stats = stats['laterality']
                msg += f"Laterality Classifier:\n"
                msg += f"  Test Accuracy: {lat_stats['test_accuracy']:.1%}\n"
                msg += f"  Cross-validation: {lat_stats['cv_mean']:.1%} ± {lat_stats['cv_std']:.1%}\n\n"
            
            msg += f"Top 5 Important Features:\n"
            for feat, importance in stats['top_features'][:5]:
                msg += f"  {feat}: {importance:.3f}\n"
            
            msg += f"\nModel saved to: models/part_classifier.pkl"
            
            QtWidgets.QMessageBox.information(self, "Training Complete", msg)
            
            self.status_label.setText("Model trained successfully!")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Training Error",
                f"Failed to train model:\n{e}\n\n"
                f"Make sure you have:\n"
                f"1. Scanned folders first\n"
                f"2. At least 10 training samples\n"
                f"3. scikit-learn installed: pip install scikit-learn"
            )
            import traceback
            traceback.print_exc()
        finally:
            self.btn_train.setEnabled(True)


class MigrationPlannerDialog(QtWidgets.QDialog):
    """Dialog for planning and executing archive reorganization"""
    
    LICENSE_TYPES = ["Commercial", "Personal", "Fan-Art", "Stock", "Client Work", "Unknown"]
    ASSET_CATEGORIES = ["Character", "Prop", "Environment", "Accessory", "Vehicle", "Effect", "Unknown"]
    
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Archive Migration Planner")
        self.resize(1200, 800)
        self.setModal(True)
        
        self.settings = QtCore.QSettings("ModelFinder", "ModelFinder")
        self.migration_plan = []
        self.conflicts = []
        self.quarantined = []
        
        # Initialize guardrails system
        if _PROPOSAL_AVAILABLE:
            self.guardrails = MigrationGuardrails(DB_PATH)
        else:
            self.guardrails = None
        
        self._build_ui()
        self._load_files_for_migration()
    
    def _build_ui(self):
        """Build the migration planner UI"""
        layout = QtWidgets.QVBoxLayout(self)
        
        # Title and description
        title = QtWidgets.QLabel("📦 Archive Reorganization Planner")
        title.setStyleSheet("font-weight: bold; font-size: 16px;")
        layout.addWidget(title)
        
        desc = QtWidgets.QLabel(
            "Plan migration of files to organized structure: "
            "<dest_root>/<project>/<license>/<category>/<proposed_name>"
        )
        desc.setStyleSheet("color: gray; margin-bottom: 10px;")
        layout.addWidget(desc)
        
        # Settings section
        settings_group = QtWidgets.QGroupBox("Migration Settings")
        settings_layout = QtWidgets.QFormLayout(settings_group)
        
        # Destination root
        dest_layout = QtWidgets.QHBoxLayout()
        self.dest_root_edit = QtWidgets.QLineEdit()
        self.dest_root_edit.setText(self.settings.value("dest_root", ""))
        self.dest_root_edit.textChanged.connect(self._on_settings_changed)
        dest_layout.addWidget(self.dest_root_edit)
        
        btn_browse = QtWidgets.QPushButton("Browse...")
        btn_browse.clicked.connect(self._browse_dest_root)
        dest_layout.addWidget(btn_browse)
        settings_layout.addRow("Destination Root:", dest_layout)
        
        # Default license
        self.default_license = QtWidgets.QComboBox()
        self.default_license.addItems(self.LICENSE_TYPES)
        self.default_license.currentTextChanged.connect(self._on_settings_changed)
        settings_layout.addRow("Default License:", self.default_license)
        
        # Default category
        self.default_category = QtWidgets.QComboBox()
        self.default_category.addItems(self.ASSET_CATEGORIES)
        self.default_category.currentTextChanged.connect(self._on_settings_changed)
        settings_layout.addRow("Default Category:", self.default_category)
        
        layout.addWidget(settings_group)
        
        # Files table
        files_group = QtWidgets.QGroupBox("Files to Migrate")
        files_layout = QtWidgets.QVBoxLayout(files_group)
        
        self.files_table = QtWidgets.QTableWidget()
        self.files_table.setColumnCount(8)
        self.files_table.setHorizontalHeaderLabels([
            "Current Name", "Project", "Proposed Name", "License", "Category",
            "Destination", "Status", "Conflicts"
        ])
        self.files_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.files_table.setAlternatingRowColors(True)
        self.files_table.horizontalHeader().setStretchLastSection(True)
        files_layout.addWidget(self.files_table)
        
        layout.addWidget(files_group)
        
        # Statistics
        stats_layout = QtWidgets.QHBoxLayout()
        self.lbl_total = QtWidgets.QLabel("Total: 0")
        self.lbl_ready = QtWidgets.QLabel("Ready: 0")
        self.lbl_conflicts = QtWidgets.QLabel("Conflicts: 0")
        self.lbl_missing = QtWidgets.QLabel("Missing Info: 0")
        
        for lbl in [self.lbl_total, self.lbl_ready, self.lbl_conflicts, self.lbl_missing]:
            lbl.setStyleSheet("padding: 5px; font-weight: bold;")
            stats_layout.addWidget(lbl)
        
        stats_layout.addStretch()
        layout.addLayout(stats_layout)
        
        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        
        self.btn_refresh = QtWidgets.QPushButton("🔄 Refresh Plan")
        self.btn_refresh.clicked.connect(self._refresh_migration_plan)
        btn_layout.addWidget(self.btn_refresh)
        
        self.btn_dry_run = QtWidgets.QPushButton("👁️ Dry Run")
        self.btn_dry_run.setToolTip("Preview migration without moving files")
        self.btn_dry_run.clicked.connect(self._run_dry_run)
        btn_layout.addWidget(self.btn_dry_run)
        
        self.btn_execute = QtWidgets.QPushButton("✅ Execute Migration")
        self.btn_execute.setToolTip("Perform actual file migration")
        self.btn_execute.clicked.connect(self._execute_migration)
        self.btn_execute.setEnabled(False)
        btn_layout.addWidget(self.btn_execute)
        
        self.btn_rollback = QtWidgets.QPushButton("↩️ Rollback Last 10")
        self.btn_rollback.setToolTip("Rollback the last 10 migration operations")
        self.btn_rollback.clicked.connect(self._rollback_operations)
        btn_layout.addWidget(self.btn_rollback)
        
        btn_layout.addStretch()
        
        btn_close = QtWidgets.QPushButton("Cancel")
        btn_close.clicked.connect(self.reject)
        btn_layout.addWidget(btn_close)
        
        layout.addLayout(btn_layout)
    
    def _browse_dest_root(self):
        """Browse for destination root directory"""
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Destination Root",
            self.dest_root_edit.text() or str(Path.home())
        )
        if dir_path:
            self.dest_root_edit.setText(dir_path)
            self.settings.setValue("dest_root", dir_path)
    
    def _on_settings_changed(self):
        """Handle settings change"""
        self._refresh_migration_plan()
    
    def _load_files_for_migration(self):
        """Load files that are ready for migration"""
        try:
            con = sqlite3.connect(DB_PATH)
            cur = con.cursor()
            
            # Get files with proposals
            cur.execute("""
                SELECT 
                    path, name, project_number, project_name, 
                    proposed_name, type_conf,
                    license_type, asset_category
                FROM files
                WHERE proposed_name IS NOT NULL 
                  AND proposed_name != ''
                  AND project_number IS NOT NULL
                ORDER BY project_number, proposed_name
            """)
            
            files = cur.fetchall()
            con.close()
            
            self.files_table.setRowCount(len(files))
            
            for row_idx, file_data in enumerate(files):
                path, name, proj_num, proj_name, proposed, conf, license, category = file_data
                
                # Current name
                self.files_table.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(name))
                
                # Project
                self.files_table.setItem(row_idx, 1, 
                    QtWidgets.QTableWidgetItem(f"{proj_num} - {proj_name or 'Unknown'}"))
                
                # Proposed name
                self.files_table.setItem(row_idx, 2, QtWidgets.QTableWidgetItem(proposed or ""))
                
                # License (editable combo)
                license_combo = QtWidgets.QComboBox()
                license_combo.addItems(self.LICENSE_TYPES)
                license_combo.setCurrentText(license or "Unknown")
                license_combo.currentTextChanged.connect(self._on_settings_changed)
                self.files_table.setCellWidget(row_idx, 3, license_combo)
                
                # Category (editable combo)
                category_combo = QtWidgets.QComboBox()
                category_combo.addItems(self.ASSET_CATEGORIES)
                category_combo.setCurrentText(category or "Unknown")
                category_combo.currentTextChanged.connect(self._on_settings_changed)
                self.files_table.setCellWidget(row_idx, 4, category_combo)
                
                # Placeholder for destination and status (filled by refresh)
                self.files_table.setItem(row_idx, 5, QtWidgets.QTableWidgetItem(""))
                self.files_table.setItem(row_idx, 6, QtWidgets.QTableWidgetItem(""))
                self.files_table.setItem(row_idx, 7, QtWidgets.QTableWidgetItem(""))
            
            self._refresh_migration_plan()
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load files:\n{e}")
    
    def _refresh_migration_plan(self):
        """Refresh the migration plan and detect conflicts"""
        dest_root = self.dest_root_edit.text()
        
        if not dest_root:
            return
        
        self.migration_plan = []
        self.conflicts = []
        dest_paths_seen = {}
        
        for row_idx in range(self.files_table.rowCount()):
            try:
                # Get file data
                name_item = self.files_table.item(row_idx, 0)
                proj_item = self.files_table.item(row_idx, 1)
                proposed_item = self.files_table.item(row_idx, 2)
                
                if not all([name_item, proj_item, proposed_item]):
                    continue
                
                # Get combo box values
                license_combo = self.files_table.cellWidget(row_idx, 3)
                category_combo = self.files_table.cellWidget(row_idx, 4)
                
                license = license_combo.currentText() if license_combo else "Unknown"
                category = category_combo.currentText() if category_combo else "Unknown"
                
                # Parse project
                proj_text = proj_item.text()
                proj_num = proj_text.split(" - ")[0] if " - " in proj_text else proj_text
                proj_name = proj_text.split(" - ")[1] if " - " in proj_text else ""
                
                proposed_name = proposed_item.text()
                
                # Build destination path
                dest_folder = f"{proj_num}_{proj_name.replace(' ', '_')}" if proj_name else proj_num
                dest_path = Path(dest_root) / dest_folder / license / category / proposed_name
                
                # Update table
                self.files_table.setItem(row_idx, 5, QtWidgets.QTableWidgetItem(str(dest_path)))
                
                # Check for conflicts
                status = "Ready"
                conflict_msg = ""
                
                if dest_path.exists():
                    status = "⚠️ Exists"
                    conflict_msg = "File already exists at destination"
                    self.conflicts.append((row_idx, conflict_msg))
                
                if str(dest_path) in dest_paths_seen:
                    status = "❌ Duplicate"
                    conflict_msg = f"Duplicate destination with row {dest_paths_seen[str(dest_path)]}"
                    self.conflicts.append((row_idx, conflict_msg))
                else:
                    dest_paths_seen[str(dest_path)] = row_idx
                
                if not proposed_name:
                    status = "❌ No Name"
                    conflict_msg = "Missing proposed name"
                
                self.files_table.setItem(row_idx, 6, QtWidgets.QTableWidgetItem(status))
                self.files_table.setItem(row_idx, 7, QtWidgets.QTableWidgetItem(conflict_msg))
                
                # Color code row
                color = None
                if "❌" in status:
                    color = QtGui.QColor(255, 200, 200)  # Red
                elif "⚠️" in status:
                    color = QtGui.QColor(255, 255, 200)  # Yellow
                else:
                    color = QtGui.QColor(200, 255, 200)  # Green
                
                if color:
                    for col in range(self.files_table.columnCount()):
                        item = self.files_table.item(row_idx, col)
                        if item:
                            item.setBackground(color)
                
                # Add to migration plan
                self.migration_plan.append({
                    "row": row_idx,
                    "source": name_item.text(),
                    "dest": str(dest_path),
                    "status": status,
                    "conflict": conflict_msg
                })
                
            except Exception as e:
                print(f"Error processing row {row_idx}: {e}")
        
        # Update statistics
        total = len(self.migration_plan)
        ready = sum(1 for p in self.migration_plan if "Ready" in p["status"])
        conflicts_count = len(self.conflicts)
        missing = sum(1 for p in self.migration_plan if "No Name" in p["status"])
        
        self.lbl_total.setText(f"Total: {total}")
        self.lbl_ready.setText(f"Ready: {ready}")
        self.lbl_ready.setStyleSheet("padding: 5px; font-weight: bold; color: green;")
        self.lbl_conflicts.setText(f"Conflicts: {conflicts_count}")
        self.lbl_conflicts.setStyleSheet(f"padding: 5px; font-weight: bold; color: {'red' if conflicts_count > 0 else 'gray'};")
        self.lbl_missing.setText(f"Missing Info: {missing}")
        
        # Enable execute button if no conflicts
        self.btn_execute.setEnabled(conflicts_count == 0 and ready > 0)
        
        self.files_table.resizeColumnsToContents()
    
    def _run_dry_run(self):
        """Run a dry-run simulation"""
        if not self.migration_plan:
            QtWidgets.QMessageBox.warning(self, "No Plan", "No migration plan generated.")
            return
        
        # Show dry-run results
        msg = f"Dry Run Results:\n\n"
        msg += f"Total files: {len(self.migration_plan)}\n"
        msg += f"Ready to migrate: {sum(1 for p in self.migration_plan if 'Ready' in p['status'])}\n"
        msg += f"Conflicts: {len(self.conflicts)}\n\n"
        
        if self.conflicts:
            msg += "Conflicts found:\n"
            for row, conflict in self.conflicts[:10]:
                msg += f"  Row {row + 1}: {conflict}\n"
            if len(self.conflicts) > 10:
                msg += f"  ... and {len(self.conflicts) - 10} more\n"
            msg += "\n"
        
        msg += "Sample migrations:\n"
        for p in self.migration_plan[:5]:
            msg += f"  {p['source']}\n"
            msg += f"  → {p['dest']}\n\n"
        
        if len(self.migration_plan) > 5:
            msg += f"... and {len(self.migration_plan) - 5} more files"
        
        QtWidgets.QMessageBox.information(self, "Dry Run", msg)
    
    def _execute_migration(self):
        """Execute the actual file migration with comprehensive guardrails"""
        ready_count = sum(1 for p in self.migration_plan if "Ready" in p["status"])
        
        if self.guardrails is None:
            QtWidgets.QMessageBox.warning(
                self, "Migration Unavailable",
                "Migration guardrails system is not available. Please check your installation."
            )
            return
        
        ret = QtWidgets.QMessageBox.question(
            self, "Confirm Migration",
            f"Ready to migrate {ready_count} files to organized structure.\n\n"
            f"This will:\n"
            f"• Create folder structure\n"
            f"• Move files to new locations\n"
            f"• Verify file integrity with SHA256 hashes\n"
            f"• Handle conflicts with version bumping\n"
            f"• Quarantine malformed meshes\n"
            f"• Log all operations for rollback\n"
            f"• Update database\n\n"
            f"Continue?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
        )
        
        if ret != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        
        # Show progress dialog
        progress = QtWidgets.QProgressDialog(
            "Migrating files with integrity verification...", "Cancel", 0, ready_count, self
        )
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        
        # Execute migration using guardrails system
        results = self.guardrails.execute_migration(self.migration_plan, dry_run=False)
        
        # Show results
        if results['success']:
            QtWidgets.QMessageBox.information(
                self, "Migration Complete",
                f"Successfully migrated {results['migrated']} files.\n"
                f"Skipped: {results['skipped']}\n"
                f"Failed: {results['failed']}\n\n"
                f"All operations have been logged for potential rollback."
            )
        else:
            error_msg = "\n".join([f"• {e['file']}: {e['error']}" for e in results['errors'][:5]])
            if len(results['errors']) > 5:
                error_msg += f"\n• ... and {len(results['errors']) - 5} more errors"
            
            QtWidgets.QMessageBox.warning(
                self, "Migration Completed with Errors",
                f"Migrated: {results['migrated']}\n"
                f"Failed: {results['failed']}\n"
                f"Skipped: {results['skipped']}\n\n"
                f"Errors:\n{error_msg}"
            )
        
        # Refresh the plan
        self._refresh_plan()
    
    def _rollback_operations(self):
        """Rollback recent migration operations"""
        if self.guardrails is None:
            QtWidgets.QMessageBox.warning(
                self, "Rollback Unavailable",
                "Rollback system is not available. Please check your installation."
            )
            return
        
        ret = QtWidgets.QMessageBox.question(
            self, "Confirm Rollback",
            "This will rollback the last 10 migration operations.\n\n"
            "This action cannot be undone. Continue?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
        )
        
        if ret != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        
        # Execute rollback
        results = self.guardrails.rollback_operations(10)
        
        if results['success']:
            QtWidgets.QMessageBox.information(
                self, "Rollback Complete",
                f"Successfully rolled back {results['rolled_back']} operations.\n"
                f"Failed: {results['failed']}\n\n"
                f"Files have been restored to their original locations."
            )
        else:
            error_msg = "\n".join([f"• {e['error']}" for e in results['errors'][:5]])
            if len(results['errors']) > 5:
                error_msg += f"\n• ... and {len(results['errors']) - 5} more errors"
            
            QtWidgets.QMessageBox.warning(
                self, "Rollback Completed with Errors",
                f"Rolled back: {results['rolled_back']}\n"
                f"Failed: {results['failed']}\n\n"
                f"Errors:\n{error_msg}"
            )
        
        # Refresh the plan
        self._refresh_plan()


class ProjectPickerDialog(QtWidgets.QDialog):
    """Dialog for selecting a project from reference parts"""
    def __init__(self, parent, projects: list[str]):
        super().__init__(parent)
        self.setWindowTitle("Select Project")
        self.resize(400, 300)
        self.setModal(True)
        
        layout = QtWidgets.QVBoxLayout(self)
        
        # Title
        title = QtWidgets.QLabel("Select a project for proposal generation:")
        title.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(title)
        
        # Project list
        self.project_list = QtWidgets.QListWidget()
        self.project_list.addItems(sorted(projects))
        self.project_list.setAlternatingRowColors(True)
        self.project_list.itemDoubleClicked.connect(self.accept)
        layout.addWidget(self.project_list)
        
        # Info label
        info = QtWidgets.QLabel(f"Found {len(projects)} projects in reference database")
        info.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(info)
        
        # Buttons
        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)
        
        # Select first item by default
        if projects:
            self.project_list.setCurrentRow(0)
    
    def get_selected_project(self) -> str:
        """Get the selected project number"""
        current_item = self.project_list.currentItem()
        return current_item.text() if current_item else ""


class SettingsDialog(QtWidgets.QDialog):
    """Settings dialog"""
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(520, 420)
        self.setModal(True)
        self.settings = QtCore.QSettings("ModelFinder", "ModelFinder")
        
        layout = QtWidgets.QVBoxLayout(self)
        
        # Form layout
        form = QtWidgets.QFormLayout()
        
        # Destination root
        dest_row = QtWidgets.QHBoxLayout()
        self.ed_dest_root = QtWidgets.QLineEdit()
        self.ed_dest_root.setText(self.settings.value("dest_root", ""))
        dest_row.addWidget(self.ed_dest_root)
        btn_pick_dest = QtWidgets.QPushButton("…")
        btn_pick_dest.clicked.connect(lambda: self._pick_dir(self.ed_dest_root))
        dest_row.addWidget(btn_pick_dest)
        form.addRow("Destination Root:", dest_row)
        
        # Cache directory
        cache_row = QtWidgets.QHBoxLayout()
        self.ed_cache_dir = QtWidgets.QLineEdit()
        self.ed_cache_dir.setText(self.settings.value("cache_dir", str(DB_DIR / "thumbnails")))
        cache_row.addWidget(self.ed_cache_dir)
        btn_pick_cache = QtWidgets.QPushButton("…")
        btn_pick_cache.clicked.connect(lambda: self._pick_dir(self.ed_cache_dir))
        cache_row.addWidget(btn_pick_cache)
        form.addRow("Thumbnail Cache:", cache_row)
        
        # Confidence threshold
        self.sp_conf_thresh = QtWidgets.QDoubleSpinBox()
        self.sp_conf_thresh.setRange(0.0, 1.0)
        self.sp_conf_thresh.setSingleStep(0.01)
        self.sp_conf_thresh.setValue(float(self.settings.value("conf_threshold", 0.66)))
        form.addRow("Auto-accept Threshold:", self.sp_conf_thresh)
        
        layout.addLayout(form)
        layout.addStretch()
        
        # Button box
        btnbox = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | 
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btnbox.accepted.connect(self._save_to_settings_and_close)
        btnbox.rejected.connect(self.reject)
        layout.addWidget(btnbox)
    
    def _pick_dir(self, line_edit):
        """Pick directory"""
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Directory",
            line_edit.text() or str(Path.home())
        )
        if dir_path:
            line_edit.setText(dir_path)
    
    def _save_to_settings_and_close(self):
        """Save settings and close"""
        self.settings.setValue("dest_root", self.ed_dest_root.text())
        self.settings.setValue("cache_dir", self.ed_cache_dir.text())
        self.settings.setValue("conf_threshold", self.sp_conf_thresh.value())
        self.accept()


class DiagnosticsDialog(QtWidgets.QDialog):
    """Diagnostics dialog for health checks"""
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Diagnostics")
        self.resize(560, 420)
        self.setModal(True)
        
        layout = QtWidgets.QVBoxLayout(self)
        
        # Intro
        lbl_intro = QtWidgets.QLabel("Quick health checks for ModelFinder.")
        lbl_intro.setStyleSheet("font-weight:600;")
        layout.addWidget(lbl_intro)
        
        # Checks form
        form = QtWidgets.QFormLayout()
        
        self.badge_db = QtWidgets.QLabel("…")
        self.badge_db.setStyleSheet("padding:4px 8px;border-radius:10px;")
        form.addRow("Database reachable", self.badge_db)
        
        self.badge_files = QtWidgets.QLabel("…")
        self.badge_files.setStyleSheet("padding:4px 8px;border-radius:10px;")
        form.addRow("files table exists", self.badge_files)
        
        self.badge_ref = QtWidgets.QLabel("…")
        self.badge_ref.setStyleSheet("padding:4px 8px;border-radius:10px;")
        form.addRow("project_reference_parts exists", self.badge_ref)
        
        self.badge_faiss = QtWidgets.QLabel("…")
        self.badge_faiss.setStyleSheet("padding:4px 8px;border-radius:10px;")
        form.addRow("FAISS artifacts present", self.badge_faiss)
        
        layout.addLayout(form)
        
        # Details text
        self.txt_details = QtWidgets.QPlainTextEdit()
        self.txt_details.setReadOnly(True)
        self.txt_details.setPlaceholderText("Details / errors appear here…")
        layout.addWidget(self.txt_details)
        
        # Buttons row
        buttons_layout = QtWidgets.QHBoxLayout()
        
        btn_recheck = QtWidgets.QPushButton("Re-run checks")
        btn_recheck.clicked.connect(self._run_checks)
        buttons_layout.addWidget(btn_recheck)
        
        buttons_layout.addStretch()
        
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.reject)
        buttons_layout.addWidget(btn_close)
        
        layout.addLayout(buttons_layout)
        
        self._run_checks()
    
    def _set_badge(self, label, ok):
        """Set badge appearance"""
        if ok:
            label.setText("OK")
            label.setStyleSheet("padding:4px 8px;border-radius:10px;background:#2e7d32;color:white;")
        else:
            label.setText("FAIL")
            label.setStyleSheet("padding:4px 8px;border-radius:10px;background:#c62828;color:white;")
    
    def _run_checks(self):
        """Run all diagnostic checks"""
        self.txt_details.clear()
        details = []
        
        # Check database
        try:
            con = sqlite3.connect(DB_PATH)
            con.close()
            self._set_badge(self.badge_db, True)
            details.append("✓ Database connection successful")
        except Exception as e:
            self._set_badge(self.badge_db, False)
            details.append(f"✗ Database connection failed: {e}")
        
        # Check files table
        try:
            con = sqlite3.connect(DB_PATH)
            cur = con.cursor()
            cur.execute("PRAGMA table_info(files)")
            count = len(cur.fetchall())
            con.close()
            self._set_badge(self.badge_files, count > 0)
            details.append(f"✓ files table exists with {count} columns" if count > 0 else "✗ files table not found")
        except Exception as e:
            self._set_badge(self.badge_files, False)
            details.append(f"✗ files table check failed: {e}")
        
        # Check project_reference_parts table
        try:
            con = sqlite3.connect(DB_PATH)
            cur = con.cursor()
            cur.execute("PRAGMA table_info(project_reference_parts)")
            count = len(cur.fetchall())
            con.close()
            self._set_badge(self.badge_ref, count > 0)
            details.append(f"✓ project_reference_parts table exists with {count} columns" if count > 0 else "✗ project_reference_parts table not found")
        except Exception as e:
            self._set_badge(self.badge_ref, False)
            details.append(f"✗ project_reference_parts table check failed: {e}")
        
        # Check FAISS artifacts
        faiss_paths = [
            DB_DIR.parent / "db" / "faiss_tfidf.index",
            DB_DIR.parent / "db" / "tfidf_vectorizer.joblib"
        ]
        faiss_ok = any(p.exists() for p in faiss_paths)
        self._set_badge(self.badge_faiss, faiss_ok)
        if faiss_ok:
            existing = [p.name for p in faiss_paths if p.exists()]
            details.append(f"✓ FAISS artifacts found: {', '.join(existing)}")
        else:
            details.append("✗ No FAISS artifacts found (similarity search unavailable)")
        
        self.txt_details.setPlainText("\n".join(details))


# -----------------------------
# Main Window with Enhanced UI
# -----------------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} — 3D Asset Finder")
        self.resize(1400, 900)
        self.settings = QtCore.QSettings("ModelFinder", "ModelFinder")
        self.indexer = None
        self.threadpool = QtCore.QThreadPool.globalInstance()
        self.current_scan_roots: list[Path] = []
        self.show_all_files = False
        self.left_panel_mode = "filters"  # "filters" or "browser"
        
        # Apply dark theme first
        self.apply_dark_theme()
        
        # Build UI
        self._build_ui()
        ensure_db()
        migrate_db_for_proposals()
        
        # Ensure user corrections table exists
        if _PROPOSAL_AVAILABLE:
            try:
                ensure_user_corrections()
            except Exception as e:
                print(f"Failed to initialize user corrections table: {e}")
        
        self.model.set_rows([])
        
        # Initialize threadpool for background operations
        self.threadpool = QtCore.QThreadPool.globalInstance()

    def apply_dark_theme(self):
        """Apply professional dark theme similar to Torrblar"""
        self.current_theme = "dark"
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QToolBar {
                background-color: #404040;
                border: none;
                spacing: 3px;
                padding: 2px;
            }
            QToolButton {
                background-color: #505050;
                border: 1px solid #666666;
                border-radius: 3px;
                padding: 4px 8px;
                color: #ffffff;
            }
            QToolButton:hover {
                background-color: #606060;
                border-color: #777777;
            }
            QToolButton:pressed {
                background-color: #404040;
            }
            QToolButton:checked {
                background-color: #0078d4;
                border-color: #106ebe;
            }
            QLineEdit, QComboBox {
                background-color: #3c3c3c;
                border: 1px solid #666666;
                border-radius: 3px;
                padding: 4px;
                color: #ffffff;
            }
            QLineEdit:focus, QComboBox:focus {
                border-color: #0078d4;
            }
            QTableView {
                background-color: #3c3c3c;
                alternate-background-color: #404040;
                gridline-color: #555555;
                color: #ffffff;
                border: 1px solid #666666;
            }
            QTableView::item {
                padding: 4px;
                border: none;
            }
            QTableView::item:selected {
                background-color: #0078d4;
                color: #ffffff;
            }
            QTableView::item:hover {
                background-color: #505050;
            }
            QHeaderView::section {
                background-color: #404040;
                border: 1px solid #666666;
                padding: 4px;
                color: #ffffff;
            }
            QTreeView {
                background-color: #3c3c3c;
                alternate-background-color: #404040;
                color: #ffffff;
                border: 1px solid #666666;
            }
            QTreeView::item {
                padding: 2px;
            }
            QTreeView::item:selected {
                background-color: #0078d4;
                color: #ffffff;
            }
            QTreeView::item:hover {
                background-color: #505050;
            }
            QTreeView::branch {
                background-color: #3c3c3c;
            }
            QStatusBar {
                background-color: #404040;
                color: #ffffff;
                border-top: 1px solid #666666;
            }
            QGroupBox {
                background-color: #3c3c3c;
                border: 1px solid #666666;
                border-radius: 3px;
                margin-top: 10px;
                color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QCheckBox {
                color: #ffffff;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QCheckBox::indicator:unchecked {
                border: 1px solid #666666;
                background-color: #3c3c3c;
            }
            QCheckBox::indicator:checked {
                border: 1px solid #0078d4;
                background-color: #0078d4;
            }
            QLabel {
                color: #ffffff;
            }
            QDialog {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QDialog QLabel {
                color: #ffffff;
                background-color: transparent;
            }
            QDialog QGroupBox {
                background-color: #3c3c3c;
                border: 1px solid #666666;
                border-radius: 3px;
                color: #ffffff;
                padding-top: 10px;
            }
            QDialog QGroupBox::title {
                color: #ffffff;
                background-color: #2b2b2b;
                padding: 2px 5px;
            }
            QDialog QLineEdit {
                background-color: #3c3c3c;
                border: 1px solid #666666;
                color: #ffffff;
                padding: 4px;
            }
            QDialog QListWidget {
                background-color: #3c3c3c;
                border: 1px solid #666666;
                color: #ffffff;
            }
            QDialog QTableWidget {
                background-color: #3c3c3c;
                border: 1px solid #666666;
                color: #ffffff;
            }
            QDialog QComboBox {
                background-color: #3c3c3c;
                border: 1px solid #666666;
                color: #ffffff;
            }
            QPushButton {
                background-color: #505050;
                border: 1px solid #666666;
                border-radius: 3px;
                padding: 6px 12px;
                color: #ffffff;
            }
            QPushButton:hover {
                background-color: #606060;
            }
            QPushButton:pressed {
                background-color: #404040;
            }
        """)

    def apply_light_theme(self):
        """Apply professional light theme"""
        self.current_theme = "light"
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ffffff;
                color: #000000;
            }
            QToolBar {
                background-color: #f5f5f5;
                border: none;
                spacing: 3px;
                padding: 2px;
                border-bottom: 1px solid #e0e0e0;
            }
            QToolButton {
                background-color: #ffffff;
                border: 1px solid #d0d0d0;
                border-radius: 3px;
                padding: 4px 8px;
                color: #000000;
            }
            QToolButton:hover {
                background-color: #f0f0f0;
                border-color: #c0c0c0;
            }
            QToolButton:pressed {
                background-color: #e0e0e0;
            }
            QToolButton:checked {
                background-color: #0078d4;
                border-color: #106ebe;
                color: #ffffff;
            }
            QLineEdit, QComboBox {
                background-color: #ffffff;
                border: 1px solid #d0d0d0;
                border-radius: 3px;
                padding: 4px;
                color: #000000;
            }
            QLineEdit:focus, QComboBox:focus {
                border-color: #0078d4;
            }
            QTableView {
                background-color: #ffffff;
                alternate-background-color: #f8f8f8;
                gridline-color: #e0e0e0;
                color: #000000;
                border: 1px solid #d0d0d0;
            }
            QTableView::item {
                padding: 4px;
                border: none;
            }
            QTableView::item:selected {
                background-color: #0078d4;
                color: #ffffff;
            }
            QTableView::item:hover {
                background-color: #f0f0f0;
            }
            QHeaderView::section {
                background-color: #f5f5f5;
                border: 1px solid #d0d0d0;
                padding: 4px;
                color: #000000;
            }
            QTreeView {
                background-color: #ffffff;
                alternate-background-color: #f9f9f9;
                color: #000000;
                border: 1px solid #d0d0d0;
            }
            QTreeView::item {
                padding: 2px;
            }
            QTreeView::item:selected {
                background-color: #0078d4;
                color: #ffffff;
            }
            QTreeView::item:hover {
                background-color: #f0f0f0;
            }
            QTreeView::branch {
                background-color: #ffffff;
            }
            QStatusBar {
                background-color: #f5f5f5;
                color: #000000;
                border-top: 1px solid #e0e0e0;
            }
            QGroupBox {
                background-color: #ffffff;
                border: 1px solid #d0d0d0;
                border-radius: 3px;
                margin-top: 10px;
                color: #000000;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QCheckBox {
                color: #000000;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QCheckBox::indicator:unchecked {
                border: 1px solid #d0d0d0;
                background-color: #ffffff;
            }
            QCheckBox::indicator:checked {
                border: 1px solid #0078d4;
                background-color: #0078d4;
            }
            QLabel {
                color: #000000;
            }
            QDialog {
                background-color: #ffffff;
                color: #000000;
            }
            QDialog QLabel {
                color: #000000;
                background-color: transparent;
            }
            QDialog QGroupBox {
                background-color: #f9f9f9;
                border: 1px solid #d0d0d0;
                border-radius: 3px;
                color: #000000;
                padding-top: 10px;
            }
            QDialog QGroupBox::title {
                color: #000000;
                background-color: #ffffff;
                padding: 2px 5px;
            }
            QDialog QLineEdit {
                background-color: #ffffff;
                border: 1px solid #d0d0d0;
                color: #000000;
                padding: 4px;
            }
            QDialog QListWidget {
                background-color: #ffffff;
                border: 1px solid #d0d0d0;
                color: #000000;
            }
            QDialog QTableWidget {
                background-color: #ffffff;
                border: 1px solid #d0d0d0;
                color: #000000;
            }
            QDialog QComboBox {
                background-color: #ffffff;
                border: 1px solid #d0d0d0;
                color: #000000;
            }
            QPushButton {
                background-color: #ffffff;
                border: 1px solid #d0d0d0;
                border-radius: 3px;
                padding: 6px 12px;
                color: #000000;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
            QPushButton:pressed {
                background-color: #e0e0e0;
            }
            QMenuBar {
                background-color: #f5f5f5;
                color: #000000;
                border-bottom: 1px solid #e0e0e0;
                padding: 2px;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 4px 8px;
                margin: 2px;
            }
            QMenuBar::item:selected {
                background-color: #e0e0e0;
                border-radius: 3px;
            }
            QMenuBar::item:pressed {
                background-color: #d0d0d0;
            }
            QMenu {
                background-color: #ffffff;
                color: #000000;
                border: 1px solid #d0d0d0;
                border-radius: 3px;
            }
            QMenu::item {
                padding: 6px 20px;
                margin: 1px;
            }
            QMenu::item:selected {
                background-color: #0078d4;
                color: #ffffff;
            }
            QMenu::separator {
                height: 1px;
                background-color: #e0e0e0;
                margin: 2px 0px;
            }
        """)

    def toggle_theme(self):
        """Toggle between light and dark theme"""
        if self.current_theme == "dark":
            self.apply_light_theme()
            self.btn_theme_toggle.setText("🌙")
            self.btn_theme_toggle.setToolTip("Switch to Dark Theme (Ctrl+Shift+T)")
            self.status.showMessage("Switched to Light Theme", 2000)
        else:
            self.apply_dark_theme()
            self.btn_theme_toggle.setText("💡")
            self.btn_theme_toggle.setToolTip("Switch to Light Theme (Ctrl+Shift+T)")
            self.status.showMessage("Switched to Dark Theme", 2000)

    def _create_menu_bar(self):
        """Create professional menu bar like Cursor IDE"""
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #404040;
                color: #ffffff;
                border-bottom: 1px solid #666666;
                padding: 2px;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 4px 8px;
                margin: 2px;
            }
            QMenuBar::item:selected {
                background-color: #505050;
                border-radius: 3px;
            }
            QMenuBar::item:pressed {
                background-color: #404040;
            }
            QMenu {
                background-color: #404040;
                color: #ffffff;
                border: 1px solid #666666;
                border-radius: 3px;
            }
            QMenu::item {
                padding: 6px 20px;
                margin: 1px;
            }
            QMenu::item:selected {
                background-color: #0078d4;
            }
            QMenu::separator {
                height: 1px;
                background-color: #666666;
                margin: 2px 0px;
            }
        """)
        
        # File Menu
        file_menu = menubar.addMenu("&File")
        file_menu.addAction("&New Project...", self._new_project, "Ctrl+N")
        file_menu.addAction("&Open Project...", self._open_project, "Ctrl+O")
        file_menu.addAction("&Save Project", self._save_project, "Ctrl+S")
        file_menu.addSeparator()
        file_menu.addAction("Import &Excel...", self.import_excel_dialog, "Ctrl+I")
        file_menu.addAction("&Export CSV...", self._export_csv, "Ctrl+E")
        file_menu.addSeparator()
        file_menu.addAction("E&xit", self.close, "Ctrl+Q")
        
        # Edit Menu
        edit_menu = menubar.addMenu("&Edit")
        edit_menu.addAction("&Find...", self._focus_search, "Ctrl+F")
        edit_menu.addAction("Find &Next", self._find_next, "F3")
        edit_menu.addAction("Find &Previous", self._find_previous, "Shift+F3")
        edit_menu.addSeparator()
        edit_menu.addAction("&Select All", self._select_all, "Ctrl+A")
        edit_menu.addAction("&Clear Selection", self._clear_selection, "Escape")
        edit_menu.addSeparator()
        edit_menu.addAction("&Batch Tag...", self.on_batch_tag_clicked, "Ctrl+T")
        
        # View Menu
        view_menu = menubar.addMenu("&View")
        view_menu.addAction("&Show All Files", self.toggle_show_all_files, "Ctrl+L")
        view_menu.addAction("&Refresh", self.refresh_table, "F5")
        view_menu.addSeparator()
        view_menu.addAction("&Toggle Theme", self.toggle_theme, "Ctrl+Shift+T")
        view_menu.addSeparator()
        view_menu.addAction("&Filters Panel", self._toggle_filters_panel, "Ctrl+Shift+F")
        view_menu.addAction("&Preview Panel", self._toggle_preview_panel, "Ctrl+Shift+P")
        view_menu.addSeparator()
        view_menu.addAction("&Full Screen", self._toggle_fullscreen, "F11")
        
        # Tools Menu
        tools_menu = menubar.addMenu("&Tools")
        tools_menu.addAction("&Scan Folders...", self.choose_and_scan, "Ctrl+R")
        tools_menu.addAction("Scan &All Drives...", self.scan_all_drives, "Ctrl+Shift+R")
        tools_menu.addSeparator()
        tools_menu.addAction("&Compute Geometry", self._start_geometry_compute, "Ctrl+G")
        tools_menu.addAction("&Build Similarity Index", self._rebuild_similarity_index, "Ctrl+B")
        tools_menu.addSeparator()
        tools_menu.addAction("📦 &Migrate Archive...", self._open_migration_planner, "Ctrl+M")
        tools_menu.addAction("🎓 &Train from Archive...", self._open_training_dialog, "Ctrl+Shift+T")
        tools_menu.addAction("🔄 Retrain from Corrections...", self._on_retrain_from_corrections)
        if _PROPOSAL_AVAILABLE:
            tools_menu.addAction("&Propose Names...", self.on_propose_names_clicked, "Ctrl+P")
        tools_menu.addSeparator()
        tools_menu.addAction("&Database Manager...", self._database_manager, "Ctrl+D")
        tools_menu.addAction("&Settings...", self._show_settings, "Ctrl+,")
        
        # Help Menu
        help_menu = menubar.addMenu("&Help")
        help_menu.addAction("&Documentation", self._show_docs, "F1")
        help_menu.addAction("&Keyboard Shortcuts", self._show_shortcuts, "Ctrl+Shift+K")
        help_menu.addSeparator()
        help_menu.addAction("&Diagnostics...", self._show_diagnostics, "Ctrl+Shift+D")
        help_menu.addSeparator()
        help_menu.addAction("&About ModelFinder", self._show_about, "Ctrl+Shift+A")

    def _build_ui(self):
        """Build the enhanced 3-panel UI layout"""
        # Create menu bar first
        self._create_menu_bar()
        
        # Create main layout with 3 panels
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        
        # Main horizontal layout
        main_layout = QtWidgets.QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Left panel - Filters
        self._build_filters_panel()
        main_layout.addWidget(self.filters_panel)
        
        # Center panel - Main content with toolbar and horizontal splitter
        center_widget = QtWidgets.QWidget()
        center_layout = QtWidgets.QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        
        # Top toolbar
        self._build_toolbar()
        center_layout.addWidget(self.toolbar)
        
        # Build table and preview panel
        self._build_results_table()
        self._build_preview_panel()
        
        # Horizontal splitter for table and preview
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        split.addWidget(self.table)
        split.addWidget(self.preview_panel)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 0)
        
        center_layout.addWidget(split)
        
        main_layout.addWidget(center_widget)
        
        # Status bar
        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready - Click 'Scan Folders...' to index 3D assets")

    def _build_filters_panel(self):
        """Build the left filters panel"""
        self.filters_panel = QtWidgets.QWidget()
        self.filters_panel.setMaximumWidth(250)
        self.filters_panel.setMinimumWidth(200)
        
        layout = QtWidgets.QVBoxLayout(self.filters_panel)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Build filters widget (no folder browser in left panel)
        filters_widget = self._build_filters_widget()
        layout.addWidget(filters_widget)
    
    def _build_filters_widget(self):
        """Build the filters widget for the left panel"""
        filters_widget = QtWidgets.QWidget()
        filters_layout = QtWidgets.QVBoxLayout(filters_widget)
        filters_layout.setContentsMargins(0, 0, 0, 0)
        
        # Title
        title = QtWidgets.QLabel("Filters")
        title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px;")
        filters_layout.addWidget(title)
        
        # File Type filters
        file_type_group = QtWidgets.QGroupBox("File Type")
        file_type_layout = QtWidgets.QVBoxLayout(file_type_group)
        
        self.file_type_filters = {}
        for ext in sorted(SUPPORTED_EXTS):
            cb = QtWidgets.QCheckBox(ext)
            cb.setChecked(True)
            cb.stateChanged.connect(self._on_filter_changed)
            self.file_type_filters[ext] = cb
            file_type_layout.addWidget(cb)
        
        filters_layout.addWidget(file_type_group)
        
        # Project filters
        project_group = QtWidgets.QGroupBox("Project")
        project_layout = QtWidgets.QFormLayout(project_group)
        
        self.project_filter = QtWidgets.QLineEdit()
        self.project_filter.setPlaceholderText("Project number...")
        self.project_filter.textChanged.connect(self._on_filter_changed)
        project_layout.addRow("Project #:", self.project_filter)
        
        self.part_type_filter = QtWidgets.QLineEdit()
        self.part_type_filter.setPlaceholderText("Part type...")
        self.part_type_filter.textChanged.connect(self._on_filter_changed)
        project_layout.addRow("Part Type:", self.part_type_filter)
        
        filters_layout.addWidget(project_group)
        
        # Size filters
        size_group = QtWidgets.QGroupBox("File Size")
        size_layout = QtWidgets.QFormLayout(size_group)
        
        self.min_size_filter = QtWidgets.QLineEdit()
        self.min_size_filter.setPlaceholderText("Min MB")
        self.min_size_filter.textChanged.connect(self._on_filter_changed)
        size_layout.addRow("Min Size:", self.min_size_filter)
        
        self.max_size_filter = QtWidgets.QLineEdit()
        self.max_size_filter.setPlaceholderText("Max MB")
        self.max_size_filter.textChanged.connect(self._on_filter_changed)
        size_layout.addRow("Max Size:", self.max_size_filter)
        
        filters_layout.addWidget(size_group)
        
        # Triangle count filters
        tris_group = QtWidgets.QGroupBox("Triangles")
        tris_layout = QtWidgets.QFormLayout(tris_group)
        
        self.min_tris_filter = QtWidgets.QLineEdit()
        self.min_tris_filter.setPlaceholderText("Min count")
        self.min_tris_filter.textChanged.connect(self._on_filter_changed)
        tris_layout.addRow("Min Tris:", self.min_tris_filter)
        
        self.max_tris_filter = QtWidgets.QLineEdit()
        self.max_tris_filter.setPlaceholderText("Max count")
        self.max_tris_filter.textChanged.connect(self._on_filter_changed)
        tris_layout.addRow("Max Tris:", self.max_tris_filter)
        
        filters_layout.addWidget(tris_group)
        
        # Clear filters button
        clear_btn = QtWidgets.QPushButton("Clear All Filters")
        clear_btn.clicked.connect(self._clear_filters)
        filters_layout.addWidget(clear_btn)
        
        filters_layout.addStretch()
        
        # Return the filters widget
        return filters_widget
    
    def _build_browser_widget(self):
        """Build the folder browser widget for the left panel"""
        browser_widget = QtWidgets.QWidget()
        browser_layout = QtWidgets.QVBoxLayout(browser_widget)
        browser_layout.setContentsMargins(0, 0, 0, 0)
        
        # Title
        title = QtWidgets.QLabel("Folder Browser")
        title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px;")
        browser_layout.addWidget(title)
        
        # Folder tree view with file system model
        self.folder_tree = QtWidgets.QTreeView()
        self.folder_tree.setAlternatingRowColors(True)
        self.folder_tree.setAnimated(True)
        self.folder_tree.setIndentation(20)
        self.folder_tree.setSortingEnabled(True)
        
        # Set up file system model
        self.file_system_model = QtWidgets.QFileSystemModel()
        self.file_system_model.setRootPath("")
        
        # Show only directories (folders) - no files
        self.file_system_model.setFilter(QtCore.QDir.Filter.Dirs | QtCore.QDir.Filter.NoDotAndDotDot)
        
        self.folder_tree.setModel(self.file_system_model)
        
        # Show only Name column - hide size, type, date columns
        for i in range(1, self.file_system_model.columnCount()):
            self.folder_tree.hideColumn(i)
        
        # Start at root to show all drives (Windows: This PC, shows C:, D:, E:, etc.)
        # Setting invalid index shows root level with all available drives
        self.folder_tree.setRootIndex(QtCore.QModelIndex())
        
        # Expand to show drives on Windows
        if sys.platform == 'win32':
            # On Windows, the root shows "This PC" and drives as children
            # We can optionally expand the first level to show all drives immediately
            root_index = self.file_system_model.index("")
            if root_index.isValid():
                self.folder_tree.expand(root_index)
        
        # Connect selection signal
        self.folder_tree.selectionModel().selectionChanged.connect(self._on_folder_selected)
        
        # Add context menu
        self.folder_tree.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.folder_tree.customContextMenuRequested.connect(self._show_browser_context_menu)
        
        browser_layout.addWidget(self.folder_tree)
        
        # Simple folder browser - no navigation buttons needed
        return browser_widget
    
    
    
    def _on_folder_selected(self, selected, deselected):
        """Handle folder selection in browser"""
        indexes = selected.indexes()
        if not indexes:
            return
        
        index = indexes[0]
        file_path = self.file_system_model.filePath(index)
        
        # If it's a directory, filter table to show files in that directory
        if self.file_system_model.isDir(index):
            self._filter_table_by_folder(file_path)
        # If it's a file, show it in preview automatically
        else:
            # Check if it's a 3D model file
            if Path(file_path).suffix.lower() in SUPPORTED_EXTS:
                self._preview_file(file_path)
    
    def _filter_table_by_folder(self, folder_path: str):
        """Load and display files from the selected folder"""
        try:
            # Store the current folder filter
            self.current_folder_filter = folder_path
            self.show_all_files = False
            
            # Quick load files directly from the filesystem
            self._quick_load_folder(folder_path)
            
        except Exception as e:
            print(f"Error loading folder: {e}")
            self.status.showMessage(f"Error loading folder: {e}", 3000)
    
    def _quick_load_folder(self, folder_path: str):
        """Quickly load files from a folder directly into the table"""
        try:
            folder = Path(folder_path)
            if not folder.exists() or not folder.is_dir():
                self.status.showMessage(f"Invalid folder: {folder_path}", 3000)
                return
            
            self.status.showMessage(f"Loading files from: {folder.name}...", 0)
            QtWidgets.QApplication.processEvents()  # Update UI
            
            # Find all 3D files in the folder and subfolders
            rows = []
            supported_exts = {'.stl', '.obj', '.ply', '.glb', '.fbx', '.3mf', '.off'}
            
            file_count = 0
            for file_path in folder.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in supported_exts:
                    try:
                        stat = file_path.stat()
                        
                        # Build row data: [path, name, ext, size, mtime, tags, ...]
                        row = [
                            str(file_path),           # 0: path
                            file_path.name,           # 1: name
                            file_path.suffix,         # 2: ext
                            stat.st_size,             # 3: size
                            stat.st_mtime,            # 4: mtime
                            "",                       # 5: tags (empty for now)
                            None,                     # 6: tris
                            None,                     # 7: dx
                            None,                     # 8: dy
                            None,                     # 9: dz
                            "",                       # 10: hash
                            "",                       # 11: project_number
                            "",                       # 12: project_name
                            "",                       # 13: part_name
                            "",                       # 14: proposed_name
                            0.0,                      # 15: confidence
                            False                     # 16: needs_review
                        ]
                        rows.append(tuple(row))
                        file_count += 1
                        
                        # Update UI periodically
                        if file_count % 50 == 0:
                            self.status.showMessage(f"Loading files: {file_count} found...", 0)
                            QtWidgets.QApplication.processEvents()
                            
                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}")
                        continue
            
            # Update the table model
            if hasattr(self, 'model') and self.model:
                self.model.set_rows(rows)
                self.status.showMessage(f"Loaded {file_count} files from: {folder.name}", 5000)
            else:
                self.status.showMessage("Table not initialized", 3000)
                
        except Exception as e:
            print(f"Error quick loading folder: {e}")
            self.status.showMessage(f"Error loading folder: {e}", 3000)
    
    def _apply_folder_filter(self, folder_path: str):
        """Apply folder filter to existing table data"""
        try:
            if not hasattr(self, 'model') or not self.model:
                return
            
            # Convert to Path for consistent comparison
            folder_path = Path(folder_path).resolve()
            
            # Filter the existing rows to show only files in this folder
            filtered_rows = []
            for row in self.model.all_rows:  # Assuming all_rows contains all data
                if len(row) > 0:
                    file_path = Path(row[0]).resolve()
                    try:
                        # Check if file is in the selected folder or its subfolders
                        if file_path.is_relative_to(folder_path) or file_path.parent == folder_path:
                            filtered_rows.append(row)
                    except ValueError:
                        # Path is not relative to folder, skip
                        continue
            
            # Update the model with filtered data
            self.model.rows = filtered_rows
            self.model.layoutChanged.emit()
            
            # Update status
            count = len(filtered_rows)
            self.status.showMessage(f"Showing {count} files in: {folder_path.name}", 3000)
            
        except Exception as e:
            print(f"Error applying folder filter: {e}")
            # Fallback to refresh if filtering fails
            self.refresh_table()
    
    def _clear_folder_filter(self):
        """Clear the current folder filter and show all files"""
        if hasattr(self, 'current_folder_filter'):
            self.current_folder_filter = None
        self.show_all_files = True
        if hasattr(self, 'model') and self.model and hasattr(self.model, 'all_rows'):
            self.model.rows = self.model.all_rows
            self.model.layoutChanged.emit()
        self.status.showMessage("Showing all files", 2000)
    
    def _on_search_changed(self, text: str):
        """Handle search text changes with live filtering"""
        if not hasattr(self, 'model') or not self.model or not hasattr(self.model, 'all_rows'):
            return
        
        # Clear search if text is empty
        if not text or len(text.strip()) == 0:
            self.model.rows = self.model.all_rows
            self.model.layoutChanged.emit()
            self.status.clearMessage()
            return
        
        # Search is case-insensitive
        search_term = text.lower().strip()
        
        # Filter rows by search term (search in name, path, tags, extension)
        filtered_rows = []
        for row in self.model.all_rows:
            if len(row) < 6:
                continue
            
            # Search in: name, extension, tags, path
            name = (row[1] or "").lower()
            ext = (row[2] or "").lower()
            tags = (row[5] or "").lower()
            path = (row[0] or "").lower()
            
            if (search_term in name or 
                search_term in ext or 
                search_term in tags or 
                search_term in path):
                filtered_rows.append(row)
        
        # Update table
        self.model.rows = filtered_rows
        self.model.layoutChanged.emit()
        
        # Update status
        count = len(filtered_rows)
        self.status.showMessage(f"Found {count} file(s) matching '{text}'", 3000)
    
    def _on_search_submit(self):
        """Handle Enter key in search box"""
        if hasattr(self, 'table') and self.table.model() and self.table.model().rowCount() > 0:
            # Select first result
            self.table.selectRow(0)
            self.table.setFocus()
            self.status.showMessage("Press Enter to open, or use arrow keys to browse", 3000)
    
    def _focus_search(self):
        """Focus the search input (Ctrl+F)"""
        if hasattr(self, 'search_input'):
            self.search_input.setFocus()
            self.search_input.selectAll()
            self.status.showMessage("Type to search files...", 2000)
    
    
    def _show_browser_context_menu(self, position):
        """Show context menu for browser tree view"""
        index = self.folder_tree.indexAt(position)
        if not index.isValid():
            return
        
        file_path = self.file_system_model.filePath(index)
        is_dir = self.file_system_model.isDir(index)
        
        menu = QtWidgets.QMenu(self)
        
        if is_dir:
            # Directory actions
            scan_action = menu.addAction("📁 Scan This Folder")
            scan_action.triggered.connect(lambda: self.start_indexing([Path(file_path)]))
            
            menu.addSeparator()
            
            reveal_action = menu.addAction("📂 Open in Explorer")
            reveal_action.triggered.connect(lambda: self._reveal_in_explorer(file_path))
        else:
            # File actions
            open_action = menu.addAction("🔓 Open File")
            open_action.triggered.connect(lambda: self._open_file(file_path))
            
            preview_action = menu.addAction("👁️ Preview")
            preview_action.triggered.connect(lambda: self._preview_file(file_path))
            
            menu.addSeparator()
            
            rename_action = menu.addAction("✏️ Rename (F2)")
            rename_action.triggered.connect(lambda: self._rename_file(file_path, index))
            
            menu.addSeparator()
            
            reveal_action = menu.addAction("📂 Show in Explorer")
            reveal_action.triggered.connect(lambda: self._reveal_in_explorer(file_path))
            
            copy_path_action = menu.addAction("📋 Copy Path")
            copy_path_action.triggered.connect(lambda: self._copy_path_to_clipboard(file_path))
            
            menu.addSeparator()
            
            delete_action = menu.addAction("🗑️ Delete...")
            delete_action.triggered.connect(lambda: self._delete_file(file_path))
        
        menu.exec(self.folder_tree.viewport().mapToGlobal(position))

    def _build_toolbar(self):
        """Build the top toolbar without duplicate buttons"""
        self.toolbar = QtWidgets.QToolBar()
        self.toolbar.setMovable(False)
        
        # Browse and Scan buttons
        btn_browse = QtWidgets.QToolButton()
        btn_browse.setText("📁 Browse Folder...")
        btn_browse.setToolTip("Open a folder to scan and index (recommended for fast, focused scans)")
        btn_browse.clicked.connect(self._browse_and_scan_folder)
        self.toolbar.addWidget(btn_browse)
        
        btn_scan = QtWidgets.QToolButton()
        btn_scan.setText("📁 Scan Folders")
        btn_scan.setToolTip("Select multiple folders to scan for 3D assets (STL, OBJ, FBX, etc.)")
        btn_scan.clicked.connect(self.choose_and_scan)
        self.toolbar.addWidget(btn_scan)
        
        btn_scan_all = QtWidgets.QToolButton()
        btn_scan_all.setText("💾 Scan All Drives")
        btn_scan_all.setToolTip("Scan all fixed drives")
        btn_scan_all.clicked.connect(self.scan_all_drives)
        self.toolbar.addWidget(btn_scan_all)
        
        self.toolbar.addSeparator()
        
        # Search
        self.search_input = QtWidgets.QLineEdit()
        self.search_input.setPlaceholderText("Search by name or tags...")
        self.search_input.setMinimumWidth(250)
        self.search_input.setMaximumWidth(400)
        self.search_input.setClearButtonEnabled(True)
        self.search_input.textChanged.connect(self._on_search_changed)
        self.search_input.returnPressed.connect(self._on_search_submit)
        self.search_input.setToolTip("Search files by name, extension, or tags - results update automatically (Ctrl+F)")
        
        search_label = QtWidgets.QLabel("🔍 ")
        search_label.setStyleSheet("font-size: 14px;")
        self.toolbar.addWidget(search_label)
        self.toolbar.addWidget(self.search_input)
        
        # Also keep reference as search_edit for backward compatibility
        self.search_edit = self.search_input
        
        self.toolbar.addSeparator()
        
        # Export CSV
        self.btn_export_csv = QtWidgets.QToolButton()
        self.btn_export_csv.setText("📊 Export CSV")
        self.btn_export_csv.setToolTip("Export current view to CSV")
        self.btn_export_csv.clicked.connect(self._export_csv)
        self.toolbar.addWidget(self.btn_export_csv)
        
        self.toolbar.addSeparator()
        
        # Advanced features (moved to menu - auto-computed during scan)
        # Geometry is now auto-computed during scanning for better UX
        
        self.toolbar.addSeparator()
        
        # Proposal and Excel import (only add once)
        if _PROPOSAL_AVAILABLE:
            self.btn_propose = QtWidgets.QToolButton()
            self.btn_propose.setText("🎯 Propose Names")
            self.btn_propose.setToolTip("Generate name proposals using reference parts")
            self.btn_propose.clicked.connect(self.on_propose_names_clicked)
            self.toolbar.addWidget(self.btn_propose)
        
        self.btn_import_excel = QtWidgets.QToolButton()
        self.btn_import_excel.setText("📋 Import Excel")
        self.btn_import_excel.setToolTip("Import Excel files with project part labels")
        self.btn_import_excel.clicked.connect(self.import_excel_dialog)
        self.toolbar.addWidget(self.btn_import_excel)
        
        self.toolbar.addSeparator()
        
        # View toggle
        self.btn_show_all = QtWidgets.QToolButton()
        self.btn_show_all.setText("👁️ Show All Files")
        self.btn_show_all.setCheckable(True)
        self.btn_show_all.setToolTip("Toggle between current scan vs all files")
        self.btn_show_all.clicked.connect(self.toggle_show_all_files)
        self.toolbar.addWidget(self.btn_show_all)
        
        self.toolbar.addSeparator()
        
        # Batch Tag button
        self.btn_batch_tag = QtWidgets.QToolButton()
        self.btn_batch_tag.setText("🏷️ Batch Tag")
        self.btn_batch_tag.setToolTip("Multi-select files with Ctrl+click, then apply tags")
        self.btn_batch_tag.clicked.connect(self.on_batch_tag_clicked)
        self.toolbar.addWidget(self.btn_batch_tag)
        
        # Add spacer to push theme toggle to the right
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        self.toolbar.addWidget(spacer)
        
        # Theme toggle - positioned at far right
        self.btn_theme_toggle = QtWidgets.QToolButton()
        self.btn_theme_toggle.setText("💡")
        self.btn_theme_toggle.setCheckable(True)
        self.btn_theme_toggle.setChecked(True)  # Start in dark mode
        self.btn_theme_toggle.setToolTip("Toggle Light/Dark Theme (Ctrl+Shift+T)")
        self.btn_theme_toggle.clicked.connect(self.toggle_theme)
        self.toolbar.addWidget(self.btn_theme_toggle)

    def _build_results_table(self):
        """Build the main results table with enhanced columns"""
        # Table
        self.table = QtWidgets.QTableView()
        self.model = EnhancedFileTableModel()
        self.table.setModel(self.model)
        
        # Table configuration
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked | QtWidgets.QAbstractItemView.EditTrigger.SelectedClicked)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)  # Allow multi-select with Ctrl
        self.table.setAlternatingRowColors(True)
        
        # Enable inline editing for Tags column only
        self.table.setItemDelegateForColumn(4, InlineEditDelegate(self.table))
        self.table.setSortingEnabled(True)
        
        # Event handlers
        self.table.clicked.connect(self._on_single_click)
        self.table.doubleClicked.connect(self._on_double_click)
        
        # Multi-select with right-click context menu
        self.table.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_context_menu)

    def _build_preview_panel(self):
        """Build the right preview/metadata panel with thumbnail"""
        self.preview_panel = QtWidgets.QWidget()
        self.preview_panel.setMinimumWidth(250)
        self.preview_panel.setMaximumWidth(320)
        v = QtWidgets.QVBoxLayout(self.preview_panel)
        v.setContentsMargins(10,10,10,10); v.setSpacing(8)
        
        title = QtWidgets.QLabel("Preview / Details")
        title.setStyleSheet("font-weight:600;font-size:14px;")
        v.addWidget(title)

        # Thumbnail cache and preview
        self.thumbnail_cache = ThumbnailCache(DB_DIR / "thumbnails")
        self.preview_thumbnail = QtWidgets.QLabel("No Preview\n(select a file)")
        self.preview_thumbnail.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preview_thumbnail.setMinimumHeight(200)
        self.preview_thumbnail.setStyleSheet(
            "QLabel{border:2px solid #666;border-radius:4px;background:#2b2b2b;padding:4px;color:#ccc;}"
        )
        self.preview_thumbnail.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.preview_thumbnail.customContextMenuRequested.connect(self._preview_context_menu)
        v.addWidget(self.preview_thumbnail)

        # quick info row
        quick = QtWidgets.QWidget(); hb = QtWidgets.QHBoxLayout(quick); hb.setContentsMargins(0,0,0,0)
        self.info_type = QtWidgets.QLabel("Type: N/A")
        self.info_tris  = QtWidgets.QLabel("Tris: N/A")
        hb.addWidget(self.info_type); hb.addWidget(self.info_tris); hb.addStretch(1)
        v.addWidget(quick)

        # metadata
        box = QtWidgets.QGroupBox("File Details"); form = QtWidgets.QFormLayout(box)
        self.meta_name = QtWidgets.QLabel("N/A"); self.meta_name.setWordWrap(True)
        self.meta_size = QtWidgets.QLabel("N/A")
        self.meta_modified = QtWidgets.QLabel("N/A")
        self.meta_path = QtWidgets.QLabel("N/A"); self.meta_path.setWordWrap(True)
        self.meta_path.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.meta_tags = QtWidgets.QLabel("N/A"); self.meta_tags.setWordWrap(True)
        self.meta_dimensions = QtWidgets.QLabel("N/A")

        form.addRow("Name:", self.meta_name)
        form.addRow("Size:", self.meta_size)
        form.addRow("Modified:", self.meta_modified)
        form.addRow("Path:", self.meta_path)
        form.addRow("Tags:", self.meta_tags)
        form.addRow("Dimensions:", self.meta_dimensions)
        v.addWidget(box)

        v.addStretch(1)
        self.current_preview_file = None

    # Event handlers
    def _on_filter_changed(self):
        """Handle filter changes with validation"""
        try:
            # Validate filter inputs before applying
            self._validate_filters()
            self.refresh_table()
        except Exception as e:
            print(f"Filter error: {e}")
            # Reset to safe defaults
            self._clear_filters()
            self.refresh_table()
    
    def _validate_filters(self):
        """Validate filter inputs"""
        # Validate file size filters
        try:
            min_size_text = self.min_size_input.text().strip()
            if min_size_text:
                min_size = float(min_size_text)
                if min_size < 0:
                    self.min_size_input.setText("0")
        except ValueError:
            self.min_size_input.setText("0")
        
        try:
            max_size_text = self.max_size_input.text().strip()
            if max_size_text:
                max_size = float(max_size_text)
                if max_size < 0:
                    self.max_size_input.setText("")
        except ValueError:
            self.max_size_input.setText("")
        
        # Validate triangle count filters
        try:
            min_tris_text = self.min_tris_input.text().strip()
            if min_tris_text:
                min_tris = int(min_tris_text)
                if min_tris < 0:
                    self.min_tris_input.setText("0")
        except ValueError:
            self.min_tris_input.setText("0")
        
        try:
            max_tris_text = self.max_tris_input.text().strip()
            if max_tris_text:
                max_tris = int(max_tris_text)
                if max_tris < 0:
                    self.max_tris_input.setText("")
        except ValueError:
            self.max_tris_input.setText("")

    def _clear_filters(self):
        """Clear all filters"""
        for cb in self.file_type_filters.values():
            cb.setChecked(True)
        self.project_filter.clear()
        self.part_type_filter.clear()
        self.min_size_filter.clear()
        self.max_size_filter.clear()
        self.min_tris_filter.clear()
        self.max_tris_filter.clear()
        self.refresh_table()

    def _on_single_click(self, index):
        """Handle single click on table row"""
        if index.isValid():
            row = self.model.rows[index.row()]
            self._update_preview_for_row(row)

    def _on_double_click(self, index):
        """Handle double click to open file"""
        if index.isValid():
            row = self.model.rows[index.row()]
            path = row[0] if row else None
            if path and Path(path).exists():
                os.startfile(path)

    def _update_preview_for_row(self, row: tuple):
        """Update preview panel with row data"""
        if not row: return
        path, name, ext, size, mtime, tags = row[:6]
        self.current_preview_file = path
        self.meta_name.setText(name or "N/A")
        self.meta_size.setText(f"{(size or 0)/(1024*1024):.2f} MB")
        self.meta_modified.setText(time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime)) if mtime else "N/A")
        disp = path if len(path) <= 60 else "..." + path[-57:]
        self.meta_path.setText(disp); self.meta_path.setToolTip(path)
        self.meta_tags.setText(tags or "No tags")
        self.info_type.setText(f"Type: {ext.upper() if ext else 'N/A'}")

        # geometry (if present in row ordering)
        tris = row[6] if len(row) > 6 else None
        if tris:
            self.info_tris.setText(f"Tris: {tris:,}")
        else:
            self.info_tris.setText("Tris: Computing...")
            # Trigger geometry computation for this file if missing
            if ext.lower() in ['.stl', '.obj', '.fbx', '.ply', '.glb']:
                worker = GeometryWorker([path])
                QtCore.QThreadPool.globalInstance().start(worker)
                # Refresh preview after geometry computation
                QtCore.QTimer.singleShot(3000, lambda: self._refresh_geometry_info(path))
        
        if len(row) > 9:
            dx, dy, dz = row[7], row[8], row[9]
            if all([dx,dy,dz]):
                self.meta_dimensions.setText(f"{dx:.1f} × {dy:.1f} × {dz:.1f}")
            else:
                self.meta_dimensions.setText("Computing...")
        else:
            self.meta_dimensions.setText("Computing...")
        
        # Load thumbnail preview
        self._load_thumbnail(path, name, ext)
    
    def _refresh_geometry_info(self, file_path: str):
        """Refresh geometry information after computation"""
        try:
            con = sqlite3.connect(DB_PATH)
            cur = con.cursor()
            cur.execute("SELECT tris, dim_x, dim_y, dim_z FROM files WHERE path = ?", (file_path,))
            result = cur.fetchone()
            con.close()
            
            if result and result[0]:
                tris, dx, dy, dz = result
                self.info_tris.setText(f"Tris: {tris:,}")
                if all([dx, dy, dz]):
                    self.meta_dimensions.setText(f"{dx:.1f} × {dy:.1f} × {dz:.1f}")
                else:
                    self.meta_dimensions.setText("N/A")
            else:
                self.info_tris.setText("Tris: Failed")
                self.meta_dimensions.setText("Failed")
        except Exception as e:
            print(f"Error refreshing geometry info: {e}")
    
    def _load_thumbnail(self, file_path: str, name: str, ext: str):
        """Load thumbnail for the given file"""
        if not file_path or not Path(file_path).exists():
            self.preview_thumbnail.setPixmap(QtGui.QPixmap())
            self.preview_thumbnail.setText(f"No Preview\n{name or ''}")
            return
        
        # 1) try cache
        cached = self.thumbnail_cache.get_thumbnail(file_path)
        if cached:
            self._display_thumbnail(cached)
            return

        # 2) For 3D files, generate thumbnail immediately
        if ext.lower() in ['.stl', '.obj', '.fbx', '.ply', '.glb']:
            self.preview_thumbnail.setText("Generating 3D preview...")
            # Generate thumbnail immediately in background
            worker = ThumbnailGenWorker(file_path, self.thumbnail_cache, 256)
            QtCore.QThreadPool.globalInstance().start(worker)
            # Refresh after generation
            QtCore.QTimer.singleShot(2000, lambda: self._refresh_if_cache_ready(file_path))
            return

        # 3) For other files, show neutral placeholder
        ph = QtGui.QPixmap(160, 160)
        ph.fill(QtGui.QColor("#2b2b2b"))
        painter = QtGui.QPainter(ph)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        pen = QtGui.QPen(QtGui.QColor("#555")); pen.setWidth(3); painter.setPen(pen)
        painter.setBrush(QtGui.QColor("#3a3a3a"))
        rect = ph.rect().adjusted(18, 18, -18, -18)
        painter.drawRoundedRect(rect, 12, 12)
        painter.setPen(QtGui.QPen(QtGui.QColor("#9aa7ff")))
        painter.drawText(ph.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, (ext or "").upper() or "FILE")
        painter.end()
        self._display_thumbnail(ph)

        # 4) kick off background generation for other files if enabled
        if self.settings.value("thumbs.autogen", "true").lower() == "true":
            self._schedule_thumbnail_generate(file_path)

    def _display_thumbnail(self, pm: QtGui.QPixmap):
        """Display a thumbnail pixmap"""
        if not pm or pm.isNull():
            self.preview_thumbnail.setPixmap(QtGui.QPixmap()); self.preview_thumbnail.setText("No Preview"); return
        scaled = pm.scaled(self.preview_thumbnail.size() - QtCore.QSize(10,10),
                           QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
        self.preview_thumbnail.setPixmap(scaled)
    
    def _preview_context_menu(self, pos):
        """Show context menu for preview thumbnail"""
        if not self.current_preview_file: return
        m = QtWidgets.QMenu(self.preview_thumbnail)
        m.addAction("🖼️ Generate Thumbnail", self._generate_thumbnail)
        
        # Add large preview option for 3D files
        ext = Path(self.current_preview_file).suffix.lower()
        if ext in ['.stl', '.obj', '.fbx', '.ply', '.glb']:
            m.addAction("🔍 Large Preview", self._generate_large_preview)
            m.addAction("🐛 Debug Preview", self._debug_preview)
        
        m.addAction("🗑️ Clear Thumbnail Cache", self._clear_thumbnail_cache)
        m.addSeparator()
        m.addAction("🔧 Open in External Viewer", self._open_in_external_viewer)
        m.exec(self.preview_thumbnail.mapToGlobal(pos))

    def _generate_thumbnail(self):
        """Generate thumbnail for current file"""
        if not self.current_preview_file:
            return
        worker = ThumbnailGenWorker(self.current_preview_file, self.thumbnail_cache, 256)
        QtCore.QThreadPool.globalInstance().start(worker)
        self.status.showMessage("Generating thumbnail…", 2500)
        QtCore.QTimer.singleShot(1200, lambda: self._refresh_if_cache_ready(self.current_preview_file))
    
    def _schedule_thumbnail_generate(self, file_path: str):
        """Schedule background thumbnail generation"""
        worker = ThumbnailGenWorker(file_path, self.thumbnail_cache, 256)
        QtCore.QThreadPool.globalInstance().start(worker)
        QtCore.QTimer.singleShot(1200, lambda: self._refresh_if_cache_ready(file_path))
    
    def _refresh_if_cache_ready(self, file_path: str):
        """Refresh thumbnail display if cache is ready"""
        pm = self.thumbnail_cache.get_thumbnail(file_path)
        if pm and not pm.isNull() and self.current_preview_file == file_path:
            self._display_thumbnail(pm)

    def _clear_thumbnail_cache(self):
        """Clear all cached thumbnails"""
        ret = QtWidgets.QMessageBox.question(self, "Clear Cache",
            "Clear all cached thumbnails?", QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
        if ret != QtWidgets.QMessageBox.StandardButton.Yes: return
        try:
            import shutil
            if self.thumbnail_cache.cache_dir.exists():
                shutil.rmtree(self.thumbnail_cache.cache_dir)
            self.thumbnail_cache.cache_dir.mkdir(parents=True, exist_ok=True)
            self.thumbnail_cache.memory_cache.clear()
            self.status.showMessage("Thumbnail cache cleared", 3000)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to clear cache: {e}")
    
    def _open_in_external_viewer(self):
        """Open current preview file in external viewer"""
        if self.current_preview_file and Path(self.current_preview_file).exists():
            os.startfile(self.current_preview_file)
    
    
    def _generate_thumbnail(self):
        """Generate thumbnail for current file"""
        try:
            if not self.current_preview_file:
                QtWidgets.QMessageBox.warning(self, "No File Selected", 
                    "No file is currently selected for preview.")
                return
            
            if not Path(self.current_preview_file).exists():
                QtWidgets.QMessageBox.warning(self, "File Not Found", 
                    "Selected file no longer exists.")
                return
            
            # Generate thumbnail in background
            worker = ThumbnailGenWorker(self.current_preview_file, self.thumbnail_cache, 256)
            QtCore.QThreadPool.globalInstance().start(worker)
            self.status.showMessage("Generating thumbnail...", 2500)
            
            # Refresh preview after generation
            QtCore.QTimer.singleShot(1500, lambda: self._refresh_if_cache_ready(self.current_preview_file))
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Thumbnail Error", f"Failed to generate thumbnail: {e}")

    def _generate_large_preview(self):
        """Generate a large preview image using solid_renderer"""
        try:
            if not self.current_preview_file:
                QtWidgets.QMessageBox.warning(self, "No File Selected", 
                    "No file is currently selected for preview.")
                return
            
            if not Path(self.current_preview_file).exists():
                QtWidgets.QMessageBox.warning(self, "File Not Found", 
                    "Selected file no longer exists.")
                return
            
            # Check if it's a 3D file
            ext = Path(self.current_preview_file).suffix.lower()
            if ext not in ['.stl', '.obj', '.fbx', '.ply', '.glb']:
                QtWidgets.QMessageBox.information(self, "Not a 3D File", 
                    "Large preview is only available for 3D model files.")
                return
            
            # Generate large preview using solid_renderer
            from solid_renderer import render_mesh_to_image
            from PIL.ImageQt import ImageQt
            
            self.status.showMessage("Generating large preview...", 3000)
            
            # Create a larger preview (512x512) with optimized settings for large files
            # Check file size to determine appropriate face limit for large preview
            file_size_mb = Path(self.current_preview_file).stat().st_size / (1024 * 1024)
            if file_size_mb > 200:  # Very large files (200MB+) - very conservative
                max_faces = 75000
            elif file_size_mb > 100:  # Large files (100-200MB)
                max_faces = 100000
            else:  # Normal files
                max_faces = 150000
            
            img = render_mesh_to_image(
                file_path=self.current_preview_file,
                size=(512, 512),
                bg_rgba=(245, 245, 245, 255),  # Clean white background
                face_rgb=(220, 220, 240),      # Light blue-gray for mesh
                outline_rgb=(160, 160, 180),   # Subtle edge outlines
                outline_width=1,
                max_faces=max_faces,           # Optimized for large files
                draw_edges=True
            )
            
            # Convert to QPixmap and display in a new window
            qimage = ImageQt(img)
            pixmap = QtGui.QPixmap.fromImage(qimage)
            
            # Create preview window
            preview_window = QtWidgets.QDialog(self)
            preview_window.setWindowTitle(f"3D Preview - {Path(self.current_preview_file).name}")
            preview_window.setModal(False)
            preview_window.resize(600, 600)
            
            layout = QtWidgets.QVBoxLayout(preview_window)
            
            # Add image label
            image_label = QtWidgets.QLabel()
            image_label.setPixmap(pixmap.scaled(512, 512, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation))
            image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(image_label)
            
            # Add buttons
            button_layout = QtWidgets.QHBoxLayout()
            
            save_btn = QtWidgets.QPushButton("Save Image...")
            save_btn.clicked.connect(lambda: self._save_preview_image(img, preview_window))
            button_layout.addWidget(save_btn)
            
            close_btn = QtWidgets.QPushButton("Close")
            close_btn.clicked.connect(preview_window.close)
            button_layout.addWidget(close_btn)
            
            layout.addLayout(button_layout)
            
            preview_window.show()
            # Show file size and face limit for large files
            if file_size_mb > 100:
                self.status.showMessage(f"Large preview generated ({file_size_mb:.1f}MB, {max_faces} faces)", 3000)
            else:
                self.status.showMessage("Large preview generated", 2000)
            
        except ImportError:
            QtWidgets.QMessageBox.warning(self, "Solid Renderer Not Available", 
                "The solid_renderer module is not available. Please ensure it's installed.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Preview Error", f"Failed to generate large preview: {e}")

    def _save_preview_image(self, img, parent_window):
        """Save the preview image to a file"""
        try:
            from pathlib import Path
            
            # Get default filename
            base_name = Path(self.current_preview_file).stem
            default_filename = f"{base_name}_preview.png"
            
            # Open save dialog
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                parent_window,
                "Save Preview Image",
                default_filename,
                "PNG Images (*.png);;JPEG Images (*.jpg);;All Files (*)"
            )
            
            if filename:
                img.save(filename)
                self.status.showMessage(f"Preview saved to {filename}", 3000)
                
        except Exception as e:
            QtWidgets.QMessageBox.critical(parent_window, "Save Error", f"Failed to save image: {e}")

    def _debug_preview(self):
        """Debug preview generation with diagnostic options"""
        try:
            if not self.current_preview_file:
                QtWidgets.QMessageBox.warning(self, "No File Selected", 
                    "No file is currently selected for preview.")
                return
            
            if not Path(self.current_preview_file).exists():
                QtWidgets.QMessageBox.warning(self, "File Not Found", 
                    "Selected file no longer exists.")
                return
            
            # Check if it's a 3D file
            ext = Path(self.current_preview_file).suffix.lower()
            if ext not in ['.stl', '.obj', '.fbx', '.ply', '.glb']:
                QtWidgets.QMessageBox.information(self, "Not a 3D File", 
                    "Debug preview is only available for 3D model files.")
                return
            
            from solid_renderer import render_mesh_to_image
            from PIL.ImageQt import ImageQt
            
            self.status.showMessage("Generating debug preview...", 3000)
            
            # Create debug preview with diagnostic options
            img = render_mesh_to_image(
                file_path=self.current_preview_file,
                size=(512, 512),
                bg_rgba=(245, 245, 245, 255),
                face_rgb=(220, 220, 240),
                outline_rgb=(160, 160, 180),
                outline_width=1,
                max_faces=150000,
                draw_edges=True,
                debug_no_downsample=True,  # Disable downsampling to see full mesh
                debug_force_hull=False    # Set to True to force convex hull
            )
            
            # Convert to QPixmap and display in a new window
            qimage = ImageQt(img)
            pixmap = QtGui.QPixmap.fromImage(qimage)
            
            # Create debug preview window
            debug_window = QtWidgets.QDialog(self)
            debug_window.setWindowTitle(f"Debug Preview - {Path(self.current_preview_file).name}")
            debug_window.setModal(False)
            debug_window.resize(700, 700)
            
            layout = QtWidgets.QVBoxLayout(debug_window)
            
            # Add debug info
            info_label = QtWidgets.QLabel("Debug Mode: No downsampling enabled")
            info_label.setStyleSheet("font-weight: bold; color: #ff6b6b;")
            layout.addWidget(info_label)
            
            # Add image label
            image_label = QtWidgets.QLabel()
            image_label.setPixmap(pixmap.scaled(512, 512, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation))
            image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(image_label)
            
            # Add debug buttons
            button_layout = QtWidgets.QHBoxLayout()
            
            hull_btn = QtWidgets.QPushButton("Force Convex Hull")
            hull_btn.clicked.connect(lambda: self._debug_with_hull(debug_window))
            button_layout.addWidget(hull_btn)
            
            save_btn = QtWidgets.QPushButton("Save Image...")
            save_btn.clicked.connect(lambda: self._save_preview_image(img, debug_window))
            button_layout.addWidget(save_btn)
            
            close_btn = QtWidgets.QPushButton("Close")
            close_btn.clicked.connect(debug_window.close)
            button_layout.addWidget(close_btn)
            
            layout.addLayout(button_layout)
            
            debug_window.show()
            self.status.showMessage("Debug preview generated", 2000)
            
        except ImportError:
            QtWidgets.QMessageBox.warning(self, "Solid Renderer Not Available", 
                "The solid_renderer module is not available. Please ensure it's installed.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Debug Preview Error", f"Failed to generate debug preview: {e}")

    def _debug_with_hull(self, parent_window):
        """Generate debug preview with forced convex hull"""
        try:
            from solid_renderer import render_mesh_to_image
            from PIL.ImageQt import ImageQt
            
            # Force convex hull to rule out degenerate geometry
            img = render_mesh_to_image(
                file_path=self.current_preview_file,
                size=(512, 512),
                bg_rgba=(245, 245, 245, 255),
                face_rgb=(220, 220, 240),
                outline_rgb=(160, 160, 180),
                outline_width=1,
                max_faces=150000,
                draw_edges=True,
                debug_no_downsample=True,
                debug_force_hull=True  # Force convex hull
            )
            
            # Update the preview in the same window
            qimage = ImageQt(img)
            pixmap = QtGui.QPixmap.fromImage(qimage)
            
            # Find and update the image label
            for child in parent_window.findChildren(QtWidgets.QLabel):
                if child.pixmap() is not None:
                    child.setPixmap(pixmap.scaled(512, 512, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation))
                    break
            
            # Update info label
            for child in parent_window.findChildren(QtWidgets.QLabel):
                if "Debug Mode" in child.text():
                    child.setText("Debug Mode: Forced convex hull (no downsampling)")
                    break
                    
        except Exception as e:
            QtWidgets.QMessageBox.critical(parent_window, "Debug Hull Error", f"Failed to generate hull preview: {e}")

    def _open_in_explorer(self):
        """Open selected file in Windows Explorer"""
        idx = self.table.currentIndex()
        if idx.isValid():
            row = self.model.rows[idx.row()]
            path = row[0] if row else None
            if path:
                os.startfile(str(Path(path).parent))


    def _context_menu(self, position):
        """Show context menu for table"""
        # Implementation for context menu
        pass
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for file operations"""
        # F2 - Rename selected file
        if event.key() == QtCore.Qt.Key.Key_F2:
            selected_rows = self.table.selectionModel().selectedRows()
            if len(selected_rows) == 1:
                row_idx = selected_rows[0].row()
                self._rename_file_from_table(row_idx)
            event.accept()
            return
        
        # Space - Preview selected file
        elif event.key() == QtCore.Qt.Key.Key_Space:
            selected_rows = self.table.selectionModel().selectedRows()
            if len(selected_rows) == 1:
                row_idx = selected_rows[0].row()
                if row_idx < len(self.model.rows):
                    file_path = self.model.rows[row_idx][0]
                    self._preview_file(file_path)
            event.accept()
            return
        
        # Delete - Delete selected files
        elif event.key() == QtCore.Qt.Key.Key_Delete:
            selected_rows = self.table.selectionModel().selectedRows()
            if selected_rows:
                self._delete_selected_files(selected_rows)
            event.accept()
            return
        
        # Pass other events to parent
        super().keyPressEvent(event)

    # Placeholder methods for existing functionality
    def _browse_and_scan_folder(self):
        """Browse and scan a single folder (recommended approach)"""
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Folder to Scan and Index",
            str(Path.home()),
            QtWidgets.QFileDialog.Option.ShowDirsOnly
        )
        
        if folder:
            folder_path = Path(folder)
            if folder_path.exists() and folder_path.is_dir():
                self.status.showMessage(f"Scanning folder: {folder_path.name}...", 0)
                QtWidgets.QApplication.processEvents()  # Update UI
                
                # Use the same indexing pipeline as regular scan
                self.start_indexing([folder_path])
            else:
                QtWidgets.QMessageBox.warning(self, "Invalid Folder", "Selected folder does not exist or is not accessible.")

    def choose_and_scan(self):
        """Choose folders to scan"""
        dlg = QtWidgets.QFileDialog(self, "Select one or more folders")
        dlg.setFileMode(QtWidgets.QFileDialog.FileMode.Directory)
        dlg.setOption(QtWidgets.QFileDialog.Option.ShowDirsOnly, True)
        dlg.setDirectory(str(Path.home()))
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            folders = [Path(p) for p in dlg.selectedFiles()]
            self.start_indexing(folders)

    def scan_all_drives(self):
        """Scan all fixed drives"""
        drives = list_fixed_drives()
        if not drives:
            QtWidgets.QMessageBox.warning(self, APP_NAME, "No fixed drives found.")
            return
        
        ret = QtWidgets.QMessageBox.question(
            self, APP_NAME,
            f"Scan all fixed drives? This may take a while.\n\n{', '.join([str(d) for d in drives])}"
        )
        if ret == QtWidgets.QMessageBox.StandardButton.Yes:
            self.start_indexing(drives)

    def toggle_show_all_files(self):
        """Toggle between current scan and all files"""
        self.show_all_files = self.btn_show_all.isChecked()
        if self.show_all_files:
            self.btn_show_all.setText("👁️ Show Current Scan")
            self.status.showMessage("Showing all files in database", 3000)
        else:
            self.btn_show_all.setText("👁️ Show All Files")
            self.status.showMessage("Showing current scan results only", 3000)
        self.refresh_table()

    def refresh_table(self):
        """Refresh the table with current filters"""
        # Get active file type filters
        active_exts = {ext for ext, cb in self.file_type_filters.items() if cb.isChecked()}
        if not active_exts:
            active_exts = None
        
        # Get search term
        search_term = self.search_edit.text()
        
        # Query database
        rows = query_files(search_term, active_exts)
        
        # Apply additional filters
        filtered_rows = []
        for row in rows:
            # Filter by current scan roots (unless "Show All Files" is enabled)
            if not self.show_all_files and self.current_scan_roots:
                file_path = Path(row[0]) if row and row[0] else None
                if file_path:
                    # Check if file is under any of the current scan roots
                    is_in_current_scan = False
                    for root in self.current_scan_roots:
                        try:
                            # Use resolve() to handle symlinks and normalize paths
                            if file_path.resolve().is_relative_to(Path(root).resolve()):
                                is_in_current_scan = True
                                break
                        except (ValueError, OSError):
                            # Fallback to string comparison if resolve() fails
                            if str(file_path).startswith(str(root)):
                                is_in_current_scan = True
                                break
                    
                    if not is_in_current_scan:
                        continue
            
            # Project filter
            if self.project_filter.text().strip():
                project_num = row[11] if len(row) > 11 else ""
                if self.project_filter.text().strip().lower() not in str(project_num).lower():
                    continue
            
            # Part type filter
            if self.part_type_filter.text().strip():
                part_name = row[12] if len(row) > 12 else ""
                if self.part_type_filter.text().strip().lower() not in str(part_name).lower():
                    continue
            
            # Size filters
            size_mb = (row[3] / (1024*1024)) if len(row) > 3 and row[3] else 0
            if self.min_size_filter.text().strip():
                try:
                    min_size = float(self.min_size_filter.text())
                    if size_mb < min_size:
                        continue
                except ValueError:
                    pass
            
            if self.max_size_filter.text().strip():
                try:
                    max_size = float(self.max_size_filter.text())
                    if size_mb > max_size:
                        continue
                except ValueError:
                    pass
            
            # Triangle filters
            tris = row[6] if len(row) > 6 and row[6] else 0
            if self.min_tris_filter.text().strip():
                try:
                    min_tris = int(self.min_tris_filter.text())
                    if tris < min_tris:
                        continue
                except ValueError:
                    pass
            
            if self.max_tris_filter.text().strip():
                try:
                    max_tris = int(self.max_tris_filter.text())
                    if tris > max_tris:
                        continue
                except ValueError:
                    pass
            
            filtered_rows.append(row)
        
        self.model.set_rows(filtered_rows)
        self.table.resizeColumnsToContents()

    def start_indexing(self, roots):
        """Start indexing folders"""
        # Ask user if they want to clear existing data for a fresh scan
        if self.current_scan_roots:  # Only ask if there's existing data
            ret = QtWidgets.QMessageBox.question(
                self, APP_NAME,
                "Clear existing database and start fresh?\n\n"
                "Yes = Clear all data, show only new scan\n"
                "No = Keep existing data, add new files",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
            )
            if ret == QtWidgets.QMessageBox.StandardButton.Yes:
                self._clear_database()
        
        self.current_scan_roots = roots
        self.status.showMessage(f"Indexing {len(roots)} folders...", 5000)
        
        # Start actual indexing in background thread
        self.indexer = Indexer(roots, ["windows\\winsxs", "$recycle.bin", "system volume information"])
        self.indexer.progress.connect(self._on_index_progress)
        self.indexer.finished_ok.connect(self._on_index_complete)
        self.indexer.error.connect(self._on_index_error)
        self.indexer.start()

    def _on_index_progress(self, found, scanned):
        """Handle indexing progress updates"""
        self.status.showMessage(f"Indexing... found {found} files, scanned {scanned} directories")

    def _on_index_complete(self):
        """Handle indexing completion"""
        self.status.showMessage("Indexing complete - auto-processing files...", 5000)
        self.refresh_table()  # Refresh table with new data
        
        # Auto-compute geometry and thumbnails for better UX
        QtCore.QTimer.singleShot(1000, self._auto_process_new_files)
    
    def _auto_process_new_files(self):
        """Automatically process newly scanned files for better UX"""
        try:
            # Get all files that need geometry computation
            con = sqlite3.connect(DB_PATH)
            cur = con.cursor()
            cur.execute("SELECT path FROM files WHERE tris IS NULL LIMIT 50")  # Process first 50 files
            files_to_process = [row[0] for row in cur.fetchall()]
            con.close()
            
            if files_to_process:
                self.status.showMessage(f"Auto-computing geometry for {len(files_to_process)} files...", 0)
                
                # Start geometry computation in background
                worker = GeometryWorker(files_to_process)
                QtCore.QThreadPool.globalInstance().start(worker)
                
                # Schedule thumbnail generation after geometry
                QtCore.QTimer.singleShot(3000, self._auto_generate_thumbnails)
            else:
                self.status.showMessage("All files already processed", 3000)
                
        except Exception as e:
            print(f"Auto-processing error: {e}")
    
    def _auto_generate_thumbnails(self):
        """Automatically generate thumbnails for new files"""
        try:
            # Get files that might need thumbnails
            con = sqlite3.connect(DB_PATH)
            cur = con.cursor()
            cur.execute("SELECT path FROM files LIMIT 20")  # Generate thumbnails for first 20 files
            files_for_thumbnails = [row[0] for row in cur.fetchall()]
            con.close()
            
            if files_for_thumbnails:
                self.status.showMessage("Auto-generating thumbnails...", 0)
                
                # Generate thumbnails in background
                for file_path in files_for_thumbnails[:10]:  # Limit to 10 for performance
                    if Path(file_path).exists():
                        worker = ThumbnailGenWorker(file_path, self.thumbnail_cache, 256)
                        QtCore.QThreadPool.globalInstance().start(worker)
                
                # Refresh display after thumbnails
                QtCore.QTimer.singleShot(2000, self.refresh_table)
                self.status.showMessage("Auto-processing complete", 3000)
                
        except Exception as e:
            print(f"Auto-thumbnail generation error: {e}")

    def _on_index_error(self, error_msg):
        """Handle indexing errors"""
        QtWidgets.QMessageBox.critical(self, "Indexing Error", f"Indexing failed: {error_msg}")
        self.status.clearMessage()

    def _clear_database(self):
        """Clear all files from the database"""
        try:
            con = sqlite3.connect(DB_PATH)
            cur = con.cursor()
            cur.execute("DELETE FROM files")
            con.commit()
            con.close()
            self.status.showMessage("Database cleared", 3000)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, APP_NAME, f"Failed to clear database: {e}")

    def _show_context_menu(self, position):
        """Show context menu with file operations and batch actions"""
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            return
        
        # Get the first selected file path
        first_row = selected_rows[0].row()
        if first_row >= len(self.model.rows):
            return
        file_path = self.model.rows[first_row][0]  # Path is first element
        
        # Create context menu
        context_menu = QtWidgets.QMenu(self)
        
        # File operations (single file only)
        if len(selected_rows) == 1:
            open_file_action = context_menu.addAction("🔓 Open File")
            open_file_action.triggered.connect(lambda: self._open_file(file_path))
            
            preview_action = context_menu.addAction("👁️ Preview")
            preview_action.triggered.connect(lambda: self._preview_file(file_path))
            
            context_menu.addSeparator()
            
            rename_action = context_menu.addAction("✏️ Rename (F2)")
            rename_action.triggered.connect(lambda: self._rename_file_from_table(first_row))
            
            # Check if file has a proposed name
            row_data = self.model.rows[first_row]
            proposed_name = row_data[14] if len(row_data) > 14 and row_data[14] else None
            
            if proposed_name:
                rename_proposed_action = context_menu.addAction("🎯 Rename to Proposed")
                rename_proposed_action.triggered.connect(lambda: self._rename_to_proposed(first_row))
            
            context_menu.addSeparator()
        
        # Batch operations
        batch_tag_action = context_menu.addAction(f"🏷️ Batch Tag ({len(selected_rows)} file{'s' if len(selected_rows) > 1 else ''})")
        batch_tag_action.triggered.connect(lambda: self._show_batch_tag_dialog_for_selection(selected_rows))
        
        # Find Similar Files action (only for single selection)
        find_sim_action = None
        if len(selected_rows) == 1 and _SIM_AVAILABLE:
            find_sim_action = context_menu.addAction("🔎 Find Similar Files")
            find_sim_action.setToolTip("Find files with similar names and tags")
            find_sim_action.triggered.connect(lambda: self._find_similar_from_table(first_row))
        
        context_menu.addSeparator()
        
        # Explorer actions
        if len(selected_rows) == 1:
            reveal_action = context_menu.addAction("📂 Show in Explorer")
            reveal_action.triggered.connect(lambda: self._reveal_in_explorer(file_path))
            
            copy_path_action = context_menu.addAction("📋 Copy Path")
            copy_path_action.triggered.connect(lambda: self._copy_path_to_clipboard(file_path))
        else:
            open_folders_action = context_menu.addAction("📁 Open Selected Folders in Explorer")
            open_folders_action.triggered.connect(lambda: self._open_selected_in_explorer(selected_rows))
        
        context_menu.addSeparator()
        
        # Delete action
        delete_action = context_menu.addAction(f"🗑️ Delete ({len(selected_rows)} file{'s' if len(selected_rows) > 1 else ''})...")
        delete_action.triggered.connect(lambda: self._delete_selected_files(selected_rows))
        
        # Show context menu
        context_menu.exec(self.table.mapToGlobal(position))
    
    def _find_similar_for_row(self, row_idx: int):
        """Find similar files for the given row index"""
        return find_similar_by_row(self, row_idx)
    
    def _find_similar_from_table(self, row_idx: int):
        """Find similar files and show results dialog"""
        try:
            results = self._find_similar_for_row(row_idx)
            if results:
                FindSimilarResultsDialog(self, results).exec()
            else:
                QtWidgets.QMessageBox.information(
                    self, "Find Similar",
                    "No similar files found.\n\n"
                    "Try building the similarity index first:\n"
                    "Tools → Build Similarity Index"
                )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to find similar files: {e}")
    
    def _rename_to_proposed(self, row_idx: int):
        """Rename file to its proposed name"""
        try:
            if row_idx >= len(self.model.rows):
                return
            
            row = self.model.rows[row_idx]
            file_path = row[0]
            
            # Check if we have a proposed name in the row data
            proposed_name = None
            if len(row) > 10:  # Check if proposed name column exists
                proposed_name = row[10]  # Adjust index based on your column structure
            
            if not proposed_name:
                QtWidgets.QMessageBox.information(self, "No Proposed Name", 
                    "No proposed name available for this file.")
                return
            
            # Get directory and create new path
            old_path = Path(file_path)
            new_path = old_path.parent / proposed_name
            
            if new_path.exists():
                reply = QtWidgets.QMessageBox.question(
                    self, "File Exists", 
                    f"File {proposed_name} already exists. Overwrite?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                )
                if reply != QtWidgets.QMessageBox.Yes:
                    return
            
            # Rename the file
            old_path.rename(new_path)
            
            # Update database
            self._update_tags_in_db(str(new_path), "renamed")
            
            # Refresh table
            self.refresh_table()
            
            QtWidgets.QMessageBox.information(self, "File Renamed", 
                f"Renamed to: {proposed_name}")
                
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Rename Error", f"Failed to rename file: {e}")
    
    # File operation methods
    def _open_file(self, file_path: str):
        """Open file with default application"""
        try:
            import subprocess
            if sys.platform == "win32":
                os.startfile(file_path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", file_path])
            else:
                subprocess.Popen(["xdg-open", file_path])
            self.status.showMessage(f"Opened: {Path(file_path).name}", 2000)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to open file:\n{e}")
    
    def _reveal_in_explorer(self, file_path: str):
        """Reveal file in file explorer"""
        try:
            import subprocess
            file_path = str(Path(file_path).resolve())
            if sys.platform == "win32":
                subprocess.Popen(["explorer", "/select,", file_path])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", file_path])
            else:
                # Linux - open parent folder
                subprocess.Popen(["xdg-open", str(Path(file_path).parent)])
            self.status.showMessage(f"Revealed: {Path(file_path).name}", 2000)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to reveal file:\n{e}")
    
    def _copy_path_to_clipboard(self, file_path: str):
        """Copy file path to clipboard"""
        try:
            clipboard = QtWidgets.QApplication.clipboard()
            clipboard.setText(str(file_path))
            self.status.showMessage(f"Copied to clipboard: {file_path}", 3000)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to copy path:\n{e}")
    
    def _rename_file(self, file_path: str, tree_index=None):
        """Rename a file (from folder browser)"""
        old_path = Path(file_path)
        new_name, ok = QtWidgets.QInputDialog.getText(
            self, "Rename File",
            f"Rename:\n{old_path.name}\n\nNew name:",
            QtWidgets.QLineEdit.EchoMode.Normal,
            old_path.name
        )
        
        if ok and new_name and new_name != old_path.name:
            try:
                new_path = old_path.parent / new_name
                old_path.rename(new_path)
                
                # Update database
                con = sqlite3.connect(DB_PATH)
                cur = con.cursor()
                cur.execute("UPDATE files SET path = ?, name = ? WHERE path = ?",
                          (str(new_path), new_name, str(old_path)))
                con.commit()
                con.close()
                
                self.refresh_table()
                self.status.showMessage(f"Renamed to: {new_name}", 3000)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Error", f"Failed to rename file:\n{e}")
    
    def _rename_file_from_table(self, row_idx: int):
        """Rename a file (from table view)"""
        if row_idx >= len(self.model.rows):
            return
        file_path = self.model.rows[row_idx][0]
        self._rename_file(file_path)
    
    def _rename_to_proposed(self, row_idx: int):
        """Rename file to its proposed name"""
        if row_idx >= len(self.model.rows):
            return
        
        row_data = self.model.rows[row_idx]
        old_path = Path(row_data[0])
        proposed_name = row_data[14] if len(row_data) > 14 and row_data[14] else None
        confidence = row_data[15] if len(row_data) > 15 and row_data[15] else 0
        
        if not proposed_name:
            QtWidgets.QMessageBox.warning(self, "Error", "No proposed name available for this file.")
            return
        
        # Show confirmation with confidence
        conf_pct = int(confidence * 100) if confidence else 0
        ret = QtWidgets.QMessageBox.question(
            self, "Rename to Proposed",
            f"Rename file to proposed name?\n\n"
            f"Current: {old_path.name}\n"
            f"Proposed: {proposed_name}\n"
            f"Confidence: {conf_pct}%\n\n"
            f"This will rename the file on disk and update the database.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
        )
        
        if ret == QtWidgets.QMessageBox.StandardButton.Yes:
            try:
                new_path = old_path.parent / proposed_name
                
                # Check if destination exists
                if new_path.exists() and new_path != old_path:
                    QtWidgets.QMessageBox.warning(
                        self, "Error",
                        f"File already exists:\n{new_path.name}\n\nCannot rename."
                    )
                    return
                
                # Rename file
                old_path.rename(new_path)
                
                # Update database
                con = sqlite3.connect(DB_PATH)
                cur = con.cursor()
                cur.execute("""
                    UPDATE files 
                    SET path = ?, name = ?, status = 'renamed'
                    WHERE path = ?
                """, (str(new_path), proposed_name, str(old_path)))
                con.commit()
                con.close()
                
                # Log user correction for active learning
                if _PROPOSAL_AVAILABLE:
                    try:
                        # Extract part type and laterality from proposed name
                        part_name = proposed_name.lower()
                        laterality = ("left" if "left_" in part_name else
                                    "right" if "right_" in part_name else "center")
                        
                        # Extract part type (remove laterality prefix)
                        if laterality != "center":
                            part_type = part_name.replace(f"{laterality}_", "").split("_")[-1]
                        else:
                            part_type = part_name.split("_")[-1].split(".")[0]
                        
                        add_user_correction(
                            file_path=str(new_path),
                            old_name=old_path.name,
                            new_name=proposed_name,
                            project_number=None,  # Could extract from proposed name if needed
                            part_type=part_type,
                            laterality=laterality,
                            confidence=confidence
                        )
                    except Exception as e:
                        print(f"Failed to log user correction: {e}")
                
                self.refresh_table()
                self.status.showMessage(f"Renamed to: {proposed_name}", 3000)
                
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Error", f"Failed to rename file:\n{e}")
    
    def _delete_file(self, file_path: str):
        """Delete a single file"""
        ret = QtWidgets.QMessageBox.question(
            self, "Delete File",
            f"Are you sure you want to delete:\n\n{Path(file_path).name}\n\nThis cannot be undone!",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
        )
        
        if ret == QtWidgets.QMessageBox.StandardButton.Yes:
            try:
                Path(file_path).unlink()
                
                # Remove from database
                con = sqlite3.connect(DB_PATH)
                cur = con.cursor()
                cur.execute("DELETE FROM files WHERE path = ?", (file_path,))
                con.commit()
                con.close()
                
                self.refresh_table()
                self.status.showMessage(f"Deleted: {Path(file_path).name}", 3000)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Error", f"Failed to delete file:\n{e}")
    
    def _delete_selected_files(self, selected_rows):
        """Delete multiple selected files"""
        file_paths = []
        for row_index in selected_rows:
            row = row_index.row()
            if row < len(self.model.rows):
                file_paths.append(self.model.rows[row][0])
        
        if not file_paths:
            return
        
        # Confirm deletion
        ret = QtWidgets.QMessageBox.question(
            self, "Delete Files",
            f"Are you sure you want to delete {len(file_paths)} file(s)?\n\nThis cannot be undone!",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
        )
        
        if ret == QtWidgets.QMessageBox.StandardButton.Yes:
            deleted_count = 0
            failed_count = 0
            
            con = sqlite3.connect(DB_PATH)
            cur = con.cursor()
            
            for file_path in file_paths:
                try:
                    Path(file_path).unlink()
                    cur.execute("DELETE FROM files WHERE path = ?", (file_path,))
                    deleted_count += 1
                except Exception as e:
                    failed_count += 1
                    print(f"Failed to delete {file_path}: {e}")
            
            con.commit()
            con.close()
            
            self.refresh_table()
            
            msg = f"Deleted {deleted_count} file(s)"
            if failed_count > 0:
                msg += f", {failed_count} failed"
            self.status.showMessage(msg, 3000)
    
    def _preview_file(self, file_path: str):
        """Preview file using Windows native preview or thumbnail"""
        # First, update the preview panel with thumbnail
        try:
            # Find the row in the model
            for i, row in enumerate(self.model.rows):
                if row[0] == file_path:
                    self._update_preview_for_row(row)
                    break
            
            # Generate/load thumbnail if available
            if hasattr(self, 'thumb_cache') and _GEO_AVAILABLE:
                thumb = self.thumb_cache.get_thumbnail(file_path)
                if thumb:
                    self.preview_label.setPixmap(thumb.scaled(
                        256, 256,
                        QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                        QtCore.Qt.TransformationMode.SmoothTransformation
                    ))
                else:
                    # Queue thumbnail generation
                    worker = ThumbnailGenWorker(file_path, self.thumb_cache, 256)
                    self.threadpool.start(worker)
                    self.preview_label.setText("Generating preview...")
            
            # Use Windows native preview (Space bar style quick look)
            self._show_windows_preview(file_path)
            
        except Exception as e:
            print(f"Preview error: {e}")
    
    def _show_windows_preview(self, file_path: str):
        """Show file in Windows native preview pane or 3D Viewer app"""
        try:
            # On Windows 10/11, we can open the file which will use the default viewer
            # For 3D files, this usually opens Windows 3D Viewer or Mixed Reality Viewer
            if sys.platform == "win32":
                # Option 1: Just open with default app (3D Viewer for .stl, .obj, etc.)
                os.startfile(file_path)
                self.status.showMessage(f"Opening preview: {Path(file_path).name}", 2000)
            else:
                # On other platforms, just open with default app
                import subprocess
                if sys.platform == "darwin":
                    subprocess.Popen(["open", file_path])
                else:
                    subprocess.Popen(["xdg-open", file_path])
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to preview file:\n{e}")

    def _show_batch_tag_dialog_for_selection(self, selected_rows):
        """Show batch tag dialog for selected rows"""
        tag_text, ok = QtWidgets.QInputDialog.getText(
            self, "Batch Tag", 
            f"Enter tags for {len(selected_rows)} selected file(s):\n\n"
            "Use commas to separate multiple tags (e.g., 'character,weapon,prototype')"
        )
        
        if ok and tag_text.strip():
            self._apply_batch_tags_to_selected_rows(selected_rows, tag_text.strip())

    def _apply_batch_tags_to_selected_rows(self, selected_rows, tag_text):
        """Apply batch tags to selected rows"""
        updated_count = 0
        for idx in selected_rows:
            row = idx.row()
            if 0 <= row < len(self.model.rows):
                file_path = self.model.rows[row][0]  # Get file path from first column
                
                # Update tags in database
                if self._update_tags_in_db(file_path, tag_text):
                    # Update the local model data
                    row_data = list(self.model.rows[row])
                    row_data[5] = tag_text  # Update tags column (index 5)
                    self.model.rows[row] = tuple(row_data)
                    updated_count += 1
        
        # Refresh the table to show changes
        if updated_count > 0:
            self.model.dataChanged.emit(
                self.model.index(0, 4),
                self.model.index(len(self.model.rows) - 1, 4)
            )
            
            # Show success message
            QtWidgets.QMessageBox.information(
                self, "Batch Tag Complete",
                f"Successfully tagged {updated_count} file(s)!\n\n"
                f"Applied tags: {tag_text}"
            )
            
            self.status.showMessage(f"Batch tagged {updated_count} files", 3000)

    def _open_selected_in_explorer(self, selected_rows):
        """Open selected files/folders in Windows Explorer"""
        for idx in selected_rows:
            row = idx.row()
            if 0 <= row < len(self.model.rows):
                path = self.model.rows[row][0]
                if Path(path).exists():
                    # Open the parent folder and select the file
                    import subprocess
                    subprocess.run(['explorer', '/select,', str(path)])

    def _simulate_indexing_complete(self):
        """Simulate indexing completion"""
        self.status.showMessage("Indexing complete", 5000)
        self.refresh_table()

    # Menu action methods
    def _new_project(self):
        """Create a new project"""
        QtWidgets.QMessageBox.information(self, "New Project", "New Project functionality coming soon!")

    def _open_project(self):
        """Open an existing project"""
        QtWidgets.QMessageBox.information(self, "Open Project", "Open Project functionality coming soon!")

    def _save_project(self):
        """Save current project"""
        QtWidgets.QMessageBox.information(self, "Save Project", "Project saved!")

    def _find_dialog(self):
        """Open find dialog"""
        text, ok = QtWidgets.QInputDialog.getText(self, "Find", "Enter search text:")
        if ok and text:
            self.search_edit.setText(text)
            self.refresh_table()

    def _find_next(self):
        """Find next occurrence (F3)"""
        if not hasattr(self, 'table') or not self.table.model():
            return
        
        current_row = self.table.currentIndex().row()
        total_rows = self.table.model().rowCount()
        
        if total_rows == 0:
            self.status.showMessage("No results to navigate", 2000)
            return
        
        # Move to next row (wrap around)
        next_row = (current_row + 1) % total_rows
        self.table.selectRow(next_row)
        self.table.scrollTo(self.table.model().index(next_row, 0))
        self.status.showMessage(f"Result {next_row + 1} of {total_rows}", 2000)

    def _find_previous(self):
        """Find previous occurrence (Shift+F3)"""
        if not hasattr(self, 'table') or not self.table.model():
            return
        
        current_row = self.table.currentIndex().row()
        total_rows = self.table.model().rowCount()
        
        if total_rows == 0:
            self.status.showMessage("No results to navigate", 2000)
            return
        
        # Move to previous row (wrap around)
        prev_row = (current_row - 1) % total_rows
        self.table.selectRow(prev_row)
        self.table.scrollTo(self.table.model().index(prev_row, 0))
        self.status.showMessage(f"Result {prev_row + 1} of {total_rows}", 2000)

    def _select_all(self):
        """Select all items in table"""
        self.table.selectAll()

    def _clear_selection(self):
        """Clear table selection"""
        self.table.clearSelection()

    def _toggle_filters_panel(self):
        """Toggle filters panel visibility"""
        self.filters_panel.setVisible(not self.filters_panel.isVisible())

    def _toggle_preview_panel(self):
        """Toggle preview panel visibility"""
        self.preview_panel.setVisible(not self.preview_panel.isVisible())

    def _toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _database_manager(self):
        """Open database manager"""
        dlg = DatabaseManagerDialog(self)
        dlg.exec()

    def _show_settings(self):
        """Show settings dialog"""
        dlg = SettingsDialog(self)
        dlg.exec()
    
    def _open_training_dialog(self):
        """Open ML training dialog"""
        dlg = TrainingDialog(self)
        dlg.exec()
        # Model might be updated, so status message
        self.status.showMessage("Training session complete", 3000)
    
    def _on_retrain_from_corrections(self):
        """Retrain model from user corrections"""
        if not _PROPOSAL_AVAILABLE:
            QtWidgets.QMessageBox.warning(
                self, "Retrain Unavailable",
                "Active learning system is not available. Please check your installation."
            )
            return
        
        try:
            rep = retrain_from_corrections(pop_after_train=True)  # clears used corrections if training ok
            QtWidgets.QMessageBox.information(
                self, "Retrain Complete", 
                f"{rep.get('msg','Done')}\nUsed: {rep.get('used',0)}  Skipped: {rep.get('skipped',0)}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Retrain Failed",
                f"Failed to retrain from corrections: {e}"
            )
    
    def _open_migration_planner(self):
        """Open migration planner dialog"""
        dlg = MigrationPlannerDialog(self)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            # Refresh table after successful migration
            self.refresh_table()
            self.status.showMessage("Migration completed successfully", 5000)
    
    def _show_diagnostics(self):
        """Show diagnostics dialog"""
        dlg = DiagnosticsDialog(self)
        dlg.exec()

    def _show_docs(self):
        """Show documentation"""
        QtWidgets.QMessageBox.information(self, "Documentation", "Documentation coming soon!")

    def _show_shortcuts(self):
        """Show keyboard shortcuts"""
        shortcuts = """
Keyboard Shortcuts:

File:
  Ctrl+N    - New Project
  Ctrl+O    - Open Project
  Ctrl+S    - Save Project
  Ctrl+I    - Import Excel
  Ctrl+E    - Export CSV
  Ctrl+Q    - Exit

Edit:
  Ctrl+F    - Find
  F3        - Find Next
  Shift+F3  - Find Previous
  Ctrl+A    - Select All
  Escape    - Clear Selection
  Ctrl+T    - Batch Tag
  Ctrl+Click - Multi-select files
  Right-Click - Context menu with Batch Tag option

View:
  Ctrl+L    - Show All Files
  F5        - Refresh
  Ctrl+Shift+T - Toggle Theme
  F11       - Full Screen

Tools:
  Ctrl+R    - Scan Folders
  Ctrl+G    - Compute Geometry
  Ctrl+B    - Build Similarity Index
  Ctrl+P    - Propose Names (if available)
        """
        QtWidgets.QMessageBox.information(self, "Keyboard Shortcuts", shortcuts)

    def _show_about(self):
        """Show about dialog"""
        about_text = f"""
{APP_NAME} - 3D Asset Finder

Professional 3D model management and organization tool.

Features:
• Advanced filtering and search
• 3D preview and metadata display
• Project-based organization
• Excel import for reference parts
• AI-powered name proposals
• Similarity search and matching

Version: 2.0 Enhanced UI
Built with PySide6 and modern UI design principles.
        """
        QtWidgets.QMessageBox.about(self, f"About {APP_NAME}", about_text)

    # CSV Export functionality
    def _export_csv(self): 
        """Export current view to CSV"""
        dlg = ExportCSVDialog(self)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            options = dlg.get_options()
            self._do_export_csv(options)
    
    def _do_export_csv(self, options):
        """Perform CSV export with given options"""
        # Get file name
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export CSV",
            str(Path.home() / f"modelfinder_export_{time.strftime('%Y%m%d_%H%M%S')}.csv"),
            "CSV Files (*.csv);;All Files (*)"
        )
        if not fn:
            return
        
        try:
            # Get rows from model
            rows = self.model.rows
            
            # Apply filters
            if options['path_contains']:
                rows = [r for r in rows if options['path_contains'].lower() in r[0].lower()]
            
            if options['exts']:
                rows = [r for r in rows if r[2] in options['exts']]
            
            # Apply limit
            if options['limit'] > 0:
                rows = rows[:options['limit']]
            
            # Write CSV
            with open(fn, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Header
                if options['include_header']:
                    writer.writerow(options['columns'])
                
                # Data rows
                for row in rows:
                    csv_row = []
                    for col in options['columns']:
                        if col == "Name":
                            csv_row.append(row[1] if len(row) > 1 else "")
                        elif col == "Extension":
                            csv_row.append(row[2] if len(row) > 2 else "")
                        elif col == "Size (MB)":
                            size = row[3] if len(row) > 3 else 0
                            csv_row.append(f"{size / (1024*1024):.2f}" if size else "0.00")
                        elif col == "Modified":
                            mtime = row[4] if len(row) > 4 else 0
                            csv_row.append(time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime)) if mtime else "")
                        elif col == "Tags":
                            csv_row.append(row[5] if len(row) > 5 else "")
                        elif col == "Path":
                            csv_row.append(row[0] if len(row) > 0 else "")
                        elif col == "Parent Folder":
                            csv_row.append(str(Path(row[0]).parent) if len(row) > 0 else "")
                        elif col == "Drive":
                            csv_row.append(str(Path(row[0]).drive) if len(row) > 0 else "")
                    writer.writerow(csv_row)
            
            QtWidgets.QMessageBox.information(
                self, "Export Complete",
                f"Exported {len(rows)} rows to:\n{fn}"
            )
            self.status.showMessage(f"Exported {len(rows)} rows to CSV", 5000)
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Export Failed",
                f"Failed to export CSV:\n{e}"
            )
    
    
    def _start_geometry_compute(self):
        """Start geometry computation for selected files"""
        try:
            selected = self.table.selectionModel().selectedRows()
            if not selected:
                QtWidgets.QMessageBox.information(self, "Compute Geometry", "Select one or more files first.")
                return
            
            paths = []
            for idx in selected:
                row = self.model.rows[idx.row()]
                if row and Path(row[0]).exists():
                    paths.append(row[0])
            
            if not paths:
                QtWidgets.QMessageBox.information(self, "Compute Geometry", "No valid files selected.")
                return
            
            self.status.showMessage(f"Computing geometry for {len(paths)} file(s)…")
            worker = GeometryWorker(paths)
            QtCore.QThreadPool.globalInstance().start(worker)
            
            # Quick refresh a bit later
            QtCore.QTimer.singleShot(1500, self.refresh_table)
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Geometry Compute Error", f"Failed to start geometry computation: {e}")
    
    def _rebuild_similarity_index(self): 
        """Rebuild similarity index"""
        try:
            if not _SIM_AVAILABLE:
                QtWidgets.QMessageBox.warning(
                    self, "Similarity Search",
                    "Similarity search requires additional libraries:\n\n"
                    "pip install numpy faiss-cpu scikit-learn joblib"
                )
                return
        
            self.status.showMessage("Building similarity index...", 0)
            QtWidgets.QApplication.processEvents()  # Update UI
            
            ok, msg = build_similarity_index()
            if ok:
                QtWidgets.QMessageBox.information(self, "Similarity Index", msg)
                self.status.showMessage(msg, 5000)
            else:
                QtWidgets.QMessageBox.critical(self, "Similarity Index", f"Failed to build index:\n{msg}")
                self.status.clearMessage()
                
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Similarity Index Error", f"Failed to rebuild similarity index: {e}")
            self.status.clearMessage()
    
    def import_excel_dialog(self): 
        """Import Excel files with project part labels"""
        try:
            # Check if required libraries are available
            import pandas as pd
            import openpyxl
        except ImportError:
            QtWidgets.QMessageBox.warning(
                self, APP_NAME,
                "Excel import requires additional libraries.\n\n"
                "Please install them:\n"
                "pip install pandas openpyxl"
            )
            return
        
        # File selection dialog
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Excel File with Project Parts",
            str(Path.home()),
            "Excel Files (*.xlsx *.xls);;All Files (*)"
        )
        
        if not file_path:
            return
        
        # Create import dialog with options
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Import Excel - Configure Options")
        dlg.resize(500, 400)
        
        layout = QtWidgets.QVBoxLayout(dlg)
        
        # File info
        file_group = QtWidgets.QGroupBox("File Information")
        file_layout = QtWidgets.QFormLayout(file_group)
        
        file_name = Path(file_path).name
        file_size = Path(file_path).stat().st_size / (1024 * 1024)
        
        file_layout.addRow("File:", QtWidgets.QLabel(file_name))
        file_layout.addRow("Size:", QtWidgets.QLabel(f"{file_size:.2f} MB"))
        
        # Try to detect project info from filename
        detected_project = ""
        detected_name = ""
        
        # Common patterns: "300868_ProjectName.xlsx" or "300868-Project Name.xlsx"
        import re
        match = re.match(r"(\d+)[_\-\s]+(.+?)\.xls", file_name)
        if match:
            detected_project = match.group(1)
            detected_name = match.group(2).replace("_", " ").replace("-", " ")
        
        layout.addWidget(file_group)
        
        # Import options
        options_group = QtWidgets.QGroupBox("Import Options")
        options_layout = QtWidgets.QFormLayout(options_group)
        
        # Project number
        project_num = QtWidgets.QLineEdit()
        project_num.setText(detected_project)
        project_num.setPlaceholderText("e.g., 300868")
        options_layout.addRow("Project Number:", project_num)
        
        # Project name
        project_name = QtWidgets.QLineEdit()
        project_name.setText(detected_name)
        project_name.setPlaceholderText("e.g., Character Assets")
        options_layout.addRow("Project Name:", project_name)
        
        # Sheet selection (will be populated after reading file)
        sheet_combo = QtWidgets.QComboBox()
        sheet_combo.setToolTip("Select which Excel sheet to import")
        
        # Column mapping
        col_group = QtWidgets.QGroupBox("Column Mapping")
        col_layout = QtWidgets.QFormLayout(col_group)
        
        part_col = QtWidgets.QComboBox()
        part_col.setEditable(True)
        part_col.setCurrentText("Part Name")
        
        desc_col = QtWidgets.QComboBox()
        desc_col.setEditable(True)
        desc_col.setCurrentText("Description")
        
        qty_col = QtWidgets.QComboBox()
        qty_col.setEditable(True)
        qty_col.setCurrentText("Quantity")
        
        col_layout.addRow("Part Name Column:", part_col)
        col_layout.addRow("Description Column:", desc_col)
        col_layout.addRow("Quantity Column:", qty_col)
        
        options_layout.addRow("Excel Sheet:", sheet_combo)
        options_layout.addRow(col_group)
        
        # Import to database option
        import_to_db = QtWidgets.QCheckBox("Import directly to database")
        import_to_db.setChecked(True)
        import_to_db.setToolTip("Add parts to ModelFinder database for auto-suggestions")
        options_layout.addRow(import_to_db)
        
        # Skip duplicates option
        skip_dupes = QtWidgets.QCheckBox("Skip duplicate entries")
        skip_dupes.setChecked(True)
        options_layout.addRow(skip_dupes)
        
        layout.addWidget(options_group)
        
        # Preview area
        preview_group = QtWidgets.QGroupBox("Preview (first 5 rows)")
        preview_layout = QtWidgets.QVBoxLayout(preview_group)
        
        preview_table = QtWidgets.QTableWidget()
        preview_table.setMaximumHeight(150)
        preview_layout.addWidget(preview_table)
        
        layout.addWidget(preview_group)
        
        # Load Excel file and populate UI
        try:
            # Read Excel file
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            sheet_combo.addItems(sheet_names)
            
            def update_preview():
                """Update preview when sheet changes"""
                sheet_name = sheet_combo.currentText()
                if not sheet_name:
                    return
                
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=5)
                    
                    # Update column combos
                    columns = df.columns.tolist()
                    for combo in [part_col, desc_col, qty_col]:
                        current = combo.currentText()
                        combo.clear()
                        combo.addItems(columns)
                        if current in columns:
                            combo.setCurrentText(current)
                    
                    # Update preview table
                    preview_table.setRowCount(min(5, len(df)))
                    preview_table.setColumnCount(len(columns))
                    preview_table.setHorizontalHeaderLabels(columns)
                    
                    for row in range(min(5, len(df))):
                        for col in range(len(columns)):
                            value = str(df.iloc[row, col])
                            preview_table.setItem(row, col, QtWidgets.QTableWidgetItem(value))
                    
                    preview_table.resizeColumnsToContents()
                    
                except Exception as e:
                    print(f"Error updating preview: {e}")
            
            sheet_combo.currentTextChanged.connect(update_preview)
            update_preview()  # Initial preview
            
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                dlg, "Read Error",
                f"Could not read Excel file:\n{e}"
            )
        
        # Buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | 
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(dlg.accept)
        button_box.rejected.connect(dlg.reject)
        layout.addWidget(button_box)
        
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        
        # Process import
        self._process_excel_import(
            file_path,
            {
                'project_number': project_num.text().strip(),
                'project_name': project_name.text().strip(),
                'sheet_name': sheet_combo.currentText(),
                'part_col': part_col.currentText(),
                'desc_col': desc_col.currentText(),
                'qty_col': qty_col.currentText(),
                'import_to_db': import_to_db.isChecked(),
                'skip_duplicates': skip_dupes.isChecked()
            }
        )

    def _process_excel_import(self, file_path, options):
        """Process the Excel import with the given options"""
        try:
            import pandas as pd
            
            # Read the Excel file
            df = pd.read_excel(file_path, sheet_name=options['sheet_name'])
            
            if df.empty:
                QtWidgets.QMessageBox.warning(self, APP_NAME, "The selected sheet is empty.")
                return
            
            # Extract relevant columns
            parts = []
            
            for _, row in df.iterrows():
                try:
                    part_name = str(row[options['part_col']]).strip() if options['part_col'] in df.columns else ""
                    description = str(row[options['desc_col']]).strip() if options['desc_col'] in df.columns else ""
                    
                    # Try to get quantity, default to 1
                    quantity = 1
                    if options['qty_col'] in df.columns:
                        try:
                            quantity = int(row[options['qty_col']])
                        except (ValueError, TypeError):
                            quantity = 1
                    
                    if part_name and part_name.lower() not in ['nan', 'none', '']:
                        parts.append({
                            'project_number': options['project_number'],
                            'project_name': options['project_name'],
                            'part_name': part_name,
                            'description': description,
                            'quantity': quantity,
                            'original_label': part_name  # Keep original for reference
                        })
                except Exception as e:
                    print(f"Error processing row: {e}")
                    continue
            
            if not parts:
                QtWidgets.QMessageBox.warning(
                    self, APP_NAME,
                    "No valid parts found in the Excel file.\n"
                    "Check that the column names are correct."
                )
                return
            
            # Import to database if requested
            if options['import_to_db']:
                imported = self._import_parts_to_db(parts, options['skip_duplicates'])
                
                QtWidgets.QMessageBox.information(
                    self, "Import Complete",
                    f"Successfully imported {imported} parts from {len(parts)} total.\n\n"
                    f"Project: {options['project_number']} - {options['project_name']}\n\n"
                    "These parts will now be used for name proposals."
                )
                
                self.status.showMessage(f"Imported {imported} parts from Excel", 5000)
            else:
                # Just show what was found
                QtWidgets.QMessageBox.information(
                    self, "Excel Parsed",
                    f"Found {len(parts)} parts in the Excel file.\n\n"
                    f"Project: {options['project_number']} - {options['project_name']}"
                )
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Import Failed",
                f"Failed to import Excel file:\n{e}"
            )

    def _import_parts_to_db(self, parts, skip_duplicates=True):
        """Import parts to the project_reference_parts table"""
        imported = 0
        
        try:
            con = sqlite3.connect(DB_PATH)
            cur = con.cursor()
            
            for part in parts:
                try:
                    if skip_duplicates:
                        # Check if already exists
                        cur.execute(
                            """SELECT COUNT(*) FROM project_reference_parts 
                            WHERE project_number=? AND part_name=?""",
                            (part['project_number'], part['part_name'])
                        )
                        if cur.fetchone()[0] > 0:
                            continue
                    
                    # Insert the part
                    cur.execute(
                        """INSERT INTO project_reference_parts 
                        (project_number, project_name, part_name, original_label, description, quantity)
                        VALUES (?, ?, ?, ?, ?, ?)""",
                        (
                            part['project_number'],
                            part['project_name'],
                            part['part_name'],
                            part['original_label'],
                            part.get('description', ''),
                            part.get('quantity', 1)
                        )
                    )
                    imported += 1
                    
                except sqlite3.IntegrityError:
                    # Duplicate, skip
                    continue
                except Exception as e:
                    print(f"Error inserting part: {e}")
                    continue
            
            con.commit()
            con.close()
            
        except Exception as e:
            print(f"Database error: {e}")
        
        return imported
    
    def _selected_rows_to_meta(self):
        """Convert selected table rows to RowMeta objects"""
        metas = []
        selected = self.table.selectionModel().selectedRows()
        for idx in selected:
            row = self.model.rows[idx.row()]
            metas.append(RowMeta(
                path=row[0],
                name=row[1],
                ext=row[2],
                tags=row[5] if len(row) > 5 else ""
            ))
        return metas

    def on_propose_names_clicked(self):
        """Handle Propose Names button click with reference picker"""
        # Get available projects from reference parts
        available_projects = []
        try:
            available_projects = get_all_projects(str(DB_PATH))
        except Exception as e:
            print(f"Error loading projects: {e}")
        
        if not available_projects:
            QtWidgets.QMessageBox.warning(
                self, "No Reference Projects",
                "No reference projects found in database.\n\n"
                "Please import an Excel file first using:\n"
                "File → Import Excel..."
            )
            return
        
        # Show project picker dialog
        dialog = ProjectPickerDialog(self, available_projects)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        
        project_number = dialog.get_selected_project()
        if not project_number:
            return
        
        rows = self._selected_rows_to_meta()
        if not rows:
            QtWidgets.QMessageBox.information(self, "Propose Names", "Select at least one file.")
            return
        
        self.btn_propose.setEnabled(False)
        self.status.showMessage(f"Generating proposals for project {project_number}...")
        
        # Create and start worker
        worker = ProposeWorker(propose_for_rows, rows, str(DB_PATH), project_number.strip())
        worker.signals.finished.connect(self._apply_proposals_to_ui)
        worker.signals.error.connect(self._on_propose_error)
        self.threadpool.start(worker)

    def _apply_proposals_to_ui(self, proposals):
        """Apply proposals to UI and database"""
        if not proposals:
            self.btn_propose.setEnabled(True)
            self.status.showMessage("No proposals generated.", 3000)
            return
        
        # Update database with proposals
        batch_update_proposals(proposals)
        self.btn_propose.setEnabled(True)
        
        # Show summary
        needs_review = sum(1 for p in proposals if p.get("needs_review", False))
        auto_accept = len(proposals) - needs_review
        
        QtWidgets.QMessageBox.information(
            self, "Proposal Summary",
            f"{len(proposals)} processed\n"
            f"{auto_accept} auto-accepted\n"
            f"{needs_review} need review"
        )
        self.refresh_table()

    def _on_propose_error(self, msg):
        """Handle proposal errors"""
        QtWidgets.QMessageBox.critical(self, "Proposal Error", msg)
        self.btn_propose.setEnabled(True)
        self.status.clearMessage()

    def on_batch_tag_clicked(self):
        """Handle batch tag button click"""
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            QtWidgets.QMessageBox.information(self, "Batch Tag", "Please select files first.\n\nUse Ctrl+click to select multiple files.")
            return
        
        # Show batch tag dialog
        tag_text, ok = QtWidgets.QInputDialog.getText(
            self, "Batch Tag", 
            f"Enter tags for {len(selected_rows)} selected file(s):\n\n"
            "Use commas to separate multiple tags (e.g., 'character,weapon,prototype')"
        )
        
        if not ok or not tag_text.strip():
            return
        
        # Apply tags to all selected rows
        updated_count = 0
        for idx in selected_rows:
            row = self.model.rows[idx.row()]
            file_path = row[0]  # Get file path from first column
            
            # Update tags in database
            if self._update_tags_in_db(file_path, tag_text.strip()):
                # Update the local model data
                row_data = list(self.model.rows[idx.row()])
                row_data[5] = tag_text.strip()  # Update tags column (index 5)
                self.model.rows[idx.row()] = tuple(row_data)
                updated_count += 1
        
        # Refresh the table to show changes
        self.model.dataChanged.emit(
            self.model.index(0, 4),
            self.model.index(len(self.model.rows) - 1, 4)
        )
        
        # Show success message
        QtWidgets.QMessageBox.information(
            self, "Batch Tag Complete",
            f"Successfully updated tags for {updated_count} file(s).\n\n"
            f"Applied tags: {tag_text.strip()}"
        )
        
        self.status.showMessage(f"Batch tagged {updated_count} files", 3000)

    def _update_tags_in_db(self, file_path, tags):
        """Update tags in database - helper method"""
        try:
            con = sqlite3.connect(DB_PATH)
            cur = con.cursor()
            cur.execute("UPDATE files SET tags = ? WHERE path = ?", (tags, file_path))
            con.commit()
            con.close()
            return True
        except Exception as e:
            print(f"Error updating tags for {file_path}: {e}")
            return False

# -----------------------------
# Application Entry Point
# -----------------------------

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion("2.0")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
