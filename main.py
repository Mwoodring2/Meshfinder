"""
ModelFinder — MVP (Windows)
PySide6-based desktop app to index and search 3D assets: .obj, .fbx, .stl, .ma, .mb

Features
- One-click scan of chosen folders or all fixed drives
- Background indexing thread writes to SQLite (in %APPDATA%/ModelFinder/index.db)
- Fast keyword search with live filtering
- Open file / open in Explorer
- Tagging (comma-separated keywords) with inline edit, persisted to DB
- Settings persisted via QSettings

Packaging
- pip install -r requirements.txt
- pyinstaller --noconfirm --windowed --icon app.ico main.py

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
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

# Optional deps for geometry metadata & thumbnails
try:
    import trimesh
    _GEO_AVAILABLE = True
except Exception:
    _GEO_AVAILABLE = False

# --- Similarity search (TF‑IDF + FAISS) imports ---
# Optional at runtime; used by the "Find Similar…" context action and the builder
try:
    import numpy as np
    import faiss
    import joblib
    from sklearn.preprocessing import normalize as _sk_normalize
    from sklearn.feature_extraction.text import TfidfVectorizer
    _SIM_AVAILABLE = True
except Exception:
    _SIM_AVAILABLE = False

APP_NAME = "ModelFinder"
DB_DIR = Path(os.environ.get("APPDATA", str(Path.home() / ".model_finder"))) / APP_NAME
DB_PATH = DB_DIR / "index.db"
SUPPORTED_EXTS = {".obj", ".fbx", ".stl", ".ma", ".mb"}

# -----------------------------
# Data layer
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
            tags TEXT DEFAULT ''
        );
        """
    )
    # Indices
    cur.execute("CREATE INDEX IF NOT EXISTS idx_name ON files(name);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ext ON files(ext);")

    # --- Schema migration: geometry metadata (Phase 2) ---
    # SQLite lacks IF NOT EXISTS for columns; run ALTERs wrapped in try/except.
    for ddl in [
        "ALTER TABLE files ADD COLUMN tris INTEGER",
        "ALTER TABLE files ADD COLUMN dim_x REAL",
        "ALTER TABLE files ADD COLUMN dim_y REAL",
        "ALTER TABLE files ADD COLUMN dim_z REAL",
        "ALTER TABLE files ADD COLUMN volume REAL",
        "ALTER TABLE files ADD COLUMN watertight INTEGER"
    ]:
        try:
            cur.execute(ddl)
        except Exception:
            pass
    con.commit()
    con.close()


def upsert_file(path: Path):
    try:
        stat = path.stat()
    except Exception:
        return
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO files(path, name, ext, size, mtime)
        VALUES(?,?,?,?,?)
        ON CONFLICT(path) DO UPDATE SET
            size=excluded.size,
            mtime=excluded.mtime
        """,
        (str(path), path.name, path.suffix.lower(), stat.st_size, stat.st_mtime),
    )
    con.commit()
    con.close()


def query_files(term: str, exts: set[str] | None = None, limit: int = 5000):
    term = term.strip()
    sql = "SELECT path,name,ext,size,mtime,tags FROM files"
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


def update_tags(path: str, tags: str):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("UPDATE files SET tags=? WHERE path=?", (tags, path))
    con.commit()
    con.close()

# -----------------------------
# Indexing worker (thread)
# -----------------------------

class Indexer(QtCore.QThread):
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

    def run(self):
        try:
            found = 0
            scanned = 0
            for root in self.roots:
                for dirpath, dirnames, filenames in os.walk(root, topdown=True):
                    # prune excluded dirs
                    dirnames[:] = [d for d in dirnames if not self._is_excluded(Path(dirpath) / d)]
                    if self._stop.is_set():
                        self.finished_ok.emit()
                        return
                    for fn in filenames:
                        scanned += 1
                        ext = Path(fn).suffix.lower()
                        if ext in SUPPORTED_EXTS:
                            p = Path(dirpath) / fn
                            upsert_file(p)
                            found += 1
                            self.indexed.emit(str(p))
                        if scanned % 500 == 0:
                            self.progress.emit(found, scanned)
            self.progress.emit(found, scanned)
            self.finished_ok.emit()
        except Exception as e:
            self.error.emit(str(e))

    def _is_excluded(self, path: Path) -> bool:
        p = str(path).lower()
        return any(x in p for x in self.excludes)

# -----------------------------
# Similarity Builder (TF‑IDF + FAISS) — runs in background
# -----------------------------
class SimilarityBuilder(QtCore.QThread):
    message = QtCore.Signal(str)
    finished_ok = QtCore.Signal(int)  # num indexed
    error = QtCore.Signal(str)

    def __init__(self, db_path: Path, art_dir: Path):
        super().__init__()
        self.db_path = db_path
        self.art_dir = art_dir

    def _build_text(self, path: str, name: str, ext: str, tags: str) -> str:
        p = Path(path or "")
        parents = []
        if p.parent.name: parents.append(p.parent.name)
        if p.parent.parent and p.parent.parent.name: parents.append(p.parent.parent.name)
        parts = [Path(name or "").stem, (ext or "").lstrip("."), tags or "", *parents]
        return " ".join(x for x in parts if x).strip()

    def run(self):
        try:
            if not _SIM_AVAILABLE:
                raise RuntimeError("Similarity deps missing (numpy, faiss, joblib, scikit-learn)")
            self.art_dir.mkdir(parents=True, exist_ok=True)
            con = sqlite3.connect(self.db_path)
            cur = con.cursor()
            cur.execute("SELECT id, path, name, ext, tags FROM files ORDER BY id ASC")
            rows = cur.fetchall()
            con.close()
            if not rows:
                raise RuntimeError("No rows in database. Scan some folders first.")
            texts, ids = [], []
            for _id, path, name, ext, tags in rows:
                t = self._build_text(path, name, ext, tags)
                if t:
                    texts.append(t)
                    ids.append(_id)
            if not texts:
                raise RuntimeError("All rows produced empty text. Add tags or verify filenames.")
            self.message.emit("Vectorizing…")
            vec = TfidfVectorizer(
                lowercase=True,
                sublinear_tf=True,
                token_pattern=r"(?u)\b[\w\-\.\#/]+\b",
                ngram_range=(1, 2),
                min_df=2,
                max_features=75000,
            )
            X = vec.fit_transform(texts).astype("float32")
            from sklearn.preprocessing import normalize
            X = normalize(X, norm="l2", copy=False)
            X = X.toarray().astype("float32")
            d = X.shape[1]
            self.message.emit("Building FAISS index…")
            index = faiss.IndexFlatIP(d)
            index.add(X)
            # save artifacts
            faiss.write_index(index, str(self.art_dir / "faiss_tfidf.index"))
            joblib.dump(vec, self.art_dir / "tfidf_vectorizer.joblib")
            np.save(self.art_dir / "faiss_ids.npy", np.asarray(ids, dtype=np.int64))
            self.finished_ok.emit(len(ids))
        except Exception as e:
            self.error.emit(str(e))

# -----------------------------
# Utils (Windows)
# -----------------------------

def list_fixed_drives() -> list[Path]:
    # Windows: get logical drives and filter by DRIVE_FIXED
    drives = []
    try:
        kernel32 = ctypes.windll.kernel32
        GetLogicalDrives = kernel32.GetLogicalDrives
        bitmask = GetLogicalDrives()
        for i in range(26):
            if bitmask & (1 << i):
                drive = f"{string.ascii_uppercase[i]}:\\"
                # DRIVE_FIXED == 3
                type_code = ctypes.windll.kernel32.GetDriveTypeW(drive)
                if type_code == 3:
                    drives.append(Path(drive))
    except Exception:
        # Fallback: common C:, D:, E:
        for d in ["C:\\", "D:\\", "E:\\"]:
            if Path(d).exists():
                drives.append(Path(d))
    return drives

# -----------------------------
# Geometry helpers (Phase 2)
# -----------------------------

def _compute_geometry(path: Path):
    """Return (tris, dim_x, dim_y, dim_z, volume, watertight) or None on failure."""
    if not _GEO_AVAILABLE:
        return None
    try:
        mesh = trimesh.load(str(path), force='mesh', skip_materials=True)
        # Some files may contain scenes; take the first geometry if needed
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())
        if not isinstance(mesh, trimesh.Trimesh):
            return None
        tris = int(len(mesh.faces)) if mesh.faces is not None else 0
        dx, dy, dz = [float(x) for x in (mesh.extents if mesh.extents is not None else (0,0,0))]
        vol = float(mesh.volume) if getattr(mesh, 'volume', None) else None
        wt = int(bool(getattr(mesh, 'is_watertight', False)))
        return tris, dx, dy, dz, vol, wt
    except Exception:
        return None


def _update_geometry_in_db(p: str, g):
    if g is None:
        return
    tris, dx, dy, dz, vol, wt = g
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "UPDATE files SET tris=?, dim_x=?, dim_y=?, dim_z=?, volume=?, watertight=? WHERE path=?",
        (tris, dx, dy, dz, vol, wt, p)
    )
    con.commit(); con.close()


class GeometryWorker(QtCore.QThread):
    message = QtCore.Signal(str)
    progress = QtCore.Signal(int, int)  # done, total
    finished_ok = QtCore.Signal(int)
    error = QtCore.Signal(str)

    def __init__(self, limit: int | None = None):
        super().__init__()
        self.limit = limit
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        if not _GEO_AVAILABLE:
            self.error.emit("Geometry deps missing (install trimesh, numpy)")
            return
        try:
            con = sqlite3.connect(DB_PATH)
            cur = con.cursor()
            # Candidates: supported extensions, no geometry yet
            q = (
                "SELECT path FROM files WHERE ext IN (?,?,?,?,?) "
                "AND (tris IS NULL OR dim_x IS NULL OR volume IS NULL)"
            )
            cur.execute(q, tuple(sorted(SUPPORTED_EXTS)))
            rows = [r[0] for r in cur.fetchall()]
            con.close()
            total = len(rows)
            if self.limit:
                rows = rows[: self.limit]
                total = len(rows)
            done = 0
            for p in rows:
                if self._stop.is_set():
                    break
                g = _compute_geometry(Path(p))
                if g is not None:
                    _update_geometry_in_db(p, g)
                done += 1
                if done % 25 == 0:
                    self.progress.emit(done, total)
            self.progress.emit(done, total)
            self.finished_ok.emit(done)
        except Exception as e:
            self.error.emit(str(e))

# -----------------------------
# UI
# -----------------------------

class FileTableModel(QtCore.QAbstractTableModel):
    headers = ["Name", "Extension", "Size (MB)", "Modified", "Tags", "Path"]

    def __init__(self):
        super().__init__()
        self.rows: list[tuple] = []

    def set_rows(self, rows: list[tuple]):
        self.beginResetModel()
        self.rows = rows
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.rows)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self.headers)

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        path, name, ext, size, mtime, tags = self._unpack(self.rows[index.row()])
        col = index.column()
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if col == 0: return name
            if col == 1: return ext
            if col == 2: return f"{size/1024/1024:.2f}"
            if col == 3: return time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))
            if col == 4: return tags
            if col == 5: return path
        if role == QtCore.Qt.ItemDataRole.EditRole and col == 4:
            return tags
        return None

    def headerData(self, section, orientation, role):
        if role == QtCore.Qt.ItemDataRole.DisplayRole and orientation == QtCore.Qt.Orientation.Horizontal:
            return self.headers[section]
        return None

    def flags(self, index):
        fl = super().flags(index)
        if index.column() == 4:
            fl |= QtCore.Qt.ItemFlag.ItemIsEditable
        return fl

    def setData(self, index, value, role):
        if role == QtCore.Qt.ItemDataRole.EditRole and index.column() == 4:
            row = self.rows[index.row()]
            path = row[0]
            update_tags(path, str(value))
            # refresh from DB or update local
            lst = list(row)
            lst[5] = str(value)
            self.rows[index.row()] = tuple(lst)
            self.dataChanged.emit(index, index)
            return True
        return False

    @staticmethod
    def _unpack(row):
        # row is (path,name,ext,size,mtime,tags)
        return row[0], row[1], row[2], row[3], row[4], row[5]


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} — 3D Asset Finder")
        self.resize(1100, 700)
        self.settings = QtCore.QSettings("ModelFinder", "ModelFinder")
        self.indexer: Indexer | None = None
        self._build_ui()
        ensure_db()
        self.refresh_table()

    def _build_ui(self):
        # Toolbar with search
        toolbar = QtWidgets.QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Geometry compute (Phase 2)
        self.btn_compute_geo = QtWidgets.QToolButton()
        self.btn_compute_geo.setText("Compute Geometry")
        self.btn_compute_geo.setToolTip("Compute tri counts, bounds, and volume in background")
        self.btn_compute_geo.clicked.connect(self._start_geometry_compute)
        toolbar.addWidget(self.btn_compute_geo)

        # Path Filter (Phase 1)
        self.path_filter = QtWidgets.QLineEdit()
        self.path_filter.setPlaceholderText("Filter by folder path… (paste part of a path and press Enter)")
        self.path_filter.returnPressed.connect(self.refresh_table)
        toolbar.addWidget(self.path_filter)

        # Export CSV (Phase 1 + extras)
        self.btn_export_csv = QtWidgets.QToolButton()
        self.btn_export_csv.setText("Export CSV")
        self.btn_export_csv.setToolTip("Export the current table view to a CSV file")
        self.btn_export_csv.clicked.connect(self._export_csv)
        toolbar.addWidget(self.btn_export_csv)

        # Bulk Tag (Phase 1.5)
        self.btn_bulk_tag = QtWidgets.QToolButton()
        self.btn_bulk_tag.setText("Bulk Tag…")
        self.btn_bulk_tag.setToolTip("Auto-tag rows by folder keywords")
        self.btn_bulk_tag.clicked.connect(self._bulk_tag_dialog)
        toolbar.addWidget(self.btn_bulk_tag)

        # Build FAISS button (runs in background)
        self.btn_build_sim = QtWidgets.QToolButton()
        self.btn_build_sim.setText("Rebuild Similarity Index")
        self.btn_build_sim.clicked.connect(self._rebuild_similarity_index)
        toolbar.addWidget(self.btn_build_sim)

        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.setPlaceholderText("Search by name or tags…")
        self.search_edit.returnPressed.connect(self.refresh_table)
        toolbar.addWidget(self.search_edit)

        self.ext_filter = QtWidgets.QComboBox()
        self.ext_filter.addItem("All")
        for e in sorted(SUPPORTED_EXTS):
            self.ext_filter.addItem(e)
        self.ext_filter.currentIndexChanged.connect(self.refresh_table)
        toolbar.addWidget(self.ext_filter)

        btn_scan = QtWidgets.QToolButton()
        btn_scan.setText("Scan Folders…")
        btn_scan.clicked.connect(self.choose_and_scan)
        toolbar.addWidget(btn_scan)

        btn_scan_all = QtWidgets.QToolButton()
        btn_scan_all.setText("Scan All Drives")
        btn_scan_all.clicked.connect(self.scan_all_drives)
        toolbar.addWidget(btn_scan_all)

        # Central table
        self.table = QtWidgets.QTableView()
        self.model = FileTableModel()
        self.table.setModel(self.model)
        # Highlight behavior: single-click selects full row (no inline edit)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        # Ensure full-row selection (single selection)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        # Ensure the selected row is visibly highlighted
        self.table.setStyleSheet(
            "QTableView::item:selected{background:palette(highlight);color:palette(highlighted-text);}"
        )
        # Single left-click handler
        self.table.clicked.connect(self._on_single_click)
        self.table.doubleClicked.connect(self.open_item)
        self.table.setSortingEnabled(False)

        # context menu
        self.table.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._context_menu)

        # Status bar
        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)

        # Layout
        # --- Preview Pane (below the table) ---
        self.preview_img = QtWidgets.QLabel()
        self.preview_img.setMinimumHeight(140)
        self.preview_img.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preview_img.setStyleSheet("QLabel { background-color: #111; border: 1px solid #333; color: #bbb; }")
        self.preview_meta = QtWidgets.QTextEdit()
        self.preview_meta.setReadOnly(True)
        self.preview_meta.setFixedHeight(120)

        preview_box = QtWidgets.QWidget()
        pv = QtWidgets.QVBoxLayout(preview_box)
        pv.setContentsMargins(6, 6, 6, 6)
        pv.setSpacing(6)
        pv.addWidget(self.preview_img)
        pv.addWidget(self.preview_meta)

        # Splitter holds table (top) + preview (bottom)
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        split.addWidget(self.table)
        split.addWidget(preview_box)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 1)

        wrapper = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(wrapper)
        v.setContentsMargins(0,0,0,0)
        v.addWidget(split)
        self.setCentralWidget(wrapper)

    # ---- actions ----
    def choose_and_scan(self):
        dlg = QtWidgets.QFileDialog(self, "Select one or more folders")
        dlg.setFileMode(QtWidgets.QFileDialog.FileMode.Directory)
        dlg.setOption(QtWidgets.QFileDialog.Option.ShowDirsOnly, True)
        dlg.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog, False)
        dlg.setOption(QtWidgets.QFileDialog.Option.ReadOnly, True)
        dlg.setDirectory(str(Path.home()))
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            folders = [Path(p) for p in dlg.selectedFiles()]
            self.start_indexing(folders)

    def scan_all_drives(self):
        drives = list_fixed_drives()
        if not drives:
            QtWidgets.QMessageBox.warning(self, APP_NAME, "No fixed drives found.")
            return
        ret = QtWidgets.QMessageBox.question(
            self, APP_NAME,
            f"Scan all fixed drives? This may take a while.\n\n{', '.join([d.as_posix() for d in drives])}"
        )
        if ret == QtWidgets.QMessageBox.StandardButton.Yes:
            self.start_indexing(drives)

    def start_indexing(self, roots: list[Path]):
        if self.indexer and self.indexer.isRunning():
            QtWidgets.QMessageBox.information(self, APP_NAME, "Indexing is already running.")
            return
        excludes = ["windows\\winsxs", "$recycle.bin", "system volume information"]
        self.indexer = Indexer(roots, excludes)
        self.indexer.progress.connect(self.on_progress)
        self.indexer.indexed.connect(self.on_indexed)
        self.indexer.finished_ok.connect(self.on_index_finish)
        self.indexer.error.connect(self.on_index_error)
        self.status.showMessage("Indexing…")
        self.indexer.start()

    def on_progress(self, found: int, scanned: int):
        self.status.showMessage(f"Indexing… scanned {scanned:,} files, found {found:,} assets")

    def on_indexed(self, path: str):
        # Incremental UX: could append to table; simplest is refresh every N
        pass

    def on_index_finish(self):
        self.status.showMessage("Indexing complete", 5000)
        self.refresh_table()

    def on_index_error(self, msg: str):
        QtWidgets.QMessageBox.critical(self, APP_NAME, f"Indexing error: {msg}")
        self.status.clearMessage()

    def refresh_table(self):
        term = self.search_edit.text()
        chosen = self.ext_filter.currentText()
        exts = None if chosen == "All" else {chosen}
        rows = query_files(term, exts)
        # Phase 1: apply optional path filter (substring match, case-insensitive)
        pf = self.path_filter.text().strip()
        if pf:
            low = pf.lower()
            rows = [r for r in rows if low in (r[0] or "").lower()]  # r[0] = path
        self.model.set_rows(rows)
        self.table.resizeColumnsToContents()

    def _current_row_path(self) -> str | None:
        idx = self.table.currentIndex()
        if not idx.isValid():
            return None
        row = self.model.rows[idx.row()]
        return row[0]  # path

    def open_item(self, index: QtCore.QModelIndex):
        row = self.model.rows[index.row()]
        path = row[0]
        try:
            os.startfile(path)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, APP_NAME, f"Cannot open file: {e}")

    def _context_menu(self, pos):
        index = self.table.indexAt(pos)
        if not index.isValid():
            return

        menu = QtWidgets.QMenu(self)
        actOpen = menu.addAction("Open")
        actReveal = menu.addAction("Show in Explorer")
        actCopyPath = menu.addAction("Copy Path")

        # New actions
        actRename = menu.addAction("Rename…")
        actFindSimilar = menu.addAction("Find Similar…")

        actEditTags = None
        if index.column() == 4:  # Tags column
            actEditTags = menu.addAction("Edit Tags…")

        action = menu.exec(self.table.viewport().mapToGlobal(pos))
        if not action:
            return

        path = self._current_row_path()
        if not path:
            return

        if action == actOpen:
            self._open_path(path)
        elif action == actReveal:
            self._reveal_in_explorer(path)
        elif action == actCopyPath:
            QtWidgets.QApplication.clipboard().setText(path)
        elif action == actRename:
            self._rename_item(index)
        elif action == actFindSimilar:
            self._find_similar_from_index(index)
        elif actEditTags and action == actEditTags:
            self._edit_tags_dialog(index)

    def _open_path(self, path: str):
        try:
            os.startfile(path)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, APP_NAME, f"Cannot open file: {e}")

    def _reveal_in_explorer(self, path: str):
        try:
            QtCore.QProcess.startDetached("explorer", ["/select,", path])
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, APP_NAME, f"Cannot reveal file: {e}")

    def _rename_item(self, index: QtCore.QModelIndex):
        # Safe on-disk rename + DB update
        try:
            row = self.model.rows[index.row()]
        except Exception:
            return
        old_path = Path(row[0])
        cur_base = old_path.stem
        cur_ext = old_path.suffix  # includes dot

        new_base, ok = QtWidgets.QInputDialog.getText(
            self,
            "Rename File",
            f"Rename in folder:\n{old_path.parent}\n\nCurrent name (without extension):",
            QtWidgets.QLineEdit.Normal,
            cur_base,
        )
        if not ok or not new_base.strip():
            return

        new_name = new_base.strip() + cur_ext
        new_path = old_path.parent / new_name

        if new_path.exists():
            ret = QtWidgets.QMessageBox.question(
                self,
                "Overwrite?",
                f"\"{new_name}\" already exists. Replace it?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            )
            if ret != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        try:
            os.replace(str(old_path), str(new_path))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, APP_NAME, f"Rename failed:\n{e}")
            return

        # Update DB: path, name, ext, size, mtime
        try:
            stat = new_path.stat()
            con = sqlite3.connect(DB_PATH)
            cur = con.cursor()
            cur.execute(
                "UPDATE files SET path=?, name=?, ext=?, size=?, mtime=? WHERE path=?",
                (str(new_path), new_name, new_path.suffix.lower(), stat.st_size, stat.st_mtime, str(old_path)),
            )
            con.commit()
            con.close()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, APP_NAME, f"Renamed on disk but DB update failed:\n{e}")

        self.status.showMessage(f"Renamed to {new_name}", 4000)
        self.refresh_table()

    def _export_csv(self):
        # Open export options dialog, then export according to user's choices
        if not self.model.rows:
            QtWidgets.QMessageBox.information(self, APP_NAME, "No rows to export.")
            return
        opts = self._get_export_options()
        if not opts:
            return  # user canceled

        # Build rows to export (apply optional path/extension filters and row cap)
        rows = list(self.model.rows)
        if opts.get("path_contains"):
            low = opts["path_contains"].lower()
            rows = [r for r in rows if low in (r[0] or "").lower()]  # r[0] = path
        if opts.get("exts"):
            exts = {e.lower() for e in opts["exts"]}
            rows = [r for r in rows if (r[2] or "").lower() in exts]  # r[2] = ext
        if opts.get("limit") and opts["limit"] > 0:
            rows = rows[: opts["limit"]]
        if not rows:
            QtWidgets.QMessageBox.information(self, APP_NAME, "No rows matched the export filters.")
            return

        suggested = str(Path.home() / "export.csv")
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export CSV", suggested, "CSV Files (*.csv)")
        if not fn:
            return

        # Map column names to indices in our row tuple
        all_cols = ["Name", "Extension", "Size (MB)", "Modified", "Tags", "Path", "Parent Folder", "Drive"]
        selected_cols = opts.get("columns") or all_cols
        idx_map = {
            "Path": 0,
            "Name": 1,
            "Extension": 2,
            "Size (MB)": 3,
            "Modified": 4,
            "Tags": 5,
        }

        try:
            import csv
            with open(fn, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if opts.get("include_header", True):
                    w.writerow(selected_cols)
                for row in rows:
                    path, name, ext, size, mtime, tags = row
                    # Normalize to export-ready
                    normalized = {
                        "Path": path,
                        "Name": name,
                        "Extension": ext,
                        "Size (MB)": f"{(size or 0)/1024/1024:.2f}",
                        "Modified": time.strftime('%Y-%m-%d %H:%M', time.localtime(mtime)) if mtime else "",
                        "Tags": tags or "",
                        "Parent Folder": self._parent_folder(path),
                        "Drive": self._drive_letter(path),
                    }
                    w.writerow([normalized[c] for c in selected_cols])
            self.status.showMessage(f"Exported {len(rows):,} rows → {fn}", 6000)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, APP_NAME, f"Export failed: {e}")

    def _parent_folder(self, path: str) -> str:
        try:
            return str(Path(path).parent.name)
        except Exception:
            return ""

    def _drive_letter(self, path: str) -> str:
        try:
            p = Path(path)
            if p.drive:
                return p.drive.rstrip('\\')
            if path.startswith('\\\\'):
                parts = path.split('\\')
                return f"\\\\{parts[2]}" if len(parts) > 2 else "\\\\"
            return ""
        except Exception:
            return ""

    def _get_export_options(self):
        # Dialog allowing user to pick columns, limit, ext filter, and path contains
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Export Options")
        dlg.resize(420, 420)
        layout = QtWidgets.QVBoxLayout(dlg)

        # Columns checklist
        grp_cols = QtWidgets.QGroupBox("Columns")
        vcols = QtWidgets.QVBoxLayout(grp_cols)
        all_cols = ["Name", "Extension", "Size (MB)", "Modified", "Tags", "Path", "Parent Folder", "Drive"]
        checks = []
        for col in all_cols:
            cb = QtWidgets.QCheckBox(col)
            cb.setChecked(True)
            vcols.addWidget(cb)
            checks.append(cb)
        layout.addWidget(grp_cols)

        # Filters
        form = QtWidgets.QFormLayout()
        path_edit = QtWidgets.QLineEdit()
        path_edit.setPlaceholderText("Substring match (optional)")
        form.addRow("Path contains:", path_edit)

        # Extension multi-select via simple checklist
        ext_box = QtWidgets.QGroupBox("Extensions")
        vext = QtWidgets.QHBoxLayout(ext_box)
        ext_checks = []
        for e in sorted(SUPPORTED_EXTS):
            cb = QtWidgets.QCheckBox(e)
            cb.setChecked(False)
            vext.addWidget(cb)
            ext_checks.append(cb)
        form.addRow(ext_box)

        # Limit rows
        limit_spin = QtWidgets.QSpinBox()
        limit_spin.setRange(0, 10_000_000)
        limit_spin.setValue(min(5000, len(self.model.rows)))
        limit_spin.setSpecialValueText("No limit")
        form.addRow("Max rows:", limit_spin)

        layout.addLayout(form)

        # Header toggle
        header_cb = QtWidgets.QCheckBox("Include header row")
        header_cb.setChecked(True)
        layout.addWidget(header_cb)

        # Buttons
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        layout.addWidget(btns)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)

        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None

        chosen_cols = [cb.text() for cb in checks if cb.isChecked()]
        chosen_exts = [cb.text() for cb in ext_checks if cb.isChecked()]

        return {
            "columns": chosen_cols or all_cols,
            "path_contains": path_edit.text().strip(),
            "exts": chosen_exts,  # empty means no extra filtering
            "limit": int(limit_spin.value()),
            "include_header": bool(header_cb.isChecked()),
        }

    # ---- Similarity helpers ----
    def _edit_tags_dialog(self, index: QtCore.QModelIndex):
        current = self.model.data(index, QtCore.Qt.ItemDataRole.DisplayRole) or ""
        new_tags, ok = QtWidgets.QInputDialog.getText(
            self, "Edit Tags", "Enter tags (comma-separated):",
            QtWidgets.QLineEdit.Normal, current
        )
        if ok:
            self.model.setData(index, new_tags, QtCore.Qt.ItemDataRole.EditRole)

    def _find_similar_from_index(self, index: QtCore.QModelIndex):
        if not _SIM_AVAILABLE:
            QtWidgets.QMessageBox.warning(self, APP_NAME, "Similarity features require numpy/faiss/joblib/scikit-learn. Install them first.")
            return
        # Build query text from the selected row
        row = self.model.rows[index.row()]
        path, name, ext, size, mtime, tags = row
        stem = Path(name).stem if name else ""
        # include two parent folder names for extra context
        pp = Path(path)
        parents = []
        if pp.parent.name: parents.append(pp.parent.name)
        if pp.parent.parent and pp.parent.parent.name: parents.append(pp.parent.parent.name)
        qtext = " ".join(x for x in [stem, ext.lstrip("."), tags, *parents] if x).strip()
        if not qtext:
            QtWidgets.QMessageBox.information(self, APP_NAME, "No text available for this item (empty name/tags). Add tags first.")
            return
        try:
            from pathlib import Path as _P
            ART_DIR = _P("db")
            VEC_PATH = ART_DIR / "tfidf_vectorizer.joblib"
            IDX_PATH = ART_DIR / "faiss_tfidf.index"
            IDS_PATH = ART_DIR / "faiss_ids.npy"
            if not (VEC_PATH.exists() and IDX_PATH.exists() and IDS_PATH.exists()):
                QtWidgets.QMessageBox.warning(self, APP_NAME, "FAISS artifacts not found. Run scripts\build_faiss.py first.")
                return
            vec = joblib.load(VEC_PATH)
            index_faiss = faiss.read_index(str(IDX_PATH))
            ids = np.load(IDS_PATH)
            q = vec.transform([qtext]).astype("float32")
            q = _sk_normalize(q, norm="l2", copy=False).toarray().astype("float32")
            D, I = index_faiss.search(q, 15)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, APP_NAME, f"Similarity search failed: {e}")
            return
        # Fetch rows by DB ids
        try:
            con = sqlite3.connect(DB_PATH)
            cur = con.cursor()
            results = []
            for rank, (idx, score) in enumerate(zip(I[0], D[0]), 1):
                if idx < 0: continue
                aid = int(ids[idx])
                cur.execute("SELECT path, name, ext, tags FROM files WHERE id=?", (aid,))
                r = cur.fetchone()
                if r:
                    rpath, rname, rext, rtags = r
                    # avoid name+ext duplication in display
                    display = rname if rname.lower().endswith((rext or '').lower()) else f"{rname}{rext}"
                    results.append((float(score), aid, display, rpath, rtags))
        finally:
            con.close()
        self._show_similar_results(results)

    def _on_single_click(self, index: QtCore.QModelIndex):
        # Single left-click: select row, update status and preview
        if not index.isValid():
            return
        self.table.selectRow(index.row())
        try:
            row = self.model.rows[index.row()]  # (path,name,ext,size,mtime,tags)
            name = row[1]; ext = row[2]; path = row[0]
            display = name if name.lower().endswith((ext or '').lower()) else f"{name}{ext}"
            self.status.showMessage(f"Selected: {display} — {path}")
            self._update_preview_for_row(row)
        except Exception:
            pass

    def _update_preview_for_row(self, row: tuple):
        # row = (path,name,ext,size,mtime,tags)
        path, name, ext, size, mtime, tags = row
        base = name if name.lower().endswith((ext or '').lower()) else f"{name}{ext}"
        # Placeholder: later replace with real image
        self.preview_img.setText("🧩 Preview coming soon")
        # Fetch geometry metadata if present
        try:
            con = sqlite3.connect(DB_PATH)
            cur = con.cursor()
            cur.execute("SELECT tris, dim_x, dim_y, dim_z, volume, watertight FROM files WHERE path=?", (path,))
            geo = cur.fetchone() or (None,)*6
            con.close()
        except Exception:
            geo = (None,)*6
        tris, dx, dy, dz, vol, wt = geo
        wt_txt = {1: "Yes", 0: "No", None: "?"}.get(wt, "?")
        self.preview_meta.setPlainText(
            f"Name: {base}\n"
            f"Ext:  {ext}\n"
            f"Size: {size/1024/1024:.2f} MB\n"
            f"Modified: {time.strftime('%Y-%m-%d %H:%M', time.localtime(mtime)) if mtime else ''}\n"
            f"Tags: {tags or ''}\n"
            f"Path: {path}\n\n"
            f"Triangles: {'' if tris is None else tris}\n"
            f"Bounds (mm): {'' if dx is None else f'{dx:.1f} × {dy:.1f} × {dz:.1f}'}\n"
            f"Volume (mm³): {'' if vol is None else f'{vol:.0f}'}\n"
            f"Watertight: {wt_txt}"
        )

    def _bulk_tag_dialog(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Bulk Tag by Folder Keywords")
        dlg.resize(560, 460)
        v = QtWidgets.QVBoxLayout(dlg)

        info = QtWidgets.QLabel(
            "Enter one rule per line as 'keyword => tag'.\n"
            "Matches are case-insensitive and run against the full file path."
        )
        info.setWordWrap(True)
        v.addWidget(info)

        edit = QtWidgets.QPlainTextEdit()
        edit.setPlainText("starwars => Star Wars\nmarvel => Marvel\ndc => DC\nlotr => Lord of the Rings")
        v.addWidget(edit)

        # Scope: current view vs entire database
        scope_box = QtWidgets.QGroupBox("Scope")
        hb = QtWidgets.QHBoxLayout(scope_box)
        rb_view = QtWidgets.QRadioButton("Current view only")
        rb_all = QtWidgets.QRadioButton("Entire database")
        rb_view.setChecked(True)
        hb.addWidget(rb_view); hb.addWidget(rb_all); hb.addStretch(1)
        v.addWidget(scope_box)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        v.addWidget(btns)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)

        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        # Parse rules
        rules = []
        for line in edit.toPlainText().splitlines():
            if "=>" in line:
                k, t = line.split("=>", 1)
                k = k.strip().lower(); t = t.strip()
                if k and t:
                    rules.append((k, t))
        if not rules:
            QtWidgets.QMessageBox.information(self, APP_NAME, "No valid rules provided.")
            return

        # Confirm if running across the whole DB
        if rb_all.isChecked():
            ret = QtWidgets.QMessageBox.question(
                self, APP_NAME,
                "This will scan the entire database and may take a while. Proceed?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            )
            if ret != QtWidgets.QMessageBox.StandardButton.Yes:
                return
            # Setup progress dialog
            progress = QtWidgets.QProgressDialog("Bulk tagging entire database…", "Cancel", 0, 0, self)
            progress.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
        else:
            progress = None

        updated = 0
        try:
            if rb_view.isChecked():
                # Work on current rows in memory for speed
                candidates = [(r[0], r[5] or "") for r in list(self.model.rows)]  # (path, tags)
            else:
                # Load all rows from DB (path, tags)
                con = sqlite3.connect(DB_PATH)
                cur = con.cursor()
                cur.execute("SELECT path, COALESCE(tags,'') FROM files")
                candidates = cur.fetchall()
                con.close()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, APP_NAME, f"Load failed: {e}")
            return

        for idx, (path, tags) in enumerate(candidates):
            if progress and progress.wasCanceled():
                break
            lowp = (path or "").lower()
            add = {t for k, t in rules if k in lowp}
            if not add:
                continue
            existing = {x.strip() for x in (tags or "").split(',') if x.strip()}
            new_tags = ", ".join(sorted(existing.union(add)))
            try:
                update_tags(path, new_tags)
                updated += 1
            except Exception:
                pass
            if progress and idx % 500 == 0:
                progress.setLabelText(f"Tagged {updated:,} items so far…")
                QtWidgets.QApplication.processEvents()

        self.status.showMessage(f"Bulk tagged {updated:,} items.", 7000)
        self.refresh_table()

        if updated:
            ret = QtWidgets.QMessageBox.question(
                self, APP_NAME,
                "Rebuild the similarity index now to include new tags?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            )
            if ret == QtWidgets.QMessageBox.StandardButton.Yes:
                self._rebuild_similarity_index()

    def _start_geometry_compute(self):
        if not _GEO_AVAILABLE:
            QtWidgets.QMessageBox.warning(self, APP_NAME, "Geometry features require: pip install trimesh numpy")
            return
        if getattr(self, 'geo_worker', None) and self.geo_worker.isRunning():
            QtWidgets.QMessageBox.information(self, APP_NAME, "Geometry compute is already running.")
            return
        self.geo_worker = GeometryWorker(limit=None)
        self.geo_worker.message.connect(lambda m: self.status.showMessage(m))
        self.geo_worker.progress.connect(lambda d,t: self.status.showMessage(f"Geometry: {d}/{t} processed"))
        self.geo_worker.finished_ok.connect(lambda n: self.status.showMessage(f"Geometry compute finished for {n:,} items", 6000))
        self.geo_worker.error.connect(lambda e: QtWidgets.QMessageBox.critical(self, APP_NAME, f"Geometry compute failed: {e}"))
        self.status.showMessage("Computing geometry in background…")
        self.geo_worker.start()

    def _show_similar_results(self, results: list[tuple]):
        if not results:
            QtWidgets.QMessageBox.information(self, APP_NAME, "No similar items found.")
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Find Similar — Results")
        dlg.resize(800, 500)
        layout = QtWidgets.QVBoxLayout(dlg)
        view = QtWidgets.QTreeWidget()
        view.setHeaderLabels(["Score", "ID", "Name", "Path", "Tags"])
        view.setColumnWidth(2, 240)
        for score, aid, name, path, tags in results:
            item = QtWidgets.QTreeWidgetItem([
                f"{score:.3f}", str(aid), name, path, tags or ""
            ])
            view.addTopLevelItem(item)
        view.itemDoubleClicked.connect(lambda it, col: self._open_path(it.text(3)))
        layout.addWidget(view)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)
        dlg.exec()

    # ---- FAISS rebuild UI wiring ----
    def _rebuild_similarity_index(self):
        if not _SIM_AVAILABLE:
            QtWidgets.QMessageBox.warning(self, APP_NAME,
                "Similarity deps missing. In PowerShell, run:\n\n"
                "pip install faiss-cpu scikit-learn numpy joblib")
            return
        self.btn_build_sim.setEnabled(False)
        self.status.showMessage("Rebuilding similarity index… (runs in background)")
        self.sim_builder = SimilarityBuilder(DB_PATH, Path("db"))
        self.sim_builder.message.connect(lambda m: self.status.showMessage(m))
        self.sim_builder.finished_ok.connect(self._on_sim_ok)
        self.sim_builder.error.connect(self._on_sim_err)
        self.sim_builder.start()

    def _on_sim_ok(self, n: int):
        self.status.showMessage(f"Similarity index rebuilt for {n:,} items.", 5000)
        self.btn_build_sim.setEnabled(True)

    def _on_sim_err(self, msg: str):
        QtWidgets.QMessageBox.critical(self, APP_NAME, f"Similarity rebuild failed: {msg}")
        self.btn_build_sim.setEnabled(True)

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()