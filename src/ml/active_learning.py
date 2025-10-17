# app/ml/active_learning.py
"""
Active Learning scaffold
- Records user corrections (already saved in DB via dataio/db.py helpers)
- Builds a small delta-dataset from corrections
- Retrains/updates the part classifier incrementally
- Safe: no asserts; graceful fallbacks

Assumes:
- geometry_features.extract_features(path) -> Dict[str, float] (22 features)
- part_classifier.save_model / load_model, and train_random_forest(X, y, **kw)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple
from pathlib import Path
import sqlite3
import json
import traceback

# Minimal deps (sklearn is already in your project for RandomForest)
try:
    from sklearn.ensemble import RandomForestClassifier
    import joblib
except Exception:
    RandomForestClassifier = None
    joblib = None

# --- Config ------------------------------------------------------------------

DEFAULT_MODEL_PATH = Path("models/part_classifier.pkl")
DEFAULT_DB_PATH    = Path("db/modelfinder.db")

# --- Data classes ------------------------------------------------------------

@dataclass
class Correction:
    id: int
    file_path: str
    old_name: str | None
    new_name: str
    project_number: str | None
    part_type: str | None
    laterality: str | None
    confidence: float | None
    corrected_utc: str | None

# --- Utilities ---------------------------------------------------------------

def _connect(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(str(db_path))

def _get_corrections(con: sqlite3.Connection, limit: int | None = None) -> List[Correction]:
    cur = con.cursor()
    q = """
      SELECT id, file_path, old_name, new_name, project_number, part_type, laterality, confidence, corrected_at
      FROM user_corrections
      ORDER BY id ASC
    """
    if limit:
        q += f" LIMIT {int(limit)}"
    rows = cur.execute(q).fetchall()
    out: List[Correction] = []
    for r in rows:
        out.append(Correction(*r))
    return out

def _pop_ids(con: sqlite3.Connection, ids: Iterable[int]) -> None:
    ids = list(ids)
    if not ids:
        return
    qmarks = ",".join("?" * len(ids))
    cur = con.cursor()
    cur.execute(f"DELETE FROM user_corrections WHERE id IN ({qmarks})", ids)
    con.commit()

# --- Feature extraction bridge ----------------------------------------------

def _extract_features_for_paths(paths: List[str]) -> Tuple[List[str], List[List[float]]]:
    """
    Delegates to your geometry features module.
    Returns (labels_placeholder, X_features) where labels_placeholder is same length as paths (not used here).
    """
    try:
        from .geometry_features import extract_geometric_features  # your function returning dict of floats
    except Exception:
        # Soft failure path — return empty to avoid crashes
        return [], []
    X: List[List[float]] = []
    for p in paths:
        try:
            feats: Dict[str, float] = extract_geometric_features(p)
            if feats:
                # keep column order stable by sorting keys
                vec = [feats[k] for k in sorted(feats.keys())]
                X.append(vec)
            else:
                X.append([])
        except Exception:
            X.append([])
    return ["" for _ in paths], X

def _labels_from_corrections(corrections: List[Correction]) -> List[str]:
    """
    Choose label granularity. Here we join laterality+part_type when laterality exists.
    Examples: 'left_foot', 'right_hand', 'head'
    """
    labels: List[str] = []
    for c in corrections:
        pt = (c.part_type or "").strip().lower()
        lat = (c.laterality or "").strip().lower()
        lab = f"{lat}_{pt}" if lat and pt else (pt or "part")
        labels.append(lab)
    return labels

# --- Model IO ----------------------------------------------------------------

def _load_model(model_path: Path):
    if joblib is None or not model_path.exists():
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None

def _save_model(model, model_path: Path) -> bool:
    if joblib is None or model is None:
        return False
    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        return True
    except Exception:
        return False

def _train_new_model(X: List[List[float]], y: List[str], n_estimators: int = 200, random_state: int = 42):
    if RandomForestClassifier is None:
        return None
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=None, n_jobs=-1, random_state=random_state
    )
    clf.fit(X, y)
    return clf

def _warm_update(existing_model, X: List[List[float]], y: List[str]):
    """
    RandomForest has no partial_fit; we can:
      - re-train from scratch using (existing + new) if you persist base data,
      - or (simple) re-train on *only* corrected samples as a quick update.
    Here we retrain on corrected samples for speed; you can upgrade to full retrain later.
    """
    return _train_new_model(X, y)

# --- Public API --------------------------------------------------------------

def retrain_from_corrections(
    db_path: Path = DEFAULT_DB_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    min_samples: int = 8,
    pop_after_train: bool = False
) -> Dict[str, object]:
    """
    Loads corrections → builds delta dataset → trains or updates the classifier → saves pkl.
    Returns a small report dict (safe to print in UI).
    """
    report = {"ok": False, "used": 0, "skipped": 0, "saved": False, "model_path": str(model_path)}
    try:
        con = _connect(db_path)
        corrections = _get_corrections(con)
        if not corrections:
            report["msg"] = "No corrections to learn from."
            return report

        # Build labels and features
        paths = [c.file_path for c in corrections]
        y = _labels_from_corrections(corrections)
        _, X = _extract_features_for_paths(paths)

        # Filter out failures (empty feature vectors)
        X2: List[List[float]] = []
        y2: List[str] = []
        ids2: List[int] = []
        for corr, xvec, lab in zip(corrections, X, y):
            if xvec and all(v is not None for v in xvec):
                X2.append(xvec); y2.append(lab); ids2.append(corr.id)
            else:
                report["skipped"] += 1

        report["used"] = len(X2)
        if len(X2) < min_samples:
            report["msg"] = f"Not enough samples to retrain (need ≥{min_samples}, got {len(X2)})."
            return report

        # Train / update model
        model = _load_model(model_path)
        new_model = _warm_update(model, X2, y2)
        saved = _save_model(new_model, model_path)
        report["saved"] = bool(saved)
        report["ok"] = bool(new_model and saved)
        if pop_after_train and report["ok"]:
            _pop_ids(con, ids2)

        report["msg"] = "Retrained from corrections." if report["ok"] else "Training failed to produce a model."
        return report
    except Exception as e:
        report["error"] = str(e)
        report["trace"] = traceback.format_exc(limit=2)
        return report


# --- Legacy compatibility wrapper -------------------------------------------

class ActiveLearningSystem:
    """Legacy compatibility wrapper for the new streamlined API."""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
    
    def log_user_correction(self, file_path: str, original_name: str, corrected_name: str,
                           project_number: str = None, part_type: str = None, 
                           laterality: str = None, confidence: float = None,
                           correction_type: str = "rename") -> bool:
        """Log a user correction for active learning."""
        try:
            con = sqlite3.connect(self.db_path)
            cur = con.cursor()
            
            cur.execute("""
                INSERT INTO user_corrections 
                (file_path, original_name, corrected_name, project_number, part_type, 
                 laterality, confidence, correction_type, corrected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(file_path),
                original_name,
                corrected_name,
                project_number,
                part_type,
                laterality,
                confidence,
                correction_type,
                sqlite3.datetime.datetime.utcnow().isoformat()
            ))
            
            con.commit()
            con.close()
            return True
            
        except Exception as e:
            print(f"Failed to log user correction: {e}")
            return False
    
    def retrain_from_corrections(self) -> Dict[str, Any]:
        """Retrain the model using user corrections."""
        result = retrain_from_corrections(self.db_path)
        return {
            'success': result.get('ok', False),
            'samples_used': result.get('used', 0),
            'error': result.get('error', result.get('msg', 'Unknown error'))
        }
