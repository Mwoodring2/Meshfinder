# src/ml/active_learning.py
"""
Active Learning Scaffold for ModelFinder Pro
Handles corrections logging and retraining from user feedback.
"""

import os
import sqlite3
import joblib
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def retrain_from_corrections(
    db_path: str = "db/modelfinder.db",
    model_path: str = "models/part_classifier.pkl",
    pop_after_train: bool = True,
    min_samples: int = 5
) -> Dict[str, Any]:
    """
    Retrain the part classifier from user corrections.
    
    Args:
        db_path: Path to SQLite database
        model_path: Path to save/load the trained model
        pop_after_train: If True, mark corrections as used after training
        min_samples: Minimum samples required for training
        
    Returns:
        Dictionary with training results and statistics
    """
    try:
        # Load corrections from database
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        
        # Get unused corrections
        corrections = cur.execute("""
            SELECT file_path, old_name, new_name, project_number, part_type, laterality, confidence
            FROM user_corrections 
            WHERE used_for_training = 0
            ORDER BY corrected_utc ASC
        """).fetchall()
        
        if len(corrections) < min_samples:
            con.close()
            return {
                "success": False,
                "msg": f"Not enough corrections for training (need {min_samples}, have {len(corrections)})",
                "used": 0,
                "skipped": len(corrections)
            }
        
        # Extract features for each correction
        training_data = []
        labels_part_type = []
        labels_laterality = []
        
        for correction in corrections:
            file_path, old_name, new_name, project_number, part_type, laterality, confidence = correction
            
            try:
                # Try to extract geometric features
                from .geometry_features import extract_geometric_features
                features = extract_geometric_features(Path(file_path))
                
                if features and len(features) > 0:
                    training_data.append(list(features.values()))
                    labels_part_type.append(part_type or "unknown")
                    labels_laterality.append(laterality or "center")
                else:
                    print(f"⚠️ Could not extract features from {file_path}")
                    
            except Exception as e:
                print(f"⚠️ Error processing {file_path}: {e}")
                continue
        
        if len(training_data) < min_samples:
            con.close()
            return {
                "success": False,
                "msg": f"Not enough valid features extracted (need {min_samples}, have {len(training_data)})",
                "used": 0,
                "skipped": len(corrections)
            }
        
        # Convert to numpy arrays
        X = np.array(training_data)
        y_part_type = np.array(labels_part_type)
        y_laterality = np.array(labels_laterality)
        
        # Create label encoders
        le_part_type = LabelEncoder()
        le_laterality = LabelEncoder()
        
        y_part_type_encoded = le_part_type.fit_transform(y_part_type)
        y_laterality_encoded = le_laterality.fit_transform(y_laterality)
        
        # Train classifiers
        clf_part_type = RandomForestClassifier(n_estimators=50, random_state=42)
        clf_laterality = RandomForestClassifier(n_estimators=50, random_state=42)
        
        clf_part_type.fit(X, y_part_type_encoded)
        clf_laterality.fit(X, y_laterality_encoded)
        
        # Create combined model
        model_data = {
            'part_type_classifier': clf_part_type,
            'laterality_classifier': clf_laterality,
            'part_type_encoder': le_part_type,
            'laterality_encoder': le_laterality,
            'feature_names': list(features.keys()) if features else []
        }
        
        # Ensure models directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        joblib.dump(model_data, model_path)
        
        # Mark corrections as used if requested
        if pop_after_train:
            correction_ids = [c[0] for c in corrections[:len(training_data)]]
            if correction_ids:
                placeholders = ','.join('?' * len(correction_ids))
                cur.execute(f"""
                    UPDATE user_corrections 
                    SET used_for_training = 1 
                    WHERE file_path IN ({placeholders})
                """, correction_ids)
                con.commit()
        
        con.close()
        
        return {
            "success": True,
            "msg": f"Successfully retrained from {len(training_data)} corrections",
            "used": len(training_data),
            "skipped": len(corrections) - len(training_data),
            "part_types": list(le_part_type.classes_),
            "laterality_types": list(le_laterality.classes_)
        }
        
    except Exception as e:
        return {
            "success": False,
            "msg": f"Training failed: {str(e)}",
            "used": 0,
            "skipped": len(corrections) if 'corrections' in locals() else 0
        }

def get_correction_stats(db_path: str = "db/modelfinder.db") -> Dict[str, Any]:
    """Get statistics about user corrections."""
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        
        # Total corrections
        total = cur.execute("SELECT COUNT(*) FROM user_corrections").fetchone()[0]
        
        # Used corrections
        used = cur.execute("SELECT COUNT(*) FROM user_corrections WHERE used_for_training = 1").fetchone()[0]
        
        # Unused corrections
        unused = total - used
        
        # Recent corrections (last 7 days)
        recent = cur.execute("""
            SELECT COUNT(*) FROM user_corrections 
            WHERE corrected_utc > datetime('now', '-7 days')
        """).fetchone()[0]
        
        con.close()
        
        return {
            "total": total,
            "used": used,
            "unused": unused,
            "recent": recent
        }
        
    except Exception as e:
        return {
            "total": 0,
            "used": 0,
            "unused": 0,
            "recent": 0,
            "error": str(e)
        }

def clear_old_corrections(db_path: str = "db/modelfinder.db", days_old: int = 30) -> int:
    """Clear corrections older than specified days."""
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        
        cur.execute("""
            DELETE FROM user_corrections 
            WHERE corrected_utc < datetime('now', '-{} days')
        """.format(days_old))
        
        deleted = cur.rowcount
        con.commit()
        con.close()
        
        return deleted
        
    except Exception as e:
        print(f"Error clearing old corrections: {e}")
        return 0