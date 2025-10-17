"""
Part Type Classifier - ML model for recognizing part types from geometry

Trains a RandomForestClassifier on geometric features to predict:
- Part type (foot, hand, head, torso, etc.)
- Laterality (left, right, center)

Uses training_samples table populated by archive_trainer.py
"""
import json
import sqlite3
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    import joblib
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")


class PartClassifier:
    """
    Machine learning classifier for part type recognition.
    
    Uses geometric features to classify parts:
    - Part type: head, torso, arm, hand, leg, foot, base, accessory, prop
    - Laterality: left, right, center
    """
    
    PART_TYPES = ['head', 'torso', 'arm', 'hand', 'leg', 'foot', 'base', 'accessory', 'prop', 'unknown']
    LATERALITY_TYPES = ['left', 'right', 'center']
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.part_model = None
        self.laterality_model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.training_stats = {}
        
        if not _SKLEARN_AVAILABLE:
            print("Warning: sklearn not available, classifier disabled")
    
    def load_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load training samples from database.
        
        Returns:
            X: Feature matrix (n_samples, n_features)
            y_part: Part type labels
            y_lat: Laterality labels
        """
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        
        cur.execute("""
            SELECT features_json, part_type, laterality 
            FROM training_samples 
            WHERE features_json IS NOT NULL 
              AND part_type IS NOT NULL
              AND part_type != 'unknown'
        """)
        
        X, y_part, y_lat = [], [], []
        
        for row in cur.fetchall():
            features_json, part_type, laterality = row
            
            if features_json:
                try:
                    features = json.loads(features_json)
                    
                    # Extract feature values in consistent order
                    if not self.feature_names:
                        self.feature_names = sorted(features.keys())
                    
                    feature_values = [features.get(k, 0.0) for k in self.feature_names]
                    
                    X.append(feature_values)
                    y_part.append(part_type or 'unknown')
                    y_lat.append(laterality or 'center')
                    
                except Exception as e:
                    print(f"Error loading features: {e}")
                    continue
        
        con.close()
        
        if len(X) == 0:
            raise ValueError("No training data found. Run 'Train from Archive' first.")
        
        return np.array(X), np.array(y_part), np.array(y_lat)
    
    def train(self, n_estimators: int = 100, max_depth: Optional[int] = 15) -> Dict:
        """
        Train both part type and laterality classifiers.
        
        Args:
            n_estimators: Number of trees in random forest
            max_depth: Maximum depth of trees (None = unlimited)
            
        Returns:
            Dictionary with training statistics
        """
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for training")
        
        # Load training data
        X, y_part, y_lat = self.load_training_data()
        
        if len(X) < 10:
            raise ValueError(f"Need at least 10 training samples, found {len(X)}")
        
        print(f"Training with {len(X)} samples...")
        print(f"Features: {len(self.feature_names)}")
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_part_train, y_part_test, y_lat_train, y_lat_test = train_test_split(
            X_scaled, y_part, y_lat, test_size=0.2, random_state=42, stratify=y_part
        )
        
        # Train part type classifier
        print("\nTraining part type classifier...")
        self.part_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'  # Handle imbalanced classes
        )
        self.part_model.fit(X_train, y_part_train)
        
        # Evaluate part type model
        part_train_score = self.part_model.score(X_train, y_part_train)
        part_test_score = self.part_model.score(X_test, y_part_test)
        
        # Cross-validation for part type
        part_cv_scores = cross_val_score(
            self.part_model, X_scaled, y_part, cv=5, scoring='accuracy'
        )
        
        # Train laterality classifier (if enough samples)
        lat_train_score = 0.0
        lat_test_score = 0.0
        lat_cv_scores = []
        
        if len(set(y_lat)) > 1:  # Only train if we have multiple classes
            print("Training laterality classifier...")
            self.laterality_model = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            self.laterality_model.fit(X_train, y_lat_train)
            
            lat_train_score = self.laterality_model.score(X_train, y_lat_train)
            lat_test_score = self.laterality_model.score(X_test, y_lat_test)
            lat_cv_scores = cross_val_score(
                self.laterality_model, X_scaled, y_lat, cv=5, scoring='accuracy'
            )
        
        # Feature importance
        feature_importance = dict(zip(
            self.feature_names,
            self.part_model.feature_importances_
        ))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Store statistics
        self.training_stats = {
            'n_samples': len(X),
            'n_features': len(self.feature_names),
            'part_type': {
                'train_accuracy': float(part_train_score),
                'test_accuracy': float(part_test_score),
                'cv_mean': float(part_cv_scores.mean()),
                'cv_std': float(part_cv_scores.std()),
                'classes': self.part_model.classes_.tolist(),
                'n_classes': len(self.part_model.classes_)
            },
            'laterality': {
                'train_accuracy': float(lat_train_score),
                'test_accuracy': float(lat_test_score),
                'cv_mean': float(lat_cv_scores.mean()) if len(lat_cv_scores) > 0 else 0.0,
                'cv_std': float(lat_cv_scores.std()) if len(lat_cv_scores) > 0 else 0.0,
                'classes': self.laterality_model.classes_.tolist() if self.laterality_model else [],
                'n_classes': len(self.laterality_model.classes_) if self.laterality_model else 0
            },
            'top_features': top_features
        }
        
        # Print summary
        print("\n" + "="*50)
        print("TRAINING COMPLETE")
        print("="*50)
        print(f"\nPart Type Classifier:")
        print(f"  Training accuracy: {part_train_score:.2%}")
        print(f"  Test accuracy: {part_test_score:.2%}")
        print(f"  Cross-validation: {part_cv_scores.mean():.2%} ± {part_cv_scores.std():.2%}")
        print(f"  Classes: {len(self.part_model.classes_)}")
        
        if self.laterality_model:
            print(f"\nLaterality Classifier:")
            print(f"  Training accuracy: {lat_train_score:.2%}")
            print(f"  Test accuracy: {lat_test_score:.2%}")
            print(f"  Cross-validation: {lat_cv_scores.mean():.2%} ± {lat_cv_scores.std():.2%}")
        
        print(f"\nTop 5 Important Features:")
        for feat, importance in top_features[:5]:
            print(f"  {feat:20s}: {importance:.3f}")
        
        return self.training_stats
    
    def predict(self, features: Dict[str, float]) -> Tuple[str, float, str, float]:
        """
        Predict part type and laterality from geometric features.
        
        Args:
            features: Dictionary of geometric features
            
        Returns:
            (part_type, part_confidence, laterality, lat_confidence)
        """
        if not _SKLEARN_AVAILABLE or self.part_model is None:
            return 'unknown', 0.0, 'center', 0.0
        
        try:
            # Extract features in correct order
            feature_values = [features.get(k, 0.0) for k in self.feature_names]
            features_array = np.array(feature_values).reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features_array)
            
            # Predict part type
            part_type = self.part_model.predict(features_scaled)[0]
            part_proba = self.part_model.predict_proba(features_scaled)[0]
            part_confidence = float(part_proba.max())
            
            # Predict laterality
            laterality = 'center'
            lat_confidence = 1.0
            
            if self.laterality_model:
                laterality = self.laterality_model.predict(features_scaled)[0]
                lat_proba = self.laterality_model.predict_proba(features_scaled)[0]
                lat_confidence = float(lat_proba.max())
            
            return part_type, part_confidence, laterality, lat_confidence
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return 'unknown', 0.0, 'center', 0.0
    
    def predict_with_details(self, features: Dict[str, float]) -> Dict:
        """
        Predict with detailed probability breakdown.
        
        Returns dictionary with top predictions and their probabilities.
        """
        if not _SKLEARN_AVAILABLE or self.part_model is None:
            return {'part_type': 'unknown', 'confidence': 0.0}
        
        try:
            feature_values = [features.get(k, 0.0) for k in self.feature_names]
            features_array = np.array(feature_values).reshape(1, -1)
            features_scaled = self.scaler.transform(features_array)
            
            # Get all probabilities for part type
            part_proba = self.part_model.predict_proba(features_scaled)[0]
            part_classes = self.part_model.classes_
            
            # Sort by probability
            sorted_indices = np.argsort(part_proba)[::-1]
            
            top_predictions = [
                {
                    'part_type': part_classes[i],
                    'probability': float(part_proba[i])
                }
                for i in sorted_indices[:5]  # Top 5
            ]
            
            result = {
                'part_type': part_classes[sorted_indices[0]],
                'confidence': float(part_proba[sorted_indices[0]]),
                'top_predictions': top_predictions
            }
            
            # Add laterality if available
            if self.laterality_model:
                lat_proba = self.laterality_model.predict_proba(features_scaled)[0]
                lat_classes = self.laterality_model.classes_
                lat_idx = np.argmax(lat_proba)
                
                result['laterality'] = lat_classes[lat_idx]
                result['laterality_confidence'] = float(lat_proba[lat_idx])
            
            return result
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return {'part_type': 'unknown', 'confidence': 0.0}
    
    def save(self, model_path: str = 'models/part_classifier.pkl'):
        """Save trained models to disk"""
        if not _SKLEARN_AVAILABLE:
            return False
        
        if self.part_model is None:
            raise ValueError("No model to save. Train first.")
        
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save models and metadata
        save_data = {
            'part_model': self.part_model,
            'laterality_model': self.laterality_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'training_stats': self.training_stats
        }
        
        joblib.dump(save_data, model_path)
        print(f"Model saved to: {model_path}")
        return True
    
    def load(self, model_path: str = 'models/part_classifier.pkl') -> bool:
        """Load trained models from disk"""
        if not _SKLEARN_AVAILABLE:
            return False
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            return False
        
        try:
            save_data = joblib.load(model_path)
            
            self.part_model = save_data['part_model']
            self.laterality_model = save_data.get('laterality_model')
            self.scaler = save_data['scaler']
            self.feature_names = save_data['feature_names']
            self.training_stats = save_data.get('training_stats', {})
            
            print(f"Model loaded from: {model_path}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        if self.part_model is None:
            return {'loaded': False}
        
        return {
            'loaded': True,
            'n_features': len(self.feature_names),
            'part_classes': self.part_model.classes_.tolist() if hasattr(self.part_model, 'classes_') else [],
            'laterality_classes': self.laterality_model.classes_.tolist() if self.laterality_model and hasattr(self.laterality_model, 'classes_') else [],
            'training_stats': self.training_stats
        }


if __name__ == "__main__":
    # Test the classifier
    import sys
    
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        db_path = "db/modelfinder.db"
    
    print(f"Using database: {db_path}")
    
    classifier = PartClassifier(db_path)
    
    # Try to load existing model
    if classifier.load():
        print("\nLoaded existing model")
        info = classifier.get_model_info()
        print(f"Part classes: {info['part_classes']}")
        print(f"Laterality classes: {info['laterality_classes']}")
    else:
        print("\nNo existing model found. Training new model...")
        
        try:
            stats = classifier.train()
            
            # Save the trained model
            classifier.save()
            
            print("\nModel trained and saved successfully!")
            
        except Exception as e:
            print(f"\nTraining failed: {e}")
            print("\nMake sure you have:")
            print("  1. Run 'Train from Archive' in the UI first")
            print("  2. Or use: python -m src.ml.archive_trainer <folder_path>")

