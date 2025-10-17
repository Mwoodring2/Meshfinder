"""
Archive Trainer - Learn from well-organized folders

Scans user's existing well-organized 3D file archive to:
1. Extract geometric features from meshes
2. Learn naming patterns (project + character + part)
3. Build training dataset for ML classifier
4. Train model to recognize part types from geometry + context
"""
import re
import sqlite3
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json

from src.ml.geometry_features import extract_geometric_features, detect_laterality
from src.utils.naming import extract_project_number


# Common part type patterns
PART_PATTERNS = {
    'head': r'\b(head|skull|face|cranium)\b',
    'torso': r'\b(torso|chest|body|trunk)\b',
    'arm': r'\b(arm|bicep|forearm|upper_arm)\b',
    'hand': r'\b(hand|palm|fist|fingers?)\b',
    'leg': r'\b(leg|thigh|calf|upper_leg|lower_leg)\b',
    'foot': r'\b(foot|feet|boot|shoe)\b',
    'base': r'\b(base|stand|platform|pedestal)\b',
    'accessory': r'\b(accessory|attach|addon|extra)\b',
    'prop': r'\b(prop|item|object|weapon|tool)\b',
}

LATERALITY_PATTERNS = {
    'left': r'\b(left|l_|_l\b)',
    'right': r'\b(right|r_|_r\b)',
}


class ArchiveTrainer:
    """Train ML model from well-organized archive folders"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.training_samples = []
    
    def scan_training_folders(self, folder_paths: List[str], 
                             progress_callback=None) -> Dict[str, int]:
        """
        Scan folders to build training dataset.
        
        Args:
            folder_paths: List of well-organized folder paths to scan
            progress_callback: Optional callback(current, total, message)
            
        Returns:
            Statistics dict with counts
        """
        stats = {
            'total_files': 0,
            'features_extracted': 0,
            'patterns_matched': 0,
            'errors': 0
        }
        
        # Collect all 3D files
        all_files = []
        for folder in folder_paths:
            folder_path = Path(folder)
            if not folder_path.exists():
                continue
            
            # Find all 3D files
            for ext in ['.stl', '.obj', '.fbx', '.glb', '.gltf']:
                all_files.extend(folder_path.rglob(f'*{ext}'))
        
        stats['total_files'] = len(all_files)
        
        # Process each file
        for idx, file_path in enumerate(all_files):
            if progress_callback:
                progress_callback(idx + 1, len(all_files), f"Processing {file_path.name}")
            
            try:
                sample = self._process_training_file(file_path)
                
                if sample:
                    self.training_samples.append(sample)
                    stats['features_extracted'] += 1
                    
                    if sample['part_type'] != 'unknown':
                        stats['patterns_matched'] += 1
                        
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                stats['errors'] += 1
        
        return stats
    
    def _process_training_file(self, file_path: Path) -> Optional[Dict]:
        """
        Process a single file to create a training sample.
        
        Extracts:
        - Geometric features
        - Part type from filename
        - Project/character from folder name
        - Laterality
        - Context
        """
        # Extract geometric features
        features = extract_geometric_features(str(file_path))
        
        if not features:
            return None
        
        # Parse filename for part type
        filename_lower = file_path.stem.lower()
        
        part_type = 'unknown'
        for part, pattern in PART_PATTERNS.items():
            if re.search(pattern, filename_lower, re.IGNORECASE):
                part_type = part
                break
        
        # Detect laterality
        laterality = 'center'
        for side, pattern in LATERALITY_PATTERNS.items():
            if re.search(pattern, filename_lower, re.IGNORECASE):
                laterality = side
                break
        
        # Also try geometric laterality detection
        if laterality == 'center':
            geo_laterality = detect_laterality(str(file_path))
            if geo_laterality:
                laterality = geo_laterality
        
        # Extract context from folder structure
        folder_name = file_path.parent.name
        project_number = extract_project_number(folder_name)
        
        # Try to extract character name from folder
        character_name = self._extract_character_name(folder_name)
        
        # License type from folder hierarchy
        license_type = self._infer_license_from_path(file_path)
        
        sample = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'features': features,
            'part_type': part_type,
            'laterality': laterality,
            'project_number': project_number or 'unknown',
            'character_name': character_name or 'unknown',
            'license_type': license_type,
            'folder_context': folder_name,
            'source': 'archive_scan'
        }
        
        return sample
    
    def _extract_character_name(self, folder_name: str) -> Optional[str]:
        """
        Extract character/project name from folder.
        
        Examples:
        "300668_Yoda_PF" -> "Yoda"
        "Superman_Base" -> "Superman"
        "300915_ManBat" -> "ManBat"
        """
        # Remove project number
        name = re.sub(r'^\d+[_-]?', '', folder_name)
        
        # Remove common suffixes
        name = re.sub(r'[_-]?(PF|STD|EX|parts?)$', '', name, flags=re.IGNORECASE)
        
        # Clean up
        name = name.replace('_', ' ').strip()
        
        return name if name else None
    
    def _infer_license_from_path(self, file_path: Path) -> str:
        """
        Infer license type from folder hierarchy.
        
        Look for keywords like "commercial", "personal", "client", "fanart"
        """
        path_str = str(file_path).lower()
        
        if 'commercial' in path_str or 'client' in path_str:
            return 'Commercial'
        elif 'personal' in path_str or 'hobby' in path_str:
            return 'Personal'
        elif 'fanart' in path_str or 'fan' in path_str:
            return 'Fan-Art'
        elif 'stock' in path_str:
            return 'Stock'
        else:
            return 'Unknown'
    
    def save_training_data(self):
        """Save training samples to database"""
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        
        # Create training_samples table if not exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS training_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT,
                file_name TEXT,
                project_number TEXT,
                character_name TEXT,
                part_type TEXT,
                laterality TEXT,
                license_type TEXT,
                folder_context TEXT,
                features_json TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                source TEXT
            )
        """)
        
        # Insert samples
        for sample in self.training_samples:
            cur.execute("""
                INSERT INTO training_samples 
                (file_path, file_name, project_number, character_name, 
                 part_type, laterality, license_type, folder_context,
                 features_json, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sample['file_path'],
                sample['file_name'],
                sample['project_number'],
                sample['character_name'],
                sample['part_type'],
                sample['laterality'],
                sample['license_type'],
                sample['folder_context'],
                json.dumps(sample['features']),
                sample['source']
            ))
        
        con.commit()
        con.close()
        
        print(f"Saved {len(self.training_samples)} training samples to database")
    
    def load_training_data(self) -> List[Dict]:
        """Load training samples from database"""
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        
        try:
            cur.execute("""
                SELECT file_path, file_name, project_number, character_name,
                       part_type, laterality, license_type, folder_context,
                       features_json, source
                FROM training_samples
            """)
            
            samples = []
            for row in cur.fetchall():
                sample = {
                    'file_path': row[0],
                    'file_name': row[1],
                    'project_number': row[2],
                    'character_name': row[3],
                    'part_type': row[4],
                    'laterality': row[5],
                    'license_type': row[6],
                    'folder_context': row[7],
                    'features': json.loads(row[8]),
                    'source': row[9]
                }
                samples.append(sample)
            
            con.close()
            return samples
            
        except sqlite3.OperationalError:
            # Table doesn't exist yet
            con.close()
            return []
    
    def get_training_statistics(self) -> Dict:
        """Get statistics about training data"""
        samples = self.load_training_data()
        
        if not samples:
            return {
                'total_samples': 0,
                'part_types': {},
                'projects': set(),
                'laterality': {}
            }
        
        # Count part types
        part_counts = {}
        laterality_counts = {}
        projects = set()
        
        for sample in samples:
            part_type = sample['part_type']
            part_counts[part_type] = part_counts.get(part_type, 0) + 1
            
            laterality = sample['laterality']
            laterality_counts[laterality] = laterality_counts.get(laterality, 0) + 1
            
            if sample['project_number'] != 'unknown':
                projects.add(sample['project_number'])
        
        return {
            'total_samples': len(samples),
            'part_types': part_counts,
            'projects': projects,
            'laterality': laterality_counts
        }


if __name__ == "__main__":
    # Test the trainer
    import sys
    
    if len(sys.argv) > 1:
        test_folder = sys.argv[1]
        print(f"Training from folder: {test_folder}")
        
        trainer = ArchiveTrainer("db/modelfinder.db")
        
        stats = trainer.scan_training_folders([test_folder])
        
        print("\nTraining Statistics:")
        print(f"  Total files: {stats['total_files']}")
        print(f"  Features extracted: {stats['features_extracted']}")
        print(f"  Patterns matched: {stats['patterns_matched']}")
        print(f"  Errors: {stats['errors']}")
        
        # Save to database
        trainer.save_training_data()
        
        # Show training data stats
        training_stats = trainer.get_training_statistics()
        print(f"\nTraining dataset:")
        print(f"  Total samples: {training_stats['total_samples']}")
        print(f"  Part types: {training_stats['part_types']}")
        print(f"  Projects: {len(training_stats['projects'])}")
    else:
        print("Usage: python -m src.ml.archive_trainer <folder_path>")

