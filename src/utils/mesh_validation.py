"""
Mesh validation and quarantine system for ModelFinder Pro.

Provides comprehensive validation of 3D meshes and quarantine functionality
for malformed or problematic files.
"""
import hashlib
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

try:
    import trimesh
    _TRIMESH_AVAILABLE = True
except ImportError:
    _TRIMESH_AVAILABLE = False


class MeshValidator:
    """Validates 3D meshes and manages quarantine system."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.quarantine_reasons = {
            'empty_mesh': 'Mesh has no vertices or faces',
            'invalid_geometry': 'Mesh has invalid geometry (NaN, inf values)',
            'non_manifold': 'Mesh is non-manifold (holes, gaps)',
            'duplicate_vertices': 'Mesh has duplicate vertices',
            'zero_area_faces': 'Mesh has faces with zero area',
            'load_error': 'Failed to load mesh file',
            'unsupported_format': 'File format not supported',
            'corrupted_file': 'File appears to be corrupted',
            'missing_file': 'File does not exist',
            'permission_error': 'Insufficient permissions to read file'
        }
    
    def validate_mesh(self, file_path: str) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Validate a 3D mesh file.
        
        Args:
            file_path: Path to the mesh file
            
        Returns:
            Tuple of (is_valid, error_reason, validation_details)
        """
        if not _TRIMESH_AVAILABLE:
            return False, 'trimesh_not_available', {'error': 'trimesh library not available'}
        
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            return False, 'missing_file', {'error': 'File does not exist'}
        
        # Check file size
        try:
            file_size = file_path.stat().st_size
            if file_size == 0:
                return False, 'empty_file', {'error': 'File is empty', 'size': 0}
        except PermissionError:
            return False, 'permission_error', {'error': 'Permission denied'}
        
        # Try to load the mesh
        try:
            mesh = trimesh.load(str(file_path))
            
            # Check if mesh loaded successfully
            if mesh is None:
                return False, 'load_error', {'error': 'Failed to load mesh', 'size': file_size}
            
            # Validate mesh properties
            validation_details = {
                'vertices': len(mesh.vertices) if hasattr(mesh, 'vertices') else 0,
                'faces': len(mesh.faces) if hasattr(mesh, 'faces') else 0,
                'volume': float(mesh.volume) if hasattr(mesh, 'volume') else 0.0,
                'bounds': mesh.bounds.tolist() if hasattr(mesh, 'bounds') else None,
                'file_size': file_size
            }
            
            # Check for empty mesh
            if validation_details['vertices'] == 0 or validation_details['faces'] == 0:
                return False, 'empty_mesh', validation_details
            
            # Check for invalid geometry
            if hasattr(mesh, 'vertices'):
                vertices = mesh.vertices
                if (vertices != vertices).any() or (vertices == float('inf')).any():
                    return False, 'invalid_geometry', validation_details
            
            # Check for duplicate vertices
            if hasattr(mesh, 'vertices') and len(mesh.vertices) > 0:
                unique_vertices = len(set(tuple(v) for v in mesh.vertices))
                if unique_vertices < len(mesh.vertices) * 0.8:  # More than 20% duplicates
                    return False, 'duplicate_vertices', validation_details
            
            # Check for zero area faces
            if hasattr(mesh, 'faces') and len(mesh.faces) > 0:
                face_areas = mesh.face_areas
                zero_area_count = sum(1 for area in face_areas if area == 0)
                if zero_area_count > len(face_areas) * 0.1:  # More than 10% zero area
                    return False, 'zero_area_faces', validation_details
            
            # Check if mesh is watertight (optional warning, not quarantine)
            if hasattr(mesh, 'is_watertight'):
                validation_details['is_watertight'] = mesh.is_watertight
            
            # Check if mesh is manifold (optional warning, not quarantine)
            if hasattr(mesh, 'is_winding_consistent'):
                validation_details['is_winding_consistent'] = mesh.is_winding_consistent
            
            return True, None, validation_details
            
        except Exception as e:
            return False, 'load_error', {
                'error': str(e),
                'size': file_size,
                'exception_type': type(e).__name__
            }
    
    def quarantine_mesh(self, file_path: str, reason: str, error_details: str = "") -> bool:
        """
        Quarantine a malformed mesh.
        
        Args:
            file_path: Path to the mesh file
            reason: Reason for quarantine (must be in quarantine_reasons)
            error_details: Additional error details
            
        Returns:
            True if successfully quarantined
        """
        if reason not in self.quarantine_reasons:
            raise ValueError(f"Invalid quarantine reason: {reason}")
        
        try:
            file_path = Path(file_path)
            file_size = file_path.stat().st_size if file_path.exists() else 0
            sha256_hash = self._calculate_sha256(file_path) if file_path.exists() else None
            
            con = sqlite3.connect(self.db_path)
            cur = con.cursor()
            
            cur.execute("""
                INSERT OR REPLACE INTO quarantined_meshes 
                (file_path, quarantine_reason, error_details, quarantined_at, file_size, sha256)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                str(file_path),
                reason,
                error_details,
                datetime.utcnow().isoformat(),
                file_size,
                sha256_hash
            ))
            
            con.commit()
            con.close()
            return True
            
        except Exception as e:
            print(f"Failed to quarantine mesh {file_path}: {e}")
            return False
    
    def is_quarantined(self, file_path: str) -> bool:
        """Check if a file is quarantined."""
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        
        cur.execute("""
            SELECT 1 FROM quarantined_meshes 
            WHERE file_path = ? AND resolved = 0
        """, (str(file_path),))
        
        result = cur.fetchone() is not None
        con.close()
        return result
    
    def get_quarantined_files(self) -> List[Dict[str, Any]]:
        """Get all quarantined files."""
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        
        cur.execute("""
            SELECT file_path, quarantine_reason, error_details, quarantined_at, 
                   file_size, resolution_attempts, resolved
            FROM quarantined_meshes 
            WHERE resolved = 0
            ORDER BY quarantined_at DESC
        """)
        
        results = []
        for row in cur.fetchall():
            results.append({
                'file_path': row[0],
                'quarantine_reason': row[1],
                'error_details': row[2],
                'quarantined_at': row[3],
                'file_size': row[4],
                'resolution_attempts': row[5],
                'resolved': bool(row[6])
            })
        
        con.close()
        return results
    
    def resolve_quarantine(self, file_path: str) -> bool:
        """Mark a quarantined file as resolved."""
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        
        cur.execute("""
            UPDATE quarantined_meshes 
            SET resolved = 1 
            WHERE file_path = ?
        """, (str(file_path),))
        
        affected = cur.rowcount
        con.commit()
        con.close()
        return affected > 0
    
    def _calculate_sha256(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception:
            return ""


class FileIntegrityChecker:
    """Handles file integrity verification for migrations."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            print(f"Failed to calculate hash for {file_path}: {e}")
            return ""
    
    def verify_file_integrity(self, file_path: str, expected_hash: str = None, 
                            expected_size: int = None) -> Tuple[bool, str]:
        """
        Verify file integrity using hash and/or size.
        
        Args:
            file_path: Path to the file
            expected_hash: Expected SHA256 hash (optional)
            expected_size: Expected file size in bytes (optional)
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not Path(file_path).exists():
            return False, "File does not exist"
        
        # Check file size if provided
        if expected_size is not None:
            actual_size = Path(file_path).stat().st_size
            if actual_size != expected_size:
                return False, f"Size mismatch: expected {expected_size}, got {actual_size}"
        
        # Check hash if provided
        if expected_hash:
            actual_hash = self.calculate_file_hash(file_path)
            if actual_hash != expected_hash:
                return False, f"Hash mismatch: expected {expected_hash[:16]}..., got {actual_hash[:16]}..."
        
        return True, ""
    
    def log_operation(self, operation: str, source_path: str, dest_path: str = None,
                     status: str = "success", details: str = "", 
                     sha256_before: str = None, sha256_after: str = None,
                     file_size_before: int = None, file_size_after: int = None,
                     version_bump: int = 0, user_confirmed: bool = False):
        """Log an operation to the operations_log table."""
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        
        cur.execute("""
            INSERT INTO operations_log 
            (timestamp, operation, source_path, dest_path, status, details,
             sha256_before, sha256_after, file_size_before, file_size_after,
             version_bump, user_confirmed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            operation,
            source_path,
            dest_path,
            status,
            details,
            sha256_before,
            sha256_after,
            file_size_before,
            file_size_after,
            version_bump,
            1 if user_confirmed else 0
        ))
        
        con.commit()
        con.close()
    
    def get_rollback_operations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent operations for rollback."""
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        
        cur.execute("""
            SELECT id, timestamp, operation, source_path, dest_path, status,
                   sha256_before, sha256_after, file_size_before, file_size_after
            FROM operations_log 
            WHERE status = 'success' AND operation IN ('move', 'copy', 'rename')
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (limit,))
        
        results = []
        for row in cur.fetchall():
            results.append({
                'id': row[0],
                'timestamp': row[1],
                'operation': row[2],
                'source_path': row[3],
                'dest_path': row[4],
                'status': row[5],
                'sha256_before': row[6],
                'sha256_after': row[7],
                'file_size_before': row[8],
                'file_size_after': row[9]
            })
        
        con.close()
        return results
