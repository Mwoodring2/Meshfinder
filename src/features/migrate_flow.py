"""
Enhanced migration flow with comprehensive guardrails for ModelFinder Pro.

Implements:
- Never overwrite protection with version bumping
- Hash verification for file integrity
- Malformed mesh quarantine
- Comprehensive operation logging
- Rollback capability
"""
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from ..utils.mesh_validation import MeshValidator, FileIntegrityChecker
from ..ml.active_learning import ActiveLearningSystem


class MigrationGuardrails:
    """Comprehensive guardrails for file migration operations."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.mesh_validator = MeshValidator(db_path)
        self.integrity_checker = FileIntegrityChecker(db_path)
        self.active_learning = ActiveLearningSystem(db_path)
    
    def plan_migration(self, source_files: List[str], dest_root: str, 
                      project_number: str, license_type: str = "unknown",
                      asset_category: str = "unknown") -> Dict[str, Any]:
        """
        Plan a migration with comprehensive conflict detection and validation.
        
        Args:
            source_files: List of source file paths
            dest_root: Destination root directory
            project_number: Project number for organization
            license_type: License type for folder structure
            asset_category: Asset category for folder structure
            
        Returns:
            Dictionary with migration plan and conflicts
        """
        plan = []
        conflicts = []
        quarantined = []
        
        for source_file in source_files:
            source_path = Path(source_file)
            
            # Skip if file doesn't exist
            if not source_path.exists():
                conflicts.append({
                    'file': source_file,
                    'type': 'missing_file',
                    'message': 'Source file does not exist'
                })
                continue
            
            # Check if file is quarantined
            if self.mesh_validator.is_quarantined(source_file):
                quarantined.append({
                    'file': source_file,
                    'reason': 'File is quarantined due to malformed mesh'
                })
                continue
            
            # Validate mesh if it's a 3D file
            if source_path.suffix.lower() in ['.stl', '.obj', '.ply', '.off']:
                is_valid, error_reason, validation_details = self.mesh_validator.validate_mesh(source_file)
                if not is_valid:
                    # Quarantine the malformed mesh
                    self.mesh_validator.quarantine_mesh(
                        source_file, 
                        error_reason, 
                        str(validation_details)
                    )
                    quarantined.append({
                        'file': source_file,
                        'reason': f'Malformed mesh: {error_reason}',
                        'details': validation_details
                    })
                    continue
            
            # Generate destination path
            dest_path = self._generate_dest_path(
                source_path, dest_root, project_number, license_type, asset_category
            )
            
            # Check for conflicts
            conflict_type, conflict_message = self._check_dest_conflict(source_path, dest_path)
            
            if conflict_type:
                conflicts.append({
                    'file': source_file,
                    'dest': str(dest_path),
                    'type': conflict_type,
                    'message': conflict_message
                })
            
            # Add to plan
            plan.append({
                'source': source_file,
                'dest': str(dest_path),
                'status': 'Ready' if not conflict_type else 'Conflict',
                'conflict_type': conflict_type,
                'file_size': source_path.stat().st_size,
                'sha256': self.integrity_checker.calculate_file_hash(source_file)
            })
        
        return {
            'plan': plan,
            'conflicts': conflicts,
            'quarantined': quarantined,
            'ready_count': len([p for p in plan if p['status'] == 'Ready']),
            'conflict_count': len(conflicts),
            'quarantined_count': len(quarantined)
        }
    
    def execute_migration(self, migration_plan: List[Dict[str, Any]], 
                         dry_run: bool = True) -> Dict[str, Any]:
        """
        Execute migration with comprehensive safety checks.
        
        Args:
            migration_plan: List of migration operations
            dry_run: If True, only simulate the migration
            
        Returns:
            Dictionary with execution results
        """
        results = {
            'success': True,
            'migrated': 0,
            'failed': 0,
            'skipped': 0,
            'errors': [],
            'operations': []
        }
        
        for operation in migration_plan:
            if operation['status'] != 'Ready':
                results['skipped'] += 1
                continue
            
            source_path = Path(operation['source'])
            dest_path = Path(operation['dest'])
            
            try:
                if dry_run:
                    # Simulate the operation
                    results['operations'].append({
                        'operation': 'simulate_move',
                        'source': str(source_path),
                        'dest': str(dest_path),
                        'status': 'simulated'
                    })
                    results['migrated'] += 1
                else:
                    # Execute the actual migration
                    success, error_msg = self._execute_single_migration(operation)
                    
                    if success:
                        results['migrated'] += 1
                    else:
                        results['failed'] += 1
                        results['errors'].append({
                            'file': str(source_path),
                            'error': error_msg
                        })
                    
                    results['operations'].append({
                        'operation': 'move',
                        'source': str(source_path),
                        'dest': str(dest_path),
                        'status': 'success' if success else 'failed',
                        'error': error_msg if not success else None
                    })
            
            except Exception as e:
                results['failed'] += 1
                results['errors'].append({
                    'file': str(source_path),
                    'error': str(e)
                })
                results['operations'].append({
                    'operation': 'move',
                    'source': str(source_path),
                    'dest': str(dest_path),
                    'status': 'failed',
                    'error': str(e)
                })
        
        results['success'] = results['failed'] == 0
        return results
    
    def _execute_single_migration(self, operation: Dict[str, Any]) -> Tuple[bool, str]:
        """Execute a single migration operation with full safety checks."""
        source_path = Path(operation['source'])
        dest_path = Path(operation['dest'])
        
        # Calculate hash before move
        sha256_before = self.integrity_checker.calculate_file_hash(str(source_path))
        file_size_before = source_path.stat().st_size
        
        # Create destination directory
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Handle destination conflicts with version bumping
        final_dest_path = self._resolve_dest_conflict(dest_path)
        
        # Perform the move
        try:
            shutil.move(str(source_path), str(final_dest_path))
            
            # Verify the move was successful
            if not final_dest_path.exists():
                return False, "File not found at destination after move"
            
            # Calculate hash after move
            sha256_after = self.integrity_checker.calculate_file_hash(str(final_dest_path))
            file_size_after = final_dest_path.stat().st_size
            
            # Verify integrity
            if sha256_before != sha256_after:
                # Try to restore original file
                if source_path.parent.exists():
                    shutil.move(str(final_dest_path), str(source_path))
                return False, "File integrity check failed - hashes don't match"
            
            if file_size_before != file_size_after:
                return False, "File size mismatch after move"
            
            # Log the successful operation
            version_bump = 1 if final_dest_path != dest_path else 0
            self.integrity_checker.log_operation(
                operation='move',
                source_path=str(source_path),
                dest_path=str(final_dest_path),
                status='success',
                details=f'Migrated to organized structure',
                sha256_before=sha256_before,
                sha256_after=sha256_after,
                file_size_before=file_size_before,
                file_size_after=file_size_after,
                version_bump=version_bump,
                user_confirmed=True
            )
            
            # Update database
            self._update_file_record(str(source_path), str(final_dest_path))
            
            return True, ""
            
        except Exception as e:
            return False, str(e)
    
    def _resolve_dest_conflict(self, dest_path: Path) -> Path:
        """Resolve destination conflicts with version bumping."""
        if not dest_path.exists():
            return dest_path
        
        # Try version bumping
        base_path = dest_path.parent / dest_path.stem
        extension = dest_path.suffix
        
        version = 1
        while True:
            versioned_path = Path(f"{base_path}_v{version}{extension}")
            if not versioned_path.exists():
                return versioned_path
            version += 1
            
            # Prevent infinite loop
            if version > 999:
                raise Exception(f"Too many version conflicts for {dest_path}")
    
    def _check_dest_conflict(self, source_path: Path, dest_path: Path) -> Tuple[Optional[str], str]:
        """Check for destination conflicts."""
        if not dest_path.exists():
            return None, ""
        
        # Check if it's the same file
        if source_path.resolve() == dest_path.resolve():
            return "same_file", "Source and destination are the same file"
        
        # Check if destination exists
        if dest_path.exists():
            return "file_exists", f"Destination file already exists: {dest_path.name}"
        
        return None, ""
    
    def _generate_dest_path(self, source_path: Path, dest_root: str, 
                           project_number: str, license_type: str, 
                           asset_category: str) -> Path:
        """Generate destination path following the organized structure."""
        # Extract filename components
        filename = source_path.name
        extension = source_path.suffix
        
        # Create organized path: <dest_root>/<project>/<license>/<category>/<filename>
        dest_path = Path(dest_root) / project_number / license_type / asset_category / filename
        return dest_path
    
    def _update_file_record(self, old_path: str, new_path: str):
        """Update file record in database after migration."""
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        
        # Update the file record
        cur.execute("""
            UPDATE files 
            SET path = ?, migration_dest = ?, migration_status = 'migrated'
            WHERE path = ?
        """, (new_path, new_path, old_path))
        
        con.commit()
        con.close()
    
    def rollback_operations(self, operation_count: int = 10) -> Dict[str, Any]:
        """
        Rollback recent operations.
        
        Args:
            operation_count: Number of recent operations to rollback
            
        Returns:
            Dictionary with rollback results
        """
        # Get recent operations
        operations = self.integrity_checker.get_rollback_operations(operation_count)
        
        if not operations:
            return {
                'success': False,
                'error': 'No operations to rollback',
                'rolled_back': 0
            }
        
        results = {
            'success': True,
            'rolled_back': 0,
            'failed': 0,
            'errors': []
        }
        
        # Rollback operations in reverse order
        for operation in reversed(operations):
            try:
                source_path = Path(operation['source_path'])
                dest_path = Path(operation['dest_path'])
                
                # Check if destination still exists
                if not dest_path.exists():
                    results['failed'] += 1
                    results['errors'].append({
                        'operation_id': operation['id'],
                        'error': f'Destination file not found: {dest_path}'
                    })
                    continue
                
                # Verify integrity before rollback
                expected_hash = operation.get('sha256_after')
                if expected_hash:
                    actual_hash = self.integrity_checker.calculate_file_hash(str(dest_path))
                    if actual_hash != expected_hash:
                        results['failed'] += 1
                        results['errors'].append({
                            'operation_id': operation['id'],
                            'error': f'File integrity check failed before rollback'
                        })
                        continue
                
                # Perform rollback
                shutil.move(str(dest_path), str(source_path))
                
                # Log rollback operation
                self.integrity_checker.log_operation(
                    operation='rollback',
                    source_path=str(dest_path),
                    dest_path=str(source_path),
                    status='success',
                    details=f'Rolled back operation {operation["id"]}'
                )
                
                results['rolled_back'] += 1
                
            except Exception as e:
                results['failed'] += 1
                results['errors'].append({
                    'operation_id': operation['id'],
                    'error': str(e)
                })
        
        results['success'] = results['failed'] == 0
        return results
