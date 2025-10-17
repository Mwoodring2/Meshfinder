"""
File migration operations with atomic transactions and rollback support.

Provides dry-run, execute, and rollback capabilities for safe file operations.
"""
import os
import shutil
import json
from pathlib import Path
from typing import List, Dict
from ..dataio.db import update_file_record, log_op


def dry_run_migration(plans: List[Dict]) -> Dict:
    """
    Simulate migration without making changes.
    
    Args:
        plans: List of migration proposals
    
    Returns:
        Dictionary with dry-run results:
        - success_count: Number of files that would succeed
        - conflict_count: Number of files with conflicts
        - conflicts: List of conflict details
        - size_total: Total size to be moved
    """
    results = {
        "success_count": 0,
        "conflict_count": 0,
        "conflicts": [],
        "size_total": 0,
        "operations": []
    }
    
    for plan in plans:
        src = Path(plan["from"])
        dst = Path(plan["to"])
        
        # Check source exists
        if not src.exists():
            results["conflicts"].append({
                "type": "source_missing",
                "from": str(src),
                "to": str(dst),
                "message": "Source file does not exist"
            })
            results["conflict_count"] += 1
            continue
        
        # Check destination conflict
        if dst.exists():
            results["conflicts"].append({
                "type": "destination_exists",
                "from": str(src),
                "to": str(dst),
                "message": "Destination file already exists"
            })
            results["conflict_count"] += 1
            continue
        
        # Check parent directory can be created
        if not dst.parent.exists():
            try:
                # Dry run - don't actually create
                results["operations"].append({
                    "action": "create_dir",
                    "path": str(dst.parent)
                })
            except Exception as e:
                results["conflicts"].append({
                    "type": "mkdir_error",
                    "from": str(src),
                    "to": str(dst),
                    "message": f"Cannot create directory: {e}"
                })
                results["conflict_count"] += 1
                continue
        
        # Success case
        results["success_count"] += 1
        results["size_total"] += src.stat().st_size
        results["operations"].append({
            "action": "move",
            "from": str(src),
            "to": str(dst),
            "size": src.stat().st_size
        })
    
    return results


def execute_migration(plans: List[Dict], user: str = "system", skip_conflicts: bool = True) -> Dict:
    """
    Execute migration with atomic operations and logging.
    
    Args:
        plans: List of migration proposals
        user: Username for operation log
        skip_conflicts: If True, skip conflicting files; if False, raise error
    
    Returns:
        Dictionary with execution results:
        - success_count: Number of successfully migrated files
        - error_count: Number of errors
        - errors: List of error details
        - rollback_data: Data needed for rollback
    """
    results = {
        "success_count": 0,
        "error_count": 0,
        "errors": [],
        "rollback_data": []
    }
    
    for plan in plans:
        src = Path(plan["from"])
        dst = Path(plan["to"])
        
        try:
            # Pre-flight checks
            if not src.exists():
                raise FileNotFoundError(f"Source file not found: {src}")
            
            if dst.exists():
                if skip_conflicts:
                    results["errors"].append({
                        "from": str(src),
                        "to": str(dst),
                        "error": "Destination exists (skipped)"
                    })
                    results["error_count"] += 1
                    continue
                else:
                    raise FileExistsError(f"Destination exists: {dst}")
            
            # Create destination directory
            dst.parent.mkdir(parents=True, exist_ok=True)
            
            # Execute move (atomic operation)
            shutil.move(str(src), str(dst))
            
            # Update database record
            update_file_record(str(src), str(dst), plan["fields"])
            
            # Log operation
            detail = json.dumps({
                "fields": plan["fields"],
                "needs_review": plan.get("needs_review", False)
            })
            log_op("move", str(src), str(dst), detail, user)
            
            # Store rollback data
            results["rollback_data"].append({
                "from": str(dst),
                "to": str(src),
                "fields": plan["fields"]
            })
            
            results["success_count"] += 1
            
        except Exception as e:
            results["errors"].append({
                "from": str(src),
                "to": str(dst),
                "error": str(e)
            })
            results["error_count"] += 1
    
    return results


def rollback_migration(rollback_data: List[Dict], user: str = "system") -> Dict:
    """
    Rollback a migration operation.
    
    Args:
        rollback_data: Data from execute_migration results
        user: Username for operation log
    
    Returns:
        Dictionary with rollback results
    """
    results = {
        "success_count": 0,
        "error_count": 0,
        "errors": []
    }
    
    for item in rollback_data:
        src = Path(item["from"])  # Current location
        dst = Path(item["to"])    # Original location
        
        try:
            if not src.exists():
                raise FileNotFoundError(f"File not found: {src}")
            
            # Create destination directory
            dst.parent.mkdir(parents=True, exist_ok=True)
            
            # Move back
            shutil.move(str(src), str(dst))
            
            # Update database record (revert to original path, reset status)
            fields = item["fields"].copy()
            fields["status"] = "discovered"  # Reset status
            update_file_record(str(src), str(dst), fields)
            
            # Log rollback operation
            log_op("rollback", str(src), str(dst), "", user)
            
            results["success_count"] += 1
            
        except Exception as e:
            results["errors"].append({
                "from": str(src),
                "to": str(dst),
                "error": str(e)
            })
            results["error_count"] += 1
    
    return results


def validate_migration_plan(plans: List[Dict]) -> Dict:
    """
    Validate migration plan before execution.
    
    Args:
        plans: List of migration proposals
    
    Returns:
        Validation results with warnings and errors
    """
    validation = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "stats": {
            "total": len(plans),
            "needs_review": 0,
            "duplicates": 0
        }
    }
    
    # Check for duplicate destinations
    destinations = {}
    for plan in plans:
        dst = plan["to"]
        if dst in destinations:
            validation["errors"].append({
                "type": "duplicate_destination",
                "files": [destinations[dst], plan["from"]],
                "destination": dst
            })
            validation["stats"]["duplicates"] += 1
            validation["valid"] = False
        else:
            destinations[dst] = plan["from"]
        
        # Count items needing review
        if plan.get("needs_review", False):
            validation["stats"]["needs_review"] += 1
    
    # Warn if many files need review
    if validation["stats"]["needs_review"] > len(plans) * 0.5:
        validation["warnings"].append({
            "type": "high_review_rate",
            "message": f"{validation['stats']['needs_review']} of {len(plans)} files need review (>50%)"
        })
    
    return validation

