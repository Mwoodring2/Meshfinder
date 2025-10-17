"""
Background worker threads for ModelFinder UI operations.

Provides QRunnable workers for long-running operations like proposal generation,
database operations, and file processing without blocking the UI thread.
"""
from PySide6.QtCore import QObject, Signal, QRunnable, Slot
from typing import List, Dict, Any, Callable, Optional


class ProposeResult(QObject):
    """Signal emitter for proposal worker results."""
    finished = Signal(list)   # list of proposals (dicts)
    error = Signal(str)
    progress = Signal(int, int)  # current, total


class ProposeWorker(QRunnable):
    """
    Background worker for proposal generation.
    
    Runs proposal generation in a separate thread to avoid blocking the UI.
    Emits signals for completion, errors, and progress updates.
    """
    
    def __init__(
        self,
        func: Callable,
        rows: List[Any],
        db_path: str,
        project_number: str
    ):
        """
        Initialize proposal worker.
        
        Args:
            func: Function to call (e.g., propose_for_rows)
            rows: List of RowMeta objects
            db_path: Path to SQLite database
            project_number: Project number for reference lookup
        """
        super().__init__()
        self.func = func
        self.rows = rows
        self.db_path = db_path
        self.project_number = project_number
        self.signals = ProposeResult()
        self.setAutoDelete(True)

    @Slot()
    def run(self):
        """Execute the proposal generation."""
        try:
            # Call the proposal function
            out = self.func(self.rows, self.db_path, self.project_number)
            self.signals.finished.emit(out)
        except Exception as e:
            self.signals.error.emit(str(e))


class BatchUpdateResult(QObject):
    """Signal emitter for batch update worker results."""
    finished = Signal(int)  # number of rows updated
    error = Signal(str)
    progress = Signal(int, int)  # current, total


class BatchUpdateWorker(QRunnable):
    """
    Background worker for batch database updates.
    
    Updates multiple file records with proposal data without blocking the UI.
    """
    
    def __init__(self, update_func: Callable, proposals: List[Dict[str, Any]]):
        """
        Initialize batch update worker.
        
        Args:
            update_func: Function to call (e.g., batch_update_proposals)
            proposals: List of proposal dictionaries
        """
        super().__init__()
        self.update_func = update_func
        self.proposals = proposals
        self.signals = BatchUpdateResult()
        self.setAutoDelete(True)
    
    @Slot()
    def run(self):
        """Execute the batch update."""
        try:
            # Call the update function
            updated = self.update_func(self.proposals)
            self.signals.finished.emit(updated)
        except Exception as e:
            self.signals.error.emit(str(e))


class MigrationResult(QObject):
    """Signal emitter for migration worker results."""
    finished = Signal(dict)  # result dictionary
    error = Signal(str)
    progress = Signal(int, int, str)  # current, total, current_file


class MigrationWorker(QRunnable):
    """
    Background worker for file migration operations.
    
    Executes file migration (move/rename) operations in a separate thread
    with progress reporting.
    """
    
    def __init__(
        self,
        migrate_func: Callable,
        plans: List[Dict[str, Any]],
        user: str = "system",
        skip_conflicts: bool = True
    ):
        """
        Initialize migration worker.
        
        Args:
            migrate_func: Function to call (e.g., execute_migration)
            plans: List of migration plan dictionaries
            user: Username for operation log
            skip_conflicts: If True, skip conflicting files
        """
        super().__init__()
        self.migrate_func = migrate_func
        self.plans = plans
        self.user = user
        self.skip_conflicts = skip_conflicts
        self.signals = MigrationResult()
        self.setAutoDelete(True)
    
    @Slot()
    def run(self):
        """Execute the migration."""
        try:
            # Call the migration function
            result = self.migrate_func(
                self.plans,
                user=self.user,
                skip_conflicts=self.skip_conflicts
            )
            self.signals.finished.emit(result)
        except Exception as e:
            self.signals.error.emit(str(e))


class ExcelImportResult(QObject):
    """Signal emitter for Excel import worker results."""
    finished = Signal(dict)  # result dictionary with stats
    error = Signal(str)
    progress = Signal(str)  # status message


class ExcelImportWorker(QRunnable):
    """
    Background worker for Excel label import.
    
    Imports part labels from Excel files into the reference parts database.
    """
    
    def __init__(
        self,
        import_func: Callable,
        excel_path: str,
        db_path: str,
        project_number: Optional[str] = None,
        project_name: Optional[str] = None
    ):
        """
        Initialize Excel import worker.
        
        Args:
            import_func: Function to call for import
            excel_path: Path to Excel file
            db_path: Path to SQLite database
            project_number: Optional project number override
            project_name: Optional project name override
        """
        super().__init__()
        self.import_func = import_func
        self.excel_path = excel_path
        self.db_path = db_path
        self.project_number = project_number
        self.project_name = project_name
        self.signals = ExcelImportResult()
        self.setAutoDelete(True)
    
    @Slot()
    def run(self):
        """Execute the Excel import."""
        try:
            # Call the import function
            result = self.import_func(
                self.excel_path,
                self.db_path,
                project_number=self.project_number,
                project_name=self.project_name
            )
            self.signals.finished.emit(result)
        except Exception as e:
            self.signals.error.emit(str(e))


class GeometryComputeResult(QObject):
    """Signal emitter for geometry computation results."""
    finished = Signal(int)  # number processed
    error = Signal(str)
    progress = Signal(int, int)  # current, total


class GeometryComputeWorker(QRunnable):
    """
    Background worker for geometry computation.
    
    Computes mesh statistics (triangle count, dimensions, volume) for 3D files.
    """
    
    def __init__(self, compute_func: Callable, file_paths: List[str]):
        """
        Initialize geometry compute worker.
        
        Args:
            compute_func: Function to call for computation
            file_paths: List of file paths to process
        """
        super().__init__()
        self.compute_func = compute_func
        self.file_paths = file_paths
        self.signals = GeometryComputeResult()
        self.setAutoDelete(True)
    
    @Slot()
    def run(self):
        """Execute the geometry computation."""
        try:
            # Call the compute function
            processed = self.compute_func(self.file_paths)
            self.signals.finished.emit(processed)
        except Exception as e:
            self.signals.error.emit(str(e))

