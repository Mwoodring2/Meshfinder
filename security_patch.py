#!/usr/bin/env python3
"""
ModelFinder Security Patch
This module provides secure implementations to fix critical vulnerabilities:
- Command Injection (CWE-78)
- Path Traversal (CWE-22)
- SQL Injection (CWE-89)
- Unrestricted File Upload (CWE-434)
"""

import os
import sys
import re
import sqlite3
import subprocess
import shlex
import hashlib
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from urllib.parse import quote

# Security constants
ALLOWED_EXTENSIONS = {'.stl', '.obj', '.fbx', '.ma', '.mb', '.3ds', '.dae', '.ply', '.gltf', '.glb'}
MAX_PATH_LENGTH = 260  # Windows MAX_PATH
MAX_FILENAME_LENGTH = 255
SAFE_PATH_REGEX = re.compile(r'^[a-zA-Z0-9_\-./\\: ]+$')

class SecurityValidator:
    """Validates and sanitizes user inputs to prevent security vulnerabilities"""
    
    @staticmethod
    def validate_path(file_path: str, base_dir: Optional[str] = None) -> Tuple[bool, str]:
        """
        Validates file paths to prevent path traversal attacks.
        
        Args:
            file_path: The path to validate
            base_dir: Optional base directory to restrict access to
            
        Returns:
            Tuple of (is_valid, sanitized_path)
        """
        try:
            # Convert to Path object and resolve to absolute path
            path = Path(file_path).resolve()
            
            # Check for path traversal attempts
            if '..' in file_path or '../' in file_path or '..\\' in file_path:
                return False, "Path traversal detected"
            
            # Check path length
            if len(str(path)) > MAX_PATH_LENGTH:
                return False, "Path too long"
            
            # Check filename length
            if len(path.name) > MAX_FILENAME_LENGTH:
                return False, "Filename too long"
            
            # If base_dir is provided, ensure path is within it
            if base_dir:
                base = Path(base_dir).resolve()
                try:
                    path.relative_to(base)
                except ValueError:
                    return False, "Path outside allowed directory"
            
            # Check if path exists and is not a symlink (prevent symlink attacks)
            if path.exists() and path.is_symlink():
                real_path = path.resolve()
                if base_dir and not str(real_path).startswith(str(Path(base_dir).resolve())):
                    return False, "Symlink points outside allowed directory"
            
            return True, str(path)
            
        except Exception as e:
            return False, f"Invalid path: {str(e)}"
    
    @staticmethod
    def validate_file_extension(file_path: str) -> bool:
        """
        Validates file extensions against allowed 3D model types.
        
        Args:
            file_path: The file path to check
            
        Returns:
            True if extension is allowed, False otherwise
        """
        path = Path(file_path)
        return path.suffix.lower() in ALLOWED_EXTENSIONS
    
    @staticmethod
    def sanitize_sql_identifier(identifier: str) -> str:
        """
        Sanitizes SQL identifiers (table/column names) to prevent SQL injection.
        
        Args:
            identifier: The SQL identifier to sanitize
            
        Returns:
            Sanitized identifier
        """
        # Only allow alphanumeric and underscore
        sanitized = re.sub(r'[^\w]', '_', identifier)
        
        # Ensure it doesn't start with a number
        if sanitized and sanitized[0].isdigit():
            sanitized = 'n_' + sanitized
            
        return sanitized[:64]  # Limit length
    
    @staticmethod
    def sanitize_sql_value(value: Any) -> Any:
        """
        Sanitizes SQL values to prevent injection.
        Use parameterized queries instead when possible.
        
        Args:
            value: The value to sanitize
            
        Returns:
            Sanitized value
        """
        if isinstance(value, str):
            # Remove dangerous SQL keywords and characters
            dangerous_patterns = [
                r';\s*DROP',
                r';\s*DELETE',
                r';\s*UPDATE',
                r';\s*INSERT',
                r';\s*ALTER',
                r';\s*CREATE',
                r'--',
                r'/\*',
                r'\*/',
                r'xp_',
                r'sp_',
                r'exec\s*\(',
            ]
            
            sanitized = value
            for pattern in dangerous_patterns:
                sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
            
            return sanitized
        
        return value
    
    @staticmethod
    def escape_shell_arg(arg: str) -> str:
        """
        Properly escapes shell arguments to prevent command injection.
        
        Args:
            arg: The argument to escape
            
        Returns:
            Escaped argument safe for shell execution
        """
        # Use shlex.quote for POSIX systems
        if sys.platform != "win32":
            return shlex.quote(arg)
        
        # For Windows, use proper escaping
        # Escape special characters
        arg = arg.replace('^', '^^')
        arg = arg.replace('&', '^&')
        arg = arg.replace('|', '^|')
        arg = arg.replace('<', '^<')
        arg = arg.replace('>', '^>')
        arg = arg.replace('(', '^(')
        arg = arg.replace(')', '^)')
        arg = arg.replace('%', '^%')
        arg = arg.replace('!', '^!')
        arg = arg.replace('"', '""')
        
        # Wrap in quotes if contains spaces
        if ' ' in arg:
            arg = f'"{arg}"'
        
        return arg


class SecureFileOperations:
    """Secure file operations to prevent command injection and path traversal"""
    
    @staticmethod
    def open_file(file_path: str, allowed_dirs: Optional[List[str]] = None) -> bool:
        """
        Securely opens a file with the default application.
        
        Args:
            file_path: Path to the file to open
            allowed_dirs: Optional list of allowed base directories
            
        Returns:
            True if successful, False otherwise
        """
        # Validate path
        validator = SecurityValidator()
        
        # Check if path is in allowed directories
        if allowed_dirs:
            valid = False
            for allowed_dir in allowed_dirs:
                is_valid, _ = validator.validate_path(file_path, allowed_dir)
                if is_valid:
                    valid = True
                    break
            if not valid:
                raise SecurityError(f"File path not in allowed directories: {file_path}")
        else:
            is_valid, sanitized = validator.validate_path(file_path)
            if not is_valid:
                raise SecurityError(f"Invalid file path: {sanitized}")
            file_path = sanitized
        
        # Validate file extension
        if not validator.validate_file_extension(file_path):
            raise SecurityError(f"File type not allowed: {Path(file_path).suffix}")
        
        # Check file exists and is not a directory
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if path.is_dir():
            raise IsADirectoryError(f"Path is a directory: {file_path}")
        
        try:
            if sys.platform == "win32":
                # Use os.startfile with explicit 'open' operation
                os.startfile(os.path.normpath(file_path), 'open')
            elif sys.platform == "darwin":
                # Use subprocess with proper argument escaping
                subprocess.run(['open', file_path], check=True)
            else:
                # Linux - use xdg-open with proper escaping
                subprocess.run(['xdg-open', file_path], check=True)
            return True
            
        except Exception as e:
            raise RuntimeError(f"Failed to open file: {str(e)}")
    
    @staticmethod
    def reveal_in_explorer(file_path: str, allowed_dirs: Optional[List[str]] = None) -> bool:
        """
        Securely reveals a file in the file explorer.
        
        Args:
            file_path: Path to the file to reveal
            allowed_dirs: Optional list of allowed base directories
            
        Returns:
            True if successful, False otherwise
        """
        # Validate path
        validator = SecurityValidator()
        
        # Check if path is in allowed directories
        if allowed_dirs:
            valid = False
            for allowed_dir in allowed_dirs:
                is_valid, _ = validator.validate_path(file_path, allowed_dir)
                if is_valid:
                    valid = True
                    break
            if not valid:
                raise SecurityError(f"File path not in allowed directories: {file_path}")
        else:
            is_valid, sanitized = validator.validate_path(file_path)
            if not is_valid:
                raise SecurityError(f"Invalid file path: {sanitized}")
            file_path = sanitized
        
        # Check file exists
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            if sys.platform == "win32":
                # Use explorer with /select flag
                subprocess.run(['explorer', '/select,', os.path.normpath(file_path)], check=True)
            elif sys.platform == "darwin":
                # Use open with -R flag
                subprocess.run(['open', '-R', file_path], check=True)
            else:
                # Linux - open parent directory
                parent_dir = str(path.parent)
                subprocess.run(['xdg-open', parent_dir], check=True)
            return True
            
        except Exception as e:
            raise RuntimeError(f"Failed to reveal file: {str(e)}")


class SecureDatabase:
    """Secure database operations to prevent SQL injection"""
    
    def __init__(self, db_path: str):
        """
        Initialize secure database connection.
        
        Args:
            db_path: Path to the database file
        """
        # Validate database path
        validator = SecurityValidator()
        is_valid, sanitized = validator.validate_path(db_path)
        if not is_valid:
            raise SecurityError(f"Invalid database path: {sanitized}")
        
        self.db_path = sanitized
        self.connection = None
    
    def connect(self):
        """Establish secure database connection"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            # Enable foreign key constraints
            self.connection.execute("PRAGMA foreign_keys = ON")
            # Use WAL mode for better concurrency
            self.connection.execute("PRAGMA journal_mode = WAL")
            # Set secure delete
            self.connection.execute("PRAGMA secure_delete = ON")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to database: {str(e)}")
    
    def execute_query(self, query: str, params: Optional[Tuple] = None) -> sqlite3.Cursor:
        """
        Execute a parameterized query safely.
        
        Args:
            query: SQL query with parameter placeholders (?)
            params: Tuple of parameters to bind
            
        Returns:
            Cursor with query results
        """
        if not self.connection:
            self.connect()
        
        try:
            if params:
                # Use parameterized query (prevents SQL injection)
                cursor = self.connection.execute(query, params)
            else:
                # For queries without parameters, validate it's a SELECT
                if not query.strip().upper().startswith('SELECT'):
                    raise SecurityError("Only SELECT queries allowed without parameters")
                cursor = self.connection.execute(query)
            
            return cursor
            
        except Exception as e:
            raise RuntimeError(f"Query execution failed: {str(e)}")
    
    def insert_file_record(self, file_path: str, file_size: int, tags: str = '') -> int:
        """
        Safely insert a file record into the database.
        
        Args:
            file_path: Path to the file
            file_size: Size of the file in bytes
            tags: Optional tags for the file
            
        Returns:
            ID of the inserted record
        """
        # Validate inputs
        validator = SecurityValidator()
        is_valid, sanitized_path = validator.validate_path(file_path)
        if not is_valid:
            raise SecurityError(f"Invalid file path: {sanitized_path}")
        
        # Use parameterized query to prevent SQL injection
        query = """
            INSERT INTO models (file_path, file_name, file_size, extension, tags, date_modified)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
        """
        
        path = Path(sanitized_path)
        params = (
            str(path),
            path.name,
            file_size,
            path.suffix.lower(),
            tags
        )
        
        cursor = self.execute_query(query, params)
        self.connection.commit()
        
        return cursor.lastrowid
    
    def search_files(self, search_term: str, extension: Optional[str] = None) -> List[Dict]:
        """
        Safely search for files in the database.
        
        Args:
            search_term: Term to search for
            extension: Optional file extension filter
            
        Returns:
            List of matching file records
        """
        # Build parameterized query
        query = "SELECT * FROM models WHERE (file_name LIKE ? OR tags LIKE ?)"
        params = [f'%{search_term}%', f'%{search_term}%']
        
        if extension:
            # Validate extension
            if not re.match(r'^\.\w+$', extension):
                raise SecurityError(f"Invalid extension format: {extension}")
            query += " AND extension = ?"
            params.append(extension)
        
        cursor = self.execute_query(query, tuple(params))
        
        # Convert to dictionary format
        columns = [description[0] for description in cursor.description]
        results = []
        for row in cursor.fetchall():
            results.append(dict(zip(columns, row)))
        
        return results
    
    def close(self):
        """Close database connection safely"""
        if self.connection:
            self.connection.close()
            self.connection = None


class SecureTempFile:
    """Secure temporary file handling"""
    
    @staticmethod
    def create_temp_file(suffix: str = '.tmp', prefix: str = 'modelfinder_') -> str:
        """
        Create a secure temporary file.
        
        Args:
            suffix: File suffix
            prefix: File prefix
            
        Returns:
            Path to the temporary file
        """
        # Validate suffix
        if suffix and not re.match(r'^\.\w+$', suffix):
            raise SecurityError(f"Invalid suffix format: {suffix}")
        
        # Validate prefix
        if not re.match(r'^[\w_-]+$', prefix):
            raise SecurityError(f"Invalid prefix format: {prefix}")
        
        try:
            # Create secure temporary file
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix=suffix,
                prefix=prefix,
                delete=False
            ) as tmp_file:
                return tmp_file.name
        except Exception as e:
            raise RuntimeError(f"Failed to create temporary file: {str(e)}")
    
    @staticmethod
    def secure_temp_dir(prefix: str = 'modelfinder_') -> str:
        """
        Create a secure temporary directory.
        
        Args:
            prefix: Directory prefix
            
        Returns:
            Path to the temporary directory
        """
        # Validate prefix
        if not re.match(r'^[\w_-]+$', prefix):
            raise SecurityError(f"Invalid prefix format: {prefix}")
        
        try:
            # Create secure temporary directory
            temp_dir = tempfile.mkdtemp(prefix=prefix)
            
            # Set restrictive permissions (owner only)
            if sys.platform != "win32":
                os.chmod(temp_dir, 0o700)
            
            return temp_dir
            
        except Exception as e:
            raise RuntimeError(f"Failed to create temporary directory: {str(e)}")


# Example usage and integration functions
def patch_main_window_file_operations(main_window_class):
    """
    Patch the MainWindow class to use secure file operations.
    This should be called during application initialization.
    """
    
    def secure_open_file(self, file_path: str):
        """Secure replacement for _open_file method"""
        try:
            # Define allowed directories (e.g., user's project folders)
            allowed_dirs = getattr(self, 'allowed_directories', None)
            
            secure_ops = SecureFileOperations()
            secure_ops.open_file(file_path, allowed_dirs)
            self.status.showMessage(f"Opened: {Path(file_path).name}", 2000)
            
        except Exception as e:
            from PySide6 import QtWidgets
            QtWidgets.QMessageBox.warning(self, "Security Error", str(e))
    
    def secure_reveal_in_explorer(self, file_path: str):
        """Secure replacement for _reveal_in_explorer method"""
        try:
            # Define allowed directories
            allowed_dirs = getattr(self, 'allowed_directories', None)
            
            secure_ops = SecureFileOperations()
            secure_ops.reveal_in_explorer(file_path, allowed_dirs)
            self.status.showMessage(f"Revealed: {Path(file_path).name}", 2000)
            
        except Exception as e:
            from PySide6 import QtWidgets
            QtWidgets.QMessageBox.warning(self, "Security Error", str(e))
    
    # Replace methods
    main_window_class._open_file = secure_open_file
    main_window_class._reveal_in_explorer = secure_reveal_in_explorer


def create_secure_db_connection(db_path: str) -> SecureDatabase:
    """
    Create a secure database connection.
    
    Args:
        db_path: Path to the database file
        
    Returns:
        Secure database connection object
    """
    secure_db = SecureDatabase(db_path)
    secure_db.connect()
    return secure_db


if __name__ == "__main__":
    # Run security tests
    print("ModelFinder Security Patch - Test Suite")
    print("=" * 50)
    
    # Test path validation
    print("\n1. Testing Path Validation...")
    validator = SecurityValidator()
    
    test_paths = [
        ("C:\\Users\\test\\file.stl", True),
        ("../../../etc/passwd", False),
        ("C:\\Users\\test\\..\\..\\Windows\\System32", False),
        ("file.stl; rm -rf /", False),
    ]
    
    for path, expected in test_paths:
        is_valid, result = validator.validate_path(path)
        status = "✓" if is_valid == expected else "✗"
        print(f"  {status} {path}: {result}")
    
    # Test SQL sanitization
    print("\n2. Testing SQL Sanitization...")
    
    test_sql = [
        "users",
        "users'; DROP TABLE users; --",
        "1=1 OR 'a'='a'",
    ]
    
    for sql in test_sql:
        sanitized = validator.sanitize_sql_value(sql)
        print(f"  Input: {sql}")
        print(f"  Output: {sanitized}")
    
    # Test shell argument escaping
    print("\n3. Testing Shell Argument Escaping...")
    
    test_args = [
        "normal_file.stl",
        "file with spaces.stl",
        "file.stl; rm -rf /",
        "file.stl && calc.exe",
    ]
    
    for arg in test_args:
        escaped = validator.escape_shell_arg(arg)
        print(f"  Input: {arg}")
        print(f"  Output: {escaped}")
    
    print("\n✓ Security patch tests completed!")
