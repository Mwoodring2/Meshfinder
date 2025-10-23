#!/usr/bin/env python3
"""
Security Patch Application Script
Applies security patches to main_enhanced.py
"""

import os
import shutil
from pathlib import Path

def apply_security_patches():
    """Apply security patches to the main application"""
    
    print("🔒 ModelFinder Security Patch Application")
    print("=" * 50)
    
    # Check if security_patch.py exists
    if not Path("security_patch.py").exists():
        print("❌ security_patch.py not found!")
        print("Please ensure security_patch.py is in the current directory")
        return False
    
    # Backup original file
    if Path("main_enhanced.py").exists():
        shutil.copy("main_enhanced.py", "main_enhanced_backup.py")
        print("✅ Created backup: main_enhanced_backup.py")
    
    # Copy security patch
    shutil.copy("security_patch.py", "security_patch.py")
    print("✅ Security patch module ready")
    
    # Apply patches to main_enhanced.py
    print("\n🔧 Applying security patches...")
    
    # Read current main_enhanced.py
    with open("main_enhanced.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Add security imports at the top
    security_imports = '''
# Security imports
from security_patch import (
    SecurityValidator,
    SecureFileOperations, 
    SecureDatabase,
    SecureTempFile
)

# Security configuration
ALLOWED_EXTENSIONS = {'.stl', '.obj', '.fbx', '.ma', '.mb', '.3ds', '.dae', '.ply', '.gltf', '.glb'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB
'''
    
    # Find the import section and add security imports
    if "from security_patch import" not in content:
        # Add after existing imports
        import_end = content.find("from PySide6 import QtCore")
        if import_end > 0:
            content = content[:import_end] + security_imports + "\n" + content[import_end:]
        else:
            # Add at the beginning if no PySide6 import found
            content = security_imports + "\n" + content
    
    # Replace vulnerable subprocess calls
    vulnerable_patterns = [
        ('subprocess.Popen(["open", file_path])', 'SecureFileOperations().open_file(file_path)'),
        ('subprocess.Popen(["xdg-open", file_path])', 'SecureFileOperations().open_file(file_path)'),
        ('subprocess.Popen(["explorer", "/select,", file_path])', 'SecureFileOperations().reveal_in_explorer(file_path)'),
        ('os.startfile(file_path)', 'SecureFileOperations().open_file(file_path)'),
    ]
    
    for old, new in vulnerable_patterns:
        content = content.replace(old, new)
    
    # Write patched content
    with open("main_enhanced.py", "w", encoding="utf-8") as f:
        f.write(content)
    
    print("✅ Security patches applied successfully!")
    print("\n📋 Next steps:")
    print("1. Run the security test suite: python test_security.py")
    print("2. Test the application: python main_enhanced.py")
    print("3. Verify all vulnerabilities are fixed")
    
    return True

if __name__ == "__main__":
    apply_security_patches()
