#!/usr/bin/env python3
"""
ModelFinder GPU Renderer Setup Script
Automatically installs dependencies and integrates GPU rendering
"""

import sys
import os
import subprocess
from pathlib import Path
import shutil

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Error: Python 3.7+ required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_dependencies():
    """Install required packages"""
    print("\n📦 Installing GPU rendering dependencies...")
    
    packages = [
        "moderngl",      # GPU rendering
        "trimesh",       # 3D model loading
        "pillow",        # Image processing
        "numpy",         # Array operations
        "pycollada",     # Additional format support (optional)
    ]
    
    for package in packages:
        print(f"   Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"   ✓ {package} installed")
        except subprocess.CalledProcessError:
            print(f"   ⚠ Failed to install {package}")
            return False
    
    return True

def test_opengl():
    """Test if OpenGL context can be created"""
    print("\n🎮 Testing OpenGL support...")
    
    try:
        import moderngl
        ctx = moderngl.create_context(standalone=True)
        version = ctx.version_code
        ctx.release()
        
        print(f"   ✓ OpenGL {version} context created successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ OpenGL test failed: {e}")
        print("\n   Troubleshooting:")
        print("   - Update your graphics drivers")
        print("   - On Linux, install: sudo apt-get install libegl1-mesa-dev")
        print("   - On headless servers, install: xvfb and osmesa")
        return False

def integrate_with_project():
    """Integrate GPU rendering with existing project"""
    print("\n🔧 Integrating with ModelFinder...")
    
    # Check if main file exists
    main_files = ["main_enhanced.py", "main.py"]
    main_file = None
    
    for file in main_files:
        if Path(file).exists():
            main_file = file
            break
    
    if not main_file:
        print("   ⚠ No main application file found")
        print("   Please manually integrate using the guide")
        return False
    
    print(f"   Found main file: {main_file}")
    
    # Backup original file
    backup_file = f"{main_file}.backup"
    shutil.copy2(main_file, backup_file)
    print(f"   ✓ Created backup: {backup_file}")
    
    # Add integration code
    integration_code = '''
# GPU Preview Integration (added by setup script)
try:
    from preview_integration import integrate_with_main_window, GPU_AVAILABLE
    
    if GPU_AVAILABLE:
        # Automatically add GPU preview dock to MainWindow
        integrate_with_main_window(MainWindow)
        print("✅ GPU preview rendering enabled!")
    else:
        print("ℹ️ GPU preview not available (moderngl not installed)")
except ImportError as e:
    print(f"ℹ️ Preview integration not loaded: {e}")
'''
    
    # Read current file
    with open(main_file, 'r') as f:
        content = f.read()
    
    # Check if already integrated
    if "preview_integration" in content:
        print("   ℹ️ Already integrated")
        return True
    
    # Find where to insert (after imports, before MainWindow class)
    import_end = content.rfind("from ")
    if import_end == -1:
        import_end = content.rfind("import ")
    
    if import_end != -1:
        # Find end of import line
        newline_pos = content.find("\n", import_end)
        if newline_pos != -1:
            # Insert after imports
            new_content = (
                content[:newline_pos + 1] +
                "\n" + integration_code + "\n" +
                content[newline_pos + 1:]
            )
            
            # Write updated file
            with open(main_file, 'w') as f:
                f.write(new_content)
            
            print("   ✓ Integration code added")
            return True
    
    print("   ⚠ Could not auto-integrate. Please add manually.")
    return False

def create_test_script():
    """Create a test script to verify installation"""
    print("\n📝 Creating test script...")
    
    test_code = '''#!/usr/bin/env python3
"""Test script for GPU renderer"""

from pathlib import Path
from gpu_renderer import GPURenderer, RenderSettings

# Test rendering
def test_gpu_renderer():
    print("Testing GPU Renderer...")
    
    try:
        # Create renderer
        settings = RenderSettings(
            width=512,
            height=512,
            samples=4,
            model_color=(0.7, 0.7, 0.8)
        )
        
        renderer = GPURenderer(settings)
        print("✓ Renderer created")
        
        # Find a test file
        test_files = list(Path(".").glob("*.stl")) + list(Path(".").glob("*.obj"))
        
        if test_files:
            test_file = str(test_files[0])
            print(f"✓ Testing with: {test_file}")
            
            # Render
            image = renderer.render_file(test_file, "test_render.png")
            
            if image:
                print(f"✓ Rendered successfully: {image.size}")
                print("✓ Saved to: test_render.png")
            else:
                print("✗ Rendering failed")
        else:
            print("ℹ️ No test files found (*.stl or *.obj)")
        
        renderer.cleanup()
        print("✓ Cleanup complete")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gpu_renderer()
'''
    
    with open("test_gpu_renderer.py", "w") as f:
        f.write(test_code)
    
    print("   ✓ Created test_gpu_renderer.py")
    return True

def main():
    """Main setup function"""
    print("=" * 60)
    print("ModelFinder GPU Renderer Setup")
    print("=" * 60)
    
    # Check requirements
    if not check_python_version():
        return 1
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Failed to install dependencies")
        return 1
    
    # Test OpenGL
    if not test_opengl():
        print("\n⚠️ OpenGL test failed, but continuing...")
        print("   GPU rendering may not work on this system")
    
    # Integrate with project
    integrate_with_project()
    
    # Create test script
    create_test_script()
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ Setup Complete!")
    print("=" * 60)
    
    print("\nNext steps:")
    print("1. Run test script: python test_gpu_renderer.py")
    print("2. Start ModelFinder and check for GPU preview dock")
    print("3. Select a 3D model to see GPU-accelerated preview")
    
    print("\nFiles created:")
    print("- gpu_renderer.py (core rendering engine)")
    print("- preview_integration.py (UI integration)")
    print("- test_gpu_renderer.py (test script)")
    
    print("\nFor manual integration, see GPU_INTEGRATION_GUIDE.md")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
