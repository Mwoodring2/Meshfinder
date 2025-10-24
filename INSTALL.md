# ModelFinder - Installation Guide

## Quick Start

### Windows Users
1. **Install ML dependencies only:**
   ```cmd
   install_ml_deps.bat
   ```

2. **Install all dependencies:**
   ```cmd
   install_all_deps.bat
   ```

### Linux/Mac Users
```bash
./install_deps.sh
```

### Manual Installation
```bash
pip install -r requirements.txt
```

## Dependencies

### Core 3D Processing
- **trimesh** - 3D mesh loading and processing
- **numpy** - Numerical computations
- **pillow** - Image processing for previews

### Machine Learning (Similarity Search)
- **faiss-cpu** - Fast similarity search
- **scikit-learn** - ML algorithms
- **joblib** - Parallel processing

### Excel Import
- **pandas** - Data manipulation
- **openpyxl** - Excel file reading

### GUI Framework
- **PySide6** - Qt-based user interface

## Virtual Environment (Recommended)

### Create Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Troubleshooting

### FAISS Installation Issues
If `faiss-cpu` fails to install:
```bash
# Try conda instead
conda install -c conda-forge faiss-cpu

# Or use pip with specific version
pip install faiss-cpu==1.7.4
```

### PySide6 Issues
If GUI installation fails:
```bash
# Try alternative Qt binding
pip install PyQt6

# Or use system package manager
# Ubuntu/Debian: sudo apt install python3-pyside6
# macOS: brew install pyside6
```

### Large File Support
For very large STL files (200MB+), ensure you have sufficient RAM:
- **Minimum**: 8GB RAM
- **Recommended**: 16GB+ RAM
- **For 300MB+ files**: 32GB+ RAM

## Features Enabled

After installation, ModelFinder provides:
- ✅ 3D mesh preview generation
- ✅ Similarity search between models
- ✅ Excel import with auto-header detection
- ✅ Large file support (200-300MB+ STL files)
- ✅ Robust mesh analysis and diagnostics
