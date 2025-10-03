# Setup Guide

## Requirements
- Windows 10/11
- Python 3.10+ (tested on 3.12/3.13)
- Git (optional)

## Step 1: Clone or unzip project

```powershell
cd E:\File storage for 3d printing\woodring_print_files\Meshfinder
```

## Step 2: Create virtual environment

```powershell
python -m venv .venv
```

## Step 3: Activate virtual environment

```powershell
# If activation is blocked by policy, bypass just for this shell:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
. .venv\Scripts\Activate.ps1
```

## Step 4: Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements-core.txt
```

## Step 5: Test installation

```powershell
# Test the scanner
python scripts/scan_folder.py --help

# Test the searcher
python scripts/search_cli.py --help
```

## Quick Start

### 1. Scan Your First Folder
```powershell
# Full command with all options
python scripts/scan_folder.py --root data/raw --out data --db db/modelfinder.db --faiss db/faiss.index --posters

# Or using virtual environment Python directly
.\.venv\Scripts\python.exe scripts\scan_folder.py --root data/raw --out data --db db/modelfinder.db --faiss db/faiss.index --posters

# Or scan a specific folder
python scripts/scan_folder.py --root "C:\Your3DModels" --verbose
```

### 2. Search Your Models
```powershell
# Find similar models to a specific file
python scripts/search_cli.py --like-path "data/raw/example.obj" --topk 10 --reveal 1

# Search by asset ID
python scripts/search_cli.py --like-id 1 --topk 5

# Search with filters
python scripts/search_cli.py --like-id 1 --ext .stl --max-tris 10000 --watertight true
```

## Configuration

### Database Location
By default, the database is stored at `db/modelfinder.db`. You can change this:

```powershell
# Use custom database location
python scripts/scan_folder.py --root "C:\My3DModels" --db "C:\MyData\models.db"
python scripts/search_cli.py --like-id 1 --db "C:\MyData\models.db"
```

### Supported File Formats
ModelFinder supports these 3D file formats:
- **STL** (.stl) - ASCII and Binary
- **OBJ** (.obj) - With materials
- **GLB** (.glb) - glTF binary (exports)
- **ZTL** (.ztl) - ZBrush format (recorded only)
- **FBX** (.fbx) - Planned via Assimp
- **PLY** (.ply) - Planned via Assimp
- **3DS** (.3ds) - Planned via Assimp
- **DAE** (.dae) - Planned via Assimp

## Troubleshooting

### Common Issues

#### "ModuleNotFoundError" when running scripts
```powershell
# Solution: Install missing dependencies
pip install -r requirements-core.txt
```

#### "Database not found" error
```powershell
# Solution: Run a scan first to create the database
python scripts/scan_folder.py --root "path/to/your/models"
```

#### "No 3D models found" message
- Check that your folder contains supported file types
- Verify file extensions are lowercase (e.g., `.stl` not `.STL`)
- Try scanning with specific extensions: `--extensions .stl .obj`

#### Memory issues with large models
- Reduce the number of parallel workers in config
- Process models in smaller batches
- Consider using a machine with more RAM

### Performance Tips

1. **Use SSD storage** for better I/O performance
2. **Close other applications** when scanning large collections
3. **Scan incrementally** - add new models rather than rescanning everything
4. **Use specific file types** when you know what you're looking for

### Getting Help

1. **Check the logs** in the `logs/` directory
2. **Use verbose mode** for detailed output: `--verbose`
3. **Check database statistics**: `python scripts/search_cli.py --stats`
4. **Review the documentation** in the `docs/` folder

## Next Steps

Once you have ModelFinder running:

1. **Explore the UI** (when available) for a graphical interface
2. **Set up automated scanning** with scheduled tasks
3. **Customize the configuration** in `config.json`
4. **Contribute to the project** by reporting bugs or suggesting features

## Advanced Usage

### Building Standalone Executable
```powershell
# Create a standalone executable
scripts\build_exe.bat

# The executable will be created in dist\ModelFinder.exe
```

### Custom Configuration
Create a `config.json` file to customize behavior:

```json
{
  "scanning": {
    "supported_extensions": [".stl", ".obj", ".fbx"],
    "max_file_size": 50000000
  },
  "conversion": {
    "output_format": "glb",
    "generate_previews": true
  }
}
```

---

**Happy 3D Model Hunting!** 🎯