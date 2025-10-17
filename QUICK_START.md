# ModelFinder Quick Start Guide

## 🚀 Launch Application

```bash
# Activate virtual environment
.venv\Scripts\activate

# Run ModelFinder
python main.py
```

## 📝 Common Commands

### Import Excel Labels
```bash
python scripts\import_label_excel_to_modelfinder.py ^
  --in "E:\path\to\project_labels.xlsx" ^
  --db "db\modelfinder.db"
```

### Run Stress Test
```bash
python scripts\stress_test.py ^
  --folder "E:\path\to\project_folder" ^
  --db "db\modelfinder.db"
```

### Run Database Migration
```bash
python scripts\migrate_db_schema.py
```

## 🎯 Workflow Steps

### 1. Import Reference Parts (One-time per project)
```bash
python scripts\import_label_excel_to_modelfinder.py ^
  --in "300868_Superman_PF.xlsx" ^
  --db "db\modelfinder.db"
```

### 2. Launch UI
```bash
python main.py
```

### 3. Scan Folders
- Click **"Scan Folders..."** or **"Scan All Drives"**
- Select folders containing 3D files
- Wait for indexing to complete

### 4. Generate Proposals
- Select files (or leave none selected for all)
- Click **"🎯 Propose Names"**
- Enter project number (e.g., `300868`)
- Wait for background processing

### 5. Review Results
- 🟢 Green = Ready (≥92% confidence)
- 🟡 Yellow = Check (66-92% confidence)
- 🔴 Red = Review (< 66% confidence)

### 6. Edit if Needed
- Double-click to edit Project #, Project Name, or Part Name
- Use Review Panel for detailed editing
- Right-click for context menu actions

### 7. Migrate Files
- Click **"Migrate Selected"**
- Confirm operation
- Files renamed and moved to organized structure

## 🎨 UI Features

### Toolbar Buttons
- **Scan Folders...** - Select specific folders to index
- **Scan All Drives** - Scan all fixed drives (Windows)
- **🎯 Propose Names** - Generate name proposals
- **📋 Review Queue** - Show files needing review
- **Export CSV** - Export current view
- **Bulk Tag...** - Auto-tag by folder keywords
- **Rebuild Similarity Index** - Update FAISS index

### Table Columns
| Column | Description |
|--------|-------------|
| Name | Original filename |
| Extension | File type (.stl, .obj, etc.) |
| Size (MB) | File size |
| Modified | Last modified date |
| Tags | User-defined tags |
| Path | Full file path |
| **Project #** | Project number (e.g., 300868) |
| **Project Name** | Slugified project name |
| **Part Name** | Matched part name |
| **Conf.** | Confidence score (color-coded) |
| **Proposed Name** | Canonical filename |
| **Needs Review** | Yes/No flag |

### Keyboard Shortcuts
- **Ctrl+P** - Propose Names
- **Ctrl+L** - Focus Part Name field
- **Ctrl+R** - Review Queue
- **Enter** - Apply changes (in review panel)
- **Tab** - Next field

### Context Menu (Right-click)
- **Open** - Open file in default application
- **Show in Explorer** - Reveal file location
- **Copy Path** - Copy full path to clipboard
- **Copy Proposed Name** - Copy canonical name
- **Set Project # from Folder** - Auto-extract
- **Batch Fill Project Name** - Fill from parent folder
- **Rename...** - Rename file on disk
- **Find Similar...** - Similarity search

## 🔧 Configuration

### Database Location
```
%APPDATA%\ModelFinder\index.db
```

### Supported File Types
- `.stl` - Stereolithography
- `.obj` - Wavefront OBJ
- `.fbx` - Autodesk FBX
- `.ma` / `.mb` - Maya
- `.glb` / `.gltf` - GL Transmission Format

### Confidence Thresholds
```python
≥92% - Auto-accept (green)
66-92% - Consider review (yellow)
<66% - Manual review required (red)
```

## 🐛 Troubleshooting

### "PySide6 not found"
```bash
.venv\Scripts\python.exe -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org PySide6
```

### "No reference parts found"
1. Check project number is correct
2. Run Excel import first
3. Verify database path

### "Database error"
```bash
# Run migration to create tables
python scripts\migrate_db_schema.py
```

### SSL Certificate Errors
Add `--trusted-host` flags to pip commands:
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org <package>
```

## 📊 Performance Tips

### For Large Projects (10,000+ files)
1. Scan in smaller batches
2. Use path filter to narrow results
3. Enable only needed columns
4. Use Review Queue to focus on issues

### For Slow Networks
1. Work offline (after initial scan)
2. Database is local (no network needed)
3. Import Excel files locally

## 📚 Documentation

- `README.md` - General overview
- `docs/REFERENCE_PARTS_INTEGRATION.md` - Complete reference system
- `docs/MAINWINDOW_PROPOSE_INTEGRATION.md` - UI integration
- `docs/TABLE_COLUMNS_GUIDE.md` - Table setup
- `docs/SYSTEM_BEHAVIOR_AND_ENHANCEMENTS.md` - Workflow details
- `STRESS_TEST_RESULTS.md` - Performance validation

## 🎯 Example: Complete Workflow

```bash
# 1. Import labels
python scripts\import_label_excel_to_modelfinder.py ^
  --in "E:\Labels\300868_Superman.xlsx" ^
  --db "db\modelfinder.db"

# Result: ✓ Inserted 25 parts

# 2. Run stress test (optional)
python scripts\stress_test.py ^
  --folder "E:\Models\300868_Superman" ^
  --db "db\modelfinder.db"

# Result: ✓ 95% auto-accept rate

# 3. Launch UI
python main.py

# 4. In UI:
#    - Click "Scan Folders..."
#    - Select "E:\Models\300868_Superman"
#    - Wait for scan (22 files found)
#    - Click "Propose Names"
#    - Enter "300868"
#    - Review results (21 green, 1 yellow)
#    - Click "Migrate Selected"
#    - Done! Files organized to canonical names
```

## 🆘 Support

### Check Logs
```
logs\modelfinder.log
```

### Database Inspection
```bash
python scripts\inspect_db.py
```

### Get Statistics
```python
from src.dataio.reference_parts import get_reference_stats
stats = get_reference_stats("db/modelfinder.db")
print(stats)
# {'total_projects': 5, 'total_parts': 127, 'avg_parts_per_project': 25.4}
```

## ✅ Status Check

Run this to verify everything is working:

```bash
# Check Python version
python --version

# Check dependencies
python -c "import PySide6; print('PySide6:', PySide6.__version__)"

# Check database
python -c "from pathlib import Path; print('DB exists:', Path('db/modelfinder.db').exists())"

# Run stress test
python scripts\stress_test.py --folder "E:\path\to\test" --db "db\modelfinder.db"
```

---

**Quick Help**: For any issues, check the documentation in the `docs/` folder or run the stress test to validate your setup.

**Version**: 0.1.0 (MVP - Production Ready)
**Last Updated**: 2025-01-14

