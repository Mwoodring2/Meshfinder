# 🎉 ModelFinder: Complete Archive Organization System

## What You Now Have

A **complete end-to-end system** for organizing your 12-year 3D file archive using **AI that learns from YOUR structure and understands geometry**.

---

## 🎯 Core Problem Solved

**Before:** Generic files named `part2.stl` scattered everywhere  
**After:** Intelligent organization: `300668_yoda_left_foot.stl` in `300668_Yoda_PF/Commercial/Character/`

---

## 🚀 Complete Workflow

### 1. **Scan Your Archive**
```
Tools → Scan Folders
```
- Select folders containing 3D models
- Files indexed to database with metadata
- Geometry computed (triangles, dimensions, volume)

### 2. **Import Reference Data** (Optional)
```
File → Import Excel
```
- Load project parts lists from Excel
- Builds vocabulary for fuzzy matching
- Links project numbers to part names

### 3. **Train from Well-Organized Folders** (NEW! 🔥)
```
Tools → Train from Archive
```
- Select your best-organized project folders
- System learns:
  - **Geometric patterns** (what a foot/hand/head looks like)
  - **Naming conventions** (your 12-year evolution)
  - **Folder structure** (project/license/category)
- Builds ML classifier that recognizes parts by shape

### 4. **Generate Intelligent Proposals**
```
Select files → 🎯 Propose Names
```
- AI analyzes:
  - ✅ Mesh geometry (shape, size, orientation)
  - ✅ Parent folder context (project, character)
  - ✅ Reference database (known parts)
  - ✅ Naming patterns (learned conventions)
- Generates names like: `300668_yoda_left_foot.stl`
- Shows confidence scores (geometry + text + context)

### 5. **Review & Quick-Fix**
```
Table View: Color-coded by confidence
```
- 🟢 **Green rows (≥85%)**: High confidence, ready to go
- 🟡 **Yellow rows (66-84%)**: Medium, double-check
- 🔴 **Red rows (<66%)**: Low confidence, needs review

**Quick actions:**
- **F2**: Rename manually
- **Right-click → Rename to Proposed**: One-click rename
- **Space**: Preview in Windows 3D Viewer
- Edit license/category inline

### 6. **Plan Migration**
```
Tools → 📦 Migrate Archive (Ctrl+M)
```
- Preview entire reorganization
- Destination structure: `<root>/<project>/<license>/<category>/<file>`
- **Dry-run** shows all moves before executing
- **Conflict detection**:
  - Files that already exist
  - Duplicate destinations
  - Missing information
- Real-time statistics

### 7. **Execute Migration**
```
Click "Execute Migration" in planner
```
- Creates organized folder structure
- Moves files atomically (safe)
- Updates database paths
- Logs all operations (rollback capable)
- Progress tracking with cancel option

---

## 🧠 ML Intelligence Features

### Geometric Learning
The system **understands shape**, not just text:

**Example: Recognizing a foot**
```python
Features extracted:
- Elongated shape (length >> width)
- Flat bottom (base plane)
- Arch curvature
- Asymmetric (left vs right)
- Typical size ratio

→ ML classifies: "foot" (85% confidence)
→ Geometric laterality: "left" (72% confidence)
→ Context from folder: "300668_Yoda"
→ Final proposal: "300668_yoda_left_foot.stl"
```

### Context Awareness
```
Folder: E:/Archive/300668_Yoda_PF/
File: part2.stl

System analyzes:
✓ Project number: 300668 (from folder name)
✓ Character: Yoda (from folder name)
✓ Geometry: Foot-shaped mesh
✓ Laterality: Left-sided
✓ License: Commercial (from folder path)
✓ Category: Character (from context)

→ Proposes: "300668_yoda_left_foot.stl"
→ Destination: "300668_Yoda_PF/Commercial/Character/"
```

### Active Learning
System improves from your corrections:
1. **Proposes:** `300668_yoda_foot.stl` (70%)
2. **You correct:** `300668_yoda_left_foot.stl`
3. **System learns:** This geometry pattern = "left foot" specifically
4. **Next time:** Higher confidence, better predictions

---

## 📊 Database Schema

### Files Table
```sql
CREATE TABLE files (
    path TEXT PRIMARY KEY,
    name TEXT,
    ext TEXT,
    size INTEGER,
    mtime REAL,
    tags TEXT,
    
    -- Geometry (computed)
    tris INTEGER,
    dim_x, dim_y, dim_z REAL,
    volume REAL,
    watertight INTEGER,
    
    -- AI Proposals
    project_number TEXT,
    project_name TEXT,
    part_name TEXT,
    proposed_name TEXT,
    type_conf REAL,
    
    -- Organization
    license_type TEXT,        -- Commercial, Personal, Fan-Art, etc.
    asset_category TEXT,      -- Character, Prop, Environment, etc.
    
    -- Migration
    migration_dest TEXT,      -- Planned destination path
    migration_status TEXT,    -- pending, migrated, failed
    status TEXT               -- renamed, reviewed, etc.
);
```

### Training Samples (ML)
```sql
CREATE TABLE training_samples (
    file_path TEXT,
    project_number TEXT,
    character_name TEXT,
    part_type TEXT,           -- head, foot, hand, etc.
    laterality TEXT,          -- left, right, center
    license_type TEXT,
    features_json TEXT,       -- Geometric features
    source TEXT               -- 'archive_scan' or 'user_correction'
);
```

### Operations Log (Audit Trail)
```sql
CREATE TABLE operations_log (
    timestamp TEXT,
    operation TEXT,           -- MIGRATE, RENAME, DELETE
    source_path TEXT,
    dest_path TEXT,
    status TEXT,              -- SUCCESS, FAILED
    details TEXT
);
```

---

## 🎨 UI Features

### Main Window (3-Panel Layout)
```
┌────────────┬─────────────────────────────┬────────────┐
│  FILTERS   │     FILE TABLE             │  PREVIEW   │
│            │  (color-coded confidence)  │            │
│ [Filters]  │ ┌────────────────────────┐ │ [Thumb]    │
│ [Browser]  │ │ 🟢 yoda_left_foot.stl  │ │            │
│            │ │ 🟡 part_unknown.obj    │ │ [Metadata] │
│  File      │ │ 🔴 mesh2.stl           │ │            │
│  Types     │ └────────────────────────┘ │ [Geometry] │
│  □ .stl    │                            │            │
│  □ .obj    │  Stats: 1234 files        │ Tris: 4.5K │
│  □ .fbx    │  Ready: 890 (72%)         │ Vol: 125cm³│
│            │  Review: 344 (28%)        │            │
│  Project   │                            │            │
│  [300668]  │                            │            │
│            │                            │            │
│  Size      │                            │            │
│  Min: 1MB  │                            │            │
│  Max: 100MB│                            │            │
└────────────┴─────────────────────────────┴────────────┘
```

### Folder Browser Mode
- Tree view of file system
- Real-time file browsing (no indexing needed)
- Right-click context menu:
  - Scan This Folder
  - Open in Explorer
  - Preview File

### Migration Planner
```
┌─────────────────────────────────────────────────────┐
│  📦 Archive Reorganization Planner                  │
├─────────────────────────────────────────────────────┤
│  Destination: E:/Organized_Archive/                 │
│  Default License: [Commercial ▼]                    │
│  Default Category: [Character ▼]                    │
├─────────────────────────────────────────────────────┤
│  Current Name    │ Destination              │Status │
│  part2.stl       │ 300668.../left_foot.stl  │ 🟢 Ready│
│  mesh_unknown    │ 300915.../hand_left.obj  │ 🟡 Exists│
│  test.obj        │ ERROR_NO_PROJECT         │ 🔴 Error│
├─────────────────────────────────────────────────────┤
│  Total: 1234 │ Ready: 890 │ Conflicts: 12          │
├─────────────────────────────────────────────────────┤
│  [🔄 Refresh] [👁️ Dry Run] [✅ Execute Migration]  │
└─────────────────────────────────────────────────────┘
```

---

## ⌨️ Keyboard Shortcuts

### Global
- `Ctrl+R` - Scan Folders
- `Ctrl+P` - Propose Names
- `Ctrl+M` - Migration Planner
- `Ctrl+F` - Find/Search
- `F5` - Refresh Table
- `F11` - Full Screen

### File Operations
- `F2` - Rename selected file
- `Space` - Preview in Windows 3D Viewer
- `Delete` - Delete selected file(s)
- `Ctrl+C` - Copy selected paths

### View
- `Ctrl+L` - Toggle Show All Files
- `Ctrl+Shift+F` - Toggle Filters Panel
- `Ctrl+Shift+P` - Toggle Preview Panel
- `Ctrl+Shift+T` - Toggle Dark/Light Theme

---

## 📁 Organized Archive Structure

### Final Result
```
E:/Organized_Archive/
├── 300668_Yoda_PF/
│   ├── Commercial/
│   │   ├── Character/
│   │   │   ├── 300668_yoda_head.stl
│   │   │   ├── 300668_yoda_left_foot.stl
│   │   │   ├── 300668_yoda_right_foot.stl
│   │   │   ├── 300668_yoda_left_hand.obj
│   │   │   └── 300668_yoda_torso.stl
│   │   ├── Accessory/
│   │   │   ├── 300668_yoda_lightsaber.fbx
│   │   │   └── 300668_yoda_cloak.obj
│   │   └── Prop/
│   │       └── 300668_yoda_base.stl
│   └── metadata.json
│
├── 300915_ManBat_Parts/
│   ├── Fan-Art/
│   │   ├── Character/
│   │   │   ├── 300915_manbat_head.stl
│   │   │   ├── 300915_manbat_wings_left.obj
│   │   │   └── 300915_manbat_wings_right.obj
│   │   └── Accessory/
│   │       └── 300915_manbat_ears.stl
│   └── metadata.json
│
└── operations_log.json
```

---

## 🔧 CLI Tools

### Database Initialization
```bash
python scripts/init_db.py
```

### Training from Archive
```bash
python -m src.ml.archive_trainer "E:/MyArchive/BestProjects/"
```

### Feature Extraction Test
```bash
python -m src.ml.geometry_features test_file.stl
```

### Comprehensive Test
```bash
python scripts/comprehensive_stress_test.py --cleanup
```

---

## 🎓 Usage Examples

### Example 1: First-Time Setup
```bash
# 1. Initialize database
python scripts/init_db.py

# 2. Launch app
launch.bat

# 3. In app: Scan your archive
Tools → Scan Folders → Select root folder

# 4. Import reference data (if you have Excel)
File → Import Excel → Select parts list

# 5. Train from best projects
Tools → Train from Archive → Select 3-5 well-organized folders

# 6. Generate proposals
Select files → 🎯 Propose Names → Select project

# 7. Review and migrate
Tools → 📦 Migrate Archive → Dry Run → Execute
```

### Example 2: Daily Workflow
```bash
# Launch app
launch.bat

# Scan new files
Ctrl+R → Select new folder

# Quick propose
Select files → Ctrl+P → Pick project

# Quick rename
Right-click → Rename to Proposed

# Or migrate in batch
Ctrl+M → Review plan → Execute
```

### Example 3: Correcting ML
```bash
# System proposes: "300668_yoda_part.stl"
# You know it's a foot

# Method 1: Manual rename (F2)
Rename to: "300668_yoda_left_foot.stl"

# Method 2: Edit in migration planner
Ctrl+M → Edit row → Change part type → Re-propose

# System learns from correction
# Next similar geometry → Higher confidence "foot" prediction
```

---

## 📈 Benefits

✅ **Unified naming** across 12 years of files  
✅ **Organized structure** by project/license/category  
✅ **Intelligent AI** that learns from YOUR patterns  
✅ **Geometric recognition** (shape-based, not just text)  
✅ **Context-aware** (folder structure informs decisions)  
✅ **Safe operations** (dry-run, rollback capable)  
✅ **Audit trail** (every operation logged)  
✅ **Incremental improvement** (learns from corrections)  

---

## 🔮 Future Enhancements

- [ ] Deep learning for complex part recognition
- [ ] Visual similarity search (find similar meshes)
- [ ] Auto-detect duplicates across projects
- [ ] Batch quality scoring
- [ ] Team model sharing
- [ ] Version control integration
- [ ] Cloud backup integration

---

## 📚 Documentation

- `ARCHIVE_ML_DESIGN.md` - ML system architecture
- `MIGRATION_STATUS.md` - Migration from main.py
- `3D_VIEWER_IMPLEMENTATION.md` - 3D viewer docs
- `docs/` - Full documentation folder

---

## 🆘 Support

### Common Issues

**Q: Proposals have low confidence**  
A: Train from more well-organized folders. System needs examples.

**Q: Wrong part types detected**  
A: Correct them (F2) - system will learn. Or retrain model.

**Q: Migration conflicts**  
A: Resolve in planner before executing. Check for duplicates.

**Q: Geometry features not extracting**  
A: Ensure trimesh is installed: `pip install trimesh`

---

## 🎉 You're Ready!

Your archive organization system is **complete and production-ready**:

1. ✅ Database initialized
2. ✅ ML training ready
3. ✅ Geometric feature extraction
4. ✅ Intelligent proposals
5. ✅ Migration planner
6. ✅ Complete UI
7. ✅ All features tested

**Start organizing your 12-year archive today!** 🚀

