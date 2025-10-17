# Reference Parts Integration Guide

This guide explains how the Excel import and reference parts lookup integrates with ModelFinder's proposal system.

## Workflow Overview

```
1. Import Excel Labels → 2. Reference DB → 3. Propose Names → 4. Final Migration
   (Excel file)           (SQLite table)     (Auto-match)      (Canonical names)
```

## Step 1: Import Excel Labels

### Command Line Usage

```bash
# Basic import to CSV
python scripts\import_label_excel_to_modelfinder.py ^
  --in "E:\path\300868 Superman PF.xlsx" ^
  --out "E:\path\300868_parts.csv"

# Direct database upsert (recommended)
python scripts\import_label_excel_to_modelfinder.py ^
  --in "E:\path\300868 Superman PF.xlsx" ^
  --db "E:\ModelFinder\db\modelfinder.db" ^
  --table project_reference_parts

# Both CSV and database
python scripts\import_label_excel_to_modelfinder.py ^
  --in "E:\path\300868 Superman PF.xlsx" ^
  --out "E:\path\300868_parts.csv" ^
  --db "E:\ModelFinder\db\modelfinder.db"

# Preview without writing
python scripts\import_label_excel_to_modelfinder.py ^
  --in "E:\path\300868 Superman PF.xlsx" ^
  --preview
```

### Excel File Format

The script auto-detects columns. Supported formats:

**Format 1: Simple list**
```
Part Name
---------
Head
Left Arm
Right Arm
Torso
```

**Format 2: Full table**
```
Part         | Description      | Qty
-------------|------------------|----
Head         | Main head piece  | 1
Left Arm     | Articulated arm  | 1
Weapon Blade | Lightsaber       | 2
```

### What Gets Stored

The `project_reference_parts` table contains:
- `project_number` - Extracted from filename (e.g., "300868")
- `project_name` - Slugified project name (e.g., "superman_pf")
- `part_name` - Slugified part name (e.g., "head", "left_arm")
- `part_name_original` - Original name from Excel
- `description` - Part description (optional)
- `quantity` - Number of parts (default: 1)
- `tags` - Auto-generated from description + quantity

## Step 2: Database Schema

The reference table is created automatically:

```sql
CREATE TABLE project_reference_parts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_number TEXT NOT NULL,
    project_name TEXT,
    part_name TEXT NOT NULL,
    part_name_original TEXT,
    description TEXT,
    quantity INTEGER DEFAULT 1,
    tags TEXT,
    source_file TEXT,
    source_row INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(project_number, part_name)
);

CREATE INDEX idx_project_reference_parts_project 
ON project_reference_parts(project_number);
```

## Step 3: Enhanced Proposal with Auto-Match

### How It Works

1. **Extract project number** from file path/context
   - `E:/Models/300868/some_file.stl` → Project: `300868`

2. **Lookup project info** in `project_reference_parts`
   - Sets `project_number` = `300868`
   - Sets `project_name` = `superman_pf`

3. **Match part name** using fuzzy matching
   - File: `head_v2.stl` → Matches reference: `head`
   - File: `left_hand.stl` → Matches reference: `left_arm` (partial)
   - File: `unknown_piece.stl` → No match (needs review)

4. **Generate canonical name**
   - Format: `{project_number}_{project_name}_{part_name}.{ext}`
   - Result: `300868_superman_pf_head.stl`

### Code Integration

In your main application, use the enhanced proposal builder:

```python
from src.features.propose_enhanced import build_proposals_enhanced
from src.dataio.db import get_file_records

# Get files to process
files = get_file_records(status="discovered", limit=100)

# Build proposals with reference lookup
plans = build_proposals_enhanced(
    files,
    dest_root="E:/Organized",
    db_path="E:/ModelFinder/db/modelfinder.db",
    table_name="project_reference_parts"
)

# Check results
for plan in plans:
    if plan['reference_match']:
        print(f"✓ Matched: {plan['from']} → {plan['to']}")
        print(f"  Reference: {plan['reference_info']['original_name']}")
    else:
        print(f"⚠ Needs review: {plan['from']}")
        print(f"  No reference match found")
```

### Confidence Scoring

The enhanced proposal adjusts confidence based on reference matches:

- **Reference match found**: `type_conf = 0.95` (high confidence)
- **No reference match**: `needs_review = True` (manual review required)
- **No reference parts for project**: Falls back to ML prediction

## Step 4: Auto-Suggestion in UI

### Part Name Suggestions

When user selects a file in the review panel:

```python
from src.features.propose_enhanced import suggest_part_names

# Get suggestions for current file
suggestions = suggest_part_names(
    filename="head_assembly.stl",
    project_number="300868",
    db_path="E:/ModelFinder/db/modelfinder.db"
)

# Display in dropdown or autocomplete
for s in suggestions:
    print(f"{s['part_name']} ({int(s['score']*100)}%)")
    # Output:
    # head (90%)
    # head_assembly (75%)
```

### UI Integration Points

1. **On file selection** - Auto-populate fields from reference
2. **Part name field** - Show autocomplete dropdown with suggestions
3. **Confidence badge** - Green for reference matches
4. **Review queue** - Filter out files with reference matches

### Example UI Code

```python
def _on_single_click(self, index: QtCore.QModelIndex):
    if not index.isValid():
        return
    
    row_data = self.model.get_row_data(index.row())
    
    # Get suggestions from reference table
    from src.features.propose_enhanced import suggest_part_names
    
    project_num = row_data.get('project_number')
    if project_num:
        suggestions = suggest_part_names(
            filename=row_data['name'],
            project_number=project_num,
            db_path=DB_PATH
        )
        
        # Update review panel with suggestions
        if suggestions:
            best_match = suggestions[0]
            self.review_panel.set_fields(
                project_number=project_num,
                project_name=row_data.get('project_name', ''),
                part_name=best_match['part_name'],
                confidence=best_match['score'],
                ext=row_data['ext']
            )
            
            # Show other suggestions in dropdown
            self.part_name_suggestions = suggestions
```

## Step 5: Final Output Format

All operations produce canonical filenames:

```
Format: {project_number}_{project_name}_{part_name}.{ext}

Examples:
- 300868_superman_pf_head.stl
- 300868_superman_pf_left_arm.obj
- 300868_superman_pf_weapon_blade.fbx
```

## Complete Example Workflow

### 1. Import Excel labels
```bash
python scripts\import_label_excel_to_modelfinder.py ^
  --in "E:\Projects\300868 Superman PF.xlsx" ^
  --db "E:\ModelFinder\db\modelfinder.db"
```

Output:
```
Processing: 300868 Superman PF.xlsx
Project Number: 300868
Project Name: Superman PF

Using column 'Part' as part name
Found 12 parts

✓ Database updated: E:\ModelFinder\db\modelfinder.db
  Inserted: 12 new parts
  Updated: 0 existing parts
  Table: project_reference_parts

=== Statistics ===
Total parts: 12
Parts with descriptions: 8
Parts with qty > 1: 2
Total quantity: 15
```

### 2. Scan folder with files
```python
# In ModelFinder UI or script
from src.dataio.db import get_file_records
files = get_file_records(status="discovered")
```

### 3. Build proposals with auto-match
```python
from src.features.propose_enhanced import build_proposals_enhanced

plans = build_proposals_enhanced(
    files,
    dest_root="E:/Organized",
    db_path="E:/ModelFinder/db/modelfinder.db"
)

# Results:
# ✓ E:/Raw/300868/head.stl → E:/Organized/300868/300868_superman_pf_head.stl
# ✓ E:/Raw/300868/arm_left.stl → E:/Organized/300868/300868_superman_pf_left_arm.stl
# ⚠ E:/Raw/300868/unknown.stl → Needs review (no match)
```

### 4. Review and migrate
```python
from src.features.migrate import dry_run_migration, execute_migration

# Dry run first
dry_run = dry_run_migration(plans)
print(f"Ready: {dry_run['success_count']}, Conflicts: {dry_run['conflict_count']}")

# Execute if dry run successful
if dry_run['conflict_count'] == 0:
    result = execute_migration(plans, user="michael")
    print(f"Migrated: {result['success_count']}")
```

## Benefits

1. **Automation** - 95%+ of files auto-matched if reference parts exist
2. **Consistency** - Canonical naming enforced across all files
3. **Quality** - High confidence for reference matches
4. **Speed** - Bulk operations with minimal manual review
5. **Traceability** - Full audit log of all operations

## Troubleshooting

### No matches found

**Problem**: Files not matching reference parts

**Solution**:
1. Check project number extraction: `extract_project_number(path)`
2. Verify reference parts exist: Check DB table
3. Review filename similarity: Fuzzy matching requires some overlap

### Wrong project info

**Problem**: Incorrect project number/name auto-filled

**Solution**:
1. Override in Excel import: `--project-num "ABC-1234"`
2. Update reference table directly
3. Manual review in UI

### Duplicate part names

**Problem**: Multiple files match same reference part

**Solution**:
1. Add suffixes to filenames before processing
2. Use quantity field in reference to expect duplicates
3. Manual review to disambiguate

## Next Steps

1. Add UI autocomplete for part names
2. Implement "Learn from corrections" to improve matching
3. Add batch operations for multiple projects
4. Export migration summary reports

