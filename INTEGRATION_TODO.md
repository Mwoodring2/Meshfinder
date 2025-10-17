# ModelFinder Integration TODO

## Current Status

✅ **Backend Components Built** (All working and tested)
- Excel import system
- Proposal generation engine
- Fuzzy matching (Jaccard similarity)
- Database schema extensions
- Background workers (Qt threading)
- Enhanced table model with color coding
- Stress test validation

❌ **UI Integration Pending** (Not yet added to main.py)
- "Propose Names" button
- New table columns
- Review panel
- Project picker
- Context menu additions

## What Your main.py Currently Has

The existing `main.py` (1198 lines) includes:
- Basic file scanning (Scan Folders, Scan All Drives)
- Table with columns: Name, Extension, Size, Modified, Tags, Path
- Context menu: Open, Show in Explorer, Copy Path, Rename, Find Similar
- Tagging system
- CSV export
- Similarity search (FAISS)
- Geometry computation
- Bulk tagging

## What Needs to Be Added

### Critical Changes to main.py

#### 1. Add Imports (Top of file)
```python
# Add after existing imports
from PySide6.QtCore import QThreadPool
from src.features.propose_from_reference import propose_for_rows, RowMeta
from src.ui.workers import ProposeWorker
from src.dataio.db import update_proposal, batch_update_proposals
```

#### 2. Replace FileTableModel with EnhancedFileTableModel
```python
# Replace the existing FileTableModel class (around line 380-442)
# with the EnhancedFileTableModel from docs/TABLE_COLUMNS_GUIDE.md
# This adds 6 new columns: Project #, Project Name, Part Name, Conf., Proposed Name, Needs Review
```

#### 3. Add Thread Pool (In MainWindow.__init__)
```python
# Around line 450, after self.indexer initialization
self.threadpool = QThreadPool.globalInstance()
```

#### 4. Add "Propose Names" Button (In _build_ui method)
```python
# Around line 492, in toolbar setup, add:
self.btn_propose = QtWidgets.QToolButton()
self.btn_propose.setText("🎯 Propose Names")
self.btn_propose.setToolTip("Generate name proposals using reference parts")
self.btn_propose.clicked.connect(self.on_propose_names_clicked)
toolbar.addWidget(self.btn_propose)
```

#### 5. Add Three New Methods to MainWindow
```python
# Add these methods to the MainWindow class:

def _selected_rows_to_meta(self) -> list[RowMeta]:
    """Convert selected table rows to RowMeta objects."""
    # See docs/MAINWINDOW_PROPOSE_INTEGRATION.md line 87-110

def on_propose_names_clicked(self):
    """Handle Propose Names button click."""
    # See docs/MAINWINDOW_PROPOSE_INTEGRATION.md line 113-195

def _apply_proposals_to_ui_and_db(self, proposals: list[dict]):
    """Apply proposal results to UI and database."""
    # See docs/MAINWINDOW_PROPOSE_INTEGRATION.md line 204-293
```

#### 6. Update refresh_table Method
```python
# Modify existing refresh_table() to handle new columns
# See docs/MAINWINDOW_PROPOSE_INTEGRATION.md line 385-425
```

## Integration Complexity

### Minimal Integration (30 minutes)
Just add "Propose Names" functionality:
1. Add imports
2. Add thread pool
3. Add button
4. Add 3 methods
5. Test

Result: Can generate proposals, but won't see all columns

### Full Integration (2-3 hours)
Complete feature set:
1. Minimal integration above
2. Replace table model (adds 6 columns)
3. Update refresh_table
4. Add review panel (optional)
5. Update context menu
6. Test thoroughly

## Recommended Approach

### Option A: Incremental Integration (Recommended)

**Phase 1** (30 min): Add basic proposal generation
- Add imports, thread pool, button, and 3 methods
- Keep existing table model
- Test with current UI

**Phase 2** (1 hour): Enhance table
- Replace with EnhancedFileTableModel
- Add color coding
- Update column widths

**Phase 3** (1 hour): Polish
- Add review panel
- Add keyboard shortcuts
- Update context menu

### Option B: Use Separate UI File

Create `main_enhanced.py` with all new features:
```bash
# Copy current main.py
cp main.py main_enhanced.py

# Apply all changes to main_enhanced.py
# Keep main.py as fallback

# Run enhanced version
python main_enhanced.py
```

This way you can test without breaking existing functionality.

### Option C: Command-Line Only (Works Now)

Skip UI integration entirely and use scripts:
```bash
# Import labels
python scripts/import_label_excel_to_modelfinder.py --in labels.xlsx --db db/modelfinder.db

# Generate proposals
python scripts/stress_test.py --folder "E:/path" --db db/modelfinder.db --update

# View in current UI
python main.py
```

## File Modifications Required

### Files to Modify
1. ✏️ `main.py` - Add proposal system integration
   - ~100 lines to add
   - ~50 lines to modify

### Files Already Complete (No changes needed)
- ✅ `src/features/propose_from_reference.py`
- ✅ `src/ui/workers.py`
- ✅ `src/dataio/db.py`
- ✅ `src/dataio/reference_parts.py`
- ✅ `src/utils/normalize.py`
- ✅ `src/utils/naming.py`
- ✅ All scripts in `scripts/`

## Testing Plan

### After Integration

1. **Test Imports**
```bash
python -c "from src.features.propose_from_reference import propose_for_rows; print('✓ Imports work')"
```

2. **Test Database**
```bash
python scripts/migrate_db_schema.py
```

3. **Import Test Data**
```bash
# Create test Excel or use existing
python scripts/import_label_excel_to_modelfinder.py --in test.xlsx --db db/modelfinder.db
```

4. **Launch UI**
```bash
python main.py
```

5. **Test Workflow**
- Scan a folder
- Click "Propose Names"
- Enter project number
- Verify proposals appear
- Check color coding
- Test migration

## Complete Integration Guide

All code snippets and step-by-step instructions are in:
- `docs/MAINWINDOW_PROPOSE_INTEGRATION.md` - Complete UI integration
- `docs/TABLE_COLUMNS_GUIDE.md` - Table model update
- `docs/SYSTEM_BEHAVIOR_AND_ENHANCEMENTS.md` - System overview

## Quick Decision Matrix

| Approach | Time | Risk | Features |
|----------|------|------|----------|
| Command-line only | 0 min | None | Backend only |
| Minimal integration | 30 min | Low | Basic proposals |
| Full integration | 2-3 hrs | Medium | All features |
| Separate file | 2-3 hrs | None | All features + fallback |

## My Recommendation

**Start with Separate File Approach**:

1. Create `main_enhanced.py` as a copy of `main.py`
2. Follow integration guides to add all features
3. Test thoroughly with `main_enhanced.py`
4. Once stable, merge back to `main.py`

This gives you:
- ✅ Safety (original still works)
- ✅ Full testing (no pressure)
- ✅ Easy rollback (if needed)
- ✅ All features (complete system)

## Next Steps

Choose one:

**A)** I can help you create `main_enhanced.py` with all integrations
**B)** I can make minimal changes to current `main.py` (just add button)
**C)** You prefer to use command-line tools only

Which would you like?

