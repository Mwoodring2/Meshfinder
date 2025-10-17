# UI Review Panel Integration Guide

This guide explains how to integrate the human-in-the-loop review panel into the main ModelFinder application.

## Components Created

### 1. **ReviewPanel** (`src/ui/review_panel.py`)
Right-side panel with:
- 3 editable fields (Project #, Project Name, Part Name)
- Live preview of proposed filename
- Confidence badge with color coding
- Action buttons (Apply, Dry-run, Migrate, Quarantine)
- Keyboard shortcuts (Enter=apply, Tab=next field, Ctrl+L=focus part name)

### 2. **ReviewTableModel** (`src/ui/review_table_model.py`)
Enhanced table model with columns:
- Original: Name, Extension, Size, Modified, Tags, Path
- **New**: Project #, Project Name, Part Name, Conf., Proposed Name, Status

### 3. **Review Helpers** (`src/ui/review_helpers.py`)
Utility functions for:
- Review queue filtering
- Batch operations
- Field validation
- Statistics

## Integration Steps

### Step 1: Import New Components in `main.py`

```python
from src.ui.review_panel import ReviewPanel
from src.ui.review_table_model import ReviewTableModel
from src.ui.review_helpers import (
    needs_review_filter,
    extract_project_from_folder,
    batch_fill_project_name,
    copy_proposed_name,
    ReviewQueueStats
)
from src.ml.project_extractor import propose_fields
```

### Step 2: Replace FileTableModel with ReviewTableModel

In `MainWindow._build_ui()`:

```python
# Replace this:
# self.model = FileTableModel()

# With this:
self.model = ReviewTableModel()
```

### Step 3: Add Review Panel to UI Layout

In `MainWindow._build_ui()`, after creating the table:

```python
# Create review panel
self.review_panel = ReviewPanel()
self.review_panel.setMaximumWidth(350)
self.review_panel.fields_changed.connect(self._on_review_fields_changed)
self.review_panel.apply_requested.connect(self._on_apply_to_selected)

# Update the splitter to include review panel
# Replace existing splitter with horizontal split
h_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
h_split.addWidget(self.table)
h_split.addWidget(self.review_panel)
h_split.setStretchFactor(0, 3)  # Table gets 3/4 width
h_split.setStretchFactor(1, 1)  # Panel gets 1/4 width

# Then add to vertical splitter with preview
split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
split.addWidget(h_split)
split.addWidget(preview_box)
split.setStretchFactor(0, 4)
split.setStretchFactor(1, 1)
```

### Step 4: Update Table Selection Handler

```python
def _on_single_click(self, index: QtCore.QModelIndex):
    if not index.isValid():
        return
    
    self.table.selectRow(index.row())
    row_data = self.model.get_row_data(index.row())
    
    # Update status bar
    self.status.showMessage(f"Selected: {row_data['name']} — {row_data['path']}")
    
    # Update preview pane
    self._update_preview_for_row(row_data)
    
    # Update review panel
    self.review_panel.set_fields(
        project_number=row_data.get('project_number', ''),
        project_name=row_data.get('project_name', ''),
        part_name=row_data.get('part_name', ''),
        confidence=row_data.get('type_conf'),
        ext=row_data.get('ext', '')
    )
```

### Step 5: Add Review Panel Event Handlers

```python
def _on_review_fields_changed(self, fields: dict):
    """Handle field changes in review panel."""
    # Optional: auto-save or mark as modified
    pass

def _on_apply_to_selected(self):
    """Apply edited fields to selected row."""
    idx = self.table.currentIndex()
    if not idx.isValid():
        return
    
    fields = self.review_panel.get_fields()
    row_data = self.model.get_row_data(idx.row())
    
    # Validate fields
    from src.ui.review_helpers import validate_fields
    valid, error = validate_fields(
        fields['project_number'],
        fields['project_name'],
        fields['part_name']
    )
    
    if not valid:
        QtWidgets.QMessageBox.warning(self, "Validation Error", error)
        return
    
    # Update database
    from src.dataio.db import update_file_record
    proposed_name = self.review_panel.get_proposed_name()
    
    # Update local model
    self.model.setData(self.model.index(idx.row(), 6), fields['project_number'], QtCore.Qt.ItemDataRole.EditRole)
    self.model.setData(self.model.index(idx.row(), 7), fields['project_name'], QtCore.Qt.ItemDataRole.EditRole)
    self.model.setData(self.model.index(idx.row(), 8), fields['part_name'], QtCore.Qt.ItemDataRole.EditRole)
    self.model.setData(self.model.index(idx.row(), 10), proposed_name, QtCore.Qt.ItemDataRole.EditRole)
    
    self.status.showMessage(f"Applied: {proposed_name}", 3000)
```

### Step 6: Add Review Queue Filter

In toolbar, add a button for review queue:

```python
btn_review_queue = QtWidgets.QToolButton()
btn_review_queue.setText("📋 Review Queue")
btn_review_queue.setToolTip("Show files needing review")
btn_review_queue.clicked.connect(self._show_review_queue)
toolbar.addWidget(btn_review_queue)
```

```python
def _show_review_queue(self):
    """Filter to show only files needing review."""
    from src.ui.review_helpers import needs_review_filter
    
    # Get all rows
    all_rows = list(self.model.rows)
    
    # Filter for review
    review_rows = []
    for row in all_rows:
        row_dict = {
            "project_number": row[6] if len(row) > 6 else None,
            "type_conf": row[9] if len(row) > 9 else 0,
            "status": row[11] if len(row) > 11 else "discovered"
        }
        if needs_review_filter(row_dict):
            review_rows.append(row)
    
    self.model.set_rows(review_rows)
    self.table.resizeColumnsToContents()
    
    # Show stats
    from src.ui.review_helpers import ReviewQueueStats
    stats = ReviewQueueStats([{
        "type_conf": r[9] if len(r) > 9 else 0,
        "status": r[11] if len(r) > 11 else "discovered"
    } for r in review_rows])
    
    self.status.showMessage(f"Review Queue: {len(review_rows)} files need attention", 5000)
```

### Step 7: Add Context Menu Actions

Update `_context_menu()`:

```python
def _context_menu(self, pos):
    index = self.table.indexAt(pos)
    if not index.isValid():
        return
    
    menu = QtWidgets.QMenu(self)
    
    # Existing actions
    actOpen = menu.addAction("Open")
    actReveal = menu.addAction("Show in Explorer")
    actCopyPath = menu.addAction("Copy Path")
    menu.addSeparator()
    
    # NEW: Review actions
    actCopyProposed = menu.addAction("📋 Copy Proposed Name")
    actSetProjFromFolder = menu.addAction("📁 Set Project # from Folder")
    actBatchFillProject = menu.addAction("📂 Batch Fill Project Name from Parent")
    menu.addSeparator()
    
    # Existing actions
    actRename = menu.addAction("Rename…")
    actFindSimilar = menu.addAction("Find Similar…")
    
    action = menu.exec(self.table.viewport().mapToGlobal(pos))
    if not action:
        return
    
    row_data = self.model.get_row_data(index.row())
    
    # Handle new actions
    if action == actCopyProposed:
        from src.ui.review_helpers import copy_proposed_name
        proposed = copy_proposed_name(row_data)
        QtWidgets.QApplication.clipboard().setText(proposed)
        self.status.showMessage(f"Copied: {proposed}", 3000)
    
    elif action == actSetProjFromFolder:
        from src.ui.review_helpers import extract_project_from_folder
        proj = extract_project_from_folder(row_data['path'])
        self.review_panel.project_number_edit.setText(proj)
        self.status.showMessage(f"Set project #: {proj}", 3000)
    
    elif action == actBatchFillProject:
        # Get all selected rows (or current row if none selected)
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            selected = [index]
        
        from src.ui.review_helpers import batch_fill_project_name
        rows_data = [self.model.get_row_data(idx.row()) for idx in selected]
        updates = batch_fill_project_name(rows_data)
        
        # Apply updates
        for idx_offset, proj_name in updates.items():
            row_idx = selected[idx_offset].row()
            self.model.setData(
                self.model.index(row_idx, 7),
                proj_name,
                QtCore.Qt.ItemDataRole.EditRole
            )
        
        self.status.showMessage(f"Updated {len(updates)} project names", 3000)
    
    # ... existing action handlers ...
```

### Step 8: Update refresh_table() for New Columns

```python
def refresh_table(self):
    term = self.search_edit.text()
    chosen = self.ext_filter.currentText()
    exts = None if chosen == "All" else {chosen}
    
    # Get base rows
    rows = query_files(term, exts)
    
    # Apply path filter
    pf = self.path_filter.text().strip()
    if pf:
        low = pf.lower()
        rows = [r for r in rows if low in (r[0] or "").lower()]
    
    # Enhance rows with metadata (if not already present)
    enhanced_rows = []
    for r in rows:
        if len(r) == 6:  # Old format: (path, name, ext, size, mtime, tags)
            # Propose fields
            from src.ml.project_extractor import propose_fields
            from src.utils.naming import canonical_name
            
            ctx = " ".join([r[1], r[5] or "", Path(r[0]).parent.name])
            fields = propose_fields(r[0], {}, ctx)
            
            proposed = canonical_name(
                fields['project_number'],
                fields['project_name'],
                fields['part_name'],
                r[2].lstrip('.')
            )
            
            # Extend row
            enhanced_rows.append(r + (
                fields['project_number'],
                fields['project_name'],
                fields['part_name'],
                fields['type_conf'],
                proposed,
                'discovered'
            ))
        else:
            enhanced_rows.append(r)
    
    self.model.set_rows(enhanced_rows)
    self.table.resizeColumnsToContents()
```

## Keyboard Shortcuts Summary

- **Enter** - Apply fields to selected row
- **Tab** - Jump to next field (Project # → Project Name → Part Name)
- **Ctrl+L** - Focus Part Name field
- **Ctrl+R** - Show Review Queue (add this shortcut)

## Testing Checklist

- [ ] Review panel updates when row is selected
- [ ] Fields are editable and update preview
- [ ] Apply button updates row and database
- [ ] Review queue filter works correctly
- [ ] Context menu actions function properly
- [ ] Confidence badges show correct colors
- [ ] Keyboard shortcuts work as expected
- [ ] Proposed name preview updates in real-time

## Next Steps

1. Implement dry-run and migration buttons in review panel
2. Add undo/redo for field changes
3. Persist review queue as saved filter
4. Add batch approve for high-confidence items
5. Integrate with existing similarity search

