# MainWindow Propose Names Integration

This guide shows how to integrate the "Propose Names" feature into the ModelFinder MainWindow.

## Step 1: Add Imports

Add these imports to the top of `main.py`:

```python
from PySide6.QtWidgets import QInputDialog, QMessageBox
from PySide6.QtCore import QThreadPool

from src.features.propose_from_reference import propose_for_rows, RowMeta
from src.ui.workers import ProposeWorker
from src.dataio import db as dbio
```

## Step 2: Initialize Thread Pool in `__init__`

In `MainWindow.__init__`, add the thread pool:

```python
def __init__(self):
    super().__init__()
    self.setWindowTitle(f"{APP_NAME} — 3D Asset Finder")
    self.resize(1100, 700)
    self.settings = QtCore.QSettings("ModelFinder", "ModelFinder")
    self.indexer: Indexer | None = None
    
    # Add thread pool for background workers
    self.threadpool = QThreadPool.globalInstance()
    
    self._build_ui()
    ensure_db()
    self.refresh_table()
```

## Step 3: Add Toolbar Action

In `MainWindow._build_ui()`, add the "Propose Names" button to the toolbar:

```python
def _build_ui(self):
    # Existing toolbar setup
    toolbar = QtWidgets.QToolBar()
    toolbar.setMovable(False)
    self.addToolBar(toolbar)
    
    # ... existing buttons ...
    
    # Add Propose Names button
    self.btn_propose = QtWidgets.QToolButton()
    self.btn_propose.setText("🎯 Propose Names")
    self.btn_propose.setToolTip("Generate name proposals using reference parts")
    self.btn_propose.clicked.connect(self.on_propose_names_clicked)
    toolbar.addWidget(self.btn_propose)
    
    # ... rest of UI ...
```

## Step 4: Add Helper Method to Convert Rows

Add this method to convert selected table rows to RowMeta format:

```python
def _selected_rows_to_meta(self) -> list[RowMeta]:
    """
    Convert selected table rows to RowMeta objects for proposal generation.
    
    Returns:
        List of RowMeta objects
    """
    metas = []
    selected = self.table.selectionModel().selectedRows()
    
    if not selected:
        # If nothing selected, use all visible rows
        for row_idx in range(self.model.rowCount()):
            row_data = self.model.get_row_data(row_idx)
            metas.append(RowMeta(
                path=row_data["path"],
                name=row_data["name"],
                ext=row_data["ext"],
                tags=row_data.get("tags", "")
            ))
    else:
        # Use only selected rows
        for idx in selected:
            row_data = self.model.get_row_data(idx.row())
            metas.append(RowMeta(
                path=row_data["path"],
                name=row_data["name"],
                ext=row_data["ext"],
                tags=row_data.get("tags", "")
            ))
    
    return metas
```

## Step 5: Add Propose Names Handler

Add the main handler for the "Propose Names" button:

```python
def on_propose_names_clicked(self):
    """
    Handle "Propose Names" button click.
    
    Workflow:
    1. Ask user for project number
    2. Get selected rows (or all if none selected)
    3. Run proposal generation in background worker
    4. Update UI and database with results
    """
    # 1) Ask for project number
    project_number, ok = QtWidgets.QInputDialog.getText(
        self,
        "Project Number",
        "Enter project number (e.g., 300868):\n\n"
        "This will match files against reference parts for this project.",
        QtWidgets.QLineEdit.EchoMode.Normal,
        ""
    )
    
    if not ok or not project_number.strip():
        return
    
    # 2) Get rows to process
    rows = self._selected_rows_to_meta()
    
    if not rows:
        QtWidgets.QMessageBox.information(
            self,
            "Propose Names",
            "No files to process. Please scan folders first."
        )
        return
    
    # Check if reference parts exist
    from src.dataio.reference_parts import load_reference_parts
    try:
        project_name, parts_map = load_reference_parts(
            str(dbio.DB_PATH),
            project_number.strip()
        )
        
        if not parts_map:
            ret = QtWidgets.QMessageBox.question(
                self,
                "No Reference Parts",
                f"No reference parts found for project {project_number}.\n\n"
                "Proposals will use ML prediction only (lower confidence).\n\n"
                "Continue anyway?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
            )
            if ret != QtWidgets.QMessageBox.StandardButton.Yes:
                return
    except Exception as e:
        QtWidgets.QMessageBox.warning(
            self,
            "Database Error",
            f"Could not check reference parts: {e}"
        )
        return
    
    # 3) Disable button and show progress
    self.btn_propose.setEnabled(False)
    self.status.showMessage(f"Generating proposals for {len(rows)} files...")
    
    # 4) Run worker in background
    worker = ProposeWorker(
        propose_for_rows,
        rows,
        str(dbio.DB_PATH),
        project_number.strip()
    )
    
    worker.signals.finished.connect(self._apply_proposals_to_ui_and_db)
    worker.signals.error.connect(self._on_propose_error)
    
    self.threadpool.start(worker)


def _on_propose_error(self, error_msg: str):
    """Handle proposal generation error."""
    self.btn_propose.setEnabled(True)
    self.status.clearMessage()
    QtWidgets.QMessageBox.critical(
        self,
        "Propose Names Error",
        f"Failed to generate proposals:\n\n{error_msg}"
    )
```

## Step 6: Add Results Handler

Add the handler to apply proposal results:

```python
def _apply_proposals_to_ui_and_db(self, proposals: list[dict]):
    """
    Update UI table and persist proposal fields to database.
    
    Args:
        proposals: List of proposal dictionaries from worker
    """
    if not proposals:
        self.btn_propose.setEnabled(True)
        self.status.showMessage("No proposals generated.", 3000)
        return
    
    # Batch update database
    updated = 0
    for p in proposals:
        # Persist fields to database
        dbio.update_proposal(p["from"], p)
        
        # Update UI model if using ReviewTableModel
        # Find the row by path
        for row_idx in range(self.model.rowCount()):
            row_data = self.model.get_row_data(row_idx)
            if row_data["path"] == p["from"]:
                # Update the row with proposal data
                current_row = list(self.model.rows[row_idx])
                
                # Update proposal fields (indices depend on your model)
                if len(current_row) >= 12:
                    current_row[6] = p["project_number"]      # Project #
                    current_row[7] = p["project_name"]        # Project Name
                    current_row[8] = p["part_name"]           # Part Name
                    current_row[9] = p["conf"]                # Confidence
                    current_row[10] = p["proposed_name"]      # Proposed Name
                    current_row[11] = "discovered"            # Status
                    
                    self.model.rows[row_idx] = tuple(current_row)
                    
                    # Emit data changed for the row
                    top_left = self.model.index(row_idx, 6)
                    bottom_right = self.model.index(row_idx, 11)
                    self.model.dataChanged.emit(top_left, bottom_right)
                
                updated += 1
                break
    
    # Re-enable button
    self.btn_propose.setEnabled(True)
    
    # Show summary
    needs_review = sum(1 for p in proposals if p.get("needs_review", False))
    auto_accept = len(proposals) - needs_review
    
    self.status.showMessage(
        f"Proposals: {updated} updated | "
        f"Auto-accept: {auto_accept} | "
        f"Needs review: {needs_review}",
        7000
    )
    
    # Show detailed dialog
    self._show_proposal_summary(proposals)


def _show_proposal_summary(self, proposals: list[dict]):
    """Show summary dialog with proposal statistics."""
    from src.features.propose_from_reference import summary_stats
    
    stats = summary_stats(proposals)
    
    summary_text = (
        f"Proposal Generation Complete\n\n"
        f"Total files: {stats['total']}\n"
        f"Auto-accept (≥66%): {stats['auto_accept']}\n"
        f"Needs review (<66%): {stats['needs_review']}\n"
        f"Average confidence: {int(stats['avg_confidence'] * 100)}%\n\n"
        f"Projects:\n"
    )
    
    for proj, count in stats['by_project'].items():
        summary_text += f"  {proj}: {count} files\n"
    
    QtWidgets.QMessageBox.information(
        self,
        "Proposal Summary",
        summary_text
    )
```

## Step 7: Add Keyboard Shortcut (Optional)

Add a keyboard shortcut for quick access:

```python
def _build_ui(self):
    # ... in toolbar setup ...
    
    # Add keyboard shortcut: Ctrl+P for Propose Names
    shortcut_propose = QtWidgets.QShortcut(
        QtGui.QKeySequence("Ctrl+P"),
        self
    )
    shortcut_propose.activated.connect(self.on_propose_names_clicked)
```

## Complete Example Usage Flow

### User Workflow:

1. **Import Reference Parts** (once per project):
   ```bash
   python scripts\import_label_excel_to_modelfinder.py ^
     --in "E:\Projects\300868 Superman PF.xlsx" ^
     --db "%APPDATA%\ModelFinder\index.db"
   ```

2. **Scan Folder** in UI:
   - Click "Scan Folders..." or "Scan All Drives"
   - Wait for indexing to complete

3. **Select Files** (optional):
   - Select specific rows in table
   - Or leave none selected to process all visible files

4. **Generate Proposals**:
   - Click "🎯 Propose Names" (or press Ctrl+P)
   - Enter project number: `300868`
   - Wait for background processing

5. **Review Results**:
   - Green confidence badge = auto-accept
   - Red/yellow badge = needs review
   - Use Review Panel to edit if needed

6. **Migrate Files**:
   - Click "Migrate Selected" to execute
   - Files renamed to canonical format

### Expected Output:

```
Original: E:/Raw/300868/head.stl
Proposed: 300868_superman_pf_head.stl
Confidence: 100% ✓

Original: E:/Raw/300868/left_hand.obj
Proposed: 300868_superman_pf_left_arm.obj
Confidence: 75% ⚠ (needs review)

Original: E:/Raw/300868/unknown_piece.fbx
Proposed: 300868_superman_pf_part.fbx
Confidence: 0% ⚠ (needs review)
```

## Database Schema Notes

The proposals update these columns in the `files` table:
- `project_number` - Extracted or user-provided
- `project_name` - From reference or "unknown"
- `part_name` - Matched from reference or generic
- `type_conf` - Confidence score (0.0 to 1.0)
- `status` - Remains "discovered" until migration

## Troubleshooting

### "No reference parts found"
- Run Excel import first
- Check project number is correct
- Verify database path

### "Worker thread error"
- Check database connectivity
- Verify file paths are accessible
- Check Python dependencies installed

### UI not updating
- Ensure model.dataChanged.emit() is called
- Verify row indices match your table model
- Check model.get_row_data() returns correct format

## Next Steps

1. Add progress bar for long operations
2. Implement "Undo" for proposals
3. Add bulk approve/reject buttons
4. Save proposal results to CSV for review

