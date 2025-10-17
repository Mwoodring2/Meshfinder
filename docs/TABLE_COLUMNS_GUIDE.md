# Table Columns Enhancement Guide

This guide shows how to add the new proposal columns to the ModelFinder main table.

## New Columns to Add

The following columns display proposal data and guide the user's review workflow:

| Column | Type | Description | Color Coding |
|--------|------|-------------|--------------|
| Project # | Text | Project number (e.g., "300915") | - |
| Project Name | Text | Slugified project name (e.g., "manbat") | - |
| Part Name | Text | Slugified part name (e.g., "base_part") | - |
| Proposed Name | Text | Full canonical filename | - |
| Conf. | Percentage | Confidence score (0-100%) | Green ≥92%, Yellow 66-92%, Red <66% |
| Needs Review | Yes/No | Flag for manual review | Red = Yes, Green = No |

## Implementation Steps

### Step 1: Update Table Model

Replace `FileTableModel` with `ReviewTableModel` or extend it:

```python
# In main.py, update the model class

class EnhancedFileTableModel(QtCore.QAbstractTableModel):
    """Enhanced table model with proposal columns."""
    
    headers = [
        "Name",           # 0
        "Extension",      # 1  
        "Size (MB)",      # 2
        "Modified",       # 3
        "Tags",           # 4
        "Path",           # 5
        "Project #",      # 6  ← NEW
        "Project Name",   # 7  ← NEW
        "Part Name",      # 8  ← NEW
        "Conf.",          # 9  ← NEW
        "Proposed Name",  # 10 ← NEW
        "Needs Review"    # 11 ← NEW
    ]
    
    def __init__(self):
        super().__init__()
        self.rows = []
    
    def set_rows(self, rows):
        """
        Set table rows.
        
        Row format: (path, name, ext, size, mtime, tags,
                     project_number, project_name, part_name,
                     type_conf, proposed_name, needs_review)
        """
        self.beginResetModel()
        self.rows = rows
        self.endResetModel()
    
    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.rows)
    
    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self.headers)
    
    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        
        row = self.rows[index.row()]
        col = index.column()
        
        # Unpack row (handle variable length for backward compatibility)
        path = row[0] if len(row) > 0 else ""
        name = row[1] if len(row) > 1 else ""
        ext = row[2] if len(row) > 2 else ""
        size = row[3] if len(row) > 3 else 0
        mtime = row[4] if len(row) > 4 else 0
        tags = row[5] if len(row) > 5 else ""
        project_num = row[6] if len(row) > 6 else ""
        project_name = row[7] if len(row) > 7 else ""
        part_name = row[8] if len(row) > 8 else ""
        type_conf = row[9] if len(row) > 9 else None
        proposed_name = row[10] if len(row) > 10 else ""
        needs_review = row[11] if len(row) > 11 else True
        
        # Display role
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if col == 0: return name
            if col == 1: return ext
            if col == 2: return f"{size/1024/1024:.2f}"
            if col == 3: return time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime)) if mtime else ""
            if col == 4: return tags or ""
            if col == 5: return path
            if col == 6: return project_num or ""
            if col == 7: return project_name or ""
            if col == 8: return part_name or ""
            if col == 9: 
                if type_conf is not None:
                    return f"{int(type_conf * 100)}%"
                return ""
            if col == 10: return proposed_name or ""
            if col == 11: return "Yes" if needs_review else "No"
        
        # Background color for Conf. column
        if role == QtCore.Qt.ItemDataRole.BackgroundRole and col == 9:
            if type_conf is not None:
                if type_conf >= 0.92:
                    return QtGui.QColor("#d4edda")  # Light green
                elif type_conf >= 0.66:
                    return QtGui.QColor("#fff3cd")  # Light yellow
                else:
                    return QtGui.QColor("#f8d7da")  # Light red
        
        # Background color for Needs Review column
        if role == QtCore.Qt.ItemDataRole.BackgroundRole and col == 11:
            if needs_review:
                return QtGui.QColor("#f8d7da")  # Light red (needs attention)
            else:
                return QtGui.QColor("#d4edda")  # Light green (ready)
        
        # Text color for Needs Review
        if role == QtCore.Qt.ItemDataRole.ForegroundRole and col == 11:
            if needs_review:
                return QtGui.QColor("#721c24")  # Dark red text
            else:
                return QtGui.QColor("#155724")  # Dark green text
        
        # Font weight for Needs Review
        if role == QtCore.Qt.ItemDataRole.FontRole and col == 11:
            font = QtGui.QFont()
            font.setBold(True)
            return font
        
        # Tooltip
        if role == QtCore.Qt.ItemDataRole.ToolTipRole:
            if col == 9 and type_conf is not None:
                conf_pct = int(type_conf * 100)
                if type_conf >= 0.92:
                    return f"High confidence ({conf_pct}%) - Auto-accept"
                elif type_conf >= 0.66:
                    return f"Medium confidence ({conf_pct}%) - Consider review"
                else:
                    return f"Low confidence ({conf_pct}%) - Needs review"
            if col == 11:
                if needs_review:
                    return "Manual review required - Low confidence or missing data"
                else:
                    return "Ready for migration - High confidence match"
        
        return None
    
    def headerData(self, section, orientation, role):
        if role == QtCore.Qt.ItemDataRole.DisplayRole and orientation == QtCore.Qt.Orientation.Horizontal:
            return self.headers[section]
        return None
    
    def flags(self, index):
        fl = super().flags(index)
        col = index.column()
        # Make tags and metadata fields editable
        if col in [4, 6, 7, 8]:  # Tags, Project #, Project Name, Part Name
            fl |= QtCore.Qt.ItemFlag.ItemIsEditable
        return fl
    
    def setData(self, index, value, role):
        if role == QtCore.Qt.ItemDataRole.EditRole:
            row_idx = index.row()
            col = index.column()
            
            row = list(self.rows[row_idx])
            
            # Extend row to have all columns if needed
            while len(row) < 12:
                row.append(None if len(row) < 11 else True)
            
            if col == 4:  # Tags
                row[5] = str(value)
            elif col == 6:  # Project #
                row[6] = str(value)
            elif col == 7:  # Project Name
                row[7] = str(value)
            elif col == 8:  # Part Name
                row[8] = str(value)
            
            self.rows[row_idx] = tuple(row)
            self.dataChanged.emit(index, index)
            return True
        
        return False
    
    def get_row_data(self, row_idx: int) -> dict:
        """Get row data as dictionary."""
        if row_idx < 0 or row_idx >= len(self.rows):
            return {}
        
        row = self.rows[row_idx]
        return {
            "path": row[0] if len(row) > 0 else "",
            "name": row[1] if len(row) > 1 else "",
            "ext": row[2] if len(row) > 2 else "",
            "size": row[3] if len(row) > 3 else 0,
            "mtime": row[4] if len(row) > 4 else 0,
            "tags": row[5] if len(row) > 5 else "",
            "project_number": row[6] if len(row) > 6 else "",
            "project_name": row[7] if len(row) > 7 else "",
            "part_name": row[8] if len(row) > 8 else "",
            "type_conf": row[9] if len(row) > 9 else None,
            "proposed_name": row[10] if len(row) > 10 else "",
            "needs_review": row[11] if len(row) > 11 else True
        }
```

### Step 2: Update MainWindow to Use New Model

In `MainWindow.__init__`:

```python
def __init__(self):
    super().__init__()
    self.setWindowTitle(f"{APP_NAME} — 3D Asset Finder")
    self.resize(1200, 800)  # Wider window for more columns
    self.settings = QtCore.QSettings("ModelFinder", "ModelFinder")
    self.indexer: Indexer | None = None
    self.threadpool = QThreadPool.globalInstance()
    self._build_ui()
    ensure_db()
    self.refresh_table()
```

In `MainWindow._build_ui`:

```python
def _build_ui(self):
    # ... toolbar setup ...
    
    # Central table with new model
    self.table = QtWidgets.QTableView()
    self.model = EnhancedFileTableModel()  # Use enhanced model
    self.table.setModel(self.model)
    
    # Table configuration
    self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
    self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
    self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
    self.table.setAlternatingRowColors(True)
    self.table.setSortingEnabled(False)
    
    # Enable horizontal scrolling for many columns
    self.table.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
    
    # Set column widths
    self.table.setColumnWidth(0, 200)   # Name
    self.table.setColumnWidth(1, 80)    # Extension
    self.table.setColumnWidth(2, 100)   # Size
    self.table.setColumnWidth(3, 150)   # Modified
    self.table.setColumnWidth(4, 150)   # Tags
    self.table.setColumnWidth(5, 300)   # Path
    self.table.setColumnWidth(6, 100)   # Project #
    self.table.setColumnWidth(7, 150)   # Project Name
    self.table.setColumnWidth(8, 150)   # Part Name
    self.table.setColumnWidth(9, 80)    # Conf.
    self.table.setColumnWidth(10, 300)  # Proposed Name
    self.table.setColumnWidth(11, 100)  # Needs Review
    
    # ... rest of UI ...
```

### Step 3: Update refresh_table() Method

Update the method to include proposal data:

```python
def refresh_table(self):
    """Refresh table with enhanced columns."""
    term = self.search_edit.text()
    chosen = self.ext_filter.currentText()
    exts = None if chosen == "All" else {chosen}
    
    # Get base rows from database
    rows = query_files(term, exts)
    
    # Apply path filter
    pf = self.path_filter.text().strip()
    if pf:
        low = pf.lower()
        rows = [r for r in rows if low in (r[0] or "").lower()]
    
    # Enhance rows with proposal columns if not present
    enhanced_rows = []
    for r in rows:
        if len(r) >= 12:
            # Already has all columns
            enhanced_rows.append(r)
        else:
            # Extend row with empty proposal columns
            enhanced_rows.append(r + (
                "",      # project_number
                "",      # project_name
                "",      # part_name
                None,    # type_conf
                "",      # proposed_name
                True     # needs_review (default)
            ))
    
    self.model.set_rows(enhanced_rows)
    
    # Resize columns to content (optional)
    # self.table.resizeColumnsToContents()
```

### Step 4: Add Filter for "Needs Review"

Add a button to filter the review queue:

```python
# In toolbar
btn_review_queue = QtWidgets.QToolButton()
btn_review_queue.setText("⚠️ Review Queue")
btn_review_queue.setToolTip("Show only files needing review")
btn_review_queue.clicked.connect(self._show_review_queue)
toolbar.addWidget(btn_review_queue)
```

```python
def _show_review_queue(self):
    """Filter to show only files needing review."""
    all_rows = list(self.model.rows)
    
    # Filter for needs_review = True
    review_rows = [r for r in all_rows if len(r) > 11 and r[11]]
    
    self.model.set_rows(review_rows)
    self.status.showMessage(f"Review Queue: {len(review_rows)} files need attention", 5000)
```

### Step 5: Add Color Legend (Optional)

Add a legend to help users understand the color coding:

```python
def _build_ui(self):
    # ... after table setup ...
    
    # Color legend
    legend = QtWidgets.QLabel(
        "🟢 High conf. (≥92%)  |  "
        "🟡 Medium conf. (66-92%)  |  "
        "🔴 Low conf. (<66%)  |  "
        "🟢 Ready  |  "
        "🔴 Needs Review"
    )
    legend.setStyleSheet("padding: 4px; background: #f8f9fa; border: 1px solid #dee2e6;")
    
    # Add to layout
    # ...
```

## Visual Examples

### Confidence Colors

```
┌─────────┬──────────────┐
│ Score   │ Background   │
├─────────┼──────────────┤
│ ≥92%    │ 🟢 Green     │
│ 66-92%  │ 🟡 Yellow    │
│ <66%    │ 🔴 Red       │
└─────────┴──────────────┘
```

### Needs Review Colors

```
┌──────────┬──────────────────┬──────────┐
│ Value    │ Background       │ Text     │
├──────────┼──────────────────┼──────────┤
│ Yes      │ 🔴 Light Red     │ Dark Red │
│ No       │ 🟢 Light Green   │ Dark Grn │
└──────────┴──────────────────┴──────────┘
```

## Sample Data Display

```
Name              | Ext  | Project # | Part Name  | Conf. | Needs Review
------------------|------|-----------|------------|-------|-------------
10_base_part.stl  | .stl | 300915    | base_part  | 90%   | No  🟢
head_v2.stl       | .stl | 300915    | head       | 100%  | No  🟢
unknown.stl       | .stl | 300915    | part       | 0%    | Yes 🔴
left_hand.obj     | .obj | 300915    | left_arm   | 75%   | No  🟡
```

## Database Query Update

Update `query_files()` in database.py to include new columns:

```python
def query_files(term: str, exts: set[str] | None = None, limit: int = 5000):
    term = term.strip()
    sql = """
        SELECT path, name, ext, size, mtime, tags,
               project_number, project_name, part_name,
               type_conf, '' as proposed_name,
               CASE WHEN type_conf < 0.66 OR project_number IS NULL THEN 1 ELSE 0 END as needs_review
        FROM files
    """
    args = []
    clauses = []
    
    if term:
        like = f"%{term.lower()}%"
        clauses.append("(LOWER(name) LIKE ? OR LOWER(tags) LIKE ?)")
        args.extend([like, like])
    
    if exts:
        placeholders = ",".join(["?"] * len(exts))
        clauses.append(f"ext IN ({placeholders})")
        args.extend([e.lower() for e in exts])
    
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    
    sql += " ORDER BY mtime DESC LIMIT ?"
    args.append(limit)
    
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(sql, args)
    rows = cur.fetchall()
    con.close()
    
    return rows
```

## Testing Checklist

- [ ] All columns display correctly
- [ ] Colors apply to Conf. and Needs Review columns
- [ ] Tooltips show on hover
- [ ] Column widths are appropriate
- [ ] Editable columns work (Tags, Project #, etc.)
- [ ] Review Queue filter works
- [ ] Sorting works (if enabled)
- [ ] Horizontal scrolling works for many columns
- [ ] Performance is acceptable with 1000+ rows

## Troubleshooting

### Columns not showing
- Verify model headers list is complete
- Check column count in `columnCount()`
- Ensure row data has all fields

### Colors not applying
- Check `BackgroundRole` returns `QtGui.QColor`
- Verify confidence values are floats (0.0-1.0)
- Ensure needs_review is boolean

### Edit not working
- Check `flags()` includes `ItemIsEditable`
- Verify `setData()` returns True
- Ensure column index matches editable columns

## Next Steps

1. Add sorting by clicking column headers
2. Add column visibility toggle
3. Add export with selected columns
4. Add batch edit for multiple rows
5. Add custom column ordering

