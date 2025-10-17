"""
Enhanced table model for file review with migration metadata.
"""
import time
from PySide6 import QtCore, QtGui


class ReviewTableModel(QtCore.QAbstractTableModel):
    """
    Table model with migration metadata columns.
    
    Columns: Name, Extension, Size (MB), Modified, Tags, Path,
             Project #, Project Name, Part Name, Conf., Proposed Name, Status
    """
    
    headers = [
        "Name", "Extension", "Size (MB)", "Modified", "Tags", "Path",
        "Project #", "Project Name", "Part Name", "Conf.", "Proposed Name", "Status"
    ]
    
    def __init__(self):
        super().__init__()
        self.rows = []
    
    def set_rows(self, rows):
        """
        Set table rows.
        
        Expected row format: (path, name, ext, size, mtime, tags,
                             project_number, project_name, part_name,
                             type_conf, proposed_name, status)
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
        
        # Unpack row data
        path, name, ext, size, mtime, tags = row[:6]
        project_number = row[6] if len(row) > 6 else None
        project_name = row[7] if len(row) > 7 else None
        part_name = row[8] if len(row) > 8 else None
        type_conf = row[9] if len(row) > 9 else None
        proposed_name = row[10] if len(row) > 10 else None
        status = row[11] if len(row) > 11 else "discovered"
        
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if col == 0: return name
            if col == 1: return ext
            if col == 2: return f"{size/1024/1024:.2f}"
            if col == 3: return time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))
            if col == 4: return tags or ""
            if col == 5: return path
            if col == 6: return project_number or ""
            if col == 7: return project_name or ""
            if col == 8: return part_name or ""
            if col == 9: 
                if type_conf is not None:
                    return f"{int(type_conf * 100)}%"
                return ""
            if col == 10: return proposed_name or ""
            if col == 11: return status or "discovered"
        
        # Background color for confidence column
        if role == QtCore.Qt.ItemDataRole.BackgroundRole and col == 9:
            if type_conf is not None:
                if type_conf >= 0.92:
                    return QtGui.QColor("#d4edda")  # Light green
                elif type_conf >= 0.7:
                    return QtGui.QColor("#fff3cd")  # Light yellow
                else:
                    return QtGui.QColor("#f8d7da")  # Light red
        
        # Badge styling for status column
        if role == QtCore.Qt.ItemDataRole.BackgroundRole and col == 11:
            if status == "migrated":
                return QtGui.QColor("#d4edda")  # Green
            elif status == "quarantined":
                return QtGui.QColor("#f8d7da")  # Red
            elif status == "staged":
                return QtGui.QColor("#cfe2ff")  # Blue
        
        # Make certain columns editable
        if role == QtCore.Qt.ItemDataRole.EditRole:
            if col == 4: return tags or ""  # Tags
            if col == 6: return project_number or ""
            if col == 7: return project_name or ""
            if col == 8: return part_name or ""
        
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
            
            # Update the row data
            row = list(self.rows[row_idx])
            
            if col == 4:  # Tags
                row[5] = str(value)
            elif col == 6:  # Project #
                if len(row) > 6:
                    row[6] = str(value)
            elif col == 7:  # Project Name
                if len(row) > 7:
                    row[7] = str(value)
            elif col == 8:  # Part Name
                if len(row) > 8:
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
            "path": row[0],
            "name": row[1],
            "ext": row[2],
            "size": row[3],
            "mtime": row[4],
            "tags": row[5] if len(row) > 5 else "",
            "project_number": row[6] if len(row) > 6 else None,
            "project_name": row[7] if len(row) > 7 else None,
            "part_name": row[8] if len(row) > 8 else None,
            "type_conf": row[9] if len(row) > 9 else None,
            "proposed_name": row[10] if len(row) > 10 else None,
            "status": row[11] if len(row) > 11 else "discovered"
        }

