"""
Human-in-the-loop review panel for file migration.

Provides interactive UI for reviewing and editing migration proposals.
"""
from PySide6 import QtCore, QtGui, QtWidgets
from pathlib import Path
from ..utils.naming import canonical_name, extract_project_number
from ..ml import DEFAULT_THRESHOLD


class ReviewPanel(QtWidgets.QWidget):
    """
    Review panel for editing file metadata and previewing proposed names.
    
    Signals:
        fields_changed: Emitted when any field is edited
        apply_requested: Emitted when Apply to Selected is clicked
    """
    
    fields_changed = QtCore.Signal(dict)  # {project_number, project_name, part_name}
    apply_requested = QtCore.Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_ext = ""
        self._build_ui()
    
    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Title
        title = QtWidgets.QLabel("📝 Review & Edit Metadata")
        title.setStyleSheet("font-weight: bold; font-size: 14px; padding: 4px;")
        layout.addWidget(title)
        
        # Editable fields
        form = QtWidgets.QFormLayout()
        form.setSpacing(8)
        
        self.project_number_edit = QtWidgets.QLineEdit()
        self.project_number_edit.setPlaceholderText("ABC-1234 or PRJ001")
        self.project_number_edit.textChanged.connect(self._on_field_changed)
        form.addRow("Project #:", self.project_number_edit)
        
        self.project_name_edit = QtWidgets.QLineEdit()
        self.project_name_edit.setPlaceholderText("Star Wars Droid")
        self.project_name_edit.textChanged.connect(self._on_field_changed)
        form.addRow("Project Name:", self.project_name_edit)
        
        self.part_name_edit = QtWidgets.QLineEdit()
        self.part_name_edit.setPlaceholderText("r2d2_body")
        self.part_name_edit.textChanged.connect(self._on_field_changed)
        form.addRow("Part Name:", self.part_name_edit)
        
        layout.addLayout(form)
        
        # Confidence badge
        conf_layout = QtWidgets.QHBoxLayout()
        conf_layout.addWidget(QtWidgets.QLabel("Confidence:"))
        self.conf_badge = QtWidgets.QLabel("--")
        self.conf_badge.setStyleSheet(
            "padding: 4px 8px; border-radius: 3px; background: #666; color: white; font-weight: bold;"
        )
        conf_layout.addWidget(self.conf_badge)
        conf_layout.addStretch()
        layout.addLayout(conf_layout)
        
        # Live preview
        preview_box = QtWidgets.QGroupBox("📋 Proposed Name Preview")
        pv_layout = QtWidgets.QVBoxLayout(preview_box)
        self.preview_label = QtWidgets.QLabel("projectnumber_projectname_partname.ext")
        self.preview_label.setStyleSheet(
            "font-family: 'Consolas', 'Courier New', monospace; "
            "padding: 8px; background: #f0f0f0; border: 1px solid #ccc; border-radius: 3px;"
        )
        self.preview_label.setWordWrap(True)
        self.preview_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        pv_layout.addWidget(self.preview_label)
        layout.addWidget(preview_box)
        
        # Action buttons
        btn_layout = QtWidgets.QVBoxLayout()
        btn_layout.setSpacing(6)
        
        self.btn_apply = QtWidgets.QPushButton("✓ Apply to Selected")
        self.btn_apply.setToolTip("Apply these fields to the selected file (Enter)")
        self.btn_apply.clicked.connect(self.apply_requested)
        self.btn_apply.setShortcut(QtCore.Qt.Key.Key_Return)
        btn_layout.addWidget(self.btn_apply)
        
        self.btn_dry_run = QtWidgets.QPushButton("🔍 Dry-run Rename")
        self.btn_dry_run.setToolTip("Preview the rename operation")
        self.btn_dry_run.clicked.connect(self._on_dry_run)
        btn_layout.addWidget(self.btn_dry_run)
        
        self.btn_migrate = QtWidgets.QPushButton("➡️ Migrate Selected")
        self.btn_migrate.setToolTip("Execute migration for selected files")
        self.btn_migrate.clicked.connect(self._on_migrate)
        btn_layout.addWidget(self.btn_migrate)
        
        self.btn_quarantine = QtWidgets.QPushButton("⚠️ Send to Quarantine")
        self.btn_quarantine.setToolTip("Mark selected files for manual review")
        self.btn_quarantine.clicked.connect(self._on_quarantine)
        btn_layout.addWidget(self.btn_quarantine)
        
        layout.addLayout(btn_layout)
        layout.addStretch()
        
        # Keyboard shortcuts
        # Ctrl+L to focus Part Name field
        shortcut_focus_part = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+L"), self)
        shortcut_focus_part.activated.connect(lambda: self.part_name_edit.setFocus())
        
        # Tab to jump to next field
        self.project_number_edit.installEventFilter(self)
        self.project_name_edit.installEventFilter(self)
        self.part_name_edit.installEventFilter(self)
    
    def eventFilter(self, obj, event):
        """Handle Tab key for jumping to next field."""
        if event.type() == QtCore.QEvent.Type.KeyPress:
            if event.key() == QtCore.Qt.Key.Key_Tab:
                if obj == self.project_number_edit:
                    self.project_name_edit.setFocus()
                    return True
                elif obj == self.project_name_edit:
                    self.part_name_edit.setFocus()
                    return True
        return super().eventFilter(obj, event)
    
    def set_fields(self, project_number: str, project_name: str, part_name: str, 
                   confidence: float = None, ext: str = ""):
        """
        Set the fields from current row data.
        
        Args:
            project_number: Project number/ID
            project_name: Project name
            part_name: Part name
            confidence: Confidence score (0-1)
            ext: File extension
        """
        self.current_ext = ext.lstrip(".")
        
        # Block signals during update to avoid triggering preview
        self.project_number_edit.blockSignals(True)
        self.project_name_edit.blockSignals(True)
        self.part_name_edit.blockSignals(True)
        
        self.project_number_edit.setText(project_number or "")
        self.project_name_edit.setText(project_name or "")
        self.part_name_edit.setText(part_name or "")
        
        self.project_number_edit.blockSignals(False)
        self.project_name_edit.blockSignals(False)
        self.part_name_edit.blockSignals(False)
        
        # Update confidence badge
        if confidence is not None:
            self._update_confidence_badge(confidence)
        else:
            self.conf_badge.setText("--")
            self.conf_badge.setStyleSheet(
                "padding: 4px 8px; border-radius: 3px; background: #666; color: white; font-weight: bold;"
            )
        
        # Update preview
        self._update_preview()
    
    def _update_confidence_badge(self, confidence: float):
        """Update confidence badge with color coding."""
        pct = int(confidence * 100)
        self.conf_badge.setText(f"{pct}%")
        
        if confidence >= DEFAULT_THRESHOLD:
            # High confidence: green
            color = "#2ecc71"
        elif confidence >= 0.7:
            # Medium confidence: yellow
            color = "#f39c12"
        else:
            # Low confidence: red
            color = "#e74c3c"
        
        self.conf_badge.setStyleSheet(
            f"padding: 4px 8px; border-radius: 3px; background: {color}; color: white; font-weight: bold;"
        )
    
    def _on_field_changed(self):
        """Handle field changes and update preview."""
        self._update_preview()
        self.fields_changed.emit(self.get_fields())
    
    def _update_preview(self):
        """Update the proposed name preview."""
        fields = self.get_fields()
        if all([fields["project_number"], fields["project_name"], fields["part_name"], self.current_ext]):
            proposed = canonical_name(
                fields["project_number"],
                fields["project_name"],
                fields["part_name"],
                self.current_ext
            )
            self.preview_label.setText(proposed)
        else:
            self.preview_label.setText("(incomplete - fill all fields)")
    
    def get_fields(self) -> dict:
        """Get current field values."""
        return {
            "project_number": self.project_number_edit.text().strip(),
            "project_name": self.project_name_edit.text().strip(),
            "part_name": self.part_name_edit.text().strip()
        }
    
    def get_proposed_name(self) -> str:
        """Get the current proposed name."""
        return self.preview_label.text()
    
    def _on_dry_run(self):
        """Handle dry-run rename button."""
        # Emit signal to parent to handle dry-run
        QtWidgets.QMessageBox.information(
            self, "Dry-run", 
            f"Would rename to:\n{self.get_proposed_name()}\n\n(Full dry-run coming soon)"
        )
    
    def _on_migrate(self):
        """Handle migrate button."""
        # Emit signal to parent to handle migration
        QtWidgets.QMessageBox.information(
            self, "Migrate", 
            "Migration feature will be handled by main window"
        )
    
    def _on_quarantine(self):
        """Handle quarantine button."""
        # Emit signal to parent to handle quarantine
        QtWidgets.QMessageBox.information(
            self, "Quarantine", 
            "Quarantine feature will be handled by main window"
        )

