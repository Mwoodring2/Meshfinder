#!/usr/bin/env python3
"""
ModelFinder GPU Rendering Integration
Adds high-quality 3D preview capabilities to the ModelFinder UI
"""

import sys
import os
from pathlib import Path
from typing import Optional, Tuple
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSlider, QCheckBox, QComboBox, QGroupBox, QSpinBox,
    QDockWidget, QSizePolicy, QMessageBox
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSize
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor

try:
    from gpu_renderer import GPURenderer, HybridRenderer, RenderSettings
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("GPU renderer not available. Some features will be disabled.")

try:
    from PIL import Image
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class RenderThread(QThread):
    """Background thread for GPU rendering to prevent UI freezing"""
    
    finished_signal = Signal(object)  # Emits PIL Image
    error_signal = Signal(str)
    progress_signal = Signal(int)
    
    def __init__(self, renderer, file_path, settings):
        super().__init__()
        self.renderer = renderer
        self.file_path = file_path
        self.settings = settings
        
    def run(self):
        """Run rendering in background"""
        try:
            self.progress_signal.emit(25)
            
            # Update renderer settings
            self.renderer.settings = self.settings
            
            self.progress_signal.emit(50)
            
            # Render the model
            image = self.renderer.render_file(self.file_path)
            
            self.progress_signal.emit(100)
            
            if image:
                self.finished_signal.emit(image)
            else:
                self.error_signal.emit("Failed to render model")
                
        except Exception as e:
            self.error_signal.emit(str(e))


class Preview3DWidget(QWidget):
    """
    3D Preview widget with GPU-accelerated rendering.
    Can be embedded in the ModelFinder main window.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.current_file = None
        self.renderer = None
        self.render_thread = None
        self.current_image = None
        
        # Initialize renderer
        self._init_renderer()
        
        # Setup UI
        self._setup_ui()
        
    def _init_renderer(self):
        """Initialize the GPU renderer"""
        if not GPU_AVAILABLE:
            return
        
        try:
            # Try GPU renderer first
            self.settings = RenderSettings(
                width=512,
                height=512,
                samples=4,
                model_color=(0.7, 0.7, 0.75),
                metallic=0.15,
                roughness=0.4
            )
            
            self.renderer = HybridRenderer(prefer_gpu=True, settings=self.settings)
            
        except Exception as e:
            print(f"Failed to initialize renderer: {e}")
            self.renderer = None
    
    def _setup_ui(self):
        """Setup the preview widget UI"""
        layout = QVBoxLayout(self)
        
        # Preview area
        self.preview_label = QLabel()
        self.preview_label.setMinimumSize(512, 512)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("""
            QLabel {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #f0f0f2, stop: 1 #d8d8da
                );
                border: 2px solid #999;
                border-radius: 4px;
            }
        """)
        self.preview_label.setText("No model loaded\n\nSelect a 3D file to preview")
        layout.addWidget(self.preview_label)
        
        # Control panel
        controls_group = QGroupBox("Render Settings")
        controls_layout = QVBoxLayout()
        
        # Quality settings
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("Quality:"))
        
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["Preview (Fast)", "Normal", "High", "Ultra"])
        self.quality_combo.setCurrentIndex(1)
        self.quality_combo.currentIndexChanged.connect(self._on_quality_changed)
        quality_layout.addWidget(self.quality_combo)
        
        quality_layout.addStretch()
        controls_layout.addLayout(quality_layout)
        
        # Material settings
        material_layout = QHBoxLayout()
        
        material_layout.addWidget(QLabel("Metallic:"))
        self.metallic_slider = QSlider(Qt.Horizontal)
        self.metallic_slider.setRange(0, 100)
        self.metallic_slider.setValue(15)
        self.metallic_slider.valueChanged.connect(self._on_material_changed)
        material_layout.addWidget(self.metallic_slider)
        
        material_layout.addWidget(QLabel("Roughness:"))
        self.roughness_slider = QSlider(Qt.Horizontal)
        self.roughness_slider.setRange(0, 100)
        self.roughness_slider.setValue(40)
        self.roughness_slider.valueChanged.connect(self._on_material_changed)
        material_layout.addWidget(self.roughness_slider)
        
        controls_layout.addLayout(material_layout)
        
        # Display options
        options_layout = QHBoxLayout()
        
        self.auto_rotate_check = QCheckBox("Auto Rotate")
        self.auto_rotate_check.stateChanged.connect(self._on_auto_rotate_changed)
        options_layout.addWidget(self.auto_rotate_check)
        
        self.wireframe_check = QCheckBox("Wireframe")
        self.wireframe_check.stateChanged.connect(self._on_wireframe_changed)
        options_layout.addWidget(self.wireframe_check)
        
        self.shadows_check = QCheckBox("Shadows")
        self.shadows_check.setChecked(True)
        self.shadows_check.stateChanged.connect(self._on_shadows_changed)
        options_layout.addWidget(self.shadows_check)
        
        options_layout.addStretch()
        controls_layout.addLayout(options_layout)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.render_button = QPushButton("🎨 Render")
        self.render_button.clicked.connect(self._render_current)
        button_layout.addWidget(self.render_button)
        
        self.export_button = QPushButton("💾 Export Image")
        self.export_button.clicked.connect(self._export_image)
        self.export_button.setEnabled(False)
        button_layout.addWidget(self.export_button)
        
        self.reset_button = QPushButton("↺ Reset View")
        self.reset_button.clicked.connect(self._reset_view)
        button_layout.addWidget(self.reset_button)
        
        layout.addLayout(button_layout)
        
        # Status bar
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("QLabel { color: #666; }")
        layout.addWidget(self.status_label)
        
        # Auto-rotate timer
        self.rotate_timer = QTimer()
        self.rotate_timer.timeout.connect(self._rotate_model)
        self.rotation_angle = 0
    
    def load_file(self, file_path: str):
        """
        Load and preview a 3D model file.
        
        Args:
            file_path: Path to the 3D model file
        """
        if not self.renderer:
            self._show_no_renderer_message()
            return
        
        self.current_file = file_path
        self.status_label.setText(f"Loading: {Path(file_path).name}")
        
        # Start rendering
        self._render_current()
    
    def _render_current(self):
        """Render the current file"""
        if not self.current_file or not self.renderer:
            return
        
        # Disable controls during rendering
        self.render_button.setEnabled(False)
        self.status_label.setText("Rendering...")
        
        # Update settings
        self._update_render_settings()
        
        # Start render thread
        self.render_thread = RenderThread(
            self.renderer,
            self.current_file,
            self.settings
        )
        
        self.render_thread.finished_signal.connect(self._on_render_complete)
        self.render_thread.error_signal.connect(self._on_render_error)
        self.render_thread.progress_signal.connect(self._on_render_progress)
        
        self.render_thread.start()
    
    def _on_render_complete(self, image):
        """Handle completed render"""
        self.current_image = image
        
        # Convert PIL Image to QPixmap
        if PIL_AVAILABLE and image:
            # Convert to QImage
            image_array = np.array(image)
            height, width, channel = image_array.shape
            bytes_per_line = 4 * width
            
            if channel == 4:
                format = QImage.Format_RGBA8888
            else:
                format = QImage.Format_RGB888
            
            qimage = QImage(
                image_array.data,
                width,
                height,
                bytes_per_line,
                format
            )
            
            # Scale to fit label
            pixmap = QPixmap.fromImage(qimage)
            scaled_pixmap = pixmap.scaled(
                self.preview_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            self.preview_label.setPixmap(scaled_pixmap)
            
            # Update status
            file_name = Path(self.current_file).name
            self.status_label.setText(f"Rendered: {file_name} ({width}x{height})")
            
            # Enable export
            self.export_button.setEnabled(True)
        
        # Re-enable controls
        self.render_button.setEnabled(True)
    
    def _on_render_error(self, error_msg):
        """Handle render error"""
        self.status_label.setText(f"Error: {error_msg}")
        self.preview_label.setText(f"Failed to render model\n\n{error_msg}")
        self.render_button.setEnabled(True)
    
    def _on_render_progress(self, progress):
        """Update render progress"""
        self.status_label.setText(f"Rendering... {progress}%")
    
    def _update_render_settings(self):
        """Update render settings from UI controls"""
        # Quality settings
        quality_index = self.quality_combo.currentIndex()
        quality_settings = [
            (256, 256, 1),   # Preview
            (512, 512, 4),   # Normal
            (1024, 1024, 8), # High
            (2048, 2048, 16) # Ultra
        ]
        
        width, height, samples = quality_settings[quality_index]
        self.settings.width = width
        self.settings.height = height
        self.settings.samples = samples
        
        # Material settings
        self.settings.metallic = self.metallic_slider.value() / 100.0
        self.settings.roughness = self.roughness_slider.value() / 100.0
        
        # Display options
        self.settings.edge_thickness = 1.5 if self.wireframe_check.isChecked() else 0.0
        self.settings.use_shadows = self.shadows_check.isChecked()
        
        # Rotation
        if hasattr(self, 'rotation_angle'):
            self.settings.rotation = (0, self.rotation_angle, 0)
    
    def _on_quality_changed(self):
        """Handle quality change"""
        if self.current_file:
            self._render_current()
    
    def _on_material_changed(self):
        """Handle material property change"""
        if self.current_file and self.sender() == self.metallic_slider:
            # Only re-render on release for sliders
            if not self.metallic_slider.isSliderDown():
                self._render_current()
        elif self.current_file and self.sender() == self.roughness_slider:
            if not self.roughness_slider.isSliderDown():
                self._render_current()
    
    def _on_wireframe_changed(self):
        """Handle wireframe toggle"""
        if self.current_file:
            self._render_current()
    
    def _on_shadows_changed(self):
        """Handle shadows toggle"""
        if self.current_file:
            self._render_current()
    
    def _on_auto_rotate_changed(self):
        """Handle auto-rotate toggle"""
        if self.auto_rotate_check.isChecked():
            self.rotate_timer.start(50)  # 20 FPS
        else:
            self.rotate_timer.stop()
    
    def _rotate_model(self):
        """Rotate the model for auto-rotate"""
        self.rotation_angle = (self.rotation_angle + 2) % 360
        if self.current_file:
            self._render_current()
    
    def _reset_view(self):
        """Reset view to default"""
        self.rotation_angle = 0
        self.metallic_slider.setValue(15)
        self.roughness_slider.setValue(40)
        self.wireframe_check.setChecked(False)
        self.shadows_check.setChecked(True)
        
        if self.current_file:
            self._render_current()
    
    def _export_image(self):
        """Export the rendered image"""
        if not self.current_image:
            return
        
        from PySide6.QtWidgets import QFileDialog
        
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Export Rendered Image",
            f"{Path(self.current_file).stem}_render.png",
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
        )
        
        if file_name:
            self.current_image.save(file_name)
            self.status_label.setText(f"Exported: {Path(file_name).name}")
    
    def _show_no_renderer_message(self):
        """Show message when renderer is not available"""
        QMessageBox.information(
            self,
            "GPU Renderer Not Available",
            "The GPU renderer is not available.\n\n"
            "To enable GPU rendering, install ModernGL:\n"
            "pip install moderngl trimesh pillow\n\n"
            "The application will continue with basic functionality."
        )
    
    def cleanup(self):
        """Clean up resources"""
        if self.render_thread and self.render_thread.isRunning():
            self.render_thread.terminate()
            self.render_thread.wait()
        
        if self.renderer:
            self.renderer.cleanup()


def create_preview_dock(main_window) -> Optional[QDockWidget]:
    """
    Create a dockable 3D preview widget for the main window.
    
    Args:
        main_window: The main application window
        
    Returns:
        QDockWidget containing the preview widget, or None if not available
    """
    if not GPU_AVAILABLE:
        print("GPU renderer not available. Preview dock will not be created.")
        return None
    
    # Create dock widget
    dock = QDockWidget("3D Preview (GPU Accelerated)", main_window)
    dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
    
    # Create preview widget
    preview_widget = Preview3DWidget()
    dock.setWidget(preview_widget)
    
    # Set size hints
    preview_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
    preview_widget.setMinimumSize(400, 500)
    
    return dock


def integrate_with_main_window(main_window_class):
    """
    Monkey-patch the main window class to add GPU preview functionality.
    
    Args:
        main_window_class: The MainWindow class to patch
    """
    
    original_init = main_window_class.__init__
    
    def new_init(self, *args, **kwargs):
        # Call original init
        original_init(self, *args, **kwargs)
        
        # Add preview dock
        if GPU_AVAILABLE:
            self.preview_dock = create_preview_dock(self)
            if self.preview_dock:
                self.addDockWidget(Qt.RightDockWidgetArea, self.preview_dock)
                
                # Connect to table selection
                if hasattr(self, 'file_table'):
                    self.file_table.itemSelectionChanged.connect(
                        lambda: _on_file_selected(self)
                    )
    
    def _on_file_selected(main_window):
        """Handle file selection in main window"""
        if not hasattr(main_window, 'preview_dock'):
            return
        
        current_row = main_window.file_table.currentRow()
        if current_row >= 0:
            # Get file path from table (assuming it's in column 1)
            file_path = main_window.file_table.item(current_row, 1).text()
            
            # Load in preview
            preview_widget = main_window.preview_dock.widget()
            if preview_widget and Path(file_path).exists():
                preview_widget.load_file(file_path)
    
    # Replace init
    main_window_class.__init__ = new_init


# Usage example
if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication, QMainWindow
    
    app = QApplication(sys.argv)
    
    # Create test window
    window = QMainWindow()
    window.setWindowTitle("GPU Preview Test")
    window.resize(1200, 800)
    
    # Create and set preview widget
    preview = Preview3DWidget()
    window.setCentralWidget(preview)
    
    # Test with a file
    test_file = "test_model.stl"
    if Path(test_file).exists():
        preview.load_file(test_file)
    
    window.show()
    sys.exit(app.exec())
