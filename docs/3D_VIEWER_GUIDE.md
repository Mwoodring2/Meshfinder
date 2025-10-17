# 3D Viewer Integration Guide

## Overview

ModelFinder now includes a real-time OpenGL 3D viewer in the preview panel. This allows you to visualize 3D models directly in the application without opening external software.

## Features

### ✨ Core Capabilities
- **Real-time 3D Rendering** - OpenGL-based mesh visualization
- **Multi-format Support** - STL, OBJ, FBX, PLY, GLTF/GLB, and more (via trimesh)
- **Interactive Camera** - Mouse controls for rotation and zoom
- **Auto-centering** - Meshes are automatically centered and scaled to fit
- **Lighting & Shading** - Professional lighting with smooth shading
- **Color-coded Normals** - Visual feedback based on surface orientation
- **XYZ Axes** - Reference axes for orientation

### 🎮 Mouse Controls
- **Left-Click + Drag** - Rotate the model
- **Mouse Wheel** - Zoom in/out
- **Auto-fit** - Models are automatically scaled to fit the viewport

### 🎨 Visual Features
- **Smooth Shading** - OpenGL smooth shading for better appearance
- **Dynamic Lighting** - Directional light with ambient, diffuse, and specular components
- **Face Culling** - Back-face culling for better performance
- **Depth Testing** - Proper depth ordering

## Technical Details

### Dependencies
```txt
PySide6>=6.6        # Qt framework
PyOpenGL            # OpenGL bindings
PyOpenGL-accelerate # Performance acceleration
trimesh             # 3D mesh loading
numpy               # Numerical operations
```

### Architecture
- **GLViewer Widget** (`src/ui/gl_viewer.py`)
  - Extends `QOpenGLWidget` for Qt integration
  - Uses trimesh for mesh loading
  - Implements custom OpenGL rendering pipeline

### Supported File Formats
The viewer supports all formats that trimesh can load:
- **STL** - Stereolithography (binary and ASCII)
- **OBJ** - Wavefront OBJ
- **PLY** - Polygon File Format
- **GLB/GLTF** - GL Transmission Format
- **FBX** - Autodesk FBX (requires FBX SDK)
- **3MF** - 3D Manufacturing Format
- **OFF** - Object File Format
- **And many more...**

## Usage

### In Main Application
1. **Select a file** in the table
2. **3D preview loads automatically** in the right panel
3. **Interact with the model**:
   - Drag to rotate
   - Scroll to zoom
   - View from any angle

### API Usage
```python
from src.ui.gl_viewer import GLViewer

# Create viewer widget
viewer = GLViewer()

# Load a mesh
success = viewer.load_mesh("path/to/model.stl")

# Clear the viewer
viewer.clear_mesh()
```

## Performance Considerations

### Optimization Strategies
- **Face Culling** - Only renders front-facing triangles
- **Lazy Loading** - Meshes load only when selected
- **Auto-scaling** - Large meshes are normalized to unit scale
- **Depth Testing** - Efficient z-buffer operations

### Mesh Size Recommendations
- **Small meshes** (< 100K triangles): Instant rendering
- **Medium meshes** (100K - 500K triangles): Smooth performance
- **Large meshes** (> 500K triangles): May have slight loading delay
- **Very large meshes** (> 1M triangles): Consider LOD implementation

## Future Enhancements

### Planned Features
- [ ] **Wireframe mode** - Toggle between solid and wireframe
- [ ] **Material/Texture support** - Display textures and materials
- [ ] **Multiple meshes** - View assemblies
- [ ] **Measurement tools** - Distance and angle measurements
- [ ] **Export screenshots** - Save preview images
- [ ] **Animation support** - Play animated models
- [ ] **Level of Detail (LOD)** - Simplify large meshes for performance
- [ ] **Screenshot export** - Save thumbnails

### Advanced Features (Future)
- [ ] **Ray tracing** - High-quality rendering
- [ ] **PBR Materials** - Physically-based rendering
- [ ] **Environment maps** - HDR lighting
- [ ] **Cross-sections** - Cut planes for inspection
- [ ] **Analysis overlays** - Show normals, UVs, etc.

## Troubleshooting

### Common Issues

**Q: Viewer shows a wireframe cube instead of my model**
- A: The mesh failed to load. Check:
  - File format is supported
  - File is not corrupted
  - File path is correct

**Q: Model appears black**
- A: Lighting issue. Try rotating the model to see if it's a normal direction problem.

**Q: Performance is slow**
- A: Large mesh. Consider:
  - Reducing mesh complexity
  - Closing other OpenGL applications
  - Updating graphics drivers

**Q: Viewer not available**
- A: Missing dependencies. Install:
  ```bash
  pip install PyOpenGL PyOpenGL-accelerate trimesh
  ```

### Fallback Mode
If OpenGL is not available, the viewer automatically falls back to a text label showing the filename. This ensures the application works even on systems without OpenGL support.

## Credits
- **OpenGL** - 3D graphics API
- **Trimesh** - Python 3D mesh library
- **PySide6** - Qt for Python
- **PyOpenGL** - Python OpenGL bindings

