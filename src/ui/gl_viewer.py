"""
OpenGL 3D Model Viewer Widget
Lightweight 3D mesh viewer for ModelFinder preview panel
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from PySide6 import QtCore, QtWidgets, QtOpenGLWidgets
from OpenGL.GL import *
from OpenGL.GLU import *
import trimesh


class GLViewer(QtOpenGLWidgets.QOpenGLWidget):
    """OpenGL widget for rendering 3D meshes"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.mesh = None
        self.rotation_x = 0
        self.rotation_y = 0
        self.zoom = -5.0
        self.last_pos = None
        
        # Enable mouse tracking for interactive rotation
        self.setMouseTracking(True)
        self.setMinimumSize(200, 200)
        
    def load_mesh(self, file_path: str | Path):
        """Load a 3D mesh from file with robust error handling"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                print(f"File not found: {file_path}")
                self.mesh = None
                self.update()
                return False
            
            # Check file size (skip very large files)
            file_size = file_path.stat().st_size
            if file_size > 100 * 1024 * 1024:  # 100MB limit
                print(f"File too large for preview: {file_size / (1024*1024):.1f}MB")
                self.mesh = None
                self.update()
                return False
            
            # Load mesh using trimesh with timeout protection
            try:
                self.mesh = trimesh.load(str(file_path), force='mesh')
            except Exception as load_error:
                print(f"Failed to load mesh with trimesh: {load_error}")
                self.mesh = None
                self.update()
                return False
            
            # Validate and process mesh
            if isinstance(self.mesh, trimesh.Trimesh):
                # Check if mesh has valid geometry
                if len(self.mesh.vertices) == 0 or len(self.mesh.faces) == 0:
                    print("Mesh has no geometry")
                    self.mesh = None
                    self.update()
                    return False
                
                # Center the mesh at origin
                try:
                    self.mesh.vertices -= self.mesh.centroid
                except Exception as e:
                    print(f"Failed to center mesh: {e}")
                
                # Scale to fit in view (normalize to unit cube)
                try:
                    extents = self.mesh.extents
                    if extents is not None and max(extents) > 0:
                        scale = 2.0 / max(extents)
                        self.mesh.vertices *= scale
                except Exception as e:
                    print(f"Failed to scale mesh: {e}")
            else:
                print("Loaded object is not a trimesh.Trimesh")
                self.mesh = None
                self.update()
                return False
            
            # Reset camera
            self.rotation_x = 0
            self.rotation_y = 0
            self.zoom = -5.0
            
            # Trigger repaint
            self.update()
            return True
            
        except Exception as e:
            print(f"Error loading mesh: {e}")
            self.mesh = None
            self.update()
            return False
    
    def clear_mesh(self):
        """Clear the current mesh"""
        self.mesh = None
        self.update()
    
    def initializeGL(self):
        """Initialize OpenGL context"""
        # Set background color (dark gray)
        glClearColor(0.2, 0.2, 0.2, 1.0)
        
        # Enable depth testing
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        
        # Enable lighting
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        
        # Set light properties
        glLightfv(GL_LIGHT0, GL_POSITION, [1.0, 1.0, 1.0, 0.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        
        # Set material properties
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.5, 0.5, 0.5, 1.0])
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 32.0)
        
        # Enable smooth shading
        glShadeModel(GL_SMOOTH)
        
        # Enable color material
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    
    def resizeGL(self, w: int, h: int):
        """Handle widget resize"""
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        # Set perspective projection
        aspect = w / h if h > 0 else 1.0
        gluPerspective(45.0, aspect, 0.1, 100.0)
        
        glMatrixMode(GL_MODELVIEW)
    
    def paintGL(self):
        """Render the 3D scene"""
        # Clear buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Reset modelview matrix
        glLoadIdentity()
        
        # Apply camera transformations
        glTranslatef(0.0, 0.0, self.zoom)
        glRotatef(self.rotation_x, 1.0, 0.0, 0.0)
        glRotatef(self.rotation_y, 0.0, 1.0, 0.0)
        
        # Draw the mesh if loaded
        if self.mesh and isinstance(self.mesh, trimesh.Trimesh):
            self._draw_mesh()
        else:
            # Draw placeholder text
            self._draw_placeholder()
        
        # Draw axis indicators
        self._draw_axes()
    
    def _draw_mesh(self):
        """Draw the loaded mesh"""
        if not self.mesh:
            return
            
        # Enable face culling for better performance
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        
        # Draw triangles
        glBegin(GL_TRIANGLES)
        for face in self.mesh.faces:
            # Get vertices and normals
            v0, v1, v2 = self.mesh.vertices[face]
            
            # Calculate face normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            normal = normal / (np.linalg.norm(normal) + 1e-10)
            
            # Set color based on normal direction (for visual interest)
            color = abs(normal)
            glColor3f(color[0] * 0.7 + 0.3, color[1] * 0.7 + 0.3, color[2] * 0.7 + 0.3)
            
            # Draw triangle
            glNormal3fv(normal)
            glVertex3fv(v0)
            glVertex3fv(v1)
            glVertex3fv(v2)
        glEnd()
        
        glDisable(GL_CULL_FACE)
    
    def _draw_axes(self):
        """Draw XYZ axes for reference"""
        glDisable(GL_LIGHTING)
        glLineWidth(2.0)
        
        glBegin(GL_LINES)
        # X axis (red)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(1.0, 0.0, 0.0)
        
        # Y axis (green)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 1.0, 0.0)
        
        # Z axis (blue)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 1.0)
        glEnd()
        
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)
    
    def _draw_placeholder(self):
        """Draw placeholder when no mesh is loaded"""
        glDisable(GL_LIGHTING)
        glColor3f(0.5, 0.5, 0.5)
        
        # Draw a simple wireframe cube as placeholder
        size = 1.0
        glBegin(GL_LINES)
        # Bottom face
        glVertex3f(-size, -size, -size)
        glVertex3f(size, -size, -size)
        
        glVertex3f(size, -size, -size)
        glVertex3f(size, -size, size)
        
        glVertex3f(size, -size, size)
        glVertex3f(-size, -size, size)
        
        glVertex3f(-size, -size, size)
        glVertex3f(-size, -size, -size)
        
        # Top face
        glVertex3f(-size, size, -size)
        glVertex3f(size, size, -size)
        
        glVertex3f(size, size, -size)
        glVertex3f(size, size, size)
        
        glVertex3f(size, size, size)
        glVertex3f(-size, size, size)
        
        glVertex3f(-size, size, size)
        glVertex3f(-size, size, -size)
        
        # Vertical edges
        glVertex3f(-size, -size, -size)
        glVertex3f(-size, size, -size)
        
        glVertex3f(size, -size, -size)
        glVertex3f(size, size, -size)
        
        glVertex3f(size, -size, size)
        glVertex3f(size, size, size)
        
        glVertex3f(-size, -size, size)
        glVertex3f(-size, size, size)
        glEnd()
        
        glEnable(GL_LIGHTING)
    
    def mousePressEvent(self, event):
        """Handle mouse press for rotation"""
        self.last_pos = event.pos()
    
    def mouseMoveEvent(self, event):
        """Handle mouse drag for rotation"""
        if event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            if self.last_pos is not None:
                dx = event.x() - self.last_pos.x()
                dy = event.y() - self.last_pos.y()
                
                self.rotation_y += dx * 0.5
                self.rotation_x += dy * 0.5
                
                self.last_pos = event.pos()
                self.update()
    
    def wheelEvent(self, event):
        """Handle mouse wheel for zoom"""
        delta = event.angleDelta().y() / 120.0
        self.zoom += delta * 0.5
        
        # Clamp zoom
        self.zoom = max(-50.0, min(-1.0, self.zoom))
        
        self.update()

