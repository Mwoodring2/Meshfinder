#!/usr/bin/env python3
"""
GPU-Accelerated 3D Model Renderer for ModelFinder
Uses ModernGL for hardware-accelerated rendering with proper shading and anti-aliasing
Provides Windows Explorer-quality previews for STL, OBJ, and other 3D formats
"""

import sys
import math
import struct
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass

try:
    import moderngl
    import trimesh
    from PIL import Image
    MODERNGL_AVAILABLE = True
except ImportError:
    MODERNGL_AVAILABLE = False
    moderngl = None
    trimesh = None
    Image = None
    print("Warning: ModernGL not installed. GPU rendering disabled.")
    print("Install with: pip install moderngl trimesh pillow")

# Shader sources for professional-quality rendering
VERTEX_SHADER = """
#version 330 core

in vec3 in_position;
in vec3 in_normal;

out vec3 frag_position;
out vec3 frag_normal;
out vec3 view_position;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 normal_matrix;

void main() {
    vec4 world_position = model * vec4(in_position, 1.0);
    frag_position = world_position.xyz;
    frag_normal = normalize(mat3(normal_matrix) * in_normal);
    view_position = (view * world_position).xyz;
    gl_Position = projection * view * world_position;
}
"""

FRAGMENT_SHADER = """
#version 330 core

in vec3 frag_position;
in vec3 frag_normal;
in vec3 view_position;

out vec4 out_color;

// Material properties
uniform vec3 material_color;
uniform float material_metallic;
uniform float material_roughness;
uniform float material_ambient;

// Lighting
uniform vec3 light_positions[3];
uniform vec3 light_colors[3];
uniform float light_intensities[3];
uniform vec3 camera_position;

// Options
uniform bool use_smooth_shading;
uniform bool use_shadows;
uniform float edge_thickness;

// Improved lighting calculation with PBR-like shading
vec3 calculate_lighting(vec3 normal, vec3 view_dir, vec3 base_color) {
    vec3 total_light = vec3(0.0);
    
    // Ambient lighting with hemisphere lighting
    vec3 up = vec3(0.0, 1.0, 0.0);
    float hemisphere = dot(normal, up) * 0.5 + 0.5;
    vec3 ambient = mix(vec3(0.05, 0.05, 0.08), vec3(0.15, 0.15, 0.18), hemisphere);
    ambient *= material_ambient;
    total_light += ambient * base_color;
    
    // Process each light source
    for (int i = 0; i < 3; i++) {
        vec3 light_dir = normalize(light_positions[i] - frag_position);
        float distance = length(light_positions[i] - frag_position);
        
        // Diffuse lighting (Lambert)
        float diff = max(dot(normal, light_dir), 0.0);
        
        // Specular lighting (Blinn-Phong with roughness)
        vec3 halfway = normalize(light_dir + view_dir);
        float spec_strength = max(dot(normal, halfway), 0.0);
        float shininess = mix(256.0, 4.0, material_roughness);
        float spec = pow(spec_strength, shininess);
        
        // Attenuation
        float attenuation = light_intensities[i] / (1.0 + 0.09 * distance + 0.032 * distance * distance);
        
        // Combine diffuse and specular
        vec3 diffuse = diff * light_colors[i] * base_color;
        vec3 specular = spec * light_colors[i] * mix(vec3(0.04), base_color, material_metallic);
        
        // Fresnel effect (rim lighting)
        float fresnel = pow(1.0 - max(dot(normal, view_dir), 0.0), 2.0);
        specular += fresnel * light_colors[i] * 0.1;
        
        total_light += (diffuse + specular) * attenuation;
    }
    
    return total_light;
}

// Edge detection for wireframe-like effect
float edge_factor() {
    vec3 d = fwidth(view_position);
    vec3 a3 = smoothstep(vec3(0.0), d * edge_thickness, view_position);
    return min(min(a3.x, a3.y), a3.z);
}

void main() {
    vec3 normal = normalize(frag_normal);
    
    // Flip normal if facing away (two-sided lighting)
    vec3 view_dir = normalize(camera_position - frag_position);
    if (dot(normal, view_dir) < 0.0) {
        normal = -normal;
    }
    
    // Calculate lighting
    vec3 color = calculate_lighting(normal, view_dir, material_color);
    
    // Apply gamma correction for more realistic appearance
    color = pow(color, vec3(1.0 / 2.2));
    
    // Optional edge highlighting
    if (edge_thickness > 0.0) {
        float edge = edge_factor();
        color = mix(vec3(0.1), color, edge);
    }
    
    out_color = vec4(color, 1.0);
}
"""

# Simplified shaders for fallback/preview mode
SIMPLE_VERTEX_SHADER = """
#version 330 core
in vec3 in_position;
in vec3 in_normal;
out vec3 v_normal;
uniform mat4 mvp;
void main() {
    v_normal = in_normal;
    gl_Position = mvp * vec4(in_position, 1.0);
}
"""

SIMPLE_FRAGMENT_SHADER = """
#version 330 core
in vec3 v_normal;
out vec4 out_color;
uniform vec3 light_dir;
uniform vec3 model_color;
void main() {
    float light = max(dot(normalize(v_normal), light_dir), 0.0) * 0.8 + 0.2;
    out_color = vec4(model_color * light, 1.0);
}
"""

@dataclass
class RenderSettings:
    """Configuration for rendering quality and style"""
    width: int = 512
    height: int = 512
    samples: int = 4  # Anti-aliasing samples
    background_color: Tuple[float, float, float] = (0.95, 0.95, 0.97)
    model_color: Tuple[float, float, float] = (0.7, 0.7, 0.75)
    metallic: float = 0.1
    roughness: float = 0.4
    use_shadows: bool = True
    edge_thickness: float = 0.0  # Set > 0 for wireframe overlay
    auto_center: bool = True
    auto_scale: bool = True
    rotation: Tuple[float, float, float] = (0, 0, 0)  # Euler angles in degrees


class GPURenderer:
    """Hardware-accelerated 3D model renderer using ModernGL"""
    
    def __init__(self, settings: Optional[RenderSettings] = None):
        """
        Initialize the GPU renderer.
        
        Args:
            settings: Render configuration settings
        """
        if not MODERNGL_AVAILABLE:
            raise RuntimeError("ModernGL is not installed. Please install with: pip install moderngl")
        
        self.settings = settings or RenderSettings()
        self.ctx = None
        self.program = None
        self.simple_program = None
        self.fbo = None
        self.mesh_cache = {}
        
        # Initialize OpenGL context
        self._init_context()
        
    def _init_context(self):
        """Initialize ModernGL context and shaders"""
        try:
            # Create standalone context (no window needed)
            self.ctx = moderngl.create_context(standalone=True, backend='egl')
        except:
            try:
                # Fallback to different backend
                self.ctx = moderngl.create_context(standalone=True)
            except Exception as e:
                raise RuntimeError(f"Failed to create OpenGL context: {e}")
        
        # Compile shaders
        try:
            self.program = self.ctx.program(
                vertex_shader=VERTEX_SHADER,
                fragment_shader=FRAGMENT_SHADER
            )
        except Exception as e:
            print(f"Failed to compile advanced shaders: {e}")
            # Fallback to simple shaders
            self.program = self.ctx.program(
                vertex_shader=SIMPLE_VERTEX_SHADER,
                fragment_shader=SIMPLE_FRAGMENT_SHADER
            )
        
        # Create framebuffer for offscreen rendering
        self._create_framebuffer()
        
    def _create_framebuffer(self):
        """Create framebuffer for offscreen rendering with MSAA"""
        width = self.settings.width
        height = self.settings.height
        samples = self.settings.samples
        
        # Create multisampled framebuffer for anti-aliasing
        if samples > 1:
            # Create MSAA render targets
            color_rbo = self.ctx.renderbuffer_multisampled(
                (width, height), samples, dtype='f4'
            )
            depth_rbo = self.ctx.depth_renderbuffer_multisampled(
                (width, height), samples
            )
            self.msaa_fbo = self.ctx.framebuffer(
                color_attachments=[color_rbo],
                depth_attachment=depth_rbo
            )
            
            # Create resolve framebuffer
            color_tex = self.ctx.texture((width, height), 4, dtype='f4')
            depth_tex = self.ctx.depth_texture((width, height))
            self.fbo = self.ctx.framebuffer(
                color_attachments=[color_tex],
                depth_attachment=depth_tex
            )
        else:
            # Single-sampled framebuffer
            color_tex = self.ctx.texture((width, height), 4, dtype='f4')
            depth_tex = self.ctx.depth_texture((width, height))
            self.fbo = self.ctx.framebuffer(
                color_attachments=[color_tex],
                depth_attachment=depth_tex
            )
            self.msaa_fbo = None
    
    def load_mesh(self, file_path: str):
        """
        Load a 3D mesh from file.
        
        Args:
            file_path: Path to the 3D model file
            
        Returns:
            Trimesh object or None if loading fails
        """
        if not MODERNGL_AVAILABLE or trimesh is None:
            return None
            
        # Check cache first
        if file_path in self.mesh_cache:
            return self.mesh_cache[file_path]
        
        try:
            # Load mesh with trimesh
            mesh = trimesh.load(file_path, force='mesh')
            
            # Convert to single mesh if it's a scene
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            
            # Ensure we have a valid mesh
            if not isinstance(mesh, trimesh.Trimesh):
                mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
            
            # Fix mesh issues
            mesh.fix_normals()
            mesh.remove_degenerate_faces()
            mesh.remove_duplicate_faces()
            mesh.remove_unreferenced_vertices()
            
            # Cache the mesh
            self.mesh_cache[file_path] = mesh
            
            return mesh
            
        except Exception as e:
            print(f"Failed to load mesh {file_path}: {e}")
            return None
    
    def _prepare_mesh_data(self, mesh) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare mesh data for GPU upload.
        
        Args:
            mesh: Trimesh object
            
        Returns:
            Tuple of (vertices, normals) as numpy arrays
        """
        # Center and scale mesh if requested
        if self.settings.auto_center:
            mesh.vertices -= mesh.centroid
        
        if self.settings.auto_scale:
            scale = 2.0 / np.max(mesh.extents)
            mesh.vertices *= scale
        
        # Apply rotation if specified
        if any(self.settings.rotation) and trimesh is not None:
            rotation_matrix = trimesh.transformations.euler_matrix(
                math.radians(self.settings.rotation[0]),
                math.radians(self.settings.rotation[1]),
                math.radians(self.settings.rotation[2])
            )
            mesh.apply_transform(rotation_matrix)
        
        # Get vertices and normals for all faces
        vertices = mesh.vertices[mesh.faces].reshape(-1, 3)
        
        # Calculate face normals if vertex normals not available
        if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals.shape[0] == mesh.vertices.shape[0]:
            normals = mesh.vertex_normals[mesh.faces].reshape(-1, 3)
        else:
            # Use face normals repeated for each vertex
            face_normals = mesh.face_normals
            normals = np.repeat(face_normals, 3, axis=0)
        
        return vertices.astype('f4'), normals.astype('f4')
    
    def _create_matrices(self, aspect_ratio: float) -> Dict[str, np.ndarray]:
        """
        Create transformation matrices for rendering.
        
        Args:
            aspect_ratio: Width/height ratio
            
        Returns:
            Dictionary of transformation matrices
        """
        # View matrix (camera position)
        eye = np.array([0, 0, 3], dtype='f4')
        center = np.array([0, 0, 0], dtype='f4')
        up = np.array([0, 1, 0], dtype='f4')
        
        view = self._look_at(eye, center, up)
        
        # Projection matrix
        fov = math.radians(45)
        projection = self._perspective(fov, aspect_ratio, 0.1, 100.0)
        
        # Model matrix
        model = np.identity(4, dtype='f4')
        
        # Normal matrix (inverse transpose of model-view)
        mv = np.dot(view, model)
        normal_matrix = np.linalg.inv(mv[:3, :3]).T
        normal_matrix = np.pad(normal_matrix, ((0, 1), (0, 1)), constant_values=0)
        normal_matrix[3, 3] = 1
        
        return {
            'model': model,
            'view': view,
            'projection': projection,
            'normal_matrix': normal_matrix,
            'mvp': np.dot(np.dot(projection, view), model)
        }
    
    def _look_at(self, eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
        """Create a look-at view matrix"""
        z = eye - center
        z = z / np.linalg.norm(z)
        
        x = np.cross(up, z)
        x = x / np.linalg.norm(x)
        
        y = np.cross(z, x)
        
        mat = np.identity(4, dtype='f4')
        mat[:3, 0] = x
        mat[:3, 1] = y
        mat[:3, 2] = z
        mat[:3, 3] = eye
        
        return np.linalg.inv(mat)
    
    def _perspective(self, fov: float, aspect: float, near: float, far: float) -> np.ndarray:
        """Create a perspective projection matrix"""
        f = 1.0 / math.tan(fov / 2.0)
        
        mat = np.zeros((4, 4), dtype='f4')
        mat[0, 0] = f / aspect
        mat[1, 1] = f
        mat[2, 2] = (far + near) / (near - far)
        mat[2, 3] = (2 * far * near) / (near - far)
        mat[3, 2] = -1
        
        return mat
    
    def render_to_image(self, mesh) -> Image.Image:
        """
        Render a mesh to a PIL Image with GPU acceleration.
        
        Args:
            mesh: Trimesh object to render
            
        Returns:
            PIL Image with rendered mesh
        """
        # Prepare mesh data
        vertices, normals = self._prepare_mesh_data(mesh)
        
        # Create vertex buffer object
        vbo_data = np.hstack([vertices, normals]).astype('f4')
        vbo = self.ctx.buffer(vbo_data.tobytes())
        
        # Create vertex array object
        vao = self.ctx.vertex_array(
            self.program,
            [(vbo, '3f 3f', 'in_position', 'in_normal')]
        )
        
        # Set up rendering target
        if self.msaa_fbo:
            self.msaa_fbo.use()
        else:
            self.fbo.use()
        
        # Clear framebuffer
        bg = self.settings.background_color
        self.ctx.clear(bg[0], bg[1], bg[2], 1.0)
        
        # Enable depth testing and face culling
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
        
        # Calculate matrices
        aspect = self.settings.width / self.settings.height
        matrices = self._create_matrices(aspect)
        
        # Set uniforms
        if 'model' in self.program:
            self.program['model'].write(matrices['model'].T.astype('f4').tobytes())
            self.program['view'].write(matrices['view'].T.astype('f4').tobytes())
            self.program['projection'].write(matrices['projection'].T.astype('f4').tobytes())
            self.program['normal_matrix'].write(matrices['normal_matrix'].T.astype('f4').tobytes())
            
            # Material properties
            self.program['material_color'].value = self.settings.model_color
            self.program['material_metallic'].value = self.settings.metallic
            self.program['material_roughness'].value = self.settings.roughness
            self.program['material_ambient'].value = 0.3
            
            # Lighting setup (3-point lighting)
            light_positions = [
                (2.0, 2.0, 2.0),   # Key light
                (-2.0, 1.0, 1.0),  # Fill light
                (0.0, -2.0, 3.0)   # Rim light
            ]
            light_colors = [
                (1.0, 1.0, 0.95),  # Warm white
                (0.5, 0.5, 0.6),   # Cool gray
                (0.3, 0.3, 0.4)    # Blue rim
            ]
            light_intensities = [1.0, 0.5, 0.3]
            
            for i, (pos, color, intensity) in enumerate(zip(light_positions, light_colors, light_intensities)):
                self.program[f'light_positions[{i}]'].value = pos
                self.program[f'light_colors[{i}]'].value = color
                self.program[f'light_intensities[{i}]'].value = intensity
            
            self.program['camera_position'].value = (0, 0, 3)
            self.program['edge_thickness'].value = self.settings.edge_thickness
        else:
            # Simple shader uniforms
            self.program['mvp'].write(matrices['mvp'].T.astype('f4').tobytes())
            self.program['light_dir'].value = (0.5, 0.5, 0.5)
            self.program['model_color'].value = self.settings.model_color
        
        # Render the mesh
        vao.render(moderngl.TRIANGLES)
        
        # Resolve MSAA if enabled
        if self.msaa_fbo:
            self.ctx.copy_framebuffer(self.fbo, self.msaa_fbo)
        
        # Read pixels from framebuffer
        pixels = self.fbo.color_attachments[0].read()
        
        # Convert to PIL Image
        image_data = np.frombuffer(pixels, dtype='f4').reshape(
            (self.settings.height, self.settings.width, 4)
        )
        
        # Convert from float to uint8 and flip vertically
        image_data = (np.clip(image_data, 0, 1) * 255).astype('uint8')
        image_data = np.flipud(image_data)
        
        # Create PIL Image (RGBA)
        image = Image.fromarray(image_data, mode='RGBA')
        
        # Clean up
        vao.release()
        vbo.release()
        
        return image
    
    def render_file(self, file_path: str, output_path: Optional[str] = None) -> Optional[Image.Image]:
        """
        Render a 3D model file to an image.
        
        Args:
            file_path: Path to the 3D model file
            output_path: Optional path to save the rendered image
            
        Returns:
            PIL Image or None if rendering fails
        """
        # Load mesh
        mesh = self.load_mesh(file_path)
        if not mesh:
            return None
        
        # Render to image
        image = self.render_to_image(mesh)
        
        # Save if output path provided
        if output_path:
            image.save(output_path)
            print(f"Rendered image saved to: {output_path}")
        
        return image
    
    def create_thumbnail(self, file_path: str, size: Tuple[int, int] = (256, 256)) -> Optional[Image.Image]:
        """
        Create a thumbnail preview of a 3D model.
        
        Args:
            file_path: Path to the 3D model file
            size: Thumbnail size (width, height)
            
        Returns:
            PIL Image thumbnail or None if rendering fails
        """
        # Update settings for thumbnail size
        original_size = (self.settings.width, self.settings.height)
        self.settings.width, self.settings.height = size
        
        # Re-create framebuffer with new size
        self._create_framebuffer()
        
        # Render the model
        image = self.render_file(file_path)
        
        # Restore original size
        self.settings.width, self.settings.height = original_size
        self._create_framebuffer()
        
        return image
    
    def cleanup(self):
        """Release GPU resources"""
        if self.fbo:
            self.fbo.release()
        if hasattr(self, 'msaa_fbo') and self.msaa_fbo:
            self.msaa_fbo.release()
        if self.program:
            self.program.release()
        if self.ctx:
            self.ctx.release()


class HybridRenderer:
    """
    Hybrid renderer that automatically selects GPU or CPU rendering.
    Falls back to CPU rendering if GPU is not available.
    """
    
    def __init__(self, prefer_gpu: bool = True, settings: Optional[RenderSettings] = None):
        """
        Initialize hybrid renderer.
        
        Args:
            prefer_gpu: Try to use GPU rendering if available
            settings: Render settings
        """
        self.settings = settings or RenderSettings()
        self.gpu_renderer = None
        self.cpu_renderer = None
        self.use_gpu = False
        
        if prefer_gpu and MODERNGL_AVAILABLE:
            try:
                self.gpu_renderer = GPURenderer(self.settings)
                self.use_gpu = True
                print("GPU rendering enabled (ModernGL)")
            except Exception as e:
                print(f"GPU rendering not available: {e}")
                print("Falling back to CPU rendering")
        
        if not self.use_gpu:
            # Import and initialize CPU renderer as fallback
            try:
                from solid_renderer import render_mesh_to_image as cpu_render
                self.cpu_render = cpu_render
                print("CPU rendering enabled (fallback)")
            except ImportError:
                print("Warning: No CPU renderer available. Install dependencies.")
    
    def render_file(self, file_path: str, output_path: Optional[str] = None) -> Optional[Image.Image]:
        """
        Render a 3D model using the best available method.
        
        Args:
            file_path: Path to 3D model file
            output_path: Optional output path for rendered image
            
        Returns:
            Rendered image or None if rendering fails
        """
        if self.use_gpu and self.gpu_renderer:
            return self.gpu_renderer.render_file(file_path, output_path)
        elif hasattr(self, 'cpu_render'):
            # CPU renderer fallback
            try:
                image = self.cpu_render(file_path, size=(self.settings.width, self.settings.height))
                if output_path and image:
                    image.save(output_path)
                return image
            except Exception as e:
                print(f"CPU render failed: {e}")
                return None
        
        print("No renderer available")
        return None
    
    def create_thumbnail(self, file_path: str, size: Tuple[int, int] = (256, 256)) -> Optional[Image.Image]:
        """Create a thumbnail of a 3D model"""
        if self.use_gpu and self.gpu_renderer:
            return self.gpu_renderer.create_thumbnail(file_path, size)
        elif hasattr(self, 'cpu_render'):
            try:
                return self.cpu_render(file_path, size=size)
            except Exception as e:
                print(f"CPU thumbnail failed: {e}")
                return None
        return None
    
    def cleanup(self):
        """Clean up resources"""
        if self.gpu_renderer:
            self.gpu_renderer.cleanup()


def main():
    """Test the GPU renderer with example files"""
    print("=" * 60)
    print("ModelFinder GPU Renderer Test")
    print("=" * 60)
    
    # Check if ModernGL is available
    if not MODERNGL_AVAILABLE:
        print("\n⚠️  ModernGL is not installed!")
        print("To enable GPU rendering, install with:")
        print("  pip install moderngl trimesh pillow")
        return
    
    # Test files (you can modify these paths)
    test_files = [
        "test_model.stl",
        "test_model.obj",
        "test_model.fbx"
    ]
    
    # Create renderer with custom settings
    settings = RenderSettings(
        width=1024,
        height=1024,
        samples=8,  # High-quality anti-aliasing
        model_color=(0.6, 0.7, 0.8),
        metallic=0.2,
        roughness=0.3,
        edge_thickness=0.0  # Set > 0 for wireframe overlay
    )
    
    try:
        # Initialize GPU renderer
        renderer = GPURenderer(settings)
        print("\n✅ GPU Renderer initialized successfully!")
        print(f"OpenGL Version: {renderer.ctx.version_code}")
        
        # Test rendering
        for test_file in test_files:
            if Path(test_file).exists():
                print(f"\nRendering: {test_file}")
                
                # Render full quality
                image = renderer.render_file(test_file, f"{test_file}_rendered.png")
                if image:
                    print(f"  ✓ Full render: {image.size}")
                
                # Create thumbnail
                thumb = renderer.create_thumbnail(test_file, size=(256, 256))
                if thumb:
                    thumb.save(f"{test_file}_thumb.png")
                    print(f"  ✓ Thumbnail: {thumb.size}")
        
        # Test hybrid renderer
        print("\n" + "=" * 40)
        print("Testing Hybrid Renderer")
        print("=" * 40)
        
        hybrid = HybridRenderer(prefer_gpu=True, settings=settings)
        print(f"Using GPU: {hybrid.use_gpu}")
        
        # Clean up
        renderer.cleanup()
        hybrid.cleanup()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
