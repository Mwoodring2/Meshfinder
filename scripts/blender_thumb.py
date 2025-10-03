# scripts/blender_thumb.py
# Usage (from PowerShell wrapper below):
# "C:\Program Files\Blender Foundation\Blender 4.4\blender.exe" -b -P scripts\blender_thumb.py -- <in_glb> <out_png> <size> <db_path> <orig_asset_path>

import bpy, sys, math, sqlite3, json
from mathutils import Vector
from pathlib import Path

def enable_addons():
    # Ensure bundled importers are available
    for mod in ("io_scene_gltf2", "io_scene_obj", "io_mesh_stl"):
        try:
            bpy.ops.preferences.addon_enable(module=mod)
        except Exception:
            pass

def import_mesh(p: Path):
    ext = p.suffix.lower()
    if ext == ".obj":
        bpy.ops.import_scene.obj(filepath=str(p))
    elif ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(p))
    elif ext == ".stl":
        bpy.ops.import_mesh.stl(filepath=str(p))
    else:
        raise RuntimeError(f"Unsupported extension: {ext}")

def join_all_meshes():
    objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not objs:
        return None
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    return bpy.context.view_layer.objects.active

def frame_camera(obj):
    # center on origin, compute extent
    bbox = [Vector(b) for b in obj.bound_box]
    min_v = Vector((min(v.x for v in bbox), min(v.y for v in bbox), min(v.z for v in bbox)))
    max_v = Vector((max(v.x for v in bbox), max(v.y for v in bbox), max(v.z for v in bbox)))
    center = (min_v + max_v) * 0.5
    extent = (max_v - min_v).length
    obj.location -= center

    cam_data = bpy.data.cameras.new("Cam")
    cam = bpy.data.objects.new("Cam", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    dist = 2.2 * extent if extent > 0 else 2.0
    cam.location = (0, -dist, dist * 0.6)
    cam.rotation_euler = (math.radians(60), 0, 0)

    # Key light
    key = bpy.data.lights.new(name="Key", type='SUN')
    key_obj = bpy.data.objects.new(name="Key", object_data=key)
    key_obj.rotation_euler = (math.radians(60), 0, math.radians(30))
    bpy.context.collection.objects.link(key_obj)

def main():
    argv = sys.argv
    argv = argv[argv.index("--")+1:] if "--" in argv else []
    if len(argv) < 5:
        print("Usage: blender -b -P blender_thumb.py -- <in_glb> <out_png> <size> <db_path> <orig_asset_path>")
        sys.exit(2)

    in_path = Path(argv[0])
    out_path = Path(argv[1])
    size = int(argv[2])
    db_path = Path(argv[3])
    orig_asset_path = argv[4]  # path in your SQLite `assets.path`

    # Clean scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    # Fast & robust: Eevee + transparent film
    scene.render.engine = 'BLENDER_EEVEE'
    scene.eevee.taa_render_samples = 8
    scene.render.film_transparent = True
    scene.render.resolution_x = size
    scene.render.resolution_y = size

    enable_addons()

    # Import
    import_mesh(in_path)
    obj = join_all_meshes()
    if obj is None:
        print("No mesh objects after import.")
        sys.exit(4)

    frame_camera(obj)

    # Render
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(out_path)
    bpy.ops.render.render(write_still=True)
    print(f"Saved {out_path}")

    # Update SQLite poster_path for this asset row
    try:
        conn = sqlite3.connect(str(db_path))
        with conn:
            conn.execute(
                "UPDATE assets SET poster_path=? WHERE path=?",
                (str(out_path), orig_asset_path)
            )
        conn.close()
        print("DB updated")
    except Exception as e:
        print(f"DB update failed: {e}")

if __name__ == "__main__":
    main()
