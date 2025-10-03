#!/usr/bin/env python3
import argparse, sqlite3, subprocess, sys, tempfile, json, os, shutil
from pathlib import Path

SUPPORTED_EXTS = {".glb", ".gltf", ".obj", ".stl", ".ply"}

def guess_mesh_col(cols):
    # Prefer any column containing 'glb' then 'path'
    for c in cols:
        lc = c.lower()
        if "glb" in lc and "path" in lc:
            return c
    for c in cols:
        if c.lower() == "glb_path":
            return c
    for c in cols:
        if c.lower() == "path":
            return c
    # fallbacks people commonly use
    for c in cols:
        if "file" in c.lower() or "relpath" in c.lower():
            return c
    return None

def rows_from_db(db_path, limit=None):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("PRAGMA table_info(assets)")
    cols = [r["name"] for r in cur.fetchall()]
    mesh_col = guess_mesh_col(cols)
    if not mesh_col:
        raise SystemExit("Could not detect mesh path column in assets table")

    # include id to build default poster names
    select_cols = ["id", mesh_col]
    maybe_name = [c for c in cols if c.lower() in ("name","basename","title")]
    if maybe_name: select_cols += [maybe_name[0]]

    cur.execute(f"SELECT {', '.join(select_cols)} FROM assets")
    rows = []
    for r in cur.fetchall():
        rid = r["id"]
        pval = r[mesh_col]
        if not pval:
            continue
        rows.append({
            "id": rid,
            "name": (r[maybe_name[0]] if maybe_name else str(rid)),
            "mesh": pval
        })
    if limit:
        rows = rows[:limit]
    return rows

BLENDER_TEMPLATE = r"""
import bpy, sys, json, os
from math import radians

argv = sys.argv
argv = argv[argv.index('--')+1:] if '--' in argv else []
cfg = json.loads(argv[0])

# Reset scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Render settings
bpy.context.scene.render.engine = 'BLENDER_EEVEE'
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.resolution_x = cfg["size"]
bpy.context.scene.render.resolution_y = cfg["size"]
bpy.context.scene.render.film_transparent = True

# World (light gray)
bpy.data.worlds["World"].color = (0.04, 0.04, 0.04)

# Camera
cam_data = bpy.data.cameras.new("Cam")
cam = bpy.data.objects.new("Cam", cam_data)
bpy.context.collection.objects.link(cam)
bpy.context.scene.camera = cam
cam.location = (2.5, -2.5, 2.0)
cam.rotation_euler = (radians(60), 0, radians(45))

# Light
light_data = bpy.data.lights.new(name="Key", type='AREA')
light_data.energy = 3000
light = bpy.data.objects.new(name="Key", object_data=light_data)
bpy.context.collection.objects.link(light)
light.location = (3, -3, 4)

# Import mesh
path = cfg["inpath"]
ext = os.path.splitext(path)[1].lower()
if ext in (".glb", ".gltf"):
    bpy.ops.import_scene.gltf(filepath=path)
elif ext == ".obj":
    bpy.ops.wm.obj_import(filepath=path)
elif ext == ".stl":
    bpy.ops.import_mesh.stl(filepath=path)
elif ext == ".ply":
    bpy.ops.wm.ply_import(filepath=path)
else:
    raise RuntimeError(f"Unsupported ext: {ext}")

# Select all imported meshes and join (optional but yields one object for framing)
for obj in list(bpy.context.scene.objects):
    obj.select_set(False)
meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
if not meshes:
    raise RuntimeError("No mesh objects after import")
for m in meshes: m.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]

# Shade smooth (best effort)
try:
    bpy.ops.object.shade_smooth()
except Exception:
    pass

# Frame & fit camera
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
bpy.ops.view3d.camera_to_view_selected()  # requires context override; fallback below if it fails

# Simple bounds-based camera distance adjustment
bpy.ops.object.select_all(action='DESELECT')
for o in meshes: o.select_set(True)
b = bpy.context.selected_objects[0].bound_box
# Use bpy.ops.view3d.camera_to_view_selected via override if available
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        with bpy.context.temp_override(area=area, region=area.regions[-1], space_data=area.spaces.active):
            try:
                bpy.ops.view3d.camera_to_view_selected()
            except Exception:
                pass
        break

# Render
bpy.context.scene.render.filepath = cfg["outpath"]
bpy.ops.render.render(write_still=True)
"""

def ensure_tmp_blender_driver():
    tmp = Path(tempfile.gettempdir()) / "blender_render_driver.py"
    tmp.write_text(BLENDER_TEMPLATE, encoding="utf-8")
    return str(tmp)

def find_import_path(row, repo_root):
    p = Path(row["mesh"])
    # allow relative to repo
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    if p.exists():
        return str(p)
    # try common derived GLB locations based on id
    try_id = str(row["id"])
    candidates = [
        repo_root / "data" / "glb" / f"{try_id}.glb",
        repo_root / "data" / "glb" / f"{try_id}.gltf",
    ]
    for c in candidates:
        if c.exists(): return str(c)
    # last resort: original path literal
    return str(Path(row["mesh"]))

def run_blender(blender_exe, cfg_json):
    return subprocess.call([
        blender_exe, "-b", "-P", ensure_tmp_blender_driver(), "--", cfg_json
    ])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--blender", required=True)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = rows_from_db(args.db, args.limit)
    print(f"[INFO] Rendering posters: n={len(rows)} -> {outdir}")

    ok = 0; fail = 0
    for i, r in enumerate(rows, 1):
        inpath = find_import_path(r, repo_root)
        ext = Path(inpath).suffix.lower()
        if ext not in SUPPORTED_EXTS:
            print(f"[SKIP] {r['id']} unsupported ext: {ext}")
            continue
        outpath = outdir / f"{r['id']}.png"
        cfg = {
            "inpath": str(inpath),
            "outpath": str(outpath),
            "size": int(args.size)
        }
        code = run_blender(args.blender, json.dumps(cfg))
        if code == 0 and outpath.exists():
            ok += 1
            print(f"[OK] {i}/{len(rows)} -> {outpath}")
        else:
            fail += 1
            print(f"[FAIL] {i}/{len(rows)} id={r['id']} ({Path(inpath).name}) code={code}")

    print(f"[SUMMARY] ok={ok} fail={fail}")

if __name__ == "__main__":
    main()
