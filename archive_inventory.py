# archive_inventory.py
import os
from pathlib import Path
from collections import defaultdict
import json
from datetime import datetime

def scan_3d_archive(root_path):
    """
    First scan of your 3D archive to understand what you have
    """
    stats = defaultdict(int)
    file_list = []
    size_by_extension = defaultdict(int)
    
    print(f"🔍 Scanning {root_path}...")
    
    # Scan everything
    for root, dirs, files in os.walk(root_path):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            
            # Skip non-3D files
            if ext not in ['.ztl', '.obj', '.stl', '.fbx', '.ply', '.glb', '.gltf', '.3ds', '.dae']:
                continue
            
            try:
                size = os.path.getsize(file_path)
                stats[ext] += 1
                size_by_extension[ext] += size
                stats['total_files'] += 1
                stats['total_size'] += size
                
                # Keep track of large files (>100MB)
                if size > 100_000_000:
                    stats['large_files'] += 1
                
                # Store file info
                file_list.append({
                    'path': file_path,
                    'name': file,
                    'extension': ext,
                    'size_mb': round(size / 1_000_000, 2),
                    'modified': os.path.getmtime(file_path)
                })
                
            except Exception as e:
                stats['errors'] += 1
                print(f"❌ Error reading {file}: {e}")
    
    # Print summary
    print("\n" + "="*50)
    print("📊 ARCHIVE SUMMARY")
    print("="*50)
    print(f"📁 Total 3D files: {stats['total_files']:,}")
    print(f"💾 Total size: {stats['total_size'] / 1_000_000_000:.2f} GB")
    print(f"📏 Average file size: {stats['total_size'] / max(stats['total_files'], 1) / 1_000_000:.2f} MB")
    print(f"⚠️  Large files (>100MB): {stats['large_files']}")
    
    print("\n📈 Breakdown by file type:")
    for ext, count in sorted(stats.items()):
        if ext.startswith('.'):
            size_gb = size_by_extension[ext] / 1_000_000_000
            print(f"  {ext}: {count:,} files ({size_gb:.2f} GB)")
    
    # Find oldest and newest files
    if file_list:
        file_list.sort(key=lambda x: x['modified'])
        oldest = file_list[0]
        newest = file_list[-1]
        oldest_date = datetime.fromtimestamp(oldest['modified']).strftime('%Y-%m-%d')
        newest_date = datetime.fromtimestamp(newest['modified']).strftime('%Y-%m-%d')
        
        print(f"\n📅 Date range: {oldest_date} to {newest_date}")
        print(f"🏛️ Oldest file: {oldest['name']}")
        print(f"🆕 Newest file: {newest['name']}")
    
    # Save detailed inventory
    output_file = f"3d_archive_inventory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'summary': dict(stats),
            'files': file_list[:1000]  # Save first 1000 for review
        }, f, indent=2)
    
    print(f"\n💾 Detailed inventory saved to: {output_file}")
    
    return stats, file_list

if __name__ == "__main__":
    # CHANGE THIS to your actual archive path
    archive_path = input("Enter the full path to your 3D archive folder: ").strip()
    
    if not os.path.exists(archive_path):
        print(f"❌ Path doesn't exist: {archive_path}")
    else:
        stats, files = scan_3d_archive(archive_path)
        
        # Show top 10 largest files
        print("\n🏋️ Top 10 largest files:")
        files.sort(key=lambda x: x['size_mb'], reverse=True)
        for f in files[:10]:
            print(f"  {f['size_mb']:.1f} MB - {f['name']}")