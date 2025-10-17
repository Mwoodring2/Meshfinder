"""
Stress test for ModelFinder proposal system.

Tests the complete workflow:
1. Scan folder
2. Generate proposals
3. Update database
4. Validate results

Usage:
    python scripts/stress_test.py --folder "E:/path/to/300915_ManBat"
"""
import argparse
import sys
import time
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    import os
    os.system('')  # Enable ANSI escape sequences
    sys.stdout.reconfigure(encoding='utf-8')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.naming import extract_project_number
from src.utils.normalize import ascii_slug, guess_part_from_filename
from src.dataio.reference_parts import load_reference_parts
from src.features.propose_from_reference import propose_for_rows, RowMeta, summary_stats
from src.dataio.db import update_proposal, batch_update_proposals


SUPPORTED_EXTS = {".obj", ".fbx", ".stl", ".ma", ".mb", ".glb", ".gltf"}


def scan_folder(folder_path: str) -> list[dict]:
    """
    Scan folder for 3D files.
    
    Args:
        folder_path: Path to scan
    
    Returns:
        List of file dictionaries
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    files = []
    for ext in SUPPORTED_EXTS:
        for file_path in folder.rglob(f"*{ext}"):
            files.append({
                "path": str(file_path),
                "name": file_path.name,
                "ext": file_path.suffix,
                "size": file_path.stat().st_size,
                "parent": file_path.parent.name
            })
    
    return files


def test_project_number_extraction(folder_path: str):
    """Test project number extraction."""
    print("=" * 80)
    print("TEST 1: Project Number Extraction")
    print("=" * 80)
    
    folder_name = Path(folder_path).name
    project_num = extract_project_number(folder_name)
    
    print(f"Folder name: {folder_name}")
    print(f"Extracted project number: {project_num or 'NONE'}")
    
    if project_num:
        print("✓ PASS: Project number extracted")
    else:
        print("✗ FAIL: Could not extract project number")
        print("  Tip: Folder should contain pattern like '300915' or 'ABC-1234'")
    
    print()
    return project_num


def test_reference_parts_lookup(db_path: str, project_number: str):
    """Test reference parts lookup."""
    print("=" * 80)
    print("TEST 2: Reference Parts Lookup")
    print("=" * 80)
    
    if not project_number:
        print("⊘ SKIP: No project number to lookup")
        print()
        return None, {}
    
    try:
        project_name, parts_map = load_reference_parts(db_path, project_number)
        
        print(f"Project number: {project_number}")
        print(f"Project name: {project_name or 'NOT FOUND'}")
        print(f"Reference parts: {len(parts_map)}")
        
        if parts_map:
            print("\nReference vocabulary:")
            for i, (part_slug, original) in enumerate(list(parts_map.items())[:10], 1):
                print(f"  {i}. {part_slug} (original: {original})")
            if len(parts_map) > 10:
                print(f"  ... and {len(parts_map) - 10} more")
            print("✓ PASS: Reference parts found")
        else:
            print("⚠ WARNING: No reference parts found")
            print("  Tip: Run import_label_excel_to_modelfinder.py first")
        
        print()
        return project_name, parts_map
    
    except Exception as e:
        print(f"✗ FAIL: Error loading reference parts: {e}")
        print()
        return None, {}


def test_file_scanning(folder_path: str):
    """Test file scanning."""
    print("=" * 80)
    print("TEST 3: File Scanning")
    print("=" * 80)
    
    try:
        files = scan_folder(folder_path)
        
        print(f"Folder: {folder_path}")
        print(f"Files found: {len(files)}")
        
        # Group by extension
        by_ext = {}
        for f in files:
            ext = f["ext"]
            by_ext[ext] = by_ext.get(ext, 0) + 1
        
        print("\nBy extension:")
        for ext, count in sorted(by_ext.items()):
            print(f"  {ext}: {count}")
        
        # Show sample files
        print("\nSample files:")
        for i, f in enumerate(files[:5], 1):
            print(f"  {i}. {f['name']} ({f['size'] / 1024:.1f} KB)")
        if len(files) > 5:
            print(f"  ... and {len(files) - 5} more")
        
        if files:
            print("✓ PASS: Files scanned successfully")
        else:
            print("✗ FAIL: No supported files found")
            print(f"  Supported: {', '.join(SUPPORTED_EXTS)}")
        
        print()
        return files
    
    except Exception as e:
        print(f"✗ FAIL: Error scanning folder: {e}")
        print()
        return []


def test_fuzzy_matching(files: list[dict], parts_vocab: set[str]):
    """Test fuzzy matching."""
    print("=" * 80)
    print("TEST 4: Fuzzy Matching")
    print("=" * 80)
    
    if not files:
        print("⊘ SKIP: No files to match")
        print()
        return
    
    if not parts_vocab:
        print("⊘ SKIP: No reference vocabulary")
        print()
        return
    
    print(f"Files: {len(files)}")
    print(f"Vocabulary: {len(parts_vocab)} parts")
    print()
    
    matches = []
    no_matches = []
    
    for f in files[:10]:  # Test first 10
        filename = f["name"]
        best_match, score = guess_part_from_filename(filename, parts_vocab)
        
        if best_match and score > 0:
            matches.append((filename, best_match, score))
            status = "✓" if score >= 0.66 else "⚠"
            print(f"{status} {filename}")
            print(f"   → {best_match} ({int(score * 100)}%)")
        else:
            no_matches.append(filename)
            print(f"✗ {filename}")
            print(f"   → No match")
    
    if len(files) > 10:
        print(f"\n... skipped {len(files) - 10} files for brevity")
    
    print(f"\nMatches: {len(matches)}/{min(10, len(files))}")
    print(f"No matches: {len(no_matches)}/{min(10, len(files))}")
    
    if matches:
        avg_score = sum(s for _, _, s in matches) / len(matches)
        print(f"Average score: {int(avg_score * 100)}%")
        print("✓ PASS: Fuzzy matching working")
    else:
        print("⚠ WARNING: No matches found")
    
    print()


def test_proposal_generation(files: list[dict], db_path: str, project_number: str):
    """Test proposal generation."""
    print("=" * 80)
    print("TEST 5: Proposal Generation")
    print("=" * 80)
    
    if not files:
        print("⊘ SKIP: No files to process")
        print()
        return []
    
    if not project_number:
        print("⊘ SKIP: No project number")
        print()
        return []
    
    # Convert files to RowMeta
    rows = [
        RowMeta(
            path=f["path"],
            name=f["name"],
            ext=f["ext"],
            tags=""
        )
        for f in files
    ]
    
    print(f"Generating proposals for {len(rows)} files...")
    
    try:
        start_time = time.time()
        proposals = propose_for_rows(rows, db_path, project_number)
        elapsed = time.time() - start_time
        
        print(f"✓ Generated {len(proposals)} proposals in {elapsed:.2f}s")
        print(f"  Speed: {len(proposals) / elapsed:.1f} files/sec")
        print()
        
        # Show statistics
        stats = summary_stats(proposals)
        print("Statistics:")
        print(f"  Total: {stats['total']}")
        print(f"  Auto-accept (≥66%): {stats['auto_accept']}")
        print(f"  Needs review (<66%): {stats['needs_review']}")
        print(f"  Average confidence: {int(stats['avg_confidence'] * 100)}%")
        print()
        
        # Show sample proposals
        print("Sample proposals:")
        for i, p in enumerate(proposals[:5], 1):
            status = "✓" if not p["needs_review"] else "⚠"
            conf = int(p["conf"] * 100)
            print(f"{status} {i}. {Path(p['from']).name}")
            print(f"     → {p['proposed_name']} ({conf}%)")
        
        if len(proposals) > 5:
            print(f"  ... and {len(proposals) - 5} more")
        
        print()
        print("✓ PASS: Proposal generation successful")
        print()
        
        return proposals
    
    except Exception as e:
        print(f"✗ FAIL: Proposal generation error: {e}")
        import traceback
        traceback.print_exc()
        print()
        return []


def test_database_update(proposals: list[dict], dry_run: bool = True):
    """Test database update."""
    print("=" * 80)
    print(f"TEST 6: Database Update {'(DRY RUN)' if dry_run else ''}")
    print("=" * 80)
    
    if not proposals:
        print("⊘ SKIP: No proposals to update")
        print()
        return
    
    if dry_run:
        print("Dry run mode - no actual database changes")
        print(f"Would update {len(proposals)} records")
        
        # Show what would be updated
        print("\nSample updates:")
        for i, p in enumerate(proposals[:3], 1):
            print(f"{i}. {Path(p['from']).name}")
            print(f"   project_number: {p['project_number']}")
            print(f"   project_name: {p['project_name']}")
            print(f"   part_name: {p['part_name']}")
            print(f"   confidence: {int(p['conf'] * 100)}%")
        
        print()
        print("✓ PASS: Dry run successful")
        print("  Tip: Use --update flag to actually update database")
    else:
        print("Updating database...")
        try:
            start_time = time.time()
            updated = batch_update_proposals(proposals)
            elapsed = time.time() - start_time
            
            print(f"✓ Updated {updated} records in {elapsed:.2f}s")
            print("✓ PASS: Database update successful")
        except Exception as e:
            print(f"✗ FAIL: Database update error: {e}")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Stress test ModelFinder proposal system"
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Folder to scan (e.g., 'E:/path/300915_ManBat')"
    )
    parser.add_argument(
        "--db",
        default="db/modelfinder.db",
        help="Database path (default: db/modelfinder.db)"
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Actually update database (default: dry run)"
    )
    parser.add_argument(
        "--project-num",
        help="Override project number (auto-detected from folder name)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("MODELFINDER STRESS TEST")
    print("=" * 80)
    print(f"Folder: {args.folder}")
    print(f"Database: {args.db}")
    print(f"Mode: {'UPDATE' if args.update else 'DRY RUN'}")
    print("=" * 80)
    print()
    
    # Run tests
    project_number = test_project_number_extraction(args.folder)
    
    if args.project_num:
        print(f"Overriding with user-provided project number: {args.project_num}")
        project_number = args.project_num
        print()
    
    project_name, parts_vocab = test_reference_parts_lookup(args.db, project_number)
    
    files = test_file_scanning(args.folder)
    
    test_fuzzy_matching(files, parts_vocab)
    
    proposals = test_proposal_generation(files, args.db, project_number)
    
    test_database_update(proposals, dry_run=not args.update)
    
    # Final summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Project: {project_number or 'UNKNOWN'}")
    print(f"Files scanned: {len(files)}")
    print(f"Reference parts: {len(parts_vocab)}")
    print(f"Proposals generated: {len(proposals)}")
    
    if proposals:
        auto = sum(1 for p in proposals if not p["needs_review"])
        review = len(proposals) - auto
        print(f"Auto-accept: {auto}")
        print(f"Needs review: {review}")
        print(f"Acceptance rate: {int(auto / len(proposals) * 100)}%")
    
    print()
    
    if proposals and not args.update:
        print("💡 Tip: Add --update flag to persist proposals to database")
    
    print()
    print("✅ STRESS TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

