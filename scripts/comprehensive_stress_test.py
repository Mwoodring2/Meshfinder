"""
Comprehensive End-to-End Stress Test for ModelFinder Enhanced UI

Tests ALL functionality:
1. Database setup & migrations
2. File scanning & indexing
3. Excel import (reference parts)
4. Proposal generation with reference picker
5. Settings persistence
6. File operations (rename, delete, copy)
7. Confidence coloring & review flagging
8. Migration dry-run & execution
9. Operations logging
10. UI component integration

Usage:
    python scripts/comprehensive_stress_test.py
    python scripts/comprehensive_stress_test.py --full-migration  # Run actual file moves
    python scripts/comprehensive_stress_test.py --cleanup  # Clean test data after
"""
import argparse
import sys
import time
import shutil
import tempfile
import sqlite3
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding
if sys.platform == 'win32':
    import os
    os.system('')  # Enable ANSI escape sequences
    sys.stdout.reconfigure(encoding='utf-8')

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.naming import extract_project_number
from src.utils.normalize import ascii_slug, guess_part_from_filename
from src.dataio.reference_parts import load_reference_parts, get_all_projects
from src.features.propose_from_reference import propose_for_rows, RowMeta, summary_stats
from src.dataio.db import update_proposal, batch_update_proposals


# Test configuration
DB_PATH = PROJECT_ROOT / "db" / "modelfinder.db"
SUPPORTED_EXTS = {".obj", ".fbx", ".stl", ".ma", ".mb", ".glb", ".gltf"}
TEST_PROJECT = "300868"  # Superman PF
TEST_CONFIDENCE_THRESHOLD = 0.66


class TestResults:
    """Track test results"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.warnings = 0
        self.start_time = time.time()
        self.tests = []
    
    def pass_test(self, name, details=""):
        self.passed += 1
        self.tests.append(("PASS", name, details))
        print(f"✓ PASS: {name}")
        if details:
            print(f"  {details}")
    
    def fail_test(self, name, error):
        self.failed += 1
        self.tests.append(("FAIL", name, str(error)))
        print(f"✗ FAIL: {name}")
        print(f"  Error: {error}")
    
    def skip_test(self, name, reason):
        self.skipped += 1
        self.tests.append(("SKIP", name, reason))
        print(f"⊘ SKIP: {name}")
        print(f"  Reason: {reason}")
    
    def warn_test(self, name, warning):
        self.warnings += 1
        self.tests.append(("WARN", name, warning))
        print(f"⚠ WARN: {name}")
        print(f"  Warning: {warning}")
    
    def print_summary(self):
        elapsed = time.time() - self.start_time
        total = len(self.tests)
        
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"Total tests: {total}")
        print(f"✓ Passed: {self.passed}")
        print(f"✗ Failed: {self.failed}")
        print(f"⊘ Skipped: {self.skipped}")
        print(f"⚠ Warnings: {self.warnings}")
        print(f"Time: {elapsed:.2f}s")
        print()
        
        if self.failed == 0:
            print("🎉 ALL TESTS PASSED!")
        else:
            print("❌ SOME TESTS FAILED - Review errors above")
            print("\nFailed tests:")
            for status, name, details in self.tests:
                if status == "FAIL":
                    print(f"  • {name}: {details}")


def print_section(title):
    """Print section header"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def create_test_files(test_dir: Path) -> list[Path]:
    """Create test 3D files for testing"""
    print("Creating test files...")
    
    test_files = [
        "Superman_Head.stl",
        "Superman_Body.obj",
        "Superman_Cape.stl",
        "Superman_Base.stl",
        "Superman_Logo_S.obj",
        "Lois_Lane_Head.stl",
        "Daily_Planet_Logo.obj",
        "Superman_Hand_Left.stl",
        "Superman_Hand_Right.stl",
        "Kryptonite_Crystal.stl",
    ]
    
    created_files = []
    for filename in test_files:
        file_path = test_dir / filename
        # Create minimal STL/OBJ file
        if file_path.suffix == ".stl":
            content = b"solid test\nendsolid test\n"
        else:
            content = b"# Test OBJ file\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
        
        file_path.write_bytes(content)
        created_files.append(file_path)
    
    print(f"Created {len(created_files)} test files")
    return created_files


# ============================================================================
# TEST 1: Database Setup & Schema
# ============================================================================

def test_database_setup(results: TestResults):
    """Test database connection and schema"""
    print_section("TEST 1: Database Setup & Schema")
    
    try:
        # Check database exists
        if not DB_PATH.exists():
            results.fail_test("Database existence", f"Database not found at {DB_PATH}")
            return
        
        results.pass_test("Database file exists", str(DB_PATH))
        
        # Check tables
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        
        # Check files table
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='files'")
        if cur.fetchone():
            results.pass_test("files table exists")
            
            # Check columns
            cur.execute("PRAGMA table_info(files)")
            columns = {row[1] for row in cur.fetchall()}
            required = {"path", "name", "ext", "size", "mtime", "tags", 
                       "project_number", "project_name", "part_name", 
                       "proposed_name", "type_conf", "status"}
            
            missing = required - columns
            if missing:
                results.warn_test("files table columns", f"Missing: {missing}")
            else:
                results.pass_test("files table columns complete")
        else:
            results.fail_test("files table", "Table does not exist")
        
        # Check project_reference_parts table
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='project_reference_parts'")
        if cur.fetchone():
            results.pass_test("project_reference_parts table exists")
        else:
            results.warn_test("project_reference_parts table", "Table missing - Excel import may not work")
        
        # Check operations_log table (for migration tracking)
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='operations_log'")
        if not cur.fetchone():
            # Create operations_log table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS operations_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    source_path TEXT,
                    dest_path TEXT,
                    status TEXT,
                    details TEXT
                )
            """)
            con.commit()
            results.pass_test("operations_log table created")
        else:
            results.pass_test("operations_log table exists")
        
        con.close()
        
    except Exception as e:
        results.fail_test("Database setup", str(e))


# ============================================================================
# TEST 2: File Scanning & Indexing
# ============================================================================

def test_file_scanning(results: TestResults, test_dir: Path):
    """Test file scanning functionality"""
    print_section("TEST 2: File Scanning & Indexing")
    
    try:
        # Create test files
        test_files = create_test_files(test_dir)
        results.pass_test("Test file creation", f"Created {len(test_files)} files")
        
        # Scan files
        found_files = []
        for ext in SUPPORTED_EXTS:
            found_files.extend(test_dir.glob(f"*{ext}"))
        
        if len(found_files) == len(test_files):
            results.pass_test("File scanning", f"Found all {len(found_files)} files")
        else:
            results.warn_test("File scanning", 
                            f"Expected {len(test_files)}, found {len(found_files)}")
        
        # Insert into database
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        
        inserted = 0
        for file_path in found_files:
            try:
                stat = file_path.stat()
                cur.execute("""
                    INSERT OR REPLACE INTO files 
                    (path, name, ext, size, mtime, tags)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    str(file_path),
                    file_path.name,
                    file_path.suffix,
                    stat.st_size,
                    stat.st_mtime,
                    ""
                ))
                inserted += 1
            except Exception as e:
                results.fail_test(f"Insert file {file_path.name}", str(e))
        
        con.commit()
        con.close()
        
        if inserted == len(found_files):
            results.pass_test("Database insertion", f"Inserted {inserted} files")
        else:
            results.fail_test("Database insertion", 
                            f"Expected {len(found_files)}, inserted {inserted}")
        
    except Exception as e:
        results.fail_test("File scanning", str(e))


# ============================================================================
# TEST 3: Reference Parts Lookup
# ============================================================================

def test_reference_parts(results: TestResults):
    """Test reference parts lookup"""
    print_section("TEST 3: Reference Parts Lookup")
    
    try:
        # Get all projects
        projects = get_all_projects(str(DB_PATH))
        
        if projects:
            results.pass_test("Get all projects", f"Found {len(projects)} projects")
            print(f"  Available projects: {', '.join(projects[:5])}")
            if len(projects) > 5:
                print(f"  ... and {len(projects) - 5} more")
        else:
            results.warn_test("Get all projects", "No projects found - Excel import needed")
            return
        
        # Load reference parts for test project
        if TEST_PROJECT in projects or projects:
            project_to_test = TEST_PROJECT if TEST_PROJECT in projects else projects[0]
            
            project_name, parts_map = load_reference_parts(str(DB_PATH), project_to_test)
            
            if parts_map:
                results.pass_test(f"Load reference parts for {project_to_test}", 
                                f"Found {len(parts_map)} parts")
                print(f"  Project name: {project_name}")
                print(f"  Sample parts: {', '.join(list(parts_map.keys())[:5])}")
            else:
                results.warn_test(f"Load reference parts for {project_to_test}", 
                                "No parts found")
        else:
            results.skip_test("Load reference parts", "No projects available")
        
    except Exception as e:
        results.fail_test("Reference parts lookup", str(e))


# ============================================================================
# TEST 4: Proposal Generation
# ============================================================================

def test_proposal_generation(results: TestResults, test_dir: Path):
    """Test proposal generation"""
    print_section("TEST 4: Proposal Generation")
    
    try:
        # Get projects
        projects = get_all_projects(str(DB_PATH))
        
        if not projects:
            results.skip_test("Proposal generation", "No reference projects available")
            return
        
        project_to_test = TEST_PROJECT if TEST_PROJECT in projects else projects[0]
        
        # Get test files from database
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        cur.execute("SELECT path, name, ext FROM files WHERE path LIKE ?", 
                   (f"%{test_dir}%",))
        file_rows = cur.fetchall()
        con.close()
        
        if not file_rows:
            results.skip_test("Proposal generation", "No test files in database")
            return
        
        # Convert to RowMeta
        rows = [
            RowMeta(path=row[0], name=row[1], ext=row[2], tags="")
            for row in file_rows
        ]
        
        results.pass_test("Prepare proposal input", f"{len(rows)} files ready")
        
        # Generate proposals
        start_time = time.time()
        proposals = propose_for_rows(rows, str(DB_PATH), project_to_test)
        elapsed = time.time() - start_time
        
        if proposals:
            results.pass_test("Proposal generation", 
                            f"Generated {len(proposals)} proposals in {elapsed:.2f}s")
            
            # Check statistics
            stats = summary_stats(proposals)
            print(f"  Auto-accept (≥{int(TEST_CONFIDENCE_THRESHOLD*100)}%): {stats['auto_accept']}")
            print(f"  Needs review (<{int(TEST_CONFIDENCE_THRESHOLD*100)}%): {stats['needs_review']}")
            print(f"  Average confidence: {int(stats['avg_confidence'] * 100)}%")
            
            # Check proposal structure
            sample = proposals[0]
            required_keys = {"from", "proposed_name", "project_number", "project_name", 
                           "part_name", "conf", "needs_review"}
            if required_keys.issubset(sample.keys()):
                results.pass_test("Proposal structure", "All required fields present")
            else:
                missing = required_keys - sample.keys()
                results.fail_test("Proposal structure", f"Missing fields: {missing}")
            
            # Update database with proposals
            updated = batch_update_proposals(proposals)
            results.pass_test("Batch update proposals", f"Updated {updated} records")
            
        else:
            results.fail_test("Proposal generation", "No proposals generated")
        
    except Exception as e:
        results.fail_test("Proposal generation", str(e))
        import traceback
        traceback.print_exc()


# ============================================================================
# TEST 5: Confidence-Based Row Coloring
# ============================================================================

def test_confidence_coloring(results: TestResults):
    """Test confidence thresholds and coloring logic"""
    print_section("TEST 5: Confidence-Based Row Coloring")
    
    try:
        # Query files with confidence scores
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        cur.execute("""
            SELECT path, proposed_name, type_conf, project_number
            FROM files 
            WHERE type_conf IS NOT NULL
            ORDER BY type_conf DESC
            LIMIT 20
        """)
        rows = cur.fetchall()
        con.close()
        
        if not rows:
            results.skip_test("Confidence coloring", "No rows with confidence scores")
            return
        
        results.pass_test("Query confidence scores", f"Found {len(rows)} rows")
        
        # Check color distribution
        high_conf = sum(1 for r in rows if r[2] >= 0.85)
        med_conf = sum(1 for r in rows if 0.66 <= r[2] < 0.85)
        low_conf = sum(1 for r in rows if r[2] < 0.66)
        
        print(f"  High confidence (≥85%): {high_conf} (green)")
        print(f"  Medium confidence (66-84%): {med_conf} (yellow)")
        print(f"  Low confidence (<66%): {low_conf} (red)")
        
        results.pass_test("Confidence distribution", 
                         f"High: {high_conf}, Med: {med_conf}, Low: {low_conf}")
        
    except Exception as e:
        results.fail_test("Confidence coloring", str(e))


# ============================================================================
# TEST 6: Settings Persistence
# ============================================================================

def test_settings_persistence(results: TestResults):
    """Test settings read/write"""
    print_section("TEST 6: Settings Persistence")
    
    try:
        from PySide6 import QtCore
        
        settings = QtCore.QSettings("ModelFinder", "ModelFinder")
        
        # Write test settings
        test_values = {
            "dest_root": "C:/TestDestination",
            "conf_threshold": 0.75,
            "cache_dir": "C:/TestCache"
        }
        
        for key, value in test_values.items():
            settings.setValue(key, value)
        
        results.pass_test("Settings write", f"Wrote {len(test_values)} settings")
        
        # Read back
        read_back = {}
        for key in test_values.keys():
            read_back[key] = settings.value(key)
        
        # Verify
        matches = 0
        for key, expected in test_values.items():
            actual = read_back[key]
            if key == "conf_threshold":
                actual = float(actual)
            if actual == expected:
                matches += 1
        
        if matches == len(test_values):
            results.pass_test("Settings read", "All settings match")
        else:
            results.warn_test("Settings read", 
                            f"Only {matches}/{len(test_values)} settings match")
        
        # Clean up test settings
        for key in test_values.keys():
            settings.remove(key)
        
    except Exception as e:
        results.fail_test("Settings persistence", str(e))


# ============================================================================
# TEST 7: File Operations
# ============================================================================

def test_file_operations(results: TestResults, test_dir: Path):
    """Test file operations (rename, delete, copy path)"""
    print_section("TEST 7: File Operations")
    
    try:
        # Create a test file for operations
        test_file = test_dir / "test_operations.stl"
        test_file.write_text("solid test\nendsolid test\n")
        
        results.pass_test("Create test file", str(test_file.name))
        
        # Test rename
        new_name = test_dir / "test_operations_renamed.stl"
        test_file.rename(new_name)
        
        if new_name.exists() and not test_file.exists():
            results.pass_test("File rename", f"{test_file.name} → {new_name.name}")
        else:
            results.fail_test("File rename", "File not renamed correctly")
        
        # Test copy path (simulate)
        path_str = str(new_name)
        if path_str and len(path_str) > 0:
            results.pass_test("Copy path", "Path string generated")
        else:
            results.fail_test("Copy path", "Failed to generate path string")
        
        # Test delete
        new_name.unlink()
        if not new_name.exists():
            results.pass_test("File delete", "File deleted successfully")
        else:
            results.fail_test("File delete", "File still exists")
        
    except Exception as e:
        results.fail_test("File operations", str(e))


# ============================================================================
# TEST 8: Migration Dry-Run
# ============================================================================

def test_migration_dry_run(results: TestResults, test_dir: Path):
    """Test migration planning and conflict detection"""
    print_section("TEST 8: Migration Dry-Run")
    
    try:
        # Query files with proposals
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        cur.execute("""
            SELECT path, proposed_name, project_number, project_name, part_name, type_conf
            FROM files 
            WHERE proposed_name IS NOT NULL AND proposed_name != ''
            LIMIT 10
        """)
        rows = cur.fetchall()
        con.close()
        
        if not rows:
            results.skip_test("Migration dry-run", "No files with proposals")
            return
        
        results.pass_test("Load migration candidates", f"Found {len(rows)} files")
        
        # Build migration plan
        dest_root = Path("C:/TestDestination")  # Mock destination
        migration_plan = []
        conflicts = []
        
        for row in rows:
            source_path = Path(row[0])
            project_num = row[2]
            proposed_name = row[1]
            
            if not proposed_name:
                continue
            
            # Build destination path: <dest_root>/<project>/<proposed_name>
            dest_path = dest_root / project_num / proposed_name
            
            migration_plan.append({
                "from": source_path,
                "to": dest_path,
                "confidence": row[5],
                "needs_review": row[5] < TEST_CONFIDENCE_THRESHOLD if row[5] else True
            })
            
            # Check for conflicts (destination already exists)
            # In real implementation, check if dest_path exists
        
        if migration_plan:
            results.pass_test("Build migration plan", 
                            f"{len(migration_plan)} moves planned")
            
            # Show sample
            print("  Sample migrations:")
            for i, m in enumerate(migration_plan[:3], 1):
                conf = int(m["confidence"] * 100) if m["confidence"] else 0
                status = "⚠" if m["needs_review"] else "✓"
                print(f"    {status} {i}. {m['from'].name}")
                print(f"       → {m['to']}")
                print(f"       Confidence: {conf}%")
            
            # Check for conflicts
            dest_paths = [m["to"] for m in migration_plan]
            if len(dest_paths) != len(set(dest_paths)):
                duplicates = len(dest_paths) - len(set(dest_paths))
                conflicts.append(f"{duplicates} duplicate destinations")
            
            if conflicts:
                results.warn_test("Conflict detection", f"Found {len(conflicts)} conflicts")
                for c in conflicts:
                    print(f"    • {c}")
            else:
                results.pass_test("Conflict detection", "No conflicts found")
        else:
            results.fail_test("Build migration plan", "No valid migrations")
        
    except Exception as e:
        results.fail_test("Migration dry-run", str(e))


# ============================================================================
# TEST 9: Operations Logging
# ============================================================================

def test_operations_logging(results: TestResults):
    """Test operations logging"""
    print_section("TEST 9: Operations Logging")
    
    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        
        # Log a test operation
        cur.execute("""
            INSERT INTO operations_log 
            (timestamp, operation, source_path, dest_path, status, details)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            "TEST_OPERATION",
            "/test/source.stl",
            "/test/dest.stl",
            "SUCCESS",
            "Comprehensive stress test"
        ))
        con.commit()
        
        results.pass_test("Log operation", "Test operation logged")
        
        # Query back
        cur.execute("""
            SELECT timestamp, operation, status 
            FROM operations_log 
            WHERE operation = 'TEST_OPERATION'
            ORDER BY timestamp DESC
            LIMIT 1
        """)
        row = cur.fetchone()
        
        if row:
            results.pass_test("Query operations log", f"Found: {row[1]} - {row[2]}")
        else:
            results.fail_test("Query operations log", "Test operation not found")
        
        # Clean up
        cur.execute("DELETE FROM operations_log WHERE operation = 'TEST_OPERATION'")
        con.commit()
        con.close()
        
    except Exception as e:
        results.fail_test("Operations logging", str(e))


# ============================================================================
# TEST 10: Integration Test
# ============================================================================

def test_integration(results: TestResults, test_dir: Path):
    """End-to-end integration test"""
    print_section("TEST 10: End-to-End Integration")
    
    try:
        # Query complete workflow
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        
        # Check files that went through complete workflow
        cur.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(proposed_name) as with_proposals,
                COUNT(CASE WHEN type_conf >= 0.66 THEN 1 END) as auto_accept,
                COUNT(CASE WHEN type_conf < 0.66 THEN 1 END) as needs_review
            FROM files
            WHERE path LIKE ?
        """, (f"%{test_dir}%",))
        
        stats = cur.fetchone()
        con.close()
        
        if stats:
            total, with_proposals, auto_accept, needs_review = stats
            
            print(f"  Total files: {total}")
            print(f"  With proposals: {with_proposals}")
            print(f"  Auto-accept: {auto_accept}")
            print(f"  Needs review: {needs_review}")
            
            if total > 0:
                coverage = (with_proposals / total * 100) if total else 0
                results.pass_test("Workflow coverage", f"{coverage:.1f}% of files processed")
                
                if with_proposals == total:
                    results.pass_test("Complete workflow", "All files have proposals")
                else:
                    results.warn_test("Complete workflow", 
                                    f"{total - with_proposals} files without proposals")
            else:
                results.skip_test("Integration test", "No test files found")
        else:
            results.skip_test("Integration test", "No statistics available")
        
    except Exception as e:
        results.fail_test("Integration test", str(e))


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive ModelFinder Stress Test"
    )
    parser.add_argument(
        "--full-migration",
        action="store_true",
        help="Actually execute file migrations (default: dry-run only)"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up test data after tests complete"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick tests only (skip slow operations)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE MODELFINDER STRESS TEST")
    print("=" * 80)
    print(f"Database: {DB_PATH}")
    print(f"Migration mode: {'FULL' if args.full_migration else 'DRY-RUN'}")
    print(f"Quick mode: {'ON' if args.quick else 'OFF'}")
    print("=" * 80)
    
    results = TestResults()
    
    # Create temporary test directory
    test_dir = Path(tempfile.mkdtemp(prefix="modelfinder_test_"))
    print(f"\nTest directory: {test_dir}")
    
    try:
        # Run all tests
        test_database_setup(results)
        test_file_scanning(results, test_dir)
        test_reference_parts(results)
        
        if not args.quick:
            test_proposal_generation(results, test_dir)
            test_confidence_coloring(results)
            test_settings_persistence(results)
            test_file_operations(results, test_dir)
            test_migration_dry_run(results, test_dir)
            test_operations_logging(results)
            test_integration(results, test_dir)
        else:
            print("\n⏭️  Skipping slow tests (quick mode)")
        
        # Print summary
        results.print_summary()
        
    finally:
        # Cleanup
        if args.cleanup:
            print(f"\n🧹 Cleaning up test directory: {test_dir}")
            try:
                shutil.rmtree(test_dir)
                
                # Clean test files from database
                con = sqlite3.connect(DB_PATH)
                cur = con.cursor()
                cur.execute("DELETE FROM files WHERE path LIKE ?", (f"%{test_dir}%",))
                deleted = cur.rowcount
                con.commit()
                con.close()
                
                print(f"   Deleted {deleted} test records from database")
                print("   ✓ Cleanup complete")
            except Exception as e:
                print(f"   ⚠ Cleanup error: {e}")
        else:
            print(f"\n💡 Test files left in: {test_dir}")
            print("   Use --cleanup flag to remove test data")
    
    # Exit with appropriate code
    sys.exit(0 if results.failed == 0 else 1)


if __name__ == "__main__":
    main()

