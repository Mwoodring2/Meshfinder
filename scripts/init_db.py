"""Initialize ModelFinder database with proper schema"""
import sqlite3
from pathlib import Path

DB_PATH = Path("db/modelfinder.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

print(f"Initializing database: {DB_PATH}")

con = sqlite3.connect(DB_PATH)
cur = con.cursor()

# Create files table
cur.execute("""
    CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY,
        path TEXT UNIQUE,
        name TEXT,
        ext TEXT,
        size INTEGER,
        mtime REAL,
        tags TEXT DEFAULT '',
        tris INTEGER,
        dim_x REAL,
        dim_y REAL,
        dim_z REAL,
        volume REAL,
        watertight INTEGER,
        project_number TEXT,
        project_name TEXT,
        part_name TEXT,
        proposed_name TEXT,
        type_conf REAL,
        project_conf REAL,
        sha256 TEXT,
        status TEXT,
        license_type TEXT DEFAULT 'unknown',
        asset_category TEXT DEFAULT 'unknown',
        migration_dest TEXT,
        migration_status TEXT
    )
""")

# Create indices
cur.execute("CREATE INDEX IF NOT EXISTS idx_name ON files(name)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_ext ON files(ext)")

# Create reference parts table
cur.execute("""
    CREATE TABLE IF NOT EXISTS project_reference_parts(
        id INTEGER PRIMARY KEY,
        project_number TEXT,
        project_name TEXT,
        part_name TEXT,
        original_label TEXT,
        description TEXT,
        quantity INTEGER DEFAULT 1,
        UNIQUE(project_number, project_name, part_name) ON CONFLICT IGNORE
    )
""")

# Create operations log table with enhanced fields for guardrails
cur.execute("""
    CREATE TABLE IF NOT EXISTS operations_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        operation TEXT NOT NULL,
        source_path TEXT,
        dest_path TEXT,
        status TEXT,
        details TEXT,
        sha256_before TEXT,
        sha256_after TEXT,
        file_size_before INTEGER,
        file_size_after INTEGER,
        version_bump INTEGER DEFAULT 0,
        user_confirmed INTEGER DEFAULT 0
    )
""")

# Create user corrections table for active learning
cur.execute("""
    CREATE TABLE IF NOT EXISTS user_corrections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_path TEXT NOT NULL,
        original_name TEXT,
        corrected_name TEXT,
        project_number TEXT,
        part_type TEXT,
        laterality TEXT,
        confidence REAL,
        correction_type TEXT,  -- 'rename', 'proposal_override', 'classification_fix'
        corrected_at TEXT NOT NULL,
        used_for_training INTEGER DEFAULT 0
    )
""")

# Create malformed mesh quarantine table
cur.execute("""
    CREATE TABLE IF NOT EXISTS quarantined_meshes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_path TEXT UNIQUE NOT NULL,
        quarantine_reason TEXT NOT NULL,
        error_details TEXT,
        quarantined_at TEXT NOT NULL,
        file_size INTEGER,
        sha256 TEXT,
        resolution_attempts INTEGER DEFAULT 0,
        resolved INTEGER DEFAULT 0
    )
""")

# Create training samples table (if not exists)
cur.execute("""
    CREATE TABLE IF NOT EXISTS training_samples (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_path TEXT UNIQUE NOT NULL,
        features_json TEXT NOT NULL,
        part_type TEXT,
        laterality TEXT,
        project_number TEXT,
        project_name TEXT,
        confidence REAL,
        created_at TEXT NOT NULL
    )
""")

con.commit()
con.close()

print("Database initialized successfully")
print("Tables: files, project_reference_parts, operations_log, user_corrections, quarantined_meshes, training_samples")

