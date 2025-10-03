# ModelFinder Architecture

This document provides an overview of the ModelFinder system architecture, components, and data flow.

## System Overview

ModelFinder is a 3D model indexing and search system that uses machine learning to organize, analyze, and find 3D assets. The system is designed to be modular, extensible, and performant.

## Architecture Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Input    │    │   CLI Tools     │    │   PyQt UI       │
│   (Commands)    │    │   (Scripts)     │    │   (Future)      │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │     Application Layer     │
                    │   (Search & Management)   │
                    └─────────────┬─────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │      Core Services        │
                    │  (Indexer, Embedder, QA)  │
                    └─────────────┬─────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │      Data Layer           │
                    │  (SQLite + FAISS + Files) │
                    └───────────────────────────┘
```

## Core Components

### 1. Data Layer

#### SQLite Database (`db/modelfinder.db`)
- **Purpose**: Stores metadata and file information
- **Schema**: 
  - `models` table: File paths, geometry stats, metadata
  - Indexes: Fast searching by filename, type, geometry
- **Benefits**: Portable, ACID compliance, SQL queries

#### FAISS Index (`db/faiss.index`)
- **Purpose**: Vector similarity search
- **Features**: 
  - Embeddings for 3D model features
  - Fast similarity search
  - Clustering and classification
- **Benefits**: High performance, scalable

#### File System (`data/`)
- **`raw/`**: Original 3D model files
- **`glb/`**: Normalized GLB exports
- **`posters/`**: Preview images
- **`metrics/`**: Computed statistics

### 2. Core Services

#### Indexer (`src/indexer/`)
- **File Scanning**: Recursive folder traversal
- **Format Detection**: Multi-format 3D file support
- **Geometry Analysis**: Extract mesh statistics
- **Metadata Extraction**: File properties, timestamps
- **Conversion Pipeline**: Convert to standardized formats

#### Embedder (`src/embedder/`)
- **Feature Extraction**: 3D model feature vectors
- **Embedding Generation**: ML-based representations
- **FAISS Integration**: Index management and search
- **Similarity Metrics**: Distance calculations

#### QA (`src/qa/`)
- **Model Validation**: Check file integrity
- **Repair Tools**: Fix common mesh issues
- **Quality Metrics**: Assess model quality
- **Batch Processing**: Handle large collections

### 3. Application Layer

#### Search Engine
- **Text Search**: Filename and metadata queries
- **Geometric Search**: Filter by mesh properties
- **Similarity Search**: Find similar models
- **Hybrid Search**: Combine multiple criteria

#### Management Interface
- **Database Operations**: CRUD operations
- **Batch Processing**: Bulk operations
- **Configuration**: System settings
- **Monitoring**: Performance and status

### 4. User Interface

#### CLI Tools (`scripts/`)
- **`scan_folder.py`**: Index 3D models
- **`search_cli.py`**: Search and query
- **`build_exe.bat`**: Create standalone executable

#### PyQt UI (`src/ui/`) - Future
- **Model Browser**: Visual file browser
- **Search Interface**: Graphical search tools
- **Preview Viewer**: 3D model previews
- **Settings Panel**: Configuration management

## Data Flow

### 1. Indexing Process
```
3D Files → Scanner → Geometry Analysis → Metadata Extraction → Database Storage
    ↓
Format Detection → Conversion Pipeline → GLB Export → Preview Generation
```

### 2. Search Process
```
Query → Parser → Search Engine → Database Query → Results Ranking → Output
    ↓
Similarity Search → FAISS Index → Vector Comparison → Similarity Scores
```

### 3. Update Process
```
File Changes → Change Detection → Incremental Update → Database Sync
    ↓
New Files → Re-indexing → Embedding Update → Index Rebuild
```

## Technology Stack

### Core Technologies
- **Python 3.8+**: Main programming language
- **SQLite**: Relational database
- **FAISS**: Vector similarity search
- **Trimesh**: 3D mesh processing
- **NumPy**: Numerical computing

### 3D Processing
- **Trimesh**: Mesh loading and analysis
- **PyGLTF**: GLB/glTF file handling
- **Assimp**: Multi-format 3D file support
- **OpenCV**: Image processing for previews

### Machine Learning
- **Scikit-learn**: Traditional ML algorithms
- **PyTorch**: Deep learning models
- **Transformers**: Pre-trained models

### User Interface
- **Click**: Command-line interface
- **PyQt6**: Desktop GUI framework
- **TQDM**: Progress indicators

## Performance Considerations

### Scalability
- **Database Indexing**: Optimized queries with proper indexes
- **FAISS Clustering**: Efficient vector operations
- **Lazy Loading**: Load data only when needed
- **Caching**: Cache frequently accessed data

### Memory Management
- **Streaming Processing**: Process large files in chunks
- **Garbage Collection**: Proper cleanup of 3D objects
- **Memory Pooling**: Reuse objects when possible

### I/O Optimization
- **Parallel Processing**: Multi-threaded file operations
- **Batch Operations**: Group database operations
- **Compression**: Compress stored data when beneficial

## Security Considerations

### Data Protection
- **File Access**: Secure file system operations
- **Database Security**: SQLite access controls
- **Input Validation**: Sanitize user inputs

### Privacy
- **Local Processing**: No cloud dependencies
- **Data Encryption**: Optional encryption for sensitive data
- **Access Logging**: Audit trail for operations

## Extensibility

### Plugin Architecture
- **Format Support**: Easy addition of new file formats
- **Search Algorithms**: Pluggable search methods
- **UI Components**: Modular interface elements

### API Design
- **RESTful API**: Future web interface support
- **Plugin Interface**: Third-party extensions
- **Configuration**: Flexible system configuration

## Deployment Options

### Standalone Executable
- **PyInstaller**: Single-file distribution
- **Portable**: No installation required
- **Cross-platform**: Windows, Linux, macOS

### Development Environment
- **Virtual Environment**: Isolated dependencies
- **Docker**: Containerized deployment
- **CI/CD**: Automated testing and deployment

## Future Enhancements

### Phase 2
- **Web Interface**: Browser-based access
- **Cloud Sync**: Multi-device synchronization
- **AI Tagging**: Automatic model categorization

### Phase 3
- **Collaborative Features**: Sharing and collaboration
- **Advanced Analytics**: Usage statistics and insights
- **Integration**: CAD software integration

---

This architecture provides a solid foundation for a scalable, maintainable 3D model management system that can grow with your needs.














