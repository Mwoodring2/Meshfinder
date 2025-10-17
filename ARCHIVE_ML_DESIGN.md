# Archive ML Training System Design

## Goal
Train ML to learn proper file naming from existing well-organized structure + geometric features.

## Problem Statement
Current system: Simple fuzzy text matching against Excel references
- ❌ Doesn't understand geometry (can't tell foot from hand)
- ❌ Doesn't learn from existing good structure
- ❌ Ignores folder context
- ❌ Names generic files as "part2" instead of "300668_yoda_left_foot"

## Solution: Geometric + Context Learning

### Phase 1: Training Data Extraction
**Learn from existing well-organized folders**

Input: User's 12-year archive with good naming
```
E:/Archive/300668_Yoda_PF/
  ├── 300668_yoda_head.stl          ← HEAD geometry + naming pattern
  ├── 300668_yoda_left_foot.obj     ← FOOT geometry + naming pattern  
  ├── 300668_yoda_left_hand.stl     ← HAND geometry + naming pattern
  └── 300668_yoda_base.stl          ← BASE geometry + naming pattern
```

Extract:
1. **Geometric features** (using trimesh):
   - Bounding box aspect ratio
   - Volume
   - Principal axes (orientation)
   - Surface area
   - Compactness (sphere-like vs elongated)
   - Vertex distribution
   - Convexity

2. **Naming patterns**:
   - Project number from folder: "300668"
   - Character name from folder: "yoda"
   - Part type from filename: "left_foot", "head", "hand"

3. **Context**:
   - Parent folder structure
   - Sibling files (other parts in same project)
   - License/project type metadata

### Phase 2: Part Type Classification
**Train classifier to recognize part types from geometry**

Training:
```
Input: Mesh geometry features
Output: Part type (head, foot, hand, torso, base, accessory)

Model: Random Forest or Neural Network
Features: [bbox_ratio_xy, bbox_ratio_xz, volume, compactness, ...]
Labels: Extracted from filename patterns in training set
```

### Phase 3: Context-Aware Naming
**Combine geometry, context, and patterns**

```python
def generate_intelligent_name(mesh_file, folder_context):
    # 1. Extract geometric features
    features = extract_mesh_features(mesh_file)
    
    # 2. Classify part type (foot, hand, head, etc)
    part_type, confidence = classify_part_type(features)
    
    # 3. Get context from folder
    project_num, character = parse_folder_context(folder_context)
    
    # 4. Detect laterality (left/right) from geometry
    laterality = detect_laterality(mesh_file)
    
    # 5. Build intelligent name
    if project_num and character and part_type:
        name = f"{project_num}_{character}_{laterality}_{part_type}.{ext}"
        return name, confidence
    else:
        return suggest_best_match()
```

### Phase 4: Incremental Learning
**System gets smarter as user corrects**

User workflow:
1. System proposes: "300668_yoda_foot.stl" (70% confidence)
2. User corrects to: "300668_yoda_left_foot.stl"
3. System learns:
   - This geometry pattern = "left foot" not just "foot"
   - Updates training data
   - Improves future predictions

## Implementation Steps

### Step 1: Training Data Scanner
```python
class ArchiveTrainer:
    def scan_training_folders(self, paths):
        """Scan well-organized folders to build training set"""
        for folder in paths:
            for file in get_3d_files(folder):
                # Extract geometric features
                features = compute_features(file)
                
                # Parse naming pattern
                project, character, part = parse_filename(file.name)
                context = parse_folder_structure(file.parent)
                
                # Store training sample
                training_samples.append({
                    'features': features,
                    'part_type': part,
                    'project': project,
                    'character': character,
                    'context': context,
                    'filename': file.name
                })
```

### Step 2: Feature Extraction
```python
def extract_mesh_features(mesh_path):
    """Extract geometric features for ML"""
    mesh = trimesh.load(mesh_path)
    
    bbox = mesh.bounding_box.extents
    features = {
        'bbox_x': bbox[0],
        'bbox_y': bbox[1],
        'bbox_z': bbox[2],
        'aspect_xy': bbox[0] / bbox[1],
        'aspect_xz': bbox[0] / bbox[2],
        'aspect_yz': bbox[1] / bbox[2],
        'volume': mesh.volume,
        'surface_area': mesh.area,
        'compactness': mesh.volume / (mesh.area ** 1.5),
        'tri_count': len(mesh.faces),
        'convexity': mesh.volume / mesh.convex_hull.volume,
        'is_watertight': mesh.is_watertight,
        'principal_axis_1': pca_axis_1(mesh.vertices),
        'principal_axis_2': pca_axis_2(mesh.vertices),
        'centroid_z': mesh.centroid[2],  # Height indicator
    }
    return features
```

### Step 3: Part Type Classifier
```python
from sklearn.ensemble import RandomForestClassifier

class PartTypeClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)
        self.part_types = ['head', 'torso', 'arm', 'hand', 'leg', 
                          'foot', 'base', 'accessory', 'prop']
    
    def train(self, training_samples):
        X = [s['features'] for s in training_samples]
        y = [s['part_type'] for s in training_samples]
        self.model.fit(X, y)
    
    def predict(self, features):
        prediction = self.model.predict_proba([features])[0]
        part_type = self.part_types[prediction.argmax()]
        confidence = prediction.max()
        return part_type, confidence
```

### Step 4: Enhanced Proposal System
```python
def propose_intelligent_name(file_path, folder_context):
    # Load mesh and extract features
    features = extract_mesh_features(file_path)
    
    # Classify part type using ML
    part_type, geo_confidence = part_classifier.predict(features)
    
    # Parse folder context
    project_num = extract_project_number(folder_context)
    character = extract_character_name(folder_context)
    
    # Detect laterality
    laterality = detect_left_right(features)
    
    # Check reference database for confirmation
    ref_parts = load_reference_parts(project_num)
    text_match, text_confidence = fuzzy_match(part_type, ref_parts)
    
    # Combine confidences
    final_confidence = (geo_confidence * 0.6) + (text_confidence * 0.4)
    
    # Build final name
    if laterality:
        proposed = f"{project_num}_{character}_{laterality}_{part_type}"
    else:
        proposed = f"{project_num}_{character}_{part_type}"
    
    return proposed, final_confidence, {
        'geometry_confidence': geo_confidence,
        'text_confidence': text_confidence,
        'detected_type': part_type,
        'detected_laterality': laterality
    }
```

## UI Integration

### New Menu Items:
1. **Tools → Train from Archive** - Scan well-organized folders
2. **Tools → Retrain Model** - Update with user corrections
3. **View → Show Geometry Insights** - Debug what ML sees

### Enhanced Proposal Dialog:
```
┌─────────────────────────────────────────────────┐
│ 📦 part2.stl                                    │
│                                                 │
│ Geometric Analysis:                             │
│   Shape: Foot-like (85% confidence)             │
│   Laterality: Left (72% confidence)             │
│   Size: 120mm x 45mm x 200mm                    │
│                                                 │
│ Folder Context:                                 │
│   Project: 300668                               │
│   Character: Yoda                               │
│   Category: Character parts                     │
│                                                 │
│ Proposed Name:                                  │
│   300668_yoda_left_foot.stl                     │
│                                                 │
│ Combined Confidence: 82%                        │
│   └─ Geometry match: 85%                        │
│   └─ Text match: 75%                            │
│   └─ Context match: 90%                         │
└─────────────────────────────────────────────────┘
```

## Training Workflow

1. **Initial Training**:
   - User selects 3-5 well-organized project folders
   - System scans ~1000 files
   - Extracts geometric features + naming patterns
   - Trains initial classifier
   - Saves model to `db/part_classifier.pkl`

2. **Active Learning**:
   - User reviews proposals
   - Corrects incorrect names
   - System retrains on corrections
   - Model improves over time

3. **Transfer Learning**:
   - System learns project-specific patterns
   - Adapts to user's naming conventions
   - Recognizes license types from folder structure
   - Understands asset categories from context

## Database Schema Updates

```sql
-- Training samples table
CREATE TABLE training_samples (
    id INTEGER PRIMARY KEY,
    file_path TEXT,
    project_number TEXT,
    character_name TEXT,
    part_type TEXT,
    laterality TEXT,
    license_type TEXT,
    features_json TEXT,  -- JSON of geometric features
    timestamp TEXT,
    source TEXT  -- 'archive_scan' or 'user_correction'
);

-- Model performance tracking
CREATE TABLE model_metrics (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    accuracy REAL,
    precision REAL,
    recall REAL,
    training_samples_count INTEGER,
    model_version TEXT
);

-- User corrections for active learning
CREATE TABLE user_corrections (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    original_proposal TEXT,
    user_correction TEXT,
    file_path TEXT,
    confidence_before REAL,
    accepted BOOLEAN
);
```

## Benefits

1. **Intelligent naming**: Recognizes part types from geometry
2. **Context-aware**: Uses folder structure + project info
3. **Learns from you**: Improves with your corrections
4. **Handles unknowns**: Works on files with no reference data
5. **Laterality detection**: Automatically detects left/right
6. **Confidence scoring**: Shows why it made each decision
7. **Transfer learning**: Adapts to your specific conventions

## Future Enhancements

1. **Deep learning**: Use neural nets for complex part recognition
2. **Similar part search**: Find visually similar parts across projects
3. **Auto-categorization**: Detect character vs prop vs environment from geometry
4. **Quality scoring**: Flag low-quality meshes automatically
5. **Batch processing**: Train on entire archive overnight
6. **Model sharing**: Export trained model for team use

