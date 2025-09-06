# Project Structure & Organization

## Directory Architecture

### Root Level Organization
```
segment-anything/
├── core/                    # Meta SAM implementation (DO NOT MODIFY)
├── features/               # Custom anime extraction features
├── tools/                  # Executable scripts and utilities
├── tests/                  # Test suite (unit + integration)
├── config/                 # Configuration files
├── docs/                   # Documentation
├── bin/shell/              # Shell scripts (linter, etc.)
├── deprecated/             # Legacy/obsolete files
└── logs/                   # Log files and execution history
```

## Core Principles

### 1. Separation of Concerns
- **core/**: Original Meta Facebook SAM implementation - keep unchanged
- **features/**: All custom anime character extraction functionality
- **tools/**: Direct execution scripts organized by purpose
- **tests/**: Comprehensive test coverage matching feature structure

### 2. Feature Organization
```
features/
├── extraction/             # Character extraction pipeline
│   ├── commands/          # CLI entry points
│   ├── models/            # SAM/YOLO wrapper classes  
│   └── pipeline/          # Processing pipeline
├── evaluation/            # Quality assessment system
│   ├── metrics/           # SCI, PLA, PLE calculators
│   └── utils/             # Evaluation utilities
├── processing/            # Pre/post-processing
└── common/                # Shared utilities
    ├── hooks/             # Initialization hooks
    ├── notification/      # Pushover integration
    └── performance/       # Performance monitoring
```

### 3. Tools Organization
```
tools/
├── batch/                 # Batch processing scripts
├── core/                  # Main pipeline executables
├── progress_tracker/      # Google Sheets integration
├── scripts/               # Utility scripts
├── testing/               # Test execution scripts
└── utils/                 # General utilities
```

## File Naming Conventions

### Python Files
- **Snake case**: `extract_character.py`, `quality_dashboard.py`
- **Descriptive names**: Clearly indicate functionality
- **Avoid abbreviations**: Use full words for clarity

### Configuration Files
- **YAML preferred**: `pipeline_config.yaml`, `author_config.yaml`
- **JSON for data**: API keys, structured data
- **Python for complex config**: `workspace_config.py`

### Test Files
- **Prefix with test_**: `test_extraction.py`, `test_quality_metrics.py`
- **Mirror structure**: Match the module being tested
- **Descriptive test names**: `test_character_extraction_pipeline()`

## Security & File Management

### Image File Security (CRITICAL)
```bash
# NEVER commit image files - treat as confidential
*.jpg
*.png
*.jpeg
*.bmp
*.tiff
*.webp

# Approved output paths only
/mnt/c/AItools/lora/train/yado/
```

### Model File Management
```bash
# Large model files (use Git LFS)
*.pth
*.pt
sam_vit_*.pth
yolov8*.pt
```

### Temporary File Cleanup
```bash
# Regular cleanup targets
*.pid
*.tmp
auto_execution_log.json
*_progress.json
```

## Import Conventions

### Relative Imports Within Features
```python
# Within features/ package
from .models.sam_wrapper import SAMWrapper
from ..common.notification import send_pushover
```

### Absolute Imports for Cross-Package
```python
# From tools/ to features/
from features.extraction.pipeline import ExtractionPipeline
from features.evaluation.metrics import calculate_sci
```

### Core SAM Integration
```python
# Always use absolute imports for core SAM
from segment_anything import SamPredictor, sam_model_registry
```

## Documentation Structure

### Required Documentation
- **README.md**: Project overview and quick start
- **CHANGELOG.md**: Version history and changes
- **docs/workflows/**: Development process documentation
- **Function docstrings**: All public functions must have docstrings

### Documentation Standards
```python
def extract_character(image_path: str, output_dir: str) -> bool:
    """Extract character from image using SAM+YOLO pipeline.
    
    Args:
        image_path: Path to input image file
        output_dir: Directory for extracted character images
        
    Returns:
        True if extraction successful, False otherwise
        
    Raises:
        FileNotFoundError: If image_path doesn't exist
        ValueError: If output_dir is not writable
    """
```

## Quality Standards

### Code Organization Rules
1. **Single Responsibility**: Each module has one clear purpose
2. **Dependency Direction**: Features can use core, not vice versa
3. **Configuration Driven**: Avoid hardcoded paths/values
4. **Error Handling**: Comprehensive error handling and logging
5. **Type Hints**: Use type hints for all function signatures

### File Size Guidelines
- **Python modules**: Keep under 500 lines when possible
- **Configuration files**: Split large configs into logical sections
- **Test files**: One test file per module, group related tests

### Prohibited Patterns
- **Circular imports**: Features importing from tools
- **Hardcoded paths**: Use configuration files
- **Global state**: Minimize global variables
- **Deep nesting**: Keep directory nesting under 4 levels