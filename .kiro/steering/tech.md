# Technology Stack & Build System

## Core Technologies

### Machine Learning Framework
- **PyTorch**: Primary ML framework (>=1.7.0)
- **Segment Anything**: Meta's SAM model for image segmentation
- **Ultralytics YOLOv8**: Character detection (>=8.0.0)
- **OpenCV**: Image processing (>=4.5.0)

### Python Environment
- **Python**: >=3.8 required
- **Package Management**: pip with requirements.txt
- **Virtual Environment**: sam-env (recommended)

### Key Dependencies
```
torch>=1.7.0
torchvision>=0.8.1
opencv-python>=4.5.0
ultralytics>=8.0.0
segment-anything
scikit-image>=0.18.0
```

### Development Tools
- **Code Quality**: black (==23.*), isort (==5.12.0), flake8, mypy
- **Testing**: pytest (>=6.0.0)
- **Linting**: Automated via bin/shell/linter.sh

## Build & Development Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv sam-env
source sam-env/bin/activate  # Linux/Mac
# sam-env\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .  # Install in development mode
```

### Code Quality & Testing
```bash
# Run full linting suite
./bin/shell/linter.sh

# Individual tools
black -l 100 .           # Code formatting
isort . --atomic         # Import sorting
flake8 .                 # Style checking
mypy --exclude 'setup.py|notebooks' .  # Type checking

# Run tests
python -m pytest tests/
```

### Main Execution Commands
```bash
# Character extraction (main pipeline)
python features/extraction/commands/extract_character.py

# Quality dashboard generation
python tools/core/quality_dashboard.py

# Batch processing
python tools/batch/kana08_enhanced_stable_batch.py

# Interactive extraction
python features/extraction/commands/quick_interactive.py
```

### Model Management
```bash
# Download SAM models (required)
# Models: sam_vit_h_4b8939.pth, sam_vit_b_01ec64.pth
# Place in project root

# YOLO models (auto-downloaded)
# yolov8n.pt, yolov8x.pt, yolov8x6_animeface.pt
```

## Configuration Management

### Key Config Files
- `config/pipeline_config.yaml`: Main pipeline configuration
- `config/workspace_config.py`: Workspace paths and settings
- `config/author_config.yaml`: Author-specific settings
- `setup.py`: Package configuration and dependencies

### Environment Variables
- CUDA support strongly recommended for GPU acceleration
- Minimum 8GB RAM, 6GB VRAM recommended

## Architecture Patterns

### Directory Structure
- `core/`: Original Meta SAM implementation (unchanged)
- `features/`: Custom anime character extraction features
- `tools/`: Executable scripts and utilities
- `tests/`: Test suite (unit + integration)
- `config/`: Configuration files

### Code Organization
- **Separation of Concerns**: Core SAM vs custom features
- **Pipeline Architecture**: Modular extraction pipeline
- **Quality-First**: Comprehensive testing and evaluation
- **Configuration-Driven**: YAML-based configuration system