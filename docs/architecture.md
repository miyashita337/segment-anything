# System Architecture

## High-Level Components

1. Input Processing
   - Image loading and validation
   - Format conversion
   - Resolution adjustment

2. Detection Pipeline
   - YOLO character detection
   - Bounding box generation
   - Confidence filtering

3. Segmentation Pipeline
   - SAM model initialization
   - Prompt generation from YOLO boxes
   - Mask generation and refinement

4. Post-Processing
   - Mask cleanup and optimization
   - Character extraction
   - Output generation

## Data Flow
1. Input image → YOLO detection
2. YOLO boxes → SAM prompts
3. SAM prompts → Segmentation masks
4. Masks → Extracted characters

## Key Interfaces
- CLI interface
- Configuration management
- Logging system
- Progress tracking
- Error handling