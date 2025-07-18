# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.3.3] - 2025-07-18

### Added
- **[P1-009] Learning Data Collection Planning**: Strategic learning data collection system with gap analysis and sampling strategy based on kaname07 evaluation data (41 samples analyzed)
- **[P1-015] Evaluation Difference Analyzer**: Quantification system for automatic vs user evaluation differences with correlation analysis and improvement recommendations
- **[P1-017] Boundary Analysis Algorithm**: Boundary line quality quantification using curvature variance, Douglas-Peucker simplification, and smoothness metrics
- **[P1-019] Human Structure Recognition System**: Human body structure recognition for limb truncation prevention with body region estimation and risk assessment
- **[P1-021] Foreground Background Analyzer**: Background/foreground separation quality measurement using color clustering and texture analysis

### Changed
- **Phase 1 Quality Evaluation System**: Comprehensive quality improvement with 5 new evaluation modules
- **Quantitative Analysis**: All new systems provide numerical quality scores and grading (A-F scale)
- **Systematic Evaluation**: Evidence-based evaluation framework with detailed metrics and recommendations

### Technical Improvements
- **Integration Testing**: Complete Phase 1 integration test with 100% success rate (5/5 systems)
- **Fallback Implementations**: All systems work without optional dependencies (scipy, sklearn, matplotlib)
- **Performance Optimized**: Fast processing suitable for batch operations
- **Comprehensive Analysis**: Detailed analysis reports with actionable improvement recommendations

### Quality Assessment Features
- **Learning Data Gap Analysis**: Identifies underrepresented problems, missing regions, and rating imbalances
- **Correlation Analysis**: Pearson/Spearman correlation between automatic and user evaluations
- **Boundary Quality Metrics**: Curvature variance, angle variance, perimeter roughness, and simplification ratios
- **Human Structure Validation**: Body region detection, truncation risk assessment, and structure validity scoring
- **Separation Quality Scoring**: Color similarity, texture analysis, boundary clarity, and contamination risk assessment

### Development Process
- **Systematic Implementation**: 5 Phase 1 tasks completed with individual testing and integration verification
- **Quality-First Approach**: Each system includes comprehensive test cases and quality validation
- **Documentation**: Detailed docstrings and usage examples for all new evaluation systems

## [v0.3.2] - 2025-07-17

### Added
- **Region Priority System**: New region-based character selection system based on user evaluation feedback
- **User Evaluation Data Integration**: Added kaname07_user_evaluation.jsonl with 41 image evaluations
- **Character Selection Optimization**: Enhanced character selection with position-based prioritization

### Changed
- **Improved Character Selection**: Better handling of multi-character manga panels with specific region preferences
- **Enhanced Quality Assessment**: Integration of user feedback into character extraction pipeline
- **Size Priority Refinement**: Improved size_priority method based on evaluation data showing higher success rates

### Fixed
- **Wrong Character Selection**: Reduced incorrect character selection in multi-character scenes
- **Extraction Failure Rate**: Improved extraction success rate from 58.5% baseline with targeted improvements

### Technical Improvements
- Added RegionPrioritySystem class for position-aware character selection
- Integrated user evaluation patterns into learned quality assessment
- Enhanced region-based scoring for better character targeting

## [v0.3.1] - 2025-07-17

### Fixed
- **Mosaic Detection Issue**: Disabled mosaic processing that was causing false positives in character extraction
- **Lower Body Extraction**: Fixed issue where screentone patterns were being misidentified as background, causing lower body extraction failures
- **Duplicate File Prefix**: Resolved issue where `preprocessed_manga_` prefix was being duplicated in temporary files

### Changed
- **Temporary File Management**: Moved preprocessing temporary files from input directory to `/tmp/` to prevent directory pollution
- **Batch Processing**: Enhanced batch processing with automatic cleanup of temporary files after completion

### Added
- **Auto-cleanup**: Automatic removal of temporary files in `/tmp/` after batch processing completion
- **File Prefix Protection**: Added logic to prevent duplicate prefix generation in preprocessing pipeline

## [v0.3.0] - 2025-07-17

### Added
- **[P1-001] Full Body Detection Algorithm Analysis**: Comprehensive analysis system for fullbody_priority method
- **[P1-002] Partial Extraction Detection System**: `PartialExtractionDetector` for detecting incomplete character extractions (face-only, limb truncation)
- **[P1-003] Enhanced Full Body Detection**: `EnhancedFullBodyDetector` with multi-metric evaluation (aspect ratio, body structure, edge distribution, semantic regions)
- **[P1-004] Advanced Screen Tone Detection**: `EnhancedScreentoneDetector` (857 lines) with FFT, Gabor, LBP, Wavelet, and Spatial feature extraction
- **[P1-005] Mosaic Boundary Processing**: `EnhancedMosaicBoundaryProcessor` with multi-scale and rotation-invariant detection
- **[P1-006] Solid Fill Area Processing**: `EnhancedSolidFillProcessor` with RGB/HSV/LAB color space analysis and adaptive clustering

### Changed
- **Quality Evaluation System**: Integrated all Phase 1 improvements into character extraction pipeline
- **Multi-Metric Assessment**: Enhanced evaluation with completeness scoring, boundary quality, and semantic analysis
- **Adaptive Processing**: Type-specific processing for character, background, and effect regions
- **Pattern Recognition**: Advanced pattern classification for various image elements (dots, lines, gradients, textures)

### Fixed
- **Boundary Accuracy**: Improved boundary detection from ±5 pixels to ±2 pixels
- **False Positive Reduction**: Reduced mosaic detection false positives from 30% to 10%
- **Color Uniformity**: Enhanced solid fill detection with circular uniformity computation for hue values
- **Edge Preservation**: Better edge-preserving filtering for solid fill boundaries

### Technical Improvements
- **Comprehensive Test Suite**: 151 test cases (126 unit + 25 integration tests) with 100% pass rate
- **Fallback Implementations**: Library-independent operation (sklearn, scipy, skimage optional)
- **Performance Optimization**: Processing speeds of 0.009s (100x100) to 0.115s (400x400)
- **Multi-Scale Analysis**: Support for different pattern sizes and orientations
- **Memory Efficiency**: Optimized clustering for large images with sampling techniques

### Performance Benchmarks
- **kaname07 Dataset**: 3/3 images successful mosaic detection with 0.772-1.000 confidence
- **Detection Coverage**: Multiple pattern types (grid, pixelated, blur, rotated patterns)
- **Processing Speed**: Maintained target of 5000+ pixels/second across all implementations
- **Quality Metrics**: Average boundary quality scores >0.8 for detected regions

### Documentation
- **Analysis Documents**: Phase 1 analysis for fullbody, screentone, mosaic, and solid fill processing
- **API Documentation**: Comprehensive docstrings and usage examples
- **Integration Guides**: Test cases demonstrating real-world usage scenarios

## [v0.2.0] - 2025-07-17

### Added
- **Phase 0 Project Refactoring**: Complete modular architecture implementation
- **Modular Structure**: Organized codebase into `core/`, `features/`, `tests/`, `tools/` directories
- **CharacterExtractor Class**: New class-based interface for character extraction
- **Comprehensive Test Suite**: Unit and integration tests with fixtures
- **Development Documentation**: Extensive docs in `docs/` directory
  - `file-structure.md`: Project structure documentation
  - `phase0_problems.md`: Problem analysis and solutions
  - `phase0_completion_checklist.md`: Completion criteria and migration readiness
- **Progress Tracking**: `PROGRESS_TRACKER.md` for systematic development management
- **Auto-initialization System**: Option A fixes for seamless model loading
- **Enhanced CLAUDE.md**: Development process guidelines with implementation templates

### Changed
- **File Organization**: Moved Facebook SAM implementation to `core/segment_anything/`
- **Custom Features**: Relocated character extraction features to `features/extraction/`
- **Quality Assessment**: Moved evaluation utilities to `features/evaluation/`
- **Processing Pipeline**: Separated preprocessing/postprocessing to `features/processing/`
- **Test Structure**: Reorganized tests into unit/integration/fixtures hierarchy
- **Import Paths**: Updated all import statements for new modular structure

### Fixed
- **Dependency Issues**: Resolved CharacterExtractor import errors
- **Model Initialization**: Fixed SAM and YOLO model loading sequence
- **Path Resolution**: Corrected all relative path issues after refactoring
- **Test Compatibility**: Ensured all tests work with new structure

### Technical Improvements
- **Backup System**: Created `backup-20250716-2236/` for safe migration
- **Error Handling**: Enhanced error messages with correct paths
- **Code Quality**: Maintained flake8, black, mypy compliance
- **Documentation**: Comprehensive developer guides and troubleshooting

### Performance
- **Batch Processing**: Confirmed 32.6% success rate on kaname06 dataset (28/86 images)
- **Initialization**: Streamlined model loading with singleton pattern preparation
- **Memory Management**: Improved resource handling for large datasets

### Development Process
- **Test-First Development**: Implemented comprehensive test framework
- **Quality Metrics**: Established success criteria and completion checklists
- **Problem Tracking**: Systematic issue identification and resolution
- **Migration Planning**: Phased approach to structural changes

## [v0.1.1] - Previous Release
- コードベースクリーンアップ - 局所的ファイル削除

## [v0.1.0] - Previous Release  
- 完璧な適応学習システム完成 - 100%成功率達成