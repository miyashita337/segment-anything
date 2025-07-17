# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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