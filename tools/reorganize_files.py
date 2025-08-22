"""Script to reorganize project files into appropriate directories."""
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

def get_file_categories() -> Dict[str, List[str]]:
    return {
        'docs': [
            'BACKUP_PLAN.md', 'CHANGELOG.md', 'CODE_OF_CONDUCT.md',
            'CONTRIBUTING.md', 'HUMAN_LABEL_LEARNING_PROJECT.md',
            'PH2-001_ROOT_CAUSE_ANALYSIS.md', 'PROGRESS_TRACKER.md',
            'PROJECT_SETTINGS.md', 'QC_COMPREHENSIVE_REPORT.md',
            'README_Phase3.md', 'folder_structure.md',
            'gemini_competition_context.md', 'gpt4o_consultation_summary.md',
            'lora_training_evaluation.md'
        ],
        'tests': [
            'test_evaluation_report.txt', 'test_extract_with_notification.py',
            'test_extraction_notification.py', 'test_grayscale_fix.py',
            'test_p1_005_demo.py', 'test_p1_006_demo.py',
            'test_p1_007_demo.py', 'test_p1_010_demo.py',
            'test_p1_015_demo.py', 'test_pose_landmark_visualization.py',
            'test_pushover.py', 'test_pushover_notification.py',
            'test_small_character_area_threshold.py'
        ],
        'deprecated': [
            'P1_016_batch_log.txt', 'auto_execution_log.json',
            'baseline_extraction.log', 'phase1_extraction.pid',
            'phase2_completion_message.txt', 'phase2_fixed_completion_message.txt',
            'priority_highest.json', 'progress_history.json',
            'quality_history.json', 'sam-env-windows-backup.txt',
            'small_character_threshold_test_20250726_170937.json'
        ],
        'features': [
            'analyze_detection_failures.py', 'analyze_quality_trend.py',
            'auto_progress_system.py', 'benchmark_human_vs_ai.py',
            'create_phase1_extraction_report.py', 'final_visual_summary.py',
            'fix_dashboard_all_images_path.py', 'fix_qcc021_extended.py',
            'generate_improvement_comparison.py', 'integrated_dashboard_server.py',
            'send_pushover_images.py'
        ]
    }

def create_directories(root_dir: Path) -> None:
    """Create necessary directories if they don't exist."""
    dirs = ['docs', 'tests', 'deprecated', 'features']
    for dir_name in dirs:
        dir_path = root_dir / dir_name
        dir_path.mkdir(exist_ok=True)

def move_files(root_dir: Path, categories: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    """Move files to their designated directories using git mv."""
    moved_files = []
    for category, files in categories.items():
        for file in files:
            src = root_dir / file
            if not src.exists():
                continue
            dst = root_dir / category / file
            if src.exists():
                os.system(f'git mv {src} {dst}')
                moved_files.append((str(src), str(dst)))
    return moved_files

def update_imports(moved_files: List[Tuple[str, str]]) -> None:
    """Update import statements in Python files."""
    for src, dst in moved_files:
        if dst.endswith('.py'):
            with open(dst, 'r') as f:
                content = f.read()
            # Update relative imports
            updated_content = content.replace('from .', 'from ...')
            with open(dst, 'w') as f:
                f.write(updated_content)

def main() -> None:
    root_dir = Path(__file__).parent.parent
    categories = get_file_categories()
    create_directories(root_dir)
    moved_files = move_files(root_dir, categories)
    update_imports(moved_files)

if __name__ == '__main__':
    main()