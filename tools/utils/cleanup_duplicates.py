#!/usr/bin/env python3
"""
Cleanup script for duplicate suffix files in QCC-011 workspace.

Purpose: Remove or rename files with duplicate suffixes like:
- extracted_kana08_0001_multi_char_detection_multi_char_detection.jpg
- Extract correct single-suffix version if not exists

Created for: QCC-011 duplicate processing issue resolution
"""

import os
import shutil
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from features.common.file_utils import clean_duplicate_suffixes


def cleanup_duplicate_files(directory: str, dry_run: bool = True):
    """
    Clean up duplicate suffix files in the specified directory.

    Args:
        directory: Directory to clean up
        dry_run: If True, only print what would be done without making changes
    """
    directory_path = Path(directory)
    if not directory_path.exists():
        print(f"❌ Directory not found: {directory}")
        return

    print(f"🧹 Cleaning up duplicate files in: {directory}")
    print(f"📋 Mode: {'DRY RUN' if dry_run else 'ACTUAL EXECUTION'}")

    duplicate_files = []

    # Find all files with duplicate suffixes
    for file_path in directory_path.glob("*_multi_char_detection_multi_char_detection*"):
        duplicate_files.append(file_path)

    if not duplicate_files:
        print("✅ No duplicate suffix files found")
        return

    print(f"🔍 Found {len(duplicate_files)} duplicate suffix files:")

    for duplicate_file in duplicate_files:
        print(f"  📁 {duplicate_file.name}")

        # Generate clean filename
        clean_filename = clean_duplicate_suffixes(duplicate_file.name)
        clean_path = duplicate_file.parent / clean_filename

        print(f"     → {clean_filename}")

        if not dry_run:
            try:
                # Check if clean version already exists
                if clean_path.exists():
                    # Compare file sizes to decide which to keep
                    duplicate_size = duplicate_file.stat().st_size
                    clean_size = clean_path.stat().st_size

                    if duplicate_size > clean_size:
                        print(f"     ✅ Replacing smaller clean file with larger duplicate")
                        shutil.move(str(duplicate_file), str(clean_path))
                    else:
                        print(
                            f"     🗑️ Removing duplicate (clean version exists and is larger/equal)"
                        )
                        duplicate_file.unlink()
                else:
                    # Rename duplicate to clean version
                    print(f"     ✅ Renaming to clean version")
                    shutil.move(str(duplicate_file), str(clean_path))

            except Exception as e:
                print(f"     ❌ Error: {e}")
        else:
            # Dry run - just show what would happen
            if clean_path.exists():
                duplicate_size = duplicate_file.stat().st_size
                clean_size = clean_path.stat().st_size
                if duplicate_size > clean_size:
                    print(
                        f"     → Would replace smaller clean file ({clean_size} bytes) with larger duplicate ({duplicate_size} bytes)"
                    )
                else:
                    print(
                        f"     → Would remove duplicate ({duplicate_size} bytes), clean exists ({clean_size} bytes)"
                    )
            else:
                print(f"     → Would rename to clean version")

    print()
    print("📊 Summary:")
    print(f"  Files processed: {len(duplicate_files)}")
    if not dry_run:
        print("  ✅ Cleanup completed successfully")
    else:
        print("  📋 Dry run completed - no changes made")
        print("  🚀 To execute changes, run with --execute flag")


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Cleanup duplicate suffix files in QCC-011 workspace"
    )
    parser.add_argument(
        "--directory",
        "-d",
        default="/mnt/c/AItools/lora/train/yado/tracker-workspace/QCC-011/extraction/",
        help="Directory to clean up",
    )
    parser.add_argument(
        "--execute", action="store_true", help="Actually perform the cleanup (default is dry run)"
    )

    args = parser.parse_args()

    print("🧹 QCC-011 Duplicate File Cleanup Tool")
    print("=" * 50)

    cleanup_duplicate_files(args.directory, dry_run=not args.execute)


if __name__ == "__main__":
    main()
