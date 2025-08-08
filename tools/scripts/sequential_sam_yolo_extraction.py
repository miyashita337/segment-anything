#!/usr/bin/env python3
"""
Sequential SAM+YOLO Extraction for P1-B004
Memory-efficient single-file processing approach
"""

import subprocess
import sys
import time
from pathlib import Path

def run_single_extraction(input_file, output_file):
    """Run single file extraction with SAM+YOLO"""
    cmd = [
        sys.executable,
        "features/extraction/commands/extract_character.py",
        str(input_file),
        "-o", str(output_file),
        "--verbose"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"SUCCESS: {input_file.name}")
            return True
        else:
            print(f"FAILED: {input_file.name}")
            print(f"Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {input_file.name}")
        return False
    except Exception as e:
        print(f"ERROR: {input_file.name} - {e}")
        return False

def main():
    print("=" * 60)
    print("Sequential SAM+YOLO Extraction for P1-B004")
    print("=" * 60)
    
    input_dir = Path("C:/AItools/lora/train/yado/org/kana08")
    output_dir = Path("C:/AItools/lora/train/yado/tracker-workspace/P1-B004/extraction")
    
    # Get all input files
    input_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    input_files = [f for f in input_files if not f.name.startswith('.')]
    input_files.sort()
    
    print(f"Total files: {len(input_files)}")
    
    # Check existing outputs
    existing_outputs = list(output_dir.glob("*_extracted.*"))
    existing_names = {f.name.replace('_extracted.jpg', '').replace('_extracted.png', '') for f in existing_outputs}
    
    print(f"Already processed: {len(existing_outputs)}")
    
    success_count = 0
    total_files = len(input_files)
    
    for i, input_file in enumerate(input_files, 1):
        base_name = input_file.stem
        if base_name in existing_names:
            print(f"[{i:2d}/{total_files}] SKIP: {input_file.name} (already exists)")
            success_count += 1
            continue
        
        print(f"[{i:2d}/{total_files}] PROCESSING: {input_file.name}")
        output_file = output_dir / f"{base_name}_extracted.jpg"
        
        if run_single_extraction(input_file, output_file):
            success_count += 1
        
        # Brief pause between files to allow memory cleanup
        time.sleep(2)
        
        # Progress update every 5 files
        if i % 5 == 0:
            print(f"Progress: {success_count}/{i} files completed ({success_count/i*100:.1f}%)")
    
    print("=" * 60)
    print(f"FINAL RESULT: {success_count}/{total_files} files processed")
    
    # Final verification
    final_outputs = list(output_dir.glob("*_extracted.*"))
    print(f"Final output count: {len(final_outputs)}")
    
    if len(final_outputs) >= 24:  # Allow some tolerance
        print("SUCCESS: Most files processed successfully")
        return 0
    else:
        print("PARTIAL: Some files may need retry")
        return 1

if __name__ == "__main__":
    exit(main())