#!/usr/bin/env python3
"""
P1-B004: Simple Extraction Continuance (Unicode Safe)
"""

import numpy as np
import cv2

import sys
import time
from pathlib import Path


def simple_extract(input_path, output_path):
    """Simple SAM-like extraction without dependencies"""
    try:
        # Read image
        img = cv2.imread(str(input_path))
        if img is None:
            print(f"ERROR: Failed to read {input_path}")
            return False

        h, w = img.shape[:2]
        print(f"Processing: {input_path.name} ({w}x{h})")

        # Simple center-crop like approach (emergency fallback)
        # Use edge detection for basic character extraction
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply basic threshold for character detection
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Find contours (basic character detection)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find largest contour (main character)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w_box, h_box = cv2.boundingRect(largest_contour)

            # Extract with padding
            pad = 20
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + w_box + pad)
            y2 = min(h, y + h_box + pad)

            extracted = img[y1:y2, x1:x2]
        else:
            # Fallback: center crop
            crop_size = min(w, h) * 0.8
            cx, cy = w // 2, h // 2
            x1 = max(0, int(cx - crop_size // 2))
            y1 = max(0, int(cy - crop_size // 2))
            x2 = min(w, int(cx + crop_size // 2))
            y2 = min(h, int(cy + crop_size // 2))
            extracted = img[y1:y2, x1:x2]

        # Resize to standard size
        if extracted.size > 0:
            extracted = cv2.resize(extracted, (512, 512))

            # Save as JPG (same format as old QC-KANA08)
            output_path = output_path.with_suffix(".jpg")
            success = cv2.imwrite(str(output_path), extracted, [cv2.IMWRITE_JPEG_QUALITY, 95])

            if success:
                print(f"SUCCESS: {output_path.name}")
                return True
            else:
                print(f"ERROR: Failed to save {output_path}")
                return False
        else:
            print(f"ERROR: Empty extraction for {input_path.name}")
            return False

    except Exception as e:
        print(f"ERROR: {e}")
        return False


def main():
    print("=" * 60)
    print("P1-B004: Simple Extraction Continuance")
    print("=" * 60)

    input_dir = Path("/mnt/c/AItools/lora/train/yado/org/kana08")
    output_dir = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace/P1-B004/extraction")

    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all input files
    input_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    input_files = [f for f in input_files if not f.name.startswith(".")]

    print(f"Input files: {len(input_files)}")

    # Check what's already done
    existing_outputs = list(output_dir.glob("*_extracted.*"))
    existing_names = {
        f.name.replace("_extracted.jpg", "").replace("_extracted.png", "") for f in existing_outputs
    }

    print(f"Already processed: {len(existing_outputs)}")

    # Process remaining files
    success_count = 0
    for i, input_file in enumerate(input_files, 1):
        base_name = input_file.stem
        if base_name in existing_names:
            print(f"[{i:2d}/{len(input_files)}] SKIP: {input_file.name} (already processed)")
            success_count += 1
            continue

        print(f"[{i:2d}/{len(input_files)}] ", end="")
        output_path = output_dir / f"{base_name}_extracted"

        if simple_extract(input_file, output_path):
            success_count += 1

        time.sleep(0.1)  # Brief pause

    print("=" * 60)
    print(f"COMPLETE: {success_count}/{len(input_files)} files processed")

    # Final verification
    final_outputs = list(output_dir.glob("*_extracted.*"))
    print(f"Final output count: {len(final_outputs)}")

    return 0 if success_count == len(input_files) else 1


if __name__ == "__main__":
    exit(main())
