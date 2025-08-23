#!/bin/bash

echo "Starting extraction for all 26 images..."

INPUT_DIR="C:/AItools/lora/train/yado/org/kana08"
OUTPUT_DIR="C:/AItools/lora/train/yado/tracker-workspace/TEST-20250803/extraction"

count=0
success=0

for file in "$INPUT_DIR"/*.jpg; do
    if [ -f "$file" ]; then
        filename=$(basename "$file" .jpg)
        count=$((count + 1))
        echo "[$count/26] Processing: $filename.jpg"
        
        sam-env/bin/python3 features/extraction/commands/extract_character.py \
            "$file" \
            -o "$OUTPUT_DIR/${filename}_extracted.jpg" \
            2>/dev/null
        
        if [ $? -eq 0 ]; then
            echo "  SUCCESS"
            success=$((success + 1))
        else
            echo "  FAILED"
        fi
    fi
done

echo "================================"
echo "Completed: $success/$count files"
echo "================================"