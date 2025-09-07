#!/bin/bash
# KIRO-003 バッチ処理スクリプト

set -e

INPUT_LIST="/mnt/c/AItools/lora/train/kiri/tracker-workspace/KIRO-003/input_files.txt"
OUTPUT_DIR="/mnt/c/AItools/lora/train/kiri/tracker-workspace/KIRO-003/extraction"
LOG_FILE="/mnt/c/AItools/lora/train/kiri/tracker-workspace/KIRO-003/extraction_log.txt"

# 仮想環境アクティベート
source sam-env/bin/activate

echo "=== KIRO-003 品質重視抽出開始 ===" | tee -a "$LOG_FILE"
echo "開始時刻: $(date)" | tee -a "$LOG_FILE"

SUCCESS_COUNT=0
TOTAL_COUNT=0

# 各ファイルを処理
while IFS= read -r input_file; do
    if [ -f "$input_file" ]; then
        TOTAL_COUNT=$((TOTAL_COUNT + 1))
        BASENAME=$(basename "$input_file" .jpg)
        OUTPUT_FILE="${OUTPUT_DIR}/${BASENAME}_extracted.jpg"
        
        echo "[$TOTAL_COUNT/30] 処理中: $input_file" | tee -a "$LOG_FILE"
        
        # 抽出実行（出力はディレクトリ指定、検証緩和）
        if python features/extraction/commands/extract_character.py \
            "$input_file" \
            -o "$OUTPUT_DIR" \
            --quality-method balanced \
            --sam-optimization-profile p1_020_balanced \
            --no-strict-validation \
            >> "$LOG_FILE" 2>&1; then
            
            if [ -f "$OUTPUT_FILE" ]; then
                SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
                echo "  ✅ 成功: $OUTPUT_FILE" | tee -a "$LOG_FILE"
            else
                echo "  ❌ 失敗: 出力ファイルなし" | tee -a "$LOG_FILE"
            fi
        else
            echo "  ❌ エラー: コマンド実行失敗" | tee -a "$LOG_FILE"
        fi
    fi
done < "$INPUT_LIST"

echo "=== 処理完了 ===" | tee -a "$LOG_FILE"
echo "成功: $SUCCESS_COUNT / $TOTAL_COUNT" | tee -a "$LOG_FILE"
echo "成功率: $(echo "scale=1; $SUCCESS_COUNT * 100 / $TOTAL_COUNT" | bc)%" | tee -a "$LOG_FILE"
echo "終了時刻: $(date)" | tee -a "$LOG_FILE"