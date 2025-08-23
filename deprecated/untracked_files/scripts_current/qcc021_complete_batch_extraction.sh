#!/bin/bash
# QCC-021完全バッチ抽出: 8ディレクトリ×424枚→379枚以上で統計的妥当性達成

set -e

echo "=========================================="
echo "🎯 QCC-021 完全バッチ抽出開始"
echo "   目標: 424枚→379枚以上抽出で統計的妥当性達成"
echo "   方法: 修復済みextract_character.py使用"
echo "=========================================="
echo ""

# 出力ディレクトリ設定
OUTPUT_BASE="/mnt/c/AItools/lora/train/yado/tracker-workspace/QCC-021-EXTENDED"
EXTRACTION_DIR="${OUTPUT_BASE}/extraction"
mkdir -p "$EXTRACTION_DIR"

# プロジェクトルートに移動
cd /mnt/c/AItools/segment-anything
export PYTHONPATH=/mnt/c/AItools/segment-anything:$PYTHONPATH

echo "📁 出力先: $EXTRACTION_DIR"
echo ""

# 8つの入力ディレクトリ定義（424枚合計）
declare -a INPUT_DIRS=(
    "/mnt/c/AItools/lora/train/yado/org/kana03"    # 25枚
    "/mnt/c/AItools/lora/train/yado/org/kana04"    # 28枚  
    "/mnt/c/AItools/lora/train/yado/org/kana05"    # 38枚
    "/mnt/c/AItools/lora/train/yado/org/kana06"    # 31枚
    "/mnt/c/AItools/lora/train/yado/org/kana07"    # 41枚
    "/mnt/c/AItools/lora/train/yado/org/kana08"    # 43枚
    "/mnt/c/AItools/lora/train/yado/org/kana09"    # 56枚
    "/mnt/c/AItools/lora/train/kiri/aichikan"      # 162枚（推定）
)

# 開始時点での抽出済み数確認
initial_count=$(find "$EXTRACTION_DIR" -name "*.jpg" | wc -l)
echo "🔢 開始時抽出済み: ${initial_count}枚"
echo ""

# 各ディレクトリを順次処理
total_processed=0
for i in "${!INPUT_DIRS[@]}"; do
    input_dir="${INPUT_DIRS[$i]}"
    dir_name=$(basename "$input_dir")
    
    echo "============================================"
    echo "📂 処理中 ($(($i + 1))/8): $dir_name"
    echo "   入力: $input_dir"
    echo "============================================"
    
    # ディレクトリ存在確認
    if [ ! -d "$input_dir" ]; then
        echo "⚠️ スキップ: ディレクトリが存在しません - $input_dir"
        continue
    fi
    
    # 画像数確認
    image_count=$(find "$input_dir" -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" | wc -l)
    if [ "$image_count" -eq 0 ]; then
        echo "⚠️ スキップ: 画像ファイルが見つかりません - $input_dir"
        continue
    fi
    
    echo "✅ 検出: ${image_count}枚の画像"
    
    # extract_character.py実行
    extraction_log="${OUTPUT_BASE}/${dir_name}_extraction.log"
    before_count=$(find "$EXTRACTION_DIR" -name "*.jpg" | wc -l)
    
    echo "🚀 抽出開始: $(date '+%H:%M:%S')"
    
    # 修復済みextract_character.pyで抽出実行
    python3 features/extraction/commands/extract_character.py \
        "$input_dir" \
        -o "$EXTRACTION_DIR" \
        --batch \
        --verbose \
        --enable-author-adaptation \
        > "$extraction_log" 2>&1
    
    extraction_result=$?
    after_count=$(find "$EXTRACTION_DIR" -name "*.jpg" | wc -l)
    extracted_this_round=$((after_count - before_count))
    
    if [ $extraction_result -eq 0 ]; then
        echo "✅ 完了: ${dir_name} - ${extracted_this_round}枚抽出"
        total_processed=$((total_processed + extracted_this_round))
    else
        echo "❌ エラー: ${dir_name} - ログ確認: $extraction_log"
    fi
    
    echo "📊 累計抽出: ${after_count}枚"
    echo ""
    
    # 379枚達成チェック
    if [ "$after_count" -ge 379 ]; then
        echo "🎯✅ 統計的妥当性達成！（379枚以上）"
        echo "   現在: ${after_count}枚"
        break
    fi
done

echo "=========================================="
echo "🏁 QCC-021 完全バッチ抽出完了"
echo ""

# 最終結果
final_count=$(find "$EXTRACTION_DIR" -name "*.jpg" | wc -l)
newly_extracted=$((final_count - initial_count))

echo "📊 最終結果:"
echo "   開始時: ${initial_count}枚"
echo "   最終: ${final_count}枚"  
echo "   新規抽出: ${newly_extracted}枚"
echo ""

# 統計的妥当性判定
if [ "$final_count" -ge 379 ]; then
    echo "🎯✅ 統計的妥当性達成！"
    if [ "$final_count" -ge 393 ]; then
        echo "🌟 推奨サンプル数も達成！（393枚以上）"
    fi
else
    shortage=$((379 - final_count))
    echo "⚠️ 統計的妥当性未達成（あと${shortage}枚必要）"
fi

echo ""
echo "📁 抽出結果: $EXTRACTION_DIR"
echo "🔗 次のステップ: ./tools/scripts/run_quality_workflow.sh QCC-021-EXTENDED"
echo "=========================================="

# 完了通知
if command -v windows-notify >/dev/null 2>&1; then
    windows-notify -t "Claude Code" -m "QCC-021完全バッチ抽出完了: ${final_count}枚抽出（目標379枚）"
fi