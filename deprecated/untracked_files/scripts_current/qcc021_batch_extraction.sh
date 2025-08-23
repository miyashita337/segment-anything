#!/bin/bash
# QCC-021: 統計的妥当性達成用バッチ抽出（シンプル版）
# 既存のextract_character.pyを活用した424枚→379枚以上の抽出

set -e  # エラー時停止

echo "================================================================================"
echo "🎯 QCC-021: 統計的妥当性達成用バッチ抽出開始"
echo "   必要サンプル数: 379枚追加（推奨393枚）"
echo "   入力総数: 424枚（yado: 266枚、kiri: 158枚）"
echo "================================================================================"

# 出力ディレクトリ設定
OUTPUT_DIR="/mnt/c/AItools/lora/train/yado/tracker-workspace/QCC-021-EXTENDED/extraction"
mkdir -p "$OUTPUT_DIR"

echo "📁 出力先: $OUTPUT_DIR"
echo ""

# プロジェクトルートに移動・PYTHONPATH設定
cd /mnt/c/AItools/segment-anything
export PYTHONPATH=/mnt/c/AItools/segment-anything:$PYTHONPATH

# カウンタ初期化
total_before=$(ls "$OUTPUT_DIR"/*.jpg 2>/dev/null | wc -l || echo 0)
echo "🔢 開始時抽出済み: ${total_before}枚"

# yado作者ディレクトリ（7個）
echo ""
echo "👤 yado作者処理開始..."
for dir in kana03 kana04 kana05 kana06 kana07 kana08 kana09; do
    input_path="/mnt/c/AItools/lora/train/yado/org/$dir"
    echo "  🔄 処理中: $dir"
    
    if [ ! -d "$input_path" ]; then
        echo "  ❌ ディレクトリ未発見: $input_path"
        continue
    fi
    
    # extract_character.py実行（バックグラウンド）
    nohup python3 features/extraction/commands/extract_character.py \
        "$input_path" \
        -o "$OUTPUT_DIR" \
        --batch \
        --verbose > "$OUTPUT_DIR/${dir}_extraction.log" 2>&1 &
    
    extraction_pid=$!
    echo "  🔄 バックグラウンド実行開始: PID $extraction_pid"
    
    # 完了待機
    wait $extraction_pid || {
        echo "  ⚠️ エラーが発生しましたが続行します: $dir"
    }
    
    echo "  ✅ 完了: $dir"
done

echo ""
echo "🎨 kiri作者処理開始..."
kiri_path="/mnt/c/AItools/lora/train/kiri/aichikan"
if [ -d "$kiri_path" ]; then
    echo "  🔄 処理中: aichikan"
    nohup python3 features/extraction/commands/extract_character.py \
        "$kiri_path" \
        -o "$OUTPUT_DIR" \
        --batch \
        --verbose > "$OUTPUT_DIR/aichikan_extraction.log" 2>&1 &
    
    extraction_pid=$!
    echo "  🔄 バックグラウンド実行開始: PID $extraction_pid"
    
    # 完了待機
    wait $extraction_pid || {
        echo "  ⚠️ エラーが発生しましたが続行します: aichikan"
    }
    echo "  ✅ 完了: aichikan"
else
    echo "  ❌ ディレクトリ未発見: $kiri_path"
fi

echo ""
echo "================================================================================"
echo "🏁 QCC-021 バッチ抽出完了"

# 結果集計
total_after=$(ls "$OUTPUT_DIR"/*.jpg 2>/dev/null | wc -l || echo 0)
newly_extracted=$((total_after - total_before))

echo "📊 結果サマリー:"
echo "   抽出前: ${total_before}枚"
echo "   抽出後: ${total_after}枚"
echo "   新規抽出: ${newly_extracted}枚"

# 統計的妥当性判定
if [ "$total_after" -ge 379 ]; then
    echo "🎯 ✅ 統計的妥当性達成！（379枚以上）"
    if [ "$total_after" -ge 393 ]; then
        echo "🌟 推奨サンプル数も達成！（393枚以上）"
    fi
else
    shortage=$((379 - total_after))
    echo "⚠️ 統計的妥当性未達成（あと${shortage}枚必要）"
fi

echo ""
echo "📋 出力先: $OUTPUT_DIR"
echo "🔗 品質ワークフロー実行用コマンド:"
echo "   ./tools/scripts/run_quality_workflow.sh QCC-021-EXTENDED"
echo "================================================================================"