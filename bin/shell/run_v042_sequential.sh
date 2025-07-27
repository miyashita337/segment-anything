#!/bin/bash

# v0.0.42 複合スコア機能 - 安全なシーケンシャル実行スクリプト

echo "=== v0.0.42 複合スコア機能テスト - シーケンシャル実行 ==="
echo "開始時刻: $(date)"

# 基本パラメータ
INPUT_DIR="/mnt/c/AItools/lora/train/yadokugaeru/org/kaname03"
OUTPUT_BASE="/mnt/c/AItools/lora/train/yadokugaeru/clipped_boundingbox"
SCRIPT="python3 sam_yolo_character_segment.py"

# 実行基準リスト
CRITERIA=("balanced" "size_priority" "fullbody_priority" "central_priority" "confidence_priority")
SUFFIXES=("" "_size" "_fullbody" "_central" "_confidence")

# 各基準を順次実行
for i in "${!CRITERIA[@]}"; do
    criteria="${CRITERIA[$i]}"
    suffix="${SUFFIXES[$i]}"
    output_dir="${OUTPUT_BASE}/kaname03_0_0_42${suffix}"
    log_file="kaname03_0_0_42_${criteria}.log"
    
    echo ""
    echo "🎯 基準: $criteria"
    echo "📁 出力: $output_dir"
    echo "📝 ログ: $log_file"
    
    # 出力ディレクトリ作成
    mkdir -p "$output_dir"
    
    # メモリ状況確認
    echo "💾 実行前メモリ状況:"
    free -h | grep "Mem:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -1
    
    # 実行開始
    echo "🚀 実行開始: $(date)"
    
    $SCRIPT --mode reproduce-auto \
        --input_dir "$INPUT_DIR" \
        --output_dir "$output_dir" \
        --multi_character_criteria "$criteria" \
        > "$log_file" 2>&1
    
    exit_code=$?
    echo "📊 実行終了: $(date), 終了コード: $exit_code"
    
    # 結果確認
    if [ -d "$output_dir" ]; then
        file_count=$(ls -1 "$output_dir"/*.jpg 2>/dev/null | wc -l)
        echo "✅ 出力ファイル数: $file_count 枚"
    else
        echo "❌ 出力ディレクトリが作成されていません"
    fi
    
    # ログサイズ確認
    if [ -f "$log_file" ]; then
        log_size=$(wc -l < "$log_file")
        echo "📄 ログ行数: $log_size 行"
        
        # エラーチェック
        error_count=$(grep -c "❌\|Error\|Exception" "$log_file" 2>/dev/null || echo "0")
        if [ "$error_count" -gt 0 ]; then
            echo "⚠️ エラー検出: $error_count 個"
            echo "最新エラー:"
            grep "❌\|Error\|Exception" "$log_file" | tail -3
        fi
    fi
    
    # メモリクリア
    echo "🧹 メモリクリア"
    python3 -c "
import gc
import torch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('GPU メモリクリア完了')
"
    
    # 60秒待機（プロセス分離）
    if [ $((i + 1)) -lt ${#CRITERIA[@]} ]; then
        echo "⏰ 60秒待機中..."
        sleep 60
    fi
done

echo ""
echo "🎉 全テスト完了: $(date)"

# 最終結果サマリ
echo ""
echo "=== 最終結果サマリ ==="
for i in "${!CRITERIA[@]}"; do
    criteria="${CRITERIA[$i]}"
    suffix="${SUFFIXES[$i]}"
    output_dir="${OUTPUT_BASE}/kaname03_0_0_42${suffix}"
    
    if [ -d "$output_dir" ]; then
        file_count=$(ls -1 "$output_dir"/*.jpg 2>/dev/null | wc -l)
        echo "✅ $criteria: $file_count 枚完了"
    else
        echo "❌ $criteria: 失敗"
    fi
done

echo ""
echo "📊 詳細ログ:"
for criteria in "${CRITERIA[@]}"; do
    log_file="kaname03_0_0_42_${criteria}.log"
    if [ -f "$log_file" ]; then
        echo "  $log_file: $(wc -l < "$log_file") 行"
    fi
done