#!/bin/bash

# v0.0.42 複合スコア機能 - Resume機能付き実行スクリプト
# 既存完了分をスキップし、未完了分のみ実行

echo "=== v0.0.42 複合スコア機能 - Resume実行 ==="
echo "開始時刻: $(date)"

# 基本パラメータ
INPUT_DIR="/mnt/c/AItools/lora/train/yadokugaeru/org/kaname03"
OUTPUT_BASE="/mnt/c/AItools/lora/train/yadokugaeru/clipped_boundingbox"
SCRIPT="python3 sam_yolo_character_segment.py"

# 実行基準リスト
CRITERIA=("balanced" "size_priority" "fullbody_priority" "central_priority" "confidence_priority")
SUFFIXES=("" "_size" "_fullbody" "_central" "_confidence")

# 完了判定関数
check_completion() {
    local output_dir="$1"
    local criteria="$2"
    
    if [ ! -d "$output_dir" ]; then
        echo "❌ ディレクトリ未作成"
        return 1
    fi
    
    local file_count=$(ls -1 "$output_dir"/*.jpg 2>/dev/null | wc -l)
    if [ "$file_count" -gt 0 ]; then
        echo "✅ 完了済み ($file_count 枚)"
        return 0
    else
        echo "⚠️ 未完了 (0 枚)"
        return 1
    fi
}

# 実行時間予測
total_pending=0
for i in "${!CRITERIA[@]}"; do
    criteria="${CRITERIA[$i]}"
    suffix="${SUFFIXES[$i]}"
    output_dir="${OUTPUT_BASE}/kaname03_0_0_42${suffix}"
    
    if ! check_completion "$output_dir" "$criteria" >/dev/null 2>&1; then
        ((total_pending++))
    fi
done

echo "📊 実行予測:"
echo "   未完了基準数: $total_pending"
echo "   予想実行時間: $((total_pending * 30))分"

# 各基準の状況確認と実行
executed_count=0
for i in "${!CRITERIA[@]}"; do
    criteria="${CRITERIA[$i]}"
    suffix="${SUFFIXES[$i]}"
    output_dir="${OUTPUT_BASE}/kaname03_0_0_42${suffix}"
    log_file="kaname03_0_0_42_${criteria}.log"
    
    echo ""
    echo "🎯 基準: $criteria ($((i+1))/${#CRITERIA[@]})"
    echo "📁 出力: $output_dir"
    echo "📝 ログ: $log_file"
    
    # 完了状況確認
    echo -n "🔍 完了状況: "
    if check_completion "$output_dir" "$criteria"; then
        echo "⏭️ スキップ（既に完了済み）"
        continue
    fi
    
    # 実行前準備
    mkdir -p "$output_dir"
    
    # システム状況確認
    echo "💾 実行前状況:"
    free -h | grep "Mem:" | awk '{print "   メモリ使用: " $3 "/" $2}'
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -1 | awk -F',' '{print "   GPU VRAM: " $1 "MB/" $2 "MB"}'
    fi
    
    # 実行開始
    echo "🚀 実行開始: $(date)"
    start_time=$(date +%s)
    
    $SCRIPT --mode reproduce-auto \
        --input_dir "$INPUT_DIR" \
        --output_dir "$output_dir" \
        --multi_character_criteria "$criteria" \
        > "$log_file" 2>&1
    
    exit_code=$?
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo "📊 実行終了: $(date)"
    echo "⏱️ 実行時間: ${duration}秒 ($((duration/60))分$((duration%60))秒)"
    echo "🔢 終了コード: $exit_code"
    
    # 結果確認
    if [ -d "$output_dir" ]; then
        file_count=$(ls -1 "$output_dir"/*.jpg 2>/dev/null | wc -l)
        if [ "$file_count" -gt 0 ]; then
            echo "✅ 処理成功: $file_count 枚出力"
            ((executed_count++))
        else
            echo "❌ 処理失敗: ファイル出力なし"
        fi
    else
        echo "❌ 処理失敗: 出力ディレクトリ未作成"
    fi
    
    # ログ確認
    if [ -f "$log_file" ]; then
        log_size=$(wc -l < "$log_file")
        echo "📄 ログサイズ: $log_size 行"
        
        # エラーチェック
        error_count=$(grep -c "❌\|Error\|Exception\|エラー" "$log_file" 2>/dev/null || echo "0")
        if [ "$error_count" -gt 0 ]; then
            echo "⚠️ エラー検出: $error_count 個"
            echo "最新エラー:"
            grep "❌\|Error\|Exception\|エラー" "$log_file" | tail -2
        fi
    fi
    
    # メモリクリア（次の実行のため）
    if [ $((i + 1)) -lt ${#CRITERIA[@]} ]; then
        echo "🧹 メモリクリア中..."
        python3 -c "
import gc
import torch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('GPU メモリクリア完了')
else:
    print('メモリクリア完了')
" 2>/dev/null
        
        echo "⏰ 30秒待機中..."
        sleep 30
    fi
done

echo ""
echo "🎉 Resume実行完了: $(date)"
echo "📊 今回実行分: $executed_count 基準"

# 最終結果サマリ
echo ""
echo "=== 全基準実行状況 ==="
total_completed=0
for i in "${!CRITERIA[@]}"; do
    criteria="${CRITERIA[$i]}"
    suffix="${SUFFIXES[$i]}"
    output_dir="${OUTPUT_BASE}/kaname03_0_0_42${suffix}"
    
    if [ -d "$output_dir" ]; then
        file_count=$(ls -1 "$output_dir"/*.jpg 2>/dev/null | wc -l)
        if [ "$file_count" -gt 0 ]; then
            echo "✅ $criteria: $file_count 枚完了"
            ((total_completed++))
        else
            echo "❌ $criteria: 失敗 (0 枚)"
        fi
    else
        echo "❌ $criteria: 未実行"
    fi
done

echo ""
echo "📈 全体完了率: $total_completed/${#CRITERIA[@]} 基準 ($((total_completed * 100 / ${#CRITERIA[@]}))%)"

if [ "$total_completed" -eq "${#CRITERIA[@]}" ]; then
    echo "🎯 全基準完了！次は結果比較分析を実行してください"
    exit 0
else
    echo "⚠️ 未完了の基準があります。ログを確認してください"
    exit 1
fi