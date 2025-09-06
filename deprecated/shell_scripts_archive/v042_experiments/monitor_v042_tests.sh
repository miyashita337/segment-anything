#!/bin/bash

# v0.0.42 複合スコア機能テスト監視スクリプト

echo "=== v0.0.42 複合スコア機能テスト監視 ==="
echo "開始時刻: $(date)"

while true; do
    echo ""
    echo "=== $(date) ==="
    
    # 実行中プロセス確認
    echo "📊 実行中プロセス:"
    PROCESSES=$(ps aux | grep sam_yolo_character_segment | grep -v grep | wc -l)
    echo "   アクティブプロセス数: $PROCESSES"
    
    if [ $PROCESSES -eq 0 ]; then
        echo "✅ 全プロセス完了"
        break
    fi
    
    # 各基準の進捗確認
    echo "📁 出力ファイル数:"
    
    for variant in "balanced" "size" "fullbody" "central" "confidence"; do
        if [ "$variant" = "balanced" ]; then
            dir="/mnt/c/AItools/lora/train/yadokugaeru/clipped_boundingbox/kaname03_0_0_42"
        else
            dir="/mnt/c/AItools/lora/train/yadokugaeru/clipped_boundingbox/kaname03_0_0_42_$variant"
        fi
        
        if [ -d "$dir" ]; then
            count=$(ls -1 "$dir"/*.jpg 2>/dev/null | wc -l)
            echo "   $variant: $count 枚"
        else
            echo "   $variant: ディレクトリなし"
        fi
    done
    
    # 最新ログ確認
    echo "📝 最新処理状況:"
    if [ -f "kaname03_0_0_42_balanced.log" ]; then
        last_line=$(tail -1 kaname03_0_0_42_balanced.log | grep "進捗:")
        if [ ! -z "$last_line" ]; then
            echo "   balanced: $last_line"
        fi
    fi
    
    sleep 30
done

echo ""
echo "🎉 全テスト完了時刻: $(date)"

# 最終結果サマリ
echo ""
echo "=== 最終結果サマリ ==="
for variant in "balanced" "size" "fullbody" "central" "confidence"; do
    if [ "$variant" = "balanced" ]; then
        dir="/mnt/c/AItools/lora/train/yadokugaeru/clipped_boundingbox/kaname03_0_0_42"
    else
        dir="/mnt/c/AItools/lora/train/yadokugaeru/clipped_boundingbox/kaname03_0_0_42_$variant"
    fi
    
    if [ -d "$dir" ]; then
        count=$(ls -1 "$dir"/*.jpg 2>/dev/null | wc -l)
        echo "✅ $variant: $count 枚完了"
    else
        echo "❌ $variant: 失敗"
    fi
done