#!/bin/bash
# 品質保証ワークフロー完全自動化スクリプト
# Usage: ./run_quality_workflow.sh PH2-001

set -e  # エラー時に停止

TRACKER_ID=$1
WORKSPACE_BASE="/mnt/c/AItools/lora/train/yado/clipped_boundingbox/workspace"
OUTPUT_DIR="${WORKSPACE_BASE}/${TRACKER_ID}"

# 引数チェック
if [ -z "$TRACKER_ID" ]; then
    echo "❌ エラー: トラッカーIDを指定してください"
    echo "使用法: ./run_quality_workflow.sh PH2-001"
    echo ""
    echo "利用可能なトラッカーID:"
    echo "  PH2-001: システム全体性能評価・ボトルネック特定"
    echo "  PH2-002: アーキテクチャ最適化・安定性確保"
    exit 1
fi

# トラッカーID形式チェック
if [[ ! "$TRACKER_ID" =~ ^PH[0-9]+-[0-9]{3}$ ]]; then
    echo "❌ エラー: 無効なトラッカーID形式: $TRACKER_ID"
    echo "正しい形式: PH{Phase番号}-{3桁連番} (例: PH2-001)"
    exit 1
fi

echo "🔄 品質保証ワークフロー開始: ${TRACKER_ID}"
echo "📁 出力ディレクトリ: ${OUTPUT_DIR}"

# 1. ワークスペース準備
echo "📂 ワークスペース準備中..."
mkdir -p "${OUTPUT_DIR}"/{extraction,quality,dashboard,tests}

# 既存の抽出結果確認
if [ -d "${OUTPUT_DIR}/extraction" ] && [ "$(ls -A ${OUTPUT_DIR}/extraction)" ]; then
    echo "ℹ️  既存の抽出結果が見つかりました: ${OUTPUT_DIR}/extraction"
    read -p "抽出パイプラインをスキップしますか? (y/N): " skip_extraction
    if [[ $skip_extraction =~ ^[Yy]$ ]]; then
        SKIP_EXTRACTION=true
    else
        SKIP_EXTRACTION=false
    fi
else
    SKIP_EXTRACTION=false
fi

# 2. 抽出パイプライン実行（バックグラウンド）
if [ "$SKIP_EXTRACTION" = false ]; then
    echo "🚀 抽出パイプライン開始（バックグラウンド実行）"
    echo "⚠️  Windows ハングアップ防止のため、バックグラウンド実行します"
    
    # kana05の10枚を使用（Phase 1と同じデータセット）
    INPUT_DIR="/mnt/c/AItools/segment-anything/test_sample_kana05"
    
    if [ ! -d "$INPUT_DIR" ]; then
        echo "❌ エラー: 入力ディレクトリが見つかりません: $INPUT_DIR"
        echo "入力画像を準備してください"
        exit 1
    fi
    
    EXTRACTION_LOG="${OUTPUT_DIR}/${TRACKER_ID}_extraction.log"
    
    # バックグラウンド実行開始
    nohup python3 tools/sam_yolo_character_segment.py \
        --mode reproduce-auto \
        --input_dir "$INPUT_DIR" \
        --output_dir "${OUTPUT_DIR}/extraction/" \
        > "$EXTRACTION_LOG" 2>&1 &
    
    EXTRACTION_PID=$!
    echo "📊 抽出プロセスID: $EXTRACTION_PID"
    echo "📝 実行ログ: $EXTRACTION_LOG"
    
    # プロセス監視開始
    echo "⏳ 抽出パイプライン実行中... (進捗監視)"
    
    # 3分間隔で進捗確認
    while kill -0 $EXTRACTION_PID 2>/dev/null; do
        echo "🔄 $(date '+%H:%M:%S') - 抽出処理継続中..."
        if [ -f "$EXTRACTION_LOG" ]; then
            tail -n 3 "$EXTRACTION_LOG" | sed 's/^/    /'
        fi
        sleep 180  # 3分待機
    done
    
    # 完了確認
    wait $EXTRACTION_PID
    EXTRACTION_EXIT_CODE=$?
    
    if [ $EXTRACTION_EXIT_CODE -eq 0 ]; then
        echo "✅ 抽出パイプライン完了"
    else
        echo "❌ 抽出パイプライン失敗 (終了コード: $EXTRACTION_EXIT_CODE)"
        echo "📝 ログを確認してください: $EXTRACTION_LOG"
        exit 1
    fi
else
    echo "⏭️  抽出パイプラインをスキップしました"
fi

# 3. 抽出結果レポート生成
echo "📊 抽出結果レポート生成中..."
python3 create_phase1_extraction_report.py \
    --input_dir "${OUTPUT_DIR}/extraction/" \
    --output_file "${OUTPUT_DIR}/extraction_result.json"

# 4. 品質チェック3コマンド実行
echo "🔍 品質チェック3コマンド実行中..."

# 4-1. 統合品質チェック
echo "  📈 統合品質チェック実行..."
python3 tools/unified_quality_checker.py \
    --results "${OUTPUT_DIR}/extraction_result.json" \
    --output "${OUTPUT_DIR}/quality/unified_quality_report.json"

# 4-2. ダッシュボード生成
echo "  📊 ダッシュボード生成..."
if [ -f "${OUTPUT_DIR}/quality/unified_quality_report.json" ]; then
    python3 tools/quality_dashboard.py \
        --report "${OUTPUT_DIR}/quality/unified_quality_report.json" \
        --output "${OUTPUT_DIR}/dashboard/"
else
    echo "⚠️  統合品質レポートが見つかりません。ダッシュボード生成をスキップします。"
fi

# 4-3. 客観指標テスト
echo "  🎯 客観指標テスト実行..."
python3 tools/run_objective_evaluation.py \
    --batch "${OUTPUT_DIR}/extraction/" \
    --output "${OUTPUT_DIR}/tests/objective_metrics_test.json"

# 5. 改善効果測定（ベースライン比較）
echo "📊 改善効果測定実行中..."
if [ -d "${WORKSPACE_BASE}/baseline" ] && [ "$(ls -A ${WORKSPACE_BASE}/baseline)" ]; then
    python3 generate_improvement_comparison.py \
        --baseline "${WORKSPACE_BASE}/baseline/" \
        --current "${OUTPUT_DIR}/" \
        --output "${OUTPUT_DIR}/improvement_report.json"
else
    echo "⚠️  ベースラインデータが見つかりません。比較分析をスキップします。"
    echo "ℹ️  初回実行時は正常です。今回の結果をベースラインとして保存できます。"
fi

# 6. 実行結果サマリー生成
echo "📋 実行結果サマリー生成中..."
SUMMARY_FILE="${OUTPUT_DIR}/workflow_summary.txt"

cat > "$SUMMARY_FILE" << EOF
品質保証ワークフロー実行結果
================================

トラッカーID: ${TRACKER_ID}
実行日時: $(date '+%Y-%m-%d %H:%M:%S')
実行者: $(whoami)

🎯 実行ステップ
✅ ワークスペース準備
$([ "$SKIP_EXTRACTION" = false ] && echo "✅ 抽出パイプライン実行" || echo "⏭️  抽出パイプライン（スキップ）")
✅ 抽出結果レポート生成
✅ 統合品質チェック
✅ ダッシュボード生成
✅ 客観指標テスト
$([ -f "${OUTPUT_DIR}/improvement_report.json" ] && echo "✅ 改善効果測定" || echo "⏭️  改善効果測定（ベースラインなし）")

📁 生成ファイル
- 抽出結果: ${OUTPUT_DIR}/extraction/
- 品質レポート: ${OUTPUT_DIR}/quality/unified_quality_report.json
- ダッシュボード: ${OUTPUT_DIR}/dashboard/dashboard.html
- テスト結果: ${OUTPUT_DIR}/tests/
$([ -f "${OUTPUT_DIR}/improvement_report.json" ] && echo "- 改善レポート: ${OUTPUT_DIR}/improvement_report.json")

📊 次のステップ
1. ダッシュボードで品質確認: file://${OUTPUT_DIR}/dashboard/dashboard.html
2. 実装完了報告テンプレートの記入
3. 品質劣化がないことを確認
4. git commit（品質確認後のみ）

EOF

echo ""
echo "✅ 品質保証ワークフロー完了: ${TRACKER_ID}"
echo "📋 実行サマリー: $SUMMARY_FILE"
echo ""
echo "🔗 ダッシュボード: file://${OUTPUT_DIR}/dashboard/dashboard.html"
echo ""

# Windows通知（オプション）
if command -v windows-notify >/dev/null 2>&1; then
    windows-notify -t "Claude Code" -m "品質保証ワークフロー完了: ${TRACKER_ID}"
fi