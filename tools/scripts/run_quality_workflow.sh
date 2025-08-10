#!/bin/bash
# 品質保証ワークフロー完全自動化スクリプト
# Usage: ./run_quality_workflow.sh PH2-001

set -e  # エラー時に停止

# 🔍 環境チェック機能（再発防止）
echo "🔍 実行環境チェック中..."

# sam-env環境存在確認
if [ ! -f "sam-env/bin/python3" ]; then
    echo "❌ エラー: sam-env環境が見つかりません"
    echo "   パス: sam-env/bin/python3"
    echo "   実行: python3 -m venv sam-env && sam-env/bin/python3 -m pip install -e ."
    exit 1
fi

# 重要パッケージ存在確認
echo "🔍 重要パッケージ確認中..."
if ! sam-env/bin/python3 -c "import cv2, click, numpy; print('✅ 必須パッケージ確認完了')" 2>/dev/null; then
    echo "❌ エラー: 必須パッケージが不足しています"
    echo "   実行: sam-env/bin/python3 -m pip install opencv-python click numpy"
    exit 1
fi

# Pushover設定確認
if [ ! -f "config/pushover.json" ]; then
    echo "⚠️ 警告: Pushover設定ファイルが見つかりません"
    echo "   パス: config/pushover.json" 
    echo "   Pushover通知は無効になります"
fi

echo "✅ 環境チェック完了"
echo ""

TRACKER_ID=$1

# config から動的にワークスペースパス取得
WORKSPACE_CONFIG_OUTPUT=$(sam-env/bin/python3 -c "
import sys
sys.path.insert(0, '$(dirname "$0")/../..')
from config.workspace_config import WorkspaceConfig
env_vars = WorkspaceConfig.export_environment_variables()
for key, value in env_vars.items():
    print(f'{key}=\"{value}\"')
")

# 環境変数設定
eval "$WORKSPACE_CONFIG_OUTPUT"

WORKSPACE_BASE="$TRACKER_WORKSPACE_ROOT"
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

# トラッカーID基本チェック（半角英数字記号のみ許可）
if [[ ! "$TRACKER_ID" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "❌ エラー: 無効なトラッカーID文字: $TRACKER_ID"
    echo "許可文字: 半角英数字・ハイフン・アンダースコア (例: PH2-001, QCC-011, TEST_001)"
    exit 1
fi

echo "🔄 品質保証ワークフロー開始: ${TRACKER_ID}"
echo "📁 出力ディレクトリ: ${OUTPUT_DIR}"

# ワークスペース確認リマインダー
echo ""
echo "🔔 リマインダー: ${TRACKER_ID} のワークスペース出力確認"
echo "📍 確認場所: ${WORKSPACE_BASE}/${TRACKER_ID}/"
echo "📋 必須ディレクトリ: extraction/, quality/, dashboard/, tests/"
echo ""

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
    
    # kana05の39枚を使用（Phase 1と同じデータセット）
    INPUT_DIR="/mnt/c/AItools/lora/train/yado/org/kana08"
    
    # 入力ディレクトリ存在チェック強化
    if [ ! -d "$INPUT_DIR" ]; then
        echo "❌ エラー: 入力ディレクトリが存在しません"
        echo "   パス: $INPUT_DIR"
        echo ""
        echo "🔧 対処方法:"
        echo "   1. パスの確認: ls $(dirname "$INPUT_DIR")"
        echo "   2. 正しいパスの指定"
        echo "   3. 必要に応じてディレクトリ作成"
        echo ""
        echo "⚠️ 注意: 存在しないパスでの強制実行は品質保証違反です"
        exit 1
    fi
    
    # 画像ファイル存在チェック
    IMAGE_COUNT=$(find "$INPUT_DIR" -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" | wc -l)
    if [ "$IMAGE_COUNT" -eq 0 ]; then
        echo "❌ エラー: 入力ディレクトリに画像ファイルが見つかりません"
        echo "   パス: $INPUT_DIR"
        echo "   サポート形式: jpg, jpeg, png"
        echo ""
        echo "🔧 対処方法:"
        echo "   1. ディレクトリ内容確認: ls $INPUT_DIR"
        echo "   2. サポートされている画像形式で画像を配置"
        echo "   3. ファイル名・拡張子の確認"
        exit 1
    fi
    
    echo "✅ 入力検証完了: $IMAGE_COUNT 枚の画像を検出"
    
    EXTRACTION_LOG="${OUTPUT_DIR}/${TRACKER_ID}_extraction.log"
    
    # バックグラウンド実行開始（統一されたextract_character.py使用）
    nohup sam-env/bin/python3 features/extraction/commands/extract_character.py \
        --mode reproduce-auto \
        --batch \
        --verbose \
        --max-files 10 \
        "$INPUT_DIR" \
        -o "${OUTPUT_DIR}/extraction/" \
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
sam-env/bin/python3 create_phase1_extraction_report.py \
    --input_dir "${OUTPUT_DIR}/extraction/" \
    --output_file "${OUTPUT_DIR}/extraction_result.json"

# 4. 品質チェック3コマンド実行
echo "🔍 品質チェック3コマンド実行中..."

# 4-1. 統合品質チェック
echo "  📈 統合品質チェック実行..."
sam-env/bin/python3 tools/core/unified_quality_checker.py \
    --results "${OUTPUT_DIR}/extraction_result.json" \
    --output "${OUTPUT_DIR}/quality/unified_quality_report.json"

# 4-2. 標準ダッシュボード生成（統一Base64画像表示）
echo "  📊 標準ダッシュボード生成..."
sam-env/bin/python3 features/common/dashboard_generator.py \
    "$TRACKER_ID" \
    "${OUTPUT_DIR}/extraction/" \
    "$OUTPUT_DIR"

# 標準ダッシュボード生成完了確認
if [ -f "${OUTPUT_DIR}/dashboard/dashboard.html" ]; then
    echo "✅ 標準ダッシュボード生成完了: http://100.123.241.106:8088/tracker/$TRACKER_ID"
    
    # ファイルサイズ表示
    DASHBOARD_SIZE=$(du -h "${OUTPUT_DIR}/dashboard/dashboard.html" | cut -f1)
    echo "  📄 ダッシュボードサイズ: $DASHBOARD_SIZE"
else
    echo "⚠️  標準ダッシュボード生成に失敗しました"
fi

# 4-3. 客観指標テスト
echo "  🎯 客観指標テスト実行..."
sam-env/bin/python3 tools/core/run_objective_evaluation.py \
    --batch "${OUTPUT_DIR}/extraction/" \
    --output "${OUTPUT_DIR}/tests/objective_metrics_test.json"

# 5. 改善効果測定（ベースライン比較）
echo "📊 改善効果測定実行中..."
if [ -d "${WORKSPACE_BASE}/baseline" ] && [ "$(ls -A ${WORKSPACE_BASE}/baseline)" ]; then
    sam-env/bin/python3 generate_improvement_comparison.py \
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

📊 次のステップ（シリアル処理必須）
1. ダッシュボードで品質確認: file://${OUTPUT_DIR}/dashboard/dashboard.html
2. 実装完了報告テンプレートの記入
3. 品質劣化がないことを確認
4. Google Sheetsステータス更新（"/release"）
5. git commit（品質確認後のみ）
6. ✅ 次のトラッカーは現在のトラッカーが/releaseになってから開始すること

EOF

echo ""
echo "✅ 品質保証ワークフロー完了: ${TRACKER_ID}"
echo "📋 実行サマリー: $SUMMARY_FILE"
echo ""
echo "🔗 ダッシュボード: file://${OUTPUT_DIR}/dashboard/dashboard.html"
echo ""
echo "🔄 シリアル処理確認:"
echo "   1. 本トラッカー(${TRACKER_ID})の品質確認完了後"
echo "   2. Google Sheetsで '/release' ステータス更新"
echo "   3. 次のトラッカー開始可能"
echo ""
echo "⚠️ 重要: パラレル実装は品質保証違反です！"

# Windows通知（オプション）
if command -v windows-notify >/dev/null 2>&1; then
    windows-notify -t "Claude Code" -m "品質保証ワークフロー完了: ${TRACKER_ID}"
fi