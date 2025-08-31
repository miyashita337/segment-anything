#!/bin/bash
# 品質保証ワークフロー完全自動化スクリプト
# Usage: 
#   ./run_quality_workflow.sh TRACKER_ID                                    # 従来実行
#   ./run_quality_workflow.sh TRACKER_ID --use-subagent /input/dir          # SubAgent統合実行
#   ./run_quality_workflow.sh TRACKER_ID --subagent-register /input/dir     # SubAgent段階1: 登録
#   ./run_quality_workflow.sh TRACKER_ID --subagent-monitor                 # SubAgent段階2: 監視
#   ./run_quality_workflow.sh TRACKER_ID --subagent-collect                 # SubAgent段階3: 収集

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

# SubAgent実行モード確認
USE_SUBAGENT=false
SUBAGENT_STAGE=""
INPUT_DIR=""

# 既存の --use-subagent モード（保持）
if [ "$2" = "--use-subagent" ]; then
    USE_SUBAGENT=true
    INPUT_DIR="$3"
    echo "🤖 SubAgentモード有効: TaskOrchestratorを使用した統合実行"

# 新規の段階実行モード
elif [ "$2" = "--subagent-register" ]; then
    SUBAGENT_STAGE="register"
    INPUT_DIR="$3"
    echo "🚀 SubAgent段階1: タスク登録実行"
elif [ "$2" = "--subagent-monitor" ]; then
    SUBAGENT_STAGE="monitor"
    echo "👁️ SubAgent段階2: タスク監視実行"
elif [ "$2" = "--subagent-collect" ]; then
    SUBAGENT_STAGE="collect"
    echo "📊 SubAgent段階3: 結果収集実行"
fi

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
    
    # SubAgent実行モード分岐
    if [ "$USE_SUBAGENT" = true ]; then
        echo "🤖 SubAgent TaskOrchestratorによる実行"
        sam-env/bin/python3 tools/scripts/run_workflow_with_subagent.py \
            "$TRACKER_ID" \
            "$INPUT_DIR" \
            --max-files 10
        exit $?
    fi
    
    # SubAgent段階実行モード分岐
    if [ -n "$SUBAGENT_STAGE" ]; then
        echo "🎯 SubAgent段階実行: $SUBAGENT_STAGE"
        echo "   トラッカーID: $TRACKER_ID"
        echo "   実行段階: $SUBAGENT_STAGE"
        echo ""
        
        case "$SUBAGENT_STAGE" in
            "register")
                echo "🚀 段階1: タスク登録開始"
                if [ -z "$INPUT_DIR" ]; then
                    echo "❌ エラー: 入力ディレクトリが指定されていません"
                    echo "使用法: $0 $TRACKER_ID --subagent-register /path/to/input"
                    exit 1
                fi
                
                sam-env/bin/python3 tools/queue/async_stage_manager.py register \
                    "$TRACKER_ID" \
                    "$INPUT_DIR" \
                    --task-type extraction \
                    --max-files 10 \
                    --quality-method balanced
                exit $?
                ;;
                
            "monitor")
                echo "👁️ 段階2: タスク監視開始"
                sam-env/bin/python3 tools/queue/async_stage_manager.py monitor \
                    "$TRACKER_ID"
                exit $?
                ;;
                
            "collect")
                echo "📊 段階3: 結果収集開始"
                sam-env/bin/python3 tools/queue/async_stage_manager.py collect \
                    "$TRACKER_ID"
                exit $?
                ;;
                
            *)
                echo "❌ エラー: 不明な段階: $SUBAGENT_STAGE"
                exit 1
                ;;
        esac
    fi
    
    # QUAL-033: 厳密パス検証システム統合
    echo ""
    echo "🔍 QUAL-033 厳密パス検証システム"
    echo "   デフォルトパスは無効化されています。明示的にパスを指定してください。"
    echo ""
    
    # 入力ディレクトリの対話的入力
    while true; do
        echo "📁 画像入力ディレクトリを指定してください:"
        echo "   例: /mnt/c/AItools/lora/train/yado/org/kana08/"
        echo "   例: /mnt/c/AItools/lora/train/kiri/org/work01/"
        echo ""
        read -p "🔍 入力パス > " INPUT_DIR
        
        # 空入力チェック
        if [ -z "$INPUT_DIR" ]; then
            echo "❌ エラー: パスの入力が必要です（デフォルト値は無効化されています）"
            echo ""
            continue
        fi
        
        # 入力ディレクトリ存在チェック（QUAL-033準拠）
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
            echo ""
            continue
        fi
        
        # 画像ファイル存在チェック（QUAL-033準拠）
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
            echo ""
            continue
        fi
        
        # 検証成功
        echo "✅ 入力パス検証成功: $INPUT_DIR"
        echo "✅ 画像ファイル: $IMAGE_COUNT 枚を検出"
        echo ""
        break
    done
    
    EXTRACTION_LOG="${OUTPUT_DIR}/${TRACKER_ID}_extraction.log"
    
    # バックグラウンド実行開始（QUAL-033厳密検証統合）
    nohup sam-env/bin/python3 features/extraction/commands/extract_character.py \
        --mode reproduce-auto \
        --batch \
        --verbose \
        --max-files 10 \
        --strict-validation \
        --require-author-structure \
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

# 4-2. 統合ダッシュボード生成（一元化されたシステム）
echo "  📊 統合ダッシュボード生成..."
sam-env/bin/python3 tools/scripts/unified_dashboard_wrapper.py \
    "$TRACKER_ID" \
    "${OUTPUT_DIR}/extraction/" \
    "$OUTPUT_DIR"

# 統合ダッシュボード生成完了確認
if [ -f "${OUTPUT_DIR}/dashboard/dashboard.html" ]; then
    echo "✅ 統合ダッシュボード生成完了: http://100.123.241.106:8088/tracker/$TRACKER_ID"
    
    # ファイルサイズ表示
    DASHBOARD_SIZE=$(du -h "${OUTPUT_DIR}/dashboard/dashboard.html" | cut -f1)
    echo "  📄 ダッシュボードサイズ: $DASHBOARD_SIZE"
    echo "  🎯 システム: 統合ダッシュボード（一元化）"
else
    echo "⚠️  統合ダッシュボード生成に失敗しました"
fi

# 4-3. 客観指標テスト
echo "  🎯 客観指標テスト実行..."
sam-env/bin/python3 tools/core/run_objective_evaluation.py \
    --batch "${OUTPUT_DIR}/extraction/" \
    --output "${OUTPUT_DIR}/tests/objective_metrics_test.json"

# 5. 統合統計分析（INCI-004対応）
echo "🔬 統合統計分析実行中 (INCI-004)..."

# 最新の完了済みトラッカーをベースラインとして自動検索
BASELINE_TRACKER=""
if [ -d "$WORKSPACE_BASE" ]; then
    # トラッカーワークスペースを更新日時順でソート（最新順）
    LATEST_TRACKER=$(ls -t "$WORKSPACE_BASE" | grep -E "^(QUAL|INTG|INCI)-[0-9]" | grep -v "$TRACKER_ID" | head -n 1)
    
    if [ -n "$LATEST_TRACKER" ] && [ -f "${WORKSPACE_BASE}/${LATEST_TRACKER}/extraction_result.json" ]; then
        BASELINE_TRACKER="$LATEST_TRACKER"
        echo "🎯 ベースライントラッカー自動検出: $BASELINE_TRACKER"
        
        # INCI-004統合統計分析実行
        echo "  📊 Cohen's d計算・Welch t検定・Google Sheets更新..."
        sam-env/bin/python3 tools/progress_tracker/universal_statistical_analyzer.py \
            --current "$TRACKER_ID" \
            --baseline "$BASELINE_TRACKER" \
            --verbose > "${OUTPUT_DIR}/statistical_analysis_result.txt" 2>&1
        
        if [ $? -eq 0 ]; then
            echo "✅ 統合統計分析完了"
            echo "📊 分析結果: ${OUTPUT_DIR}/statistical_analysis_result.txt"
            
            # 結果の要約を表示
            if [ -f "${OUTPUT_DIR}/statistical_analysis_result.txt" ]; then
                echo "  📈 分析サマリー:"
                grep -E "(Cohen's d:|改善率:|統計的有意性:|実用的意義:)" "${OUTPUT_DIR}/statistical_analysis_result.txt" | sed 's/^/    /'
            fi
        else
            echo "⚠️  統合統計分析に失敗しました"
            echo "📝 詳細エラー: ${OUTPUT_DIR}/statistical_analysis_result.txt"
        fi
    else
        echo "⚠️  適切なベーストラッカーが見つかりません"
        echo "ℹ️  利用可能トラッカー: $(ls "$WORKSPACE_BASE" | grep -E "^(QUAL|INTG|INCI)-[0-9]" | grep -v "$TRACKER_ID" | tr '\n' ' ')"
        echo "ℹ️  統計分析をスキップしました。手動実行する場合："
        echo "    sam-env/bin/python3 tools/progress_tracker/universal_statistical_analyzer.py --current $TRACKER_ID --baseline BASELINE_ID"
    fi
else
    echo "⚠️  ワークスペースベースディレクトリが見つかりません: $WORKSPACE_BASE"
fi

# 6. 改善効果測定（レガシー・後方互換用）
echo "📊 改善効果測定（レガシー）..."
if [ -d "${WORKSPACE_BASE}/baseline" ] && [ "$(ls -A ${WORKSPACE_BASE}/baseline)" ]; then
    sam-env/bin/python3 generate_improvement_comparison.py \
        --baseline "${WORKSPACE_BASE}/baseline/" \
        --current "${OUTPUT_DIR}/" \
        --output "${OUTPUT_DIR}/improvement_report.json"
else
    echo "⚠️  レガシーベースラインデータが見つかりません。スキップします。"
fi

# 7. 実行結果サマリー生成
echo "📋 実行結果サマリー生成中..."
SUMMARY_FILE="${OUTPUT_DIR}/workflow_summary.txt"

cat > "$SUMMARY_FILE" << EOF
品質保証ワークフロー実行結果（INCI-004対応）
=============================================

トラッカーID: ${TRACKER_ID}
実行日時: $(date '+%Y-%m-%d %H:%M:%S')
実行者: $(whoami)
ベースライン: ${BASELINE_TRACKER:-"未設定"}

🎯 実行ステップ
✅ ワークスペース準備
$([ "$SKIP_EXTRACTION" = false ] && echo "✅ 抽出パイプライン実行" || echo "⏭️  抽出パイプライン（スキップ）")
✅ 抽出結果レポート生成
✅ 統合品質チェック
✅ ダッシュボード生成
✅ 客観指標テスト
$([ -f "${OUTPUT_DIR}/statistical_analysis_result.txt" ] && echo "✅ 統合統計分析（INCI-004）" || echo "⏭️  統合統計分析（ベースラインなし）")
$([ -f "${OUTPUT_DIR}/improvement_report.json" ] && echo "✅ 改善効果測定（レガシー）" || echo "⏭️  改善効果測定（レガシー）")

📁 生成ファイル
- 抽出結果: ${OUTPUT_DIR}/extraction/
- 品質レポート: ${OUTPUT_DIR}/quality/unified_quality_report.json
- ダッシュボード: ${OUTPUT_DIR}/dashboard/dashboard.html
- テスト結果: ${OUTPUT_DIR}/tests/
$([ -f "${OUTPUT_DIR}/statistical_analysis_result.txt" ] && echo "- 統計分析結果: ${OUTPUT_DIR}/statistical_analysis_result.txt")
$([ -f "${OUTPUT_DIR}/improvement_report.json" ] && echo "- 改善レポート: ${OUTPUT_DIR}/improvement_report.json")

📊 統計分析サマリー（INCI-004）
$([ -f "${OUTPUT_DIR}/statistical_analysis_result.txt" ] && {
    echo "$(grep -E "(Cohen's d:|改善率:|統計的有意性:|実用的意義:)" "${OUTPUT_DIR}/statistical_analysis_result.txt" | sed 's/^/- /')"
} || echo "- 統計分析が実行されませんでした")

📊 次のステップ（シリアル処理必須）
1. ダッシュボードで品質確認: file://${OUTPUT_DIR}/dashboard/dashboard.html
2. 統計分析結果確認: ${OUTPUT_DIR}/statistical_analysis_result.txt
3. Google Sheetsの統計列データ確認（X-AC列）
4. 実装完了報告テンプレートの記入
5. 品質劣化がないことを確認
6. Google Sheetsステータス更新（"/release"）
7. git commit（品質確認後のみ）
8. ✅ 次のトラッカーは現在のトラッカーが/releaseになってから開始すること

EOF

echo ""
echo "✅ 品質保証ワークフロー完了: ${TRACKER_ID}"
echo "📋 実行サマリー: $SUMMARY_FILE"
echo ""
echo "🔗 ダッシュボード: file://${OUTPUT_DIR}/dashboard/dashboard.html"
echo ""
echo "📋 ダッシュボード品質保証チェックリスト:"
echo "   詳細確認: docs/checklists/dashboard_quality_checklist.md"
echo "   🚨 毎回実行必須 - 統計データ・品質分布の完全性確認"
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