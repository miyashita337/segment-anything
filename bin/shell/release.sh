#!/bin/bash
# /release コマンド実装
# Usage: ./release.sh [PH2-001] [--patch|--minor|--major]

set -e

TRACKER_ID=$1
VERSION_TYPE=${2:-"--patch"}  # デフォルトはパッチバージョン
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# 引数チェック
if [ -z "$TRACKER_ID" ]; then
    echo "❌ エラー: トラッカーIDを指定してください"
    echo "使用法: ./release.sh PH2-001 [--patch|--minor|--major]"
    exit 1
fi

# PROGRESS_TRACKER.md のパス
TRACKER_FILE="/mnt/c/AItools/segment-anything/docs/workflows/PROGRESS_TRACKER.md"
WORKSPACE_BASE="/mnt/c/AItools/lora/train/yado/clipped_boundingbox/workspace"
TASK_DIR="${WORKSPACE_BASE}/${TRACKER_ID}"

echo "🚀 リリースプロセス開始: ${TRACKER_ID}"
echo "📁 タスクディレクトリ: ${TASK_DIR}"

# 1. タスク完了確認
echo "📋 タスク完了確認中..."

if [ ! -d "$TASK_DIR" ]; then
    echo "❌ エラー: タスクディレクトリが見つかりません: $TASK_DIR"
    exit 1
fi

# 必須ファイル確認
REQUIRED_FILES=(
    "extraction_result.json"
    "quality/unified_quality_report.json"
    "dashboard/dashboard.html"
    "tests/objective_metrics_test.json"
)

MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "${TASK_DIR}/${file}" ]; then
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    echo "❌ エラー: 以下の必須ファイルが見つかりません:"
    for file in "${MISSING_FILES[@]}"; do
        echo "  - $file"
    done
    echo ""
    echo "💡 品質保証ワークフローを完全実行してください:"
    echo "   ./tools/run_quality_workflow.sh $TRACKER_ID"
    exit 1
fi

echo "✅ タスク完了確認済み"

# 2. 品質基準確認
echo "🔍 品質基準確認中..."

# 統合品質レポートから品質スコア取得
QUALITY_REPORT="${TASK_DIR}/quality/unified_quality_report.json"
if [ -f "$QUALITY_REPORT" ]; then
    OVERALL_SCORE=$(python3 -c "
import json
with open('$QUALITY_REPORT', 'r') as f:
    data = json.load(f)
print(data.get('overall_score', 0))
")
    
    STATUS=$(python3 -c "
import json
with open('$QUALITY_REPORT', 'r') as f:
    data = json.load(f)
print(data.get('status', 'UNKNOWN'))
")
    
    echo "📊 品質スコア: $OVERALL_SCORE"
    echo "🏆 ステータス: $STATUS"
    
    # 品質基準チェック (50%以上でリリース可能)
    MEETS_CRITERIA=$(python3 -c "print('true' if float('$OVERALL_SCORE') >= 0.5 else 'false')")
    
    if [ "$MEETS_CRITERIA" = "false" ]; then
        echo "❌ 警告: 品質スコアが基準値(50%)を下回っています"
        read -p "リリースを継続しますか? (y/N): " continue_release
        if [[ ! $continue_release =~ ^[Yy]$ ]]; then
            echo "🔄 リリースを中止しました"
            exit 1
        fi
    fi
fi

echo "✅ 品質基準確認済み"

# 3. バージョン更新
echo "📈 バージョン更新中..."

SETUP_PY="/mnt/c/AItools/segment-anything/setup.py"
if [ -f "$SETUP_PY" ]; then
    # 現在のバージョン取得
    CURRENT_VERSION=$(python3 -c "
import re
with open('$SETUP_PY', 'r') as f:
    content = f.read()
match = re.search(r'version=\"([^\"]+)\"', content)
print(match.group(1) if match else '0.1.0')
")
    
    echo "🔢 現在のバージョン: $CURRENT_VERSION"
    
    # 新しいバージョン計算
    NEW_VERSION=$(python3 -c "
import re
version = '$CURRENT_VERSION'
parts = list(map(int, version.split('.')))

if '$VERSION_TYPE' == '--major':
    parts[0] += 1
    parts[1] = 0
    parts[2] = 0
elif '$VERSION_TYPE' == '--minor':
    parts[1] += 1
    parts[2] = 0
else:  # --patch
    parts[2] += 1

print('.'.join(map(str, parts)))
")
    
    echo "🔢 新しいバージョン: $NEW_VERSION"
    
    # setup.py更新
    python3 -c "
import re
with open('$SETUP_PY', 'r') as f:
    content = f.read()

content = re.sub(r'version=\"[^\"]+\"', f'version=\"$NEW_VERSION\"', content)

with open('$SETUP_PY', 'w') as f:
    f.write(content)
"
    
    echo "✅ setup.py更新完了"
fi

# 4. PROGRESS_TRACKER.md更新
echo "📝 PROGRESS_TRACKER.md更新中..."

if [ -f "$TRACKER_FILE" ]; then
    # 完了日時と詳細情報の更新
    python3 -c "
import re
import json
from datetime import datetime

# 品質レポートから詳細情報取得
quality_details = {}
if '$QUALITY_REPORT':
    try:
        with open('$QUALITY_REPORT', 'r') as f:
            quality_data = json.load(f)
        quality_details = {
            'overall_score': quality_data.get('overall_score', 0),
            'status': quality_data.get('status', 'UNKNOWN'),
            'passed_metrics': quality_data.get('passed_metrics', 0),
            'total_metrics': quality_data.get('total_metrics', 0)
        }
    except:
        pass

# PROGRESS_TRACKER.md読み込み
with open('$TRACKER_FILE', 'r', encoding='utf-8') as f:
    content = f.read()

# タスクの完了ステータス更新
pattern = r'($TRACKER_ID:.*?status:\s*)🔄 PLANNED|⏳ IN_PROGRESS|❌ FAILED'
replacement = r'\1✅ COMPLETED'
content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)

# 完了日時追加
pattern = r'($TRACKER_ID:.*?)(\\n\\s*scope:)'
completion_info = f'''
  completion_date: $TIMESTAMP
  version: $NEW_VERSION
  quality_score: {quality_details.get('overall_score', 'N/A')}
  status: {quality_details.get('status', 'N/A')}
  passed_metrics: {quality_details.get('passed_metrics', 0)}/{quality_details.get('total_metrics', 0)}'''
replacement = r'\1' + completion_info + r'\2'
content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)

with open('$TRACKER_FILE', 'w', encoding='utf-8') as f:
    f.write(content)
"
    
    echo "✅ PROGRESS_TRACKER.md更新完了"
fi

# 5. Git コミット（オプション）
echo "📦 Git コミット準備..."

if git rev-parse --git-dir > /dev/null 2>&1; then
    echo "🔍 変更ファイル確認:"
    git status --porcelain
    
    read -p "Git コミットを作成しますか? (y/N): " create_commit
    if [[ $create_commit =~ ^[Yy]$ ]]; then
        # 変更をステージング
        git add "$SETUP_PY"
        git add "$TRACKER_FILE"
        git add "$TASK_DIR"
        
        # コミットメッセージ作成
        COMMIT_MSG="$(cat <<EOF
Release $TRACKER_ID: Version $NEW_VERSION

🎯 Task: $TRACKER_ID
📈 Version: $CURRENT_VERSION → $NEW_VERSION  
📊 Quality Score: $OVERALL_SCORE
🏆 Status: $STATUS
📅 Released: $TIMESTAMP

✅ Completed deliverables:
- Extraction pipeline execution
- Quality assessment (3 programs)
- Dashboard generation  
- Objective metrics testing
- Improvement measurement

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
        
        git commit -m "$COMMIT_MSG"
        
        # タグ作成
        git tag -a "v$NEW_VERSION" -m "Release v$NEW_VERSION: $TRACKER_ID"
        
        echo "✅ Git コミット完了: v$NEW_VERSION"
        echo "🏷️  タグ作成完了: v$NEW_VERSION"
    fi
fi

# 6. リリースサマリー生成
echo "📋 リリースサマリー生成中..."

RELEASE_SUMMARY="${TASK_DIR}/RELEASE_SUMMARY.md"

cat > "$RELEASE_SUMMARY" << EOF
# Release Summary: $TRACKER_ID

## 📋 Release Information
- **Tracker ID**: $TRACKER_ID
- **Version**: $CURRENT_VERSION → **$NEW_VERSION**
- **Release Date**: $TIMESTAMP
- **Release Type**: $VERSION_TYPE

## 📊 Quality Metrics
- **Overall Score**: $OVERALL_SCORE
- **Status**: $STATUS
- **Quality Standard**: $([ "$MEETS_CRITERIA" = "true" ] && echo "✅ PASSED" || echo "⚠️ MANUAL APPROVAL")

## 📁 Deliverables
- ✅ Extraction Pipeline Results: \`${TASK_DIR}/extraction/\`
- ✅ Quality Assessment Report: \`${TASK_DIR}/quality/unified_quality_report.json\`
- ✅ Dashboard: \`${TASK_DIR}/dashboard/dashboard.html\`
- ✅ Test Results: \`${TASK_DIR}/tests/objective_metrics_test.json\`
$([ -f "${TASK_DIR}/improvement_report.json" ] && echo "- ✅ Improvement Analysis: \`${TASK_DIR}/improvement_report.json\`" || echo "- ⏭️ Improvement Analysis: Skipped (no baseline)")

## 🔗 Quick Access
- [Dashboard](file://${TASK_DIR}/dashboard/dashboard.html)
- [Quality Report](file://${TASK_DIR}/quality/unified_quality_report.json)
- [Test Results](file://${TASK_DIR}/tests/objective_metrics_test.json)

## 📈 Next Steps
1. Review dashboard for quality confirmation
2. Validate test results meet project standards
3. Update documentation if needed
4. Prepare for next phase development

---
*Generated by Claude Code Release System v$NEW_VERSION*
EOF

echo ""
echo "🎉 リリース完了: $TRACKER_ID → v$NEW_VERSION"
echo "📋 リリースサマリー: $RELEASE_SUMMARY"
echo ""
echo "🔗 Quick Links:"
echo "   📊 Dashboard: file://${TASK_DIR}/dashboard/dashboard.html"
echo "   📝 Summary: file://${RELEASE_SUMMARY}"
echo ""

# Windows通知
if command -v windows-notify >/dev/null 2>&1; then
    windows-notify -t "Claude Code Release" -m "✅ $TRACKER_ID released as v$NEW_VERSION"
fi

echo "✅ /release プロセス完了"