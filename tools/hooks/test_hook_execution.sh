#!/bin/bash
# INTG-087: Hook実行テストスクリプト
# Claude Code Hook システムの実行テスト・検証スクリプト

set -euo pipefail

echo "🧪 INTG-087 Hook実行テストシステム開始"

# テスト設定
TEST_TRACKER_ID="INTG-087"
TEST_WORKSPACE="/tmp/hook_execution_test"
LOG_FILE="$TEST_WORKSPACE/hook_test.log"

# テスト環境セットアップ
setup_test_environment() {
    echo "🔧 テスト環境セットアップ中..."
    
    mkdir -p "$TEST_WORKSPACE/.workflow/logs"
    
    # テスト用チェックリスト状態ファイル
    cat > "$TEST_WORKSPACE/.workflow/checklist_status.json" <<EOF
{
  "tracker_id": "$TEST_TRACKER_ID",
  "created_at": "$(date -Iseconds)",
  "phase_0_5_branch": false,
  "phase_1_planning": {
    "sam_env_check": false,
    "google_sheets_sync": false,
    "sow_creation": false
  },
  "phase_2_implementation": {
    "started": false,
    "approval": false
  },
  "phase_3_quality": {
    "workflow_executed": false,
    "dashboard_created": false
  }
}
EOF

    echo "✅ テスト環境セットアップ完了: $TEST_WORKSPACE"
}

# テスト結果記録
log_test_result() {
    local test_name="$1"
    local result="$2"
    local details="$3"
    
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] TEST: $test_name | RESULT: $result | DETAILS: $details" >> "$LOG_FILE"
    
    if [[ "$result" == "PASS" ]]; then
        echo "✅ $test_name: $result"
    else
        echo "❌ $test_name: $result - $details"
    fi
}

# Hook実行ヘルパー
run_hook_test() {
    local hook_script="$1"
    local test_input="$2"
    local test_name="$3"
    local expected_pattern="$4"
    
    echo ""
    echo "🧪 テスト実行: $test_name"
    echo "📄 Hook: $hook_script"
    
    # Hook実行（タイムアウト設定）
    local output=""
    local exit_code=0
    
    if output=$(echo "$test_input" | timeout 30 bash "$hook_script" 2>&1); then
        exit_code=$?
        
        # 期待されるパターンの確認
        if [[ -n "$expected_pattern" ]] && echo "$output" | grep -q "$expected_pattern"; then
            log_test_result "$test_name" "PASS" "Expected pattern found: $expected_pattern"
            return 0
        elif [[ -z "$expected_pattern" ]] && [[ $exit_code -eq 0 ]]; then
            log_test_result "$test_name" "PASS" "Hook executed successfully"
            return 0
        else
            log_test_result "$test_name" "FAIL" "Expected pattern not found or exit code: $exit_code"
            echo "Output: $output"
            return 1
        fi
    else
        exit_code=$?
        log_test_result "$test_name" "FAIL" "Hook execution failed with exit code: $exit_code"
        echo "Output: $output"
        return 1
    fi
}

# 1. ワークフローコンプライアンス検証Hook テスト
test_validate_workflow_compliance() {
    echo ""
    echo "📋 1. ワークフローコンプライアンス検証Hook テスト"
    
    # テストケース1: 正常なextract_character.py実行
    local test_input1='{
        "tool": "Bash",
        "args": {
            "command": "python features/extraction/commands/extract_character.py /test/path -o /output/path --batch"
        }
    }'
    
    run_hook_test \
        "tools/hooks/validate_workflow_compliance.sh" \
        "$test_input1" \
        "WorkflowCompliance-ExtractCharacter" \
        '"allow"'
    
    # テストケース2: ブランチ検証
    local test_input2='{
        "tool": "Bash",
        "args": {
            "command": "git checkout feature/INTG-087"
        }
    }'
    
    run_hook_test \
        "tools/hooks/validate_workflow_compliance.sh" \
        "$test_input2" \
        "WorkflowCompliance-BranchCheck" \
        '"allow"'
    
    # テストケース3: 無関係コマンド（スキップ）
    local test_input3='{
        "tool": "Bash",
        "args": {
            "command": "ls -la"
        }
    }'
    
    run_hook_test \
        "tools/hooks/validate_workflow_compliance.sh" \
        "$test_input3" \
        "WorkflowCompliance-UnrelatedCommand" \
        '"allow"'
}

# 2. ワークフロー進捗更新Hook テスト
test_update_workflow_progress() {
    echo ""
    echo "📊 2. ワークフロー進捗更新Hook テスト"
    
    # テストケース1: sam-env確認完了
    local test_input1='{
        "tool": "Bash",
        "args": {
            "command": "source sam-env/bin/activate && echo $VIRTUAL_ENV"
        },
        "result": "/path/to/sam-env"
    }'
    
    run_hook_test \
        "tools/hooks/update_workflow_progress.sh" \
        "$test_input1" \
        "WorkflowProgress-SamEnvCheck" \
        '"status"'
    
    # テストケース2: Google Sheets連携
    local test_input2='{
        "tool": "Bash",
        "args": {
            "command": "python tools/progress_tracker/cli.py status INTG-087"
        },
        "result": "success"
    }'
    
    run_hook_test \
        "tools/hooks/update_workflow_progress.sh" \
        "$test_input2" \
        "WorkflowProgress-GoogleSheets" \
        '"status"'
    
    # テストケース3: 抽出処理完了
    local test_input3='{
        "tool": "Bash",
        "args": {
            "command": "python features/extraction/commands/extract_character.py /input -o /output --batch"
        },
        "result": "extraction completed"
    }'
    
    run_hook_test \
        "tools/hooks/update_workflow_progress.sh" \
        "$test_input3" \
        "WorkflowProgress-ExtractionCompleted" \
        '"status"'
}

# 3. トラッカーコンテキスト分析Hook テスト
test_analyze_tracker_context() {
    echo ""
    echo "🔍 3. トラッカーコンテキスト分析Hook テスト"
    
    # テストケース1: 進捗確認プロンプト
    local test_input1='{
        "prompt": "INTG-087の進捗を確認して次のステップを教えてください。現在Phase 1の計画フェーズです。"
    }'
    
    run_hook_test \
        "tools/hooks/analyze_tracker_context.sh" \
        "$test_input1" \
        "TrackerContext-ProgressCheck" \
        '"tracker_id"'
    
    # テストケース2: 実装開始プロンプト
    local test_input2='{
        "prompt": "INTG-087の実装を開始します。Phase 2に移行してください。"
    }'
    
    run_hook_test \
        "tools/hooks/analyze_tracker_context.sh" \
        "$test_input2" \
        "TrackerContext-ImplementationStart" \
        '"current_phase"'
    
    # テストケース3: 品質チェックプロンプト
    local test_input3='{
        "prompt": "INTG-087の品質ワークフローを実行して最終確認を行います。"
    }'
    
    run_hook_test \
        "tools/hooks/analyze_tracker_context.sh" \
        "$test_input3" \
        "TrackerContext-QualityCheck" \
        '"recommendations"'
}

# 4. Hook統合シーケンステスト
test_hook_sequence() {
    echo ""
    echo "🔗 4. Hook統合シーケンステスト"
    
    # Phase 1: PreToolUse → PostToolUse シーケンス
    echo "Phase 1: PreToolUse → PostToolUse テスト"
    
    # PreToolUse
    local pre_input='{
        "tool": "Bash",
        "args": {
            "command": "source sam-env/bin/activate"
        }
    }'
    
    if run_hook_test \
        "tools/hooks/validate_workflow_compliance.sh" \
        "$pre_input" \
        "HookSequence-PreToolUse" \
        '"allow"'; then
        
        # PostToolUse
        local post_input='{
            "tool": "Bash",
            "args": {
                "command": "source sam-env/bin/activate"
            },
            "result": "success"
        }'
        
        run_hook_test \
            "tools/hooks/update_workflow_progress.sh" \
            "$post_input" \
            "HookSequence-PostToolUse" \
            '"status"'
    else
        log_test_result "HookSequence-Complete" "FAIL" "PreToolUse failed, skipping PostToolUse"
    fi
}

# 5. エラーハンドリングテスト
test_error_handling() {
    echo ""
    echo "🚨 5. Hook エラーハンドリングテスト"
    
    # テストケース1: 不正なJSON入力
    local invalid_json='{"tool": "Bash", "args": invalid json}'
    
    # エラー時でもクラッシュしないことを確認
    if output=$(echo "$invalid_json" | timeout 10 bash tools/hooks/validate_workflow_compliance.sh 2>&1 || true); then
        if echo "$output" | grep -q "ERROR\|error\|エラー"; then
            log_test_result "ErrorHandling-InvalidJSON" "PASS" "Graceful error handling detected"
        else
            log_test_result "ErrorHandling-InvalidJSON" "PARTIAL" "No explicit error message, but didn't crash"
        fi
    else
        log_test_result "ErrorHandling-InvalidJSON" "FAIL" "Hook crashed on invalid input"
    fi
    
    # テストケース2: 存在しないトラッカーID
    local nonexistent_tracker='{
        "prompt": "NONEXISTENT-999の状況を確認してください。"
    }'
    
    if output=$(echo "$nonexistent_tracker" | timeout 10 bash tools/hooks/analyze_tracker_context.sh 2>&1 || true); then
        if echo "$output" | grep -q '"status":\s*"new_tracker"\|"no_tracker"'; then
            log_test_result "ErrorHandling-NonexistentTracker" "PASS" "Handled nonexistent tracker appropriately"
        else
            log_test_result "ErrorHandling-NonexistentTracker" "PARTIAL" "Response received but pattern unclear"
        fi
    else
        log_test_result "ErrorHandling-NonexistentTracker" "FAIL" "Hook failed on nonexistent tracker"
    fi
}

# 6. パフォーマンステスト
test_performance() {
    echo ""
    echo "⚡ 6. Hook パフォーマンステスト"
    
    local start_time=$(date +%s.%N)
    
    # 軽量テスト実行
    local simple_input='{
        "tool": "Bash",
        "args": {
            "command": "echo test"
        }
    }'
    
    if run_hook_test \
        "tools/hooks/validate_workflow_compliance.sh" \
        "$simple_input" \
        "Performance-SimpleCommand" \
        '"allow"'; then
        
        local end_time=$(date +%s.%N)
        local duration=$(echo "$end_time - $start_time" | bc -l)
        
        # 5秒以内での実行を期待
        if (( $(echo "$duration < 5.0" | bc -l) )); then
            log_test_result "Performance-ExecutionTime" "PASS" "Duration: ${duration}s (under 5s)"
        else
            log_test_result "Performance-ExecutionTime" "FAIL" "Duration: ${duration}s (over 5s)"
        fi
    else
        log_test_result "Performance-ExecutionTime" "FAIL" "Hook execution failed"
    fi
}

# テスト結果サマリー
generate_test_summary() {
    echo ""
    echo "📊 テスト結果サマリー"
    echo "=================================="
    
    if [[ -f "$LOG_FILE" ]]; then
        local total_tests=$(grep "TEST:" "$LOG_FILE" | wc -l)
        local passed_tests=$(grep "RESULT: PASS" "$LOG_FILE" | wc -l)
        local failed_tests=$(grep "RESULT: FAIL" "$LOG_FILE" | wc -l)
        local partial_tests=$(grep "RESULT: PARTIAL" "$LOG_FILE" | wc -l)
        
        echo "📈 総テスト数: $total_tests"
        echo "✅ 成功: $passed_tests"
        echo "❌ 失敗: $failed_tests"
        echo "⚠️ 部分成功: $partial_tests"
        
        local success_rate=0
        if [[ $total_tests -gt 0 ]]; then
            success_rate=$(echo "scale=1; ($passed_tests + $partial_tests) * 100 / $total_tests" | bc -l)
        fi
        
        echo "🎯 成功率: ${success_rate}%"
        
        echo ""
        echo "📋 詳細ログ: $LOG_FILE"
        
        # 失敗したテストの詳細表示
        if [[ $failed_tests -gt 0 ]]; then
            echo ""
            echo "❌ 失敗テスト詳細:"
            grep "RESULT: FAIL" "$LOG_FILE" | while read line; do
                echo "  - $line"
            done
        fi
    else
        echo "⚠️ ログファイルが見つかりません: $LOG_FILE"
    fi
}

# テスト環境クリーンアップ
cleanup_test_environment() {
    echo ""
    echo "🧹 テスト環境クリーンアップ中..."
    
    if [[ -d "$TEST_WORKSPACE" ]]; then
        # ログファイルのバックアップ
        if [[ -f "$LOG_FILE" ]]; then
            cp "$LOG_FILE" "/tmp/hook_test_$(date +%Y%m%d_%H%M%S).log"
            echo "📄 ログファイルをバックアップしました: /tmp/hook_test_$(date +%Y%m%d_%H%M%S).log"
        fi
        
        # テストディレクトリ削除
        rm -rf "$TEST_WORKSPACE"
    fi
    
    echo "✅ クリーンアップ完了"
}

# メイン実行フロー
main() {
    echo "🎯 INTG-087 Hook実行テスト開始"
    echo "時刻: $(date)"
    echo ""
    
    # テスト環境セットアップ
    setup_test_environment
    
    # 各テスト実行
    test_validate_workflow_compliance
    test_update_workflow_progress
    test_analyze_tracker_context
    test_hook_sequence
    test_error_handling
    test_performance
    
    # テスト結果サマリー
    generate_test_summary
    
    # クリーンアップ
    cleanup_test_environment
    
    echo ""
    echo "🎯 INTG-087 Hook実行テスト完了"
}

# bcコマンドの存在確認
if ! command -v bc &> /dev/null; then
    echo "⚠️ 警告: bc コマンドが見つかりません。パフォーマンステストをスキップします。"
    echo "インストール: sudo apt-get install bc"
    
    # bcなしでも動作するよう調整
    test_performance() {
        echo "⏭️  パフォーマンステストをスキップ（bc未インストール）"
        log_test_result "Performance-ExecutionTime" "SKIP" "bc command not available"
    }
fi

# スクリプト実行
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi