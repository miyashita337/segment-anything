# SubAgent統合システム使用ガイド

**INTG-087: SubAgent長時間タスクキューシステム完全版**

## 🎯 概要

SubAgent統合システムは、Claude Codeにおける長時間実行タスクの効率的な管理・制御を実現するシステムです。特に `extract_character.py` などの重いML処理の直接実行を防止し、キューベースでの実行を強制します。

## 🏗️ システム構成

```
SubAgent統合システム
├── tools/queue/subagent_wrapper.py          # メインキューシステム
├── features/extraction/commands/            # 統合対象スクリプト
│   └── extract_character.py                 # SubAgent必須化済み
├── tests/unit/                              # 統合テスト
│   ├── test_hook_integration.py             # Hook統合テスト
│   └── test_workflow_state.py               # ワークフロー状態テスト
└── .claude/hooks.json                       # Hook設定
```

## 🚀 基本使用方法

### 1. 環境設定

```bash
# 必要な環境変数設定
export TRACKER_ID="INTG-087"
export TRACKER_WORKSPACE="/mnt/c/AItools/lora/train/yado/tracker-workspace/INTG-087"
export PYTHONPATH="/mnt/c/AItools/segment-anything:$PYTHONPATH"
```

### 2. SubAgentキュー操作

#### タスクのエンキュー（キューに追加）
```bash
# 基本形式
python tools/queue/subagent_wrapper.py enqueue <task_name> "<command>"

# 例1: キャラクター抽出タスク
python tools/queue/subagent_wrapper.py enqueue extract_character \
  "python features/extraction/commands/extract_character.py /mnt/c/AItools/lora/train/yado/org/kana05/ -o /mnt/c/AItools/lora/train/yado/tracker-workspace/INTG-087/extraction/ --batch"

# 例2: 品質ワークフロータスク
python tools/queue/subagent_wrapper.py enqueue quality_workflow \
  "bash tools/scripts/run_quality_workflow.sh INTG-087"

# 例3: 進捗更新タスク
python tools/queue/subagent_wrapper.py enqueue progress_update \
  "python tools/progress_tracker/cli.py update INTG-087 '実装完了'"
```

#### タスクの実行
```bash
# キューから次のタスクを実行
python tools/queue/subagent_wrapper.py execute

# 実行例出力:
# ✅ タスク実行完了: completed
# 出力: 抽出処理が正常に完了しました...
```

#### キュー状況確認
```bash
# キュー状況表示
python tools/queue/subagent_wrapper.py status

# 出力例:
# {
#   "tracker_id": "INTG-087",
#   "queue_status": {
#     "total_tasks": 5,
#     "queued": 1,
#     "completed": 3,
#     "failed": 1,
#     "success_rate": 60.0
#   },
#   "performance": {
#     "average_execution_time": 245.8,
#     "total_execution_time": 1229.0
#   },
#   "next_task": "INTG-087_extract_character_1725934567"
# }
```

#### タスククリーンアップ
```bash
# 7日以上古い完了タスクを削除
python tools/queue/subagent_wrapper.py cleanup 7

# 出力例:
# ✅ クリーンアップ完了: 12件削除
```

### 3. 直接実行防止システム

#### 問題: extract_character.py の直接実行
```bash
# これは失敗します
python features/extraction/commands/extract_character.py /input/path -o /output/path

# エラー出力:
# ❌ extract_character.py の直接実行は禁止されています
# 
# 🚨 INTG-087 SubAgent統合システムにより、長時間タスクは必ず
# SubAgentキューシステム経由で実行する必要があります。
# 
# 🔧 正しい実行方法:
# 1. タスクをキューに追加:
#    python tools/queue/subagent_wrapper.py enqueue extract_character "python features/extraction/commands/extract_character.py [args...]"
# 
# 2. キューからタスクを実行:
#    python tools/queue/subagent_wrapper.py execute
```

#### 解決方法: SubAgent経由実行
```bash
# Step 1: タスクをエンキュー
python tools/queue/subagent_wrapper.py enqueue extract_character \
  "python features/extraction/commands/extract_character.py /input/path -o /output/path --batch"

# Step 2: タスク実行
python tools/queue/subagent_wrapper.py execute
```

#### 緊急時の直接実行（非推奨）
```bash
# 緊急時のみ使用 - SubAgent制限をバイパス
SUBAGENT_EXECUTION=true python features/extraction/commands/extract_character.py /input/path -o /output/path --batch
```

## 🎛️ 高度な使用方法

### 1. 優先度付きタスク管理

```python
# Python APIでの高度なタスク制御例
from tools.queue.subagent_wrapper import SubAgentTaskQueue
from pathlib import Path

# キューシステム初期化
workspace_path = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace/INTG-087")
queue = SubAgentTaskQueue(workspace_path, "INTG-087")

# 高優先度タスクの追加
queue.enqueue_task(
    task_id="urgent_extraction",
    command="python features/extraction/commands/extract_character.py /urgent/path -o /output/path --batch",
    priority=5,  # 最高優先度
    estimated_duration=600,  # 10分予想
    resource_requirements={"type": "gpu", "memory": "8GB"}
)

# 低優先度タスクの追加
queue.enqueue_task(
    task_id="background_processing",
    command="bash tools/scripts/cleanup_old_files.sh",
    priority=1,  # 最低優先度
    estimated_duration=30
)

# タスク実行
result = queue.execute_next_task()
print(f"実行結果: {result['status']}")
```

### 2. リソース監視と制御

SubAgentシステムは以下のリソース監視を行います：

```python
# リソース制限設定（subagent_wrapper.py内）
self.max_execution_time = 3600      # 1時間タイムアウト
self.max_memory_usage = 8 * GB      # 8GB メモリ制限  
self.task_timeout = 1800            # 30分タスクタイムアウト
```

**監視項目:**
- CPU使用率監視
- メモリ使用量監視  
- 実行時間監視
- CUDA利用可能性チェック
- 依存ファイル存在チェック

### 3. エラーハンドリングとレジューム

```bash
# タスク失敗時の状況確認
python tools/queue/subagent_wrapper.py status

# 失敗タスクの詳細確認（Python API使用）
python -c "
from tools.queue.subagent_wrapper import SubAgentTaskQueue
from pathlib import Path
queue = SubAgentTaskQueue(Path('/tmp'), 'INTG-087')
registry = queue._load_task_registry()
for task_id, task in registry.items():
    if task['status'] == 'failed':
        print(f'失敗タスク: {task_id}')
        print(f'エラー: {task[\"error\"]}')
"
```

## 🔧 統合テスト

### 1. Hook統合テスト実行

```bash
# Hook統合システムテスト
python -m pytest tests/unit/test_hook_integration.py -v

# 期待される出力:
# test_validate_workflow_compliance_hook PASSED
# test_update_workflow_progress_hook PASSED  
# test_analyze_tracker_context_hook PASSED
# test_hook_sequence_workflow PASSED
# test_workflow_state_persistence PASSED
# test_error_handling PASSED
```

### 2. SubAgentシステムテスト

```bash
# SubAgent機能テスト
python -c "
# SubAgentキューテスト
import tempfile
from pathlib import Path
from tools.queue.subagent_wrapper import SubAgentTaskQueue

# テスト用キュー作成
test_workspace = Path(tempfile.mkdtemp())
queue = SubAgentTaskQueue(test_workspace, 'TEST-001')

# テストタスク追加
success = queue.enqueue_task(
    task_id='test_task_001',
    command='echo \"SubAgent test successful\"',
    priority=3
)

print(f'エンキュー成功: {success}')

# キュー状況確認
status = queue.get_queue_status()
print(f'キュー状況: {status[\"queue_status\"]}')
"
```

### 3. 完全統合テスト

```bash
# Hook実行テストスイート
bash tools/hooks/test_hook_execution.sh

# 期待される結果:
# 📊 テスト結果サマリー
# ==================================
# 📈 総テスト数: 15
# ✅ 成功: 13
# ❌ 失敗: 1
# ⚠️ 部分成功: 1  
# 🎯 成功率: 93.3%
```

## 📊 パフォーマンス最適化

### 1. キュー効率化

```bash
# 完了タスクの定期クリーンアップ（推奨: 週1回）
python tools/queue/subagent_wrapper.py cleanup 7

# キュー統計分析
python tools/queue/subagent_wrapper.py status | jq '.performance'
```

### 2. リソース最適化

```python
# GPU使用率最適化のための設定例
resource_requirements = {
    "type": "gpu",
    "memory": "8GB",
    "cuda_version": "11.8",
    "batch_size": 4  # GPU性能に応じて調整
}

queue.enqueue_task(
    task_id="optimized_extraction",
    command="python features/extraction/commands/extract_character.py --batch --sam-optimization-profile p1_020_optimized",
    resource_requirements=resource_requirements
)
```

## 🛠️ トラブルシューティング

### 1. よくあるエラーと解決方法

#### エラー: `Permission denied`
```bash
# 解決: 実行権限付与
chmod +x tools/queue/subagent_wrapper.py
chmod +x tools/hooks/*.sh
```

#### エラー: `ModuleNotFoundError: No module named 'config.workspace_config'`
```bash
# 解決: Pythonパス設定
export PYTHONPATH="/mnt/c/AItools/segment-anything:$PYTHONPATH"
```

#### エラー: `psutil.NoSuchProcess`
```bash
# 解決: psutilインストール
pip install psutil
```

### 2. デバッグモード

```bash
# 詳細ログ出力でのSubAgent実行
SUBAGENT_EXECUTION=true LOG_LEVEL=DEBUG \
python features/extraction/commands/extract_character.py /input/path -o /output/path --batch --verbose
```

### 3. システム状態確認

```bash
# SubAgentシステム全体の状態確認スクリプト
cat > check_subagent_status.sh << 'EOF'
#!/bin/bash
echo "🔍 SubAgent統合システム状態確認"
echo "================================"

# Python環境確認
echo "📦 Python環境:"
python --version
which python

# 必要モジュール確認  
echo -e "\n📚 必要モジュール:"
python -c "import psutil; print('psutil: OK')" 2>/dev/null || echo "psutil: NG"
python -c "import torch; print('torch: OK')" 2>/dev/null || echo "torch: NG"

# ワークスペース確認
echo -e "\n📁 ワークスペース:"
echo "TRACKER_ID: ${TRACKER_ID:-未設定}"
echo "TRACKER_WORKSPACE: ${TRACKER_WORKSPACE:-未設定}"

# Hookシステム確認
echo -e "\n🔗 Hookシステム:"
if [[ -f ".claude/hooks.json" ]]; then
    echo "✅ .claude/hooks.json 存在"
    jq '.hooks | keys[]' .claude/hooks.json
else
    echo "❌ .claude/hooks.json 不在"
fi

# キューファイル確認
if [[ -d "${TRACKER_WORKSPACE}/.subagent_queue" ]]; then
    echo -e "\n🎯 SubAgentキュー:"
    echo "✅ キューディレクトリ存在"
    ls -la "${TRACKER_WORKSPACE}/.subagent_queue"
else
    echo -e "\n🎯 SubAgentキュー:"
    echo "⚠️ キューディレクトリ未作成"
fi
EOF

chmod +x check_subagent_status.sh
bash check_subagent_status.sh
```

## 📋 チェックリスト

### SubAgent統合システム導入前チェックリスト

- [ ] Python 3.8+ インストール済み
- [ ] psutil, torch, numpy インストール済み
- [ ] 適切な TRACKER_ID, TRACKER_WORKSPACE 環境変数設定
- [ ] .claude/hooks.json 設定済み
- [ ] tools/hooks/*.sh に実行権限付与
- [ ] workspace_config.py アクセス可能

### 運用開始前チェックリスト

- [ ] Hook統合テスト成功
- [ ] SubAgentキューシステム動作確認
- [ ] extract_character.py 直接実行防止動作確認
- [ ] キュー操作（enqueue/execute/status/cleanup）動作確認
- [ ] エラーハンドリング動作確認

---

**作成日**: 2025-09-09  
**バージョン**: INTG-087 SubAgent統合版  
**対象システム**: Claude Code Hook統合システム