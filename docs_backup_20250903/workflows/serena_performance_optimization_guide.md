# Serena MCP Server パフォーマンス最適化・運用ガイド

**最終更新**: 2025-08-02  
**目的**: SerenaとClaude Code標準ツールの効果的使い分け

---

## 🎯 概要

Serena MCP Serverはsegment-anythingプロジェクト（52,411ファイル）で性能問題が発生します。本ガイドでは最適化設定と効果的な使い分け方法を提供します。

---

## ⚠️ 確認された問題

### 📊 プロジェクト規模による問題
- **総ファイル数**: 52,411ファイル（Serenaの想定を大幅超過）
- **Pythonファイル**: 12,951ファイル
- **初期インデックス時間**: 99秒（get_symbols_overview）
- **応答時間**: 10分以上のスピン（find_file等）

### 🔧 根本原因
1. **Language Server負荷**: Node.js版Pyright（686MB RAM使用）
2. **シンボル解析**: 102,973文字の出力（制限50,000文字超過）
3. **ファイルスキャン**: 仮想環境・ログ・画像ファイル等の不要な処理

---

## ✅ 実施済み最適化

### 1. プロジェクト除外パターン強化

```yaml
# /mnt/c/AItools/segment-anything/.serena/project.yml
ignored_paths: 
  - "sam-env/**"           # 5.2GB仮想環境
  - "gemini_outputs/**"    # Gemini出力
  - "deprecated/**"        # 非推奨コード（6.7MB）
  - "temp/**"              # 一時ファイル
  - "logs/**"              # ログファイル
  - "*.log"
  - "*.pid"
  - ".serena/cache/**"     # Serenaキャッシュ
  - "backup-*/**"          # バックアップ
  - "**/__pycache__/**"    # Pythonキャッシュ
  - "**/*.pyc"
  - "**/*.pyo"
  - "**/*.egg-info/**"
  - ".mypy_cache/**"
  - ".pytest_cache/**"
  - "*.pth"
  - "*.dll"                # バイナリファイル
  - "*.so"
  - "notebooks/**"         # Jupyter notebooks
  - "demo_output/**"       # デモ出力
  - "task_dispatch_results/**"
  - "validation_reports/**"
  - "*.png"                # 画像ファイル
  - "*.jpg"
  - "*.jpeg"
  - "*.gif"
  - "*.mp4"
  - "*.avi"
```

### 2. タイムアウト短縮

```yaml
# ~/.serena/serena_config.yml
tool_timeout: 60  # 240秒から60秒に短縮
```

---

## 🚀 効果的な使い分け戦略

### Serena MCP Server（高機能・低速）
**適用場面**:
- 複雑なシンボル検索
- リファクタリング（symbol-based editing）
- プロジェクト全体の概要把握

**推奨コマンド**:
```bash
# シンボル検索（具体的なパス指定）
mcp__serena__find_symbol: name_path="ExtractionConfig", relative_path="features/extraction"

# メモリ管理
mcp__serena__write_memory: memory_name="module_structure"
mcp__serena__read_memory: memory_file_name="project_overview.md"
```

### Claude Code標準ツール（軽量・高速）
**適用場面**:
- ファイル読み書き
- 簡単な検索・置換
- バッチ処理・コマンド実行

**推奨コマンド**:
```bash
# ファイル操作
Read: file_path="/path/to/file.py"
Edit: file_path="/path/to/file.py", old_string="...", new_string="..."

# 検索
Grep: pattern="extract_character", glob="**/*.py"

# 実行
Bash: command="python test.py"
```

---

## 📋 運用ガイドライン

### ✅ Serena使用推奨ケース
1. **新機能実装時**: プロジェクト構造の把握
2. **リファクタリング**: クラス・メソッドの体系的変更
3. **デバッグ**: 複雑な依存関係の調査

### ❌ Serena使用回避ケース  
1. **単純なファイル読み込み**: `mcp__serena__read_file` → `Read`
2. **文字列置換**: `mcp__serena__replace_regex` → `Edit`
3. **ファイル検索**: `mcp__serena__find_file` → `Glob`
4. **コマンド実行**: `mcp__serena__execute_shell_command` → `Bash`

### 🔄 ハイブリッド使用例

```bash
# ステップ1: Claude標準ツールで高速調査
Grep: pattern="Command.*Pattern", glob="features/**/*.py"

# ステップ2: Serenaで詳細分析
mcp__serena__find_symbol: name_path="BaseExtractionCommand", include_body=true

# ステップ3: Claude標準ツールで実装
Edit: file_path="features/extraction/commands/new_processor.py"
```

---

## 🔧 トラブルシューティング

### 問題: 10分以上のスピン
**解決方法**:
1. **即座に中断**: `Ctrl+C` または `esc`
2. **代替手段**: Claude標準ツールに切り替え
3. **設定確認**: 除外パターンの効果確認

### 問題: "Answer too long" エラー
**解決方法**:
```bash
# より具体的なパス指定
mcp__serena__get_symbols_overview: relative_path="features/extraction/commands"

# max_answer_chars調整
mcp__serena__find_symbol: name_path="ExtractionConfig", max_answer_chars=10000
```

### 問題: Language Server異常
**解決方法**:
```bash
# Language Server再起動
mcp__serena__restart_language_server

# Serenaプロセス確認
ps aux | grep serena

# 完全再起動（最終手段）
pkill -f serena-mcp-server
```

---

## 📊 性能監視

### 定期確認コマンド
```bash
# プロセス監視
ps aux | grep -E "(pyright|serena)" | grep -v grep

# メモリ使用量
free -h

# ファイル数確認（効果測定）
find /mnt/c/AItools/segment-anything -type f | wc -l
```

### 期待される改善効果
- **応答時間**: 10分→数秒レベル
- **メモリ使用量**: 686MB→400MB以下
- **安定性**: タイムアウト回避

---

## 📝 ベストプラクティス

### 1. 段階的アプローチ
```
簡易調査（Claude標準） → 詳細分析（Serena） → 実装（Claude標準）
```

### 2. タスク別選択基準
| タスク | 推奨ツール | 理由 |
|--------|------------|------|
| ファイル読み込み | Read | 高速・安定 |
| 文字列検索 | Grep | パターンマッチ最適 |
| シンボル検索 | Serena | 意味的理解 |
| コード実行 | Bash | 実行環境最適 |
| リファクタリング | Serena | 構造理解必要 |

### 3. エラー対処フロー
```
Serenaでエラー → Claude標準で代替 → 必要に応じてSerena詳細分析
```

---

## 🎯 今後の改善計画

### 短期対策（実施済み）
- ✅ 除外パターン強化
- ✅ タイムアウト短縮  
- ✅ 使い分けガイドライン

### 中期対策（検討中）
- プロジェクト分割検討
- 選択的インデックス設定
- キャッシュ最適化

### 長期対策（将来）
- Serena設定のプロファイル化
- 自動パフォーマンス監視
- 適応的ツール選択

---

**重要**: このガイドラインに従い、適切なツール選択により開発効率を最大化してください。