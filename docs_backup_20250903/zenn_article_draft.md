# 大規模リポジトリでClaude Code MCP Serverが10分応答しない問題を解決した話

## はじめに

Claude CodeでSerena MCP Serverを使用していたところ、突然10分以上も応答が返ってこない問題に遭遇しました。

```
serena - find_file (MCP)(file_mask: "*command*", relative_path: "features/extraction")
* Spinning… (635s · ⚒ 1.2k tokens · esc to interrupt)
```

**635秒（10分以上）**のスピンは明らかに異常です。この記事では、52,411ファイルの大規模リポジトリで発生した性能問題の原因分析と解決策を共有します。

## 問題の概要

### 発生環境
- **OS**: Windows + WSL (Ubuntu)
- **MCP Server**: Serena (v0.1.3)
- **プロジェクト規模**: 52,411ファイル（Pythonファイル12,951個）
- **Language Server**: Node.js版Pyright

### 症状
- `find_file`、`search_for_pattern`等のツールで10分以上の応答遅延
- `get_symbols_overview`で99秒の処理時間
- Language Serverの高メモリ使用量（686MB）

## 原因調査の結果

Claude Codeに調査を依頼したところ、以下の根本原因が特定されました：

### 1. プロジェクト規模の問題
```bash
# 実際のファイル数確認
$ find /path/to/your-project -type f | wc -l
52411

$ find /path/to/your-project -type f -name "*.py" | wc -l
12951
```

**52,411ファイル**という規模は、MCP Serverの想定を大幅に超えていました。

### 2. Language Server (Pyright) の負荷
```bash
# プロセス確認
$ ps aux | grep pyright
user    141926  0.2  3.3 1602064 686924 ?  Sl  20:33  0:23 node ...pyright/dist/langserver.index.js
```

Node.js版のPyrightが**686MB**のメモリを使用し、初期インデックス作成に大幅な時間を要していました。

### 3. 不要ファイルの処理
- 仮想環境（5.2GB）
- ログファイル・キャッシュ
- 画像ファイル・バイナリファイル
- 廃止予定のコード

これらが全てLanguage Serverの処理対象となっており、性能を大幅に悪化させていました。

## 解決策の実装

### 1. プロジェクト除外パターンの追加

`.serena/project.yml`に以下の除外パターンを追加しました：

```yaml
ignored_paths: 
  # 仮想環境・依存関係
  - "venv/**"
  - "node_modules/**"
  - "**/__pycache__/**"
  - "**/*.pyc"
  - "**/*.pyo"
  - "**/*.egg-info/**"
  
  # キャッシュ・ログ
  - ".mypy_cache/**"
  - ".pytest_cache/**"
  - "logs/**"
  - "*.log"
  - "*.pid"
  
  # バイナリファイル
  - "*.dll"
  - "*.so"
  - "*.pth"
  
  # メディアファイル
  - "*.png"
  - "*.jpg"
  - "*.jpeg"
  - "*.gif"
  - "*.mp4"
  
  # 開発関連
  - "temp/**"
  - "deprecated/**"
  - "backup-*/**"
  - "notebooks/**"
```

### 2. タイムアウト設定の調整

`~/.serena/serena_config.yml`でタイムアウトを短縮：

```yaml
tool_timeout: 60  # 240秒から60秒に短縮
```

これにより、問題のあるクエリで長時間待機することを防げます。

## 運用改善：ツールの使い分け戦略

大規模プロジェクトでは、**Serena MCP ServerとClaude Code標準ツールの使い分け**が重要です。

### Serena MCP Server（高機能・低速）
**適用場面**：
- 複雑なシンボル検索・リファクタリング
- プロジェクト全体の構造把握

**推奨使用例**：
```bash
# 具体的なパス指定でのシンボル検索
mcp__serena__find_symbol: name_path="ExtractionConfig", relative_path="src/commands"
```

### Claude Code標準ツール（軽量・高速）
**適用場面**：
- ファイル読み書き
- 簡単な検索・置換
- コマンド実行

**推奨使用例**：
```bash
# ファイル操作
Read: file_path="/path/to/file.py"
Edit: file_path="/path/to/file.py", old_string="...", new_string="..."

# パターン検索
Grep: pattern="extract_character", glob="**/*.py"
```

### ハイブリッド戦略
```
1. Claude標準ツールで高速調査
   ↓
2. Serenaで詳細分析（必要に応じて）
   ↓
3. Claude標準ツールで実装
```

## プロジェクト規模別の対応指針

経験から導かれる、プロジェクト規模別の対応基準：

| 規模 | ファイル数 | 状況 | 推奨対応 |
|------|------------|------|----------|
| 小規模 | 〜1,000 | 問題なし | デフォルト設定 |
| 中規模 | 1,000〜10,000 | 軽微な遅延 | 基本的な除外パターン |
| 大規模 | 10,000〜50,000 | 最適化推奨 | 本記事の対策必須 |
| 超大規模 | 50,000以上 | 深刻な問題 | 段階的最適化＋使い分け |

## 他のMCP Serverでも注意が必要

この問題は**Serena固有ではありません**。Language Serverベースの全てのMCP Serverで類似の問題が発生する可能性があります：

- ファイル数に比例した処理時間の増加
- メモリ使用量の急激な増大
- 初期インデックス処理の長時間化

大規模プロジェクトでMCP Serverを使用する際は、事前に以下を確認することをお勧めします：

```bash
# プロジェクト規模の事前チェック
find . -type f | wc -l
find . -name "*.py" | wc -l  # 言語固有ファイル
du -sh .  # 総サイズ
```

## 効果と今後の運用

最適化により、以下の改善が期待されます：
- **応答時間**: 10分→数秒レベル
- **メモリ使用量**: 大幅削減
- **安定性**: タイムアウト回避

継続的な監視には以下のコマンドが有効です：

```bash
# プロセス・メモリ監視
ps aux | grep -E "(pyright|serena)" | grep -v grep
free -h

# MCP Server接続状況確認
# Claude Code設定画面で確認
```

## まとめ

大規模リポジトリでのMCP Server性能問題は、**適切な除外設定と使い分け戦略**で解決できます。

特に重要なポイント：
1. **プロジェクト規模の把握**：1万ファイル以上なら要注意
2. **除外パターンの設定**：仮想環境・ログ・バイナリファイル等
3. **ツールの使い分け**：高速処理はClaude標準ツール、詳細分析はMCP Server
4. **継続的監視**：プロセス状況とメモリ使用量の定期確認

この経験が、同様の問題に直面している方の参考になれば幸いです。大規模プロジェクトでのClaude Code活用時は、ぜひこれらの対策をご検討ください。