# 刷新ワークフロー手順書 - INCI-004対応版

**作成日**: 2025-08-26  
**対応**: INCI-004承認回数増加・統計分析システム統合改良  
**承認回数**: 5回（安心感向上）  
**統計分析**: universal_statistical_analyzer.py（BASELINE_ID必須化）

---

## 📋 **基本設定**

```bash
{トラッカーID}=INCI-004  # 例: 実際のトラッカーIDに置き換え

# 前のトラッカー(ベーストラッカー)はGoogleSheetで一番更新日付が新しいトラッカー
# INCI-004により、BASELINE_IDの指定が必須になりました
```

---

## 🚀 **Phase 1: 起票・計画フェーズ**

### * [local/Claude] GoogleSheetトラッカー起票
もしGoogleSheetにトラッカーが起票されてないなら先に起票(すでに起票済みならスルー)
```bash
PROGRESS_TRACKER_SHEET_NAME="シート1" python3 tools/progress_tracker/cli.py create {トラッカーID} --description "概要" --details "詳細な実装計画..."
```

### * [local/Claude] GoogleStatusステータス更新
```bash
PROGRESS_TRACKER_SHEET_NAME="シート1" python3 tools/progress_tracker/cli.py update {トラッカーID} "着手中"
```

### * [local/Claude] featureブランチを切って別のブランチで開発
```bash
git checkout -b feature/{トラッカーID}
```

### * [local/Claude] 概要と詳細を改めて見なおす
- [local/Claude] 概要と詳細で現在の実装と矛盾があるなら一旦ストップしてユーザーに確認
- [local/Claude] 概要と詳細を見て、そもそも何がしたいかわからない場合も一旦ストップしてユーザーに確認（無理に開始しない）ヒアリングや質問もOK
- [local/Claude] 概要と詳細を見て、Claude側で理解できたら、具体的な計画書やわかりやすい説明でユーザーに実装開始しても大丈夫か確認する

### * [ユーザー] 📋 **承認1: 計画承認**
実装開始の承認する（実装計画・品質基準・測定計画の承認）

---

## 🔧 **Phase 2: 実装・テストフェーズ**

### * [local/Claude] 実装方針確認・説明
- コア機能実装方針の説明
- 技術選択・アーキテクチャの説明  
- 実装スコープ・品質基準の確認

### * [ユーザー] 📋 **承認2: 実装方針承認**
実装方針・技術アーキテクチャの承認（コア機能実装方針・技術選択・アーキテクチャ承認）

### * [local/Claude] 実装
- [local/Claude] 追加で実装したらテストも追加で実装
- 単体テスト・統合テスト作成・実行
- 品質基準への適合確認

### * [local/Claude] テスト結果報告
- 単体テスト結果報告（全PASS確認）
- 統合テスト結果報告（全PASS確認）
- 品質基準達成状況報告

### * [ユーザー] 📋 **承認3: テスト結果承認**
テスト結果・品質基準達成の承認（単体・統合テスト結果・品質基準達成確認）

### * [local/Claude] 修正内容のpushとPullRequestの作成

---

## 📊 **Phase 3: CI・品質ワークフローフェーズ**

### * [github] CIが回る
- [github] 動作チェック
- [github] UnitTest（今あるtestディレクトリ以下のテスト全部）
- [github] テストの中には画像抽出もあるので唯一テスト用の漫画アニメキャラクターの画像をpushするのでそれが抽出できてるかを確認すること
- [github] QCチェック（今あるQC関係全部）

### * [local/Claude] PullRequestの状態を確認する（3分ごとに確認）
CIがすべてSUCCEEDされてるかチェック、jobが待っていたり実行中ならリトライ
jobがエラーなら修正して（成功するまで何度もくり返し改善）

具体的には以下を見続けてすべてのjobが成功するまで繰り返し確認
https://github.com/miyashita337/segment-anything/pull/{pull_request_id}
ですべてのCIやチェックやアクションが通ったら報告してください、それができてない場合は繰り返し改善をしてください

### * [local/Claude] githubのCIの同時並列で以下をする

### * [local/Claude] ワークフロー実行、上記トラッカーIDでワークフロー実行

### * [local/Claude] テストやデモじゃないので全画像を対象として,抽出プログラム実行
```bash
python3 features/extraction/commands/extract_character.py INPUT_DIR -o OUTPUT_DIR --batch
```

### * [local/Claude] 抽出バッチを実行(バックグラウンド)
１ファイルに５分以上かかる場合は、一旦ストップしてユーザーに確認させること

**5分タイムアウトの実装方法**
- 現在は手動監視前提
- 自動タイムアウト機能は未実装なら実装しておくこと

### * [local/Claude] ダッシュボード作成

### * [local/Claude] 品質ワークフロー結果報告
- run_quality_workflow.sh実行結果報告
- ダッシュボード品質確認結果報告
- 品質基準達成状況・劣化有無報告

### * [ユーザー] 📋 **承認4: 品質ワークフロー結果承認**
品質ワークフロー結果・ダッシュボード品質の承認（run_quality_workflow.sh結果・ダッシュボード品質確認）

---

## 📈 **Phase 4: 統計分析・最終承認フェーズ**

### * [local/Claude] 統計分析実行（INCI-004新システム）

**⚠️ 重要: INCI-004により統計分析コマンドが変更されました**

```bash
# 新しい統合統計分析システム（BASELINE_ID必須）
python3 tools/progress_tracker/universal_statistical_analyzer.py --current {TRACKER_ID} --baseline {BASELINE_ID}
```

**BASELINE_ID取得方法:**
```bash
# GoogleSheetで一番更新日付が新しい完了済みトラッカーを確認
PROGRESS_TRACKER_SHEET_NAME="シート1" python3 tools/progress_tracker/cli.py list --status "/release" | head -1
```

**統計分析実行例:**
```bash
python3 tools/progress_tracker/universal_statistical_analyzer.py --current INCI-004 --baseline QUAL-001 --verbose
```

### * [local/Claude] 統計結果をダッシュボードに統合
- 統計レポート（JSON/HTML）生成
- ダッシュボードに統計サマリー追加
- Google Sheets統計列自動更新（universal_statistical_analyzer.pyが自動実行）

### * [local/Claude] ダッシュボード確認

```bash
# 実際の必要コマンド（PRINCIPLE.mdに記載済み）
curl -u admin:[PASSWORD] http://100.123.241.106:8088/tracker/{トラッカーID}
```

- ダッシュボード内の画像もブラウザ上で表示できてることを確認できてること
- 画像はBase64ではなく画像パスであること

### * [local/Claude] 最終結果報告
- PR・統計分析・ダッシュボードの最終確認結果報告
- 全体的な品質・改善効果の総合評価
- 完了基準達成状況の最終報告

### * [ユーザー] 📋 **承認5: 最終承認**
Pull Request・統計分析・ダッシュボードの最終確認承認

### * [local/Claude] このとき修正が入ってたらもう一度commit push をしてCIを回してください
localhostの抽出プログラム実行(features/extraction/commands/extract_character.py)とCIが両方とも成功するように繰り返し改善するようにしてください

### * 両方が問題なく通ったら、ユーザーに報告

### * [ユーザー] ダッシュボード確認

---

## 🎯 **INCI-004での主要変更点**

### 承認回数増加（安心感向上）
- **変更前**: 2回承認（計画承認・最終承認）
- **変更後**: 5回承認（計画・実装方針・テスト結果・品質ワークフロー結果・最終）

### 統計分析システム統合改良
- **変更前**: 存在しないuniversal_statistical_analyzer.pyを参照
- **変更後**: 実装済みuniversal_statistical_analyzer.py使用・BASELINE_ID必須化

### 統計分析コマンド変更
```bash
# 旧（存在しないファイル）
python tools/progress_tracker/universal_statistical_analyzer.py --current {TRACKER_ID} --baseline {BASELINE_ID}

# 新（INCI-004で実装）
python3 tools/progress_tracker/universal_statistical_analyzer.py --current {TRACKER_ID} --baseline {BASELINE_ID}
```

### エラー防止強化
- BASELINE_ID未指定時のエラー表示・強制終了
- 統計分析品質保証（必須引数チェック）
- Google Sheets統計列自動更新

---

## 📋 **承認チェックリスト**

### 📋 承認1: 計画承認
- [ ] 実装計画の妥当性確認
- [ ] 品質基準・測定計画確認
- [ ] リソース要件・スケジュール確認

### 📋 承認2: 実装方針承認
- [ ] コア機能実装方針確認
- [ ] 技術選択・アーキテクチャ確認
- [ ] 実装スコープ・品質基準確認

### 📋 承認3: テスト結果承認
- [ ] 単体テスト結果確認（全PASS）
- [ ] 統合テスト結果確認（全PASS）
- [ ] 品質基準達成確認

### 📋 承認4: 品質ワークフロー結果承認
- [ ] run_quality_workflow.sh結果確認
- [ ] ダッシュボード品質確認
- [ ] 品質劣化・パフォーマンス低下有無確認

### 📋 承認5: 最終承認
- [ ] Pull Request内容確認
- [ ] 統計分析結果確認（BASELINE_ID必須）
- [ ] ダッシュボード最終確認

---

## ⚠️ **注意事項・制約**

### BASELINE_ID必須化
- INCI-004により、統計分析時のBASELINE_ID指定が必須
- 未指定時は明確なエラーメッセージ表示・強制終了
- GoogleSheetの最新完了トラッカーをベースラインとして推奨

### 承認タイミング厳守
- 各承認ポイントでユーザー確認必須
- 承認なしでの次フェーズ移行禁止
- 技術的困難時は必ずユーザー相談

### Git操作安全性
- `git add .` 絶対禁止
- 個別ファイル指定必須: `git add [具体的ファイル名]`
- deprecated/大量未追跡ファイル対策

---

**INCI-004対応完了**: 2025-08-26  
**適用開始**: 即時（次トラッカーから5回承認ワークフロー適用）  
**統計分析**: universal_statistical_analyzer.py使用・BASELINE_ID必須化対応完了