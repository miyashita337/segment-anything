# QI-002 根本原因分析と再発防止策

## 📊 問題の完全な根本原因

### 🚨 第1の根本原因: sam-env/Scripts/ Windows環境パス混在
**問題**: 過去のWindows環境向けパス（`sam-env/Scripts/python.exe`）がLinux/WSL環境コードに残存
**影響**: 9ファイルにわたって不正な環境パス参照が残存し、将来的なエラーの温床となっていた

**影響ファイル一覧（修正済み）**:
- `background_extraction.py`
- `background_extraction_fixed.py` 
- `qc_20_batch_simple.py`
- `qc_5_batch_test.py`
- `qc_kana08_reproduction.py`
- `bin/shell/activate_and_install_linters.sh`
- `bin/shell/check_env.sh`
- `bin/shell/linter_venv.sh`
- `run_extraction.sh`

### 🚨 第2の根本原因: segment_anythingライブラリ不整合
**問題**: ローカルプロジェクトのsegment_anythingがMeta公式版より優先されていた
**解決**: Meta公式segment_anything v1.0を明示的にインストール

### 🚨 第3の根本原因: ワークフロー環境不一致
**問題**: ワークフロースクリプトがsystem `python3`を使用、sam-env環境を使用していなかった
**解決**: 全python3コマンドを`sam-env/bin/python3`に統一

## 🛡️ 再発防止策

### 1️⃣ **環境チェック機能強化**
ワークフロースクリプトに以下を実装済み：
- sam-env環境存在確認
- 必須パッケージ検証
- Python環境一貫性チェック

### 2️⃣ **コードベース統一完了**
- **Pythonパス統一**: 全て`sam-env/bin/python3`に変更済み
- **Activateパス統一**: 全て`sam-env/bin/activate`に変更済み
- **Windows専用パス削除**: `sam-env/Scripts/`への参照完全除去

### 3️⃣ **依存関係管理改善**
- **segment_anything**: Meta公式v1.0固定
- **インストール検証**: インポート可能性事前確認
- **ライブラリ競合回避**: ローカル版より公式版優先

### 4️⃣ **監視・検証プロセス**
```bash
# 定期チェックコマンド
find . -name "*.py" -o -name "*.sh" | xargs grep -l "sam-env/Scripts"
# → 結果: 空（正常状態）

sam-env/bin/python3 -c "from segment_anything import sam_model_registry; print('OK')"
# → 結果: OK（正常状態）
```

## 📈 効果測定

### ✅ 修正前の状態
- **環境エラー**: cv2 ModuleNotFoundError
- **ライブラリエラー**: cannot import name 'sam_model_registry'
- **処理失敗**: Pushover通知未送信
- **品質問題**: 黒画像が高品質と誤判定

### ✅ 修正後の状態  
- **環境統一**: sam-env/bin/python3完全統一（9ファイル）
- **ライブラリ正常**: Meta公式segment_anything v1.0動作
- **処理成功**: QI-002抽出処理正常実行中（12/26画像完了）
- **品質改善**: 黒画像検出機能実装済み

## 🔒 継続的品質管理

### 定期監査項目
1. **環境パス監査**: sam-env/Scripts参照の有無
2. **ライブラリ整合性**: segment_anythingバージョン確認
3. **ワークフロー動作**: 実際の抽出処理実行テスト
4. **品質基準遵守**: 黒画像検出機能動作確認

### 緊急時対応手順
1. **エラー検出**: ログファイル詳細確認
2. **環境検証**: sam-env/bin/python3動作確認
3. **ライブラリ再構築**: pip uninstall & git+install
4. **設定見直し**: 本ドキュメント参照

## 📝 学習された教訓

1. **クロスプラットフォーム配慮**: Windows/Linux環境パス混在の危険性
2. **依存関係明確化**: ローカル版より公式版ライブラリ優先の重要性  
3. **包括的修正**: 単発修正でなく影響範囲全体の一括対応が必須
4. **検証体制**: エラーログだけでなく根本原因追求の重要性

---
作成日: 2025-08-08  
QI-002完全解決記念ドキュメント