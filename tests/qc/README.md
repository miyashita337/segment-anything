# QC（品質管理）システム

## 概要
このディレクトリは品質管理（Quality Control）システムのスクリプト群を含みます。

## ファイル構成

### 統合QCシステム
- `unified_batch_extraction.py` - 汎用バッチ抽出+品質チェック（kana08/05/07対応）
- `compatible_extraction_system.py` - QC成功版互換システム（汎用性高）

## 使用方法

### 複数データセット対応バッチ処理
```bash
python tests/qc/unified_batch_extraction.py
```

### QC互換抽出システム
```bash
python tests/qc/compatible_extraction_system.py [入力ディレクトリ] [出力ディレクトリ]
```

## 特徴
- **汎用性**: 特定データセットに依存しない設計
- **品質保証**: 抽出結果の品質を自動評価
- **統合管理**: 複数のQC手法を統一インターフェースで提供

## 歴史的経緯
2025-08-09に既存のデータセット特化QCファイル群を統合・汎用化して作成。
レガシーファイルは`deprecated/qc_legacy/`に移動済み。