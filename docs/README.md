# 📚 Segment Anything ドキュメントハブ

**最終更新**: 2025-09-03  
**プロジェクトバージョン**: v0.9.27

このディレクトリには、Segment Anythingプロジェクトの全ドキュメントが整理されています。

## 🚀 はじめに

### 重要ドキュメント
- [システム設計](core/ARCHITECTURE.md) - プロジェクトアーキテクチャ概要
- [クイックスタート](core/QUICK_START.md) - 初期セットアップガイド
- [プロジェクト原則](../PRINCIPLE.md) - 開発の基本原則

## 🔧 開発者向け

### 開発プロセス
- [環境構築](development/setup/) - セットアップガイド集
- [開発ワークフロー](workflows/) - AI-人間協調ワークフロー
- [品質管理](development/quality/) - コード品質基準
- [進捗管理](development/progress/) - タスク・進捗管理

### トラッカーシステム 🎯
- [13ステップワークフロー](checklists/tracker_workflow_checklist.md) - 必須チェックリスト
- [統合テンプレート](templates/unified_tracker_template.md) - 計画・進捗・完了報告
- [トラッカーワークスペース](tracker-workspace/) - 各トラッカーの作業領域

## 📊 技術情報

### 仕様書
- [技術仕様](technical_specifications.md) - 統合技術仕様書
- [システム仕様](technical/specifications/) - 詳細仕様書
- [API仕様](technical/api/) - API設計ドキュメント
- [パフォーマンス](technical/performance/) - 性能最適化ガイド

### 依存関係・構造
- [依存関係管理](dependency_reference.md) - パッケージ管理
- [プロジェクト構造](project_structure_reference.md) - ディレクトリ構造ガイド

## 🔌 外部連携

### インテグレーション
- [Google Sheets連携](google_sheets_reference.md) - 進捗管理システム
- [GitHub Actions](github_actions_reference.md) - CI/CD設定
- [詳細設定](integrations/) - 外部サービス連携設定

## 📈 レポート・分析

### 品質レポート
- [品質レポート](reports/quality/) - 品質評価結果
- [完了報告](reports/completion/) - 各フェーズ完了報告
- [分析結果](reports/analysis/) - 技術分析・コンサルテーション

## 🗂️ アーカイブ

### 過去資産
- [履歴文書](archive/historical/) - 過去の重要文書
- [廃止予定](archive/deprecated/) - 非推奨・廃止予定文書

## 🔍 ドキュメント検索のヒント

### 目的別検索
| 目的 | 参照先 |
|-----|--------|
| 新規参加者 | [クイックスタート](core/QUICK_START.md) → [開発ワークフロー](workflows/) |
| トラッカータスク開始 | [13ステップチェックリスト](checklists/tracker_workflow_checklist.md) |
| 品質基準確認 | [技術仕様](technical_specifications.md) → [品質管理](development/quality/) |
| 進捗報告 | [Google Sheets連携](google_sheets_reference.md) |
| トラブルシューティング | [ワークフロー](workflows/troubleshooting_guide.md) |

### カテゴリ別整理
- **計画**: templates/
- **実行**: checklists/
- **参照**: technical/, integrations/
- **報告**: reports/
- **保管**: archive/

## 📝 メンテナンス情報

### ドキュメント管理原則
1. **情報の一元化**: 各分野の情報は1つの統一リファレンスに集約
2. **重複の排除**: 同じ内容を複数のファイルに記載しない
3. **参照の徹底**: 詳細情報は統一リファレンスへの参照で済ませる
4. **定期的監査**: 重複ドキュメントの定期的な確認と統合

### 現在進行中のリファクタリング
- **KIRO-001**: ドキュメント構造最適化プロジェクト進行中
- [移行計画](refactoring/migration_mapping.md) - 段階的移行マッピング

---

**注意**: このドキュメントハブは継続的に改善されています。  
フィードバックは[Issues](https://github.com/username/segment-anything/issues)へお願いします。