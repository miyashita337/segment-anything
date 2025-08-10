# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## バージョニング修正について

**重要**: v1.0 および v1.0.1 は実際の品質・完成度に対して不適切なバージョン番号でした。
本プロジェクトは開発段階にあり、v0.x.x のバージョニングが適切です。
今後は v0.1.2 から継続し、真の安定版達成時に v1.0.0 をリリースします。

## [v0.9.7] - 2025-08-10

### Added
- **QCA-001: 作者別パラメータ適応システム** - ディレクトリ構造ベースの自動作者検出・最適化
  - `features/adaptation/`: 作者別適応システムモジュール実装
  - `AuthorParameterAdapter`: yado, kiri, zundamon作者プロファイル自動適用
  - ディレクトリ構造解析: `/train/{author}/org/{work}/` パターンからの自動識別
- **設定ファイルベースアーキテクチャ**: `config/author_config.yaml` - YAMLベースの動的設定管理
  - ハードコーディング完全除去・横展開対応システム
  - 後方互換性100%維持・フォールバック機能実装
- **汎用ダッシュボード統合**: `tools/scripts/universal_dashboard_merger.py`
  - 複数作者ワークスペース統合システム
  - Base64画像表示・品質バッジ自動生成
  - 統一URL形式: `http://100.123.241.106:8088/tracker/{TRACKER_ID}`

### Changed
- **作者別最適化パラメータ**:
  - yado: バランス型・YOLO信頼度0.07
  - kiri: 細密描写特化・YOLO信頼度0.05
  - zundamon: シンプルスタイル・YOLO信頼度0.08
- **統合テストスイート**: `tests/integration/test_qca001_author_adaptation.py` - 作者検出・パラメータ適用検証

### Fixed
- **設定システム**: AuthorParameterAdapterV2による動的設定読み込み
- **ダッシュボード統合**: 17画像統合処理（yado: 14枚、kiri: 3枚）
- **品質評価**: 高品質5枚、中品質6枚、品質スコア47.1%

## [v0.9.6] - 2025-08-10

### Added
- **重複防止システム**: `features/common/file_utils.py` - 重複サフィックス自動防止機能
  - `generate_output_filename()`: 重複サフィックス回避ファイル名生成
  - `clean_duplicate_suffixes()`: 既存重複サフィックスクリーニング
  - `is_already_processed()`: 処理済みファイル自動検出
- **包括的テストスイート**: `tests/unit/test_duplicate_prevention.py` - 12テストケースによる品質保証
- **クリーンアップツール**: `tools/utils/cleanup_duplicates.py` - 既存重複ファイル自動整理機能

### Fixed
- **重複処理根本解決**: `_multi_char_detection_multi_char_detection.jpg` パターンの完全除去
  - QCC-011実績: 6時間処理時間 → 正常時間への大幅短縮
  - 7個重複ファイルの適切処理（4個置換・3個削除）
- **MultipleCharacterDetector改善**: 重複処理スキップ機能実装
  - ファイル名重複チェック: `_multi_char_detection`含有ファイル自動スキップ
  - 既存出力ファイル検出による重複処理完全防止
- **extract_character.py最適化**: `generate_output_filename()`統合による出力ファイル名重複防止
- **clean_duplicate_suffixes修正**: 正規表現パターン改善による正確なクリーニング

### Changed
- **処理効率大幅改善**: 重複処理除去により処理時間の劇的短縮
- **QCC-011ダッシュボード更新**: 全57枚画像表示対応・重複サフィックスファイル0個達成
- **品質保証強化**: 12/12テスト合格・実動作テスト完了

### Technical Details
- **最小限修正アプローチ**: SQLite等大規模変更を回避した軽量実装
- **Google Sheets非競合**: 既存進捗管理システムと完全分離
- **後方互換性**: 既存機能への影響ゼロ保証
- **自動防止機能**: 今後の重複処理問題完全予防システム

## [v0.9.5] - 2025-08-10

### Added
- **QCC-061トラッカー**: PyTorch決定論的実行と性能の両立ハイブリッドモード提案システム
- **直接画像表示**: Base64エンコードから直接画像パスへの変更によるダッシュボード軽量化

### Changed 
- **大幅性能向上**: 決定論的実行無効化により処理速度3.6倍改善（4時間 → 67分）
- **SAMマスク最適化**: 無制限マスク生成から10マスク制限へ変更（96.5%削減）
- **ダッシュボード軽量化**: 8.5MBから43KBへの大幅サイズ削減
- **画像表示サイズ**: ユーザビリティ改善のため50%サイズで一覧表示

### Fixed
- **types.py競合修正**: 標準ライブラリとの名前衝突をcustom_types.pyリネームで解決
- **circular importエラー**: モジュール間の循環インポート問題を解決
- **メモリ効率化**: RAM 2.2GB、GPU 2.9GBでの安定稼働実現

### Performance
- **QCC-011完全成功**: 43/43枚処理で100%成功率達成
- **平均処理時間**: 93秒/枚（従来の4-5倍高速化）
- **torch.backends.cudnn.benchmark=True**: GPU最適化有効化

## [v0.9.4] - 2025-08-09

### Added
- **プロジェクト構造改善**: 新しい`deprecated/`ディレクトリ階層
  - `deprecated/qc_legacy/` - データセット特化QCスクリプト群
  - `deprecated/dashboard_legacy/` - 停止中ダッシュボードサーバー
  - `deprecated/static_dashboards/` - 静的HTMLダッシュボード
  - `deprecated/scripts_legacy/` - 背景抽出・一時スクリプト群
- **QCシステム統合**: `tests/qc/`配下に統一品質チェックシステム
  - `unified_batch_extraction.py` - 汎用バッチ抽出+品質チェック
  - `compatible_extraction_system.py` - QC成功版互換システム
- **GPT5連携**: `tools/scripts/gpt5_lora_quality_evaluator.py` - AI品質評価ツール
- **ドキュメント構造化**: `docs/`配下に重要文書集約
  - `docs/setup/` - Windows/WSL環境セットアップガイド
  - 非同期システムレポート・改善文書の移動

### Changed
- **Pushover通知システム統一化**: DRY原則適用・重複排除
- **linter環境安定性向上**: 仮想環境セットアップスクリプト改善
- **YOLO wrapper機能拡張**: パフォーマンス・信頼性向上
- **品質チェッカー機能拡張**: 境界ケース検出システム強化

### Removed
- **レガシーファイル整理**: ルートディレクトリの散在ファイル削除
- **重複QCスクリプト**: データセット特化・属人的なスクリプト群
- **停止中ダッシュボード**: 非稼働サーバーファイル
- **一時ファイル**: セッション記録・背景抽出スクリプト群

### Security
- **APIキー保護強化**: `.gitignore`にAPI設定ファイル追加
  - `config/api_keys.json` - Gemini APIキー等
  - `config/google_api_monitoring.json` - API監視設定
- **秘匿ファイル管理**: Git履歴からの漏洩防止

### Performance  
- **稼働中システム保護**: `integrated_dashboard_server.py`、`external_dashboard_server_8080.py`維持
- **プロジェクト構造最適化**: 開発・保守効率向上
- **品質保証体制**: QCシステム統合による品質向上

## [v0.9.3] - 2025-08-09

### Added
- **QI-006: 複数キャラクター検出改善システム**
  - 信頼度基準から面積基準への選択アルゴリズム変更
  - YOLO bbox → SAM prompt のHybrid方式実装
  - fullbody_priority品質評価メソッド採用
  - GPT-5評価との統合ダッシュボード実装
  - multiple_character_detector.py: 複数キャラクター検出分析機能
  - アルゴリズムテストダッシュボード追加

### Changed
- **extract_character.py** - 最大面積マスク選択ロジック実装
- **sam_wrapper.py** - generate_masks_with_bbox_promptメソッド追加
- **start.py** - グローバル変数名修正（model初期化問題解決）

### Fixed
- **複数キャラクター重複問題** - kana08_0014.jpg等の5人重複キャラクター問題改善
- **GPT-5評価精度** - 漫画キャラクター認識の限界把握と対策実装
- **SAM predictor属性エラー** - Grid方式へのフォールバック動作確認

### Performance
- **kana08データセット**: 24枚の再抽出完了（修正アルゴリズム適用）
- **Web統合ダッシュボード**: http://100.123.241.106:8088/tracker/QI-006 実装
- **抽出品質改善**: 面積基準選択により下半身のみ抽出問題の特定（QI-007チケット作成）

## [v0.9.2] - 2025-08-08

### Added
- **QI-003: 統合品質評価システム実装**
  - BoundaryCaseDetector による黒画面検出（100%精度）
  - AnimeImagePreprocessor による明度改善（+1820.8%効果実証）
  - 統合品質チェッカーの動作確認と機能テスト
  - features/common/notification/pushover_image_sender.py 実装
- **QI-004: ダッシュボード標準化・Base64画像表示システム**
  - StandardDashboardGenerator クラス実装
  - Base64画像埋め込み機能（2-3MB HTMLファイル生成）
  - 品質バッジシステム（高品質・中品質・低品質の自動判定）
  - 統一URL形式: http://100.123.241.106:8088/tracker/{TRACKER_ID}
- **QI-005: Pushover通知システム統一化**
  - 17ファイルの分散実装から共通モジュール化（20/31ファイル統一）
  - 全抽出画像添付送信機能（10枚制限対応バッチ送信）
  - tools/scripts/unify_pushover_notifications.py 作成

### Changed
- **run_quality_workflow.sh** - 標準ダッシュボード生成統合（192-208行目）
- **integrated_dashboard_server.py** - 画像アクセス制限解除（VPN+Basic認証で保護）
- **Google Sheets進捗管理** - QI-003/004/005詳細情報完全記載

### Fixed
- **QI-002/003 Pushover通知問題** - 通知が来ない問題を完全解決
- **QI-004構文エラー** - 1878行目未終了文字列修正・8KB→2.9MB復旧
- **ダッシュボード画像表示** - ブラウザ表示不具合修正（Base64フル埋め込み）
- **黒画面問題の実態把握** - QI-002で12.5%、QI-003で15.0%の黒画面検出・解決策確認

### Performance
- **黒画面検出・改善**: 明度6.6 → 126.2（+1820.8%改善実証）
- **Pushover画像送信**: QI-002（24枚）、QI-003（20枚）全画像送信完了（100%成功率）
- **ダッシュボード生成**: 2.7-2.9MB Base64画像フル埋め込み（100%表示成功）
- **通知統一化**: 64.5%統一化率・100%送信成功率

## [v0.9.1] - 2025-08-08

### Added
- **統合ダッシュボードサーバー拡張** - INTEGRATE-3-6シリーズ対応
  - 8088ポートでの統合管理システム
  - Tailscale経由の外部アクセス対応（100.123.241.106:8088）
  - Phase 3カテゴリとINTEGRATE統合パイプライン表示
- **WSL2ポートフォワーディング対応** - Windows/Mac/iPhoneからのアクセス改善
- **INTEGRATE-3-6-05評価版** - kana08データセット最終評価

### Changed
- **YOLOモデル切り替え機能** - yolov8x.pt ⟷ yolov8x6_animeface.pt
- **ワークフロードキュメント更新** - 8088統合サーバー運用方針明記
- **ダッシュボード自動認識** - tracker-workspace配下の自動検出

### Fixed
- **Basic認証処理** - 8088サーバーの認証安定性向上
- **ダッシュボード生成** - Base64埋め込みの最適化

### Performance
- INTEGRATE-3-6-05: 24/26枚成功（92.3%成功率）
- 8088サーバー: 48個のダッシュボード統合管理

## [v0.9.0] - 2025-08-05

### Added
- **Phase 3-6統合パイプライン実装** - 単一プロセスでの完全ワークフロー
  - 状態管理とレジューム機能（チェックポイントベース）
  - 多層バリデーション（入力パス・設定・依存関係）
  - Web対応HTMLダッシュボード自動生成
- **Pushover画像送信機能** - 抽出結果上位10枚の自動送信
  - ファイルサイズ制限対応（2MB以下）
  - レート制限対応（0.5秒間隔）
  - エラーハンドリングと詳細ログ
- **統合設定システム** - YAML設定ファイルによる包括的制御
- **モックテストフレームワーク** - 全Phaseの動作検証システム

### Changed
- **設計原則の明確化** - 安全性とロバスト性を運用コストより優先
- **エラーメッセージの統一** - 詳細な対処方法を含む統一フォーマット
- **ダッシュボード統合** - 既存サーバーとの自動連携強化

### Technical Details
- `tools/core/integrated_quality_pipeline.py`: メインパイプライン（1,000+ 行）
- `config/pipeline_config.yaml`: 包括的設定ファイル
- `tools/scripts/run_integrated_pipeline.sh`: Bash実行ラッパー
- `docs/workflows/integrated_pipeline_phase3_6.md`: 統合仕様書

### Performance
- Phase 3: バリデーション（1.7秒）
- Phase 4: キャラクター抽出（5枚成功、kana05データセット）
- Phase 5: 品質評価（成功率48.7%）
- Phase 6: ダッシュボード・Pushover送信（0.9秒）

## [v0.8.5] - 2025-08-04

### Added
- **P1-022 Ubuntu環境移行完了**: Windows PE32+からUbuntu python3ネイティブ環境への完全移行
  - charset issues (cp932→UTF-8) 完全解決
  - spec.md準拠（python3優先、WSL2、ubuntu仮想環境必須）
  - QC成功システム100%動作確認・バッチ抽出26/26成功率達成
  - Pushover連携でcharset問題解消確認
- **send_pushover_images.py**: Ubuntu環境対応Pushover送信システム実装
  - 高品質画像10枚の自動選択・送信機能
  - UTF-8対応でcharset エラー解消
  - QC基準サイズ(604×1166)完全再現確認

### Fixed  
- **P1-B004品質劣化問題の教訓ドキュメント化**: 包括的品質改善ガイド追加
  - 50%品質閾値過適用問題→監視専用運用への方針転換
  - 3段階処理パイプライン→デフォルト無効化で安定性確保
  - 複雑スコアリング→QC成功版単純アルゴリズムへの統一
  - シンプルファースト原則の確立

### Changed
- **品質指標運用方針**: 処理制御から監視・統計専用への転換
  - 全画像パス方式（0%閾値）+ 段階的警告システム（5%, 10%, 15%）
  - 品質データ蓄積による将来改善基盤の構築
  - ドメイン適合性重視（アニメキャラクター特化指標）

### Technical
- **Environment Migration**: sam-env Windows PE32+ → sam-env-ubuntu python3 native
- **Dependency Backup**: sam-env-windows-backup.txt環境記録保持
- **Charset Resolution**: cp932→UTF-8移行によるPushover・ログ出力問題解消
- **QC Compatibility**: 100%互換性維持・26/26バッチ抽出成功

### Documentation
- **integrated_quality_check_guide.md**: P1-B004教訓とシンプルファースト原則
- **quality_evaluation_guide.md**: 品質指標適用方針と運用ルール明確化

### Performance
- **Extraction Success**: 26/26 images (100%) on kana08 dataset with Ubuntu environment
- **Quality Reproduction**: QC reference size (604×1166) perfectly achieved
- **Charset Issues**: Complete resolution of cp932 encoding problems

この包括的な環境移行により、仕様書準拠のUbuntu環境での安定動作と、品質問題の根本的解決指針が確立されました。

## [v0.8.4] - 2025-08-04 

### Fixed
- **P1-B004品質劣化完全解決**: SAM Predictor採用によりQC成功版と100%互換実現
- **画像サイズ劣化修正**: kana08_0002で604×1166サイズ完全再現達成  
- **抽出成功率向上**: 全52ファイルで100%成功率実現
- **YOLOモデル統一**: yolov8x6_animeface.pt→yolov8x.pt移行で安定性向上

### Added
- **QC互換抽出システム**: `qc_compatible_extraction.py`で完全互換処理実装
- **LoRA学習適性評価システム**: 品質調査・評価機能追加

### Changed  
- **SAMアーキテクチャ**: AutomaticMaskGenerator → SAM Predictor移行
- **画像前処理**: max_size 1024→4096で元サイズ保持
- **マスク選択アルゴリズム**: 複雑評価システム → QC成功版シンプルアルゴリズム

## [v0.8.3] - 2025-08-03

### Added
- **QC品質調査バッチ抽出システム**: 107枚の高品質キャラクター抽出を自動実行
  - SAM+YOLO統合パイプライン（0.6秒/画像の高速処理）
  - 3フォルダ完全自動処理（KANA08: 26枚、KANA05: 39枚、KANA07: 42枚）
  - 100%成功率でのバッチ処理安定性確認
  - バックグラウンド実行・Pushover通知統合
- **LoRA学習適性評価フレームワーク**: GPT-4o・Gemini協議による専門的分析
  - 影響度分析による学習データ適性の定量評価（他キャラ混入: 5/5、テキスト混入: 3/5）
  - アニメキャラクター特化の技術的考慮事項
  - 包括的改善提案・Phase別実装ロードマップ
- **Pushover通知テストスクリプト**: 通知機能の動作確認・デバッグツール
- **包括的技術レポート**: 抽出品質・パフォーマンス・改善提案の詳細文書化

### Fixed
- **Pushover通知設定**: api_token/user_keyキー名修正により通知機能復旧
  - 設定ファイル構造の修正（"token"→"api_token", "user"→"user_key"）
  - テスト送信成功確認（ステータス200、画像添付対応）
- **WSL2アクセス問題**: エクスプローラーアクセス代替方法の提供
  - `\\wsl$\Ubuntu-22.04`パスでの直接アクセス方法案内
  - WSLネットワーク機能の診断・修復手順提供

### Technical Highlights
- **処理規模**: 1.1分で107枚完全処理（失敗0件）
- **AI協議**: 複数AIによる技術的妥当性検証・学習適性評価
- **品質保証**: SAM精密セグメンテーション＋YOLO detection統合

### Performance
- **抽出効率**: 0.6秒/画像（GPU CUDA最適化）
- **成功率**: 100%（107/107枚）
- **メモリ使用**: 安定1.6GB使用量
- **YOLO検出**: 10-25ms/画像、confidence=0.07

### Documentation
- **QC_EXTRACTION_ANALYSIS.md**: 技術分析・パフォーマンス詳細
- **LORA_TRAINING_SUITABILITY_REPORT.md**: LoRA学習適性総合評価
- **QC_FINAL_REPORT.md**: 実行サマリー・成果物一覧

## [v0.8.3] - 2025-08-02

### Validation
- **P1-016抽出プログラム統合検証完了**: extract_character.pyでの統合機能動作確認
  - P1-018バッチサイズ制御改善の統合確認
  - P1-019プロセス安定性向上（StableBatchProcessor）の統合確認
  - P1-020 SAM推論最適化（93%処理時間短縮）の統合確認
  - 実抽出テスト: 12枚成功、品質スコア60.0%、LCA精度100%、A/B評価率87.5%
  - Google Sheets `/release`ステータス更新完了

### Verification
- **統合抽出システム動作確認**: sam_yolo_character_segment.py → extract_character.py統合完了の実証
- **性能改善確認**: 処理時間65.3秒で12枚抽出、メモリ最適化-55.8MB確認
- **安定バッチ処理確認**: マイクロバッチ処理とチェックポイント機能の正常動作

## [v0.8.2] - 2025-08-02

### Documentation
- **P1-DOC-UPDATE完了**: v0.8.1修正内容を反映したドキュメント統一更新
  - `CLAUDE.md`: v0.8.1の重要修正内容（BatchMemoryManagerスコープエラー、面積比閾値、拡張子修正）を追加
  - `docs/workflows/output_directory_config.md`: v1.1に更新、新デフォルトパス設定を追加
  - 実運用検証: 12枚抽出成功、品質評価完了、ダッシュボード生成確認

### Validation
- **実抽出テスト**: kana05データセットで12枚成功抽出（品質スコア60.0%）
- **品質指標確認**: LCA精度100%、A/B評価率87.5%、FPS 0.667達成
- **ワークフロー検証**: 統合品質チェック・ダッシュボード生成・Google Sheets連携正常動作

## [v0.8.1] - 2025-08-02

### Fixed
- **BatchMemoryManagerのスコープエラーを修正**: 内部関数からのデコレータ呼び出し時のエラーを解消
- **画像抽出時の拡張子不整合を修正**: 実際の出力形式(.jpg)に合わせて修正
- これにより抽出成功率が12.8%から87.2%に大幅改善（680%向上）

### Changed
- **面積比閾値の調整**: 0.01から0.005に変更し、より小さなマスクも受け入れるように改善

## [v0.8.0] - 2025-07-31

### Added
- **P1-006階層的品質評価システム**: AIによる説明可能な品質分析機能
  - `features/evaluation/explainable_quality.py` - AI品質解析エンジン実装
  - `features/evaluation/hierarchical_quality.py` - 多層品質評価システム
  - デモ出力付き品質分析結果（高品質・中品質・低品質・特殊ケース対応）
- **P1-015大規模データセット対応メモリ最適化**: スケーラブル処理システム
  - `features/common/memory_optimizer.py` - 動的メモリ管理・バッチサイズ調整
  - `features/processing/large_dataset_processor.py` - 大規模データセット専用処理
  - メモリ圧迫検出（85%閾値）・チェックポイント機能・プログレッシブ処理
- **共通機能強化**: 堅牢性向上のための基盤システム
  - `features/common/input_validation.py` - 統一入力値検証システム
  - `features/common/retry_handler.py` - 指数バックオフ付きリトライ機能
  - `config/workspace_config.py` - ワークスペース設定統合管理

### Fixed
- **トラッカーIDフォーマット制限解除**: Google Sheets A列全フォーマット対応
  - 旧制限: `PH{Phase}-{3桁}` 固定 → 新対応: P1-XXX, PH2-XXX, T-XXX, BAT-XXX, CLAUDE-OPT-XXX等
  - `features/common/output_path_manager.py`の正規表現ベース柔軟検証に変更
  - 複合フォーマット（P1-A002-1, P1-B001）もサポート

### Performance
- **メモリ使用量最適化**: 87.4MB削減確認（P1-015実装効果）
- **バッチ処理安定性**: 92.3%成功率での大規模データセット処理

### Testing
- **テストスイート拡充**: 新機能対応の包括的テスト追加
  - `tests/integration/test_p1_015_integration.py` - 大規模データセット統合テスト
  - `tests/unit/test_explainable_quality.py` - AI品質分析テスト
  - `tests/unit/test_hierarchical_quality.py` - 階層品質評価テスト
  - `tests/unit/test_memory_optimizer.py` - メモリ最適化テスト
  - `tests/unit/test_large_dataset_processor.py` - 大規模処理テスト

## [v0.7.0] - 2025-07-29

### Added
- **ワークスペースパス一元管理システム**: 全パス設定を`config/workspace_config.py`で統合管理
  - 環境変数`TRACKER_WORKSPACE_BASE`による動的設定変更対応
  - 3つの主要ファイル統合（output_path_manager.py, validate_tracker_completion.py, run_quality_workflow.sh）
  - バリデーション・設定確認機能付き
- **P1-005自動マスク修正機能**: `features/processing/postprocessing/auto_mask_correction.py`実装
  - 適応的マスク拡張による品質改善
  - エッジ保持フィルタリングシステム
  - 手足切断防止処理
- **ドキュメント統一リファレンス化**: 技術文書の体系的整理
  - `docs/technical_specifications.md` - 統合技術仕様書
  - `docs/dependency_reference.md` - 依存関係リファレンス
  - `docs/project_structure_reference.md` - プロジェクト構造ガイド
  - `docs/github_actions_reference.md` - GitHub Actions統合文書
- **トラッカー完了検証システム**: `tools/utils/validate_tracker_completion.py`
  - 4回目仕様違反の恒久対策として実装
  - ワークスペース構造・抽出結果・品質レポート・ダッシュボードの包括検証
  - A-F評価での総合判定システム

### Changed
- **CLAUDE.md大幅強化**: トラッカーワークフロー必須要件の明文化
  - シリアル処理要件（パラレル実装禁止）の徹底
  - 入力検証強化（存在しないディレクトリでの強制実行防止）
  - チェックボックス式完了判定基準の導入
  - 禁止表現リスト（ワークスペース出力完了まで使用禁止）
- **ワークスペースパス標準化**: ハードコードされたパスを全面的に統合
  - `/mnt/c/AItools/lora/train/yado/clipped_boundingbox/workspace/` → `/mnt/c/AItools/lora/train/yado/tracker-workspace/`
  - パス変更時の影響範囲を1箇所に集約

### Fixed
- **P1-007完了検証問題**: 空の抽出ディレクトリとダッシュボードグラフ未生成問題を解決
  - 正しい入力パス（`/mnt/c/AItools/lora/train/yado/org/kana05`）での抽出実行
  - 39枚の画像を正常に処理し、完全なワークスペース出力を生成
- **入力検証の欠如**: 存在しないディレクトリでの強制実行を防止
  - `features/common/input_validation.py`による統一検証システム
  - 詳細なエラーメッセージと対処方法の提示

### Technical
- **設定管理の近代化**: 環境変数ベースの柔軟な設定システム
- **品質保証ワークフローの自動化**: シェルスクリプトでのPython設定統合
- **テストスイート拡張**: 新機能に対応した単体テスト追加
- **ドキュメント構造最適化**: 重複排除と参照性向上

### Security
- **設定ファイル一元化**: 散在していたパス設定による設定ミス防止
- **入力検証強化**: 不正なパスでの処理実行を防止

この包括的な改善により、トラッカーワークフローの品質保証が大幅に向上し、設定管理の運用性が改善されました。

## [v0.6.0] - 2025-07-28

### Added
- **P1-A003 自動テスト強化システム**: 品質劣化の事前検出機能を実装
  - 3段階テストモード (quick/standard/full)
  - 継続監視システムによる24時間365日監視
  - 品質劣化検出メカニズム
  - 自動ダッシュボード生成機能
  - ベースライン比較による品質トレンド分析
- **P1-A001 改善コード復旧プロジェクト**: 15個の改善機能を本番環境に復旧
  - true_success_analyzer.py: 真の成功分析システム
  - visual_verification_system.py: 視覚的検証システム
  - kana08_enhanced_stable_batch.py: 拡張バッチ処理
  - その他12個の改善機能
- **ツール階層化**: tools/ ディレクトリの大規模リファクタリング
  - core/: 中核機能
  - batch/: バッチ処理
  - scripts/: 実行スクリプト
  - testing/: テストツール
  - utils/: ユーティリティ
  - legacy/: レガシーコード

### Changed
- **ワークスペース構造**: PROGRESS_TRACKER.md準拠の正しいパスに修正
  - 誤: `/mnt/c/AItools/segment-anything/workspace/`
  - 正: `/mnt/c/AItools/lora/train/yado/clipped_boundingbox/workspace/`
- **tools/ ディレクトリ構造**: 機能別に整理・分類
  - 34個のファイルを適切なサブディレクトリに移動
  - 実行順序・依存関係の明確化

### Fixed
- **P1-A003 ワークスペースパス問題**: 4回の指摘後に正しいパスに修正
- **品質評価精度**: A/B評価率の正確な計算と表示
- **Google Sheets連携**: ステータス更新と品質指標の同期

### Technical
- **テストカバレッジ**: 新規ユニットテスト追加
  - test_automated_quality_testing.py (9テスト)
  - test_p1_a001_recovery.py (15機能検証)
- **継続的品質監視**: monitoring_config.json による設定管理
- **成功率改善**: P1-A001で100%成功率達成（15/15機能）

### Security
- **廃止ファイル管理**: 重要ファイルをdeprecated/tools_archive/に安全保管
- **ファイル保護チェックリスト**: file_protection_checklist.py による重要度分類

この大規模な品質改善により、自動テスト強化と改善コード復旧の基盤が確立されました。

## [v0.2.0] - 2025-07-27

### Added
- **Google Sheets優先度管理**: B列に4段階優先度システム追加（優先度最高/高/中/低）
- **タスク起票システム**: 67個の残存タスクを体系的にGoogle Sheetsに一括登録
- **22列管理システム**: 21列から22列構造への拡張（A-V列対応）
- **Progress Tracker統合**: TaskRecordモデルの優先度フィールド拡張
- **自動ドロップダウン設定**: B列優先度選択のUI改善
- **一括タスク管理**: Phase 1-4とT-seriesタスクの完全分類・起票
- **統合テストスイート**: TEST-001による全フロー検証システム

### Changed
- **データ構造拡張**: progress_tracker data modelsの22列対応
- **Google Sheets API**: update/append処理の優先度パラメータ対応  
- **StatusUpdateHook**: デフォルト優先度「優先度中」設定
- **ワークフロー文書**: タスク起票プロセスの標準化ドキュメント追加

### Fixed
- **シート構造移行**: 既存8レコードの列シフト自動処理
- **API レート制限**: Google Sheets書き込み制限への対応
- **データ整合性**: 優先度未設定時のデフォルト値処理

### Technical
- **PriorityLevel Enum**: 4段階優先度の型安全実装
- **バッチ処理最適化**: 67件中64件の起票成功（95.5%成功率）
- **重複防止機能**: find_existing_record()による既存チェック
- **自動移行ツール**: update_sheet_headers.pyによる構造変更

この大規模な機能追加により、優先度ベースの効率的なプロジェクト管理基盤が確立されました。

## [v0.1.2] - 2025-07-27

実質的には v1.0.1 と同じ内容ですが、適切なバージョン番号に修正しました。

### Fixed
- Phase 1 改善コード (kana08_stable_batch.py) を deprecated フォルダから復元
- A/B評価率が実際は 80% なのに 0% と表示されていた問題を修正
- SCI値を 0.400 から 0.853 に修正

### Added
- 柔軟なトラッカーID形式による品質保証ワークフローシステム (P1-XXX, PH2-XXX, T-XXX, BAT-XXX, CLAUDE-OPT-XXX等のGoogle Sheets A列フォーマット全対応)
- ファイル保護チェックリストツール (`tools/file_protection_checklist.py`)
- 自動品質ワークフロースクリプト (`tools/run_quality_workflow.sh`)
- 出力ディレクトリ設定ドキュメント (`docs/workflows/output_directory_config.md`)
- 実装レポートテンプレート (`docs/templates/implementation_report_template.md`)
- PH2-001 の根本原因分析 (`PH2-001_ROOT_CAUSE_ANALYSIS.md`)

### Changed
- CLAUDE.md にファイル保護原則と重要度分類を追加
- PRINCIPLE.md に着実な改善原則 ("歩みは少なくても着実な方法") を追加
- folder_structure.md に 6段階必須ファイル移動チェックリストを追加
- PROGRESS_TRACKER.md に完全な品質保証ワークフローを追加

### 開発状態
- 抽出成功率: 80% (Phase 1 改善適用時)
- 品質評価: A/B評価率 80%、SCI値 0.853
- 総合品質: 75.0% (PASS)
- **注意**: まだ開発中であり、本番利用には適していません

## [v1.0.1] - 2025-07-27 [誤ったバージョニング]

### Fixed
- Restored Phase 1 improvements (kana08_stable_batch.py) that were incorrectly moved to deprecated folder
- Fixed A/B evaluation rate showing 0% when it was actually 80%
- Corrected SCI value from 0.400 to 0.853

### Added
- Comprehensive quality assurance workflow system with flexible tracker ID format (supports P1-XXX, PH2-XXX, T-XXX, BAT-XXX, CLAUDE-OPT-XXX and all Google Sheets A column formats)
- File protection checklist tool (`tools/file_protection_checklist.py`)
- Automated quality workflow script (`tools/run_quality_workflow.sh`)
- Output directory configuration documentation (`docs/workflows/output_directory_config.md`)
- Implementation report template (`docs/templates/implementation_report_template.md`)
- Root cause analysis for PH2-001 (`PH2-001_ROOT_CAUSE_ANALYSIS.md`)

### Changed
- Enhanced CLAUDE.md with file protection principles and severity classification
- Updated PRINCIPLE.md with steady improvement principle ("歩みは少なくても着実な方法")
- Improved folder_structure.md with 6-step mandatory file movement checklist
- Expanded PROGRESS_TRACKER.md with complete quality assurance workflow

### Thanks
- @miyashita337 for identifying the Phase 1 code issue and providing guidance on steady improvement

## [v0.1.0-cleanup] - 2025-07-26

### Added
- **Comprehensive File Structure Analysis**: Created `folder_structure.md` with detailed analysis of all 383 project files
  - Categorized files by type: core, features, tests, documentation, temporary, backup, logs
  - Analyzed file purposes based on git history, code references, and conversation context
  - Documented archival strategy and retention policies
- **Organized Deprecated Archive**: Structured `deprecated/` folder with 6 categorized subdirectories
  - `backup/` - Historical backups and migration files (55 files)
  - `scripts/` - Temporary and experimental scripts (43 files)
  - `reports/` - Analysis reports and evaluation data (22 files)
  - `logs/` - Processing logs and execution records (20 files)
  - `temp_implementations/` - Prototype and test implementations (37 files)
  - `completed_phases/` - Finalized phase deliverables (13 files)
- **Archive Documentation**: Created `deprecated/README.md` with retention policy and review schedule

### Changed
- **Massive File Reduction**: Reduced project files from 383 to 24 essential files (94% reduction)
- **Preserved Core Structure**: Maintained all essential functionality in `core/`, `features/`, `tests/`, `docs/workflows/`
- **Improved Project Navigation**: Clear separation between active workflow files and archived materials
- **Updated Progress Tracking**: Enhanced `PROGRESS_TRACKER.md` with cleanup task completion (20/21 Phase 0 tasks, 95%)

### Technical Verification
- **Functionality Testing**: Verified `kana08_stable_batch.py` works perfectly after reorganization
  - Successfully processed 16/26 images (61.5% success rate)
  - Generated proper output files and reports
  - All import paths confirmed working
  - No performance degradation detected
- **Import Path Validation**: Confirmed all core dependencies accessible from deprecated location
- **Output Generation**: Verified extraction reports and logs generated correctly

### Removed (Archived to deprecated/)
- **Historical Backups**: Moved backup files and migration artifacts to `deprecated/backup/`
- **Experimental Scripts**: Archived temporary implementations and test scripts to appropriate subdirectories
- **Analysis Reports**: Moved evaluation reports and benchmark data to `deprecated/reports/`
- **Log Files**: Archived processing logs and execution records to `deprecated/logs/`
- **Completed Work**: Moved finalized phase deliverables to `deprecated/completed_phases/`

### Performance
- **File System Optimization**: 94% reduction in file clutter improves navigation and development efficiency
- **Preserved Functionality**: Zero impact on core processing capabilities
- **Maintained Quality**: All existing tests and quality checks continue to function
- **Development Workflow**: Streamlined project structure enhances workflow template usage

### Archive Policy
- **Retention Period**: 1-2 month review period for all archived files
- **Review Schedule**: Set for 2025-09-26 to determine permanent deletion or restoration
- **Safe Recovery**: All moved files preserved with original structure in deprecated/ subdirectories
- **Documentation**: Comprehensive tracking of all file movements and categorizations

### Documentation
- **Structure Analysis**: Complete file-by-file analysis in `folder_structure.md`
- **Archive Guide**: Detailed documentation of deprecated folder organization
- **Recovery Instructions**: Clear guidance for accessing archived files when needed
- **Progress Tracking**: Updated completion status and task documentation

### Development Process
- **Systematic Organization**: Methodical analysis and categorization of all project files
- **Safety First**: All files moved to archive rather than deleted for safe recovery
- **Workflow Focus**: Prioritized files needed for workflow templates and active development
- **Quality Assurance**: Verified functionality through comprehensive testing before finalization

## [v0.5.0] - 2025-07-21

### Added
- **[P1-007] Enhanced Aspect Ratio Judgment System**: Dynamic aspect ratio analysis system replacing fixed thresholds (1.2-2.5) with intelligent character style detection
  - **Character Style Detection**: Automatic detection of anime, chibi, realistic, and deformed character styles using edge density and color saturation analysis
  - **Adaptive Threshold Calculation**: Dynamic threshold computation based on character type and image characteristics
  - **Multi-Factor Quality Assessment**: Integrated aspect ratio with confidence, size, position, and edge quality metrics
  - **Confidence-Based Blending**: Automatic fallback to P1-003 system when P1-007 confidence is low, ensuring robust evaluation
  - **Protocol-Based Architecture**: Implemented using SOLID principles with dependency injection for extensibility

### Changed
- **YOLO Wrapper Integration**: Enhanced `yolo_wrapper.py` with 'aspect_ratio_enhanced' evaluation criteria
- **Backward Compatibility**: Maintained compatibility with all existing P1-001 through P1-006 systems
- **Quality Evaluation Pipeline**: Improved character extraction quality through intelligent aspect ratio assessment

### Fixed
- **Import Errors**: Resolved encoding issues and import problems in batch processing pipeline
- **Character Style Misclassification**: Eliminated fixed aspect ratio assumptions that failed on diverse character types

### Technical Improvements
- **Comprehensive Test Suite**: Added 42 new tests (28 unit + 14 integration tests) with 100% pass rate
- **Performance Validation**: Ensured no regression in processing speed (within 50x performance threshold)
- **Code Quality**: Maintained flake8, black, mypy, and isort compliance throughout implementation
- **Zero Dependencies**: Implementation uses only NumPy, OpenCV, and PyTorch - no additional dependencies required

### Architecture
- **Enhanced Aspect Ratio Analyzer**: Core `EnhancedAspectRatioAnalyzer` class with configurable threshold calculators
- **Style Detection System**: `HeuristicStyleDetector` for automatic character type classification
- **Confidence Blending**: Seamless integration between P1-007 and P1-003 evaluation systems
- **Extensible Design**: Protocol-based interfaces allow easy addition of new threshold calculation strategies

### Performance
- **Dynamic Adaptation**: Intelligent threshold adjustment based on character style improves extraction accuracy
- **Processing Efficiency**: Maintained target processing speeds while adding sophisticated analysis
- **Quality Metrics**: Enhanced quality assessment through multi-dimensional evaluation criteria

### Documentation
- **Implementation Guide**: Detailed documentation of P1-007 system architecture and usage
- **Integration Examples**: Comprehensive test cases demonstrating real-world usage scenarios
- **API Documentation**: Complete docstrings and type hints for all new functionality

### Thanks
- @miyashita337 for P1-007 Enhanced Aspect Ratio Judgment System implementation and comprehensive testing

## [v0.4.0] - 2025-07-21

### Added
- **[P1-006] Solid Fill Region Detection**: Comprehensive `SolidFillDetector` system for background/foreground separation with 388 lines of implementation
- **Claude GitHub Actions Integration**: Full automation workflow with Claude Code Action for issue-driven development
- **Comprehensive Test Suite**: 265-line test file for solid fill detection with 11 test cases (9 passing, 2 requiring fixes)
- **Manual Setup Documentation**: Complete Windows-specific setup guide for Claude GitHub Actions integration
- **Enhanced Manga Preprocessing**: Advanced effect line removal and multi-panel layout processing
- **Issue Templates**: Structured GitHub issue management for automated development workflow
- **Zenn Blog Documentation**: Complete technical blog post documenting Windows Claude Code setup challenges and solutions

### Changed
- **Phase A Animation Character Detection**: Implemented anime-specific YOLO model with GPT-4O recommended box expansion
- **Background/Foreground Separation**: Enhanced precision with solid fill region detection and color uniformity analysis
- **GitHub Actions Workflow**: Improved CI/CD configuration with proper trigger mechanisms and error handling
- **Development Automation**: Transitioned to Claude-assisted automated development with issue-driven implementation

### Fixed
- **GitHub Actions Linter Errors**: Resolved syntax and configuration issues in workflow files
- **Claude Integration Triggers**: Fixed GitHub Action implementation triggers for proper automation
- **Workflow Configuration**: Corrected invalid workflow syntax and removed non-functional configurations
- **Windows Claude Code Issues**: Documented and provided workaround for `/install-github-app` command failures

### Technical Improvements
- **Test Coverage**: 81.8% success rate (9/11 tests) for new solid fill detection system
- **Code Quality**: Maintained flake8, black, mypy, and isort compliance throughout automation integration
- **Integration Testing**: Successful merge of automated PRs with comprehensive review process
- **Documentation**: Enhanced project documentation with technical blog content and setup guides

### Performance
- **Solid Fill Detection**: Efficient region detection with configurable thresholds and adaptive processing
- **Automation Speed**: Rapid implementation cycles through Claude GitHub Actions integration
- **Phase 1 Progress**: Advanced to 27% completion (6/22 tasks) with P1-006 solid fill detection completed

### Security
- **GitHub Secrets Management**: Proper handling of Claude API tokens and authentication credentials
- **Manual Setup Process**: Secure alternative to automated installation for Windows environments

### Thanks
- @miyashita337 for core development and Claude GitHub Actions integration
- Claude Code Action community for automation framework development

## [v0.1.0] - 2025-07-20

### Added
- **CI/CD Infrastructure**: GitHub Actions workflow for automated linting and testing (`basic-ci.yml`)
- **Issue Templates**: Structured feature request templates for better project management
- **Phase 2 Character Extraction**: Enhanced manga preprocessing with effect line removal and panel splitting
- **Test Infrastructure**: Comprehensive pytest-based testing suite with 18 test cases
- **Development Tools**: Enhanced linter integration with virtual environment support
- **Project Documentation**: Improved GitHub Actions integration guides and usage documentation

### Changed
- **YOLO Threshold Optimization**: Lowered detection threshold from 0.02 to 0.005 for improved character detection
- **Module Architecture**: Updated import paths for Phase 0 refactoring compatibility
- **File Organization**: Reorganized scripts into `tools/` directory structure for better maintainability
- **Test Coverage**: Enhanced test reliability with fixed color detection and numerical type checking

### Fixed
- **Import Errors**: Resolved module import issues after Phase 0 refactoring 
- **Test Suite**: Fixed failing tests in character extraction pipeline
- **Dependency Issues**: Resolved dlib dependency with OpenCV-only fallback implementation
- **Code Quality**: Applied flake8, black, mypy, and isort standards throughout codebase

### Removed
- **Temporary Files**: Cleaned up development artifacts, backup files, and Jupyter checkpoints
- **Duplicate Scripts**: Consolidated evaluation scripts into organized directory structure

### Performance
- **Phase 2 Success Rate**: Improved from 0% to 37.5% success rate on difficult test cases
- **Processing Stability**: Enhanced memory management and error recovery in batch processing
- **Test Execution**: Optimized test performance with proper mocking and fixture management

### Thanks
- @miyashita337 for core development and GitHub Actions integration
- @harieshokunin for issue template contributions

## [v0.3.6] - 2025-07-18

### Added
- **run_batch_v035.py**: Enhanced batch processing script with comprehensive JSON results output and detailed performance metrics
- **batch_results_v035.json**: Detailed batch processing results documentation with individual image processing times and success rates
- **Performance Benchmarking**: Comprehensive processing statistics for kana09 dataset (58 images, 46.6% success rate)

### Fixed
- **Batch Processing Reliability**: Improved error handling and progress tracking for large-scale image processing
- **Results Documentation**: Enhanced logging with detailed JSON output for processing analysis
- **Performance Monitoring**: Better tracking of processing time per image and overall batch performance

### Performance Metrics
- **Processing Speed**: Average 17.3 seconds per image on kana09 dataset
- **Success Rate**: 46.6% (27/58 images) with detailed failure analysis
- **Memory Usage**: Stable memory consumption during batch processing
- **Error Classification**: Detailed categorization of processing failures for improvement targeting

### Technical Improvements
- **Batch Processing**: Enhanced batch script with real-time progress reporting and comprehensive result logging
- **Error Analysis**: Detailed failure categorization with specific error messages for each processing stage
- **Performance Tracking**: Individual image processing time recording for optimization analysis

## [v0.3.5] - 2025-07-18

### Added
- **PROJECT_SETTINGS.md**: Comprehensive project settings documentation with naming conventions, development guidelines, and mandatory checklists
- **run_batch_v034.py**: Improved batch processing script with proper naming conventions (replaced proprietary terms)
- **run_small_test_v034.py**: Small-scale test script with proper naming conventions (replaced proprietary terms)

### Changed
- **File Naming Conventions**: All Python files now use generic names without proprietary terms for better reusability and maintainability
- **Output Filename Strategy**: Batch processing scripts now preserve input filenames exactly for evaluation system compatibility
- **Function and Class Names**: Removed proprietary references from function names and comments for better code portability

### Fixed
- **Evaluation System Compatibility**: Output filenames now match input filenames exactly, ensuring proper evaluation system integration
- **File Management**: Eliminated numbered prefixes in output filenames that caused evaluation system mismatches
- **Code Maintainability**: Improved code clarity by removing dataset-specific references from core functionality

### Technical Improvements
- **Development Process**: Established mandatory pre-implementation checklists to prevent naming convention violations
- **Quality Assurance**: Added comprehensive guidelines for maintaining evaluation system compatibility
- **Code Portability**: Enhanced code reusability across different datasets and projects

### Documentation
- **Naming Standards**: Detailed documentation of Python file naming rules and output filename requirements
- **Development Guidelines**: Step-by-step checklists for developers to ensure compliance with project standards
- **Troubleshooting**: Common issues and solutions related to file naming and evaluation system integration

## [v0.3.4] - 2025-07-18

### Added
- **[P1-018] Smoothness Metrics System**: Advanced boundary line smoothness evaluation using curvature analysis, frequency domain analysis, local variation assessment, and multi-scale smoothness evaluation
- **[P1-020] Truncation Detection Algorithm**: Comprehensive limb and body part truncation detection with edge-based analysis, anatomical structure validation, and recovery suggestions
- **[P1-022] Contamination Quantifier**: Multi-modal contamination detection system with pixel-level purity analysis, color distance assessment, and texture-based contamination detection
- **[P1-016] Feedback Loop System**: Automatic feedback integration with performance trend analysis, adaptive parameter adjustment, and quality prediction improvement
- **[P1-010] Efficient Sampling Algorithm**: Multi-criteria sampling strategies including uncertainty-based, diversity-maximizing, stratified, and cost-effective sampling with active learning integration

### Changed
- **Phase 1 Quality Evaluation System**: Expanded from 5 to 10 evaluation modules with comprehensive quality assessment
- **Integrated Analysis**: All new systems provide A-F grading with detailed metrics and improvement recommendations
- **Continuous Learning**: Feedback loop system enables adaptive improvement based on evaluation data
- **Sampling Optimization**: Strategic learning data collection with efficiency maximization

### Technical Improvements
- **Complete Integration Testing**: 100% success rate (5/5 new systems) with comprehensive test suite
- **Advanced Analytics**: Multi-dimensional quality analysis with statistical validation
- **Fallback Implementations**: All systems work without optional dependencies (scipy, sklearn, matplotlib)
- **Performance Optimized**: Fast processing suitable for batch operations
- **Self-Evaluation**: Comprehensive quality assessment with A- grade (90/100 points)

### Quality Assessment Features
- **Boundary Smoothness Analysis**: Curvature variance, frequency analysis, gradient smoothness, and multi-scale evaluation
- **Truncation Prevention**: Edge-based detection, body completeness analysis, and anatomical structure validation
- **Contamination Quantification**: Color-based, texture-based, spatial, and edge contamination analysis
- **Feedback Integration**: Trend analysis, parameter adjustment, and prediction improvement
- **Sampling Efficiency**: Uncertainty-based, diversity-maximizing, and cost-effective sampling strategies

### Development Process
- **Systematic Implementation**: 5 additional Phase 1 tasks completed with individual testing and integration verification
- **Quality-First Approach**: Each system includes comprehensive test cases and quality validation
- **Documentation**: Detailed docstrings, usage examples, and self-evaluation report
- **Phase 1 Progress**: Advanced from 59% to 82% completion (18/22 tasks completed)

## [v0.3.3] - 2025-07-18

### Added
- **[P1-009] Learning Data Collection Planning**: Strategic learning data collection system with gap analysis and sampling strategy based on kana07 evaluation data (41 samples analyzed)
- **[P1-015] Evaluation Difference Analyzer**: Quantification system for automatic vs user evaluation differences with correlation analysis and improvement recommendations
- **[P1-017] Boundary Analysis Algorithm**: Boundary line quality quantification using curvature variance, Douglas-Peucker simplification, and smoothness metrics
- **[P1-019] Human Structure Recognition System**: Human body structure recognition for limb truncation prevention with body region estimation and risk assessment
- **[P1-021] Foreground Background Analyzer**: Background/foreground separation quality measurement using color clustering and texture analysis

### Changed
- **Phase 1 Quality Evaluation System**: Comprehensive quality improvement with 5 new evaluation modules
- **Quantitative Analysis**: All new systems provide numerical quality scores and grading (A-F scale)
- **Systematic Evaluation**: Evidence-based evaluation framework with detailed metrics and recommendations

### Technical Improvements
- **Integration Testing**: Complete Phase 1 integration test with 100% success rate (5/5 systems)
- **Fallback Implementations**: All systems work without optional dependencies (scipy, sklearn, matplotlib)
- **Performance Optimized**: Fast processing suitable for batch operations
- **Comprehensive Analysis**: Detailed analysis reports with actionable improvement recommendations

### Quality Assessment Features
- **Learning Data Gap Analysis**: Identifies underrepresented problems, missing regions, and rating imbalances
- **Correlation Analysis**: Pearson/Spearman correlation between automatic and user evaluations
- **Boundary Quality Metrics**: Curvature variance, angle variance, perimeter roughness, and simplification ratios
- **Human Structure Validation**: Body region detection, truncation risk assessment, and structure validity scoring
- **Separation Quality Scoring**: Color similarity, texture analysis, boundary clarity, and contamination risk assessment

### Development Process
- **Systematic Implementation**: 5 Phase 1 tasks completed with individual testing and integration verification
- **Quality-First Approach**: Each system includes comprehensive test cases and quality validation
- **Documentation**: Detailed docstrings and usage examples for all new evaluation systems

## [v0.3.2] - 2025-07-17

### Added
- **Region Priority System**: New region-based character selection system based on user evaluation feedback
- **User Evaluation Data Integration**: Added kana07_user_evaluation.jsonl with 41 image evaluations
- **Character Selection Optimization**: Enhanced character selection with position-based prioritization

### Changed
- **Improved Character Selection**: Better handling of multi-character manga panels with specific region preferences
- **Enhanced Quality Assessment**: Integration of user feedback into character extraction pipeline
- **Size Priority Refinement**: Improved size_priority method based on evaluation data showing higher success rates

### Fixed
- **Wrong Character Selection**: Reduced incorrect character selection in multi-character scenes
- **Extraction Failure Rate**: Improved extraction success rate from 58.5% baseline with targeted improvements

### Technical Improvements
- Added RegionPrioritySystem class for position-aware character selection
- Integrated user evaluation patterns into learned quality assessment
- Enhanced region-based scoring for better character targeting

## [v0.3.1] - 2025-07-17

### Fixed
- **Mosaic Detection Issue**: Disabled mosaic processing that was causing false positives in character extraction
- **Lower Body Extraction**: Fixed issue where screentone patterns were being misidentified as background, causing lower body extraction failures
- **Duplicate File Prefix**: Resolved issue where `preprocessed_manga_` prefix was being duplicated in temporary files

### Changed
- **Temporary File Management**: Moved preprocessing temporary files from input directory to `/tmp/` to prevent directory pollution
- **Batch Processing**: Enhanced batch processing with automatic cleanup of temporary files after completion

### Added
- **Auto-cleanup**: Automatic removal of temporary files in `/tmp/` after batch processing completion
- **File Prefix Protection**: Added logic to prevent duplicate prefix generation in preprocessing pipeline

## [v0.3.0] - 2025-07-17

### Added
- **[P1-001] Full Body Detection Algorithm Analysis**: Comprehensive analysis system for fullbody_priority method
- **[P1-002] Partial Extraction Detection System**: `PartialExtractionDetector` for detecting incomplete character extractions (face-only, limb truncation)
- **[P1-003] Enhanced Full Body Detection**: `EnhancedFullBodyDetector` with multi-metric evaluation (aspect ratio, body structure, edge distribution, semantic regions)
- **[P1-004] Advanced Screen Tone Detection**: `EnhancedScreentoneDetector` (857 lines) with FFT, Gabor, LBP, Wavelet, and Spatial feature extraction
- **[P1-005] Mosaic Boundary Processing**: `EnhancedMosaicBoundaryProcessor` with multi-scale and rotation-invariant detection
- **[P1-006] Solid Fill Area Processing**: `EnhancedSolidFillProcessor` with RGB/HSV/LAB color space analysis and adaptive clustering

### Changed
- **Quality Evaluation System**: Integrated all Phase 1 improvements into character extraction pipeline
- **Multi-Metric Assessment**: Enhanced evaluation with completeness scoring, boundary quality, and semantic analysis
- **Adaptive Processing**: Type-specific processing for character, background, and effect regions
- **Pattern Recognition**: Advanced pattern classification for various image elements (dots, lines, gradients, textures)

### Fixed
- **Boundary Accuracy**: Improved boundary detection from ±5 pixels to ±2 pixels
- **False Positive Reduction**: Reduced mosaic detection false positives from 30% to 10%
- **Color Uniformity**: Enhanced solid fill detection with circular uniformity computation for hue values
- **Edge Preservation**: Better edge-preserving filtering for solid fill boundaries

### Technical Improvements
- **Comprehensive Test Suite**: 151 test cases (126 unit + 25 integration tests) with 100% pass rate
- **Fallback Implementations**: Library-independent operation (sklearn, scipy, skimage optional)
- **Performance Optimization**: Processing speeds of 0.009s (100x100) to 0.115s (400x400)
- **Multi-Scale Analysis**: Support for different pattern sizes and orientations
- **Memory Efficiency**: Optimized clustering for large images with sampling techniques

### Performance Benchmarks
- **kana07 Dataset**: 3/3 images successful mosaic detection with 0.772-1.000 confidence
- **Detection Coverage**: Multiple pattern types (grid, pixelated, blur, rotated patterns)
- **Processing Speed**: Maintained target of 5000+ pixels/second across all implementations
- **Quality Metrics**: Average boundary quality scores >0.8 for detected regions

### Documentation
- **Analysis Documents**: Phase 1 analysis for fullbody, screentone, mosaic, and solid fill processing
- **API Documentation**: Comprehensive docstrings and usage examples
- **Integration Guides**: Test cases demonstrating real-world usage scenarios

## [v0.2.0] - 2025-07-17

### Added
- **Phase 0 Project Refactoring**: Complete modular architecture implementation
- **Modular Structure**: Organized codebase into `core/`, `features/`, `tests/`, `tools/` directories
- **CharacterExtractor Class**: New class-based interface for character extraction
- **Comprehensive Test Suite**: Unit and integration tests with fixtures
- **Development Documentation**: Extensive docs in `docs/` directory
  - `file-structure.md`: Project structure documentation
  - `phase0_problems.md`: Problem analysis and solutions
  - `phase0_completion_checklist.md`: Completion criteria and migration readiness
- **Progress Tracking**: `PROGRESS_TRACKER.md` for systematic development management
- **Auto-initialization System**: Option A fixes for seamless model loading
- **Enhanced CLAUDE.md**: Development process guidelines with implementation templates

### Changed
- **File Organization**: Moved Facebook SAM implementation to `core/segment_anything/`
- **Custom Features**: Relocated character extraction features to `features/extraction/`
- **Quality Assessment**: Moved evaluation utilities to `features/evaluation/`
- **Processing Pipeline**: Separated preprocessing/postprocessing to `features/processing/`
- **Test Structure**: Reorganized tests into unit/integration/fixtures hierarchy
- **Import Paths**: Updated all import statements for new modular structure

### Fixed
- **Dependency Issues**: Resolved CharacterExtractor import errors
- **Model Initialization**: Fixed SAM and YOLO model loading sequence
- **Path Resolution**: Corrected all relative path issues after refactoring
- **Test Compatibility**: Ensured all tests work with new structure

### Technical Improvements
- **Backup System**: Created `backup-20250716-2236/` for safe migration
- **Error Handling**: Enhanced error messages with correct paths
- **Code Quality**: Maintained flake8, black, mypy compliance
- **Documentation**: Comprehensive developer guides and troubleshooting

### Performance
- **Batch Processing**: Confirmed 32.6% success rate on kana06 dataset (28/86 images)
- **Initialization**: Streamlined model loading with singleton pattern preparation
- **Memory Management**: Improved resource handling for large datasets

### Development Process
- **Test-First Development**: Implemented comprehensive test framework
- **Quality Metrics**: Established success criteria and completion checklists
- **Problem Tracking**: Systematic issue identification and resolution
- **Migration Planning**: Phased approach to structural changes

## [v0.1.1] - Previous Release
- コードベースクリーンアップ - 局所的ファイル削除

## [v0.1.0] - Previous Release  
- 完璧な適応学習システム完成 - 100%成功率達成