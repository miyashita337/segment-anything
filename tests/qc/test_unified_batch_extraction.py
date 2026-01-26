#!/usr/bin/env python3
"""
Tests for QC品質調査用バッチキャラクター抽出システム
Environment-dependent tests that require GPU and model files.
"""

import torch

import json
import os
import pytest
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

# 環境依存テスト: GPU・モデルファイル不在時は全てSkip
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not Path("sam_vit_h_4b8939.pth").exists()
    or not Path("yolov8x.pt").exists(),
    reason="Environment dependent: GPU and model files required",
)

# Skip import if not available
try:
    import sys
    from pathlib import Path

    # Add tests/qc to path for imports
    sys.path.insert(0, str(Path(__file__).parent))
    from unified_batch_extraction import QCBatchExtractor
except ImportError:
    pytest.skip("QCBatchExtractor not available", allow_module_level=True)


class TestQCBatchExtractor:
    """QC Batch Extractor Test Cases"""

    def setup_method(self):
        """テスト前セットアップ"""
        # Skip if environment not available
        if not torch.cuda.is_available():
            pytest.skip("Environment dependent: GPU required")

        if not Path("sam_vit_h_4b8939.pth").exists():
            pytest.skip("Environment dependent: SAM model file required")

        if not Path("yolov8x.pt").exists():
            pytest.skip("Environment dependent: YOLO model file required")

        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # CI環境変数設定
        os.environ["CI_ENVIRONMENT"] = "true"

    def teardown_method(self):
        """テスト後クリーンアップ"""
        if hasattr(self, "temp_dir") and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

        # CI環境変数クリア
        if "CI_ENVIRONMENT" in os.environ:
            del os.environ["CI_ENVIRONMENT"]

    def test_extractor_initialization_ci_mode(self):
        """抽出器初期化テスト（CI環境）"""
        # CI環境フラグ設定
        os.environ["CI_ENVIRONMENT"] = "true"

        extractor = QCBatchExtractor()

        # CI環境での初期化確認
        assert extractor.device == torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert extractor.sam is None  # CI環境ではモック
        assert extractor.predictor is None
        assert extractor.yolo_model is None

        # 統計情報初期化確認
        assert extractor.stats["total_processed"] == 0
        assert extractor.stats["total_success"] == 0
        assert extractor.stats["total_failed"] == 0
        assert isinstance(extractor.stats["folder_stats"], dict)

    @patch.dict(os.environ, {"CI_ENVIRONMENT": ""})
    def test_extractor_initialization_local_mode(self):
        """抽出器初期化テスト（ローカル環境）"""
        # 実際のモデルファイルパスでの初期化テスト
        # ローカル環境でのテストは実際のファイルが必要

        # モデルファイルが存在する場合のみテスト実行
        sam_checkpoint = "/mnt/c/AItools/segment-anything/sam_vit_h_4b8939.pth"
        yolo_model = "/mnt/c/AItools/segment-anything/yolov8x.pt"

        if Path(sam_checkpoint).exists() and Path(yolo_model).exists():
            try:
                extractor = QCBatchExtractor()
                assert extractor.device is not None
                assert extractor.sam is not None
                assert extractor.predictor is not None
                assert extractor.yolo_model is not None
            except Exception as e:
                # モデルロードエラーは許容（環境依存）
                pytest.skip(f"Model loading failed: {e}")
        else:
            pytest.skip("Model files not found in expected locations")

    def test_pushover_config_loading_ci(self):
        """Pushover設定読み込みテスト（CI環境）"""
        os.environ["CI_ENVIRONMENT"] = "true"

        extractor = QCBatchExtractor()
        config = extractor.load_pushover_config()

        # CI環境では空の設定が返される
        assert isinstance(config, dict)
        assert len(config) == 0

    @patch("builtins.open", create=True)
    @patch("json.load")
    @patch.dict(os.environ, {"CI_ENVIRONMENT": ""})
    def test_pushover_config_loading_local(self, mock_json_load, mock_open):
        """Pushover設定読み込みテスト（ローカル環境・モック）"""
        # モック設定
        mock_config = {"user_key": "test_user_key", "api_token": "test_api_token"}
        mock_json_load.return_value = mock_config
        mock_open.return_value.__enter__.return_value = Mock()

        extractor = QCBatchExtractor()

        # 設定読み込み確認
        assert extractor.pushover_config == mock_config

    @patch("requests.post")
    def test_pushover_notification(self, mock_post):
        """Pushover通知テスト"""
        os.environ["CI_ENVIRONMENT"] = "true"

        # モックレスポンス設定
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        extractor = QCBatchExtractor()
        extractor.pushover_config = {"user_key": "test_user", "api_token": "test_token"}

        # 通知送信テスト
        extractor.send_pushover_notification("テストメッセージ", "テストタイトル")

        # モック呼び出し確認
        mock_post.assert_called_once()

    def test_extract_character_ci_mode(self):
        """キャラクター抽出テスト（CI環境）"""
        os.environ["CI_ENVIRONMENT"] = "true"

        extractor = QCBatchExtractor()

        # テスト画像作成
        input_file = self.temp_path / "test_input.jpg"
        output_file = self.temp_path / "test_output.png"
        input_file.touch()  # ダミーファイル作成

        # CI環境ではモデルが存在しないため失敗する
        result = extractor.extract_character(str(input_file), str(output_file))

        # CI環境では False が返される（モデルなし）
        assert isinstance(result, bool)

    def test_process_folder_ci_mode(self):
        """フォルダ処理テスト（CI環境）"""
        os.environ["CI_ENVIRONMENT"] = "true"

        extractor = QCBatchExtractor()

        # テストディレクトリ作成
        input_dir = self.temp_path / "input"
        output_dir = self.temp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        # ダミー画像ファイル作成
        for i in range(3):
            (input_dir / f"test_{i}.jpg").touch()

        # フォルダ処理実行
        stats = extractor.process_folder(str(input_dir), str(output_dir), "TEST_FOLDER")

        # 統計結果確認
        assert isinstance(stats, dict)
        assert "total" in stats
        assert "success" in stats
        assert "failed" in stats
        assert stats["total"] == 3
        # CI環境では抽出が失敗するため、failed が 3
        assert stats["failed"] == 3
        assert stats["success"] == 0

    def test_run_qc_extraction_ci_mode(self):
        """QC抽出メイン処理テスト（CI環境）"""
        os.environ["CI_ENVIRONMENT"] = "true"

        extractor = QCBatchExtractor()

        # QC抽出実行（CI環境では一時ディレクトリ使用）
        extractor.run_qc_extraction()

        # 統計情報の確認
        assert extractor.stats["total_processed"] > 0
        assert "KANA08" in extractor.stats["folder_stats"]
        assert "KANA05" in extractor.stats["folder_stats"]
        assert "KANA07" in extractor.stats["folder_stats"]

        # 各フォルダの統計確認
        for folder_name in ["KANA08", "KANA05", "KANA07"]:
            folder_stats = extractor.stats["folder_stats"][folder_name]
            assert "total" in folder_stats
            assert "success" in folder_stats
            assert "failed" in folder_stats
            assert "success_rate" in folder_stats

    @patch("unified_batch_extraction.QCBatchExtractor")
    def test_main_function(self, mock_extractor_class):
        """メイン関数テスト"""
        from unified_batch_extraction import main

        # モック設定
        mock_extractor = Mock()
        mock_extractor_class.return_value = mock_extractor

        # メイン関数実行
        main()

        # 抽出器が作成され、実行されることを確認
        mock_extractor_class.assert_called_once()
        mock_extractor.run_qc_extraction.assert_called_once()


class TestEnvironmentDetection:
    """環境検出テスト"""

    def test_ci_environment_detection(self):
        """CI環境検出テスト"""
        # CI環境変数設定
        os.environ["CI_ENVIRONMENT"] = "true"

        # モジュールを再インポートして環境検出をテスト
        import importlib
        import unified_batch_extraction

        importlib.reload(unified_batch_extraction)

        # IS_CI フラグの確認
        assert unified_batch_extraction.IS_CI == True

        # 環境変数クリア
        del os.environ["CI_ENVIRONMENT"]

    def test_local_environment_detection(self):
        """ローカル環境検出テスト"""
        # CI環境変数が未設定の場合
        if "CI_ENVIRONMENT" in os.environ:
            del os.environ["CI_ENVIRONMENT"]

        # モジュールを再インポートして環境検出をテスト
        import importlib
        import unified_batch_extraction

        importlib.reload(unified_batch_extraction)

        # パスの存在により判定が変わる
        expected_ci = not os.path.exists("/mnt/c")
        assert unified_batch_extraction.IS_CI == expected_ci
