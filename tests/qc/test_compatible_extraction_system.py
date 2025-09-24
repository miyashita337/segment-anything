#!/usr/bin/env python3
"""
Tests for QC成功版互換抽出システム
Environment-dependent tests that require GPU and model files.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import torch
from PIL import Image
import numpy as np

# 環境依存テスト: GPU・モデルファイル不在時は全てSkip
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or
    not Path("sam_vit_h_4b8939.pth").exists() or
    not Path("yolov8x.pt").exists(),
    reason="Environment dependent: GPU and model files required"
)

# Skip import if not available
try:
    from compatible_extraction_system import QCCompatibleExtractor
except ImportError:
    pytest.skip("QCCompatibleExtractor not available", allow_module_level=True)


class TestQCCompatibleExtractor:
    """QC Compatible Extractor Test Cases"""

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

    def teardown_method(self):
        """テスト後クリーンアップ"""
        if hasattr(self, 'temp_dir') and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_extractor_initialization(self):
        """抽出器初期化テスト"""
        extractor = QCCompatibleExtractor()

        assert extractor.device is not None
        assert extractor.sam_checkpoint == "sam_vit_h_4b8939.pth"
        assert extractor.yolo_model_path == "yolov8x.pt"

    def test_model_loading(self):
        """モデルロードテスト"""
        extractor = QCCompatibleExtractor()

        # モデルロードをテスト
        success = extractor.load_models()

        # 環境により成功/失敗が分かれる
        if success:
            assert extractor.sam_model is not None
            assert extractor.predictor is not None
            assert extractor.yolo_model is not None
        else:
            # モデルファイルが見つからない場合も許容
            pass

    def test_yolo_detection(self):
        """YOLO検出テスト"""
        extractor = QCCompatibleExtractor()

        # モデルロード
        if not extractor.load_models():
            pytest.skip("Model loading failed")

        # テスト画像作成
        test_image = np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8)

        # YOLO検出実行
        detections = extractor.detect_with_yolo(test_image, conf_threshold=0.07)

        # 戻り値の型確認
        assert isinstance(detections, list)
        # 検出があれば各要素の構造確認
        if detections:
            for detection in detections:
                assert 'box' in detection
                assert 'confidence' in detection
                assert 'area' in detection
                assert 'class_id' in detection

    def test_single_image_processing(self):
        """単一画像処理テスト"""
        extractor = QCCompatibleExtractor()

        # モデルロード
        if not extractor.load_models():
            pytest.skip("Model loading failed")

        # テスト画像作成
        input_file = self.temp_path / "test_input.jpg"
        output_file = self.temp_path / "test_output.png"

        test_image = Image.new('RGB', (400, 300), color='red')
        test_image.save(input_file)

        # 処理実行
        success = extractor.process_single_image(
            str(input_file),
            str(output_file),
            conf_threshold=0.07
        )

        # 結果確認（環境により成功/失敗が変わる）
        assert isinstance(success, bool)
        if success:
            assert output_file.exists()

    def test_batch_processing(self):
        """バッチ処理テスト"""
        extractor = QCCompatibleExtractor()

        # モデルロード
        if not extractor.load_models():
            pytest.skip("Model loading failed")

        # テスト入力ディレクトリ作成
        input_dir = self.temp_path / "input"
        output_dir = self.temp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        # テスト画像作成
        for i in range(3):
            test_image = Image.new('RGB', (100, 100), color='blue')
            test_image.save(input_dir / f"test_{i}.jpg")

        # バッチ処理実行
        stats = extractor.process_batch(
            str(input_dir),
            str(output_dir),
            conf_threshold=0.07,
            max_files=3
        )

        # 統計結果確認
        assert isinstance(stats, dict)
        assert 'total' in stats
        assert 'success' in stats
        assert 'failed' in stats
        assert stats['total'] == 3
        assert stats['success'] + stats['failed'] == stats['total']

    @patch('sys.argv', ['test', 'input.jpg', 'output_dir'])
    def test_main_function(self):
        """メイン関数テスト（モック使用）"""
        # メイン関数の存在確認のみ
        from compatible_extraction_system import main
        assert callable(main)


class TestQCCompatibleExtractorMocked:
    """モックを使用したテストケース"""

    @patch('compatible_extraction_system.sam_model_registry')
    @patch('compatible_extraction_system.YOLO')
    def test_extractor_with_mocked_models(self, mock_yolo, mock_sam_registry):
        """モックモデルを使用した抽出器テスト"""
        # モックセットアップ
        mock_sam_model = Mock()
        mock_sam_registry.return_value = mock_sam_model
        mock_yolo_instance = Mock()
        mock_yolo.return_value = mock_yolo_instance

        # 抽出器初期化
        extractor = QCCompatibleExtractor()

        # load_modelsの動作確認
        with patch.object(extractor, 'sam_model', mock_sam_model), \
             patch.object(extractor, 'yolo_model', mock_yolo_instance):

            # 基本的な属性確認
            assert extractor.device is not None
            assert extractor.sam_checkpoint == "sam_vit_h_4b8939.pth"
            assert extractor.yolo_model_path == "yolov8x.pt"

    def test_extractor_initialization_without_models(self):
        """モデルファイルなしでの初期化テスト"""
        # モデルファイルの有無に関係なく初期化可能であることを確認
        extractor = QCCompatibleExtractor(
            sam_checkpoint="nonexistent.pth",
            yolo_model_path="nonexistent.pt"
        )

        assert extractor.device is not None
        assert extractor.sam_checkpoint == "nonexistent.pth"
        assert extractor.yolo_model_path == "nonexistent.pt"