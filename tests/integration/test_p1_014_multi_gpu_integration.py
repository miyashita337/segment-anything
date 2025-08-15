#!/usr/bin/env python3
"""
P1-014: マルチGPU統合テスト
実際のファイル処理とシステム統合テスト
"""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, Mock
import numpy as np
from PIL import Image

# プロジェクトルートをパスに追加
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from features.processing.multi_gpu_sam_integration import MultiGPUSAMIntegration
from features.processing.multi_gpu_processor import MultiGPUConfig


class TestMultiGPUSAMIntegration:
    """マルチGPU SAM統合テスト"""
    
    @pytest.fixture
    def sample_images(self):
        """サンプル画像ファイル作成"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # テスト用画像ファイル作成
            image_files = []
            for i in range(3):
                # 512x512のランダム画像作成
                img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                
                img_path = temp_path / f"test_image_{i:03d}.jpg"
                img.save(img_path)
                image_files.append(str(img_path))
            
            yield image_files, str(temp_path)
    
    def test_integration_initialization(self):
        """統合システム初期化テスト"""
        integration = MultiGPUSAMIntegration("P1-014-TEST")
        
        assert integration.tracker_id == "P1-014-TEST"
        assert integration.config is not None
        assert integration.processor is not None
        assert integration.sam_yolo_script.exists()
    
    def test_performance_info_retrieval(self):
        """パフォーマンス情報取得テスト"""
        integration = MultiGPUSAMIntegration("P1-014-TEST")
        perf_info = integration.get_performance_info()
        
        assert 'multi_gpu_enabled' in perf_info
        assert 'available_gpus' in perf_info
        assert 'gpu_details' in perf_info
        assert 'load_balancing_strategy' in perf_info
        assert 'batch_size_per_gpu' in perf_info
        
        assert isinstance(perf_info['available_gpus'], int)
        assert isinstance(perf_info['gpu_details'], list)
    
    def test_file_splitting_for_gpus(self):
        """GPU用ファイル分割テスト"""
        integration = MultiGPUSAMIntegration("P1-014-TEST")
        
        files = [f"file_{i}.jpg" for i in range(10)]
        gpu_count = 3
        
        chunks = integration._split_files_for_gpus(files, gpu_count)
        
        assert len(chunks) == gpu_count
        
        # 全ファイルが分割されていることを確認
        total_files = sum(len(chunk) for chunk in chunks)
        assert total_files == len(files)
        
        # 各チャンクのサイズが均等に近いことを確認
        expected_size = len(files) // gpu_count
        for chunk in chunks:
            assert abs(len(chunk) - expected_size) <= 1
    
    @patch('subprocess.run')
    def test_fallback_to_standard_processing(self, mock_subprocess, sample_images):
        """標準処理フォールバック テスト"""
        image_files, temp_dir = sample_images
        
        # subprocess.runの成功をモック
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        integration = MultiGPUSAMIntegration("P1-014-TEST")
        
        with tempfile.TemporaryDirectory() as output_dir:
            success = integration._fallback_to_standard_processing(
                input_files=image_files,
                output_dir=output_dir,
                processing_params={'score_threshold': 0.07}
            )
            
            assert success is True
            mock_subprocess.assert_called_once()
            
            # コマンドライン引数確認
            call_args = mock_subprocess.call_args[0][0]
            assert "python3" in call_args
            assert "--mode" in call_args
            assert "reproduce-auto" in call_args
            assert "--score_threshold" in call_args
            assert "0.07" in call_args
    
    @patch('subprocess.run')
    def test_single_gpu_processing(self, mock_subprocess, sample_images):
        """単一GPU処理テスト"""
        image_files, temp_dir = sample_images
        
        # subprocess.runの成功をモック
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        config = MultiGPUConfig(use_multi_gpu=True, max_gpus=1)
        integration = MultiGPUSAMIntegration("P1-014-TEST", config)
        
        with tempfile.TemporaryDirectory() as output_dir:
            success = integration._single_gpu_processing(
                input_files=image_files,
                output_dir=output_dir,
                processing_params={'score_threshold': 0.05}
            )
            
            assert success is True
    
    def test_gpu_output_merging(self):
        """GPU出力統合テスト"""
        integration = MultiGPUSAMIntegration("P1-014-TEST")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # GPU別ディレクトリ作成
            gpu_dirs = []
            for gpu_id in [0, 1]:
                gpu_dir = temp_path / f"gpu_{gpu_id}"
                gpu_dir.mkdir()
                gpu_dirs.append(gpu_dir)
                
                # GPU別出力ファイル作成
                for i in range(2):
                    output_file = gpu_dir / f"output_{gpu_id}_{i}.jpg"
                    output_file.write_text(f"dummy content for GPU {gpu_id} file {i}")
            
            # 統合実行
            integration._merge_gpu_outputs(str(temp_path))
            
            # メインディレクトリにファイルが移動されていることを確認
            main_files = list(temp_path.glob("*.jpg"))
            assert len(main_files) == 4  # 各GPUから2ファイルずつ
            
            # GPU別ディレクトリが削除されていることを確認
            remaining_gpu_dirs = list(temp_path.glob("gpu_*"))
            assert len(remaining_gpu_dirs) == 0
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_multi_gpu_environment_detection(self, mock_device_count, mock_cuda_available):
        """マルチGPU環境検出テスト"""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 2
        
        with patch('torch.cuda.get_device_properties') as mock_get_props:
            mock_props = Mock()
            mock_props.name = "Test GPU"
            mock_props.total_memory = 8 * 1024**3
            mock_props.major_capability_version = 8
            mock_get_props.return_value = mock_props
            
            with patch('torch.cuda.device'):
                config = MultiGPUConfig(use_multi_gpu=True)
                integration = MultiGPUSAMIntegration("P1-014-TEST", config)
                
                perf_info = integration.get_performance_info()
                
                assert perf_info['multi_gpu_enabled'] is True
                assert perf_info['available_gpus'] == 2
                assert len(perf_info['gpu_details']) == 2
    
    @patch('subprocess.run')
    @patch('torch.cuda.is_available')
    def test_process_with_existing_pipeline_no_gpu(self, mock_cuda_available, mock_subprocess, sample_images):
        """GPU無環境での既存パイプライン処理テスト"""
        image_files, temp_dir = sample_images
        
        mock_cuda_available.return_value = False
        
        # subprocess成功をモック
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        integration = MultiGPUSAMIntegration("P1-014-TEST")
        
        with tempfile.TemporaryDirectory() as output_dir:
            success = integration.process_with_existing_pipeline(
                input_files=image_files,
                output_dir=output_dir
            )
            
            assert success is True
            mock_subprocess.assert_called_once()
    
    def test_configuration_variations(self):
        """設定バリエーションテスト"""
        # デフォルト設定
        integration1 = MultiGPUSAMIntegration("P1-014-TEST")
        assert integration1.config.use_multi_gpu is True
        assert integration1.config.load_balancing_strategy == 'round_robin'
        
        # カスタム設定
        custom_config = MultiGPUConfig(
            use_multi_gpu=False,
            max_gpus=1,
            load_balancing_strategy='memory_aware'
        )
        integration2 = MultiGPUSAMIntegration("P1-014-TEST", custom_config)
        assert integration2.config.use_multi_gpu is False
        assert integration2.config.max_gpus == 1
        assert integration2.config.load_balancing_strategy == 'memory_aware'


class TestEndToEndScenarios:
    """エンドツーエンドシナリオテスト"""
    
    @pytest.fixture
    def mock_sam_yolo_script(self):
        """SAM+YOLOスクリプトモック"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""#!/usr/bin/env python3
# Mock SAM+YOLO script for testing
import sys
import os
from pathlib import Path

if __name__ == "__main__":
    # 基本的な引数解析
    input_dir = None
    output_dir = None
    
    i = 0
    while i < len(sys.argv):
        if sys.argv[i] == '--input_dir' and i + 1 < len(sys.argv):
            input_dir = sys.argv[i + 1]
        elif sys.argv[i] == '--output_dir' and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
        i += 1
    
    if input_dir and output_dir:
        # 入力ファイルを出力ディレクトリにコピー（処理をシミュレート）
        import shutil
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for img_file in input_path.glob("*.jpg"):
            output_file = output_path / f"{img_file.stem}_extracted.jpg"
            shutil.copy2(img_file, output_file)
    
    sys.exit(0)
""")
            mock_script_path = Path(f.name)
        
        yield mock_script_path
        
        # クリーンアップ
        if mock_script_path.exists():
            mock_script_path.unlink()
    
    def test_complete_workflow_simulation(self, sample_images, mock_sam_yolo_script):
        """完全ワークフローシミュレーション テスト"""
        image_files, temp_input_dir = sample_images
        
        # モックスクリプトパスを設定
        integration = MultiGPUSAMIntegration("P1-014-TEST")
        integration.sam_yolo_script = mock_sam_yolo_script
        
        with tempfile.TemporaryDirectory() as output_dir:
            # 処理実行
            start_time = time.time()
            success = integration.process_with_existing_pipeline(
                input_files=image_files,
                output_dir=output_dir,
                processing_params={'score_threshold': 0.07}
            )
            end_time = time.time()
            
            # 結果確認
            assert success is True
            
            # 処理時間確認
            processing_time = end_time - start_time
            assert processing_time < 30.0  # 30秒以内で完了
            
            # 出力ファイル確認
            output_path = Path(output_dir)
            output_files = list(output_path.glob("*_extracted.jpg"))
            assert len(output_files) == len(image_files)
    
    @pytest.fixture
    def sample_images(self):
        """サンプル画像ファイル作成"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # テスト用画像ファイル作成
            image_files = []
            for i in range(5):  # 5個のテスト画像
                # 256x256の小さめの画像でテスト高速化
                img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                
                img_path = temp_path / f"test_image_{i:03d}.jpg"
                img.save(img_path)
                image_files.append(str(img_path))
            
            yield image_files, str(temp_path)


class TestPerformanceCharacteristics:
    """パフォーマンス特性テスト"""
    
    def test_memory_usage_estimation(self):
        """メモリ使用量推定テスト"""
        integration = MultiGPUSAMIntegration("P1-014-TEST")
        perf_info = integration.get_performance_info()
        
        # GPU情報の基本チェック
        assert 'available_gpus' in perf_info
        assert isinstance(perf_info['available_gpus'], int)
        assert perf_info['available_gpus'] >= 0
        
        # GPU詳細情報確認
        for gpu_detail in perf_info['gpu_details']:
            assert 'device_id' in gpu_detail
            assert 'device_name' in gpu_detail
            assert 'total_memory_gb' in gpu_detail
            assert 'is_available' in gpu_detail
    
    def test_load_balancing_effectiveness(self):
        """負荷分散効果テスト"""
        # 複数の負荷分散戦略をテスト
        strategies = ['round_robin', 'memory_aware', 'performance_aware']
        
        for strategy in strategies:
            config = MultiGPUConfig(load_balancing_strategy=strategy)
            integration = MultiGPUSAMIntegration("P1-014-TEST", config)
            
            assert integration.config.load_balancing_strategy == strategy
            
            # パフォーマンス情報取得
            perf_info = integration.get_performance_info()
            assert perf_info['load_balancing_strategy'] == strategy


@pytest.fixture
def temp_workspace():
    """一時ワークスペース"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])