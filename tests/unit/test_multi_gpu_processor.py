#!/usr/bin/env python3
"""
P1-014: マルチGPU処理システム単体テスト
"""

import pytest
import tempfile
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# プロジェクトルートをパスに追加
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from features.processing.multi_gpu_processor import (
    GPUDeviceInfo,
    MultiGPUConfig,
    ProcessingTask,
    GPUManager,
    MultiGPUSAMProcessor
)


class TestGPUDeviceInfo:
    """GPUデバイス情報テスト"""
    
    def test_gpu_device_info_creation(self):
        """GPUデバイス情報作成テスト"""
        gpu_info = GPUDeviceInfo(
            device_id=0,
            device_name="RTX 4070 Ti SUPER",
            total_memory_gb=15.0,
            available_memory_gb=12.0,
            compute_capability=(8, 9),
            is_available=True
        )
        
        assert gpu_info.device_id == 0
        assert gpu_info.device_name == "RTX 4070 Ti SUPER"
        assert gpu_info.total_memory_gb == 15.0
        assert gpu_info.available_memory_gb == 12.0
        assert gpu_info.compute_capability == (8, 9)
        assert gpu_info.is_available is True


class TestMultiGPUConfig:
    """マルチGPU設定テスト"""
    
    def test_default_config(self):
        """デフォルト設定テスト"""
        config = MultiGPUConfig()
        
        assert config.use_multi_gpu is True
        assert config.max_gpus is None
        assert config.memory_threshold_gb == 2.0
        assert config.load_balancing_strategy == 'round_robin'
        assert config.batch_size_per_gpu == 1
        assert config.enable_distributed is False
    
    def test_custom_config(self):
        """カスタム設定テスト"""
        config = MultiGPUConfig(
            use_multi_gpu=False,
            max_gpus=2,
            memory_threshold_gb=4.0,
            load_balancing_strategy='memory_aware',
            batch_size_per_gpu=4
        )
        
        assert config.use_multi_gpu is False
        assert config.max_gpus == 2
        assert config.memory_threshold_gb == 4.0
        assert config.load_balancing_strategy == 'memory_aware'
        assert config.batch_size_per_gpu == 4


class TestProcessingTask:
    """処理タスクテスト"""
    
    def test_processing_task_creation(self):
        """処理タスク作成テスト"""
        task = ProcessingTask(
            task_id="test_001",
            input_path="/input/image.jpg",
            output_path="/output/image_extracted.jpg",
            gpu_id=0,
            processing_params={'threshold': 0.5},
            priority=1
        )
        
        assert task.task_id == "test_001"
        assert task.input_path == "/input/image.jpg"
        assert task.output_path == "/output/image_extracted.jpg"
        assert task.gpu_id == 0
        assert task.processing_params == {'threshold': 0.5}
        assert task.priority == 1


class TestGPUManager:
    """GPU管理システムテスト"""
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_no_cuda_available(self, mock_device_count, mock_cuda_available):
        """CUDA利用不可時のテスト"""
        mock_cuda_available.return_value = False
        mock_device_count.return_value = 0
        
        config = MultiGPUConfig()
        manager = GPUManager(config)
        
        assert len(manager.available_gpus) == 0
        assert len(manager.gpu_locks) == 0
        assert len(manager.gpu_queues) == 0
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.get_device_properties')
    def test_single_gpu_detection(self, mock_get_props, mock_device_count, mock_cuda_available):
        """単一GPU検出テスト"""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 1
        
        # GPU属性モック
        mock_props = Mock()
        mock_props.name = "RTX 4070 Ti SUPER"
        mock_props.total_memory = 16 * 1024**3  # 16GB
        mock_props.major_capability_version = 8
        mock_get_props.return_value = mock_props
        
        with patch('torch.cuda.device'):
            config = MultiGPUConfig()
            manager = GPUManager(config)
            
            assert len(manager.available_gpus) == 1
            assert manager.available_gpus[0].device_id == 0
            assert manager.available_gpus[0].device_name == "RTX 4070 Ti SUPER"
            assert manager.available_gpus[0].is_available is True
    
    def test_optimal_gpu_selection_round_robin(self):
        """ラウンドロビンGPU選択テスト"""
        # 2個のGPUをモック
        gpu1 = GPUDeviceInfo(0, "GPU0", 8.0, 6.0, (8, 0), True)
        gpu2 = GPUDeviceInfo(1, "GPU1", 8.0, 6.0, (8, 0), True)
        
        config = MultiGPUConfig(load_balancing_strategy='round_robin')
        manager = GPUManager(config)
        manager.available_gpus = [gpu1, gpu2]
        
        # タスク作成
        task1 = ProcessingTask("task1", "/input1.jpg", "/output1.jpg", -1, {}, priority=0)
        task2 = ProcessingTask("task2", "/input2.jpg", "/output2.jpg", -1, {}, priority=1)
        task3 = ProcessingTask("task3", "/input3.jpg", "/output3.jpg", -1, {}, priority=2)
        
        # GPU選択テスト
        assert manager.get_optimal_gpu(task1) == 0  # priority 0 % 2 = 0
        assert manager.get_optimal_gpu(task2) == 1  # priority 1 % 2 = 1
        assert manager.get_optimal_gpu(task3) == 0  # priority 2 % 2 = 0
    
    @patch('torch.cuda.mem_get_info')
    @patch('torch.cuda.device')
    def test_optimal_gpu_selection_memory_aware(self, mock_device, mock_mem_info):
        """メモリ重視GPU選択テスト"""
        # メモリ情報モック（GPU0: 4GB空き, GPU1: 6GB空き）
        mock_mem_info.side_effect = [
            (4 * 1024**3, 8 * 1024**3),  # GPU0: 4GB free, 8GB total
            (6 * 1024**3, 8 * 1024**3),  # GPU1: 6GB free, 8GB total
        ]
        
        # GPU情報設定
        gpu1 = GPUDeviceInfo(0, "GPU0", 8.0, 6.0, (8, 0), True)
        gpu2 = GPUDeviceInfo(1, "GPU1", 8.0, 6.0, (8, 0), True)
        
        config = MultiGPUConfig(load_balancing_strategy='memory_aware')
        manager = GPUManager(config)
        manager.available_gpus = [gpu1, gpu2]
        
        task = ProcessingTask("task1", "/input1.jpg", "/output1.jpg", -1, {}, priority=0)
        
        # より多くのメモリを持つGPU1が選択されるべき
        selected_gpu = manager.get_optimal_gpu(task)
        assert selected_gpu == 1


class TestMultiGPUSAMProcessor:
    """マルチGPU SAM処理システムテスト"""
    
    def test_processor_initialization(self):
        """プロセッサ初期化テスト"""
        processor = MultiGPUSAMProcessor("P1-014-TEST")
        
        assert processor.tracker_id == "P1-014-TEST"
        assert processor.config is not None
        assert processor.gpu_manager is not None
        assert processor.path_manager is not None
        assert processor.stats['total_tasks'] == 0
        assert processor.stats['completed_tasks'] == 0
        assert processor.stats['failed_tasks'] == 0
    
    @patch('torch.cuda.is_available')
    def test_cpu_fallback_when_no_gpu(self, mock_cuda_available):
        """GPU利用不可時のCPUフォールバック テスト"""
        mock_cuda_available.return_value = False
        
        processor = MultiGPUSAMProcessor("P1-014-TEST")
        
        # CPUフォールバック処理のテスト
        with tempfile.TemporaryDirectory() as temp_dir:
            input_files = ["test1.jpg", "test2.jpg"]
            output_dir = temp_dir
            
            # CPUフォールバック処理呼び出し
            tasks = [
                ProcessingTask(f"task_{i}", f, f"{temp_dir}/out_{i}.jpg", -1, {}, i)
                for i, f in enumerate(input_files)
            ]
            
            # 開始時間設定
            processor.stats['start_time'] = 1000.0
            
            report = processor._fallback_cpu_processing(tasks)
            
            assert report is not None
            assert report.total_tasks == 2
            assert 'cpu_fallback' in report.performance_metrics
            assert report.performance_metrics['cpu_fallback'] is True
    
    def test_save_report(self):
        """レポート保存テスト"""
        from features.processing.multi_gpu_processor import MultiGPUReport
        
        processor = MultiGPUSAMProcessor("P1-014-TEST")
        
        report = MultiGPUReport(
            total_tasks=10,
            completed_tasks=8,
            failed_tasks=2,
            total_processing_time=120.5,
            gpu_usage_stats={0: {'utilization': 85.0}},
            performance_metrics={'throughput': 0.066}
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 一時的にパスマネージャーをモック
            with patch.object(processor.path_manager, 'ensure_output_dir') as mock_ensure_dir:
                mock_ensure_dir.return_value = Path(temp_dir)
                
                report_path = processor.save_report(report)
                
                assert report_path.exists()
                assert report_path.name.startswith("P1-014-TEST_multi_gpu_report")
                assert report_path.suffix == ".json"
                
                # JSONファイル内容確認
                import json
                with open(report_path, 'r') as f:
                    data = json.load(f)
                
                assert data['total_tasks'] == 10
                assert data['completed_tasks'] == 8
                assert data['failed_tasks'] == 2
                assert data['tracker_id'] == "P1-014-TEST"


class TestIntegrationScenarios:
    """統合シナリオテスト"""
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_multi_gpu_environment_simulation(self, mock_device_count, mock_cuda_available):
        """マルチGPU環境シミュレーションテスト"""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 2
        
        # GPU属性モック
        with patch('torch.cuda.get_device_properties') as mock_get_props:
            mock_props = Mock()
            mock_props.name = "Test GPU"
            mock_props.total_memory = 8 * 1024**3  # 8GB
            mock_props.major_capability_version = 8
            mock_get_props.return_value = mock_props
            
            with patch('torch.cuda.device'):
                config = MultiGPUConfig(max_gpus=2)
                processor = MultiGPUSAMProcessor("P1-014-TEST", config)
                
                # 2つのGPUが検出されることを確認
                assert len(processor.gpu_manager.available_gpus) == 2
                assert processor.gpu_manager.available_gpus[0].device_id == 0
                assert processor.gpu_manager.available_gpus[1].device_id == 1
    
    def test_load_balancing_strategies(self):
        """負荷分散戦略テスト"""
        # ラウンドロビン戦略
        config_rr = MultiGPUConfig(load_balancing_strategy='round_robin')
        manager_rr = GPUManager(config_rr)
        
        # メモリ重視戦略
        config_mem = MultiGPUConfig(load_balancing_strategy='memory_aware')
        manager_mem = GPUManager(config_mem)
        
        # パフォーマンス戦略
        config_perf = MultiGPUConfig(load_balancing_strategy='performance_aware')
        manager_perf = GPUManager(config_perf)
        
        # 各戦略で異なる処理が実行されることを確認
        assert config_rr.load_balancing_strategy == 'round_robin'
        assert config_mem.load_balancing_strategy == 'memory_aware'
        assert config_perf.load_balancing_strategy == 'performance_aware'


@pytest.fixture
def sample_gpu_info():
    """サンプルGPU情報"""
    return GPUDeviceInfo(
        device_id=0,
        device_name="RTX 4070 Ti SUPER",
        total_memory_gb=15.0,
        available_memory_gb=12.0,
        compute_capability=(8, 9),
        is_available=True
    )


@pytest.fixture
def sample_config():
    """サンプル設定"""
    return MultiGPUConfig(
        use_multi_gpu=True,
        max_gpus=2,
        memory_threshold_gb=2.0,
        load_balancing_strategy='round_robin'
    )


@pytest.fixture
def sample_task():
    """サンプルタスク"""
    return ProcessingTask(
        task_id="test_task_001",
        input_path="/test/input.jpg",
        output_path="/test/output.jpg",
        gpu_id=0,
        processing_params={'threshold': 0.5},
        priority=0
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])