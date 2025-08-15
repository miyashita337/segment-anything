"""
PH2-008: 復旧機能システムのユニットテスト
"""

import unittest
import tempfile
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from features.common.recovery_system import (
    RecoveryState, SystemStatus, ProcessMonitor, 
    FailureDetector, AutoRecoverySystem, RecoveryManager
)


class TestRecoveryState(unittest.TestCase):
    """RecoveryState データクラステスト"""
    
    def test_recovery_state_creation(self):
        """RecoveryState作成テスト"""
        state = RecoveryState(
            process_id="TEST-001",
            start_time=datetime.now(),
            current_phase="phase4"
        )
        
        self.assertEqual(state.process_id, "TEST-001")
        self.assertEqual(state.current_phase, "phase4")
        self.assertEqual(state.retry_count, 0)
        self.assertEqual(state.max_retries, 3)
        self.assertIsNotNone(state.error_history)
        self.assertEqual(len(state.error_history), 0)


class TestProcessMonitor(unittest.TestCase):
    """ProcessMonitor テスト"""
    
    def setUp(self):
        self.monitor = ProcessMonitor("python")
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('shutil.disk_usage')
    @patch('psutil.process_iter')
    def test_get_system_status(self, mock_process_iter, mock_disk_usage, 
                              mock_virtual_memory, mock_cpu_percent):
        """システム状態取得テスト"""
        # モック設定
        mock_cpu_percent.return_value = 45.5
        mock_virtual_memory.return_value = Mock(percent=60.2)
        mock_disk_usage.return_value = Mock(free=5 * 1024**3)  # 5GB
        mock_process = Mock()
        mock_process.name.return_value = "python"
        mock_process_iter.return_value = [mock_process]
        
        status = self.monitor.get_system_status()
        
        self.assertEqual(status.cpu_percent, 45.5)
        self.assertEqual(status.memory_percent, 60.2)
        self.assertEqual(status.disk_free_gb, 5.0)
        self.assertTrue(status.process_alive)
        self.assertIsInstance(status.timestamp, datetime)
    
    def test_detect_system_issues(self):
        """システム問題検出テスト"""
        # 正常状態
        normal_status = SystemStatus(
            cpu_percent=50.0,
            memory_percent=70.0,
            disk_free_gb=5.0,
            gpu_memory_mb=8192.0,
            process_alive=True,
            timestamp=datetime.now()
        )
        issues = self.monitor.detect_system_issues(normal_status)
        self.assertEqual(len(issues), 0)
        
        # 問題状態
        problem_status = SystemStatus(
            cpu_percent=98.0,
            memory_percent=95.0,
            disk_free_gb=0.5,
            gpu_memory_mb=8192.0,
            process_alive=False,
            timestamp=datetime.now()
        )
        issues = self.monitor.detect_system_issues(problem_status)
        self.assertEqual(len(issues), 4)
        self.assertIn("CPU使用率過高", issues[0])
        self.assertIn("メモリ使用率過高", issues[1])
        self.assertIn("ディスク容量不足", issues[2])
        self.assertIn("対象プロセスが停止中", issues[3])


class TestFailureDetector(unittest.TestCase):
    """FailureDetector テスト"""
    
    def setUp(self):
        self.detector = FailureDetector()
    
    def test_detect_failure_from_log(self):
        """ログから失敗検出テスト"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            f.write("Processing image...\n")
            f.write("CUDA out of memory: tried to allocate 2.00 GiB\n")
            f.write("Process failed\n")
            log_file = Path(f.name)
        
        try:
            has_failure, detected = self.detector.detect_failure_from_log(log_file)
            
            self.assertTrue(has_failure)
            self.assertIn("CUDA out of memory", detected)
        finally:
            log_file.unlink()
    
    def test_detect_output_failure(self):
        """出力失敗検出テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # 空ディレクトリ（失敗）
            has_failure, msg = self.detector.detect_output_failure(output_dir)
            self.assertTrue(has_failure)
            self.assertIn("抽出された画像が存在しません", msg)
            
            # 画像ファイル作成（成功）
            test_image = output_dir / "test.jpg"
            test_image.write_text("fake image data")
            
            has_failure, msg = self.detector.detect_output_failure(output_dir)
            self.assertFalse(has_failure)
            self.assertEqual(msg, "")


class TestAutoRecoverySystem(unittest.TestCase):
    """AutoRecoverySystem テスト"""
    
    def setUp(self):
        self.recovery_system = AutoRecoverySystem()
    
    def test_calculate_backoff_delay(self):
        """指数バックオフ遅延計算テスト"""
        # 1回目: 30秒
        delay1 = self.recovery_system.calculate_backoff_delay(0)
        self.assertEqual(delay1, 30)
        
        # 2回目: 60秒
        delay2 = self.recovery_system.calculate_backoff_delay(1)
        self.assertEqual(delay2, 60)
        
        # 3回目: 120秒
        delay3 = self.recovery_system.calculate_backoff_delay(2)
        self.assertEqual(delay3, 120)
        
        # 最大値制限確認
        delay_max = self.recovery_system.calculate_backoff_delay(10)
        self.assertEqual(delay_max, 300)  # max_delay
    
    @patch('time.sleep')
    def test_attempt_recovery(self, mock_sleep):
        """復旧試行テスト"""
        state = RecoveryState(
            process_id="TEST-001",
            start_time=datetime.now(),
            current_phase="phase4"
        )
        
        # ProcessMonitor をモック化
        with patch.object(self.recovery_system, 'process_monitor') as mock_monitor:
            mock_status = SystemStatus(
                cpu_percent=50.0,
                memory_percent=70.0,
                disk_free_gb=5.0,
                gpu_memory_mb=8192.0,
                process_alive=True,
                timestamp=datetime.now()
            )
            mock_monitor.get_system_status.return_value = mock_status
            mock_monitor.detect_system_issues.return_value = []
            
            result = self.recovery_system.attempt_recovery(state, "テストエラー")
            
            self.assertTrue(result)
            self.assertEqual(state.retry_count, 1)
            self.assertEqual(len(state.error_history), 1)
            mock_sleep.assert_called_once_with(60)  # 2回目の試行なので60秒
    
    def test_save_load_recovery_state(self):
        """復旧状態保存・読み込みテスト"""
        state = RecoveryState(
            process_id="TEST-001",
            start_time=datetime.now(),
            current_phase="phase4",
            retry_count=2
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            state_file = Path(f.name)
        
        try:
            # 保存テスト
            self.recovery_system.save_recovery_state(state, state_file)
            self.assertTrue(state_file.exists())
            
            # 読み込みテスト
            loaded_state = self.recovery_system.load_recovery_state(state_file)
            self.assertIsNotNone(loaded_state)
            self.assertEqual(loaded_state.process_id, "TEST-001")
            self.assertEqual(loaded_state.current_phase, "phase4")
            self.assertEqual(loaded_state.retry_count, 2)
        finally:
            if state_file.exists():
                state_file.unlink()


class TestRecoveryManager(unittest.TestCase):
    """RecoveryManager テスト"""
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.manager = RecoveryManager("TEST-001", self.temp_dir)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialize_recovery_session(self):
        """復旧セッション初期化テスト"""
        state = self.manager.initialize_recovery_session("phase4")
        
        self.assertIsNotNone(state)
        self.assertEqual(state.current_phase, "phase4")
        self.assertEqual(state.retry_count, 0)
        self.assertTrue(state.process_id.startswith("TEST-001_"))
        
        # 状態ファイル確認
        self.assertTrue(self.manager.state_file.exists())
    
    @patch('time.sleep')
    def test_handle_failure(self, mock_sleep):
        """失敗処理テスト"""
        state = self.manager.initialize_recovery_session("phase4")
        
        # ProcessMonitor をモック化
        with patch.object(self.manager.recovery_system, 'process_monitor') as mock_monitor:
            mock_status = SystemStatus(
                cpu_percent=50.0,
                memory_percent=70.0,
                disk_free_gb=5.0,
                gpu_memory_mb=8192.0,
                process_alive=True,
                timestamp=datetime.now()
            )
            mock_monitor.get_system_status.return_value = mock_status
            mock_monitor.detect_system_issues.return_value = []
            
            result = self.manager.handle_failure(state, "テスト失敗")
            
            self.assertTrue(result)
            self.assertEqual(state.retry_count, 1)
    
    def test_cleanup_recovery_session(self):
        """復旧セッション終了処理テスト"""
        # セッション開始
        self.manager.initialize_recovery_session("phase4")
        self.assertTrue(self.manager.state_file.exists())
        
        # セッション終了
        self.manager.cleanup_recovery_session()
        self.assertFalse(self.manager.state_file.exists())


def run_ph2_008_tests():
    """PH2-008テスト実行"""
    print("🧪 PH2-008: 復旧機能システム ユニットテスト実行")
    
    # テストスイート作成
    test_classes = [
        TestRecoveryState,
        TestProcessMonitor,
        TestFailureDetector,
        TestAutoRecoverySystem,
        TestRecoveryManager
    ]
    
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 結果サマリー
    total = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success = total - failures - errors
    
    print(f"\n📊 テスト結果サマリー:")
    print(f"   実行: {total}件")
    print(f"   成功: {success}件")
    print(f"   失敗: {failures}件")
    print(f"   エラー: {errors}件")
    
    if failures == 0 and errors == 0:
        print("✅ PH2-008復旧機能システム テストすべて通過")
        return True
    else:
        print("❌ PH2-008復旧機能システム テストに失敗があります")
        return False


if __name__ == "__main__":
    success = run_ph2_008_tests()
    exit(0 if success else 1)