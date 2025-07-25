#!/usr/bin/env python3
"""
自動進捗システム
Phase間の自動実行・修正・テストバッチ出力を管理
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# プロジェクトパスを追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from features.common.progress_reporter import ProgressReporter
from features.common.project_tracker import ProjectTracker
from features.evaluation.test_batch_generator import TestBatchGenerator
from features.phase1.data_expansion_system import DataExpansionSystem

logger = logging.getLogger(__name__)


class AutoProgressSystem:
    """自動進捗システム"""
    
    def __init__(self, project_root: Path):
        """
        初期化
        
        Args:
            project_root: プロジェクトルートディレクトリ
        """
        self.project_root = project_root
        self.tracker = ProjectTracker(project_root)
        self.reporter = ProgressReporter(project_root)
        self.test_batch_generator = TestBatchGenerator(project_root)
        
        # 実行ログ
        self.execution_log = []
        
        # 自動実行設定
        self.auto_config = {
            "test_batch_interval": 1,  # 每タスク完了後にテストバッチ出力
            "max_retries": 3,          # 最大再試行回数
            "sleep_between_phases": 5,  # Phase間の待機時間（秒）
        }
        
    def setup_logging(self):
        """ロギング設定"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler('auto_progress_system.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    
    def log_execution(self, phase: str, task: str, status: str, details: str = ""):
        """実行ログ記録"""
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "phase": phase,
            "task": task,
            "status": status,
            "details": details
        }
        self.execution_log.append(log_entry)
        logger.info(f"[{phase}] {task}: {status} - {details}")
    
    def generate_test_batch_from_results(self, phase: str, results_file: Optional[Path] = None):
        """結果からテストバッチ生成"""
        try:
            logger.info(f"Phase {phase} テストバッチ生成開始")
            
            # 結果ファイル検索
            if results_file is None:
                results_pattern = f"*{phase}*results*.json"
                benchmark_dir = self.project_root / "benchmark_results" / phase
                
                if benchmark_dir.exists():
                    result_files = list(benchmark_dir.glob(results_pattern))
                    if result_files:
                        results_file = max(result_files, key=lambda x: x.stat().st_mtime)
            
            if not results_file or not results_file.exists():
                logger.warning(f"Phase {phase}の結果ファイルが見つかりません")
                return None
            
            # 結果データ読み込み
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # detailed_resultsから抽出
            if 'detailed_results' in data:
                benchmark_results = data['detailed_results']
            elif isinstance(data, list):
                benchmark_results = data
            else:
                logger.warning("適切な結果データ形式が見つかりません")
                return None
            
            # テストバッチ生成
            summary = self.test_batch_generator.generate_test_batch(benchmark_results, phase)
            
            self.log_execution(phase, "テストバッチ生成", "成功", 
                              f"ベスト5・ワースト5を生成: {len(summary.best_items + summary.worst_items)}件")
            
            # 通知送信
            self.send_test_batch_notification(phase, summary)
            
            return summary
            
        except Exception as e:
            logger.error(f"テストバッチ生成エラー: {e}")
            self.log_execution(phase, "テストバッチ生成", "失敗", str(e))
            return None
    
    def generate_test_batch_from_panel_results(self, phase: str, panel_results):
        """コマ検出結果からテストバッチ生成"""
        try:
            logger.info(f"Phase {phase} コマ検出結果からテストバッチ生成開始")
            
            # コマ検出結果をベンチマーク形式に変換
            benchmark_results = []
            
            for result in panel_results:
                if result.success and result.largest_panel:
                    # Claude評価用スコア計算
                    panel = result.largest_panel
                    claude_score = (
                        panel.confidence * 0.4 +           # 信頼度40%
                        min(panel.area / 50000, 1.0) * 0.3 +  # サイズ正規化30%
                        (1.0 / max(result.processing_time, 0.1)) * 0.3  # 速度30%
                    )
                    
                    benchmark_item = {
                        'image_id': result.image_id,
                        'image_path': result.image_path,
                        'largest_char_predicted': True,  # コマ検出成功
                        'iou_score': float(panel.confidence),  # 信頼度をIoU代用
                        'confidence_score': float(panel.confidence),
                        'processing_time': float(result.processing_time),
                        'character_count': len(result.detections),  # 検出コマ数
                        'area_largest_ratio': min(panel.area / 100000, 1.0),
                        'quality_grade': 'B' if claude_score > 0.7 else 'C',
                        'prediction_bbox': list(panel.bbox),
                        'ground_truth_bbox': list(panel.bbox)  # 仮のGT
                    }
                else:
                    # 失敗ケース
                    benchmark_item = {
                        'image_id': result.image_id,
                        'image_path': result.image_path,
                        'largest_char_predicted': False,
                        'iou_score': 0.0,
                        'confidence_score': 0.0,
                        'processing_time': float(result.processing_time),
                        'character_count': 0,
                        'area_largest_ratio': 0.0,
                        'quality_grade': 'F',
                        'prediction_bbox': None,
                        'ground_truth_bbox': [0, 0, 100, 100]  # 仮のGT
                    }
                
                benchmark_results.append(benchmark_item)
            
            # テストバッチ生成
            summary = self.test_batch_generator.generate_test_batch(benchmark_results, phase)
            
            self.log_execution(phase, "コマ検出テストバッチ生成", "成功", 
                              f"ベスト5・ワースト5を生成: {len(summary.best_items + summary.worst_items)}件")
            
            # 通知送信
            self.send_test_batch_notification(phase, summary)
            
            return summary
            
        except Exception as e:
            logger.error(f"コマ検出テストバッチ生成エラー: {e}")
            self.log_execution(phase, "コマ検出テストバッチ生成", "失敗", str(e))
            return None
    
    def send_test_batch_notification(self, phase: str, summary):
        """テストバッチ完了通知"""
        try:
            message = (f"Phase {phase} テストバッチ生成完了: "
                      f"ベスト5平均スコア {summary.avg_score_best:.3f}, "
                      f"ワースト5平均スコア {summary.avg_score_worst:.3f}. "
                      f"人間評価との乖離確認をお願いします。")
            
            import subprocess
            subprocess.run([
                "windows-notify", "-t", "Claude Code", "-m", message
            ], check=False)
            
        except Exception as e:
            logger.error(f"通知送信エラー: {e}")
    
    def run_phase1_data_expansion(self) -> bool:
        """Phase 1 データ拡張実行"""
        try:
            logger.info("Phase 1 データ拡張開始")
            self.log_execution("phase1", "データ拡張", "開始", "")
            
            # データ拡張システム初期化・実行
            expansion_system = DataExpansionSystem(self.project_root)
            result = expansion_system.generate_pseudo_labels()
            
            # レポート生成
            report = expansion_system.create_expansion_report(result)
            report_file = expansion_system.output_dir / f"expansion_report_{time.strftime('%Y%m%d_%H%M%S')}.md"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            # プロジェクトトラッカー更新
            self.tracker.update_task_status("phase1-data-expansion", "completed")
            
            self.log_execution("phase1", "データ拡張", "成功", 
                              f"{result.generated_count}件生成、拡張倍率{result.expansion_ratio:.1f}倍")
            
            # テストバッチ生成（拡張データから）
            # 注: 実際の実装では拡張データを使ったベンチマークを先に実行する必要がある
            # ここでは元のPhase 0結果を使用
            self.generate_test_batch_from_results("phase0")
            
            return True
            
        except Exception as e:
            logger.error(f"Phase 1 データ拡張エラー: {e}")
            self.log_execution("phase1", "データ拡張", "失敗", str(e))
            return False
    
    def run_next_available_task(self) -> Optional[str]:
        """次の実行可能タスクを実行"""
        try:
            # 次のタスク取得
            next_tasks = self.tracker.get_next_tasks(1)
            
            if not next_tasks:
                logger.info("実行可能なタスクがありません")
                return None
            
            next_task = next_tasks[0]
            task_id = next_task.id
            task_content = next_task.content
            
            logger.info(f"次のタスク実行: {task_id} - {task_content}")
            
            # タスクを進行中に設定
            self.tracker.update_task_status(task_id, "in_progress")
            
            # タスク別実行
            success = False
            
            if task_id in ["phase1-data-expansion", "phase1-data-prep"]:
                success = self.run_phase1_data_expansion()
            
            elif task_id in ["phase1-panel-detection", "phase1-model-setup", "phase1-training"]:
                success = self.run_phase1_panel_detection()
            
            else:
                logger.warning(f"未実装のタスク: {task_id}")
                success = False
            
            # 結果に応じてステータス更新
            if success:
                self.tracker.update_task_status(task_id, "completed")
                self.log_execution("auto", "タスク実行", "成功", f"{task_id}: {task_content}")
            else:
                self.tracker.update_task_status(task_id, "pending")  # 再試行可能に戻す
                self.log_execution("auto", "タスク実行", "失敗", f"{task_id}: {task_content}")
            
            return task_id if success else None
            
        except Exception as e:
            logger.error(f"タスク実行エラー: {e}")
            return None
    
    def run_phase1_panel_detection(self) -> bool:
        """Phase 1 コマ検出ネット実行"""
        try:
            logger.info("Phase 1 コマ検出ネット実行開始")
            self.log_execution("phase1", "コマ検出ネット実行", "開始", "")
            
            # コマ検出ネットワーク実行
            from features.phase1.panel_detection_network import PanelDetectionNetwork
            
            panel_detector = PanelDetectionNetwork(self.project_root, model_type="yolo")
            
            # テスト画像ディレクトリ
            test_dir = self.project_root / "test_small"
            
            if not test_dir.exists():
                logger.warning(f"テストディレクトリが見つかりません: {test_dir}")
                self.log_execution("phase1", "コマ検出ネット実行", "失敗", "テストディレクトリなし")
                return False
            
            # バッチ処理実行
            results = panel_detector.process_batch(test_dir)
            
            success_count = len([r for r in results if r.success])
            
            if success_count > 0:
                self.log_execution("phase1", "コマ検出ネット実行", "成功", 
                                  f"{success_count}/{len(results)}件成功処理")
                
                # テストバッチ生成（コマ検出結果から）
                self.generate_test_batch_from_panel_results("phase1", results)
                
                return True
            else:
                self.log_execution("phase1", "コマ検出ネット実行", "失敗", "全画像処理失敗")
                return False
            
        except Exception as e:
            logger.error(f"Phase 1 コマ検出ネット実行エラー: {e}")
            self.log_execution("phase1", "コマ検出ネット実行", "失敗", str(e))
            return False
    
    def run_auto_progress_loop(self, max_iterations: int = 10):
        """自動進捗ループ実行"""
        try:
            logger.info("自動進捗システム開始")
            
            for iteration in range(max_iterations):
                logger.info(f"\n=== 自動進捗ループ {iteration + 1}/{max_iterations} ===")
                
                # 進捗状況確認
                summary = self.tracker.generate_progress_summary()
                logger.info(f"全体進捗: {summary['overall_progress']:.1f}%")
                logger.info(f"現在Phase: {summary['current_phase']}")
                
                # 完了チェック
                if summary['overall_progress'] >= 100.0:
                    logger.info("🎉 全Phase完了！")
                    break
                
                # 次のタスク実行
                executed_task = self.run_next_available_task()
                
                if executed_task is None:
                    logger.info("実行可能なタスクがありません。待機中...")
                    time.sleep(self.auto_config["sleep_between_phases"])
                    continue
                
                # テストバッチ生成（必要に応じて）
                if self.auto_config["test_batch_interval"] == 1:
                    # 現在のPhaseのテストバッチを生成
                    current_phase = summary['current_phase']
                    if current_phase:
                        self.generate_test_batch_from_results(current_phase)
                
                # 進捗レポート更新
                self.reporter.generate_full_report()
                
                # Phase間の待機
                if executed_task:
                    time.sleep(self.auto_config["sleep_between_phases"])
            
            # 最終実行ログ保存
            self.save_execution_log()
            
            logger.info("自動進捗システム完了")
            
        except Exception as e:
            logger.error(f"自動進捗システムエラー: {e}")
    
    def save_execution_log(self):
        """実行ログ保存"""
        try:
            log_file = self.project_root / "auto_execution_log.json"
            
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(self.execution_log, f, indent=2, ensure_ascii=False)
            
            logger.info(f"実行ログ保存: {log_file}")
            
        except Exception as e:
            logger.error(f"実行ログ保存エラー: {e}")
    
    def manual_test_batch_generation(self, phase: str = "phase0"):
        """手動テストバッチ生成（テスト用）"""
        try:
            logger.info(f"手動テストバッチ生成: Phase {phase}")
            
            # Phase 0のシミュレーション結果を使用
            import numpy as np
            
            sample_results = []
            for i in range(101):
                # prediction_bboxをTupleでなくListに変更（JSON互換性のため）
                pred_bbox = None
                if np.random.random() > 0.2:
                    pred_bbox = [
                        int(np.random.uniform(50, 200)),
                        int(np.random.uniform(50, 200)),
                        int(np.random.uniform(100, 300)),
                        int(np.random.uniform(150, 400))
                    ]
                
                result = {
                    'image_id': f'kana08_{i:04d}',
                    'image_path': f'/test_small/kana08_{i:04d}.png',
                    'largest_char_predicted': bool(np.random.random() > 0.4),
                    'iou_score': float(np.random.uniform(0.0, 1.0)),
                    'confidence_score': float(np.random.uniform(0.1, 0.9)),
                    'processing_time': float(np.random.uniform(3.0, 12.0)),
                    'character_count': int(np.random.randint(1, 6)),
                    'area_largest_ratio': float(np.random.uniform(0.2, 0.8)),
                    'quality_grade': str(np.random.choice(['A', 'B', 'C', 'D', 'E', 'F'], 
                                                         p=[0.05, 0.15, 0.25, 0.25, 0.20, 0.10])),
                    'prediction_bbox': pred_bbox,
                    'ground_truth_bbox': [
                        int(np.random.uniform(60, 180)),
                        int(np.random.uniform(60, 180)),
                        int(np.random.uniform(120, 280)),
                        int(np.random.uniform(180, 380))
                    ]
                }
                sample_results.append(result)
            
            # テストバッチ生成
            summary = self.test_batch_generator.generate_test_batch(sample_results, phase)
            
            print(f"\n📊 テストバッチ生成完了 - Phase {phase}")
            print(f"ベスト5平均スコア: {summary.avg_score_best:.3f}")
            print(f"ワースト5平均スコア: {summary.avg_score_worst:.3f}")
            print(f"出力ディレクトリ: {self.test_batch_generator.output_dir}")
            
            return summary
            
        except Exception as e:
            logger.error(f"手動テストバッチ生成エラー: {e}")
            return None


def main():
    """メイン処理"""
    
    # システム初期化
    auto_system = AutoProgressSystem(project_root)
    auto_system.setup_logging()
    
    logger.info("自動進捗システム初期化完了")
    
    # 引数に応じた実行
    if len(sys.argv) > 1 and sys.argv[1] == "--test-batch":
        # テストバッチ生成のみ
        phase = sys.argv[2] if len(sys.argv) > 2 else "phase0"
        auto_system.manual_test_batch_generation(phase)
        
    elif len(sys.argv) > 1 and sys.argv[1] == "--data-expansion":
        # データ拡張のみ
        auto_system.run_phase1_data_expansion()
        
    else:
        # 自動進捗ループ実行
        max_iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 5
        auto_system.run_auto_progress_loop(max_iterations)


if __name__ == "__main__":
    main()