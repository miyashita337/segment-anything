#!/usr/bin/env python3
"""
タスク統合レイヤー
QUAL-044: pytest、extract_character.py等の具体的なタスク統合

長時間処理タスクを標準化されたインターフェースで実行可能にする
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import logging

from .long_task_manager import LongTaskQueue
from .subagent_monitor import SubAgentIntegration

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TaskIntegration:
    """タスク統合クラス"""
    
    def __init__(self, tracker_id: str = "QUAL-044"):
        """
        初期化
        
        Args:
            tracker_id: 現在のトラッカーID
        """
        self.tracker_id = tracker_id
        self.workspace_base = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace")
        self.workspace = self.workspace_base / tracker_id
        
        # Queue と SubAgent統合初期化
        self.queue = LongTaskQueue(str(self.workspace))
        self.integration = SubAgentIntegration()
        
        # 現在のコンテキスト設定
        self.integration.set_context({
            'tracker_id': tracker_id,
            'workspace': str(self.workspace),
            'task_type': 'long_running_queue'
        })
        
        logger.info(f"TaskIntegration initialized for {tracker_id}")
    
    def execute_pytest(self, 
                      test_path: str = "tests/",
                      options: Optional[List[str]] = None,
                      coverage: bool = True) -> str:
        """
        pytest実行タスク登録
        
        Args:
            test_path: テストパス
            options: pytest追加オプション
            coverage: カバレッジ計測有無
            
        Returns:
            task_id: タスクID
        """
        logger.info(f"Registering pytest task for {test_path}")
        
        # pytestコマンド構築
        cmd_parts = ["python", "-m", "pytest", test_path, "-v"]
        
        if coverage:
            cmd_parts.extend(["--cov=.", "--cov-report=html", "--cov-report=term"])
        
        if options:
            cmd_parts.extend(options)
        
        # 出力ファイル指定
        output_file = self.workspace / "logs" / f"pytest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        cmd_parts.extend([f"--html={output_file}.html", "--self-contained-html"])
        
        command = " ".join(cmd_parts)
        
        # タスクをキューに追加
        task_id = self.queue.enqueue_task(command, "pytest")
        
        logger.info(f"pytest task enqueued: {task_id}")
        logger.info(f"Command: {command}")
        
        return task_id
    
    def execute_extract_character(self,
                                 input_dir: str,
                                 output_dir: Optional[str] = None,
                                 batch: bool = True,
                                 max_files: Optional[int] = None,
                                 quality_method: str = "balanced") -> str:
        """
        extract_character.py実行タスク登録
        
        Args:
            input_dir: 入力ディレクトリ
            output_dir: 出力ディレクトリ
            batch: バッチ処理フラグ
            max_files: 最大処理ファイル数
            quality_method: 品質評価手法
            
        Returns:
            task_id: タスクID
        """
        logger.info(f"Registering extract_character task for {input_dir}")
        
        # 出力ディレクトリデフォルト設定
        if output_dir is None:
            output_dir = str(self.workspace / "extraction")
        
        # コマンド構築
        cmd_parts = [
            "python",
            "features/extraction/commands/extract_character.py",
            input_dir,
            "-o", output_dir
        ]
        
        if batch:
            cmd_parts.append("--batch")
        
        if max_files:
            cmd_parts.extend(["--max-files", str(max_files)])
        
        cmd_parts.extend(["--quality-method", quality_method])
        cmd_parts.append("--verbose")
        
        command = " ".join(cmd_parts)
        
        # タスクをキューに追加
        task_id = self.queue.enqueue_task(command, "extract_character")
        
        logger.info(f"extract_character task enqueued: {task_id}")
        logger.info(f"Command: {command}")
        
        return task_id
    
    def execute_custom_command(self,
                              command: str,
                              task_type: str = "custom") -> str:
        """
        カスタムコマンド実行タスク登録
        
        Args:
            command: 実行コマンド
            task_type: タスクタイプ
            
        Returns:
            task_id: タスクID
        """
        logger.info(f"Registering custom task: {task_type}")
        
        # タスクをキューに追加
        task_id = self.queue.enqueue_task(command, task_type)
        
        logger.info(f"Custom task enqueued: {task_id}")
        logger.info(f"Command: {command}")
        
        return task_id
    
    def start_monitoring(self, task_id: str) -> Dict[str, Any]:
        """
        タスク監視開始（同一セッション内）
        
        Args:
            task_id: 監視対象タスクID
            
        Returns:
            監視結果
        """
        logger.info(f"Starting monitoring for task: {task_id}")
        
        # SubAgent統合による監視開始
        result = self.integration.monitor_long_task(
            task_id=task_id,
            task_command=f"Queue task {task_id}"
        )
        
        return result
    
    def start_queue_processing(self) -> None:
        """キュー処理開始"""
        logger.info("Starting queue background processing")
        self.queue.start_background_execution()
    
    def stop_queue_processing(self) -> None:
        """キュー処理停止"""
        logger.info("Stopping queue processing")
        self.queue.stop_execution()
    
    def get_queue_status(self) -> Dict[str, Any]:
        """キュー状態取得"""
        return self.queue.get_queue_status()
    
    def parse_pytest_results(self, output_file: str) -> Dict[str, Any]:
        """
        pytest結果解析
        
        Args:
            output_file: 出力ファイルパス
            
        Returns:
            解析結果
        """
        results = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'errors': 0,
            'duration': 0.0,
            'failures': []
        }
        
        try:
            with open(output_file, 'r') as f:
                content = f.read()
            
            # 結果サマリー抽出
            summary_match = re.search(
                r'(\d+) passed(?:, (\d+) failed)?(?:, (\d+) skipped)?(?:, (\d+) error)? in ([\d.]+)s',
                content
            )
            
            if summary_match:
                results['passed'] = int(summary_match.group(1) or 0)
                results['failed'] = int(summary_match.group(2) or 0)
                results['skipped'] = int(summary_match.group(3) or 0)
                results['errors'] = int(summary_match.group(4) or 0)
                results['duration'] = float(summary_match.group(5) or 0.0)
                results['total_tests'] = sum([
                    results['passed'],
                    results['failed'],
                    results['skipped'],
                    results['errors']
                ])
            
            # 失敗テスト詳細抽出
            failure_matches = re.findall(
                r'FAILED ([\w/\.]+::\w+)(?:\[[\w-]+\])? - (.+)',
                content
            )
            
            for test_name, error_msg in failure_matches:
                results['failures'].append({
                    'test': test_name,
                    'error': error_msg
                })
            
        except Exception as e:
            logger.error(f"Failed to parse pytest results: {e}")
        
        return results
    
    def parse_extraction_results(self, output_dir: str) -> Dict[str, Any]:
        """
        extract_character結果解析
        
        Args:
            output_dir: 出力ディレクトリ
            
        Returns:
            解析結果
        """
        output_path = Path(output_dir)
        result_file = output_path / "extraction_result.json"
        
        if not result_file.exists():
            return {
                'total_images': 0,
                'successful': 0,
                'failed': 0,
                'success_rate': 0.0,
                'error': 'Result file not found'
            }
        
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            total = data.get('total_images', 0)
            successful = data.get('successful_extractions', 0)
            
            return {
                'total_images': total,
                'successful': successful,
                'failed': total - successful,
                'success_rate': (successful / total * 100) if total > 0 else 0,
                'quality_scores': data.get('quality_scores', {}),
                'processing_time': data.get('total_processing_time', 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to parse extraction results: {e}")
            return {
                'total_images': 0,
                'successful': 0,
                'failed': 0,
                'success_rate': 0.0,
                'error': str(e)
            }
    
    def generate_summary_report(self, task_id: str, task_type: str) -> str:
        """
        サマリーレポート生成
        
        Args:
            task_id: タスクID
            task_type: タスクタイプ
            
        Returns:
            レポート内容
        """
        report = f"""# タスク実行レポート

## 基本情報
- **タスクID**: {task_id}
- **タスクタイプ**: {task_type}
- **トラッカーID**: {self.tracker_id}
- **実行時刻**: {datetime.now().isoformat()}

## 実行結果
"""
        
        if task_type == "pytest":
            # pytest結果追加
            log_file = self.workspace / "logs" / f"{task_id}_output.log"
            if log_file.exists():
                results = self.parse_pytest_results(str(log_file))
                report += f"""
### テスト結果
- 総テスト数: {results['total_tests']}
- 成功: {results['passed']}
- 失敗: {results['failed']}
- スキップ: {results['skipped']}
- エラー: {results['errors']}
- 実行時間: {results['duration']:.2f}秒

### 失敗テスト詳細
"""
                for failure in results['failures']:
                    report += f"- `{failure['test']}`: {failure['error']}\n"
        
        elif task_type == "extract_character":
            # 抽出結果追加
            output_dir = self.workspace / "extraction"
            results = self.parse_extraction_results(str(output_dir))
            report += f"""
### 抽出結果
- 総画像数: {results['total_images']}
- 成功: {results['successful']}
- 失敗: {results['failed']}
- 成功率: {results['success_rate']:.1f}%
- 処理時間: {results.get('processing_time', 0):.2f}秒
"""
        
        report += f"""
## 次のアクション
- 結果の詳細確認
- 失敗ケースの分析
- 必要に応じて再実行

---
*Generated by QUAL-044 Task Integration System*
"""
        
        return report


class TaskOrchestrator:
    """タスクオーケストレーター（高レベルAPI）"""
    
    def __init__(self, tracker_id: str = "QUAL-044"):
        """初期化"""
        self.integration = TaskIntegration(tracker_id)
        self.active_tasks: Dict[str, str] = {}  # task_id -> task_type
    
    def run_pytest_with_monitoring(self,
                                  test_path: str = "tests/",
                                  **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        pytest実行と監視
        
        Args:
            test_path: テストパス
            **kwargs: その他のpytestオプション
            
        Returns:
            (task_id, 監視結果)
        """
        # タスク登録
        task_id = self.integration.execute_pytest(test_path, **kwargs)
        self.active_tasks[task_id] = "pytest"
        
        # キュー処理開始
        self.integration.start_queue_processing()
        
        # 監視開始（同一セッション内）
        result = self.integration.start_monitoring(task_id)
        
        # レポート生成
        if result.get('final_status') == 'completed':
            report = self.integration.generate_summary_report(task_id, "pytest")
            result['summary_report'] = report
        
        return task_id, result
    
    def run_extraction_with_monitoring(self,
                                      input_dir: str,
                                      **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        キャラクター抽出実行と監視
        
        Args:
            input_dir: 入力ディレクトリ
            **kwargs: その他の抽出オプション
            
        Returns:
            (task_id, 監視結果)
        """
        # タスク登録
        task_id = self.integration.execute_extract_character(input_dir, **kwargs)
        self.active_tasks[task_id] = "extract_character"
        
        # キュー処理開始
        self.integration.start_queue_processing()
        
        # 監視開始（同一セッション内）
        result = self.integration.start_monitoring(task_id)
        
        # レポート生成
        if result.get('final_status') == 'completed':
            report = self.integration.generate_summary_report(task_id, "extract_character")
            result['summary_report'] = report
        
        return task_id, result
    
    def cleanup(self) -> None:
        """クリーンアップ"""
        self.integration.stop_queue_processing()
        self.active_tasks.clear()


def demonstrate_task_integration():
    """タスク統合デモンストレーション"""
    print("🎯 タスク統合レイヤーデモンストレーション")
    print("=" * 50)
    
    # オーケストレーター初期化
    orchestrator = TaskOrchestrator("QUAL-044")
    
    print("\n1️⃣ pytest実行例")
    print("   テストパス: tests/unit/")
    print("   カバレッジ: 有効")
    print("   監視: SubAgentによる同一セッション監視")
    
    # デモ結果（実際の実行はコメントアウト）
    # task_id, result = orchestrator.run_pytest_with_monitoring("tests/unit/")
    
    print("\n2️⃣ extract_character.py実行例")
    print("   入力: /mnt/c/AItools/lora/train/yado/org/kana05/")
    print("   出力: workspace/QUAL-044/extraction/")
    print("   品質手法: balanced")
    print("   監視: SubAgentによる同一セッション監視")
    
    # デモ結果（実際の実行はコメントアウト）
    # task_id, result = orchestrator.run_extraction_with_monitoring(
    #     "/mnt/c/AItools/lora/train/yado/org/kana05/",
    #     quality_method="balanced",
    #     max_files=5
    # )
    
    print("\n✅ 統合システムの特徴:")
    print("   1. 長時間処理のキュー管理")
    print("   2. 同一セッション内での監視")
    print("   3. 自動リトライ（2回まで）")
    print("   4. 結果解析とレポート生成")
    print("   5. PlanModeエスカレーション準備")
    
    # クリーンアップ
    orchestrator.cleanup()
    
    return True


def main():
    """CLI実行用メイン関数"""
    import sys
    
    if len(sys.argv) < 2:
        # デモンストレーション実行
        demonstrate_task_integration()
    else:
        command = sys.argv[1]
        
        if command == "pytest":
            # pytest実行
            orchestrator = TaskOrchestrator("QUAL-044")
            test_path = sys.argv[2] if len(sys.argv) > 2 else "tests/"
            task_id, result = orchestrator.run_pytest_with_monitoring(test_path)
            print(f"Task ID: {task_id}")
            print(f"Result: {json.dumps(result, indent=2, default=str)}")
            
        elif command == "extract":
            # extract_character実行
            orchestrator = TaskOrchestrator("QUAL-044")
            input_dir = sys.argv[2] if len(sys.argv) > 2 else "/mnt/c/AItools/lora/train/yado/org/kana05/"
            task_id, result = orchestrator.run_extraction_with_monitoring(input_dir)
            print(f"Task ID: {task_id}")
            print(f"Result: {json.dumps(result, indent=2, default=str)}")
            
        else:
            print(f"Unknown command: {command}")
            print("Usage: python task_integration.py [pytest|extract] [path]")
            sys.exit(1)


if __name__ == "__main__":
    main()