#!/usr/bin/env python3
"""
完全自動パイプライン
アニメ全身検出モデル探索とバッチ実行の自動化
"""

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .background_executor import get_task_manager
from .error_recovery import get_auto_recovery_executor
from .progress_monitor import get_progress_monitor

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AutoPipeline:
    """完全自動実行パイプライン"""
    
    def __init__(self):
        """初期化"""
        self.task_manager = get_task_manager()
        self.progress_monitor = get_progress_monitor()
        self.recovery_executor = get_auto_recovery_executor()
        self.pipeline_status = {
            'start_time': None,
            'end_time': None,
            'tasks': [],
            'results': {},
            'errors': []
        }
        
    def start_pipeline(self):
        """パイプライン開始"""
        self.pipeline_status['start_time'] = datetime.now()
        self.progress_monitor.start_monitoring()
        logger.info("Auto pipeline started")
        
    def stop_pipeline(self):
        """パイプライン停止"""
        self.pipeline_status['end_time'] = datetime.now()
        self.progress_monitor.stop_monitoring()
        logger.info("Auto pipeline stopped")
        
    def run_anime_model_search(self) -> Dict[str, Any]:
        """アニメ全身検出モデル探索"""
        logger.info("Starting anime fullbody model search...")
        
        search_results = {
            'models_found': [],
            'best_model': None,
            'test_results': {}
        }
        
        # 利用可能なモデル候補
        model_candidates = [
            {
                'name': 'yolov8x6_animeface.pt',
                'type': 'anime_specialized',
                'url': 'local',
                'description': 'アニメ顔特化モデル'
            },
            {
                'name': 'yolov8x.pt',
                'type': 'general',
                'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt',
                'description': '汎用高精度モデル'
            },
            {
                'name': 'yolov8l.pt',
                'type': 'general',
                'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt',
                'description': '汎用大規模モデル'
            }
        ]
        
        # モデル検索・テスト
        for model_info in model_candidates:
            try:
                # モデル存在確認
                model_path = model_info['name']
                if os.path.exists(model_path):
                    logger.info(f"Found model: {model_path}")
                    search_results['models_found'].append(model_info)
                    
                    # テスト実行
                    test_result = self._test_anime_model(model_path)
                    search_results['test_results'][model_path] = test_result
                    
            except Exception as e:
                logger.error(f"Error testing model {model_info['name']}: {e}")
                
        # 最適モデル選択
        if search_results['test_results']:
            best_model = max(
                search_results['test_results'].items(),
                key=lambda x: x[1].get('fullbody_score', 0)
            )
            search_results['best_model'] = best_model[0]
            
        return search_results
        
    def _test_anime_model(self, model_path: str) -> Dict[str, Any]:
        """アニメモデルテスト"""
        try:
            from ultralytics import YOLO
            
            model = YOLO(model_path)
            test_results = {
                'model_path': model_path,
                'tests': []
            }
            
            # テスト画像
            test_images = [
                "test_small/img001.jpg",
                "test_small/img002.jpg",
                "test_small/img003.jpg"
            ]
            
            fullbody_count = 0
            total_detections = 0
            
            for img_path in test_images:
                if os.path.exists(img_path):
                    results = model(img_path, conf=0.05, verbose=False)
                    
                    if results[0].boxes is not None:
                        boxes = results[0].boxes
                        detections = len(boxes)
                        total_detections += detections
                        
                        # 全身検出判定（簡易版）
                        for box in boxes.xyxy:
                            x1, y1, x2, y2 = box
                            height = y2 - y1
                            width = x2 - x1
                            aspect_ratio = height / width if width > 0 else 0
                            
                            # アスペクト比で全身判定
                            if 1.5 < aspect_ratio < 4.0:
                                fullbody_count += 1
                                
            # スコア計算
            fullbody_score = fullbody_count / max(total_detections, 1)
            
            test_results['fullbody_count'] = fullbody_count
            test_results['total_detections'] = total_detections
            test_results['fullbody_score'] = fullbody_score
            
            return test_results
            
        except Exception as e:
            logger.error(f"Model test error: {e}")
            return {'error': str(e), 'fullbody_score': 0}
            
    def run_batch_processing(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        """バッチ処理実行"""
        logger.info("Starting batch processing...")
        
        if model_path is None:
            model_path = 'yolov8x.pt'
            
        # バッチ処理パラメータ
        batch_params = {
            'input_dir': 'test_small',
            'output_dir': 'results_batch/auto_pipeline_results',
            'model_path': model_path,
            'score_threshold': 0.005,
            'quality_method': 'balanced'
        }
        
        # タスク実行
        task_id = self.task_manager.run_batch_extraction(
            batch_params['input_dir'],
            batch_params['output_dir'],
            model_path=batch_params['model_path'],
            score_threshold=batch_params['score_threshold'],
            quality_method=batch_params['quality_method']
        )
        
        self.pipeline_status['tasks'].append(task_id)
        
        # 結果待機（タイムアウト10分）
        try:
            result = self.task_manager.wait_for_completion(task_id, timeout=600)
            return result
        except TimeoutError:
            return {'error': 'Batch processing timeout', 'task_id': task_id}
            
    def collect_output_paths(self) -> List[str]:
        """出力画像パス収集"""
        logger.info("Collecting output paths...")
        
        output_paths = []
        
        # 既知の出力ディレクトリ
        output_dirs = [
            'results_batch',
            'results_batch/auto_pipeline_results',
            'results_batch/yolo_005_test',
            '/tmp'
        ]
        
        # 画像ファイル収集
        for output_dir in output_dirs:
            if os.path.exists(output_dir):
                for root, dirs, files in os.walk(output_dir):
                    for file in files:
                        if file.endswith(('.jpg', '.jpeg', '.png')):
                            full_path = os.path.join(root, file)
                            output_paths.append(full_path)
                            
        # Phase 2テスト結果
        phase2_patterns = [
            '/tmp/phase2_test_*.jpg',
            '/tmp/test_*.jpg',
            '/tmp/pipeline_test/*.jpg'
        ]
        
        import glob
        for pattern in phase2_patterns:
            matches = glob.glob(pattern)
            output_paths.extend(matches)
            
        # 重複除去
        output_paths = list(set(output_paths))
        output_paths.sort()
        
        return output_paths
        
    def generate_final_report(self) -> Dict[str, Any]:
        """最終レポート生成"""
        logger.info("Generating final report...")
        
        output_paths = self.collect_output_paths()
        
        report = {
            'pipeline_status': self.pipeline_status,
            'execution_time': str(
                self.pipeline_status['end_time'] - self.pipeline_status['start_time']
            ) if self.pipeline_status['end_time'] else 'Running',
            'output_images': {
                'count': len(output_paths),
                'paths': output_paths
            },
            'model_search_results': self.pipeline_status['results'].get('model_search', {}),
            'batch_results': self.pipeline_status['results'].get('batch_processing', {}),
            'errors': self.pipeline_status['errors']
        }
        
        # レポート保存
        report_path = Path('temp/logs/final_pipeline_report.json')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            
        return report
        
    def run_full_pipeline(self):
        """完全自動パイプライン実行"""
        self.start_pipeline()
        
        try:
            # 1. アニメモデル探索
            logger.info("=== Phase 1: Anime Model Search ===")
            model_search_results = self.recovery_executor.execute_with_recovery(
                self.run_anime_model_search
            )
            self.pipeline_status['results']['model_search'] = model_search_results
            
            # 2. 最適モデルでバッチ処理
            logger.info("=== Phase 2: Batch Processing ===")
            best_model = model_search_results.get('best_model')
            batch_results = self.recovery_executor.execute_with_recovery(
                self.run_batch_processing,
                best_model
            )
            self.pipeline_status['results']['batch_processing'] = batch_results
            
            # 3. 結果集約
            logger.info("=== Phase 3: Results Collection ===")
            final_report = self.generate_final_report()
            
            # レポート出力
            self._print_final_report(final_report)
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            self.pipeline_status['errors'].append(str(e))
            
        finally:
            self.stop_pipeline()
            
    def _print_final_report(self, report: Dict[str, Any]):
        """最終レポート表示"""
        print("\n" + "="*80)
        print("🎉 パイプライン実行完了レポート")
        print("="*80)
        
        print(f"\n⏱️ 実行時間: {report['execution_time']}")
        
        print(f"\n📊 モデル探索結果:")
        model_results = report.get('model_search_results', {})
        print(f"   発見モデル数: {len(model_results.get('models_found', []))}")
        print(f"   最適モデル: {model_results.get('best_model', 'N/A')}")
        
        print(f"\n📁 出力画像:")
        print(f"   総数: {report['output_images']['count']}")
        print(f"   保存先:")
        for path in report['output_images']['paths'][:10]:  # 最初の10件
            print(f"      - {path}")
        if len(report['output_images']['paths']) > 10:
            print(f"      ... 他 {len(report['output_images']['paths']) - 10} ファイル")
            
        if report['errors']:
            print(f"\n⚠️ エラー:")
            for error in report['errors']:
                print(f"   - {error}")
                
        print(f"\n💾 詳細レポート: temp/logs/final_pipeline_report.json")
        print("="*80 + "\n")


# シングルトンインスタンス
_auto_pipeline = None


def get_auto_pipeline() -> AutoPipeline:
    """自動パイプラインのシングルトンインスタンス取得"""
    global _auto_pipeline
    if _auto_pipeline is None:
        _auto_pipeline = AutoPipeline()
    return _auto_pipeline