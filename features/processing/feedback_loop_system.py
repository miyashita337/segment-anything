#!/usr/bin/env python3
"""
P1-016: フィードバックループシステム
処理速度改善・ボトルネック特定・適応的最適化の統合システム

目標:
- 自動フィードバック統合
- パフォーマンストレンド分析  
- 適応的パラメータ調整
- 品質予測改善
"""

import numpy as np

import json
import logging
import queue
# プロジェクトルートをパスに追加
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from features.evaluation.utils.learned_quality_assessment import (
    LearnedQualityAssessment,
    QualityPrediction,
)
from features.processing.adaptive_parameter_optimizer import (
    AdaptiveParameterOptimizer,
    ImageCharacteristics,
    OptimizationParameters,
)

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """パフォーマンス指標"""
    total_processing_time: float
    yolo_inference_time: float
    sam_inference_time: float
    postprocessing_time: float
    memory_usage_mb: float
    gpu_utilization: float
    success_rate: float
    quality_score: float
    bottleneck_stage: str
    
    
@dataclass
class ProcessingSession:
    """処理セッション情報"""
    session_id: str
    start_time: float
    end_time: Optional[float]
    image_path: str
    image_characteristics: ImageCharacteristics
    optimization_parameters: OptimizationParameters
    quality_prediction: QualityPrediction
    performance_metrics: Optional[PerformanceMetrics]
    success: bool
    error_message: Optional[str]


class PerformanceMonitor:
    """パフォーマンス監視システム"""
    
    def __init__(self):
        self.monitoring_data = []
        self.bottleneck_history = []
        self.performance_trends = {}
        self._lock = threading.Lock()
    
    def start_monitoring(self, session_id: str) -> Dict[str, float]:
        """監視開始"""
        start_metrics = {
            'session_id': session_id,
            'start_time': time.time(),
            'start_memory': self._get_memory_usage(),
            'start_gpu_memory': self._get_gpu_memory()
        }
        
        logger.debug(f"📊 監視開始: {session_id}")
        return start_metrics
    
    def record_stage_time(self, session_id: str, stage: str, duration: float):
        """ステージ別時間記録"""
        with self._lock:
            timestamp = time.time()
            self.monitoring_data.append({
                'session_id': session_id,
                'timestamp': timestamp,
                'stage': stage,
                'duration': duration,
                'memory_usage': self._get_memory_usage(),
                'gpu_memory': self._get_gpu_memory()
            })
            
        logger.debug(f"⏱️ ステージ記録: {stage} = {duration:.2f}秒")
    
    def end_monitoring(self, session_id: str, start_metrics: Dict) -> PerformanceMetrics:
        """監視終了・メトリクス計算"""
        end_time = time.time()
        total_time = end_time - start_metrics['start_time']
        
        # ステージ別時間集計
        stage_times = {}
        memory_peak = start_metrics['start_memory']
        gpu_peak = start_metrics['start_gpu_memory']
        
        with self._lock:
            session_data = [d for d in self.monitoring_data if d['session_id'] == session_id]
            
            for data in session_data:
                stage = data['stage']
                stage_times[stage] = stage_times.get(stage, 0) + data['duration']
                memory_peak = max(memory_peak, data['memory_usage'])
                gpu_peak = max(gpu_peak, data['gpu_memory'])
        
        # ボトルネック特定
        bottleneck_stage = self._identify_bottleneck(stage_times)
        
        # GPU使用率推定（簡易）
        gpu_utilization = min(gpu_peak / 16384 * 100, 100)  # 16GB GPU想定
        
        metrics = PerformanceMetrics(
            total_processing_time=total_time,
            yolo_inference_time=stage_times.get('yolo_inference', 0),
            sam_inference_time=stage_times.get('sam_inference', 0),
            postprocessing_time=stage_times.get('postprocessing', 0),
            memory_usage_mb=memory_peak,
            gpu_utilization=gpu_utilization,
            success_rate=1.0,  # セッション単位では成功/失敗
            quality_score=0.0,  # 後で品質評価結果で更新
            bottleneck_stage=bottleneck_stage
        )
        
        logger.info(f"📈 監視完了: 総時間{total_time:.1f}秒, ボトルネック: {bottleneck_stage}")
        return metrics
    
    def _identify_bottleneck(self, stage_times: Dict[str, float]) -> str:
        """ボトルネック特定"""
        if not stage_times:
            return "unknown"
        
        max_stage = max(stage_times.items(), key=lambda x: x[1])
        return max_stage[0]
    
    def _get_memory_usage(self) -> float:
        """メモリ使用量取得"""
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def _get_gpu_memory(self) -> float:
        """GPU メモリ使用量取得"""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024 / 1024
            return 0.0
        except:
            return 0.0
    
    def get_bottleneck_analysis(self) -> Dict[str, Any]:
        """ボトルネック分析結果"""
        with self._lock:
            if not self.monitoring_data:
                return {'message': '監視データがありません'}
            
            # ステージ別平均時間計算
            stage_totals = {}
            stage_counts = {}
            
            for data in self.monitoring_data[-100:]:  # 直近100レコード
                stage = data['stage']
                duration = data['duration']
                
                stage_totals[stage] = stage_totals.get(stage, 0) + duration
                stage_counts[stage] = stage_counts.get(stage, 0) + 1
            
            stage_averages = {
                stage: stage_totals[stage] / stage_counts[stage]
                for stage in stage_totals
            }
            
            # ボトルネック特定
            if stage_averages:
                bottleneck = max(stage_averages.items(), key=lambda x: x[1])
                bottleneck_percentage = (bottleneck[1] / sum(stage_averages.values())) * 100
            else:
                bottleneck = ("unknown", 0)
                bottleneck_percentage = 0
            
            return {
                'stage_averages': stage_averages,
                'primary_bottleneck': bottleneck[0],
                'bottleneck_time': bottleneck[1],
                'bottleneck_percentage': bottleneck_percentage,
                'total_samples': len(self.monitoring_data)
            }


class FeedbackLoopSystem:
    """フィードバックループシステム"""
    
    def __init__(self, tracker_id: str = "P1-016"):
        self.tracker_id = tracker_id
        
        # コンポーネント初期化
        self.parameter_optimizer = AdaptiveParameterOptimizer()
        self.quality_assessor = LearnedQualityAssessment()
        self.performance_monitor = PerformanceMonitor()
        
        # セッション管理
        self.active_sessions: Dict[str, ProcessingSession] = {}
        self.completed_sessions: List[ProcessingSession] = []
        
        # フィードバック収集
        self.feedback_queue = queue.Queue()
        self.feedback_processor_thread = None
        self.running = False
        
        # 統計
        self.processing_stats = {
            'total_sessions': 0,
            'successful_sessions': 0,
            'average_processing_time': 0.0,
            'average_quality_score': 0.0,
            'most_common_bottleneck': 'unknown',
            'optimization_improvements': 0
        }
        
        logger.info(f"🔄 P1-016 フィードバックループシステム初期化: {tracker_id}")
    
    def start_feedback_processing(self):
        """フィードバック処理開始"""
        if self.feedback_processor_thread is None or not self.feedback_processor_thread.is_alive():
            self.running = True
            self.feedback_processor_thread = threading.Thread(
                target=self._process_feedback_loop,
                daemon=True
            )
            self.feedback_processor_thread.start()
            logger.info("🔄 フィードバック処理スレッド開始")
    
    def stop_feedback_processing(self):
        """フィードバック処理停止"""
        self.running = False
        if self.feedback_processor_thread:
            self.feedback_processor_thread.join(timeout=5.0)
        logger.info("⏹️ フィードバック処理停止")
    
    def create_processing_session(self, image_path: str) -> str:
        """処理セッション作成"""
        session_id = f"session_{int(time.time())}_{len(self.active_sessions)}"
        
        # 画像特性分析
        try:
            import cv2
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"画像読み込み失敗: {image_path}")
            
            # 適応的パラメータ最適化
            optimization_params = self.parameter_optimizer.optimize_parameters_for_image(image)
            
            # 品質予測
            quality_prediction = self.quality_assessor.predict_quality_and_method(
                self.quality_assessor.analyze_image_characteristics(image_path)
            )
            
            # セッション作成
            session = ProcessingSession(
                session_id=session_id,
                start_time=time.time(),
                end_time=None,
                image_path=image_path,
                image_characteristics=self.parameter_optimizer._analyze_image_characteristics(image),
                optimization_parameters=optimization_params,
                quality_prediction=quality_prediction,
                performance_metrics=None,
                success=False,
                error_message=None
            )
            
            self.active_sessions[session_id] = session
            
            logger.info(f"📝 セッション作成: {session_id} ({Path(image_path).name})")
            logger.info(f"   予測品質: {quality_prediction.predicted_quality:.3f}")
            logger.info(f"   推奨手法: {quality_prediction.recommended_method}")
            logger.info(f"   YOLO閾値: {optimization_params.yolo_threshold:.4f}")
            
            return session_id
            
        except Exception as e:
            logger.error(f"❌ セッション作成エラー: {e}")
            return None
    
    def start_processing_monitoring(self, session_id: str) -> Optional[Dict]:
        """処理監視開始"""
        if session_id not in self.active_sessions:
            logger.error(f"❌ セッション未発見: {session_id}")
            return None
        
        start_metrics = self.performance_monitor.start_monitoring(session_id)
        return start_metrics
    
    def record_processing_stage(self, session_id: str, stage: str, duration: float):
        """処理ステージ記録"""
        self.performance_monitor.record_stage_time(session_id, stage, duration)
    
    def complete_processing_session(self, 
                                  session_id: str, 
                                  start_metrics: Dict,
                                  success: bool,
                                  actual_quality_score: Optional[float] = None,
                                  error_message: Optional[str] = None):
        """処理セッション完了"""
        if session_id not in self.active_sessions:
            logger.error(f"❌ セッション未発見: {session_id}")
            return
        
        session = self.active_sessions[session_id]
        
        # パフォーマンス メトリクス計算
        performance_metrics = self.performance_monitor.end_monitoring(session_id, start_metrics)
        
        # 実際の品質スコア設定
        if actual_quality_score is not None:
            performance_metrics.quality_score = actual_quality_score
        
        # セッション完了
        session.end_time = time.time()
        session.performance_metrics = performance_metrics
        session.success = success
        session.error_message = error_message
        
        # 完了セッションに移動
        self.completed_sessions.append(session)
        del self.active_sessions[session_id]
        
        # フィードバックキューに追加
        self.feedback_queue.put(session)
        
        # 統計更新
        self._update_processing_stats()
        
        logger.info(f"✅ セッション完了: {session_id}")
        logger.info(f"   成功: {success}")
        logger.info(f"   総時間: {performance_metrics.total_processing_time:.2f}秒")
        logger.info(f"   ボトルネック: {performance_metrics.bottleneck_stage}")
        if actual_quality_score:
            logger.info(f"   品質スコア: {actual_quality_score:.3f}")
    
    def _process_feedback_loop(self):
        """フィードバックループ処理"""
        logger.info("🔄 フィードバックループ処理開始")
        
        while self.running:
            try:
                # タイムアウト付きでフィードバック取得
                session = self.feedback_queue.get(timeout=1.0)
                
                if session.success and session.performance_metrics:
                    # 成功セッションからの学習
                    self._learn_from_successful_session(session)
                
                if session.performance_metrics:
                    # パフォーマンス分析更新
                    self._analyze_performance_trends(session)
                
                # フィードバック処理完了
                self.feedback_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ フィードバック処理エラー: {e}")
        
        logger.info("🔄 フィードバックループ処理終了")
    
    def _learn_from_successful_session(self, session: ProcessingSession):
        """成功セッションからの学習"""
        try:
            if not session.performance_metrics:
                return
            
            # 品質フィードバック学習
            quality_score = session.performance_metrics.quality_score
            if quality_score > 0:
                self.parameter_optimizer.learn_from_quality_feedback(
                    session.image_characteristics,
                    quality_score,
                    session.optimization_parameters
                )
                
                logger.debug(f"📚 品質フィードバック学習: {quality_score:.3f}")
            
            # パフォーマンス最適化学習
            if session.performance_metrics.total_processing_time > 0:
                # 高速処理の成功例を学習
                if session.performance_metrics.total_processing_time < 300:  # 5分未満
                    self._learn_performance_optimization(session)
                    
        except Exception as e:
            logger.error(f"❌ 学習処理エラー: {e}")
    
    def _learn_performance_optimization(self, session: ProcessingSession):
        """パフォーマンス最適化学習"""
        # 高速処理を実現したパラメータパターンを記録
        # 将来の類似画像で優先的に使用
        
        pattern_key = self.parameter_optimizer._generate_pattern_key(session.image_characteristics)
        processing_time = session.performance_metrics.total_processing_time
        
        # 性能向上パターンとして記録
        if not hasattr(self.parameter_optimizer, 'performance_patterns'):
            self.parameter_optimizer.performance_patterns = {}
        
        if pattern_key not in self.parameter_optimizer.performance_patterns:
            self.parameter_optimizer.performance_patterns[pattern_key] = []
        
        self.parameter_optimizer.performance_patterns[pattern_key].append({
            'processing_time': processing_time,
            'parameters': asdict(session.optimization_parameters),
            'timestamp': session.start_time
        })
        
        # 最新10件のみ保持
        if len(self.parameter_optimizer.performance_patterns[pattern_key]) > 10:
            self.parameter_optimizer.performance_patterns[pattern_key] = \
                self.parameter_optimizer.performance_patterns[pattern_key][-10:]
        
        logger.debug(f"⚡ パフォーマンス最適化学習: {pattern_key} ({processing_time:.1f}秒)")
    
    def _analyze_performance_trends(self, session: ProcessingSession):
        """パフォーマンストレンド分析"""
        if not session.performance_metrics:
            return
        
        metrics = session.performance_metrics
        
        # ボトルネック統計更新
        bottleneck = metrics.bottleneck_stage
        if bottleneck not in self.performance_monitor.bottleneck_history:
            self.performance_monitor.bottleneck_history = []
        
        self.performance_monitor.bottleneck_history.append({
            'timestamp': session.start_time,
            'bottleneck': bottleneck,
            'processing_time': metrics.total_processing_time,
            'yolo_time': metrics.yolo_inference_time,
            'sam_time': metrics.sam_inference_time
        })
        
        # 履歴サイズ制限
        if len(self.performance_monitor.bottleneck_history) > 100:
            self.performance_monitor.bottleneck_history = \
                self.performance_monitor.bottleneck_history[-100:]
    
    def _update_processing_stats(self):
        """処理統計更新"""
        if not self.completed_sessions:
            return
        
        successful_sessions = [s for s in self.completed_sessions if s.success]
        total_sessions = len(self.completed_sessions)
        
        # 基本統計
        self.processing_stats['total_sessions'] = total_sessions
        self.processing_stats['successful_sessions'] = len(successful_sessions)
        
        if successful_sessions:
            # 平均処理時間
            avg_time = np.mean([
                s.performance_metrics.total_processing_time 
                for s in successful_sessions 
                if s.performance_metrics
            ])
            self.processing_stats['average_processing_time'] = avg_time
            
            # 平均品質スコア
            quality_scores = [
                s.performance_metrics.quality_score 
                for s in successful_sessions 
                if s.performance_metrics and s.performance_metrics.quality_score > 0
            ]
            if quality_scores:
                self.processing_stats['average_quality_score'] = np.mean(quality_scores)
            
            # 最頻ボトルネック
            bottlenecks = [
                s.performance_metrics.bottleneck_stage 
                for s in successful_sessions 
                if s.performance_metrics
            ]
            if bottlenecks:
                from collections import Counter
                most_common = Counter(bottlenecks).most_common(1)
                self.processing_stats['most_common_bottleneck'] = most_common[0][0]
    
    def get_performance_analysis(self) -> Dict[str, Any]:
        """パフォーマンス分析結果取得"""
        bottleneck_analysis = self.performance_monitor.get_bottleneck_analysis()
        
        return {
            'processing_stats': self.processing_stats,
            'bottleneck_analysis': bottleneck_analysis,
            'active_sessions': len(self.active_sessions),
            'completed_sessions': len(self.completed_sessions),
            'optimization_report': self.parameter_optimizer.get_optimization_report()
        }
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """最適化推奨事項取得"""
        recommendations = []
        
        # ボトルネック分析から推奨事項生成
        bottleneck_analysis = self.performance_monitor.get_bottleneck_analysis()
        primary_bottleneck = bottleneck_analysis.get('primary_bottleneck', 'unknown')
        
        if primary_bottleneck == 'sam_inference':
            recommendations.append({
                'type': 'bottleneck_optimization',
                'priority': 'high',
                'title': 'SAM推論最適化',
                'description': 'SAM推論が主ボトルネック（5-6分）。points_per_side削減やcrop_layers最適化を推奨',
                'actions': [
                    'sam_points_per_side を 32 → 16 に削減',
                    'sam_crop_n_layers を 1 → 0 に削減',
                    'sam_pred_iou_thresh を 0.86 → 0.8 に緩和'
                ]
            })
        
        if primary_bottleneck == 'yolo_inference':
            recommendations.append({
                'type': 'bottleneck_optimization',
                'priority': 'medium',
                'title': 'YOLO推論最適化',
                'description': 'YOLO推論が主ボトルネック。モデルサイズ削減を推奨',
                'actions': [
                    'yolov8x.pt → yolov8n.pt に変更',
                    'yolo_threshold の最適化',
                    'アニメ特化YOLOモデルの使用検討'
                ]
            })
        
        # 品質・パフォーマンストレードオフ分析
        if len(self.completed_sessions) >= 5:
            recent_sessions = self.completed_sessions[-10:]
            fast_sessions = [
                s for s in recent_sessions 
                if s.performance_metrics and s.performance_metrics.total_processing_time < 240
            ]
            
            if len(fast_sessions) >= 3:
                recommendations.append({
                    'type': 'parameter_optimization',
                    'priority': 'medium',
                    'title': '高速パラメータの活用',
                    'description': f'{len(fast_sessions)}件の高速処理事例を発見。同様のパラメータ適用を推奨',
                    'actions': [
                        '高速処理パラメータパターンの自動適用',
                        '画像特性に基づく事前最適化',
                        '処理時間予測の精度向上'
                    ]
                })
        
        return recommendations
    
    def save_feedback_data(self, output_path: str) -> bool:
        """フィードバックデータ保存"""
        try:
            save_data = {
                'tracker_id': self.tracker_id,
                'saved_at': datetime.now().isoformat(),
                'processing_stats': self.processing_stats,
                'completed_sessions_count': len(self.completed_sessions),
                'performance_analysis': self.get_performance_analysis(),
                'optimization_recommendations': self.get_optimization_recommendations(),
                'learned_patterns': self.parameter_optimizer.learned_patterns,
                'bottleneck_history': self.performance_monitor.bottleneck_history[-50:]  # 最新50件
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 フィードバックデータ保存: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ フィードバックデータ保存エラー: {e}")
            return False


def create_feedback_loop_system(tracker_id: str = "P1-016") -> FeedbackLoopSystem:
    """フィードバックループシステム ファクトリ関数"""
    return FeedbackLoopSystem(tracker_id)


if __name__ == "__main__":
    # テスト実行
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🧪 P1-016 フィードバックループシステム テスト")
    
    system = create_feedback_loop_system("P1-016-TEST")
    system.start_feedback_processing()
    
    # テスト用画像でセッション作成
    test_image = "/mnt/c/AItools/lora/train/yado/org/kana05/kana05_0001.jpg"
    
    if Path(test_image).exists():
        session_id = system.create_processing_session(test_image)
        
        if session_id:
            # 模擬処理実行
            start_metrics = system.start_processing_monitoring(session_id)
            
            # 模擬ステージ処理時間記録
            system.record_processing_stage(session_id, 'yolo_inference', 30.0)
            system.record_processing_stage(session_id, 'sam_inference', 350.0)
            system.record_processing_stage(session_id, 'postprocessing', 25.0)
            
            # セッション完了
            system.complete_processing_session(
                session_id, start_metrics, 
                success=True, 
                actual_quality_score=2.5
            )
            
            # 分析結果表示
            analysis = system.get_performance_analysis()
            print(f"📊 パフォーマンス分析:")
            print(f"   処理統計: {analysis['processing_stats']}")
            print(f"   ボトルネック分析: {analysis['bottleneck_analysis']}")
            
            recommendations = system.get_optimization_recommendations()
            print(f"💡 最適化推奨事項: {len(recommendations)}件")
            for rec in recommendations:
                print(f"   - {rec['title']}: {rec['description']}")
    
    system.stop_feedback_processing()
    print("✅ テスト完了")