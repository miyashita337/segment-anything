"""
統計分析プラグイン

ダッシュボード用の統計情報を生成
"""

import logging
from features.common.dashboard_plugins import DashboardPlugin
from typing import Any, Dict, List


class Plugin(DashboardPlugin):
    """統計分析プラグイン"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @property
    def name(self) -> str:
        return "statistics"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    def execute(self, dashboard_data: Dict[str, Any], 
               plugin_settings: Dict[str, Any]) -> Dict[str, Any]:
        """統計分析実行"""
        try:
            self.logger.info("📊 統計分析プラグイン実行")
            
            quality_scores = dashboard_data.get('quality_scores', [])
            total_images = dashboard_data.get('total_images', 0)
            
            # 基本統計
            statistics = self._calculate_basic_statistics(quality_scores, total_images, plugin_settings)
            
            # 品質分布統計
            distribution_stats = self._calculate_quality_distribution(quality_scores, plugin_settings)
            statistics.update(distribution_stats)
            
            # トレンド分析（時系列データがある場合）
            if plugin_settings.get('enable_trend_analysis', False):
                trend_stats = self._calculate_trend_statistics(dashboard_data, plugin_settings)
                statistics.update(trend_stats)
            
            dashboard_data['statistics'] = statistics
            
            self.logger.info("✅ 統計分析完了")
            return dashboard_data
            
        except Exception as e:
            self.logger.error(f"統計分析プラグインエラー: {e}")
            return dashboard_data
    
    def _calculate_basic_statistics(self, quality_scores: List[float], 
                                   total_images: int,
                                   settings: Dict[str, Any]) -> Dict[str, Any]:
        """基本統計計算"""
        if not quality_scores:
            return {
                'avg_quality': 0.0,
                'min_quality': 0.0,
                'max_quality': 0.0,
                'median_quality': 0.0,
                'std_quality': 0.0,
                'success_rate': 0.0
            }
        
        import statistics as stats
        
        avg_quality = stats.mean(quality_scores)
        min_quality = min(quality_scores)
        max_quality = max(quality_scores)
        median_quality = stats.median(quality_scores)
        
        # 標準偏差
        std_quality = stats.stdev(quality_scores) if len(quality_scores) > 1 else 0.0
        
        # 成功率（中品質以上の割合）
        success_threshold = settings.get('success_threshold', 0.6)
        success_count = sum(1 for score in quality_scores if score >= success_threshold)
        success_rate = success_count / len(quality_scores)
        
        return {
            'avg_quality': avg_quality,
            'min_quality': min_quality,
            'max_quality': max_quality,
            'median_quality': median_quality,
            'std_quality': std_quality,
            'success_rate': success_rate,
            'success_count': success_count,
            'total_processed': len(quality_scores),
            'total_input': total_images
        }
    
    def _calculate_quality_distribution(self, quality_scores: List[float], 
                                       settings: Dict[str, Any]) -> Dict[str, Any]:
        """品質分布統計"""
        thresholds = settings.get('quality_thresholds', {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.3
        })
        
        distribution = {'high': 0, 'medium': 0, 'low': 0, 'poor': 0}
        
        for score in quality_scores:
            if score >= thresholds['high']:
                distribution['high'] += 1
            elif score >= thresholds['medium']:
                distribution['medium'] += 1
            elif score >= thresholds['low']:
                distribution['low'] += 1
            else:
                distribution['poor'] += 1
        
        # パーセント計算
        total = len(quality_scores) if quality_scores else 1
        distribution_percent = {
            key: (count / total) * 100 
            for key, count in distribution.items()
        }
        
        return {
            'quality_distribution': distribution,
            'quality_distribution_percent': distribution_percent,
            'high_quality_count': distribution['high'],
            'medium_quality_count': distribution['medium'],
            'low_quality_count': distribution['low'],
            'poor_count': distribution['poor']
        }
    
    def _calculate_trend_statistics(self, dashboard_data: Dict[str, Any], 
                                   settings: Dict[str, Any]) -> Dict[str, Any]:
        """トレンド統計計算（拡張機能）"""
        # 将来の拡張用
        # 履歴データがある場合のトレンド分析
        return {
            'trend_analysis_available': False,
            'trend_note': '履歴データが不足しています'
        }