"""
統合ダッシュボード生成システム

既存のダッシュボード生成システムを統合し、一元管理を提供
- StandardDashboardGeneratorをベースに機能拡張
- QI-004の画像品質解析機能を統合
- quality_dashboardのグラフ生成機能を統合
- 設定駆動型でトラッカー固有処理に対応
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import importlib.util

# 既存システムのインポート
from .dashboard_generator import StandardDashboardGenerator
from .dashboard_config import DashboardConfigManager, DashboardConfig

# QI-004統合用（オプショナル）
try:
    from ..evaluation.qi004_dashboard_optimization_system import (
        ImageQualityAnalyzer, 
        DashboardOptimizer
    )
    QI004_AVAILABLE = True
except ImportError:
    QI004_AVAILABLE = False

# quality_dashboard統合用（オプショナル）
try:
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent / "tools" / "core"))
    from quality_dashboard import QualityDashboard
    QUALITY_DASHBOARD_AVAILABLE = True
except ImportError:
    QUALITY_DASHBOARD_AVAILABLE = False


class UnifiedDashboardGenerator:
    """
    統合ダッシュボード生成システム
    
    機能:
    - 既存システムの統合利用
    - 設定駆動型のダッシュボード生成
    - プラグインベースの機能拡張
    - 一元的なAPI提供
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """初期化"""
        self.logger = logging.getLogger(__name__)
        
        # 設定管理
        self.config_manager = DashboardConfigManager(config_dir)
        
        # 既存システムの初期化
        self.standard_generator = StandardDashboardGenerator()
        
        # QI-004機能（利用可能な場合）
        if QI004_AVAILABLE:
            self.image_quality_analyzer = ImageQualityAnalyzer()
            self.dashboard_optimizer = DashboardOptimizer()
            self.logger.info("QI-004機能統合: 有効")
        else:
            self.image_quality_analyzer = None
            self.dashboard_optimizer = None
            self.logger.info("QI-004機能統合: 無効（モジュール未発見）")
        
        # quality_dashboard機能（利用可能な場合）
        if QUALITY_DASHBOARD_AVAILABLE:
            self.quality_dashboard = QualityDashboard()
            self.logger.info("QualityDashboard機能統合: 有効")
        else:
            self.quality_dashboard = None
            self.logger.info("QualityDashboard機能統合: 無効（モジュール未発見）")
        
        # プラグインシステム
        self.plugins = {}
        self._load_plugins()
    
    def generate_dashboard(self, tracker_id: str, 
                          extraction_dir: str,
                          output_dir: str,
                          config_override: Optional[Dict[str, Any]] = None) -> Path:
        """
        統一ダッシュボード生成
        
        Args:
            tracker_id: トラッカーID
            extraction_dir: 抽出画像ディレクトリ
            output_dir: 出力ディレクトリ
            config_override: 設定オーバーライド（オプション）
            
        Returns:
            Path: 生成されたダッシュボードHTMLファイルパス
        """
        self.logger.info(f"🔄 統合ダッシュボード生成開始: {tracker_id}")
        
        try:
            # 1. 設定ロード
            config = self.config_manager.load_config(tracker_id)
            if config_override:
                self._apply_config_override(config, config_override)
            
            # 2. 入出力パス設定
            config.extraction_dir = extraction_dir
            config.output_dir = output_dir
            
            # 3. 画像データ収集・解析
            dashboard_data = self._collect_and_analyze_data(config)
            
            # 4. プラグイン実行
            dashboard_data = self._execute_plugins(config, dashboard_data)
            
            # 5. ダッシュボード生成
            dashboard_path = self._generate_dashboard_content(config, dashboard_data)
            
            self.logger.info(f"✅ 統合ダッシュボード生成完了: {dashboard_path}")
            self.logger.info(f"📊 サイズ: {dashboard_path.stat().st_size / 1024:.1f}KB")
            
            return dashboard_path
            
        except Exception as e:
            self.logger.error(f"❌ ダッシュボード生成エラー {tracker_id}: {e}")
            raise
    
    def generate_quality_report_dashboard(self, tracker_id: str, 
                                        quality_report_path: str,
                                        output_dir: str) -> Path:
        """
        品質レポートベースのダッシュボード生成
        
        Args:
            tracker_id: トラッカーID
            quality_report_path: 品質レポートJSONファイルパス
            output_dir: 出力ディレクトリ
            
        Returns:
            Path: 生成されたダッシュボードHTMLファイルパス
        """
        if not QUALITY_DASHBOARD_AVAILABLE:
            raise RuntimeError("QualityDashboard機能が利用できません")
        
        self.logger.info(f"🔄 品質レポートダッシュボード生成: {tracker_id}")
        
        try:
            # quality_dashboardシステムを使用
            html_path = self.quality_dashboard.create_dashboard(
                quality_report_path, output_dir
            )
            
            return Path(html_path)
            
        except Exception as e:
            self.logger.error(f"❌ 品質レポートダッシュボード生成エラー {tracker_id}: {e}")
            raise
    
    def _collect_and_analyze_data(self, config: DashboardConfig) -> Dict[str, Any]:
        """データ収集・解析"""
        self.logger.info("📊 データ収集・解析開始")
        
        data = {
            'tracker_id': config.tracker_id,
            'title': config.title or f"{config.tracker_id} 品質評価ダッシュボード",
            'description': config.description,
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'image_paths': [],
            'quality_scores': [],
            'black_screen_indices': [],
            'total_images': 0,
            'statistics': {},
            'optimization_data': {}
        }
        
        if not config.extraction_dir or not os.path.exists(config.extraction_dir):
            self.logger.warning(f"⚠️ 抽出ディレクトリが存在しません: {config.extraction_dir}")
            return data
        
        # 1. 画像ファイル収集
        image_paths = self._collect_images(config.extraction_dir)
        data['image_paths'] = image_paths
        data['total_images'] = len(image_paths)
        
        if not image_paths:
            self.logger.warning("⚠️ 処理対象画像が見つかりません")
            return data
        
        # 2. 画像品質解析（QI-004機能利用）
        if config.quality_analysis.enable_image_analysis and self.image_quality_analyzer:
            self.logger.info("🔍 画像品質解析実行")
            quality_scores = []
            black_screen_indices = []
            
            for i, image_path in enumerate(image_paths):
                try:
                    analysis = self.image_quality_analyzer.analyze_image_quality(image_path)
                    score = analysis.get('overall_score', 0.0)
                    quality_scores.append(score)
                    
                    # 黒画面検出（品質スコアが極端に低い場合）
                    if score < 0.1:
                        black_screen_indices.append(i)
                        
                except Exception as e:
                    self.logger.warning(f"画像解析エラー {image_path}: {e}")
                    quality_scores.append(0.0)
            
            data['quality_scores'] = quality_scores
            data['black_screen_indices'] = black_screen_indices
        
        # 3. パフォーマンス最適化（QI-004機能利用）
        if self.dashboard_optimizer:
            self.logger.info("⚡ ダッシュボード最適化実行")
            try:
                optimization_result = self.dashboard_optimizer.optimize_dashboard_performance(
                    image_paths, config.output_dir or ""
                )
                data['optimization_data'] = optimization_result
            except Exception as e:
                self.logger.warning(f"最適化エラー: {e}")
        
        # 4. 統計情報計算
        if config.quality_analysis.enable_statistical_analysis:
            data['statistics'] = self._calculate_statistics(data)
        
        self.logger.info(f"📊 データ収集完了: {len(image_paths)}枚の画像を解析")
        return data
    
    def _collect_images(self, extraction_dir: str) -> List[str]:
        """画像ファイル収集"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
        image_paths = []
        
        for file_path in Path(extraction_dir).rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_paths.append(str(file_path))
        
        return sorted(image_paths)
    
    def _calculate_statistics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """統計情報計算"""
        quality_scores = data.get('quality_scores', [])
        total_images = data.get('total_images', 0)
        
        if not quality_scores:
            return {
                'avg_quality': 0.0,
                'success_count': 0,
                'poor_count': total_images,
                'quality_distribution': {'high': 0, 'medium': 0, 'low': 0, 'poor': 0}
            }
        
        # 基本統計
        avg_quality = sum(quality_scores) / len(quality_scores)
        
        # 品質分布
        thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.3
        }
        
        distribution = {'high': 0, 'medium': 0, 'low': 0, 'poor': 0}
        success_count = 0
        
        for score in quality_scores:
            if score >= thresholds['high']:
                distribution['high'] += 1
                success_count += 1
            elif score >= thresholds['medium']:
                distribution['medium'] += 1
                success_count += 1
            elif score >= thresholds['low']:
                distribution['low'] += 1
            else:
                distribution['poor'] += 1
        
        poor_count = distribution['poor']
        
        return {
            'avg_quality': avg_quality,
            'success_count': success_count,
            'poor_count': poor_count,
            'quality_distribution': distribution
        }
    
    def _execute_plugins(self, config: DashboardConfig, 
                        dashboard_data: Dict[str, Any]) -> Dict[str, Any]:
        """プラグイン実行"""
        enabled_plugins = config.plugins.enabled_plugins
        
        for plugin_name in enabled_plugins:
            if plugin_name in self.plugins:
                try:
                    self.logger.debug(f"🔌 プラグイン実行: {plugin_name}")
                    plugin_settings = config.plugins.plugin_settings.get(plugin_name, {})
                    dashboard_data = self.plugins[plugin_name].execute(
                        dashboard_data, plugin_settings
                    )
                except Exception as e:
                    self.logger.warning(f"プラグインエラー {plugin_name}: {e}")
        
        return dashboard_data
    
    def _generate_dashboard_content(self, config: DashboardConfig,
                                  dashboard_data: Dict[str, Any]) -> Path:
        """ダッシュボードコンテンツ生成"""
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # StandardDashboardGeneratorを使用してHTML生成
        dashboard_path = self.standard_generator.generate_standard_dashboard(
            dashboard_data, str(output_dir)
        )
        
        return dashboard_path
    
    def _apply_config_override(self, config: DashboardConfig, 
                             override: Dict[str, Any]):
        """設定オーバーライドを適用"""
        for key, value in override.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                config.custom_settings[key] = value
    
    def _load_plugins(self):
        """プラグインロード"""
        # プラグインディレクトリ
        plugin_dir = Path(__file__).parent / "dashboard_plugins"
        if not plugin_dir.exists():
            plugin_dir.mkdir(parents=True, exist_ok=True)
            return
        
        # プラグインファイルをロード
        for plugin_file in plugin_dir.glob("*_plugin.py"):
            try:
                spec = importlib.util.spec_from_file_location(
                    plugin_file.stem, plugin_file
                )
                if spec and spec.loader:
                    plugin_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(plugin_module)
                    
                    # プラグインクラスを取得
                    if hasattr(plugin_module, 'Plugin'):
                        plugin_instance = plugin_module.Plugin()
                        self.plugins[plugin_file.stem.replace('_plugin', '')] = plugin_instance
                        self.logger.debug(f"プラグインロード: {plugin_file.stem}")
                        
            except Exception as e:
                self.logger.warning(f"プラグインロードエラー {plugin_file}: {e}")
    
    def create_tracker_config_template(self, tracker_id: str) -> str:
        """トラッカー設定テンプレート作成"""
        return self.config_manager.create_config_template(tracker_id)
    
    def list_available_configs(self) -> List[str]:
        """利用可能な設定一覧"""
        return self.config_manager.list_available_configs()


# 互換性維持のための関数
def create_unified_dashboard(tracker_id: str, extraction_dir: str, 
                           output_dir: str, **kwargs) -> str:
    """
    統合ダッシュボード生成の簡易インターフェース
    
    Args:
        tracker_id: トラッカーID
        extraction_dir: 抽出ディレクトリ
        output_dir: 出力ディレクトリ
        **kwargs: 追加設定
        
    Returns:
        str: 生成されたHTMLファイルパス
    """
    generator = UnifiedDashboardGenerator()
    dashboard_path = generator.generate_dashboard(
        tracker_id, extraction_dir, output_dir, kwargs
    )
    return str(dashboard_path)


# 使用例とテスト
def example_usage():
    """使用例"""
    generator = UnifiedDashboardGenerator()
    
    # 基本使用例
    dashboard_path = generator.generate_dashboard(
        tracker_id="QI-004",
        extraction_dir="/workspace/QI-004/extraction",
        output_dir="/workspace/QI-004"
    )
    print(f"ダッシュボード生成: {dashboard_path}")
    
    # 設定テンプレート作成
    template_path = generator.create_tracker_config_template("QI-007")
    print(f"設定テンプレート: {template_path}")
    
    # 利用可能設定一覧
    configs = generator.list_available_configs()
    print(f"設定一覧: {configs}")


if __name__ == "__main__":
    example_usage()