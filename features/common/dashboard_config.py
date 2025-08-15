"""
統合ダッシュボード設定管理システム

統合ダッシュボード生成のための設定管理を提供
- トラッカー固有設定
- プラグイン設定
- UI/UX設定
- 画像表示オプション
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
import logging


@dataclass
class ImageDisplayConfig:
    """画像表示設定"""
    method: str = "path_reference"  # "base64" or "path_reference"
    max_size_mb: float = 0.1        # path_reference時のHTMLファイル最大サイズ
    thumbnail_size: tuple = (400, 400)
    quality_compression: int = 85
    lazy_loading: bool = True


@dataclass
class QualityAnalysisConfig:
    """品質解析設定"""
    enable_image_analysis: bool = True
    enable_statistical_analysis: bool = True
    enable_graph_generation: bool = True
    quality_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'high': 0.8,
        'medium': 0.6, 
        'low': 0.3
    })


@dataclass
class LayoutConfig:
    """レイアウト設定"""
    template: str = "standard"      # "standard", "compact", "detailed"
    grid_columns: int = 3
    enable_responsive: bool = True
    custom_css: Optional[str] = None


@dataclass
class PluginConfig:
    """プラグイン設定"""
    enabled_plugins: List[str] = field(default_factory=list)
    plugin_settings: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class DashboardConfig:
    """統合ダッシュボード設定"""
    tracker_id: str
    title: Optional[str] = None
    description: Optional[str] = None
    
    # コンポーネント設定
    image_display: ImageDisplayConfig = field(default_factory=ImageDisplayConfig)
    quality_analysis: QualityAnalysisConfig = field(default_factory=QualityAnalysisConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    plugins: PluginConfig = field(default_factory=PluginConfig)
    
    # 入出力パス
    extraction_dir: Optional[str] = None
    output_dir: Optional[str] = None
    
    # 追加設定
    custom_settings: Dict[str, Any] = field(default_factory=dict)


class DashboardConfigManager:
    """ダッシュボード設定管理"""
    
    def __init__(self, config_dir: Optional[str] = None):
        """初期化"""
        self.logger = logging.getLogger(__name__)
        
        # 設定ディレクトリ
        if config_dir is None:
            self.config_dir = Path(__file__).parent.parent.parent / "config" / "dashboard"
        else:
            self.config_dir = Path(config_dir)
        
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # デフォルト設定ファイルパス
        self.default_config_path = self.config_dir / "default_dashboard_config.yaml"
        self.tracker_configs_dir = self.config_dir / "tracker_specific"
        self.tracker_configs_dir.mkdir(parents=True, exist_ok=True)
        
        # デフォルト設定を作成（存在しない場合）
        self._ensure_default_config()
    
    def load_config(self, tracker_id: str) -> DashboardConfig:
        """
        トラッカー設定をロード
        
        Args:
            tracker_id: トラッカーID
            
        Returns:
            DashboardConfig: ダッシュボード設定
        """
        try:
            # デフォルト設定をロード
            default_config = self._load_default_config()
            
            # トラッカー固有設定をロード（存在する場合）
            tracker_config_path = self.tracker_configs_dir / f"{tracker_id.lower()}.yaml"
            if tracker_config_path.exists():
                tracker_overrides = self._load_yaml_config(tracker_config_path)
                # デフォルト設定にトラッカー設定をオーバーライド
                merged_config = self._merge_configs(default_config, tracker_overrides)
            else:
                merged_config = default_config
            
            # DashboardConfigオブジェクトに変換
            dashboard_config = self._dict_to_config(merged_config, tracker_id)
            
            self.logger.info(f"設定ロード完了: {tracker_id}")
            return dashboard_config
            
        except Exception as e:
            self.logger.error(f"設定ロードエラー {tracker_id}: {e}")
            # フォールバック: デフォルト設定
            return DashboardConfig(tracker_id=tracker_id)
    
    def save_tracker_config(self, tracker_id: str, config: DashboardConfig):
        """
        トラッカー固有設定を保存
        
        Args:
            tracker_id: トラッカーID
            config: ダッシュボード設定
        """
        try:
            config_dict = self._config_to_dict(config)
            config_path = self.tracker_configs_dir / f"{tracker_id.lower()}.yaml"
            
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
            
            self.logger.info(f"トラッカー設定保存: {config_path}")
            
        except Exception as e:
            self.logger.error(f"設定保存エラー {tracker_id}: {e}")
            raise
    
    def list_available_configs(self) -> List[str]:
        """利用可能な設定一覧を取得"""
        configs = []
        for config_file in self.tracker_configs_dir.glob("*.yaml"):
            configs.append(config_file.stem.upper())
        return sorted(configs)
    
    def create_config_template(self, tracker_id: str) -> str:
        """
        設定テンプレートを作成
        
        Args:
            tracker_id: トラッカーID
            
        Returns:
            str: 作成された設定ファイルパス
        """
        template_config = {
            'title': f'{tracker_id} 品質評価ダッシュボード',
            'description': f'{tracker_id}トラッカーの品質評価結果',
            'image_display': {
                'method': 'path_reference',
                'max_size_mb': 0.1,
                'lazy_loading': True
            },
            'quality_analysis': {
                'enable_image_analysis': True,
                'enable_statistical_analysis': True,
                'enable_graph_generation': True
            },
            'layout': {
                'template': 'standard',
                'grid_columns': 3,
                'enable_responsive': True
            },
            'plugins': {
                'enabled_plugins': ['image_quality', 'statistics']
            }
        }
        
        config_path = self.tracker_configs_dir / f"{tracker_id.lower()}.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(template_config, f, default_flow_style=False,
                     allow_unicode=True, indent=2)
        
        self.logger.info(f"設定テンプレート作成: {config_path}")
        return str(config_path)
    
    def _ensure_default_config(self):
        """デフォルト設定ファイルを確実に作成"""
        if not self.default_config_path.exists():
            default_config = {
                'title': 'Quality Evaluation Dashboard',
                'description': 'Image extraction quality evaluation results',
                'image_display': {
                    'method': 'path_reference',
                    'max_size_mb': 0.1,
                    'thumbnail_size': [400, 400],
                    'quality_compression': 85,
                    'lazy_loading': True
                },
                'quality_analysis': {
                    'enable_image_analysis': True,
                    'enable_statistical_analysis': True,
                    'enable_graph_generation': True,
                    'quality_thresholds': {
                        'high': 0.8,
                        'medium': 0.6,
                        'low': 0.3
                    }
                },
                'layout': {
                    'template': 'standard',
                    'grid_columns': 3,
                    'enable_responsive': True,
                    'custom_css': None
                },
                'plugins': {
                    'enabled_plugins': ['image_quality', 'statistics'],
                    'plugin_settings': {}
                }
            }
            
            with open(self.default_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False,
                         allow_unicode=True, indent=2)
            
            self.logger.info(f"デフォルト設定作成: {self.default_config_path}")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """デフォルト設定をロード"""
        return self._load_yaml_config(self.default_config_path)
    
    def _load_yaml_config(self, config_path: Path) -> Dict[str, Any]:
        """YAML設定ファイルをロード"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _merge_configs(self, base_config: Dict, override_config: Dict) -> Dict:
        """設定をマージ（再帰的）"""
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def _dict_to_config(self, config_dict: Dict[str, Any], tracker_id: str) -> DashboardConfig:
        """辞書からDashboardConfigオブジェクトに変換"""
        try:
            # 各コンポーネント設定を構築
            image_display = ImageDisplayConfig(**config_dict.get('image_display', {}))
            quality_analysis = QualityAnalysisConfig(**config_dict.get('quality_analysis', {}))
            layout = LayoutConfig(**config_dict.get('layout', {}))
            plugins = PluginConfig(**config_dict.get('plugins', {}))
            
            return DashboardConfig(
                tracker_id=tracker_id,
                title=config_dict.get('title'),
                description=config_dict.get('description'),
                image_display=image_display,
                quality_analysis=quality_analysis,
                layout=layout,
                plugins=plugins,
                extraction_dir=config_dict.get('extraction_dir'),
                output_dir=config_dict.get('output_dir'),
                custom_settings=config_dict.get('custom_settings', {})
            )
            
        except Exception as e:
            self.logger.error(f"設定変換エラー: {e}")
            return DashboardConfig(tracker_id=tracker_id)
    
    def _config_to_dict(self, config: DashboardConfig) -> Dict[str, Any]:
        """DashboardConfigから辞書に変換"""
        return {
            'title': config.title,
            'description': config.description,
            'image_display': {
                'method': config.image_display.method,
                'max_size_mb': config.image_display.max_size_mb,
                'thumbnail_size': list(config.image_display.thumbnail_size),
                'quality_compression': config.image_display.quality_compression,
                'lazy_loading': config.image_display.lazy_loading
            },
            'quality_analysis': {
                'enable_image_analysis': config.quality_analysis.enable_image_analysis,
                'enable_statistical_analysis': config.quality_analysis.enable_statistical_analysis,
                'enable_graph_generation': config.quality_analysis.enable_graph_generation,
                'quality_thresholds': config.quality_analysis.quality_thresholds
            },
            'layout': {
                'template': config.layout.template,
                'grid_columns': config.layout.grid_columns,
                'enable_responsive': config.layout.enable_responsive,
                'custom_css': config.layout.custom_css
            },
            'plugins': {
                'enabled_plugins': config.plugins.enabled_plugins,
                'plugin_settings': config.plugins.plugin_settings
            },
            'extraction_dir': config.extraction_dir,
            'output_dir': config.output_dir,
            'custom_settings': config.custom_settings
        }


# 使用例とテスト関数
def example_usage():
    """使用例"""
    config_manager = DashboardConfigManager()
    
    # QI-004の設定をロード
    qi004_config = config_manager.load_config("QI-004")
    print(f"QI-004設定: {qi004_config.title}")
    
    # 新しいトラッカー用の設定テンプレート作成
    template_path = config_manager.create_config_template("QI-007")
    print(f"テンプレート作成: {template_path}")


if __name__ == "__main__":
    example_usage()