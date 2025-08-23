"""
統合ダッシュボード プラグインシステム

プラグインベースでダッシュボード機能を拡張
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class DashboardPlugin(ABC):
    """ダッシュボードプラグインの基底クラス"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """プラグイン名"""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """プラグインバージョン"""
        pass
    
    @abstractmethod
    def execute(self, dashboard_data: Dict[str, Any], 
               plugin_settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        プラグイン実行
        
        Args:
            dashboard_data: ダッシュボードデータ
            plugin_settings: プラグイン設定
            
        Returns:
            Dict[str, Any]: 更新されたダッシュボードデータ
        """
        pass