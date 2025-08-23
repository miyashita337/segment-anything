"""
決定論的ダッシュボード生成システム v2.0

仕様:
- 完全決定論的出力保証（同一データに対してバイト完全一致）
- 仕様書ベース生成（config/dashboard_specification.yaml）
- 非決定要素の完全排除（datetime.now(), random等）
- 軽量HTML生成（Base64フリー、セキュリティ最適化）
- 統一URL形式: http://100.123.241.106:8088/tracker/{TRACKER_ID}
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
from datetime import datetime
import logging


class StandardDashboardGenerator:
    """標準ダッシュボード生成クラス - 決定論的バージョン"""
    
    def __init__(self):
        """初期化"""
        # 決定論的生成器を内部使用
        from .deterministic_dashboard import DeterministicDashboardGenerator
        self.deterministic_generator = DeterministicDashboardGenerator()
    
    def create_dashboard(self, tracker_id: str, workspace_dir: str, 
                        extraction_result_path: Optional[str] = None) -> str:
        """ダッシュボード作成（決定論的）
        
        Args:
            tracker_id: トラッカーID
            workspace_dir: ワークスペースディレクトリ
            extraction_result_path: 抽出結果JSONパス（オプション）
        
        Returns:
            生成されたダッシュボードファイルのパス
        """
        workspace_path = Path(workspace_dir)
        
        # 抽出結果JSONパス決定
        if extraction_result_path is None:
            extraction_result_path = workspace_path / "extraction_result.json"
        
        # 出力パス決定
        dashboard_dir = workspace_path / "dashboard"
        dashboard_file = dashboard_dir / "dashboard.html"
        
        # 決定論的生成器による生成
        return self.deterministic_generator.generate_dashboard(
            tracker_id=tracker_id,
            data_path=str(extraction_result_path),
            output_path=str(dashboard_file)
        )