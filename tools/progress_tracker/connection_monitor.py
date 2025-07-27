#!/usr/bin/env python3
"""
Google Sheets API 接続監視とユーザビリティ向上
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

from .config import get_default_config
from .sheets_client import GoogleSheetsClient
from .data_models import SheetsAPIError

logger = logging.getLogger(__name__)


@dataclass
class ConnectionHealth:
    """接続健全性情報"""
    timestamp: datetime
    success: bool
    response_time_ms: float
    error_message: Optional[str] = None


class ConnectionMonitor:
    """Google Sheets API接続監視クラス"""
    
    def __init__(self):
        self.config = get_default_config()
        self.health_history: List[ConnectionHealth] = []
        self.max_history = 100
    
    def check_connection(self) -> ConnectionHealth:
        """接続確認"""
        start_time = time.time()
        timestamp = datetime.now()
        
        try:
            client = GoogleSheetsClient(self.config)
            
            # 簡単なテスト（シート一覧取得）
            metadata = client.service.spreadsheets().get(
                spreadsheetId=self.config.spreadsheet_id
            ).execute()
            
            sheets = metadata.get('sheets', [])
            if not sheets:
                raise SheetsAPIError("シートが見つかりません")
            
            response_time = (time.time() - start_time) * 1000
            
            health = ConnectionHealth(
                timestamp=timestamp,
                success=True,
                response_time_ms=response_time
            )
            
            logger.info(f"接続確認成功: {response_time:.1f}ms")
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            
            health = ConnectionHealth(
                timestamp=timestamp,
                success=False,
                response_time_ms=response_time,
                error_message=str(e)
            )
            
            logger.warning(f"接続確認失敗: {e}")
        
        # 履歴に追加
        self.health_history.append(health)
        if len(self.health_history) > self.max_history:
            self.health_history.pop(0)
        
        return health
    
    def get_connection_summary(self) -> Dict[str, any]:
        """接続サマリー取得"""
        if not self.health_history:
            return {"status": "no_data"}
        
        recent_checks = self.health_history[-10:]
        success_count = sum(1 for check in recent_checks if check.success)
        
        if recent_checks:
            avg_response_time = sum(check.response_time_ms for check in recent_checks) / len(recent_checks)
            latest_check = recent_checks[-1]
        else:
            avg_response_time = 0
            latest_check = None
        
        return {
            "status": "healthy" if success_count >= 7 else "degraded" if success_count >= 3 else "unhealthy",
            "success_rate": success_count / len(recent_checks) * 100,
            "avg_response_time_ms": avg_response_time,
            "latest_check": latest_check,
            "total_checks": len(self.health_history),
            "recommendations": self._get_recommendations(recent_checks)
        }
    
    def _get_recommendations(self, recent_checks: List[ConnectionHealth]) -> List[str]:
        """推奨事項の生成"""
        recommendations = []
        
        if not recent_checks:
            return recommendations
        
        success_count = sum(1 for check in recent_checks if check.success)
        success_rate = success_count / len(recent_checks)
        
        if success_rate < 0.5:
            recommendations.append("🔧 API接続の問題が継続しています。認証設定を確認してください")
            recommendations.append("📋 手動でSpreadSheetを更新することを推奨します")
        
        elif success_rate < 0.8:
            recommendations.append("⚠️ API接続が不安定です。ネットワーク環境を確認してください")
        
        # レスポンス時間チェック
        avg_response = sum(check.response_time_ms for check in recent_checks) / len(recent_checks)
        if avg_response > 5000:  # 5秒以上
            recommendations.append("🐌 API応答時間が遅いです。処理タイミングを調整することを推奨します")
        
        # エラーパターン分析
        error_messages = [check.error_message for check in recent_checks if check.error_message]
        if error_messages:
            if any("Unable to parse range" in msg for msg in error_messages):
                recommendations.append("📝 シート名の設定を確認してください")
            if any("403" in msg or "401" in msg for msg in error_messages):
                recommendations.append("🔑 認証の更新が必要な可能性があります")
        
        return recommendations


def show_user_friendly_status():
    """ユーザーフレンドリーな状況表示"""
    print("\n" + "="*60)
    print("📊 SpreadSheet API 接続状況")
    print("="*60)
    
    monitor = ConnectionMonitor()
    health = monitor.check_connection()
    summary = monitor.get_connection_summary()
    
    # 状況表示
    status_icons = {
        "healthy": "✅",
        "degraded": "⚠️",
        "unhealthy": "❌",
        "no_data": "❓"
    }
    
    status = summary.get("status", "no_data")
    icon = status_icons.get(status, "❓")
    
    print(f"{icon} 接続状況: {status}")
    
    if "success_rate" in summary:
        print(f"📈 成功率: {summary['success_rate']:.1f}%")
        print(f"⚡ 平均応答時間: {summary['avg_response_time_ms']:.1f}ms")
    
    # 最新チェック結果
    if health.success:
        print(f"🌐 最新接続: ✅ 成功 ({health.response_time_ms:.1f}ms)")
    else:
        print(f"🌐 最新接続: ❌ 失敗")
        print(f"   エラー: {health.error_message}")
    
    # 推奨事項
    recommendations = summary.get("recommendations", [])
    if recommendations:
        print(f"\n💡 推奨事項:")
        for rec in recommendations:
            print(f"   {rec}")
    
    # SpreadSheet URL
    config = get_default_config()
    print(f"\n📋 SpreadSheet URL:")
    print(f"   https://docs.google.com/spreadsheets/d/{config.spreadsheet_id}/edit")
    
    print("="*60)
    
    return health.success


if __name__ == "__main__":
    show_user_friendly_status()