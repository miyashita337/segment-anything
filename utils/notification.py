"""
通知システム（Pushover）
"""

import json
import requests
import os
from pathlib import Path
from typing import Optional, Dict, Any


class PushoverNotifier:
    """Pushover通知クライアント"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: 設定ファイルパス。Noneの場合はデフォルト位置を使用
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "pushover.json"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.api_url = "https://api.pushover.net/1/messages.json"
    
    def _load_config(self) -> Optional[Dict[str, Any]]:
        """設定ファイルを読み込み"""
        try:
            if not self.config_path.exists():
                print(f"⚠️ Pushover設定ファイルが見つかりません: {self.config_path}")
                print(f"   {self.config_path}.example をコピーして設定してください")
                return None
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 必須項目チェック
            required_fields = ['token', 'user']
            for field in required_fields:
                if not config.get(field) or config[field] == f"your_{field}_here" or config[field] == f"your_{field}_key_here":
                    print(f"⚠️ Pushover設定の{field}が設定されていません")
                    return None
            
            return config
            
        except Exception as e:
            print(f"❌ Pushover設定読み込みエラー: {e}")
            return None
    
    def send_notification(self, message: str, title: Optional[str] = None, priority: int = 0) -> bool:
        """
        Pushover通知を送信
        
        Args:
            message: 通知メッセージ
            title: 通知タイトル（Noneの場合は設定ファイルのデフォルトを使用）
            priority: 優先度 (-2: 最低, -1: 低, 0: 通常, 1: 高, 2: 緊急)
        
        Returns:
            bool: 送信成功かどうか
        """
        if not self.config:
            print("⚠️ Pushover設定が無効なため通知をスキップします")
            return False
        
        try:
            # 通知データ作成
            data = {
                "token": self.config["token"],
                "user": self.config["user"],
                "message": message,
                "title": title or self.config.get("title", "Character Extraction"),
                "priority": priority
            }
            
            # デバイス指定がある場合
            if self.config.get("device"):
                data["device"] = self.config["device"]
            
            # API送信
            response = requests.post(self.api_url, data=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == 1:
                    print("📱 Pushover通知送信成功")
                    return True
                else:
                    print(f"❌ Pushover API エラー: {result.get('errors', 'Unknown error')}")
                    return False
            else:
                print(f"❌ Pushover HTTP エラー: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Pushover通信エラー: {e}")
            return False
        except Exception as e:
            print(f"❌ Pushover送信エラー: {e}")
            return False
    
    def send_batch_complete(self, successful: int, total: int, failed: int, total_time: float) -> bool:
        """
        バッチ処理完了通知を送信
        
        Args:
            successful: 成功数
            total: 総数
            failed: 失敗数
            total_time: 処理時間（秒）
        
        Returns:
            bool: 送信成功かどうか
        """
        success_rate = (successful / total * 100) if total > 0 else 0
        
        message = f"""🎯 バッチ処理完了

📊 結果:
   成功: {successful}/{total} ({success_rate:.1f}%)
   失敗: {failed}
   処理時間: {total_time:.1f}秒

⚡ 1画像あたり平均: {total_time/total:.1f}秒"""
        
        # 成功率に応じて優先度設定
        if success_rate >= 90:
            priority = 0  # 通常
            title = "✅ キャラクター抽出完了"
        elif success_rate >= 70:
            priority = 0  # 通常
            title = "⚠️ キャラクター抽出完了（一部失敗）"
        else:
            priority = 1  # 高
            title = "❌ キャラクター抽出完了（多数失敗）"
        
        return self.send_notification(message, title, priority)


def send_batch_notification(successful: int, total: int, failed: int, total_time: float) -> bool:
    """
    バッチ処理完了通知の簡易送信関数
    
    Args:
        successful: 成功数
        total: 総数
        failed: 失敗数
        total_time: 処理時間（秒）
    
    Returns:
        bool: 送信成功かどうか
    """
    notifier = PushoverNotifier()
    return notifier.send_batch_complete(successful, total, failed, total_time)


# 使用例
if __name__ == "__main__":
    # テスト通知
    notifier = PushoverNotifier()
    notifier.send_notification("テスト通知", "Character Extraction Test")