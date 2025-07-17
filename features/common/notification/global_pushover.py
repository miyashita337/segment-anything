#!/usr/bin/env python3
"""
Global Pushover Notification Module
どのプログラムからでも利用可能なPushover通知システム

Usage:
    from utils.global_pushover import notify_success, notify_error, notify_process_complete
    
    notify_success("処理完了", "バッチ処理が正常に完了しました")
    notify_error("エラー発生", "予期しないエラーが発生しました")
    notify_process_complete("抽出完了", successful=50, total=100, duration=120.5)
"""

import os
import json
import requests
import time
from pathlib import Path
from typing import Optional, Dict, Any

def find_pushover_config() -> Optional[str]:
    """Pushover設定ファイルを自動検出"""
    possible_paths = [
        # segment-anything プロジェクト内
        "/mnt/c/AItools/segment-anything/config/pushover.json",
        # manga-character-extractor-api プロジェクト内（後方互換）
        "/mnt/c/AItools/manga-character-extractor-api/config/pushover_config.json",
        # ホームディレクトリ
        os.path.expanduser("~/.pushover.json"),
        # 現在のディレクトリ
        "./pushover.json",
        "./config/pushover.json"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

def load_pushover_config() -> Optional[Dict[str, Any]]:
    """Pushover設定を読み込み"""
    config_path = find_pushover_config()
    if not config_path:
        return None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # segment-anything形式の設定
        if 'token' in config and 'user' in config:
            return {
                'api_token': config['token'],
                'user_key': config['user'],
                'enabled': True
            }
        
        # manga-character-extractor-api形式の設定
        if 'pushover' in config:
            pushover_config = config['pushover']
            return {
                'api_token': pushover_config.get('api_token'),
                'user_key': pushover_config.get('user_key'),
                'enabled': pushover_config.get('enabled', True)
            }
        
        return None
        
    except Exception as e:
        print(f"⚠️ Pushover設定の読み込みに失敗: {e}")
        return None

def send_pushover_notification(title: str, message: str, priority: int = 0, 
                              sound: str = "pushover", expire: int = 3600, 
                              retry: int = 60) -> bool:
    """
    Pushover通知を送信
    
    Args:
        title: 通知タイトル
        message: 通知メッセージ
        priority: 優先度 (-2: 最低, -1: 低, 0: 通常, 1: 高, 2: 緊急)
        sound: 通知音
        expire: 緊急通知の有効期限 (秒)
        retry: 緊急通知の再試行間隔 (秒)
        
    Returns:
        bool: 送信成功フラグ
    """
    config = load_pushover_config()
    if not config or not config.get('enabled'):
        return False
    
    api_token = config.get('api_token')
    user_key = config.get('user_key')
    
    if not api_token or not user_key:
        print("⚠️ Pushover APIトークンまたはユーザーキーが設定されていません")
        return False
    
    url = "https://api.pushover.net/1/messages.json"
    
    data = {
        "token": api_token,
        "user": user_key,
        "title": title,
        "message": message,
        "priority": priority,
        "sound": sound
    }
    
    # 緊急通知の場合は追加パラメータ
    if priority == 2:
        data.update({
            "expire": expire,
            "retry": retry
        })
    
    try:
        response = requests.post(url, data=data, timeout=10)
        
        if response.status_code == 200:
            return True
        else:
            print(f"⚠️ Pushover通知送信失敗: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Pushover通知送信エラー: {e}")
        return False

def notify_success(title: str = "処理完了", message: str = "処理が正常に完了しました") -> bool:
    """成功通知"""
    return send_pushover_notification(title, message, priority=0, sound="magic")

def notify_error(title: str = "エラー発生", message: str = "処理中にエラーが発生しました") -> bool:
    """エラー通知"""
    return send_pushover_notification(title, message, priority=1, sound="siren")

def notify_warning(title: str = "警告", message: str = "注意が必要な状況が発生しました") -> bool:
    """警告通知"""
    return send_pushover_notification(title, message, priority=0, sound="falling")

def notify_process_complete(title: str = "バッチ処理完了", 
                          successful: int = 0, total: int = 0, 
                          failed: int = 0, duration: float = 0.0) -> bool:
    """バッチ処理完了通知"""
    success_rate = (successful / total * 100) if total > 0 else 0
    
    message = f"""📊 処理結果:
✅ 成功: {successful}/{total} ({success_rate:.1f}%)
❌ 失敗: {failed}
⏱️ 処理時間: {duration:.1f}秒

🎉 バッチ処理が完了しました！"""
    
    # 成功率に応じて優先度を調整
    priority = 0 if success_rate >= 50 else 1
    sound = "magic" if success_rate >= 80 else "pushover"
    
    return send_pushover_notification(title, message, priority=priority, sound=sound)

def notify_long_process_start(title: str = "長時間処理開始", message: str = "処理を開始しました") -> bool:
    """長時間処理開始通知"""
    return send_pushover_notification(title, message, priority=0, sound="cosmic")

def notify_critical_error(title: str = "重大エラー", message: str = "重大なエラーが発生しました") -> bool:
    """重大エラー通知（緊急）"""
    return send_pushover_notification(title, message, priority=2, sound="siren", expire=3600, retry=60)

# 使用例とテスト関数
def test_notifications():
    """通知テスト"""
    print("🧪 Pushover通知テスト開始...")
    
    config = load_pushover_config()
    if not config:
        print("❌ Pushover設定が見つかりません")
        return False
    
    print(f"✅ 設定ファイル検出: {find_pushover_config()}")
    print(f"📱 APIトークン: {config['api_token'][:8]}...")
    print(f"👤 ユーザーキー: {config['user_key'][:8]}...")
    
    # テスト通知送信
    result = notify_success("テスト通知", "Global Pushover モジュールのテストです")
    
    if result:
        print("✅ テスト通知送信成功")
    else:
        print("❌ テスト通知送信失敗")
    
    return result

if __name__ == "__main__":
    test_notifications()