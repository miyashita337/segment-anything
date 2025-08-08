#!/usr/bin/env python3
"""
Pushover Image Attachment Module
抽出された全ての画像をPushover通知に添付して送信

Usage:
    from features.common.notification.pushover_image_sender import send_extraction_complete_with_images
    
    send_extraction_complete_with_images(
        title="QI-005抽出完了",
        extraction_dir="/path/to/extraction/",
        successful=20,
        total=25
    )
"""

import json
import os
import requests
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import glob


def load_pushover_config() -> Optional[Dict[str, Any]]:
    """Pushover設定を読み込み"""
    possible_paths = [
        "/mnt/c/AItools/segment-anything/config/pushover.json",
        "/mnt/c/AItools/manga-character-extractor-api/config/pushover_config.json",
        os.path.expanduser("~/.pushover.json"),
        "./pushover.json",
        "./config/pushover.json"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # segment-anything形式
                if 'api_token' in config and 'user_key' in config:
                    return {
                        'api_token': config['api_token'],
                        'user_key': config['user_key'],
                        'enabled': config.get('enabled', True)
                    }
                
                # manga-character-extractor-api形式
                if 'pushover' in config:
                    pushover_config = config['pushover']
                    return {
                        'api_token': pushover_config.get('api_token'),
                        'user_key': pushover_config.get('user_key'),
                        'enabled': pushover_config.get('enabled', True)
                    }
                    
            except Exception as e:
                print(f"⚠️ 設定ファイル読み込みエラー ({path}): {e}")
                continue
    
    return None


def get_extracted_images(extraction_dir: str) -> List[str]:
    """抽出ディレクトリから全ての画像ファイルを取得"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']
    image_files = []
    
    for ext in image_extensions:
        pattern = os.path.join(extraction_dir, ext)
        image_files.extend(glob.glob(pattern))
        # サブディレクトリも検索
        pattern_recursive = os.path.join(extraction_dir, '**', ext)
        image_files.extend(glob.glob(pattern_recursive, recursive=True))
    
    # 重複除去とソート
    image_files = sorted(list(set(image_files)))
    
    print(f"📸 検出された画像: {len(image_files)}枚")
    for img_path in image_files:
        print(f"  - {os.path.basename(img_path)}")
    
    return image_files


def send_pushover_with_image(title: str, message: str, image_path: str, 
                           priority: int = 0, sound: str = "pushover") -> bool:
    """単一画像を添付してPushover通知を送信"""
    config = load_pushover_config()
    if not config or not config.get('enabled'):
        print("⚠️ Pushover設定が無効またはありません")
        return False
    
    api_token = config.get('api_token')
    user_key = config.get('user_key')
    
    if not api_token or not user_key:
        print("⚠️ Pushover APIトークンまたはユーザーキーが設定されていません")
        return False
    
    if not os.path.exists(image_path):
        print(f"⚠️ 画像ファイルが見つかりません: {image_path}")
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
    
    try:
        with open(image_path, 'rb') as image_file:
            files = {"attachment": image_file}
            response = requests.post(url, data=data, files=files, timeout=30)
        
        if response.status_code == 200:
            print(f"✅ 画像付き通知送信成功: {os.path.basename(image_path)}")
            return True
        else:
            print(f"❌ 画像付き通知送信失敗: HTTP {response.status_code}")
            try:
                error_info = response.json()
                print(f"   エラー詳細: {error_info}")
            except:
                pass
            return False
            
    except Exception as e:
        print(f"❌ 画像付き通知送信エラー: {e}")
        return False


def send_extraction_complete_with_images(title: str, extraction_dir: str,
                                       successful: int = 0, total: int = 0,
                                       failed: int = 0, duration: float = 0.0) -> bool:
    """
    抽出完了通知を全ての画像を添付して送信
    
    Args:
        title: 通知タイトル
        extraction_dir: 抽出された画像があるディレクトリ
        successful: 成功数
        total: 総数
        failed: 失敗数
        duration: 処理時間（秒）
    
    Returns:
        bool: 送信成功フラグ
    """
    print(f"\n🚀 画像付きPushover通知送信開始")
    print(f"📁 抽出ディレクトリ: {extraction_dir}")
    
    # 画像ファイル一覧取得
    image_files = get_extracted_images(extraction_dir)
    
    if not image_files:
        print("⚠️ 抽出された画像が見つかりません。テキスト通知のみ送信します。")
        # テキストのみの通知にフォールバック
        from features.common.notification.global_pushover import notify_process_complete
        return notify_process_complete(title, successful, total, failed, duration)
    
    success_rate = (successful / total * 100) if total > 0 else 0
    
    # 最初の画像に詳細な統計情報を添付
    first_message = f"""📊 {title} - 抽出完了報告

✅ 成功: {successful}/{total} ({success_rate:.1f}%)
❌ 失敗: {failed}
⏱️ 処理時間: {duration:.1f}秒

📸 抽出画像: {len(image_files)}枚
全ての抽出画像を添付します"""
    
    sent_count = 0
    
    # 最初の画像: 詳細統計付き
    if len(image_files) > 0:
        success = send_pushover_with_image(
            title=f"{title} (1/{len(image_files)})",
            message=first_message,
            image_path=image_files[0],
            priority=0 if success_rate >= 50 else 1,
            sound="magic" if success_rate >= 80 else "pushover"
        )
        if success:
            sent_count += 1
        
        # 少し待機（API制限対策）
        time.sleep(1)
    
    # 残りの画像: シンプルな情報
    for i, image_path in enumerate(image_files[1:], 2):
        image_name = os.path.basename(image_path)
        simple_message = f"{title}\n画像 {i}/{len(image_files)}: {image_name}"
        
        success = send_pushover_with_image(
            title=f"{title} ({i}/{len(image_files)})",
            message=simple_message,
            image_path=image_path,
            priority=0,
            sound="none"  # 2枚目以降は音なし
        )
        
        if success:
            sent_count += 1
        
        # API制限対策: 1秒間隔
        time.sleep(1)
    
    print(f"\n📊 通知送信結果:")
    print(f"   送信成功: {sent_count}/{len(image_files)}枚")
    print(f"   成功率: {(sent_count/len(image_files)*100):.1f}%")
    
    return sent_count == len(image_files)


def test_image_notification():
    """画像付き通知のテスト"""
    print("🧪 画像付きPushover通知テスト開始...")
    
    config = load_pushover_config()
    if not config:
        print("❌ Pushover設定が見つかりません")
        return False
    
    # テスト用の小さな画像を作成
    test_dir = "/tmp/test_pushover_single"
    os.makedirs(test_dir, exist_ok=True)
    
    # 実際に抽出された画像があれば使用
    sample_images = [
        "/mnt/c/AItools/lora/train/yado/tracker-workspace/QI-005/extraction/kana08_0001.jpg",
        "/mnt/c/AItools/lora/train/yado/org/kana08/kana08_0001.jpg"
    ]
    
    test_image = None
    for img_path in sample_images:
        if os.path.exists(img_path):
            test_image = img_path
            break
    
    if not test_image:
        print("❌ テスト用画像が見つかりません")
        return False
    
    print(f"📸 テスト画像: {test_image}")
    
    # 単一画像テスト
    result = send_pushover_with_image(
        title="テスト通知 (画像付き)",
        message="Global Pushover画像送信モジュールのテストです",
        image_path=test_image
    )
    
    if result:
        print("✅ 画像付きテスト通知送信成功")
    else:
        print("❌ 画像付きテスト通知送信失敗")
    
    return result


if __name__ == "__main__":
    test_image_notification()