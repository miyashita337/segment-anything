#!/usr/bin/env python3
"""
Pushover通知テストスクリプト
"""

import json
import requests
from pathlib import Path

def test_pushover():
    # 設定読み込み
    config_path = Path('/mnt/c/AItools/segment-anything/config/pushover.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Pushover設定読み込み完了")
    print(f"API Token: {config['api_token'][:10]}...")
    print(f"User Key: {config['user_key'][:10]}...")
    
    # サンプル画像パス
    sample_image = "/mnt/c/AItools/lora/train/yado/tracker-workspace/workspace/QC-KANA08/extracted_kana08_0001.png"
    
    # 通知メッセージ
    message = """🎉 QC抽出完了通知（テスト）
    
📊 処理結果:
• KANA08: 26/26枚成功
• KANA05: 39/39枚成功
• KANA07: 42/42枚成功

✅ 総計: 107枚（100%成功率）
⏱️ 処理時間: 1.1分

Pushover通知が正常に動作しています！"""
    
    # 通知送信
    data = {
        'token': config['api_token'],
        'user': config['user_key'],
        'title': '🎯 QC抽出完了',
        'message': message,
        'priority': 0
    }
    
    files = {}
    if Path(sample_image).exists():
        print(f"サンプル画像添付: {sample_image}")
        files['attachment'] = open(sample_image, 'rb')
    
    print("Pushover API送信中...")
    response = requests.post(
        'https://api.pushover.net/1/messages.json',
        data=data,
        files=files
    )
    
    if files:
        files['attachment'].close()
    
    print(f"ステータスコード: {response.status_code}")
    print(f"レスポンス: {response.text}")
    
    if response.status_code == 200:
        print("✅ Pushover通知送信成功！")
    else:
        print(f"❌ Pushover通知失敗: {response.status_code}")
        print(f"詳細: {response.text}")

if __name__ == "__main__":
    test_pushover()