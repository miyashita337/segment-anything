#!/usr/bin/env python3
"""
Pushover画像送信スクリプト - Ubuntu環境対応
高品質抽出結果をPushoverに送信
"""

import requests
import json
import os
from pathlib import Path

# Pushover設定読み込み
with open('config/pushover.json', 'r') as f:
    config = json.load(f)

def send_image_to_pushover(image_path, message='', title='QC成功版完全再現'):
    """単一画像をPushoverに送信"""
    url = 'https://api.pushover.net/1/messages.json'
    
    data = {
        'token': config['api_token'],
        'user': config['user_key'],
        'message': message,
        'title': title
    }
    
    try:
        with open(image_path, 'rb') as f:
            files = {'attachment': f}
            response = requests.post(url, data=data, files=files)
        
        return response.status_code == 200, response.text
    except Exception as e:
        return False, str(e)

def main():
    """高品質画像10枚をPushoverに送信"""
    extraction_dir = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace/QC-KANA08-RESTORED-080402/extraction")
    
    # 高品質・多様性を考慮した画像選択
    selected_images = [
        # QC成功版基準画像
        ("extracted_kana08_0002.png", "QC基準: 604×1166完全再現"),
        # 高スコア画像
        ("extracted_kana08_0021.png", "最高スコア: 0.998 (1358×766)"),
        ("extracted_kana08_0001.png", "高スコア: 0.997 (902×782)"),
        ("extracted_kana08_0013.png", "高スコア: 0.995 (851×980)"),
        # 大サイズ画像
        ("extracted_kana08_0011.png", "大サイズ: 1489×1698 (0.973)"),
        ("extracted_kana08_0023.png", "大サイズ: 1354×1560 (0.974)"),
        ("extracted_kana08_0010.png", "大サイズ: 1166×968 (0.976)"),
        # 多様なサイズ・ポーズ
        ("extracted_kana08_0008.png", "縦長: 663×1334 (0.996)"),
        ("extracted_kana08_0017.png", "縦長: 909×1638 (0.964)"),
        ("extracted_kana08_0018.png", "横長: 1371×794 (0.982)"),
    ]
    
    print("🚀 Ubuntu環境Pushover送信開始")
    print(f"   対象: {len(selected_images)}枚の高品質画像")
    
    success_count = 0
    
    for i, (filename, description) in enumerate(selected_images, 1):
        image_path = extraction_dir / filename
        
        if not image_path.exists():
            print(f"❌ [{i}/{len(selected_images)}] ファイル不存在: {filename}")
            continue
        
        print(f"📤 [{i}/{len(selected_images)}] 送信中: {filename}")
        
        message = f"Ubuntu環境移行成功! {description}"
        success, result = send_image_to_pushover(image_path, message)
        
        if success:
            success_count += 1
            print(f"✅ 送信成功: {description}")
        else:
            print(f"❌ 送信失敗: {result[:100]}")
    
    print(f"\n📊 送信完了: {success_count}/{len(selected_images)}枚成功")
    print("🎯 Ubuntu環境でのQC成功版完全再現をPushoverで確認可能")

if __name__ == "__main__":
    main()