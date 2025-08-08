#!/usr/bin/env python3
"""
INTEGRATE-3-6-02抽出結果をPushoverに送信
品質劣化診断Phase 2の結果確認用（修正版）
"""

import json
import requests
from pathlib import Path
import logging
import glob
import os

def setup_logging():
    """ログ設定"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_pushover_config():
    """Pushover設定読み込み"""
    config_path = Path("config/pushover.json")
    if not config_path.exists():
        raise FileNotFoundError(f"Pushover設定ファイルが存在しません: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def send_image_to_pushover(image_path: str, title: str, message: str, config: dict, logger) -> bool:
    """個別画像をPushoverに送信"""
    try:
        with open(image_path, 'rb') as image_file:
            response = requests.post(
                "https://api.pushover.net/1/messages.json",
                data={
                    "token": config["api_token"],
                    "user": config["user_key"],
                    "title": title,
                    "message": message
                },
                files={"attachment": image_file},
                timeout=30
            )
        
        if response.status_code == 200:
            logger.info(f"✅ 送信成功: {Path(image_path).name}")
            return True
        else:
            logger.error(f"❌ 送信失敗: {Path(image_path).name} - Status: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ エラー: {Path(image_path).name} - {str(e)}")
        return False

def main():
    logger = setup_logging()
    
    try:
        # Pushover設定読み込み
        config = load_pushover_config()
        
        # 抽出結果ディレクトリ
        extraction_dir = "/mnt/c/AItools/lora/train/yado/tracker-workspace/INTEGRATE-3-6-02/extraction"
        
        # 画像ファイル取得
        image_files = glob.glob(os.path.join(extraction_dir, "kana08_*.jpg"))
        image_files.sort()
        
        if not image_files:
            logger.error("送信する画像が見つかりません")
            return
        
        # 10枚に制限
        selected_images = image_files[:10]
        
        logger.info(f"📤 INTEGRATE-3-6-02 品質修正結果をPushoverに送信開始")
        logger.info(f"📊 対象画像数: {len(selected_images)}")
        
        success_count = 0
        
        for i, image_path in enumerate(selected_images, 1):
            filename = Path(image_path).name
            title = f"INTEGRATE-3-6-02 Phase2品質修正 ({i}/10)"
            message = f"問題コード復元版\nファイル: {filename}\n安定システム使用"
            
            if send_image_to_pushover(image_path, title, message, config, logger):
                success_count += 1
        
        logger.info(f"📤 送信完了: {success_count}/{len(selected_images)} 枚成功")
        
    except Exception as e:
        logger.error(f"❌ 実行エラー: {str(e)}")

if __name__ == "__main__":
    main()