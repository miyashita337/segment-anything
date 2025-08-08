#!/usr/bin/env python3
"""
INTEGRATE-3-6-03 Pushover送信スクリプト
kana08抽出結果を10枚送信して品質確認を行う
"""

import json
import logging
import requests
from pathlib import Path
import time


def setup_logging():
    """ログ設定"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_pushover_config():
    """Pushover設定読み込み"""
    config_path = Path("/mnt/c/AItools/segment-anything/config/pushover.json")
    if not config_path.exists():
        raise FileNotFoundError(f"Pushover設定が見つかりません: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def send_image_to_pushover(image_path: Path, config: dict, logger: logging.Logger, index: int):
    """Pushoverに画像送信"""
    try:
        with open(image_path, 'rb') as f:
            files = {'attachment': (image_path.name, f, 'image/jpeg')}
            data = {
                'token': config['api_token'],
                'user': config['user_key'],
                'message': f'INTEGRATE-3-6-03 抽出結果 {index+1}/10: {image_path.name}\n'
                          f'YOLO復旧テスト: yolov8x.pt (confidence=0.07)\n'
                          f'品質チェック: アニメキャラクター抽出精度確認',
                'title': f'🔍 INTEGRATE-3-6-03 品質確認 ({index+1}/10)'
            }
            
            response = requests.post(
                "https://api.pushover.net/1/messages.json",
                data=data,
                files=files,
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info(f"✅ 送信成功: {image_path.name}")
                return True
            else:
                logger.error(f"❌ 送信失敗: {image_path.name} - Status: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
                
    except Exception as e:
        logger.error(f"❌ 送信エラー: {image_path.name} - {e}")
        return False


def main():
    """メイン処理"""
    logger = setup_logging()
    logger.info("🚀 INTEGRATE-3-6-03 Pushover送信開始")
    
    try:
        # 設定読み込み
        config = load_pushover_config()
        logger.info("✅ Pushover設定読み込み完了")
        
        # 抽出結果ディレクトリ
        extraction_dir = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace/INTEGRATE-3-6-03/extraction")
        
        if not extraction_dir.exists():
            raise FileNotFoundError(f"抽出ディレクトリが見つかりません: {extraction_dir}")
        
        # 画像ファイル取得（最初の10枚）
        image_files = sorted(extraction_dir.glob("kana08_*.jpg"))[:10]
        
        if len(image_files) < 10:
            logger.warning(f"⚠️ 画像数が不足: {len(image_files)}枚 (期待値: 10枚)")
        
        logger.info(f"📸 送信対象: {len(image_files)}枚の画像")
        
        # 送信処理
        success_count = 0
        for i, image_path in enumerate(image_files):
            logger.info(f"📤 送信中: {i+1}/{len(image_files)} - {image_path.name}")
            
            if send_image_to_pushover(image_path, config, logger, i):
                success_count += 1
            
            # レート制限対策（1秒待機）
            if i < len(image_files) - 1:
                time.sleep(1)
        
        # 結果報告
        logger.info(f"🎯 送信完了: {success_count}/{len(image_files)}枚成功")
        
        if success_count == len(image_files):
            logger.info("✅ 全画像送信完了 - Pushoverでkana08抽出品質を確認してください")
        else:
            logger.warning(f"⚠️ 一部送信失敗: {len(image_files) - success_count}枚失敗")
            
    except Exception as e:
        logger.error(f"❌ 処理エラー: {e}")
        raise


if __name__ == "__main__":
    main()