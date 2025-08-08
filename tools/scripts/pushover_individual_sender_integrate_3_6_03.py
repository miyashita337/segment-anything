#!/usr/bin/env python3
"""
INTEGRATE-3-6-03 個別画像Pushover送信スクリプト
抽出結果19枚を1枚ずつ順次送信して詳細品質確認を行う
"""

import json
import logging
import requests
from pathlib import Path
import time
from datetime import datetime


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


def get_file_size_mb(file_path: Path) -> float:
    """ファイルサイズ(MB)取得"""
    return file_path.stat().st_size / (1024 * 1024)


def send_individual_image(image_path: Path, config: dict, logger: logging.Logger, index: int, total: int):
    """個別画像をPushoverに送信"""
    try:
        file_size_mb = get_file_size_mb(image_path)
        
        with open(image_path, 'rb') as f:
            files = {'attachment': (image_path.name, f, 'image/jpeg')}
            
            # 詳細メッセージ作成
            message = (
                f"📊 INTEGRATE-3-6-03 品質チェック\n"
                f"画像: {image_path.name}\n"
                f"順序: {index}/{total}\n\n"
                f"🔧 抽出設定:\n"
                f"・YOLO: yolov8x.pt (復旧版)\n"
                f"・閾値: confidence=0.07\n"
                f"・SAM: 精密セグメンテーション\n\n"
                f"📏 ファイル情報:\n"
                f"・サイズ: {file_size_mb:.2f}MB\n"
                f"・形式: JPEG\n\n"
                f"🎯 確認ポイント:\n"
                f"・キャラクター境界の精度\n"
                f"・背景除去の品質\n"
                f"・手足切断の回避状況"
            )
            
            data = {
                'token': config['api_token'],
                'user': config['user_key'],
                'message': message,
                'title': f'🔍 品質確認 [{index}/{total}] {image_path.name}',
                'priority': 0,
                'sound': 'none' if index > 1 else 'pushover'  # 最初だけ音あり
            }
            
            response = requests.post(
                "https://api.pushover.net/1/messages.json",
                data=data,
                files=files,
                timeout=60  # 画像送信のためタイムアウト延長
            )
            
            if response.status_code == 200:
                logger.info(f"✅ 送信成功 [{index}/{total}]: {image_path.name} ({file_size_mb:.2f}MB)")
                return True
            else:
                logger.error(f"❌ 送信失敗 [{index}/{total}]: {image_path.name} - Status: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
                
    except Exception as e:
        logger.error(f"❌ 送信エラー [{index}/{total}]: {image_path.name} - {e}")
        return False


def main():
    """メイン処理"""
    logger = setup_logging()
    start_time = datetime.now()
    
    logger.info("🚀 INTEGRATE-3-6-03 個別画像Pushover送信開始")
    
    try:
        # 設定読み込み
        config = load_pushover_config()
        logger.info("✅ Pushover設定読み込み完了")
        
        # 抽出結果ディレクトリ
        extraction_dir = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace/INTEGRATE-3-6-03/extraction")
        
        if not extraction_dir.exists():
            raise FileNotFoundError(f"抽出ディレクトリが見つかりません: {extraction_dir}")
        
        # 全画像ファイル取得（ソート済み）
        image_files = sorted(extraction_dir.glob("kana08_*.jpg"))
        total_images = len(image_files)
        
        logger.info(f"📸 送信対象: {total_images}枚の画像")
        
        if total_images == 0:
            logger.warning("⚠️ 送信対象画像が見つかりません")
            return
        
        # 開始通知
        start_message = (
            f"🔍 INTEGRATE-3-6-03 個別品質チェック開始\n\n"
            f"対象: {total_images}枚の抽出画像\n"
            f"送信間隔: 2秒\n"
            f"予想時間: 約{total_images * 2 // 60}分\n\n"
            f"各画像の品質を順次確認してください。"
        )
        
        requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                'token': config['api_token'],
                'user': config['user_key'],
                'message': start_message,
                'title': '🚀 個別品質チェック開始',
                'priority': 1,  # 高優先度
                'sound': 'pushover'
            },
            timeout=30
        )
        
        # 個別送信処理
        success_count = 0
        failed_images = []
        
        for i, image_path in enumerate(image_files, 1):
            logger.info(f"📤 送信準備: {i}/{total_images} - {image_path.name}")
            
            if send_individual_image(image_path, config, logger, i, total_images):
                success_count += 1
            else:
                failed_images.append(image_path.name)
            
            # 送信間隔（2秒待機、レート制限対策）
            if i < total_images:
                logger.info(f"⏳ 待機中... (2秒)")
                time.sleep(2)
        
        # 完了通知
        end_time = datetime.now()
        duration = end_time - start_time
        
        completion_message = (
            f"✅ INTEGRATE-3-6-03 個別送信完了\n\n"
            f"📊 送信結果:\n"
            f"・成功: {success_count}/{total_images}枚\n"
            f"・失敗: {len(failed_images)}枚\n"
            f"・成功率: {success_count/total_images*100:.1f}%\n\n"
            f"⏱️ 処理時間: {duration}\n\n"
            f"🎯 次のステップ:\n"
            f"・各画像の品質を確認\n"
            f"・問題のある画像を特定\n"
            f"・必要に応じて再処理検討"
        )
        
        if failed_images:
            completion_message += f"\n\n❌ 送信失敗画像:\n" + "\n".join(failed_images)
        
        requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                'token': config['api_token'],
                'user': config['user_key'],
                'message': completion_message,
                'title': f'🎯 個別送信完了 ({success_count}/{total_images})',
                'priority': 1,
                'sound': 'magic'
            },
            timeout=30
        )
        
        logger.info("=" * 60)
        logger.info(f"✅ 全送信完了: {success_count}/{total_images}枚成功")
        logger.info(f"⏱️ 処理時間: {duration}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ メイン処理エラー: {e}")
        
        # エラー通知
        try:
            config = load_pushover_config()
            requests.post(
                "https://api.pushover.net/1/messages.json",
                data={
                    'token': config['api_token'],
                    'user': config['user_key'],
                    'message': f"❌ 個別送信処理でエラーが発生:\n{str(e)}",
                    'title': '❌ 送信処理エラー',
                    'priority': 2,  # 緊急
                    'sound': 'siren'
                },
                timeout=30
            )
        except:
            pass
        
        raise


if __name__ == "__main__":
    main()