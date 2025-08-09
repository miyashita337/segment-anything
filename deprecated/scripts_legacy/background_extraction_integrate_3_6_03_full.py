#!/usr/bin/env python3
"""
INTEGRATE-3-6-03 全画像バックグラウンド抽出スクリプト
kana08データセット全26枚を対象とした重い処理対応版
"""

import json
import logging
import requests
import subprocess
import time
import threading
from datetime import datetime
from pathlib import Path


def setup_logging():
    """ログ設定"""
    log_file = Path("logs/INTEGRATE-3-6-03_full_extraction.log")
    log_file.parent.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_pushover_config():
    """Pushover設定読み込み"""
    config_path = Path("/mnt/c/AItools/segment-anything/config/pushover.json")
    if not config_path.exists():
        raise FileNotFoundError(f"Pushover設定が見つかりません: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def notify_success(message: str, title: str = "🔧 INTEGRATE-3-6-03"):
    """Pushover通知送信"""
    try:
        config = load_pushover_config()
        
        data = {
            'token': config['api_token'],
            'user': config['user_key'],
            'message': message,
            'title': title,
            'priority': 0  # 通常優先度
        }
        
        response = requests.post(
            "https://api.pushover.net/1/messages.json",
            data=data,
            timeout=30
        )
        
        return response.status_code == 200
        
    except Exception as e:
        logging.error(f"Pushover通知エラー: {e}")
        return False


def run_full_extraction(logger):
    """全画像抽出実行"""
    try:
        logger.info("🚀 INTEGRATE-3-6-03 全画像抽出開始")
        
        # 開始通知
        notify_success(
            "kana08データセット全画像抽出を開始しました。\n"
            "対象: 26枚の画像\n"
            "モード: yolov8x.pt復旧版\n"
            "完了まで10-15分程度お待ちください。",
            "🚀 バックグラウンド抽出開始"
        )
        
        # 抽出コマンド実行
        cmd = [
            "python3",
            "tools/core/sam_yolo_character_segment.py",
            "--mode", "reproduce-auto",
            "--input_dir", "/mnt/c/AItools/lora/train/yado/org/kana08/",
            "--output_dir", "/mnt/c/AItools/lora/train/yado/tracker-workspace/INTEGRATE-3-6-03/extraction/",
            "--score_threshold", "0.07",
            "--verbose"
        ]
        
        logger.info(f"実行コマンド: {' '.join(cmd)}")
        
        # プロセス開始
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # リアルタイムログ出力
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                logger.info(f"[抽出] {output.strip()}")
        
        # プロセス終了確認
        return_code = process.poll()
        
        if return_code == 0:
            logger.info("✅ 全画像抽出完了")
            return True
        else:
            logger.error(f"❌ 抽出処理エラー: return_code={return_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ 抽出処理例外: {e}")
        return False


def count_extraction_results(logger):
    """抽出結果カウント"""
    try:
        extraction_dir = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace/INTEGRATE-3-6-03/extraction")
        
        if not extraction_dir.exists():
            return 0, 0
        
        image_files = list(extraction_dir.glob("kana08_*.jpg"))
        total_input = 24  # cover画像2枚を除く実際の対象画像数
        
        logger.info(f"📊 抽出結果: {len(image_files)}/{total_input}枚")
        return len(image_files), total_input
        
    except Exception as e:
        logger.error(f"結果カウントエラー: {e}")
        return 0, 0


def send_completion_notification(success_count: int, total_count: int, logger):
    """完了通知送信"""
    success_rate = (success_count / total_count * 100) if total_count > 0 else 0
    
    if success_rate >= 90:
        status_emoji = "✅"
        status_text = "高品質完了"
    elif success_rate >= 70:
        status_emoji = "⚠️"
        status_text = "一部課題あり"
    else:
        status_emoji = "❌"
        status_text = "要改善"
    
    message = (
        f"{status_emoji} INTEGRATE-3-6-03 全画像抽出完了\n\n"
        f"📊 結果: {success_count}/{total_count}枚 ({success_rate:.1f}%)\n"
        f"🔧 モード: yolov8x.pt復旧版\n"
        f"⚖️ 閾値: confidence=0.07\n"
        f"📁 出力: tracker-workspace/INTEGRATE-3-6-03/\n\n"
        f"ステータス: {status_text}\n"
        f"次: 品質評価・分析準備完了"
    )
    
    title = f"{status_emoji} 全画像抽出{status_text}"
    
    notify_success(message, title)
    logger.info(f"📱 完了通知送信: {success_rate:.1f}%成功率")


def main():
    """メイン処理"""
    logger = setup_logging()
    start_time = datetime.now()
    
    try:
        logger.info("=" * 60)
        logger.info("INTEGRATE-3-6-03 kana08全画像バックグラウンド抽出開始")
        logger.info(f"開始時刻: {start_time}")
        logger.info("=" * 60)
        
        # 全画像抽出実行
        extraction_success = run_full_extraction(logger)
        
        # 結果確認
        success_count, total_count = count_extraction_results(logger)
        
        # 完了通知
        send_completion_notification(success_count, total_count, logger)
        
        # 最終ログ
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("=" * 60)
        logger.info(f"処理完了時刻: {end_time}")
        logger.info(f"処理時間: {duration}")
        logger.info(f"最終結果: {success_count}/{total_count}枚 ({success_count/total_count*100:.1f}%)")
        logger.info("=" * 60)
        
        return extraction_success
        
    except Exception as e:
        logger.error(f"メイン処理エラー: {e}")
        
        # エラー通知
        notify_success(
            f"❌ INTEGRATE-3-6-03抽出処理でエラーが発生しました。\n\n"
            f"エラー: {str(e)}\n"
            f"ログを確認してください。",
            "❌ 抽出処理エラー"
        )
        return False


if __name__ == "__main__":
    main()