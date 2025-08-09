#!/usr/bin/env python3
"""
INTEGRATE-3-6-04 全画像バックグラウンド抽出スクリプト
yolov8x6_animeface.ptモデルでのkana08データセット処理
汎用モデル(yolov8x.pt)との比較検証用
"""

import json
import logging
import requests
import subprocess
import time
from datetime import datetime
from pathlib import Path


def setup_logging():
    """ログ設定"""
    log_file = Path("logs/INTEGRATE-3-6-04_animeface_extraction.log")
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


def notify_success(message: str, title: str = "🔬 INTEGRATE-3-6-04", priority: int = 0):
    """Pushover通知送信"""
    try:
        config = load_pushover_config()
        
        data = {
            'token': config['api_token'],
            'user': config['user_key'],
            'message': message,
            'title': title,
            'priority': priority
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


def run_animeface_extraction(logger):
    """yolov8x6_animeface.ptでの全画像抽出実行"""
    try:
        logger.info("🚀 INTEGRATE-3-6-04 アニメ特化モデル抽出開始")
        
        # 開始通知
        notify_success(
            "🔬 モデル比較検証開始\n\n"
            "対象: kana08データセット全画像\n"
            "モデル: yolov8x6_animeface.pt（アニメ特化版）\n"
            "比較対象: INTEGRATE-3-6-03（yolov8x.pt汎用版）\n\n"
            "処理時間: 約10-15分\n"
            "完了後、品質比較結果をお送りします。",
            "🔬 アニメ特化モデル検証開始",
            priority=1
        )
        
        # 出力ディレクトリ作成
        output_dir = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace/INTEGRATE-3-6-04/extraction")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 抽出コマンド実行（--verboseオプション削除、--anime_yoloオプション追加）
        cmd = [
            "python3",
            "tools/core/sam_yolo_character_segment.py",
            "--mode", "reproduce-auto",
            "--input_dir", "/mnt/c/AItools/lora/train/yado/org/kana08/",
            "--output_dir", str(output_dir),
            "--score_threshold", "0.07",
            "--anime_yolo"  # アニメ特化モデル使用を明示的に指定
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
            logger.info("✅ アニメ特化モデル抽出完了")
            return True
        else:
            logger.error(f"❌ 抽出処理エラー: return_code={return_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ 抽出処理例外: {e}")
        return False


def count_and_compare_results(logger):
    """抽出結果カウントと比較"""
    try:
        # INTEGRATE-3-6-04（アニメ特化版）
        anime_dir = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace/INTEGRATE-3-6-04/extraction")
        anime_files = list(anime_dir.glob("kana08_*.jpg")) if anime_dir.exists() else []
        
        # INTEGRATE-3-6-03（汎用版）
        general_dir = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace/INTEGRATE-3-6-03/extraction")
        general_files = list(general_dir.glob("kana08_*.jpg")) if general_dir.exists() else []
        
        total_input = 24  # cover画像を除く実際の対象画像数
        
        anime_count = len(anime_files)
        general_count = len(general_files)
        
        logger.info(f"📊 比較結果:")
        logger.info(f"  アニメ特化版(3-6-04): {anime_count}/{total_input}枚 ({anime_count/total_input*100:.1f}%)")
        logger.info(f"  汎用版(3-6-03): {general_count}/{total_input}枚 ({general_count/total_input*100:.1f}%)")
        
        return anime_count, general_count, total_input
        
    except Exception as e:
        logger.error(f"結果カウントエラー: {e}")
        return 0, 0, 0


def send_comparison_notification(anime_count: int, general_count: int, total_count: int, logger):
    """比較結果通知送信"""
    anime_rate = (anime_count / total_count * 100) if total_count > 0 else 0
    general_rate = (general_count / total_count * 100) if total_count > 0 else 0
    
    # 判定
    if anime_rate > general_rate:
        verdict = "✅ アニメ特化版が優秀"
        emoji = "🏆"
    elif anime_rate < general_rate:
        verdict = "⚠️ 汎用版が優秀"
        emoji = "⚠️"
    else:
        verdict = "🤝 同等の性能"
        emoji = "🤝"
    
    message = (
        f"{emoji} モデル比較検証完了\n\n"
        f"📊 抽出成功率比較:\n"
        f"・yolov8x6_animeface.pt: {anime_count}/{total_count}枚 ({anime_rate:.1f}%)\n"
        f"・yolov8x.pt: {general_count}/{total_count}枚 ({general_rate:.1f}%)\n\n"
        f"📈 差分: {abs(anime_rate - general_rate):.1f}%\n"
        f"🎯 判定: {verdict}\n\n"
        f"📁 出力ディレクトリ:\n"
        f"・3-6-04: アニメ特化版結果\n"
        f"・3-6-03: 汎用版結果（比較用）\n\n"
        f"次: 個別画像での品質詳細確認"
    )
    
    title = f"{emoji} モデル比較: {verdict}"
    
    notify_success(message, title, priority=1)
    logger.info(f"📱 比較結果通知送信完了")


def send_sample_images(logger):
    """サンプル画像送信（最初の5枚）"""
    try:
        config = load_pushover_config()
        extraction_dir = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace/INTEGRATE-3-6-04/extraction")
        
        if not extraction_dir.exists():
            return
        
        sample_images = sorted(extraction_dir.glob("kana08_*.jpg"))[:5]
        
        for i, image_path in enumerate(sample_images, 1):
            try:
                with open(image_path, 'rb') as f:
                    files = {'attachment': (image_path.name, f, 'image/jpeg')}
                    
                    message = (
                        f"🔬 アニメ特化版サンプル {i}/5\n"
                        f"画像: {image_path.name}\n"
                        f"モデル: yolov8x6_animeface.pt\n\n"
                        f"品質確認ポイント:\n"
                        f"・キャラクター境界の精度\n"
                        f"・アニメ特有の表現への対応\n"
                        f"・髪飾りや装飾品の検出"
                    )
                    
                    data = {
                        'token': config['api_token'],
                        'user': config['user_key'],
                        'message': message,
                        'title': f'🔬 サンプル [{i}/5] {image_path.name}',
                        'priority': 0,
                        'sound': 'none'
                    }
                    
                    response = requests.post(
                        "https://api.pushover.net/1/messages.json",
                        data=data,
                        files=files,
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        logger.info(f"✅ サンプル送信成功 [{i}/5]: {image_path.name}")
                    
                    time.sleep(1)  # レート制限対策
                    
            except Exception as e:
                logger.error(f"サンプル送信エラー: {e}")
                
    except Exception as e:
        logger.error(f"サンプル送信処理エラー: {e}")


def main():
    """メイン処理"""
    logger = setup_logging()
    start_time = datetime.now()
    
    try:
        logger.info("=" * 60)
        logger.info("INTEGRATE-3-6-04 アニメ特化モデル検証開始")
        logger.info(f"開始時刻: {start_time}")
        logger.info("=" * 60)
        
        # アニメ特化モデルで全画像抽出実行
        extraction_success = run_animeface_extraction(logger)
        
        # 結果確認と比較
        anime_count, general_count, total_count = count_and_compare_results(logger)
        
        # 比較結果通知
        send_comparison_notification(anime_count, general_count, total_count, logger)
        
        # サンプル画像送信
        if anime_count > 0:
            logger.info("サンプル画像送信開始...")
            send_sample_images(logger)
        
        # 最終ログ
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("=" * 60)
        logger.info(f"処理完了時刻: {end_time}")
        logger.info(f"処理時間: {duration}")
        logger.info(f"最終結果: アニメ特化版 {anime_count}/{total_count}枚 vs 汎用版 {general_count}/{total_count}枚")
        logger.info("=" * 60)
        
        return extraction_success
        
    except Exception as e:
        logger.error(f"メイン処理エラー: {e}")
        
        # エラー通知
        notify_success(
            f"❌ INTEGRATE-3-6-04処理でエラーが発生しました。\n\n"
            f"エラー: {str(e)}\n"
            f"ログを確認してください。",
            "❌ アニメ特化モデル検証エラー",
            priority=2
        )
        return False


if __name__ == "__main__":
    main()