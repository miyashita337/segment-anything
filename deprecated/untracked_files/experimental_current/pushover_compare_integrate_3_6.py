from features.common.environment_manager import (
    get_path,
    get_test_image_path,
    is_ci_environment,
    setup_test_env,
)

#!/usr/bin/env python3
"""
INTG-046-03/04 モデル比較結果送信
"""

import json
import requests
import time
from pathlib import Path


def load_pushover_config():
    config_path = Path(get_path("data", Path(get_path("data", Path("/mnt/c/AItools/segment-anything/config/pushover.json").relative_to("/mnt/c/AItools/"))).relative_to("/mnt/c/AItools/")))
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def send_comparison_results():
    """比較結果送信"""
    config = load_pushover_config()
    
    # 結果カウント
    dir_03 = Path(get_path("output", "INTG-046-03/extraction"))
    dir_04 = Path(get_path("output", "INTG-046-04/extraction"))
    
    files_03 = list(dir_03.glob("kana08_*.jpg")) if dir_03.exists() else []
    files_04 = list(dir_04.glob("kana08_*.jpg")) if dir_04.exists() else []
    
    total = 24
    count_03 = len(files_03)
    count_04 = len(files_04)
    rate_03 = count_03 / total * 100
    rate_04 = count_04 / total * 100
    
    # 判定
    if rate_04 > rate_03:
        verdict = "✅ アニメ特化版が優秀"
        emoji = "🏆"
    elif rate_04 < rate_03:
        verdict = "⚠️ 汎用版が優秀"
        emoji = "⚠️"
    else:
        verdict = "🤝 同等の性能"
        emoji = "🤝"
    
    message = (
        f"{emoji} モデル比較検証完了\n\n"
        f"📊 抽出成功率比較:\n"
        f"・INTG-046-04 (アニメ特化意図): {count_04}/{total}枚 ({rate_04:.1f}%)\n"
        f"・INTG-046-03 (汎用版): {count_03}/{total}枚 ({rate_03:.1f}%)\n\n"
        f"📈 差分: {abs(rate_04 - rate_03):.1f}%\n"
        f"🎯 判定: {verdict}\n\n"
        f"⚠️ 注意事項:\n"
        f"・3-6-04は--anime_yoloフラグ使用\n"
        f"・実際にはyolov8x.ptが使用された模様\n"
        f"・yolov8x6_animeface.ptは未ロード\n\n"
        f"次: 個別画像での品質詳細確認"
    )
    
    data = {
        'token': config['api_token'],
        'user': config['user_key'],
        'message': message,
        'title': f'{emoji} モデル比較: {verdict}',
        'priority': 1,
        'sound': 'pushover'
    }
    
    response = requests.post(
        "https://api.pushover.net/1/messages.json",
        data=data,
        timeout=30
    )
    
    print(f"📱 比較結果送信: {response.status_code}")
    
    # サンプル画像送信（3-6-04から3枚）
    sample_images = sorted(dir_04.glob("kana08_*.jpg"))[:3]
    
    for i, img_path in enumerate(sample_images, 1):
        with open(img_path, 'rb') as f:
            files = {'attachment': (img_path.name, f, 'image/jpeg')}
            
            data = {
                'token': config['api_token'],
                'user': config['user_key'],
                'message': (
                    f"🔬 3-6-04サンプル {i}/3\n"
                    f"画像: {img_path.name}\n"
                    f"意図: yolov8x6_animeface.pt使用\n"
                    f"実際: yolov8x.pt使用（--anime_yolo）\n\n"
                    f"品質確認ポイント:\n"
                    f"・キャラクター境界精度\n"
                    f"・背景除去品質"
                ),
                'title': f'🔬 サンプル [{i}/3] {img_path.name}',
                'priority': 0,
                'sound': 'none'
            }
            
            response = requests.post(
                "https://api.pushover.net/1/messages.json",
                data=data,
                files=files,
                timeout=60
            )
            
            print(f"✅ サンプル送信 [{i}/3]: {img_path.name}")
            time.sleep(1)


if __name__ == "__main__":
    send_comparison_results()