#!/usr/bin/env python3
"""
QI-006: GPT-5評価結果のPushover通知
24枚の画像それぞれにGPT-5評価とその理由を添付して個別送信
"""

import base64
import json
import requests
import sys
import time
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from features.common.notification.pushover_image_sender import (
    load_pushover_config,
    send_pushover_with_image,
)


def load_gpt5_evaluation_results():
    """GPT-5評価結果を読み込み"""
    results_file = Path(
        "/mnt/c/AItools/lora/train/yado/tracker-workspace/QI-006/quality/gpt5_lora_quality_evaluation.json"
    )

    if not results_file.exists():
        print(f"❌ GPT-5評価結果が見つかりません: {results_file}")
        return None

    with open(results_file, "r", encoding="utf-8") as f:
        return json.load(f)


def get_grade_emoji(grade):
    """グレードに対応する絵文字を返す"""
    emoji_map = {"A": "🏆", "B": "✅", "C": "⚠️", "D": "❌", "F": "🚫"}
    return emoji_map.get(grade, "❓")


def format_evaluation_message(result):
    """評価結果をメッセージフォーマットに変換"""
    image_name = result.get("image_name", "N/A")
    grade = result.get("grade", "N/A")
    emoji = get_grade_emoji(grade)

    # 評価詳細
    person_count = result.get("person_count", "不明")
    extraction_quality = result.get("extraction_quality", "不明")
    brightness = result.get("brightness", "不明")
    lora_suitability = result.get("lora_suitability", "不明")
    reason = result.get("detailed_reason", "評価理由なし")
    suggestions = result.get("improvement_suggestions", "なし")

    # メッセージ作成
    title = f"{emoji} GPT-5評価: {image_name} - グレード{grade}"

    message = f"""📊 GPT-5 LoRA品質評価結果

🎯 グレード: {grade}
👤 人物数: {person_count}
✂️ 抽出品質: {extraction_quality}
💡 明度: {brightness}
🎨 LoRA適合性: {lora_suitability}

📝 評価理由:
{reason}

💡 改善提案:
{suggestions}

ファイルサイズ: {result.get('file_size', 0):,} bytes"""

    return title, message


def send_evaluation_with_image(result, image_path):
    """評価結果と画像をPushoverで送信"""
    title, message = format_evaluation_message(result)

    try:
        # 画像ファイルの存在確認
        if not image_path.exists():
            print(f"⚠️ 画像ファイルが見つかりません: {image_path}")
            # 画像なしで送信（テキストのみ）
            config = load_pushover_config()
            if not config:
                return False

            url = "https://api.pushover.net/1/messages.json"
            data = {
                "token": config["api_token"],
                "user": config["user_key"],
                "title": title,
                "message": message,
                "priority": 0,
            }
            response = requests.post(url, data=data, timeout=30)
            success = response.status_code == 200
        else:
            # 画像付きで送信
            success = send_pushover_with_image(
                title=title, message=message, image_path=str(image_path), priority=0
            )

        if success:
            grade = result.get("grade", "N/A")
            emoji = get_grade_emoji(grade)
            print(f"  {emoji} 送信成功: {result.get('image_name')} - グレード{grade}")
        else:
            print(f"  ❌ 送信失敗: {result.get('image_name')}")

        return success

    except Exception as e:
        print(f"  ❌ エラー: {result.get('image_name')} - {e}")
        return False


def send_summary_notification(gpt5_data):
    """全体サマリーを送信"""
    summary = gpt5_data["evaluation_summary"]
    grades = gpt5_data["grade_distribution"]["grade_distribution"]

    title = "📊 GPT-5評価完了サマリー (QI-006)"

    message = f"""🎯 GPT-5 LoRA品質評価 全24枚完了

📈 評価結果分布:
🏆 A評価: {grades.get('A', 0)}枚 - LoRA学習に最適
✅ B評価: {grades.get('B', 0)}枚 - 適している
⚠️ C評価: {grades.get('C', 0)}枚 - 注意必要
❌ D評価: {grades.get('D', 0)}枚 - 問題あり
🚫 F評価: {grades.get('F', 0)}枚 - 使用不可

📊 統計:
• 総評価画像: {summary['total_images']}枚
• 評価成功: {summary['successful_evaluations']}枚
• 成功率: {summary['successful_evaluations']/summary['total_images']*100:.1f}%

💡 推奨事項:
• LoRA学習推奨: {grades.get('A', 0) + grades.get('B', 0)}枚
• 要改善: {grades.get('C', 0) + grades.get('D', 0) + grades.get('F', 0)}枚

🌐 ダッシュボード:
http://100.123.241.106:8088/tracker/QI-006

評価実行時刻: {summary.get('evaluation_timestamp', 'N/A')}"""

    # テキストのみ送信
    config = load_pushover_config()
    if not config:
        print("❌ Pushover設定が読み込めません")
        return False

    url = "https://api.pushover.net/1/messages.json"
    data = {
        "token": config["api_token"],
        "user": config["user_key"],
        "title": title,
        "message": message,
        "priority": 1,  # 高優先度
    }

    try:
        response = requests.post(url, data=data, timeout=30)
        success = response.status_code == 200

        if success:
            print(f"✅ サマリー送信成功")
        else:
            print(f"❌ サマリー送信失敗: {response.text}")

        return success
    except Exception as e:
        print(f"❌ サマリー送信エラー: {e}")
        return False


def main():
    """メイン実行関数"""
    print("📱 QI-006: GPT-5評価結果Pushover通知開始")
    print("=" * 60)

    # GPT-5評価結果読み込み
    gpt5_data = load_gpt5_evaluation_results()
    if not gpt5_data:
        return False

    # 画像ディレクトリ
    extraction_dir = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace/QI-006/extraction")

    # 詳細結果を取得
    detailed_results = gpt5_data.get("detailed_results", [])

    # 成功した評価のみをフィルタ
    successful_results = [r for r in detailed_results if r.get("status") == "success"]

    print(f"📊 送信対象: {len(successful_results)}枚の評価結果")
    print(f"📱 開始時刻: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("")

    # 個別送信カウンター
    success_count = 0
    failed_count = 0

    # 各画像の評価結果を個別送信
    for i, result in enumerate(successful_results, 1):
        image_name = result.get("image_name", "")
        image_path = extraction_dir / image_name

        print(f"📤 送信中 {i}/{len(successful_results)}: {image_name}")

        # 評価結果と画像を送信
        if send_evaluation_with_image(result, image_path):
            success_count += 1
        else:
            failed_count += 1

        # レート制限対策（1秒待機）
        if i < len(successful_results):
            time.sleep(1)

    print("")
    print("-" * 60)

    # サマリー送信
    print("📊 サマリー送信中...")
    send_summary_notification(gpt5_data)

    # 最終統計
    print("")
    print("=" * 60)
    print(f"✅ 送信完了統計:")
    print(f"   成功: {success_count}件")
    print(f"   失敗: {failed_count}件")
    print(f"   合計: {success_count + failed_count}件")
    print(f"📱 終了時刻: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    return success_count > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
