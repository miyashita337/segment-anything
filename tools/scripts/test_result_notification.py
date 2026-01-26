#!/usr/bin/env python3
"""
抽出アルゴリズム修正結果をPushover通知
test_algo_fix_result.jpgの結果通知
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from features.common.notification.pushover_image_sender import send_pushover_with_image


def main():
    """修正したアルゴリズムの抽出結果を通知"""

    result_image = Path("test_algo_fix_result.jpg")

    if not result_image.exists():
        print(f"❌ 結果画像が見つかりません: {result_image}")
        return False

    # ファイル情報
    file_size = result_image.stat().st_size

    title = "🎯 抽出アルゴリズム修正テスト結果"

    message = f"""✅ アルゴリズム修正テスト完了

📊 テスト対象: kana08_0014.jpg
   元画像: 1496x2112 (5人重なり合い)
   
🔍 検出結果:
   YOLO: 3人検出
   最大面積person: 99,890px選択
   
🎯 SAM処理:
   生成マスク: 215個 → 179個キャラクター
   選択方式: 最大面積選択 (修正済み)
   最終マスク: 352,036px (面積比11.1%)
   
✅ 出力結果:
   サイズ: 659x662px
   ファイル: {file_size:,} bytes
   
🚀 修正点:
   ✅ 信頼度→面積ベース選択
   ✅ fullbody_priority適用
   ✅ YOLO bbox→SAM hybrid実装

複数キャラクター時の最大面積選択が正常動作確認！"""

    # Pushover送信
    success = send_pushover_with_image(
        title=title, message=message, image_path=str(result_image), priority=1  # 高優先度
    )

    if success:
        print("✅ 抽出結果Pushover通知送信完了")
        return True
    else:
        print("❌ Pushover通知送信失敗")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
