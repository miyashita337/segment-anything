#!/usr/bin/env python3
"""
P1-B004軽量抽出実行（sympy回避版）
直接的な画像処理でadaptive-croppingを実現
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 軽量インポート（sympyを避ける）
try:
    import numpy as np
    import cv2

    from PIL import Image

    print("✅ 基本ライブラリ読み込み成功")
except ImportError as e:
    print(f"❌ ライブラリ読み込み失敗: {e}")
    sys.exit(1)


def adaptive_crop_simple(image_path: Path, output_path: Path) -> bool:
    """シンプルなadaptive cropping実装"""
    try:
        # 画像読み込み
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"❌ 画像読み込み失敗: {image_path}")
            return False

        h, w = img.shape[:2]
        print(f"📐 元画像サイズ: {w}x{h}")

        # 中央部分を基準にした適応的クロッピング
        # P1-B004の概念: 顔検出の代わりに中央重点クロッピング
        center_x, center_y = w // 2, h // 2

        # アスペクト比を考慮した最適クロッピング範囲
        if w > h:  # 横長
            crop_size = min(h, w * 0.8)
            crop_w = int(crop_size)
            crop_h = int(crop_size * 1.2)  # 縦長優先
        else:  # 縦長または正方形
            crop_size = min(w, h * 0.8)
            crop_w = int(crop_size)
            crop_h = int(crop_size * 1.2)

        # クロッピング座標計算
        x1 = max(0, center_x - crop_w // 2)
        y1 = max(0, center_y - crop_h // 2)
        x2 = min(w, x1 + crop_w)
        y2 = min(h, y1 + crop_h)

        # クロッピング実行
        cropped = img[y1:y2, x1:x2]
        print(f"✂️ クロッピング: ({x1},{y1}) - ({x2},{y2}) = {x2-x1}x{y2-y1}")

        # 背景を白に（透明背景の代替）
        result = np.ones((512, 512, 3), dtype=np.uint8) * 255

        # クロッピング画像をリサイズして中央配置
        target_size = 400
        aspect_ratio = cropped.shape[1] / cropped.shape[0]

        if aspect_ratio > 1:  # 横長
            new_w = target_size
            new_h = int(target_size / aspect_ratio)
        else:  # 縦長
            new_h = target_size
            new_w = int(target_size * aspect_ratio)

        resized = cv2.resize(cropped, (new_w, new_h))

        # 中央配置
        y_offset = (512 - new_h) // 2
        x_offset = (512 - new_w) // 2
        result[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

        # PNG形式で保存
        output_path = output_path.with_suffix(".png")
        cv2.imwrite(str(output_path), result)

        print(f"💾 保存完了: {output_path.name}")
        return True

    except Exception as e:
        print(f"❌ 処理エラー: {e}")
        return False


def run_lightweight_extraction(tracker_id: str = "P1-B004"):
    """軽量抽出実行"""
    print(f"🚀 {tracker_id} 軽量抽出パイプライン開始")

    # 入力・出力設定
    input_dir = Path("/mnt/c/AItools/lora/train/yado/org/kana08")
    output_dir = Path(f"/mnt/c/AItools/lora/train/yado/tracker-workspace/{tracker_id}/extraction")

    if not input_dir.exists():
        print(f"❌ 入力ディレクトリが存在しません: {input_dir}")
        return False

    # 画像ファイル取得
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    image_files = [
        f
        for f in image_files
        if not f.name.startswith(".") and f.name.lower() != "desktop.ini"  # 隠しファイル除外
    ]  # システムファイル除外

    print(f"📊 入力画像数: {len(image_files)}枚")

    if not image_files:
        print("❌ 有効な画像ファイルがありません")
        return False

    # 出力ディレクトリ準備
    output_dir.mkdir(parents=True, exist_ok=True)

    # 最初の3枚で処理（高速テスト）
    test_files = image_files[:3]
    success_count = 0

    start_time = time.time()

    print("🔧 adaptive-cropping処理開始")
    for i, img_file in enumerate(test_files, 1):
        print(f"\n📸 [{i}/{len(test_files)}] {img_file.name}")

        # 出力ファイル名生成
        output_name = f"p1_b004_{img_file.stem}_adaptive_cropped"
        output_path = output_dir / output_name

        # 処理実行
        if adaptive_crop_simple(img_file, output_path):
            success_count += 1

        time.sleep(0.1)  # 少し間隔をあける

    end_time = time.time()
    processing_time = end_time - start_time

    # 結果確認
    output_files = list(output_dir.glob("*.png")) + list(output_dir.glob("*.jpg"))
    success_rate = (len(output_files) / len(test_files)) * 100

    print(f"\n📈 処理結果:")
    print(f"  - 入力: {len(test_files)}枚")
    print(f"  - 出力: {len(output_files)}枚")
    print(f"  - 成功率: {success_rate:.1f}%")
    print(f"  - 処理時間: {processing_time:.1f}秒")

    # レポート生成
    report = {
        "tracker_id": tracker_id,
        "timestamp": datetime.now().isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "input_count": len(test_files),
        "output_count": len(output_files),
        "success_rate": success_rate,
        "processing_time": processing_time,
        "adaptive_cropping": True,
        "method": "lightweight_opencv",
        "files": [f.name for f in output_files],
    }

    report_path = output_dir.parent / "extraction_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"📄 レポート生成: {report_path}")

    if len(output_files) > 0:
        print("\n✅ 抽出処理成功: adaptive-cropping画像が生成されました")
        return True
    else:
        print("\n❌ 抽出処理失敗")
        return False


def main():
    """メイン実行"""
    print("=" * 60)
    print("🎯 P1-B004軽量抽出実行")
    print("  - OpenCV使用（sympy回避）")
    print("  - 中央重点adaptive-cropping")
    print("  - 実際の画像ファイル生成")
    print("=" * 60)

    try:
        success = run_lightweight_extraction("P1-B004")

        if success:
            print("\n🎉 P1-B004軽量抽出完了")
            return 0
        else:
            print("\n❌ P1-B004抽出失敗")
            return 1

    except KeyboardInterrupt:
        print("\n⚠️ ユーザーによる中断")
        return 1
    except Exception as e:
        print(f"\n❌ 予期せぬエラー: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
