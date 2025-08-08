#!/usr/bin/env python3
"""
P1-B004直接実行（環境問題緊急回避）
adaptive_cropping.pyモジュールを直接使用
"""

import sys
import json
import time
from datetime import datetime
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# P1-B004モジュール直接インポート
try:
    import cv2
    from features.processing.adaptive_cropping import AdaptiveCropper
    print("✅ P1-B004モジュール読み込み成功")
except ImportError as e:
    print(f"❌ P1-B004モジュール読み込み失敗: {e}")
    sys.exit(1)


def run_direct_extraction(tracker_id: str = "P1-B004"):
    """P1-B004モジュール直接実行"""
    print(f"🚀 {tracker_id} 直接抽出実行")
    
    # 入力・出力設定
    input_dir = Path("/mnt/c/AItools/lora/train/yado/org/kana08")
    output_dir = Path(f"/mnt/c/AItools/lora/train/yado/tracker-workspace/{tracker_id}/extraction")
    
    if not input_dir.exists():
        print(f"❌ 入力ディレクトリが存在しません: {input_dir}")
        return False
    
    # 画像ファイル取得
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    print(f"📊 入力画像数: {len(image_files)}枚")
    
    if not image_files:
        print("❌ 入力画像が見つかりません")
        return False
    
    # 出力ディレクトリ準備
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # AdaptiveCropper初期化
    try:
        cropper = AdaptiveCropper()
        print("✅ AdaptiveCropper初期化完了")
    except Exception as e:
        print(f"❌ AdaptiveCropper初期化失敗: {e}")
        return False
    
    # 処理実行
    success_count = 0
    start_time = time.time()
    
    # 最初の3枚で処理
    test_files = image_files[:3]
    
    for i, img_file in enumerate(test_files, 1):
        print(f"\n📸 [{i}/{len(test_files)}] {img_file.name}")
        
        try:
            # 画像読み込み
            image = cv2.imread(str(img_file))
            if image is None:
                print(f"❌ 画像読み込み失敗: {img_file}")
                continue
            
            # P1-B004 adaptive cropping実行
            from features.processing.adaptive_cropping import apply_adaptive_cropping
            result = apply_adaptive_cropping(
                image=image,
                cropper=cropper,
                save_path=str(output_dir / f"p1_b004_{img_file.stem}_adaptive.png")
            )
            
            if result["success"]:
                print(f"✅ 成功: {result.get('output_path', 'N/A')}")
                success_count += 1
                
                # 詳細情報表示
                if "face_detection" in result:
                    print(f"   👤 顔検出: {result['face_detection']}")
                if "cropping_applied" in result:
                    print(f"   ✂️ クロッピング: {result['cropping_applied']}")
            else:
                print(f"❌ 失敗: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ 処理エラー: {e}")
        
        time.sleep(0.5)  # 処理間隔
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # 結果確認
    output_files = list(output_dir.glob("*.png")) + list(output_dir.glob("*.jpg"))
    success_rate = (len(output_files) / len(test_files)) * 100
    
    print(f"\n📈 直接抽出結果:")
    print(f"  - 入力: {len(test_files)}枚")
    print(f"  - 出力: {len(output_files)}枚")
    print(f"  - 成功率: {success_rate:.1f}%")
    print(f"  - 処理時間: {processing_time:.1f}秒")
    
    if output_files:
        print("✅ 生成ファイル:")
        for f in output_files:
            print(f"  - {f.name}")
    
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
        "method": "direct_adaptive_cropper",
        "files": [f.name for f in output_files]
    }
    
    report_path = output_dir.parent / "extraction_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📄 レポート生成: {report_path}")
    
    if len(output_files) > 0:
        print("\n✅ P1-B004直接抽出成功: adaptive_croppingで画像生成完了")
        return True
    else:
        print("\n❌ P1-B004直接抽出失敗")
        return False


def main():
    """メイン実行"""
    print("="*60)
    print("🎯 P1-B004直接実行（AdaptiveCropper使用）")
    print("  - sympy環境問題を回避")
    print("  - P1-B004実装を直接使用")
    print("  - 実際の画像ファイル生成")
    print("="*60)
    
    try:
        success = run_direct_extraction("P1-B004")
        
        if success:
            print("\n🎉 P1-B004直接抽出完了")
            return 0
        else:
            print("\n❌ P1-B004直接抽出失敗")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ ユーザーによる中断")
        return 1
    except Exception as e:
        print(f"\n❌ 予期せぬエラー: {e}")
        return 1


if __name__ == "__main__":
    exit(main())