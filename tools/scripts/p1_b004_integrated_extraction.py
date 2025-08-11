#!/usr/bin/env python3
"""
P1-B004統合実行スクリプト
extract_character.pyの--adaptive-croppingを環境問題回避で実行
"""

import os
import subprocess
import sys
import time
import json
from datetime import datetime
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def run_integrated_extraction(tracker_id: str = "P1-B004"):
    """統合extract_character.pyでの抽出実行"""
    print(f"🚀 {tracker_id} 統合抽出パイプライン開始")
    
    # 入力・出力設定
    input_dir = Path("/mnt/c/AItools/lora/train/yado/org/kana08")
    output_dir = Path(f"/mnt/c/AItools/lora/train/yado/tracker-workspace/{tracker_id}/extraction")
    
    if not input_dir.exists():
        print(f"❌ 入力ディレクトリが存在しません: {input_dir}")
        return False
    
    # 画像ファイル確認
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    print(f"📊 入力画像数: {len(image_files)}枚")
    
    if not image_files:
        print("❌ 入力画像が見つかりません")
        return False
    
    # 出力ディレクトリ準備
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 単体画像ずつ処理（sympy問題回避）
    success_count = 0
    start_time = time.time()
    
    # 最初の3枚で処理
    test_files = image_files[:3]
    
    for i, img_file in enumerate(test_files, 1):
        print(f"\n📸 [{i}/{len(test_files)}] {img_file.name}")
        
        # 出力ファイル名
        output_name = f"p1_b004_{img_file.stem}_adaptive.png"
        output_path = output_dir / output_name
        
        try:
            # 単体画像処理（環境リセット効果）
            command = [
                "python3", 
                "features/extraction/commands/extract_character.py",
                str(img_file),
                "-o", str(output_path),
                "--adaptive-cropping",
                "--verbose"
            ]
            
            print(f"💻 実行: {' '.join(command[:4])} ... --adaptive-cropping")
            
            # 新しいプロセスで実行（環境分離）
            result = subprocess.run(
                command,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=60,  # 1分タイムアウト
                env={**dict(os.environ), "PYTHONPATH": str(project_root)} if 'os' in globals() else None
            )
            
            if result.returncode == 0 and output_path.exists():
                print(f"✅ 成功: {output_name}")
                success_count += 1
            else:
                print(f"❌ 失敗: {img_file.name}")
                if result.stderr:
                    print(f"  エラー: {result.stderr[:200]}...")
                    
        except subprocess.TimeoutExpired:
            print(f"⏰ タイムアウト: {img_file.name}")
        except Exception as e:
            print(f"❌ エラー: {e}")
        
        time.sleep(1)  # プロセス間隔
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # 結果確認
    output_files = list(output_dir.glob("*.png")) + list(output_dir.glob("*.jpg"))
    success_rate = (len(output_files) / len(test_files)) * 100
    
    print(f"\n📈 統合抽出結果:")
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
        "method": "integrated_extract_character",
        "files": [f.name for f in output_files]
    }
    
    report_path = output_dir.parent / "extraction_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📄 レポート生成: {report_path}")
    
    if len(output_files) > 0:
        print("\n✅ 統合抽出成功: extract_character.py --adaptive-cropping で画像生成完了")
        return True
    else:
        print("\n❌ 統合抽出失敗")
        return False


def main():
    """メイン実行"""
    import os
    
    print("="*60)
    print("🎯 P1-B004統合実行（extract_character.py使用）")
    print("  - --adaptive-cropping オプション使用")
    print("  - 環境問題回避（プロセス分離）")
    print("  - 実際の画像ファイル生成")
    print("="*60)
    
    try:
        success = run_integrated_extraction("P1-B004")
        
        if success:
            print("\n🎉 P1-B004統合抽出完了")
            return 0
        else:
            print("\n❌ P1-B004統合抽出失敗")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ ユーザーによる中断")
        return 1
    except Exception as e:
        print(f"\n❌ 予期せぬエラー: {e}")
        return 1


if __name__ == "__main__":
    exit(main())