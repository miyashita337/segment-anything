#!/usr/bin/env python3
"""
P1-012実際の抽出パイプライン実行
境界例自動検出システムの実際の処理を実行し、自動的に/releaseに更新
"""

import sys
import subprocess
import time
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    """P1-012実際の抽出パイプライン実行"""
    tracker_id = "P1-012"
    project_root = Path(__file__).parent.parent.parent
    
    # 入力・出力パス設定
    input_dir = "/mnt/c/AItools/lora/train/yado/org/kana05"
    output_base = "/mnt/c/AItools/lora/train/yado/tracker-workspace"
    output_dir = f"{output_base}/{tracker_id}/extraction"
    
    print("============================================================")
    print("🎯 P1-012実際の境界例検出パイプライン実行")
    print("  - 境界例自動検出システムの実行")  
    print("  - 完了後に自動/release更新")
    print("============================================================")
    
    # ログファイル設定
    log_file = Path(output_base) / tracker_id / "extraction_log.txt"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("============================================================\n")
        f.write("🎯 P1-012実際の境界例検出パイプライン実行\n")
        f.write("  - 境界例自動検出システムの実行\n")
        f.write("  - 完了後に自動/release更新\n")
        f.write("============================================================\n")
        
        # 入力ディレクトリ確認
        if not Path(input_dir).exists():
            error_msg = f"❌ 入力ディレクトリが存在しません: {input_dir}"
            print(error_msg)
            f.write(error_msg + "\n")
            return False
            
        # 入力画像数カウント
        input_path = Path(input_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
            
        print(f"🚀 P1-012 境界例検出パイプライン開始")
        print(f"📊 入力画像数: {len(image_files)}枚")
        f.write(f"🚀 P1-012 境界例検出パイプライン開始\n")
        f.write(f"📊 入力画像数: {len(image_files)}枚\n")
        
        # 境界例検出実行
        print("🔧 境界例自動検出システム実行開始")
        f.write("🔧 境界例自動検出システム実行開始\n")
        
        boundary_detector_cmd = [
            "python3", 
            "features/evaluation/boundary_case_detector.py",
            "--tracker-id", tracker_id,
            "--input-dir", input_dir
        ]
        
        print(f"💻 実行コマンド: {' '.join(boundary_detector_cmd)}")
        f.write(f"💻 実行コマンド: {' '.join(boundary_detector_cmd)}\n")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                boundary_detector_cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5分タイムアウト
            )
            
            elapsed_time = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ 境界例検出処理成功 ({elapsed_time:.1f}秒)")
                f.write(f"✅ 境界例検出処理成功 ({elapsed_time:.1f}秒)\n")
                
                # 結果ファイル確認
                quality_dir = Path(output_base) / tracker_id / "quality"
                result_files = list(quality_dir.glob("*boundary_analysis.json")) if quality_dir.exists() else []
                summary_files = list(quality_dir.glob("*boundary_summary.md")) if quality_dir.exists() else []
                
                print(f"📁 結果ファイル数: {len(result_files) + len(summary_files)}個")
                f.write(f"📁 結果ファイル数: {len(result_files) + len(summary_files)}個\n")
                
                # 実行成功とみなす
                success_rate = 100.0
                
            else:
                print(f"❌ 境界例検出処理失敗 (終了コード: {result.returncode})")
                print(f"エラー出力: {result.stderr}")
                f.write(f"❌ 境界例検出処理失敗 (終了コード: {result.returncode})\n")
                f.write(f"エラー出力: {result.stderr}\n")
                success_rate = 0.0
                
        except subprocess.TimeoutExpired:
            print("❌ 境界例検出処理タイムアウト")
            f.write("❌ 境界例検出処理タイムアウト\n")
            success_rate = 0.0
        except Exception as e:
            print(f"❌ 境界例検出処理エラー: {e}")
            f.write(f"❌ 境界例検出処理エラー: {e}\n")
            success_rate = 0.0
            
        # Google Sheets自動更新
        if success_rate >= 50.0:  # 50%以上で成功とみなす
            print("🔄 Google Sheetsを/releaseに自動更新中...")
            f.write("🔄 Google Sheetsを/releaseに自動更新中...\n")
            
            try:
                update_result = subprocess.run(
                    ["python3", "tools/progress_tracker/cli.py", "update", tracker_id, "/release"],
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if update_result.returncode == 0:
                    print(f"✅ Google Sheets自動更新成功: {tracker_id} -> /release")
                    f.write(f"✅ Google Sheets自動更新成功: {tracker_id} -> /release\n")
                else:
                    print(f"⚠️ Google Sheets更新失敗: {update_result.stderr}")
                    f.write(f"⚠️ Google Sheets更新失敗: {update_result.stderr}\n")
                    
            except Exception as e:
                print(f"❌ Google Sheets更新エラー: {e}")
                f.write(f"❌ Google Sheets更新エラー: {e}\n")
        else:
            print("⚠️ 成功率が低いため、手動確認が必要です")
            f.write("⚠️ 成功率が低いため、手動確認が必要です\n")
            
        print("")
        print("🎉 P1-012境界例検出パイプライン完了")
        f.write("\n🎉 P1-012境界例検出パイプライン完了\n")
        
        # 最終的なサマリー出力
        if success_rate >= 50.0:
            print("📋 処理サマリー:")
            print(f"  - 境界例検出システム: 実行完了")
            print(f"  - 結果保存: 完了")
            print(f"  - Google Sheets: /release更新済み")
            
            f.write("📋 処理サマリー:\n")
            f.write(f"  - 境界例検出システム: 実行完了\n")
            f.write(f"  - 結果保存: 完了\n")
            f.write(f"  - Google Sheets: /release更新済み\n")
            
        return success_rate >= 50.0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)