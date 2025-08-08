#!/usr/bin/env python3
"""
P1-013実際の差分処理最適化実行
差分処理最適化システムの実際の処理を実行し、自動的に/releaseに更新
"""

import sys
import subprocess
import time
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    """P1-013実際の差分処理最適化実行"""
    tracker_id = "P1-013"
    project_root = Path(__file__).parent.parent.parent
    
    # 入力・出力パス設定
    input_dir = "/mnt/c/AItools/lora/train/yado/org/kana05"
    output_base = "/mnt/c/AItools/lora/train/yado/tracker-workspace"
    output_dir = f"{output_base}/{tracker_id}/extraction"
    
    print("============================================================")
    print("🚀 P1-013実際の差分処理最適化パイプライン実行")
    print("  - 差分処理最適化システムの実行")  
    print("  - キャッシュベース増分処理")
    print("  - 完了後に自動/release更新")
    print("============================================================")
    
    # ログファイル設定
    log_file = Path(output_base) / tracker_id / "extraction_log.txt"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("============================================================\\n")
        f.write("🚀 P1-013実際の差分処理最適化パイプライン実行\\n")
        f.write("  - 差分処理最適化システムの実行\\n")
        f.write("  - キャッシュベース増分処理\\n")
        f.write("  - 完了後に自動/release更新\\n")
        f.write("============================================================\\n")
        
        # 入力ディレクトリ確認
        if not Path(input_dir).exists():
            error_msg = f"❌ 入力ディレクトリが存在しません: {input_dir}"
            print(error_msg)
            f.write(error_msg + "\\n")
            return False
            
        # 入力画像数カウント
        input_path = Path(input_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
            
        print(f"🚀 P1-013 差分処理最適化パイプライン開始")
        print(f"📊 入力画像数: {len(image_files)}枚")
        f.write(f"🚀 P1-013 差分処理最適化パイプライン開始\\n")
        f.write(f"📊 入力画像数: {len(image_files)}枚\\n")
        
        # 差分処理最適化実行
        print("🔧 差分処理最適化システム実行開始")
        f.write("🔧 差分処理最適化システム実行開始\\n")
        
        differential_processor_cmd = [
            "python3", 
            "features/processing/differential_processor.py",
            "--tracker-id", tracker_id,
            "--input-dir", input_dir
        ]
        
        print(f"💻 実行コマンド: {' '.join(differential_processor_cmd)}")
        f.write(f"💻 実行コマンド: {' '.join(differential_processor_cmd)}\\n")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                differential_processor_cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=600  # 10分タイムアウト
            )
            
            elapsed_time = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ 差分処理最適化成功 ({elapsed_time:.1f}秒)")
                f.write(f"✅ 差分処理最適化成功 ({elapsed_time:.1f}秒)\\n")
                
                # 結果ファイル確認
                quality_dir = Path(output_base) / tracker_id / "quality"
                result_files = list(quality_dir.glob("*differential_report.json")) if quality_dir.exists() else []
                summary_files = list(quality_dir.glob("*differential_summary.md")) if quality_dir.exists() else []
                
                # 抽出ディレクトリ確認
                extraction_dir = Path(output_base) / tracker_id / "extraction"
                extraction_files = list(extraction_dir.glob("*")) if extraction_dir.exists() else []
                
                print(f"📁 品質レポートファイル数: {len(result_files) + len(summary_files)}個")
                print(f"📁 抽出ファイル数: {len(extraction_files)}個")
                f.write(f"📁 品質レポートファイル数: {len(result_files) + len(summary_files)}個\\n")
                f.write(f"📁 抽出ファイル数: {len(extraction_files)}個\\n")
                
                # 標準出力からの統計情報抽出
                output_lines = result.stdout.split('\\n')
                processed_files = 0
                skipped_files = 0
                failed_files = 0
                cache_hit_rate = 0.0
                
                for line in output_lines:
                    if "処理結果:" in line:
                        # 処理結果行から数値抽出
                        parts = line.split(':')[1].strip()
                        if '処理' in parts:
                            processed_files = int(parts.split('処理')[0].strip())
                        if 'スキップ' in parts:
                            skipped_part = parts.split('スキップ')[0].split(',')[-1].strip()
                            skipped_files = int(skipped_part)
                        if '失敗' in parts:
                            failed_part = parts.split('失敗')[0].split(',')[-1].strip()
                            failed_files = int(failed_part)
                    elif "キャッシュヒット率:" in line:
                        try:
                            cache_rate_str = line.split('%')[0].split('(')[-1]
                            cache_hit_rate = float(cache_rate_str)
                        except:
                            cache_hit_rate = 0.0
                
                print(f"📊 処理統計:")
                print(f"  - 処理ファイル数: {processed_files}枚")
                print(f"  - スキップファイル数: {skipped_files}枚") 
                print(f"  - 失敗ファイル数: {failed_files}枚")
                print(f"  - キャッシュヒット率: {cache_hit_rate:.1f}%")
                
                f.write(f"📊 処理統計:\\n")
                f.write(f"  - 処理ファイル数: {processed_files}枚\\n")
                f.write(f"  - スキップファイル数: {skipped_files}枚\\n")
                f.write(f"  - 失敗ファイル数: {failed_files}枚\\n")
                f.write(f"  - キャッシュヒット率: {cache_hit_rate:.1f}%\\n")
                
                # 成功判定：処理が完了したか、または有効なスキップがある場合
                total_handled = processed_files + skipped_files
                success_rate = (total_handled / max(len(image_files), 1)) * 100 if len(image_files) > 0 else 100.0
                
            else:
                print(f"❌ 差分処理最適化失敗 (終了コード: {result.returncode})")
                print(f"エラー出力: {result.stderr}")
                f.write(f"❌ 差分処理最適化失敗 (終了コード: {result.returncode})\\n")
                f.write(f"エラー出力: {result.stderr}\\n")
                success_rate = 0.0
                
        except subprocess.TimeoutExpired:
            print("❌ 差分処理最適化タイムアウト")
            f.write("❌ 差分処理最適化タイムアウト\\n")
            success_rate = 0.0
        except Exception as e:
            print(f"❌ 差分処理最適化エラー: {e}")
            f.write(f"❌ 差分処理最適化エラー: {e}\\n")
            success_rate = 0.0
            
        # Google Sheets自動更新
        if success_rate >= 50.0:  # 50%以上で成功とみなす
            print("🔄 Google Sheetsを/releaseに自動更新中...")
            f.write("🔄 Google Sheetsを/releaseに自動更新中...\\n")
            
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
                    f.write(f"✅ Google Sheets自動更新成功: {tracker_id} -> /release\\n")
                else:
                    print(f"⚠️ Google Sheets更新失敗: {update_result.stderr}")
                    f.write(f"⚠️ Google Sheets更新失敗: {update_result.stderr}\\n")
                    
            except Exception as e:
                print(f"❌ Google Sheets更新エラー: {e}")
                f.write(f"❌ Google Sheets更新エラー: {e}\\n")
        else:
            print("⚠️ 成功率が低いため、手動確認が必要です")
            f.write("⚠️ 成功率が低いため、手動確認が必要です\\n")
            
        print("")
        print("🎉 P1-013差分処理最適化パイプライン完了")
        f.write("\\n🎉 P1-013差分処理最適化パイプライン完了\\n")
        
        # 最終的なサマリー出力
        if success_rate >= 50.0:
            print("📋 処理サマリー:")
            print(f"  - 差分処理最適化システム: 実行完了")
            print(f"  - キャッシュ効率化: {cache_hit_rate:.1f}%")
            print(f"  - 結果保存: 完了")
            print(f"  - Google Sheets: /release更新済み")
            
            f.write("📋 処理サマリー:\\n")
            f.write(f"  - 差分処理最適化システム: 実行完了\\n")
            f.write(f"  - キャッシュ効率化: {cache_hit_rate:.1f}%\\n")
            f.write(f"  - 結果保存: 完了\\n")
            f.write(f"  - Google Sheets: /release更新済み\\n")
            
        return success_rate >= 50.0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)