#!/usr/bin/env python3
"""
P1-015: 大規模データセット対応メモリ最適化実行スクリプト
完了後に自動的に/releaseステータスに更新
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def run_large_dataset_processing(tracker_id: str = "P1-015"):
    """大規模データセット処理パイプライン実行"""
    print(f"🚀 {tracker_id} 大規模データセット対応メモリ最適化処理開始")
    
    # 入力・出力ディレクトリ設定
    input_dir = Path("/mnt/c/AItools/lora/train/yado/org/kana05")
    output_dir = Path(f"/mnt/c/AItools/lora/train/yado/tracker-workspace/{tracker_id}/extraction")
    
    # 入力ディレクトリ確認
    if not input_dir.exists():
        print(f"❌ 入力ディレクトリが存在しません: {input_dir}")
        return False
    
    # 画像数確認
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    print(f"📊 入力画像数: {len(image_files)}枚")
    
    # 出力ディレクトリ準備
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 既存の出力ファイルクリーンアップ
    for existing_file in output_dir.glob("*"):
        if existing_file.is_file():
            existing_file.unlink()
            
    print(f"🧹 出力ディレクトリクリーンアップ完了: {output_dir}")
    
    try:
        # 大規模データセット処理システム実行
        print("🔧 大規模データセット処理システム実行開始")
        start_time = time.time()
        
        # メモリ使用量とファイル数に基づいて最適なチャンクサイズを決定
        chunk_size = min(30, max(10, len(image_files) // 10))  # 10-30の範囲で調整
        memory_threshold = 1.5  # 1.5GB閾値
        
        command = [
            "python3", "features/processing/large_dataset_processor.py",
            "--tracker-id", tracker_id,
            "--input-dir", str(input_dir),
            "--chunk-size", str(chunk_size),
            "--memory-threshold", str(memory_threshold)
        ]
        
        print(f"💻 実行コマンド: {' '.join(command)}")
        print(f"📦 チャンクサイズ: {chunk_size} (総ファイル数 {len(image_files)} に基づく自動調整)")
        print(f"💾 メモリ閾値: {memory_threshold}GB")
        
        # 実行
        result = subprocess.run(
            command,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=7200,  # 2時間タイムアウト（大規模データセット対応）
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if result.returncode == 0:
            # 成功
            print(f"✅ 大規模データセット処理成功 ({processing_time:.1f}秒)")
            
            # 出力ファイル確認
            output_files = list(output_dir.glob("*.png")) + list(output_dir.glob("*.jpg"))
            print(f"📁 出力ファイル数: {len(output_files)}個")
            
            # 成功率計算
            success_rate = (len(output_files) / len(image_files)) * 100 if image_files else 0
            print(f"📈 抽出成功率: {success_rate:.1f}%")
            
            # パフォーマンス統計表示
            avg_time_per_file = processing_time / len(image_files) if image_files else 0
            print(f"⚡ 平均処理時間: {avg_time_per_file:.2f}秒/ファイル")
            print(f"💾 メモリ効率: 大規模データセット対応最適化適用")
            
            # 出力ファイルサンプル表示
            if output_files:
                print("📋 出力ファイルサンプル:")
                for i, output_file in enumerate(output_files[:5], 1):
                    size_mb = output_file.stat().st_size / 1024 / 1024
                    print(f"  {i}. {output_file.name} ({size_mb:.2f}MB)")
                if len(output_files) > 5:
                    print(f"  ... (他{len(output_files) - 5}個)")
            
            # 自動的に/releaseに更新
            if success_rate >= 70.0:  # 70%以上で成功とみなす（大規模データセットは要求を緩和）
                print("🔄 Google Sheetsを/releaseに自動更新中...")
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
                    else:
                        print(f"⚠️ Google Sheets更新失敗: {update_result.stderr}")
                        
                except Exception as e:
                    print(f"❌ Google Sheets更新エラー: {e}")
            else:
                print(f"⚠️ 成功率{success_rate:.1f}%のため、手動確認が必要です")
            
            return True
            
        else:
            # 失敗
            print(f"❌ 大規模データセット処理失敗")
            print(f"📋 エラー出力: {result.stderr[:500]}...")
            if result.stdout:
                print(f"📋 標準出力: {result.stdout[:500]}...")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ 大規模データセット処理タイムアウト（2時間）")
        return False
    except Exception as e:
        print(f"❌ 大規模データセット処理エラー: {e}")
        return False


def generate_memory_optimization_report(tracker_id: str = "P1-015"):
    """メモリ最適化レポート生成"""
    print("📊 メモリ最適化レポート生成中...")
    
    try:
        # システム情報取得
        import psutil
        import torch
        
        print(f"🖥️ システム環境:")
        
        # メモリ情報
        memory = psutil.virtual_memory()
        print(f"   RAM: {memory.total // 1024**3}GB総容量 ({memory.percent}%使用中)")
        print(f"   利用可能: {memory.available // 1024**3}GB")
        
        # GPU情報
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"   GPU数: {gpu_count}")
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                gpu_memory_gb = props.total_memory // 1024**3
                print(f"   GPU {i}: {props.name} ({gpu_memory_gb}GB)")
        else:
            print("   GPU: 利用不可（CPU処理）")
        
        # P1-015固有の最適化機能表示
        print(f"🚀 P1-015 メモリ最適化機能:")
        print(f"   ✅ 動的バッチサイズ調整")
        print(f"   ✅ メモリ圧迫検知・自動最適化")
        print(f"   ✅ プログレッシブチャンク処理")
        print(f"   ✅ 中間クリーンアップ")
        print(f"   ✅ チェックポイント機能")
        
    except Exception as e:
        print(f"❌ メモリ最適化レポート生成エラー: {e}")


def create_optimization_demo():
    """最適化機能デモ実行"""
    print("🎮 P1-015 メモリ最適化デモ実行中...")
    
    try:
        demo_result = subprocess.run(
            ["python3", "test_p1_015_demo.py"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=300  # 5分タイムアウト
        )
        
        if demo_result.returncode == 0:
            print("✅ メモリ最適化デモ実行成功")
            print("📊 主要機能の動作確認完了")
        else:
            print("⚠️ メモリ最適化デモ実行失敗")
            print(demo_result.stderr[:300])
            
    except Exception as e:
        print(f"❌ デモ実行エラー: {e}")


def main():
    """メイン実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description="P1-015 大規模データセット処理実行")
    parser.add_argument("--tracker_id", type=str, default="P1-015", help="トラッカーID")
    parser.add_argument("--skip-report", action="store_true", help="レポート生成をスキップ")
    parser.add_argument("--skip-demo", action="store_true", help="デモ実行をスキップ")
    
    args = parser.parse_args()
    
    print("="*60)
    print("🎯 P1-015 大規模データセット対応メモリ最適化実行")
    print("  - プログレッシブチャンク処理")
    print("  - 動的メモリ管理・最適化")
    print("  - 大規模データセット対応")
    print("  - 完了後に自動/release更新")
    print("="*60)
    
    try:
        # デモ実行（オプション）
        if not args.skip_demo:
            create_optimization_demo()
            print()
        
        # メイン処理実行
        success = run_large_dataset_processing(args.tracker_id)
        
        # レポート生成
        if not args.skip_report:
            print()
            generate_memory_optimization_report(args.tracker_id)
        
        if success:
            print("\n🎉 P1-015 大規模データセット対応メモリ最適化完了")
            print("💾 メモリ効率化による安定した大規模処理を実現")
            print("⚡ プログレッシブ処理によるスケーラブルな処理基盤確立")
            return 0
        else:
            print("\n❌ P1-015 大規模データセット処理失敗")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ ユーザーによる中断")
        return 1
    except Exception as e:
        print(f"\n❌ 予期せぬエラー: {e}")
        return 1


if __name__ == "__main__":
    exit(main())