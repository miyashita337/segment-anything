#!/usr/bin/env python3
"""
評価用ダッシュボード生成スクリプト

元のワークフローで参照されていたが欠落していたファイル。
統一ダッシュボードシステムへのブリッジとして機能。

使用方法:
    python features/evaluation/dashboard_generator.py
    python features/evaluation/dashboard_generator.py --tracker-id TRACKER-006
"""

import argparse
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from features.common.dashboard_generator import DashboardGenerator


def main():
    """メイン実行"""
    parser = argparse.ArgumentParser(
        description="評価用ダッシュボード生成 - 統一ダッシュボードシステム経由"
    )
    parser.add_argument(
        "--tracker-id", 
        default="UNKNOWN", 
        help="トラッカーID（デフォルト: UNKNOWN）"
    )
    parser.add_argument(
        "--workspace-dir",
        help="ワークスペースディレクトリ（省略時は自動推定）"
    )
    parser.add_argument(
        "--extraction-result",
        help="抽出結果JSONファイルパス（省略時は自動推定）"
    )
    parser.add_argument(
        "--mode",
        choices=["standard", "integrated"],
        default="standard",
        help="ダッシュボードモード（デフォルト: standard）"
    )
    
    args = parser.parse_args()
    
    print(f"🎨 評価用ダッシュボード生成開始: {args.tracker_id}")
    print(f"📋 モード: {args.mode}")
    
    try:
        # ワークスペースディレクトリの推定
        if args.workspace_dir:
            workspace_dir = args.workspace_dir
        else:
            # トラッカーIDから自動推定
            workspace_base = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace")
            workspace_dir = workspace_base / args.tracker_id
            
            if not workspace_dir.exists():
                print(f"⚠️ ワークスペースディレクトリが見つかりません: {workspace_dir}")
                print(f"💡 手動指定してください: --workspace-dir /path/to/workspace")
                return 1
        
        print(f"📁 ワークスペース: {workspace_dir}")
        
        # 統一ダッシュボード生成器初期化
        generator = DashboardGenerator(
            workspace_dir=workspace_dir,
            tracker_id=args.tracker_id
        )
        
        # モードに応じたダッシュボード生成
        if args.mode == "standard":
            # 標準モード: extraction_result.jsonベース
            dashboard_path = generator.create_dashboard(
                tracker_id=args.tracker_id,
                workspace_dir=str(workspace_dir),
                extraction_result_path=args.extraction_result
            )
        else:
            # 統合モード: quality_dataベース（模擬データ使用）
            quality_data = {
                'tracker_id': args.tracker_id,
                'success_rate': 85.0,
                'total_images': 100,
                'successful_images': 85
            }
            dashboard_path = generator.generate_dashboard(quality_data)
        
        print(f"✅ ダッシュボード生成完了: {dashboard_path}")
        
        # 統合サーバー用index.html作成（TRACKER-006問題解決）
        workspace_path = Path(workspace_dir)
        index_file = workspace_path / "index.html"
        dashboard_file = workspace_path / "dashboard" / "dashboard.html"
        
        if dashboard_file.exists():
            # dashboard.htmlの内容をindex.htmlにコピー
            with open(dashboard_file, 'r', encoding='utf-8') as f:
                dashboard_content = f.read()
            
            with open(index_file, 'w', encoding='utf-8') as f:
                f.write(dashboard_content)
            
            print(f"📋 統合サーバー用index.html作成: {index_file}")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"❌ ファイルが見つかりません: {e}")
        return 1
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())