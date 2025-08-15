#!/usr/bin/env python3
"""
統合ダッシュボード生成ラッパースクリプト

既存の個別ダッシュボード生成スクリプト（qi004_dashboard_generator.py等）を
統合システムに置き換えるためのラッパー
"""

import sys
import argparse
import logging
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from features.common.unified_dashboard_generator import UnifiedDashboardGenerator


def setup_logging(verbose: bool = False):
    """ログ設定"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="統合ダッシュボード生成システム")
    parser.add_argument("tracker_id", help="トラッカーID (例: QI-004)")
    parser.add_argument("extraction_dir", help="抽出ディレクトリパス")
    parser.add_argument("output_dir", help="出力ディレクトリパス")
    parser.add_argument("--verbose", "-v", action="store_true", help="詳細ログ出力")
    parser.add_argument("--config-override", help="設定オーバーライド（JSON形式）")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"🚀 統合ダッシュボード生成開始")
        logger.info(f"   トラッカーID: {args.tracker_id}")
        logger.info(f"   抽出ディレクトリ: {args.extraction_dir}")
        logger.info(f"   出力ディレクトリ: {args.output_dir}")
        
        # 設定オーバーライド処理
        config_override = None
        if args.config_override:
            import json
            config_override = json.loads(args.config_override)
            logger.info(f"   設定オーバーライド: {config_override}")
        
        # 統合ダッシュボード生成
        generator = UnifiedDashboardGenerator()
        dashboard_path = generator.generate_dashboard(
            tracker_id=args.tracker_id,
            extraction_dir=args.extraction_dir,
            output_dir=args.output_dir,
            config_override=config_override
        )
        
        logger.info(f"✅ ダッシュボード生成完了: {dashboard_path}")
        logger.info(f"📊 ファイルサイズ: {dashboard_path.stat().st_size / 1024:.1f}KB")
        
        # 成功時の出力（既存スクリプトとの互換性）
        print(f"SUCCESS: {dashboard_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ ダッシュボード生成エラー: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)