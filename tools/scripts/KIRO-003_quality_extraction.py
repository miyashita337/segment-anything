#!/usr/bin/env python3
"""
KIRO-003 品質重視キャラクター抽出スクリプト
KIRO-002の改善版: 成功率70%+、品質スコア0.7+を目標
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from features.extraction.commands.extract_character import main as extract_main

def setup_quality_config():
    """品質重視の設定を構築"""
    return {
        "score_threshold": 0.05,  # より厳格な閾値（KIRO-002: 0.07）
        "quality_mode": "balanced",
        "retry_count": 3,
        "mask_expansion": True,
        "edge_refinement": True,
        "min_mask_size": 100,
        "max_iterations": 5
    }

def run_quality_extraction():
    """品質重視抽出の実行"""
    
    # WorkspaceConfigManagerを使って動的パス解決
    from config.workspace_config import WorkspaceConfig
    workspace_config = WorkspaceConfig()
    config = workspace_config.get_workspace_config("KIRO-003")
    
    if config:
        # 動的パス生成
        input_dir = config.get('input_path', f"/mnt/c/AItools/lora/train/{config['author_name']}/aichikan/")
        output_dir = f"{config['workspace_path']}/extraction/"
        input_list = f"{config['workspace_path']}/input_files.txt"
    else:
        # フォールバック: 従来のハードコード
        input_dir = "/mnt/c/AItools/lora/train/kiri/aichikan/"
        output_dir = "/mnt/c/AItools/lora/train/kiri/tracker-workspace/KIRO-003/extraction/"
        input_list = "/mnt/c/AItools/lora/train/kiri/tracker-workspace/KIRO-003/input_files.txt"
    
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{output_dir}/../extraction_log.txt"),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=== KIRO-003 品質重視抽出 開始 ===")
    
    # 入力ファイルリスト読み込み
    with open(input_list, 'r') as f:
        input_files = [line.strip() for line in f.readlines()]
    
    logger.info(f"対象ファイル数: {len(input_files)}")
    
    # 品質設定
    config = setup_quality_config()
    logger.info(f"品質設定: {json.dumps(config, indent=2)}")
    
    # 抽出実行の引数構築
    sys.argv = [
        "extract_character.py",
        input_dir,
        "-o", output_dir,
        "--batch",
        "--score-threshold", str(config["score_threshold"]),
        "--quality-mode", config["quality_mode"]
    ]
    
    # ファイルリストを環境変数で渡す
    os.environ["KIRO_003_INPUT_FILES"] = ",".join(input_files)
    
    try:
        # 抽出実行
        logger.info("抽出処理を開始します...")
        extract_main()
        logger.info("=== KIRO-003 品質重視抽出 完了 ===")
        
    except Exception as e:
        logger.error(f"抽出中にエラーが発生: {e}")
        raise

if __name__ == "__main__":
    run_quality_extraction()