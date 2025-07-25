#!/usr/bin/env python3
"""
Phase A2 強化SCI統合システムテスト
18枚のデータセットでの包括的評価を実行
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from features.evaluation.extraction_integrated_evaluator import ExtractionIntegratedEvaluator

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_enhanced_sci_integration():
    """強化SCI統合システムの包括的テスト"""
    
    # テストディレクトリ一覧
    test_directories = [
        "/mnt/c/AItools/lora/train/yado/org/kana05_cursor_fix",
        "/mnt/c/AItools/lora/train/yado/org/kana07_cursor_fix", 
        "/mnt/c/AItools/lora/train/yado/org/kana08_cursor_fix"
    ]
    
    logger.info("🧮 Phase A2 強化SCI統合システムテスト開始")
    logger.info("=" * 70)
    
    # 統合評価システム初期化
    evaluator = ExtractionIntegratedEvaluator()
    
    all_results = []
    total_face_detections = 0
    total_pose_detections = 0
    total_evaluations = 0
    sci_scores = []
    
    for test_dir in test_directories:
        dir_path = Path(test_dir)
        
        if not dir_path.exists():
            logger.warning(f"⚠️ ディレクトリが存在しません: {test_dir}")
            continue
            
        logger.info(f"\n📁 評価ディレクトリ: {dir_path.name}")
        logger.info("-" * 50)
        
        try:
            # 統合評価実行
            results = evaluator.evaluate_extraction_batch(str(dir_path))
            
            # 結果分析
            dir_face_detections = 0
            dir_pose_detections = 0
            dir_sci_scores = []
            
            for result in results:
                total_evaluations += 1
                
                # SCI詳細分析
                if result.enhanced_sci_details:
                    sci_detail = result.enhanced_sci_details
                    
                    # 顔検出率カウント
                    if sci_detail.face_detections:
                        dir_face_detections += 1
                        total_face_detections += 1
                    
                    # ポーズ検出率カウント
                    if sci_detail.pose_result.detected:
                        dir_pose_detections += 1
                        total_pose_detections += 1
                    
                    # SCIスコア収集
                    sci_scores.append(result.sci_score)
                    dir_sci_scores.append(result.sci_score)
                    
                    # 詳細ログ
                    logger.info(f"  📄 {result.correspondence.source_image.name}")
                    logger.info(f"    SCI総合: {result.sci_score:.3f}")
                    logger.info(f"    顔検出: {len(sci_detail.face_detections)}件")
                    logger.info(f"    ポーズ検出: {'成功' if sci_detail.pose_result.detected else '失敗'}")
                    logger.info(f"    品質レベル: {sci_detail.completeness_level}")
                    logger.info(f"    処理時間: {sci_detail.processing_time:.2f}秒")
            
            # ディレクトリ別サマリー
            dir_total = len(results)
            dir_face_rate = (dir_face_detections / dir_total) * 100 if dir_total > 0 else 0
            dir_pose_rate = (dir_pose_detections / dir_total) * 100 if dir_total > 0 else 0
            dir_sci_mean = sum(dir_sci_scores) / len(dir_sci_scores) if dir_sci_scores else 0
            
            logger.info(f"\n📊 {dir_path.name} サマリー:")
            logger.info(f"  評価件数: {dir_total}件")
            logger.info(f"  顔検出率: {dir_face_rate:.1f}% ({dir_face_detections}/{dir_total})")
            logger.info(f"  ポーズ検出率: {dir_pose_rate:.1f}% ({dir_pose_detections}/{dir_total})")
            logger.info(f"  SCI平均: {dir_sci_mean:.3f}")
            
            all_results.extend(results)
            
        except Exception as e:
            logger.error(f"❌ {dir_path.name} 評価エラー: {e}")
            continue
    
    # 全体統計
    logger.info("\n" + "=" * 70)
    logger.info("📈 Phase A2 強化SCI統合システム 全体統計")
    logger.info("=" * 70)
    
    if total_evaluations > 0:
        overall_face_rate = (total_face_detections / total_evaluations) * 100
        overall_pose_rate = (total_pose_detections / total_evaluations) * 100
        overall_sci_mean = sum(sci_scores) / len(sci_scores) if sci_scores else 0
        sci_std = (sum((x - overall_sci_mean) ** 2 for x in sci_scores) / len(sci_scores)) ** 0.5 if sci_scores else 0
        
        logger.info(f"📊 総合結果:")
        logger.info(f"  総評価件数: {total_evaluations}件")
        logger.info(f"  顔検出率: {overall_face_rate:.1f}% ({total_face_detections}/{total_evaluations})")
        logger.info(f"  ポーズ検出率: {overall_pose_rate:.1f}% ({total_pose_detections}/{total_evaluations})")
        logger.info(f"  SCI平均値: {overall_sci_mean:.3f} ± {sci_std:.3f}")
        
        # Phase A2目標達成度
        logger.info(f"\n🎯 Phase A2 目標達成度:")
        
        # 顔検出率目標: 90%
        face_target = 90.0
        face_achievement = (overall_face_rate / face_target) * 100
        face_status = "✅ 達成" if overall_face_rate >= face_target else "⚠️ 未達成"
        logger.info(f"  顔検出率: {overall_face_rate:.1f}% / {face_target}% ({face_achievement:.1f}%) {face_status}")
        
        # ポーズ検出率目標: 80%
        pose_target = 80.0
        pose_achievement = (overall_pose_rate / pose_target) * 100
        pose_status = "✅ 達成" if overall_pose_rate >= pose_target else "⚠️ 未達成"
        logger.info(f"  ポーズ検出率: {overall_pose_rate:.1f}% / {pose_target}% ({pose_achievement:.1f}%) {pose_status}")
        
        # SCI平均値目標: 0.70
        sci_target = 0.70
        sci_achievement = (overall_sci_mean / sci_target) * 100
        sci_status = "✅ 達成" if overall_sci_mean >= sci_target else "⚠️ 未達成"
        logger.info(f"  SCI平均値: {overall_sci_mean:.3f} / {sci_target:.2f} ({sci_achievement:.1f}%) {sci_status}")
        
        # レポート保存
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "phase": "A2_enhanced_sci_integration_test",
            "total_evaluated": total_evaluations,
            "face_detection_rate": overall_face_rate / 100,
            "pose_detection_rate": overall_pose_rate / 100,
            "sci_statistics": {
                "mean": overall_sci_mean,
                "std": sci_std,
                "min": min(sci_scores) if sci_scores else 0,
                "max": max(sci_scores) if sci_scores else 0,
                "count": len(sci_scores)
            },
            "target_achievements": {
                "face_detection": {
                    "target": face_target / 100,
                    "actual": overall_face_rate / 100,
                    "achievement_rate": face_achievement / 100,
                    "achieved": bool(overall_face_rate >= face_target)
                },
                "pose_detection": {
                    "target": pose_target / 100,
                    "actual": overall_pose_rate / 100,
                    "achievement_rate": pose_achievement / 100,
                    "achieved": bool(overall_pose_rate >= pose_target)
                },
                "sci_mean": {
                    "target": sci_target,
                    "actual": overall_sci_mean,
                    "achievement_rate": sci_achievement / 100,
                    "achieved": bool(overall_sci_mean >= sci_target)
                }
            }
        }
        
        report_file = f"evaluation_reports/phase_a2_enhanced_sci_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs("evaluation_reports", exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n💾 レポート保存: {report_file}")
        
        # 総合評価
        targets_achieved = sum([
            overall_face_rate >= face_target,
            overall_pose_rate >= pose_target,
            overall_sci_mean >= sci_target
        ])
        
        if targets_achieved == 3:
            logger.info(f"\n🎉 Phase A2 完全達成! (3/3 目標クリア)")
        elif targets_achieved >= 2:
            logger.info(f"\n🔥 Phase A2 良好進捗! ({targets_achieved}/3 目標クリア)")
        else:
            logger.info(f"\n📈 Phase A2 改善必要 ({targets_achieved}/3 目標クリア)")
    else:
        logger.error("❌ 評価データなし")
    
    logger.info("\n✅ Phase A2 強化SCI統合システムテスト完了")
    return all_results


if __name__ == "__main__":
    try:
        results = test_enhanced_sci_integration()
        print(f"\n🔍 テスト結果: {len(results)}件の評価を完了")
    except Exception as e:
        logger.error(f"❌ テスト実行エラー: {e}")
        sys.exit(1)