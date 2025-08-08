#!/usr/bin/env python3
"""
簡易SAM+YOLO精度向上システム
正解データとの比較による基本的な改善実装
"""

import cv2
import numpy as np
import json
from pathlib import Path
import logging

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_correct_annotations():
    """正解アノテーションの特徴分析"""
    logger.info("🔍 正解アノテーション特徴分析開始")
    
    # 正解データ読み込み
    correct_path = Path("C:/AItools/lora/train/yado/tracker-workspace/P1-B004/analysis/correct_annotations.json")
    with open(correct_path, 'r', encoding='utf-8') as f:
        correct_data = json.load(f)
    
    # 統計情報収集
    stats = {
        'aspect_ratios': [],
        'relative_areas': [],
        'center_positions': [],
        'dimensions': []
    }
    
    for filename, annotation in correct_data['annotations'].items():
        if annotation['primary_rectangle']:
            rect = annotation['primary_rectangle']
            
            # アスペクト比
            aspect_ratio = rect['width'] / rect['height'] if rect['height'] > 0 else 1.0
            stats['aspect_ratios'].append(aspect_ratio)
            
            # 相対面積
            stats['relative_areas'].append(rect['relative_width'] * rect['relative_height'])
            
            # 中心位置
            stats['center_positions'].append((rect['relative_x'] + rect['relative_width']/2, 
                                            rect['relative_y'] + rect['relative_height']/2))
            
            # 絶対サイズ
            stats['dimensions'].append((rect['width'], rect['height']))
    
    # 統計サマリー
    avg_aspect = sum(stats['aspect_ratios']) / len(stats['aspect_ratios'])
    avg_area = sum(stats['relative_areas']) / len(stats['relative_areas'])
    
    logger.info(f"📊 正解アノテーション統計:")
    logger.info(f"   平均アスペクト比: {avg_aspect:.2f}")
    logger.info(f"   平均相対面積: {avg_area:.3f} ({avg_area*100:.1f}%)")
    logger.info(f"   総アノテーション数: {len(stats['aspect_ratios'])}")
    
    return stats

def generate_improvement_recommendations():
    """改善推奨事項生成"""
    logger.info("🎯 SAM+YOLO改善推奨事項生成")
    
    stats = analyze_correct_annotations()
    
    recommendations = []
    
    # アスペクト比分析
    aspect_ratios = stats['aspect_ratios']
    avg_aspect = sum(aspect_ratios) / len(aspect_ratios)
    
    if avg_aspect < 1.0:
        recommendations.append("縦長キャラクターが多い - 縦方向の検出精度を向上させる")
    elif avg_aspect > 1.5:
        recommendations.append("横長レイアウトが多い - 横方向の検出範囲を拡大")
    else:
        recommendations.append("バランスの良いアスペクト比 - 現在の設定を維持")
    
    # 面積分析
    relative_areas = stats['relative_areas']
    avg_area = sum(relative_areas) / len(relative_areas)
    
    if avg_area > 0.3:
        recommendations.append("大きなキャラクターが多い - 面積重視の検出パラメータ推奨")
    elif avg_area < 0.1:
        recommendations.append("小さなキャラクターが多い - 精密検出パラメータ推奨")
    else:
        recommendations.append("中程度サイズのキャラクター - バランス型パラメータ推奨")
    
    # YOLO閾値推奨
    if avg_area > 0.25:
        recommendations.append("YOLO信頼度閾値: 0.05-0.07 (大キャラ検出)")
    else:
        recommendations.append("YOLO信頼度閾値: 0.03-0.05 (小キャラ検出)")
    
    # SAM設定推奨
    recommendations.append("SAM points_per_side: 32-48 (標準精度)")
    recommendations.append("品質評価: fullbody_priority または size_priority 推奨")
    
    logger.info("💡 推奨改善事項:")
    for i, rec in enumerate(recommendations, 1):
        logger.info(f"   {i}. {rec}")
    
    return recommendations

def create_optimized_extraction_config():
    """最適化抽出設定ファイル作成"""
    logger.info("⚙️ 最適化設定ファイル作成")
    
    stats = analyze_correct_annotations()
    
    # 統計に基づく最適設定計算
    avg_area = sum(stats['relative_areas']) / len(stats['relative_areas'])
    avg_aspect = sum(stats['aspect_ratios']) / len(stats['aspect_ratios'])
    
    # 設定値決定
    if avg_area > 0.25:
        # 大キャラ重視設定
        config = {
            "optimization_profile": "large_character_focus",
            "yolo_conf_threshold": 0.05,
            "yolo_iou_threshold": 0.4,
            "sam_points_per_side": 32,
            "quality_method": "size_priority",
            "weights": {
                "area": 0.4,
                "fullbody": 0.3,
                "central": 0.15,
                "grounded": 0.1,
                "confidence": 0.05
            }
        }
    elif avg_area < 0.15:
        # 小キャラ精密設定
        config = {
            "optimization_profile": "small_character_precision",
            "yolo_conf_threshold": 0.03,
            "yolo_iou_threshold": 0.3,
            "sam_points_per_side": 48,
            "quality_method": "balanced",
            "weights": {
                "area": 0.2,
                "fullbody": 0.4,
                "central": 0.2,
                "grounded": 0.15,
                "confidence": 0.05
            }
        }
    else:
        # バランス設定
        config = {
            "optimization_profile": "balanced_optimized",
            "yolo_conf_threshold": 0.04,
            "yolo_iou_threshold": 0.35,
            "sam_points_per_side": 40,
            "quality_method": "fullbody_priority",
            "weights": {
                "area": 0.25,
                "fullbody": 0.35,
                "central": 0.2,
                "grounded": 0.15,
                "confidence": 0.05
            }
        }
    
    # 統計情報追加
    config["analysis_stats"] = {
        "average_relative_area": avg_area,
        "average_aspect_ratio": avg_aspect,
        "total_annotations": len(stats['aspect_ratios'])
    }
    
    # 設定ファイル保存
    config_path = Path("C:/AItools/lora/train/yado/tracker-workspace/P1-B004/optimization/optimized_config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    logger.info(f"💾 最適化設定保存: {config_path}")
    logger.info(f"🎯 選択プロファイル: {config['optimization_profile']}")
    logger.info(f"📊 YOLO閾値: {config['yolo_conf_threshold']}")
    logger.info(f"⚙️ 品質評価: {config['quality_method']}")
    
    return config_path

def main():
    """メイン実行"""
    logger.info("=" * 60)
    logger.info("🚀 SAM+YOLO簡易最適化システム開始")
    logger.info("=" * 60)
    
    # 正解アノテーション分析
    stats = analyze_correct_annotations()
    
    # 改善推奨事項生成
    recommendations = generate_improvement_recommendations()
    
    # 最適化設定作成
    config_path = create_optimized_extraction_config()
    
    logger.info("=" * 60)
    logger.info("✅ SAM+YOLO簡易最適化完了")
    logger.info(f"📄 設定ファイル: {config_path}")
    logger.info("🔄 次のステップ: この設定で抽出テストを実行")
    logger.info("=" * 60)
    
    return 0

if __name__ == "__main__":
    exit(main())