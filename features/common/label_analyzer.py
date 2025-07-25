#!/usr/bin/env python3
"""
Label Data Analyzer - 抽出したラベルデータの詳細分析
Analysis of extracted label data for character extraction learning
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class LabelDataAnalyzer:
    """ラベルデータ分析クラス"""
    
    def __init__(self, labels_path: Path):
        """
        初期化
        
        Args:
            labels_path: 抽出済みラベルJSONファイルパス
        """
        self.labels_path = labels_path
        self.labels_data = self.load_labels()
        
    def load_labels(self) -> List[Dict[str, Any]]:
        """ラベルデータを読み込み"""
        try:
            with open(self.labels_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"ラベルデータ読み込み: {len(data)}ファイル")
            return data
        except Exception as e:
            logger.error(f"ラベルデータ読み込みエラー: {e}")
            return []
    
    def analyze_character_regions(self) -> Dict[str, Any]:
        """
        キャラクター領域の統計分析
        
        Returns:
            統計情報辞書
        """
        stats = {
            "total_files": len(self.labels_data),
            "files_with_character": 0,
            "aspect_ratios": [],
            "areas": [],
            "sizes": [],
            "positions": [],
            "fill_ratios": []
        }
        
        for label in self.labels_data:
            if label.get("character_region"):
                char_region = label["character_region"]
                bbox = char_region["bbox"]
                
                stats["files_with_character"] += 1
                
                # アスペクト比
                aspect_ratio = bbox["height"] / max(bbox["width"], 1)
                stats["aspect_ratios"].append(aspect_ratio)
                
                # 面積
                area = bbox["width"] * bbox["height"]
                stats["areas"].append(area)
                
                # サイズ（幅x高さ）
                stats["sizes"].append((bbox["width"], bbox["height"]))
                
                # 位置（画像内の相対位置）
                img_w, img_h = label["image_size"]
                rel_x = (bbox["x"] + bbox["width"]/2) / max(img_w, 1)
                rel_y = (bbox["y"] + bbox["height"]/2) / max(img_h, 1)
                stats["positions"].append((rel_x, rel_y))
                
                # 塗りつぶし比率
                if "fill_ratio" in char_region:
                    stats["fill_ratios"].append(char_region["fill_ratio"])
        
        # 統計値計算
        if stats["aspect_ratios"]:
            stats["aspect_ratio_stats"] = {
                "mean": np.mean(stats["aspect_ratios"]),
                "std": np.std(stats["aspect_ratios"]),
                "min": np.min(stats["aspect_ratios"]),
                "max": np.max(stats["aspect_ratios"]),
                "median": np.median(stats["aspect_ratios"])
            }
        
        if stats["areas"]:
            stats["area_stats"] = {
                "mean": np.mean(stats["areas"]),
                "std": np.std(stats["areas"]),
                "min": np.min(stats["areas"]),
                "max": np.max(stats["areas"]),
                "median": np.median(stats["areas"])
            }
        
        logger.info(f"キャラクター領域分析完了: {stats['files_with_character']}/{stats['total_files']}ファイル")
        
        return stats
    
    def analyze_panel_patterns(self) -> Dict[str, Any]:
        """
        コマパターンの分析
        
        Returns:
            コマパターン分析結果
        """
        panel_stats = {
            "panel_sizes": [],
            "panel_positions": [],
            "character_to_panel_ratios": [],
            "panel_types": defaultdict(int)
        }
        
        for label in self.labels_data:
            if label.get("largest_panel_box") and label.get("character_region"):
                panel = label["largest_panel_box"]["bbox"]
                char = label["character_region"]["bbox"]
                
                # パネルサイズ
                panel_area = panel["width"] * panel["height"]
                panel_stats["panel_sizes"].append(panel_area)
                
                # パネル内でのキャラクター比率
                char_area = char["width"] * char["height"]
                char_to_panel_ratio = char_area / max(panel_area, 1)
                panel_stats["character_to_panel_ratios"].append(char_to_panel_ratio)
                
                # パネルタイプ分類（アスペクト比ベース）
                panel_aspect = panel["height"] / max(panel["width"], 1)
                if panel_aspect > 1.5:
                    panel_stats["panel_types"]["vertical"] += 1
                elif panel_aspect < 0.7:
                    panel_stats["panel_types"]["horizontal"] += 1
                else:
                    panel_stats["panel_types"]["square"] += 1
        
        return panel_stats
    
    def classify_character_types(self) -> Dict[str, Any]:
        """
        キャラクター抽出タイプの分類
        
        Returns:
            キャラクタータイプ分析結果
        """
        type_stats = {
            "full_body": 0,      # 全身
            "bust_up": 0,        # バストアップ
            "face_close": 0,     # 顔アップ
            "other": 0
        }
        
        for label in self.labels_data:
            if label.get("character_region"):
                char_region = label["character_region"]
                aspect_ratio = char_region.get("aspect_ratio", 1.0)
                area = char_region.get("area", 0)
                
                # アスペクト比と面積で分類
                if aspect_ratio > 2.0 and area > 50000:
                    type_stats["full_body"] += 1
                elif 1.2 <= aspect_ratio <= 2.0:
                    type_stats["bust_up"] += 1
                elif aspect_ratio < 1.2:
                    type_stats["face_close"] += 1
                else:
                    type_stats["other"] += 1
        
        return type_stats
    
    def generate_learning_insights(self) -> Dict[str, Any]:
        """
        学習のためのインサイト生成
        
        Returns:
            学習用インサイト
        """
        char_stats = self.analyze_character_regions()
        panel_stats = self.analyze_panel_patterns()
        type_stats = self.classify_character_types()
        
        insights = {
            "dataset_summary": {
                "total_labels": char_stats["total_files"],
                "successful_extractions": char_stats["files_with_character"],
                "success_rate": char_stats["files_with_character"] / max(char_stats["total_files"], 1)
            },
            
            "character_features": {
                "typical_aspect_ratio": char_stats.get("aspect_ratio_stats", {}).get("mean", 1.5),
                "size_variation": char_stats.get("area_stats", {}).get("std", 0),
                "position_distribution": char_stats["positions"] if char_stats["positions"] else []
            },
            
            "extraction_patterns": {
                "character_types": type_stats,
                "dominant_type": max(type_stats.items(), key=lambda x: x[1])[0] if type_stats else "unknown"
            },
            
            "learning_recommendations": []
        }
        
        # 学習推奨事項の生成
        if char_stats["files_with_character"] > 80:
            insights["learning_recommendations"].append(
                "豊富なラベルデータ（80+）が利用可能。深層学習アプローチが有効。"
            )
        
        if char_stats.get("aspect_ratio_stats", {}).get("std", 0) > 0.5:
            insights["learning_recommendations"].append(
                "アスペクト比の変動が大きい。多様な抽出範囲に対応する必要がある。"
            )
        
        if type_stats.get("full_body", 0) > type_stats.get("face_close", 0):
            insights["learning_recommendations"].append(
                "全身抽出が多い。身体検出機能の強化が重要。"
            )
        
        return insights
    
    def create_visualization_report(self, output_dir: Path):
        """
        可視化レポート生成
        
        Args:
            output_dir: 出力ディレクトリ
        """
        try:
            output_dir.mkdir(exist_ok=True)
            
            char_stats = self.analyze_character_regions()
            
            # アスペクト比分布
            if char_stats["aspect_ratios"]:
                plt.figure(figsize=(10, 6))
                plt.hist(char_stats["aspect_ratios"], bins=20, alpha=0.7, color='blue', edgecolor='black')
                plt.xlabel('Aspect Ratio (Height/Width)')
                plt.ylabel('Frequency')
                plt.title('Character Region Aspect Ratio Distribution')
                plt.grid(True, alpha=0.3)
                plt.savefig(output_dir / "aspect_ratio_distribution.png", dpi=150, bbox_inches='tight')
                plt.close()
            
            # 面積分布
            if char_stats["areas"]:
                plt.figure(figsize=(10, 6))
                plt.hist(char_stats["areas"], bins=20, alpha=0.7, color='green', edgecolor='black')
                plt.xlabel('Area (pixels²)')
                plt.ylabel('Frequency')
                plt.title('Character Region Area Distribution')
                plt.grid(True, alpha=0.3)
                plt.savefig(output_dir / "area_distribution.png", dpi=150, bbox_inches='tight')
                plt.close()
            
            # 位置分布（散布図）
            if char_stats["positions"]:
                positions = np.array(char_stats["positions"])
                plt.figure(figsize=(8, 8))
                plt.scatter(positions[:, 0], positions[:, 1], alpha=0.6, color='red')
                plt.xlabel('Relative X Position')
                plt.ylabel('Relative Y Position')
                plt.title('Character Position Distribution in Images')
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                plt.grid(True, alpha=0.3)
                plt.savefig(output_dir / "position_distribution.png", dpi=150, bbox_inches='tight')
                plt.close()
            
            logger.info(f"可視化レポート生成完了: {output_dir}")
            
        except Exception as e:
            logger.error(f"可視化レポート生成エラー: {e}")
    
    def export_analysis_report(self, output_path: Path):
        """
        総合分析レポートをJSONで出力
        
        Args:
            output_path: 出力ファイルパス
        """
        try:
            insights = self.generate_learning_insights()
            char_stats = self.analyze_character_regions()
            panel_stats = self.analyze_panel_patterns()
            type_stats = self.classify_character_types()
            
            report = {
                "analysis_summary": insights,
                "character_statistics": char_stats,
                "panel_statistics": panel_stats,
                "character_types": type_stats,
                "analysis_timestamp": str(Path(__file__).stat().st_mtime)
            }
            
            # NumPy配列をリストに変換
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                return obj
            
            report = convert_numpy(report)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"分析レポート出力完了: {output_path}")
            
        except Exception as e:
            logger.error(f"分析レポート出力エラー: {e}")


def main():
    """メイン処理"""
    logging.basicConfig(level=logging.INFO)
    
    # 入力・出力パス
    labels_path = Path("/mnt/c/AItools/segment-anything/extracted_labels.json")
    output_dir = Path("/mnt/c/AItools/segment-anything/label_analysis")
    report_path = output_dir / "analysis_report.json"
    
    # 分析実行
    analyzer = LabelDataAnalyzer(labels_path)
    
    # 分析結果出力
    analyzer.export_analysis_report(report_path)
    analyzer.create_visualization_report(output_dir)
    
    # サマリー表示
    insights = analyzer.generate_learning_insights()
    
    print("\n📊 ラベルデータ分析結果:")
    print(f"  総ファイル数: {insights['dataset_summary']['total_labels']}")
    print(f"  成功抽出数: {insights['dataset_summary']['successful_extractions']}")
    print(f"  成功率: {insights['dataset_summary']['success_rate']*100:.1f}%")
    
    print(f"\n🎯 キャラクター特徴:")
    print(f"  平均アスペクト比: {insights['character_features']['typical_aspect_ratio']:.2f}")
    print(f"  支配的タイプ: {insights['extraction_patterns']['dominant_type']}")
    
    print(f"\n💡 学習推奨事項:")
    for rec in insights['learning_recommendations']:
        print(f"  - {rec}")
    
    print(f"\n📁 出力ファイル:")
    print(f"  分析レポート: {report_path}")
    print(f"  可視化ファイル: {output_dir}")


if __name__ == "__main__":
    main()