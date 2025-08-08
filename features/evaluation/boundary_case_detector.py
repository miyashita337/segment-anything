#!/usr/bin/env python3
"""
P1-012: 境界例自動検出システム
抽出品質の境界ケース（成功/失敗の境界）を自動特定し、改善点を提示
"""

import numpy as np
import cv2

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.workspace_config import WorkspaceConfig
from features.common.output_path_manager import OutputCategory, OutputPathManager

logger = logging.getLogger(__name__)


class BoundaryType(Enum):
    """境界ケースの種類"""
    POSE_BOUNDARY = "pose_boundary"        # ポーズの境界
    SIZE_BOUNDARY = "size_boundary"        # サイズの境界
    QUALITY_BOUNDARY = "quality_boundary"  # 品質の境界
    CONTEXT_BOUNDARY = "context_boundary"  # 文脈の境界
    TECHNICAL_BOUNDARY = "technical_boundary"  # 技術的境界


@dataclass
class BoundaryCase:
    """境界ケース情報"""
    image_path: str
    case_type: BoundaryType
    confidence_score: float
    quality_metrics: Dict[str, float]
    detection_reason: str
    improvement_suggestions: List[str]
    technical_details: Dict[str, Any]


@dataclass
class BoundaryAnalysisResult:
    """境界例分析結果"""
    total_images: int
    boundary_cases: List[BoundaryCase]
    summary_statistics: Dict[str, Any]
    improvement_priorities: List[Dict[str, Any]]


class BoundaryCaseDetector:
    """境界例自動検出システム"""
    
    # 品質境界しきい値
    QUALITY_THRESHOLDS = {
        'confidence_low': 0.3,      # 信頼度低境界
        'confidence_high': 0.7,     # 信頼度高境界
        'size_small': 5000,         # 小サイズ境界（ピクセル）
        'size_large': 500000,       # 大サイズ境界（ピクセル）
        'aspect_ratio_min': 0.3,    # アスペクト比最小
        'aspect_ratio_max': 3.0,    # アスペクト比最大
    }
    
    def __init__(self, tracker_id: str):
        """
        初期化
        
        Args:
            tracker_id: トラッカーID
        """
        self.tracker_id = tracker_id
        self.path_manager = OutputPathManager(tracker_id)
        self.boundary_cases: List[BoundaryCase] = []
        
        # 統計データ
        self.stats = {
            'total_processed': 0,
            'boundary_cases_found': 0,
            'case_type_counts': {bt.value: 0 for bt in BoundaryType}
        }
        
    def analyze_image_quality(self, image_path: Path) -> Dict[str, float]:
        """
        画像品質を分析
        
        Args:
            image_path: 画像パス
            
        Returns:
            品質メトリクス辞書
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return {'error': 1.0}
                
            # 基本メトリクス計算
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 1. 明度・コントラスト
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # 2. シャープネス（ラプラシアン分散）
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            
            # 3. エッジ密度
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (width * height)
            
            # 4. ノイズレベル（高周波成分）
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            noise_level = np.std(magnitude_spectrum)
            
            # 5. 彩度（カラー画像の場合）
            if len(image.shape) == 3:  # カラー画像チェック
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                saturation = np.mean(hsv[:, :, 1])
            else:
                saturation = 0.0  # グレースケール画像の場合
            
            return {
                'width': float(width),
                'height': float(height),
                'aspect_ratio': float(width / height),
                'total_pixels': float(width * height),
                'brightness': float(brightness / 255),
                'contrast': float(contrast / 255),
                'sharpness': float(min(sharpness / 1000, 1.0)),
                'edge_density': float(edge_density),
                'noise_level': float(min(noise_level / 10, 1.0)),
                'saturation': float(saturation / 255),
            }
            
        except Exception as e:
            logger.error(f"画像品質分析エラー {image_path}: {e}")
            return {'error': 1.0}
            
    def detect_pose_boundary(self, quality_metrics: Dict[str, float]) -> Tuple[bool, str, List[str]]:
        """
        ポーズ境界ケース検出
        
        Args:
            quality_metrics: 品質メトリクス
            
        Returns:
            (境界ケースフラグ, 検出理由, 改善提案リスト)
        """
        suggestions = []
        reasons = []
        
        # アスペクト比による境界検出
        aspect_ratio = quality_metrics.get('aspect_ratio', 1.0)
        if aspect_ratio < self.QUALITY_THRESHOLDS['aspect_ratio_min']:
            reasons.append(f"極端に縦長のアスペクト比: {aspect_ratio:.2f}")
            suggestions.append("横向きポーズまたは全身表示の検討")
        elif aspect_ratio > self.QUALITY_THRESHOLDS['aspect_ratio_max']:
            reasons.append(f"極端に横長のアスペクト比: {aspect_ratio:.2f}")
            suggestions.append("縦向きポーズまたは上半身中心の検討")
            
        # エッジ密度による複雑ポーズ検出
        edge_density = quality_metrics.get('edge_density', 0.5)
        if edge_density > 0.15:
            reasons.append(f"高エッジ密度（複雑ポーズ）: {edge_density:.3f}")
            suggestions.append("シンプルなポーズまたは背景簡素化")
            
        return len(reasons) > 0, "; ".join(reasons), suggestions
        
    def detect_size_boundary(self, quality_metrics: Dict[str, float]) -> Tuple[bool, str, List[str]]:
        """
        サイズ境界ケース検出
        
        Args:
            quality_metrics: 品質メトリクス
            
        Returns:
            (境界ケースフラグ, 検出理由, 改善提案リスト)
        """
        suggestions = []
        reasons = []
        
        total_pixels = quality_metrics.get('total_pixels', 100000)
        
        if total_pixels < self.QUALITY_THRESHOLDS['size_small']:
            reasons.append(f"小サイズ画像: {int(total_pixels)}px")
            suggestions.append("高解像度画像の使用推奨")
        elif total_pixels > self.QUALITY_THRESHOLDS['size_large']:
            reasons.append(f"大サイズ画像: {int(total_pixels)}px")
            suggestions.append("適切なリサイズによる処理効率化")
            
        return len(reasons) > 0, "; ".join(reasons), suggestions
        
    def detect_quality_boundary(self, quality_metrics: Dict[str, float]) -> Tuple[bool, str, List[str]]:
        """
        品質境界ケース検出
        
        Args:
            quality_metrics: 品質メトリクス
            
        Returns:
            (境界ケースフラグ, 検出理由, 改善提案リスト)
        """
        suggestions = []
        reasons = []
        
        # 明度境界
        brightness = quality_metrics.get('brightness', 0.5)
        if brightness < 0.2:
            reasons.append(f"低明度: {brightness:.2f}")
            suggestions.append("画像明度の調整または照明改善")
        elif brightness > 0.8:
            reasons.append(f"高明度: {brightness:.2f}")
            suggestions.append("露出過多の修正")
            
        # コントラスト境界
        contrast = quality_metrics.get('contrast', 0.5)
        if contrast < 0.1:
            reasons.append(f"低コントラスト: {contrast:.2f}")
            suggestions.append("コントラスト強化またはシャープネス調整")
            
        # シャープネス境界
        sharpness = quality_metrics.get('sharpness', 0.5)
        if sharpness < 0.1:
            reasons.append(f"低シャープネス（ぼやけ）: {sharpness:.2f}")
            suggestions.append("フォーカス調整またはシャープネス強化")
            
        # ノイズレベル境界
        noise_level = quality_metrics.get('noise_level', 0.5)
        if noise_level > 0.7:
            reasons.append(f"高ノイズレベル: {noise_level:.2f}")
            suggestions.append("ノイズ除去またはISO設定見直し")
            
        return len(reasons) > 0, "; ".join(reasons), suggestions
        
    def calculate_boundary_confidence(self, quality_metrics: Dict[str, float], 
                                    case_type: BoundaryType) -> float:
        """
        境界ケース信頼度計算
        
        Args:
            quality_metrics: 品質メトリクス
            case_type: 境界ケース種類
            
        Returns:
            信頼度スコア（0-1）
        """
        if case_type == BoundaryType.POSE_BOUNDARY:
            aspect_ratio = quality_metrics.get('aspect_ratio', 1.0)
            edge_density = quality_metrics.get('edge_density', 0.05)
            
            # アスペクト比の極端さ
            aspect_extremity = max(
                abs(aspect_ratio - 1.0) / 2.0,  # 1.0からの乖離
                0.0
            )
            
            # エッジ密度の高さ
            edge_complexity = min(edge_density / 0.2, 1.0)
            
            return min((aspect_extremity + edge_complexity) / 2.0, 1.0)
            
        elif case_type == BoundaryType.SIZE_BOUNDARY:
            total_pixels = quality_metrics.get('total_pixels', 100000)
            
            # サイズの極端さ
            if total_pixels < self.QUALITY_THRESHOLDS['size_small']:
                return min((self.QUALITY_THRESHOLDS['size_small'] - total_pixels) / 
                          self.QUALITY_THRESHOLDS['size_small'], 1.0)
            elif total_pixels > self.QUALITY_THRESHOLDS['size_large']:
                return min((total_pixels - self.QUALITY_THRESHOLDS['size_large']) / 
                          self.QUALITY_THRESHOLDS['size_large'], 1.0)
            return 0.0
            
        elif case_type == BoundaryType.QUALITY_BOUNDARY:
            brightness = quality_metrics.get('brightness', 0.5)
            contrast = quality_metrics.get('contrast', 0.5)
            sharpness = quality_metrics.get('sharpness', 0.5)
            
            # 品質の問題度合い
            brightness_issue = max(abs(brightness - 0.5) - 0.3, 0.0) / 0.2
            contrast_issue = max(0.1 - contrast, 0.0) / 0.1
            sharpness_issue = max(0.1 - sharpness, 0.0) / 0.1
            
            return min((brightness_issue + contrast_issue + sharpness_issue) / 3.0, 1.0)
            
        return 0.5  # デフォルト
        
    def process_image(self, image_path: Path) -> Optional[BoundaryCase]:
        """
        単一画像の境界ケース検出
        
        Args:
            image_path: 画像パス
            
        Returns:
            境界ケース（該当しない場合はNone）
        """
        try:
            # 品質メトリクス計算
            quality_metrics = self.analyze_image_quality(image_path)
            
            if 'error' in quality_metrics:
                return None
                
            # 各種境界ケース検出
            detectors = [
                (BoundaryType.POSE_BOUNDARY, self.detect_pose_boundary),
                (BoundaryType.SIZE_BOUNDARY, self.detect_size_boundary),
                (BoundaryType.QUALITY_BOUNDARY, self.detect_quality_boundary),
            ]
            
            boundary_cases = []
            
            for case_type, detector_func in detectors:
                is_boundary, reason, suggestions = detector_func(quality_metrics)
                
                if is_boundary:
                    confidence = self.calculate_boundary_confidence(quality_metrics, case_type)
                    
                    boundary_case = BoundaryCase(
                        image_path=str(image_path),
                        case_type=case_type,
                        confidence_score=confidence,
                        quality_metrics=quality_metrics,
                        detection_reason=reason,
                        improvement_suggestions=suggestions,
                        technical_details={
                            'detector_version': '1.0',
                            'analysis_timestamp': time.time(),
                        }
                    )
                    
                    boundary_cases.append(boundary_case)
                    
            # 最高信頼度の境界ケースを返す
            if boundary_cases:
                return max(boundary_cases, key=lambda x: x.confidence_score)
            
            return None
            
        except Exception as e:
            logger.error(f"画像処理エラー {image_path}: {e}")
            return None
            
    def process_directory(self, input_dir: Path) -> BoundaryAnalysisResult:
        """
        ディレクトリ内画像の境界ケース検出
        
        Args:
            input_dir: 入力ディレクトリ
            
        Returns:
            境界分析結果
        """
        logger.info(f"🔍 境界ケース検出開始: {input_dir}")
        
        # 画像ファイル収集
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(input_dir.glob(f"*{ext}"))
            image_files.extend(input_dir.glob(f"*{ext.upper()}"))
            
        logger.info(f"📁 対象画像数: {len(image_files)}枚")
        
        # 処理実行
        boundary_cases = []
        processed_count = 0
        
        for image_path in image_files:
            boundary_case = self.process_image(image_path)
            
            if boundary_case:
                boundary_cases.append(boundary_case)
                self.stats['case_type_counts'][boundary_case.case_type.value] += 1
                
            processed_count += 1
            
            if processed_count % 10 == 0:
                logger.info(f"📊 進捗: {processed_count}/{len(image_files)} "
                          f"({len(boundary_cases)}件の境界ケース発見)")
                
        # 統計計算
        self.stats['total_processed'] = processed_count
        self.stats['boundary_cases_found'] = len(boundary_cases)
        
        # 改善優先度計算
        improvement_priorities = self._calculate_improvement_priorities(boundary_cases)
        
        result = BoundaryAnalysisResult(
            total_images=processed_count,
            boundary_cases=boundary_cases,
            summary_statistics=self.stats.copy(),
            improvement_priorities=improvement_priorities
        )
        
        logger.info(f"✅ 境界ケース検出完了: {len(boundary_cases)}/{processed_count}枚 "
                   f"({len(boundary_cases)/max(processed_count, 1)*100:.1f}%)")
                   
        return result
        
    def _calculate_improvement_priorities(self, boundary_cases: List[BoundaryCase]) -> List[Dict[str, Any]]:
        """
        改善優先度計算
        
        Args:
            boundary_cases: 境界ケースリスト
            
        Returns:
            改善優先度リスト
        """
        if not boundary_cases:
            return []
            
        # カテゴリ別集計
        category_stats = {}
        
        for case in boundary_cases:
            case_type = case.case_type.value
            if case_type not in category_stats:
                category_stats[case_type] = {
                    'count': 0,
                    'total_confidence': 0.0,
                    'suggestions': set()
                }
                
            category_stats[case_type]['count'] += 1
            category_stats[case_type]['total_confidence'] += case.confidence_score
            category_stats[case_type]['suggestions'].update(case.improvement_suggestions)
            
        # 優先度計算（件数 × 平均信頼度）
        priorities = []
        
        for case_type, stats in category_stats.items():
            avg_confidence = stats['total_confidence'] / stats['count']
            priority_score = stats['count'] * avg_confidence
            
            priorities.append({
                'case_type': case_type,
                'count': stats['count'],
                'average_confidence': avg_confidence,
                'priority_score': priority_score,
                'improvement_suggestions': list(stats['suggestions']),
            })
            
        # 優先度順でソート
        priorities.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return priorities
        
    def save_results(self, result: BoundaryAnalysisResult, output_dir: Optional[Path] = None) -> Path:
        """
        結果保存
        
        Args:
            result: 分析結果
            output_dir: 出力ディレクトリ（未指定時は自動生成）
            
        Returns:
            結果ファイルパス
        """
        if output_dir is None:
            output_dir = self.path_manager.ensure_output_dir(OutputCategory.QUALITY_REPORT)
            
        # JSON形式で保存
        result_data = {
            'tracker_id': self.tracker_id,
            'analysis_timestamp': time.time(),
            'total_images': result.total_images,
            'boundary_cases_count': len(result.boundary_cases),
            'summary_statistics': result.summary_statistics,
            'improvement_priorities': result.improvement_priorities,
            'boundary_cases': [
                {
                    **asdict(case),
                    'case_type': case.case_type.value  # Enumを文字列に変換
                } for case in result.boundary_cases
            ]
        }
        
        result_file = output_dir / f"{self.tracker_id}_boundary_analysis.json"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"📄 境界分析結果保存: {result_file}")
        
        # サマリーレポート作成
        summary_file = output_dir / f"{self.tracker_id}_boundary_summary.md"
        self._generate_summary_report(result, summary_file)
        
        return result_file
        
    def _generate_summary_report(self, result: BoundaryAnalysisResult, output_file: Path):
        """
        サマリーレポート生成
        
        Args:
            result: 分析結果
            output_file: 出力ファイル
        """
        lines = []
        lines.append(f"# P1-012 境界例自動検出レポート")
        lines.append(f"")
        lines.append(f"**トラッカーID**: {self.tracker_id}")
        lines.append(f"**分析日時**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"")
        
        # サマリー
        lines.append("## 📊 分析サマリー")
        lines.append(f"- **総処理画像数**: {result.total_images}枚")
        lines.append(f"- **境界ケース発見数**: {len(result.boundary_cases)}枚")
        lines.append(f"- **境界ケース率**: {len(result.boundary_cases)/max(result.total_images, 1)*100:.1f}%")
        lines.append("")
        
        # カテゴリ別統計
        lines.append("## 🏷️ カテゴリ別統計")
        for case_type, count in result.summary_statistics['case_type_counts'].items():
            if count > 0:
                lines.append(f"- **{case_type}**: {count}件")
        lines.append("")
        
        # 改善優先度
        if result.improvement_priorities:
            lines.append("## 🎯 改善優先度")
            for i, priority in enumerate(result.improvement_priorities[:5], 1):
                lines.append(f"### {i}. {priority['case_type']} (スコア: {priority['priority_score']:.2f})")
                lines.append(f"- **件数**: {priority['count']}件")
                lines.append(f"- **平均信頼度**: {priority['average_confidence']:.2f}")
                lines.append("- **改善提案**:")
                for suggestion in priority['improvement_suggestions'][:3]:
                    lines.append(f"  - {suggestion}")
                lines.append("")
                
        # 詳細ケース
        if result.boundary_cases:
            lines.append("## 📋 代表的境界ケース")
            # 高信頼度順にソート
            sorted_cases = sorted(result.boundary_cases, key=lambda x: x.confidence_score, reverse=True)
            
            for i, case in enumerate(sorted_cases[:10], 1):
                image_name = Path(case.image_path).name
                lines.append(f"### {i}. {image_name}")
                lines.append(f"- **種類**: {case.case_type.value}")
                lines.append(f"- **信頼度**: {case.confidence_score:.2f}")
                lines.append(f"- **検出理由**: {case.detection_reason}")
                lines.append("- **改善提案**:")
                for suggestion in case.improvement_suggestions:
                    lines.append(f"  - {suggestion}")
                lines.append("")
                
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
            
        logger.info(f"📄 サマリーレポート保存: {output_file}")


def main():
    """メイン実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description="P1-012: 境界例自動検出システム")
    parser.add_argument('--tracker-id', default='P1-012', help='トラッカーID')
    parser.add_argument('--input-dir', type=Path, required=True, help='入力ディレクトリ')
    parser.add_argument('--output-dir', type=Path, help='出力ディレクトリ')
    
    args = parser.parse_args()
    
    # 検出器初期化
    detector = BoundaryCaseDetector(args.tracker_id)
    
    # 境界ケース検出実行
    result = detector.process_directory(args.input_dir)
    
    # 結果保存
    result_file = detector.save_results(result, args.output_dir)
    
    print(f"🎉 P1-012境界例自動検出完了！")
    print(f"📄 結果ファイル: {result_file}")
    print(f"📊 境界ケース: {len(result.boundary_cases)}/{result.total_images}枚")


if __name__ == "__main__":
    main()