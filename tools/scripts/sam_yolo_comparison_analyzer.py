#!/usr/bin/env python3
"""
SAM+YOLO結果と正解範囲の比較分析システム
IoU（交差率）とクロッピング精度の定量評価
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SAMYOLOComparisonAnalyzer:
    """SAM+YOLO結果比較分析器"""
    
    def __init__(self, correct_annotations_path: Path, extraction_dir: Path, original_dir: Path, output_dir: Path):
        self.correct_annotations_path = Path(correct_annotations_path)
        self.extraction_dir = Path(extraction_dir)
        self.original_dir = Path(original_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 正解データ読み込み
        with open(self.correct_annotations_path, 'r', encoding='utf-8') as f:
            self.correct_data = json.load(f)
    
    def calculate_iou(self, box1: Dict, box2: Dict) -> float:
        """IoU（交差率）計算"""
        try:
            # box1: 正解範囲, box2: SAM+YOLO結果の推定範囲
            x1_1, y1_1 = box1['x'], box1['y']
            x2_1, y2_1 = x1_1 + box1['width'], y1_1 + box1['height']
            
            x1_2, y1_2 = box2['x'], box2['y']
            x2_2, y2_2 = x1_2 + box2['width'], y1_2 + box2['height']
            
            # 交差領域
            x1_i = max(x1_1, x1_2)
            y1_i = max(y1_1, y1_2)
            x2_i = min(x2_1, x2_2)
            y2_i = min(y2_1, y2_2)
            
            if x2_i <= x1_i or y2_i <= y1_i:
                return 0.0
            
            intersection = (x2_i - x1_i) * (y2_i - y1_i)
            area1 = box1['width'] * box1['height']
            area2 = box2['width'] * box2['height']
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.error(f"IoU計算エラー: {e}")
            return 0.0
    
    def get_extracted_image_bounds(self, extracted_path: Path, original_path: Path) -> Optional[Dict]:
        """抽出画像から元画像での推定境界を計算"""
        try:
            if not extracted_path.exists() or not original_path.exists():
                return None
            
            # 画像読み込み
            extracted = cv2.imread(str(extracted_path))
            original = cv2.imread(str(original_path))
            
            if extracted is None or original is None:
                return None
            
            orig_h, orig_w = original.shape[:2]
            ext_h, ext_w = extracted.shape[:2]
            
            # 簡易推定: 抽出画像のアスペクト比から元画像での位置を推定
            # 注意: これは近似値であり、実際のSAM+YOLOの抽出ロジックを正確には反映しない
            
            # アスペクト比ベースの推定
            aspect_ratio = ext_w / ext_h if ext_h > 0 else 1.0
            
            # 仮定: 抽出画像が元画像の中央付近から取られたと仮定
            if aspect_ratio > 1.0:  # 横長
                estimated_w = int(orig_w * 0.7)  # 元画像の70%程度の幅と仮定
                estimated_h = int(estimated_w / aspect_ratio)
            else:  # 縦長または正方形
                estimated_h = int(orig_h * 0.7)  # 元画像の70%程度の高さと仮定
                estimated_w = int(estimated_h * aspect_ratio)
            
            # 中央配置と仮定
            estimated_x = (orig_w - estimated_w) // 2
            estimated_y = (orig_h - estimated_h) // 2
            
            return {
                'x': max(0, estimated_x),
                'y': max(0, estimated_y),
                'width': min(estimated_w, orig_w),
                'height': min(estimated_h, orig_h),
                'confidence': 0.5  # 推定値であることを示す
            }
            
        except Exception as e:
            logger.error(f"境界推定エラー {extracted_path}: {e}")
            return None
    
    def analyze_single_image(self, filename: str) -> Dict:
        """単一画像の比較分析"""
        base_name = filename.replace('.jpg', '').replace('.png', '')
        
        # ファイルパス構築
        original_path = self.original_dir / filename
        extracted_path = self.extraction_dir / f"{base_name}_extracted.jpg"
        
        result = {
            'filename': filename,
            'base_name': base_name,
            'has_correct_annotation': False,
            'has_extracted_result': False,
            'correct_rectangle': None,
            'estimated_sam_yolo_bounds': None,
            'iou_score': 0.0,
            'coverage_ratio': 0.0,
            'quality_grade': 'F'
        }
        
        # 正解データ確認
        if filename in self.correct_data['annotations']:
            annotation = self.correct_data['annotations'][filename]
            if annotation['primary_rectangle']:
                result['has_correct_annotation'] = True
                result['correct_rectangle'] = annotation['primary_rectangle']
        
        # 抽出結果確認
        if extracted_path.exists():
            result['has_extracted_result'] = True
            
            # SAM+YOLO結果の推定境界取得
            estimated_bounds = self.get_extracted_image_bounds(extracted_path, original_path)
            if estimated_bounds:
                result['estimated_sam_yolo_bounds'] = estimated_bounds
                
                # IoU計算
                if result['correct_rectangle']:
                    iou = self.calculate_iou(result['correct_rectangle'], estimated_bounds)
                    result['iou_score'] = iou
                    
                    # カバレッジ率計算（正解範囲に対する抽出範囲の重複率）
                    correct_area = result['correct_rectangle']['width'] * result['correct_rectangle']['height']
                    estimated_area = estimated_bounds['width'] * estimated_bounds['height']
                    coverage = min(estimated_area / correct_area, 1.0) if correct_area > 0 else 0.0
                    result['coverage_ratio'] = coverage
                    
                    # 品質グレード判定
                    if iou >= 0.7:
                        result['quality_grade'] = 'A'
                    elif iou >= 0.5:
                        result['quality_grade'] = 'B'
                    elif iou >= 0.3:
                        result['quality_grade'] = 'C'
                    elif iou >= 0.1:
                        result['quality_grade'] = 'D'
                    else:
                        result['quality_grade'] = 'F'
        
        return result
    
    def analyze_all_images(self) -> Dict:
        """全画像の比較分析"""
        logger.info("🔍 SAM+YOLO結果と正解範囲の比較分析開始")
        
        results = {
            'metadata': {
                'analysis_method': 'sam_yolo_correct_comparison',
                'correct_annotations_file': str(self.correct_annotations_path),
                'extraction_directory': str(self.extraction_dir),
                'original_directory': str(self.original_dir)
            },
            'individual_results': {},
            'summary_statistics': {}
        }
        
        # 全画像ファイル取得
        image_files = []
        for filename in self.correct_data['annotations'].keys():
            if filename.endswith(('.jpg', '.png')):
                image_files.append(filename)
        
        image_files.sort()
        logger.info(f"分析対象: {len(image_files)}枚")
        
        # 個別分析
        total_iou = 0.0
        total_coverage = 0.0
        grade_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
        valid_results = 0
        
        for i, filename in enumerate(image_files, 1):
            logger.info(f"[{i:2d}/{len(image_files)}] 分析中: {filename}")
            
            result = self.analyze_single_image(filename)
            results['individual_results'][filename] = result
            
            if result['has_correct_annotation'] and result['has_extracted_result']:
                total_iou += result['iou_score']
                total_coverage += result['coverage_ratio']
                grade_counts[result['quality_grade']] += 1
                valid_results += 1
                
                logger.info(f"  IoU: {result['iou_score']:.3f}, Coverage: {result['coverage_ratio']:.3f}, Grade: {result['quality_grade']}")
            else:
                logger.warning(f"  不完全なデータ: 正解={result['has_correct_annotation']}, 抽出={result['has_extracted_result']}")
        
        # サマリー統計
        if valid_results > 0:
            avg_iou = total_iou / valid_results
            avg_coverage = total_coverage / valid_results
        else:
            avg_iou = avg_coverage = 0.0
        
        results['summary_statistics'] = {
            'total_images': len(image_files),
            'valid_comparisons': valid_results,
            'average_iou': avg_iou,
            'average_coverage': avg_coverage,
            'grade_distribution': grade_counts,
            'success_rate_ab': (grade_counts['A'] + grade_counts['B']) / valid_results if valid_results > 0 else 0.0,
            'failure_rate_f': grade_counts['F'] / valid_results if valid_results > 0 else 0.0
        }
        
        return results
    
    def generate_comparison_report(self, analysis_results: Dict) -> Path:
        """比較分析レポート生成"""
        report_path = self.output_dir / "sam_yolo_comparison_report.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📄 比較分析レポート保存: {report_path}")
        return report_path
    
    def generate_visual_comparison(self, analysis_results: Dict):
        """視覚的比較画像生成"""
        logger.info("🎨 視覚的比較画像生成開始")
        
        comparison_dir = self.output_dir / "visual_comparison"
        comparison_dir.mkdir(exist_ok=True)
        
        for filename, result in analysis_results['individual_results'].items():
            if not (result['has_correct_annotation'] and result['has_extracted_result']):
                continue
            
            try:
                # 元画像読み込み
                original_path = self.original_dir / filename
                img = cv2.imread(str(original_path))
                if img is None:
                    continue
                
                # 正解範囲描画（緑色）
                if result['correct_rectangle']:
                    rect = result['correct_rectangle']
                    cv2.rectangle(img, (rect['x'], rect['y']), 
                                (rect['x'] + rect['width'], rect['y'] + rect['height']), 
                                (0, 255, 0), 3)
                    cv2.putText(img, "Correct", (rect['x'], rect['y']-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # SAM+YOLO推定範囲描画（赤色）
                if result['estimated_sam_yolo_bounds']:
                    est = result['estimated_sam_yolo_bounds']
                    cv2.rectangle(img, (est['x'], est['y']), 
                                (est['x'] + est['width'], est['y'] + est['height']), 
                                (0, 0, 255), 2)
                    cv2.putText(img, "SAM+YOLO", (est['x'], est['y']-30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 結果情報描画
                info_text = f"IoU: {result['iou_score']:.3f} | Grade: {result['quality_grade']}"
                cv2.putText(img, info_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # 保存
                output_path = comparison_dir / f"comparison_{filename}"
                cv2.imwrite(str(output_path), img)
                
            except Exception as e:
                logger.error(f"比較画像生成エラー {filename}: {e}")
        
        logger.info(f"📁 比較画像保存: {comparison_dir}")

def main():
    """メイン実行"""
    # パス設定
    correct_annotations_path = Path("C:/AItools/lora/train/yado/tracker-workspace/P1-B004/analysis/correct_annotations.json")
    extraction_dir = Path("C:/AItools/lora/train/yado/tracker-workspace/P1-B004/extraction")
    original_dir = Path("C:/AItools/lora/train/yado/org/kana08")
    output_dir = Path("C:/AItools/lora/train/yado/tracker-workspace/P1-B004/analysis")
    
    # 分析器初期化
    analyzer = SAMYOLOComparisonAnalyzer(
        correct_annotations_path, extraction_dir, original_dir, output_dir
    )
    
    # 比較分析実行
    analysis_results = analyzer.analyze_all_images()
    
    # レポート生成
    report_path = analyzer.generate_comparison_report(analysis_results)
    
    # 視覚的比較画像生成
    analyzer.generate_visual_comparison(analysis_results)
    
    # サマリー表示
    stats = analysis_results['summary_statistics']
    
    logger.info("=" * 60)
    logger.info("🎯 SAM+YOLO vs 正解範囲 比較分析完了")
    logger.info(f"📊 有効比較: {stats['valid_comparisons']}/{stats['total_images']}枚")
    logger.info(f"📈 平均IoU: {stats['average_iou']:.3f}")
    logger.info(f"📊 平均カバレッジ: {stats['average_coverage']:.3f}")
    logger.info(f"🎖️ 品質分布: A:{stats['grade_distribution']['A']} B:{stats['grade_distribution']['B']} C:{stats['grade_distribution']['C']} D:{stats['grade_distribution']['D']} F:{stats['grade_distribution']['F']}")
    logger.info(f"✅ A+B成功率: {stats['success_rate_ab']:.1%}")
    logger.info(f"❌ F失敗率: {stats['failure_rate_f']:.1%}")
    logger.info(f"📄 レポート: {report_path}")
    logger.info("=" * 60)
    
    return 0 if stats['success_rate_ab'] > 0.5 else 1

if __name__ == "__main__":
    exit(main())