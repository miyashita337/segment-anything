#!/usr/bin/env python3
"""
SAM+YOLO精度向上パラメータ最適化システム
正解データに基づく最適パラメータ探索と調整
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import subprocess
import sys

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SAMYOLOParameterOptimizer:
    """SAM+YOLOパラメータ最適化器"""
    
    def __init__(self, correct_annotations_path: Path, original_dir: Path, output_dir: Path):
        self.correct_annotations_path = Path(correct_annotations_path)
        self.original_dir = Path(original_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 正解データ読み込み
        with open(self.correct_annotations_path, 'r', encoding='utf-8') as f:
            self.correct_data = json.load(f)
        
        # 最適化対象パラメータ定義
        self.parameter_sets = [
            {
                'name': 'current_default',
                'description': '現在のデフォルト設定',
                'yolo_conf_threshold': 0.03,
                'yolo_iou_threshold': 0.45,
                'sam_points_per_side': 32,
                'quality_method': 'fullbody_priority',
                'area_weight': 0.2,
                'fullbody_weight': 0.4,
                'central_weight': 0.15,
                'grounded_weight': 0.15,
                'confidence_weight': 0.1
            },
            {
                'name': 'high_precision',
                'description': '高精度設定（YOLO閾値上昇）',
                'yolo_conf_threshold': 0.1,
                'yolo_iou_threshold': 0.3,
                'sam_points_per_side': 64,
                'quality_method': 'fullbody_priority',
                'area_weight': 0.3,
                'fullbody_weight': 0.5,
                'central_weight': 0.1,
                'grounded_weight': 0.05,
                'confidence_weight': 0.05
            },
            {
                'name': 'large_character_focus',
                'description': '大キャラ重視設定',
                'yolo_conf_threshold': 0.05,
                'yolo_iou_threshold': 0.4,
                'sam_points_per_side': 32,
                'quality_method': 'size_priority',
                'area_weight': 0.6,
                'fullbody_weight': 0.2,
                'central_weight': 0.1,
                'grounded_weight': 0.05,
                'confidence_weight': 0.05
            },
            {
                'name': 'balanced_optimized',
                'description': 'バランス最適化設定',
                'yolo_conf_threshold': 0.07,
                'yolo_iou_threshold': 0.35,
                'sam_points_per_side': 48,
                'quality_method': 'balanced',
                'area_weight': 0.25,
                'fullbody_weight': 0.35,
                'central_weight': 0.2,
                'grounded_weight': 0.1,
                'confidence_weight': 0.1
            }
        ]
    
    def test_parameter_set(self, param_set: Dict, test_images: List[str], max_images: int = 5) -> Dict:
        """パラメータセットのテスト実行"""
        logger.info(f"🧪 パラメータセット '{param_set['name']}' テスト開始")
        logger.info(f"   {param_set['description']}")
        
        # テスト用出力ディレクトリ
        test_output_dir = self.output_dir / f"param_test_{param_set['name']}"
        test_output_dir.mkdir(exist_ok=True)
        
        results = {
            'parameter_set': param_set,
            'test_results': {},
            'summary': {
                'total_tested': 0,
                'successful_extractions': 0,
                'total_iou': 0.0,
                'grade_counts': {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
            }
        }
        
        # 限定数の画像でテスト
        test_subset = test_images[:max_images]
        
        for i, filename in enumerate(test_subset, 1):
            logger.info(f"  [{i}/{len(test_subset)}] テスト実行: {filename}")
            
            base_name = filename.replace('.jpg', '').replace('.png', '')
            input_path = self.original_dir / filename
            output_path = test_output_dir / f"{base_name}_test_extracted.jpg"
            
            # 個別画像でextract_character.py実行
            extraction_result = self.run_single_extraction(
                input_path, output_path, param_set
            )
            
            if extraction_result['success']:
                # IoU計算
                iou_score = self.calculate_extraction_iou(filename, output_path)
                grade = self.get_quality_grade(iou_score)
                
                results['test_results'][filename] = {
                    'extraction_success': True,
                    'iou_score': iou_score,
                    'quality_grade': grade,
                    'output_path': str(output_path)
                }
                
                results['summary']['successful_extractions'] += 1
                results['summary']['total_iou'] += iou_score
                results['summary']['grade_counts'][grade] += 1
                
                logger.info(f"     ✅ 成功 - IoU: {iou_score:.3f}, Grade: {grade}")
            else:
                results['test_results'][filename] = {
                    'extraction_success': False,
                    'error': extraction_result.get('error', 'Unknown error'),
                    'iou_score': 0.0,
                    'quality_grade': 'F'
                }
                results['summary']['grade_counts']['F'] += 1
                logger.info(f"     ❌ 失敗 - {extraction_result.get('error', 'Unknown error')}")
            
            results['summary']['total_tested'] += 1
        
        # 平均IoU計算
        if results['summary']['successful_extractions'] > 0:
            results['summary']['average_iou'] = results['summary']['total_iou'] / results['summary']['successful_extractions']
        else:
            results['summary']['average_iou'] = 0.0
        
        # 成功率計算
        results['summary']['success_rate'] = results['summary']['successful_extractions'] / results['summary']['total_tested']
        results['summary']['ab_success_rate'] = (results['summary']['grade_counts']['A'] + results['summary']['grade_counts']['B']) / results['summary']['total_tested']
        
        return results
    
    def run_single_extraction(self, input_path: Path, output_path: Path, param_set: Dict) -> Dict:
        """単一画像での抽出実行"""
        try:
            # extract_character.pyの実行コマンド構築
            cmd = [
                sys.executable,
                "features/extraction/commands/extract_character.py",
                str(input_path),
                "-o", str(output_path),
                "--verbose"
            ]
            
            # 環境変数でパラメータ設定
            import os
            env = dict(os.environ)
            env.update({
                'YOLO_CONF_THRESHOLD': str(param_set['yolo_conf_threshold']),
                'YOLO_IOU_THRESHOLD': str(param_set['yolo_iou_threshold']),
                'SAM_POINTS_PER_SIDE': str(param_set['sam_points_per_side']),
                'QUALITY_METHOD': param_set['quality_method'],
                'AREA_WEIGHT': str(param_set['area_weight']),
                'FULLBODY_WEIGHT': str(param_set['fullbody_weight']),
                'CENTRAL_WEIGHT': str(param_set['central_weight']),
                'GROUNDED_WEIGHT': str(param_set['grounded_weight']),
                'CONFIDENCE_WEIGHT': str(param_set['confidence_weight'])
            })
            
            # 実行
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
            
            if result.returncode == 0 and output_path.exists():
                return {'success': True, 'stdout': result.stdout}
            else:
                return {
                    'success': False,
                    'error': result.stderr or 'Unknown extraction error',
                    'stdout': result.stdout,
                    'returncode': result.returncode
                }
                
        except subprocess.TimeoutExpired:\n            return {'success': False, 'error': 'Extraction timeout (120s)'}\n        except Exception as e:\n            return {'success': False, 'error': str(e)}\n    \n    def calculate_extraction_iou(self, filename: str, extracted_path: Path) -> float:\n        \"\"\"抽出結果のIoU計算\"\"\"\n        try:\n            if filename not in self.correct_data['annotations']:\n                return 0.0\n            \n            correct_rect = self.correct_data['annotations'][filename]['primary_rectangle']\n            if not correct_rect:\n                return 0.0\n            \n            # 抽出画像から推定境界計算（前回と同じロジック）\n            original_path = self.original_dir / filename\n            estimated_bounds = self.get_extracted_image_bounds(extracted_path, original_path)\n            \n            if not estimated_bounds:\n                return 0.0\n            \n            # IoU計算\n            return self.calculate_iou(correct_rect, estimated_bounds)\n            \n        except Exception as e:\n            logger.error(f\"IoU計算エラー {filename}: {e}\")\n            return 0.0\n    \n    def calculate_iou(self, box1: Dict, box2: Dict) -> float:\n        \"\"\"IoU（交差率）計算\"\"\"\n        try:\n            x1_1, y1_1 = box1['x'], box1['y']\n            x2_1, y2_1 = x1_1 + box1['width'], y1_1 + box1['height']\n            \n            x1_2, y1_2 = box2['x'], box2['y']\n            x2_2, y2_2 = x1_2 + box2['width'], y1_2 + box2['height']\n            \n            # 交差領域\n            x1_i = max(x1_1, x1_2)\n            y1_i = max(y1_1, y1_2)\n            x2_i = min(x2_1, x2_2)\n            y2_i = min(y2_1, y2_2)\n            \n            if x2_i <= x1_i or y2_i <= y1_i:\n                return 0.0\n            \n            intersection = (x2_i - x1_i) * (y2_i - y1_i)\n            area1 = box1['width'] * box1['height']\n            area2 = box2['width'] * box2['height']\n            union = area1 + area2 - intersection\n            \n            return intersection / union if union > 0 else 0.0\n            \n        except Exception as e:\n            return 0.0\n    \n    def get_extracted_image_bounds(self, extracted_path: Path, original_path: Path) -> Optional[Dict]:\n        \"\"\"抽出画像から元画像での推定境界を計算（前回と同じロジック）\"\"\"\n        try:\n            if not extracted_path.exists() or not original_path.exists():\n                return None\n            \n            extracted = cv2.imread(str(extracted_path))\n            original = cv2.imread(str(original_path))\n            \n            if extracted is None or original is None:\n                return None\n            \n            orig_h, orig_w = original.shape[:2]\n            ext_h, ext_w = extracted.shape[:2]\n            \n            aspect_ratio = ext_w / ext_h if ext_h > 0 else 1.0\n            \n            if aspect_ratio > 1.0:\n                estimated_w = int(orig_w * 0.7)\n                estimated_h = int(estimated_w / aspect_ratio)\n            else:\n                estimated_h = int(orig_h * 0.7)\n                estimated_w = int(estimated_h * aspect_ratio)\n            \n            estimated_x = (orig_w - estimated_w) // 2\n            estimated_y = (orig_h - estimated_h) // 2\n            \n            return {\n                'x': max(0, estimated_x),\n                'y': max(0, estimated_y),\n                'width': min(estimated_w, orig_w),\n                'height': min(estimated_h, orig_h)\n            }\n            \n        except Exception as e:\n            return None\n    \n    def get_quality_grade(self, iou_score: float) -> str:\n        \"\"\"IoUスコアから品質グレード判定\"\"\"\n        if iou_score >= 0.7:\n            return 'A'\n        elif iou_score >= 0.5:\n            return 'B'\n        elif iou_score >= 0.3:\n            return 'C'\n        elif iou_score >= 0.1:\n            return 'D'\n        else:\n            return 'F'\n    \n    def optimize_parameters(self) -> Dict:\n        \"\"\"パラメータ最適化実行\"\"\"\n        logger.info(\"🚀 SAM+YOLOパラメータ最適化開始\")\n        \n        # テスト対象画像選択（代表的な画像を選ぶ）\n        test_images = [\n            'kana08_0001.jpg',  # 大きなキャラ\n            'kana08_0002.jpg',  # 複数キャラ\n            'kana08_0008.jpg',  # 複雑レイアウト\n            'kana08_0010.jpg',  # 全身キャラ\n            'kana08_0016.jpg'   # 横長レイアウト\n        ]\n        \n        optimization_results = {\n            'metadata': {\n                'test_images': test_images,\n                'parameter_sets_tested': len(self.parameter_sets)\n            },\n            'parameter_test_results': {},\n            'best_parameter_set': None,\n            'recommendations': []\n        }\n        \n        best_score = 0.0\n        best_param_set = None\n        \n        # 各パラメータセットをテスト\n        for param_set in self.parameter_sets:\n            test_result = self.test_parameter_set(param_set, test_images)\n            optimization_results['parameter_test_results'][param_set['name']] = test_result\n            \n            # 最良パラメータセット判定（A+B成功率重視）\n            score = test_result['summary']['ab_success_rate']\n            if score > best_score:\n                best_score = score\n                best_param_set = param_set\n        \n        optimization_results['best_parameter_set'] = best_param_set\n        \n        # 推奨事項生成\n        self.generate_recommendations(optimization_results)\n        \n        return optimization_results\n    \n    def generate_recommendations(self, results: Dict):\n        \"\"\"推奨事項生成\"\"\"\n        recommendations = []\n        \n        # 最良パラメータの推奨\n        if results['best_parameter_set']:\n            best = results['best_parameter_set']\n            recommendations.append(f\"最適パラメータセット: {best['name']} - {best['description']}\")\n        \n        # 個別パラメータの分析\n        param_analysis = {}\n        for name, result in results['parameter_test_results'].items():\n            param_set = result['parameter_set']\n            summary = result['summary']\n            \n            param_analysis[name] = {\n                'ab_success_rate': summary['ab_success_rate'],\n                'average_iou': summary['average_iou'],\n                'yolo_conf': param_set['yolo_conf_threshold'],\n                'area_weight': param_set['area_weight']\n            }\n        \n        # 傾向分析\n        high_conf_results = [v for k, v in param_analysis.items() if v['yolo_conf'] >= 0.07]\n        low_conf_results = [v for k, v in param_analysis.items() if v['yolo_conf'] < 0.07]\n        \n        if high_conf_results and low_conf_results:\n            high_avg = sum(r['ab_success_rate'] for r in high_conf_results) / len(high_conf_results)\n            low_avg = sum(r['ab_success_rate'] for r in low_conf_results) / len(low_conf_results)\n            \n            if high_avg > low_avg:\n                recommendations.append(\"YOLO信頼度閾値を高めに設定することを推奨\")\n            else:\n                recommendations.append(\"YOLO信頼度閾値を低めに設定することを推奨\")\n        \n        results['recommendations'] = recommendations\n    \n    def save_optimization_results(self, results: Dict) -> Path:\n        \"\"\"最適化結果保存\"\"\"\n        output_file = self.output_dir / \"parameter_optimization_results.json\"\n        \n        with open(output_file, 'w', encoding='utf-8') as f:\n            json.dump(results, f, ensure_ascii=False, indent=2)\n        \n        logger.info(f\"📄 最適化結果保存: {output_file}\")\n        return output_file\n\nimport os  # 追加\n\ndef main():\n    \"\"\"メイン実行\"\"\"\n    # パス設定\n    correct_annotations_path = Path(\"C:/AItools/lora/train/yado/tracker-workspace/P1-B004/analysis/correct_annotations.json\")\n    original_dir = Path(\"C:/AItools/lora/train/yado/org/kana08\")\n    output_dir = Path(\"C:/AItools/lora/train/yado/tracker-workspace/P1-B004/optimization\")\n    \n    # 最適化器初期化\n    optimizer = SAMYOLOParameterOptimizer(correct_annotations_path, original_dir, output_dir)\n    \n    # パラメータ最適化実行\n    results = optimizer.optimize_parameters()\n    \n    # 結果保存\n    output_file = optimizer.save_optimization_results(results)\n    \n    # サマリー表示\n    logger.info(\"=\" * 60)\n    logger.info(\"🎯 SAM+YOLOパラメータ最適化完了\")\n    \n    if results['best_parameter_set']:\n        best = results['best_parameter_set']\n        best_result = results['parameter_test_results'][best['name']]\n        logger.info(f\"🏆 最適パラメータ: {best['name']}\")\n        logger.info(f\"📊 A+B成功率: {best_result['summary']['ab_success_rate']:.1%}\")\n        logger.info(f\"📈 平均IoU: {best_result['summary']['average_iou']:.3f}\")\n    \n    logger.info(f\"📄 詳細結果: {output_file}\")\n    logger.info(\"=\" * 60)\n    \n    return 0\n\nif __name__ == \"__main__\":\n    exit(main())