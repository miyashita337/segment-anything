#!/usr/bin/env python3
"""
改善版SAMプロンプト戦略システム
複数点・ネガティブプロンプト・境界ボックスを活用した高精度セグメンテーション
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

from segment_anything import SamPredictor, sam_model_registry

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SAMStrategyResult:
    """SAM戦略テスト結果"""
    strategy_name: str
    image_id: str
    success: bool
    iou_score: float
    processing_time: float
    mask_quality: float
    strategy_details: Dict[str, Any]


class ImprovedSAMStrategy:
    """改善版SAMプロンプト戦略"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.labels_file = project_root / "extracted_labels.json"
        self.output_dir = Path("/mnt/c/AItools/lora/train/yado/sam_improvement")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 人間ラベルデータ読み込み
        self.human_labels = self.load_human_labels()
        logger.info(f"人間ラベルデータ読み込み: {len(self.human_labels)}件")
        
        # YOLO初期化（最適閾値0.03使用）
        self.yolo_model = YOLO('yolov8n.pt')
        self.optimal_threshold = 0.03
        
        # SAM初期化
        self.init_sam()
        
    def init_sam(self):
        """SAM初期化"""
        sam_checkpoint = self.project_root / "sam_vit_h_4b8939.pth"
        if not sam_checkpoint.exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {sam_checkpoint}")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry["vit_h"](checkpoint=str(sam_checkpoint))
        sam.to(device=device)
        self.sam_predictor = SamPredictor(sam)
        
    def load_human_labels(self) -> Dict[str, Dict]:
        """人間ラベルデータ読み込み"""
        try:
            with open(self.labels_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            labels_dict = {}
            for item in data:
                filename = item['filename']
                image_id = filename.rsplit('.', 1)[0]
                
                if item.get('red_boxes') and len(item['red_boxes']) > 0:
                    first_box = item['red_boxes'][0]
                    bbox = first_box['bbox']
                    labels_dict[image_id] = {
                        'filename': filename,
                        'bbox': [bbox['x'], bbox['y'], bbox['width'], bbox['height']]
                    }
            
            return labels_dict
            
        except Exception as e:
            logger.error(f"ラベルデータ読み込みエラー: {e}")
            return {}
    
    def find_image_path(self, image_id: str) -> Optional[Path]:
        """画像ファイルパス検索"""
        search_dirs = [
            Path("/mnt/c/AItools/lora/train/yado/org/kana05_cursor"),
            Path("/mnt/c/AItools/lora/train/yado/org/kana07_cursor"),
            Path("/mnt/c/AItools/lora/train/yado/org/kana08_cursor"),
            self.project_root / "test_small"
        ]
        
        extensions = ['.jpg', '.jpeg', '.png']
        
        for dir_path in search_dirs:
            for ext in extensions:
                image_path = dir_path / f"{image_id}{ext}"
                if image_path.exists():
                    return image_path
        
        return None
    
    def calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                     bbox2: Tuple[int, int, int, int]) -> float:
        """IoU計算"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        union_area = w1 * h1 + w2 * h2 - intersection_area
        
        return intersection_area / max(union_area, 1e-6)
    
    def calculate_mask_quality(self, mask: np.ndarray) -> float:
        """マスク品質評価"""
        if mask is None or not mask.any():
            return 0.0
        
        # 連結性チェック
        contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return 0.0
        
        # 最大連結成分の割合
        largest_contour = max(contours, key=cv2.contourArea)
        largest_area = cv2.contourArea(largest_contour)
        total_area = np.sum(mask)
        
        connectivity_score = largest_area / max(total_area, 1)
        
        # 形状の滑らかさ（周囲長/面積比）
        perimeter = cv2.arcLength(largest_contour, True)
        smoothness_score = min(1.0, 4 * np.pi * largest_area / max(perimeter**2, 1))
        
        # 総合品質スコア
        quality_score = 0.7 * connectivity_score + 0.3 * smoothness_score
        return quality_score
    
    def strategy_single_center_point(self, image: np.ndarray, yolo_bbox: Tuple[int, int, int, int]) -> Tuple[Optional[np.ndarray], Dict]:
        """戦略1: 単一中心点プロンプト（従来手法）"""
        x, y, w, h = yolo_bbox
        center_x = x + w // 2
        center_y = y + h // 2
        
        input_point = np.array([[center_x, center_y]])
        input_label = np.array([1])
        
        masks, scores, _ = self.sam_predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )
        
        if masks is not None and len(masks) > 0:
            best_mask = masks[np.argmax(scores)]
            return best_mask, {
                'points_used': 1,
                'center_point': [center_x, center_y],
                'confidence': float(np.max(scores))
            }
        
        return None, {'error': 'No mask generated'}
    
    def strategy_multiple_points(self, image: np.ndarray, yolo_bbox: Tuple[int, int, int, int]) -> Tuple[Optional[np.ndarray], Dict]:
        """戦略2: 複数点プロンプト"""
        x, y, w, h = yolo_bbox
        
        # 5点パターン（中心 + 四隅寄り）
        points = [
            [x + w // 2, y + h // 2],      # 中心
            [x + w // 4, y + h // 4],      # 左上寄り
            [x + 3 * w // 4, y + h // 4],  # 右上寄り
            [x + w // 4, y + 3 * h // 4],  # 左下寄り
            [x + 3 * w // 4, y + 3 * h // 4]  # 右下寄り
        ]
        
        input_point = np.array(points)
        input_label = np.array([1, 1, 1, 1, 1])  # 全て正例
        
        masks, scores, _ = self.sam_predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )
        
        if masks is not None and len(masks) > 0:
            best_mask = masks[np.argmax(scores)]
            return best_mask, {
                'points_used': 5,
                'points': points,
                'confidence': float(np.max(scores))
            }
        
        return None, {'error': 'No mask generated'}
    
    def strategy_with_negatives(self, image: np.ndarray, yolo_bbox: Tuple[int, int, int, int]) -> Tuple[Optional[np.ndarray], Dict]:
        """戦略3: ネガティブプロンプト併用"""
        x, y, w, h = yolo_bbox
        img_h, img_w = image.shape[:2]
        
        # ポジティブポイント
        positive_points = [
            [x + w // 2, y + h // 2],      # 中心
            [x + w // 3, y + h // 3],      # 左上寄り
            [x + 2 * w // 3, y + 2 * h // 3]  # 右下寄り
        ]
        
        # ネガティブポイント（背景領域）
        negative_points = []
        margin = 50
        
        # 上下左右の背景領域にネガティブポイント配置
        if y > margin:  # 上側背景
            negative_points.append([x + w // 2, max(0, y - margin // 2)])
        if y + h + margin < img_h:  # 下側背景
            negative_points.append([x + w // 2, min(img_h - 1, y + h + margin // 2)])
        if x > margin:  # 左側背景
            negative_points.append([max(0, x - margin // 2), y + h // 2])
        if x + w + margin < img_w:  # 右側背景
            negative_points.append([min(img_w - 1, x + w + margin // 2), y + h // 2])
        
        if not negative_points:
            # フォールバック: 四隅から適当な背景点
            negative_points = [
                [margin, margin],
                [img_w - margin, margin]
            ]
        
        all_points = positive_points + negative_points
        labels = [1] * len(positive_points) + [0] * len(negative_points)
        
        input_point = np.array(all_points)
        input_label = np.array(labels)
        
        masks, scores, _ = self.sam_predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )
        
        if masks is not None and len(masks) > 0:
            best_mask = masks[np.argmax(scores)]
            return best_mask, {
                'positive_points': len(positive_points),
                'negative_points': len(negative_points),
                'all_points': all_points,
                'confidence': float(np.max(scores))
            }
        
        return None, {'error': 'No mask generated'}
    
    def strategy_bbox_prompt(self, image: np.ndarray, yolo_bbox: Tuple[int, int, int, int]) -> Tuple[Optional[np.ndarray], Dict]:
        """戦略4: 境界ボックスプロンプト"""
        x, y, w, h = yolo_bbox
        
        # YOLOボックスを少し拡張
        expansion = 0.1  # 10%拡張
        expand_w = int(w * expansion)
        expand_h = int(h * expansion)
        
        expanded_bbox = np.array([
            max(0, x - expand_w),
            max(0, y - expand_h),
            x + w + expand_w,
            y + h + expand_h
        ])
        
        masks, scores, _ = self.sam_predictor.predict(
            box=expanded_bbox[None, :],
            multimask_output=True
        )
        
        if masks is not None and len(masks) > 0:
            best_mask = masks[np.argmax(scores)]
            return best_mask, {
                'bbox_used': True,
                'original_bbox': [x, y, w, h],
                'expanded_bbox': expanded_bbox.tolist(),
                'confidence': float(np.max(scores))
            }
        
        return None, {'error': 'No mask generated'}
    
    def strategy_hybrid(self, image: np.ndarray, yolo_bbox: Tuple[int, int, int, int]) -> Tuple[Optional[np.ndarray], Dict]:
        """戦略5: ハイブリッド（境界ボックス + 複数点）"""
        x, y, w, h = yolo_bbox
        
        # 境界ボックス
        expanded_bbox = np.array([x - 5, y - 5, x + w + 5, y + h + 5])
        
        # 複数点
        points = [
            [x + w // 2, y + h // 2],      # 中心
            [x + w // 4, y + h // 4],      # 左上
            [x + 3 * w // 4, y + 3 * h // 4]  # 右下
        ]
        
        input_point = np.array(points)
        input_label = np.array([1, 1, 1])
        
        masks, scores, _ = self.sam_predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=expanded_bbox[None, :],
            multimask_output=True
        )
        
        if masks is not None and len(masks) > 0:
            best_mask = masks[np.argmax(scores)]
            return best_mask, {
                'hybrid': True,
                'points': points,
                'bbox': expanded_bbox.tolist(),
                'confidence': float(np.max(scores))
            }
        
        return None, {'error': 'No mask generated'}
    
    def test_all_strategies(self, image_id: str, label_data: Dict) -> List[SAMStrategyResult]:
        """全戦略のテスト実行"""
        image_path = self.find_image_path(image_id)
        if not image_path:
            return []
        
        # 画像読み込み
        image = cv2.imread(str(image_path))
        if image is None:
            return []
        
        # YOLO検出（最適閾値使用）
        results = self.yolo_model(image, conf=self.optimal_threshold, verbose=False)
        
        if not results or len(results[0].boxes) == 0:
            return []
        
        # 最大検出結果選択
        boxes = results[0].boxes.xyxy.cpu().numpy()
        if len(boxes) == 0:
            return []
        
        areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
        largest_idx = np.argmax(areas)
        x1, y1, x2, y2 = boxes[largest_idx]
        yolo_bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
        
        # SAM画像設定
        self.sam_predictor.set_image(image)
        
        # 各戦略をテスト
        strategies = [
            ('Single Center Point', self.strategy_single_center_point),
            ('Multiple Points', self.strategy_multiple_points),
            ('With Negatives', self.strategy_with_negatives),
            ('BBox Prompt', self.strategy_bbox_prompt),
            ('Hybrid', self.strategy_hybrid)
        ]
        
        results_list = []
        human_bbox = tuple(label_data['bbox'])
        
        for strategy_name, strategy_func in strategies:
            start_time = time.time()
            
            try:
                mask, details = strategy_func(image, yolo_bbox)
                processing_time = time.time() - start_time
                
                if mask is not None:
                    # マスクから境界ボックス計算
                    y_indices, x_indices = np.where(mask > 0)
                    
                    if len(x_indices) > 0 and len(y_indices) > 0:
                        x_min = int(np.min(x_indices))
                        x_max = int(np.max(x_indices))
                        y_min = int(np.min(y_indices))
                        y_max = int(np.max(y_indices))
                        
                        mask_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                        
                        # 評価
                        iou_score = self.calculate_iou(human_bbox, mask_bbox)
                        mask_quality = self.calculate_mask_quality(mask)
                        success = iou_score > 0.5
                        
                        results_list.append(SAMStrategyResult(
                            strategy_name=strategy_name,
                            image_id=image_id,
                            success=success,
                            iou_score=iou_score,
                            processing_time=processing_time,
                            mask_quality=mask_quality,
                            strategy_details=details
                        ))
                    else:
                        results_list.append(SAMStrategyResult(
                            strategy_name=strategy_name,
                            image_id=image_id,
                            success=False,
                            iou_score=0.0,
                            processing_time=processing_time,
                            mask_quality=0.0,
                            strategy_details={'error': 'Empty mask'}
                        ))
                else:
                    results_list.append(SAMStrategyResult(
                        strategy_name=strategy_name,
                        image_id=image_id,
                        success=False,
                        iou_score=0.0,
                        processing_time=processing_time,
                        mask_quality=0.0,
                        strategy_details=details
                    ))
                    
            except Exception as e:
                results_list.append(SAMStrategyResult(
                    strategy_name=strategy_name,
                    image_id=image_id,
                    success=False,
                    iou_score=0.0,
                    processing_time=time.time() - start_time,
                    mask_quality=0.0,
                    strategy_details={'error': str(e)}
                ))
        
        return results_list
    
    def run_strategy_comparison(self, sample_limit: int = 20) -> List[SAMStrategyResult]:
        """全戦略比較実行"""
        logger.info("SAM戦略比較開始")
        all_results = []
        
        # サンプル制限
        test_items = list(self.human_labels.items())[:sample_limit]
        
        for i, (image_id, label_data) in enumerate(test_items, 1):
            logger.info(f"処理中 [{i}/{len(test_items)}]: {image_id}")
            
            results = self.test_all_strategies(image_id, label_data)
            all_results.extend(results)
        
        return all_results
    
    def generate_strategy_report(self, results: List[SAMStrategyResult]):
        """戦略比較レポート生成"""
        # 戦略別統計
        strategy_stats = {}
        for result in results:
            strategy = result.strategy_name
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    'total': 0,
                    'success': 0,
                    'iou_scores': [],
                    'processing_times': [],
                    'quality_scores': []
                }
            
            stats = strategy_stats[strategy]
            stats['total'] += 1
            if result.success:
                stats['success'] += 1
            stats['iou_scores'].append(result.iou_score)
            stats['processing_times'].append(result.processing_time)
            stats['quality_scores'].append(result.mask_quality)
        
        # 最良戦略決定
        best_strategy = None
        best_score = 0
        
        for strategy, stats in strategy_stats.items():
            success_rate = stats['success'] / max(stats['total'], 1) * 100
            avg_iou = np.mean(stats['iou_scores'])
            combined_score = success_rate * 0.7 + avg_iou * 30  # 重み付け総合スコア
            
            if combined_score > best_score:
                best_score = combined_score
                best_strategy = strategy
        
        # レポート作成
        report = f"""# SAMプロンプト戦略比較レポート

**実行日時**: {time.strftime('%Y-%m-%d %H:%M:%S')}
**テスト画像数**: {len(set(r.image_id for r in results))}枚
**YOLO閾値**: {self.optimal_threshold} (最適化済み)

---

## 🏆 最優秀戦略

**推奨戦略**: {best_strategy}
**総合スコア**: {best_score:.1f}

---

## 📊 戦略別比較結果

| 戦略 | 成功率 | 平均IoU | 平均品質 | 平均処理時間 |
|------|--------|---------|----------|-------------|
"""
        
        for strategy, stats in strategy_stats.items():
            success_rate = stats['success'] / max(stats['total'], 1) * 100
            avg_iou = np.mean(stats['iou_scores'])
            avg_quality = np.mean(stats['quality_scores'])
            avg_time = np.mean(stats['processing_times'])
            
            marker = "⭐" if strategy == best_strategy else ""
            report += f"| {strategy} {marker} | {success_rate:.1f}% | {avg_iou:.3f} | {avg_quality:.3f} | {avg_time:.3f}s |\n"
        
        report += f"""
---

## 🎯 改善効果予測

### 従来手法 vs 最優秀戦略
- **従来**: Single Center Point
- **改善**: {best_strategy}
- **予想成功率向上**: {strategy_stats[best_strategy]['success'] / max(strategy_stats[best_strategy]['total'], 1) * 100:.1f}%

### 次のステップ
1. **{best_strategy}の本格導入**
2. **YOLO最適閾値(0.03)との統合**  
3. **エンドツーエンド性能測定**

*Generated by Improved SAM Strategy System*
"""
        
        # レポート保存
        report_path = self.output_dir / f"sam_strategy_report_{time.strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"SAM戦略レポート保存: {report_path}")
        return report_path, best_strategy, strategy_stats


def main():
    """メイン処理"""
    project_root = Path("/mnt/c/AItools/segment-anything")
    
    sam_strategy = ImprovedSAMStrategy(project_root)
    
    # 戦略比較実行（サンプル20枚）
    results = sam_strategy.run_strategy_comparison(sample_limit=20)
    
    # レポート生成
    report_path, best_strategy, stats = sam_strategy.generate_strategy_report(results)
    
    # 結果サマリー
    print(f"\n✅ SAM戦略比較完了")
    print(f"最優秀戦略: {best_strategy}")
    print(f"レポート: {report_path}")


if __name__ == "__main__":
    main()