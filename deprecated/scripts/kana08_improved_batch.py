#!/usr/bin/env python3
"""
kana08改善版バッチ抽出スクリプト
評価結果に基づく改善システム統合版
"""

import numpy as np
import cv2
import torch

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple
from ultralytics import YOLO

# プロジェクトパス追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from features.evaluation.utils.non_character_filter import NonCharacterFilter
from features.processing.limb_protection_system import LimbProtectionSystem
# 改善システムのインポート
from features.processing.preprocessing.boundary_enhancer import BoundaryEnhancer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImprovedKana08Extractor:
    """kana08改善版バッチ抽出器"""
    
    def __init__(self):
        self.input_dir = Path("/mnt/c/AItools/lora/train/yado/org/kana08")
        self.output_dir = Path("/mnt/c/AItools/lora/train/yado/clipped_boundingbox/kana08_rev_merge2")
        
        # 改善システム初期化
        logger.info("改善システム初期化中...")
        
        self.boundary_enhancer = BoundaryEnhancer(
            skin_enhancement_factor=1.4,    # 肌色強調を強化
            edge_enhancement_factor=2.2,    # エッジ強調を強化
            contrast_enhancement=1.3        # コントラスト強化を維持
        )
        
        self.non_character_filter = NonCharacterFilter()
        
        self.limb_protector = LimbProtectionSystem(
            enable_pose_estimation=True,
            enable_limb_completion=True,
            protection_margin=12  # 少し控えめに調整
        )
        
        # SAM・YOLOモデル初期化
        logger.info("SAM・YOLOモデル初期化中...")
        
        sam_checkpoint = Path("/mnt/c/AItools/segment-anything/sam_vit_h_4b8939.pth")
        if not sam_checkpoint.exists():
            logger.error(f"SAMモデルが見つかりません: {sam_checkpoint}")
            raise FileNotFoundError(f"SAMモデルファイルが必要です: {sam_checkpoint}")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry["vit_h"](checkpoint=str(sam_checkpoint))
        sam.to(device=device)
        self.sam_generator = SamAutomaticMaskGenerator(sam)
        
        self.yolo_model = YOLO('yolov8n.pt')
        
        # 設定
        self.confidence_threshold = 0.07
        
        logger.info("改善版抽出器初期化完了")
    
    def process_image(self, image_path: Path) -> Tuple[bool, Optional[str], Optional[dict]]:
        """改善版単一画像の処理"""
        try:
            start_time = time.time()
            
            # 画像読み込み
            image = cv2.imread(str(image_path))
            if image is None:
                return False, "画像読み込み失敗", None
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            processing_stats = {"original_size": image_rgb.shape}
            
            # Step 1: 境界強調前処理
            enhanced_image = self.boundary_enhancer.enhance_image_boundaries(image_rgb)
            enhancement_stats = self.boundary_enhancer.get_enhancement_stats(image_rgb, enhanced_image)
            processing_stats["enhancement"] = enhancement_stats
            
            logger.debug(f"境界強調: コントラスト改善 {enhancement_stats['contrast_improvement']:.2f}x")
            
            # Step 2: YOLO検出（改善版前処理画像使用）
            results = self.yolo_model(cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR), conf=self.confidence_threshold)
            
            if not results or len(results[0].boxes) == 0:
                return False, "キャラクター未検出", processing_stats
            
            # Step 3: SAMでマスク生成
            sam_masks = self.sam_generator.generate(enhanced_image)
            if not sam_masks:
                return False, "SAMマスク生成失敗", processing_stats
            
            # Step 4: YOLO検出結果と統合
            boxes = results[0].boxes.xyxy.cpu().numpy()
            integrated_masks = self._integrate_yolo_sam_masks(sam_masks, boxes, enhanced_image)
            
            if not integrated_masks:
                return False, "統合マスク生成失敗", processing_stats
            
            # Step 5: 非キャラクター要素フィルタリング
            filtered_masks = self.non_character_filter.filter_non_character_elements(
                integrated_masks, cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR)
            )
            
            processing_stats["filtering"] = {
                "original_candidates": len(integrated_masks),
                "filtered_candidates": len(filtered_masks)
            }
            
            if not filtered_masks:
                return False, "フィルタリング後マスクなし", processing_stats
            
            # Step 6: 最適マスク選択（サイズ・位置・品質の複合評価）
            best_mask_data = self._select_best_mask_improved(filtered_masks, enhanced_image.shape)
            
            if not best_mask_data:
                return False, "最適マスク選択失敗", processing_stats
            
            best_mask = best_mask_data['segmentation']
            
            # Step 7: 手足保護システム適用
            protected_mask, limb_analysis = self.limb_protector.protect_limbs_in_mask(
                enhanced_image, best_mask
            )
            
            processing_stats["limb_protection"] = limb_analysis
            
            # Step 8: 最終抽出処理
            extracted_image = self._extract_character_with_mask(enhanced_image, protected_mask)
            
            if extracted_image is None:
                return False, "キャラクター抽出失敗", processing_stats
            
            # Step 9: 結果保存
            output_path = self.output_dir / image_path.name
            cv2.imwrite(str(output_path), cv2.cvtColor(extracted_image, cv2.COLOR_RGB2BGR))
            
            processing_time = time.time() - start_time
            processing_stats["total_time"] = processing_time
            
            quality_info = f"改善版抽出"
            if limb_analysis["protection_applied"]:
                quality_info += f" (手足保護: {limb_analysis['protection_quality']:.2f})"
            
            return True, quality_info, processing_stats
            
        except Exception as e:
            return False, f"エラー: {str(e)}", None
    
    def _integrate_yolo_sam_masks(self, sam_masks: List[dict], yolo_boxes: np.ndarray, 
                                image: np.ndarray) -> List[dict]:
        """YOLO検出結果とSAMマスクの統合"""
        integrated_masks = []
        
        for mask_data in sam_masks:
            mask = mask_data['segmentation']
            
            # マスクの境界ボックス取得
            y_indices, x_indices = np.where(mask)
            if len(x_indices) == 0:
                continue
            
            mask_x1, mask_y1 = np.min(x_indices), np.min(y_indices)
            mask_x2, mask_y2 = np.max(x_indices), np.max(y_indices)
            
            # YOLOボックスとの重複確認
            best_overlap = 0.0
            best_yolo_score = 0.0
            
            for box in yolo_boxes:
                yolo_x1, yolo_y1, yolo_x2, yolo_y2 = box
                
                # 重複領域計算
                overlap_x1 = max(mask_x1, yolo_x1)
                overlap_y1 = max(mask_y1, yolo_y1)
                overlap_x2 = min(mask_x2, yolo_x2)
                overlap_y2 = min(mask_y2, yolo_y2)
                
                if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                    mask_area = (mask_x2 - mask_x1) * (mask_y2 - mask_y1)
                    overlap_ratio = overlap_area / max(mask_area, 1)
                    
                    if overlap_ratio > best_overlap:
                        best_overlap = overlap_ratio
                        best_yolo_score = 0.8  # YOLO信頼度（簡易版）
            
            # 十分な重複があるマスクのみ採用
            if best_overlap > 0.3:
                mask_data['yolo_overlap'] = best_overlap
                mask_data['yolo_confidence'] = best_yolo_score
                mask_data['bbox'] = (mask_x1, mask_y1, mask_x2 - mask_x1, mask_y2 - mask_y1)
                integrated_masks.append(mask_data)
        
        return integrated_masks
    
    def _select_best_mask_improved(self, masks: List[dict], image_shape: Tuple[int, int, int]) -> Optional[dict]:
        """改善版最適マスク選択"""
        if not masks:
            return None
        
        h, w = image_shape[:2]
        best_mask = None
        best_score = 0.0
        
        for mask_data in masks:
            score = 0.0
            
            # サイズスコア
            area = mask_data.get('area', 0)
            area_ratio = area / (h * w)
            if 0.02 <= area_ratio <= 0.4:  # 適切なサイズ範囲
                score += min(area_ratio / 0.2, 1.0) * 0.3
            
            # 位置スコア（中央寄り）
            bbox = mask_data.get('bbox', [0, 0, 0, 0])
            if len(bbox) >= 4:
                center_x = bbox[0] + bbox[2] / 2
                center_y = bbox[1] + bbox[3] / 2
                
                # 画像中央からの距離
                distance_from_center = np.sqrt(
                    ((center_x - w/2) / w)**2 + 
                    ((center_y - h/2) / h)**2
                )
                score += max(0, 1.0 - distance_from_center) * 0.2
            
            # YOLO重複スコア
            yolo_overlap = mask_data.get('yolo_overlap', 0)
            score += yolo_overlap * 0.3
            
            # SAM品質スコア
            stability_score = mask_data.get('stability_score', 0.5)
            score += stability_score * 0.2
            
            if score > best_score:
                best_score = score
                best_mask = mask_data
        
        return best_mask
    
    def _extract_character_with_mask(self, image: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
        """マスクを使用したキャラクター抽出"""
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        
        # マスクを3チャネルに変換
        mask_normalized = mask.astype(np.float32) / 255.0
        mask_3ch = np.stack([mask_normalized] * 3, axis=-1)
        
        # 背景を黒にしてキャラクターを抽出
        extracted = image.astype(np.float32) * mask_3ch
        
        # 境界の取得とクロップ
        y_indices, x_indices = np.where(mask > 0)
        if len(x_indices) == 0:
            return None
        
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # パディング追加
        padding = 10
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(image.shape[1], x_max + padding)
        y_max = min(image.shape[0], y_max + padding)
        
        # クロップ
        cropped = extracted[y_min:y_max, x_min:x_max]
        
        return cropped.astype(np.uint8)
    
    def run_batch(self):
        """改善版バッチ処理実行"""
        # 出力ディレクトリ作成
        self.output_dir.mkdir(exist_ok=True)
        
        # 画像ファイル取得
        image_files = sorted(list(self.input_dir.glob("*.jpg")))
        total = len(image_files)
        
        if total == 0:
            logger.error("処理する画像が見つかりません")
            return
        
        logger.info(f"改善版バッチ処理開始: {total}枚の画像")
        logger.info(f"入力: {self.input_dir}")
        logger.info(f"出力: {self.output_dir}")
        
        # 処理統計
        success_count = 0
        failed_files = []
        improvement_stats = []
        start_time = time.time()
        
        # 各画像を処理
        for i, image_path in enumerate(image_files, 1):
            logger.info(f"[{i}/{total}] 処理中: {image_path.name}")
            
            success, message, stats = self.process_image(image_path)
            
            if success:
                success_count += 1
                logger.info(f"  ✅ 成功 - {message}")
                if stats:
                    improvement_stats.append(stats)
            else:
                failed_files.append((image_path.name, message))
                logger.warning(f"  ❌ 失敗 - {message}")
            
            # 進捗表示
            if i % 5 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                remaining = avg_time * (total - i)
                logger.info(f"進捗: {i}/{total} ({i/total*100:.1f}%) - 残り時間: {remaining:.0f}秒")
        
        # 処理完了
        total_time = time.time() - start_time
        logger.info("=" * 50)
        logger.info("改善版バッチ処理完了")
        logger.info(f"総処理時間: {total_time:.1f}秒")
        logger.info(f"平均処理時間: {total_time/total:.1f}秒/画像")
        logger.info(f"成功: {success_count}/{total} ({success_count/total*100:.1f}%)")
        
        # 改善統計の表示
        if improvement_stats:
            avg_enhancement = np.mean([s["enhancement"]["contrast_improvement"] 
                                     for s in improvement_stats if "enhancement" in s])
            
            filtering_effective = sum(1 for s in improvement_stats 
                                    if "filtering" in s and 
                                    s["filtering"]["filtered_candidates"] < s["filtering"]["original_candidates"])
            
            limb_protection_applied = sum(1 for s in improvement_stats 
                                        if "limb_protection" in s and 
                                        s["limb_protection"]["protection_applied"])
            
            logger.info("🔧 改善システム統計:")
            logger.info(f"  平均コントラスト改善: {avg_enhancement:.2f}x")
            logger.info(f"  フィルタリング有効: {filtering_effective}/{len(improvement_stats)}件")
            logger.info(f"  手足保護適用: {limb_protection_applied}/{len(improvement_stats)}件")
        
        if failed_files:
            logger.info("失敗ファイル:")
            for name, reason in failed_files:
                logger.info(f"  - {name}: {reason}")


def main():
    """メイン実行"""
    extractor = ImprovedKana08Extractor()
    extractor.run_batch()


if __name__ == "__main__":
    main()