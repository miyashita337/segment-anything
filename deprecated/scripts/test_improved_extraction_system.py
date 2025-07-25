#!/usr/bin/env python3
"""
改善版抽出システムのテスト
kana08評価結果を元にした白色系部位・マスク検出・全身抽出の改善テスト
"""

import numpy as np
import cv2

import logging
import os
import sys
import time
from pathlib import Path

# プロジェクトパス追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from features.evaluation.utils.non_character_filter import NonCharacterFilter
from features.processing.limb_protection_system import LimbProtectionSystem
# 改善されたモジュールのインポート
from features.processing.preprocessing.boundary_enhancer import BoundaryEnhancer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImprovedExtractionSystem:
    """改善版抽出システム"""
    
    def __init__(self):
        """初期化"""
        self.boundary_enhancer = BoundaryEnhancer(
            skin_enhancement_factor=1.3,
            edge_enhancement_factor=1.8,
            contrast_enhancement=1.2
        )
        
        self.non_character_filter = NonCharacterFilter()
        
        self.limb_protector = LimbProtectionSystem(
            enable_pose_estimation=True,
            enable_limb_completion=True,
            protection_margin=15
        )
        
        logger.info("改善版抽出システム初期化完了")
    
    def process_test_image(self, image_path: Path, output_dir: Path) -> dict:
        """テスト画像の処理"""
        logger.info(f"処理開始: {image_path.name}")
        
        # 画像読み込み
        image = cv2.imread(str(image_path))
        if image is None:
            return {"error": "画像読み込み失敗"}
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = {
            "image_name": image_path.name,
            "original_size": image_rgb.shape,
            "processing_steps": []
        }
        
        # Step 1: 境界強調前処理
        start_time = time.time()
        enhanced_image = self.boundary_enhancer.enhance_image_boundaries(image_rgb)
        enhancement_time = time.time() - start_time
        
        # 強調統計取得
        enhancement_stats = self.boundary_enhancer.get_enhancement_stats(image_rgb, enhanced_image)
        
        results["processing_steps"].append({
            "step": "boundary_enhancement",
            "time": enhancement_time,
            "stats": enhancement_stats
        })
        
        # Step 2: 簡易マスク生成（実際のSAM/YOLO処理の代替）
        start_time = time.time()
        test_mask = self._generate_test_mask(enhanced_image)
        mask_time = time.time() - start_time
        
        results["processing_steps"].append({
            "step": "mask_generation", 
            "time": mask_time,
            "mask_area": np.sum(test_mask > 0)
        })
        
        # Step 3: 非キャラクター要素フィルタリングのテスト
        start_time = time.time()
        
        # テスト用マスクデータ作成
        test_masks = self._create_test_mask_candidates(test_mask, enhanced_image)
        filtered_masks = self.non_character_filter.filter_non_character_elements(
            test_masks, cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR)
        )
        
        filter_time = time.time() - start_time
        
        results["processing_steps"].append({
            "step": "non_character_filtering",
            "time": filter_time,
            "original_candidates": len(test_masks),
            "filtered_candidates": len(filtered_masks)
        })
        
        # Step 4: 手足保護システム
        start_time = time.time()
        protected_mask, limb_analysis = self.limb_protector.protect_limbs_in_mask(
            enhanced_image, test_mask
        )
        protection_time = time.time() - start_time
        
        results["processing_steps"].append({
            "step": "limb_protection",
            "time": protection_time,
            "analysis": limb_analysis
        })
        
        # 結果保存
        self._save_test_results(
            output_dir, image_path.stem,
            image_rgb, enhanced_image, test_mask, protected_mask
        )
        
        logger.info(f"処理完了: {image_path.name} (総時間: {sum(step['time'] for step in results['processing_steps']):.2f}秒)")
        
        return results
    
    def _generate_test_mask(self, image: np.ndarray) -> np.ndarray:
        """テスト用マスク生成（SAM/YOLOの代替）"""
        # グレースケール変換
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 適応的閾値処理
        mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # ノイズ除去
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 最大連結成分を保持
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if num_labels > 1:
            largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            mask = (labels == largest_label).astype(np.uint8) * 255
        
        return mask
    
    def _create_test_mask_candidates(self, mask: np.ndarray, image: np.ndarray) -> list:
        """テスト用マスク候補作成"""
        # 連結成分解析でマスク候補を作成
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        
        candidates = []
        for i in range(1, num_labels):
            # 各成分の情報
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            
            # マスクを抽出
            component_mask = (labels == i).astype(np.uint8) * 255
            
            candidates.append({
                "segmentation": component_mask,
                "bbox": (x, y, w, h),
                "area": area,
                "confidence": 0.8  # ダミー値
            })
        
        return candidates
    
    def _save_test_results(self, output_dir: Path, base_name: str, 
                          original: np.ndarray, enhanced: np.ndarray,
                          test_mask: np.ndarray, protected_mask: np.ndarray):
        """テスト結果の保存"""
        output_dir.mkdir(exist_ok=True)
        
        # オリジナル画像
        cv2.imwrite(
            str(output_dir / f"{base_name}_01_original.jpg"),
            cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        )
        
        # 境界強調後
        cv2.imwrite(
            str(output_dir / f"{base_name}_02_enhanced.jpg"),
            cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
        )
        
        # テストマスク
        cv2.imwrite(
            str(output_dir / f"{base_name}_03_test_mask.jpg"),
            test_mask
        )
        
        # 保護後マスク
        cv2.imwrite(
            str(output_dir / f"{base_name}_04_protected_mask.jpg"),
            protected_mask
        )
        
        # 比較画像作成
        comparison = self._create_comparison_image(
            original, enhanced, test_mask, protected_mask
        )
        cv2.imwrite(
            str(output_dir / f"{base_name}_05_comparison.jpg"),
            comparison
        )
    
    def _create_comparison_image(self, original: np.ndarray, enhanced: np.ndarray,
                               test_mask: np.ndarray, protected_mask: np.ndarray) -> np.ndarray:
        """比較画像作成"""
        h, w = original.shape[:2]
        
        # 2x2レイアウトで配置
        comparison = np.zeros((h*2, w*2, 3), dtype=np.uint8)
        
        # オリジナル（左上）
        comparison[:h, :w] = original
        
        # 強調後（右上）
        comparison[:h, w:] = enhanced
        
        # テストマスク（左下）
        test_mask_3ch = cv2.cvtColor(test_mask, cv2.COLOR_GRAY2RGB)
        comparison[h:, :w] = test_mask_3ch
        
        # 保護後マスク（右下）
        protected_mask_3ch = cv2.cvtColor(protected_mask, cv2.COLOR_GRAY2RGB)
        comparison[h:, w:] = protected_mask_3ch
        
        # ラベル追加
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "Original", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Enhanced", (w+10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Test Mask", (10, h+30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Protected", (w+10, h+30), font, 1, (255, 255, 255), 2)
        
        return cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)


def run_improvement_test():
    """改善システムのテスト実行"""
    # テスト画像選択（kana08評価で問題があった画像）
    test_images = [
        "kana08_0001.jpg",  # 上半身抽出失敗
        "kana08_0008.jpg",  # 白い足が抽出できない
        "kana08_0013.jpg",  # カチューシャ・胸部分欠損
        "kana08_0019.jpg",  # 太もものみ抽出
        "kana08_0000_cover.jpg",  # マスク誤抽出
        "kana08_0005.jpg",  # マスク誤抽出
    ]
    
    input_dir = Path("/mnt/c/AItools/lora/train/yado/org/kana08")
    output_dir = Path("/mnt/c/AItools/segment-anything/test_results_improved")
    
    system = ImprovedExtractionSystem()
    results = []
    
    logger.info(f"改善システムテスト開始: {len(test_images)}枚")
    
    for image_name in test_images:
        image_path = input_dir / image_name
        
        if image_path.exists():
            try:
                result = system.process_test_image(image_path, output_dir)
                results.append(result)
            except Exception as e:
                logger.error(f"エラー {image_name}: {e}")
                results.append({
                    "image_name": image_name,
                    "error": str(e)
                })
        else:
            logger.warning(f"画像が見つかりません: {image_path}")
    
    # 結果サマリー
    logger.info("=" * 50)
    logger.info("改善システムテスト結果サマリー")
    
    for result in results:
        if "error" not in result:
            total_time = sum(step["time"] for step in result["processing_steps"])
            logger.info(f"{result['image_name']}: 処理時間 {total_time:.2f}秒")
            
            for step in result["processing_steps"]:
                if step["step"] == "boundary_enhancement":
                    stats = step["stats"]
                    logger.info(f"  境界強調: コントラスト改善 {stats['contrast_improvement']:.2f}x")
                elif step["step"] == "non_character_filtering":
                    logger.info(f"  フィルタリング: {step['original_candidates']} → {step['filtered_candidates']} 候補")
                elif step["step"] == "limb_protection":
                    analysis = step["analysis"]
                    logger.info(f"  手足保護: 適用={analysis['protection_applied']}, 品質={analysis['protection_quality']:.2f}")
        else:
            logger.error(f"{result['image_name']}: {result['error']}")
    
    logger.info(f"テスト結果保存: {output_dir}")
    return results


if __name__ == "__main__":
    run_improvement_test()