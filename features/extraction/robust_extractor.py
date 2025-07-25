#!/usr/bin/env python3
"""
Robust Character Extractor - ロバストキャラクター抽出システム
複数手法の並列実行による最適結果選択と品質保証
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import cv2

import json
import logging
import subprocess
import tempfile
from features.common.hooks.start import get_sam_model, get_yolo_model, initialize_models
from features.evaluation.utils.face_detection import filter_non_character_masks
from features.evaluation.utils.non_character_filter import apply_non_character_filter
from features.processing.preprocessing.color_preserving_enhancer import ColorPreservingEnhancer
from features.processing.preprocessing.preprocessing import preprocess_image_pipeline
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RobustCharacterExtractor:
    """ロバスト品質保証キャラクター抽出システム"""
    
    def __init__(self):
        """初期化"""
        self.color_enhancer = ColorPreservingEnhancer(
            preserve_luminance=True,
            preserve_saturation=True,
            adaptive_enhancement=True
        )
        
        # 過去の成功結果を記録（簡易版）
        self.success_history = {}
        
        logger.info("RobustCharacterExtractor初期化完了")

    def extract_character_robust(self, 
                                input_path: Path, 
                                output_path: Path,
                                verbose: bool = False) -> Dict[str, Any]:
        """
        ロバスト品質保証キャラクター抽出
        
        Args:
            input_path: 入力画像パス
            output_path: 出力パス
            verbose: 詳細ログ出力
            
        Returns:
            抽出結果情報
        """
        logger.info(f"ロバスト抽出開始: {input_path.name}")
        
        # 基本前処理
        processed_bgr, processed_rgb, scale = preprocess_image_pipeline(str(input_path))
        if processed_bgr is None:
            return {"success": False, "error": "preprocessing_failed"}
        
        # 3つの手法で並列実行
        methods = [
            ("enhanced_system", self._extract_enhanced_system),
            ("color_preserving", self._extract_color_preserving),
            ("backup_method", self._extract_backup_method)
        ]
        
        results = {}
        best_result = None
        best_quality = 0.0
        
        for method_name, method_func in methods:
            try:
                if verbose:
                    print(f"🔄 実行中: {method_name}")
                
                result = method_func(processed_bgr, processed_rgb, input_path, verbose)
                
                if result and result.get("success", False):
                    quality_score = self._evaluate_result_quality(result, processed_rgb)
                    result["quality_score"] = quality_score
                    result["method"] = method_name
                    
                    results[method_name] = result
                    
                    if verbose:
                        print(f"✅ {method_name}: 品質スコア {quality_score:.3f}")
                    
                    # 最高品質結果を記録
                    if quality_score > best_quality:
                        best_quality = quality_score
                        best_result = result
                else:
                    if verbose:
                        print(f"❌ {method_name}: 失敗")
                        
            except Exception as e:
                logger.warning(f"{method_name} 実行エラー: {e}")
                if verbose:
                    print(f"⚠️ {method_name}: エラー - {e}")
        
        # 最適結果を保存
        if best_result:
            try:
                # 最高品質の結果を出力パスに保存
                cv2.imwrite(str(output_path), best_result["image_bgr"])
                
                final_result = {
                    "success": True,
                    "method": best_result["method"],
                    "quality_score": best_quality,
                    "size": best_result.get("size", "unknown"),
                    "face_detected": best_result.get("face_detected", False),
                    "alternative_methods": len(results),
                    "total_attempts": len(methods)
                }
                
                if verbose:
                    print(f"🎯 最適手法: {best_result['method']} (品質: {best_quality:.3f})")
                
                # 成功履歴に記録
                self.success_history[input_path.name] = {
                    "method": best_result["method"],
                    "quality": best_quality
                }
                
                return final_result
                
            except Exception as e:
                logger.error(f"結果保存エラー: {e}")
                return {"success": False, "error": f"save_failed: {e}"}
        
        else:
            # 全手法が失敗した場合 → 強制抽出を試行
            logger.warning(f"全手法失敗: {input_path.name} → 強制抽出モード実行")
            
            # 強制抽出：品質を問わず何らかの候補を抽出
            forced_result = self._force_extract_any_candidate(
                processed_bgr, processed_rgb, input_path, results, verbose
            )
            
            if forced_result and forced_result.get("success", False):
                logger.info(f"🚀 強制抽出成功: {input_path.name}")
                
                # 強制抽出結果を指定された出力パスに保存
                forced_output_path = input_path.parent / f"forced_{input_path.name}"
                if forced_output_path.exists():
                    try:
                        # 強制抽出ファイルを指定出力パスにコピー
                        import shutil
                        shutil.copy2(forced_output_path, output_path)
                        logger.info(f"📋 強制抽出結果を出力パスにコピー: {output_path}")
                        
                        # 一時ファイル削除
                        forced_output_path.unlink()
                    except Exception as e:
                        logger.warning(f"強制抽出ファイル移動エラー: {e}")
                
                return forced_result
            else:
                logger.error(f"💥 強制抽出も失敗: {input_path.name}")
                return {
                    "success": False,
                    "error": "all_methods_and_forced_extraction_failed",
                    "attempted_methods": len(methods),
                    "results": {k: v.get("error", "unknown_error") for k, v in results.items()}
                }

    def _force_extract_any_candidate(self, 
                                    processed_bgr: np.ndarray,
                                    processed_rgb: np.ndarray, 
                                    input_path: Path,
                                    previous_results: Dict[str, Any],
                                    verbose: bool = False) -> Optional[Dict[str, Any]]:
        """
        強制抽出：品質を問わず何らかの候補を抽出
        """
        try:
            logger.info(f"🚀 強制抽出モード開始: {input_path.name}")
            
            # モデル初期化確認
            if get_sam_model() is None or get_yolo_model() is None:
                initialize_models()
            
            sam_model = get_sam_model()
            yolo_model = get_yolo_model()
            
            # SAMマスク生成（全候補）
            all_masks = sam_model.generate_masks(processed_bgr)
            if not all_masks:
                logger.warning(f"SAMマスク生成失敗")
                return None
                
            logger.info(f"SAM候補数: {len(all_masks)}")
            
            # 最低限のフィルタリング（吹き出し・テキストのみ除外）
            from features.evaluation.utils.non_character_filter import apply_non_character_filter
            filtered_masks = apply_non_character_filter(all_masks, processed_rgb)
            
            # 強制抽出モード：フィルタで全て除外されても最大候補を選択
            if not filtered_masks:
                logger.warning(f"フィルタ後候補なし → 強制的に最大候補選択")
                if all_masks:
                    filtered_masks = [max(all_masks, key=lambda x: np.sum(x.get('segmentation', np.zeros((1, 1))) > 0))]
                else:
                    return None
            
            logger.info(f"最低限フィルタ後: {len(filtered_masks)}候補")
            
            # 最大のマスクを選択（面積ベース強制選択）
            best_candidate = max(filtered_masks, key=lambda x: np.sum(x.get('segmentation', np.zeros((1, 1))) > 0))
            
            if best_candidate is None:
                return None
            
            # 抽出実行
            mask = best_candidate.get('segmentation')
            if mask is None:
                return None
            
            # マスク適用して抽出
            extraction_result = self._apply_mask_and_extract(
                processed_rgb, mask, input_path.name
            )
            
            if extraction_result and extraction_result.get("extracted_rgb") is not None:
                # 保存（出力パスは呼び出し元で指定された場所）
                extracted_rgb = extraction_result["extracted_rgb"]
                extracted_bgr = cv2.cvtColor(extracted_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(input_path.parent / f"forced_{input_path.name}"), extracted_bgr)
                
                logger.info(f"✅ 強制抽出完了: {input_path.parent / f'forced_{input_path.name}'}")
                
                return {
                    "success": True,
                    "method": "forced_extraction",
                    "quality_score": 0.1,  # 低品質だが成功
                    "size": f"{extracted_rgb.shape[1]}x{extracted_rgb.shape[0]}",
                    "face_detected": False,
                    "forced": True,
                    "original_candidates": len(all_masks),
                    "filtered_candidates": len(filtered_masks)
                }
            
            return None
            
        except Exception as e:
            logger.error(f"強制抽出エラー: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _apply_mask_and_extract(self, 
                               image_rgb: np.ndarray, 
                               mask: np.ndarray, 
                               filename: str) -> Optional[Dict[str, Any]]:
        """
        マスクを適用してキャラクターを抽出
        """
        try:
            # マスクの正規化
            if mask.max() > 1:
                mask = mask.astype(np.float32) / 255.0
            
            # 3チャンネルマスクに変換
            if len(mask.shape) == 2:
                mask_3d = np.stack([mask, mask, mask], axis=2)
            else:
                mask_3d = mask
            
            # マスク適用
            masked_image = image_rgb * mask_3d
            
            # バウンディングボックス計算
            y_indices, x_indices = np.where(mask > 0.1)
            if len(y_indices) == 0 or len(x_indices) == 0:
                return None
            
            y_min, y_max = y_indices.min(), y_indices.max()
            x_min, x_max = x_indices.min(), x_indices.max()
            
            # 余白追加
            padding = 10
            h, w = image_rgb.shape[:2]
            y_min = max(0, y_min - padding)
            y_max = min(h, y_max + padding)
            x_min = max(0, x_min - padding)
            x_max = min(w, x_max + padding)
            
            # 切り出し
            cropped = masked_image[y_min:y_max, x_min:x_max]
            
            if cropped.size == 0:
                return None
            
            # uint8に変換
            cropped_uint8 = (cropped * 255).astype(np.uint8)
            
            return {
                "extracted_rgb": cropped_uint8,
                "bbox": (x_min, y_min, x_max - x_min, y_max - y_min),
                "mask_area": np.sum(mask > 0.1)
            }
            
        except Exception as e:
            logger.error(f"マスク適用エラー {filename}: {e}")
            return None

    def _extract_enhanced_system(self, 
                               processed_bgr: np.ndarray,
                               processed_rgb: np.ndarray, 
                               input_path: Path,
                               verbose: bool = False) -> Optional[Dict[str, Any]]:
        """Enhanced System手法での抽出"""
        try:
            # モデル初期化確認
            if get_sam_model() is None or get_yolo_model() is None:
                initialize_models()
            
            sam_model = get_sam_model()
            yolo_model = get_yolo_model()
            
            # SAMマスク生成
            all_masks = sam_model.generate_masks(processed_bgr)
            if not all_masks:
                return {"success": False, "error": "no_masks_generated"}
            
            # フィルタリング
            character_masks = sam_model.filter_character_masks(all_masks)
            if not character_masks:
                return {"success": False, "error": "no_character_masks"}
            
            # YOLO検出との組み合わせ
            scored_masks = yolo_model.score_masks_with_detections(character_masks, processed_bgr)
            if not scored_masks:
                return {"success": False, "error": "no_scored_masks"}
            
            # Enhanced System のフィルタリング
            filtered_masks = apply_non_character_filter(scored_masks, processed_bgr)
            validated_masks = filter_non_character_masks(filtered_masks, processed_bgr)
            
            final_masks = validated_masks if validated_masks else filtered_masks
            
            if not final_masks:
                return {"success": False, "error": "no_final_masks"}
            
            # 最適マスク選択（fullbody_priority）
            best_mask = self._select_best_mask_simple(final_masks, processed_bgr.shape, "fullbody_priority")
            
            if best_mask:
                # キャラクター抽出・保存
                extracted_image = self._extract_character_from_mask(processed_bgr, best_mask)
                cropped_image = self._crop_to_content(extracted_image)
                
                # 品質情報
                has_face = self._detect_face_simple(cropped_image)
                size_info = f"{cropped_image.shape[1]}x{cropped_image.shape[0]}"
                
                return {
                    "success": True,
                    "image_bgr": cropped_image,
                    "size": size_info,
                    "face_detected": has_face,
                    "mask_count": len(final_masks)
                }
            
            return {"success": False, "error": "mask_selection_failed"}
            
        except Exception as e:
            return {"success": False, "error": f"enhanced_system_error: {e}"}

    def _extract_color_preserving(self, 
                                processed_bgr: np.ndarray,
                                processed_rgb: np.ndarray,
                                input_path: Path,
                                verbose: bool = False) -> Optional[Dict[str, Any]]:
        """色調保持境界強調手法での抽出"""
        try:
            # 色調保持境界強調
            enhanced_rgb, metrics = self.color_enhancer.enhance_image_boundaries(processed_rgb)
            enhanced_bgr = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)
            
            # モデル初期化確認
            if get_sam_model() is None or get_yolo_model() is None:
                initialize_models()
            
            sam_model = get_sam_model()
            yolo_model = get_yolo_model()
            
            # 強化された画像でマスク生成
            all_masks = sam_model.generate_masks(enhanced_bgr)
            if not all_masks:
                return {"success": False, "error": "no_masks_generated"}
            
            # フィルタリング
            character_masks = sam_model.filter_character_masks(all_masks)
            if not character_masks:
                return {"success": False, "error": "no_character_masks"}
            
            # YOLO検出
            scored_masks = yolo_model.score_masks_with_detections(character_masks, enhanced_bgr)
            if not scored_masks:
                return {"success": False, "error": "no_scored_masks"}
            
            # 最適マスク選択
            best_mask = self._select_best_mask_simple(scored_masks, enhanced_bgr.shape, "balanced")
            
            if best_mask:
                # 元の（非強化）画像でキャラクター抽出（色調保持のため）
                extracted_image = self._extract_character_from_mask(processed_bgr, best_mask)
                cropped_image = self._crop_to_content(extracted_image)
                
                # 品質情報
                has_face = self._detect_face_simple(cropped_image)
                size_info = f"{cropped_image.shape[1]}x{cropped_image.shape[0]}"
                
                return {
                    "success": True,
                    "image_bgr": cropped_image,
                    "size": size_info,
                    "face_detected": has_face,
                    "mask_count": len(scored_masks),
                    "color_quality": metrics.get("overall_quality", 0.5)
                }
            
            return {"success": False, "error": "mask_selection_failed"}
            
        except Exception as e:
            return {"success": False, "error": f"color_preserving_error: {e}"}

    def _extract_backup_method(self, 
                             processed_bgr: np.ndarray,
                             processed_rgb: np.ndarray,
                             input_path: Path,
                             verbose: bool = False) -> Optional[Dict[str, Any]]:
        """バックアップ手法での抽出"""
        try:
            # より保守的なパラメータでの処理
            # モデル初期化確認
            if get_sam_model() is None or get_yolo_model() is None:
                initialize_models()
            
            sam_model = get_sam_model()
            yolo_model = get_yolo_model()
            
            # 基本的なマスク生成（フィルタリング少なめ）
            all_masks = sam_model.generate_masks(processed_bgr)
            if not all_masks:
                return {"success": False, "error": "no_masks_generated"}
            
            # 基本的なキャラクターフィルタのみ
            character_masks = sam_model.filter_character_masks(all_masks)
            if not character_masks:
                return {"success": False, "error": "no_character_masks"}
            
            # YOLO検出
            scored_masks = yolo_model.score_masks_with_detections(character_masks, processed_bgr)
            if not scored_masks:
                # YOLOスコアリング失敗時は元のマスクを使用
                scored_masks = character_masks
            
            # シンプルなマスク選択（面積優先）
            best_mask = self._select_best_mask_simple(scored_masks, processed_bgr.shape, "size_priority")
            
            if best_mask:
                # キャラクター抽出
                extracted_image = self._extract_character_from_mask(processed_bgr, best_mask)
                cropped_image = self._crop_to_content(extracted_image)
                
                # 品質情報
                has_face = self._detect_face_simple(cropped_image)
                size_info = f"{cropped_image.shape[1]}x{cropped_image.shape[0]}"
                
                return {
                    "success": True,
                    "image_bgr": cropped_image,
                    "size": size_info,
                    "face_detected": has_face,
                    "mask_count": len(scored_masks)
                }
            
            return {"success": False, "error": "mask_selection_failed"}
            
        except Exception as e:
            return {"success": False, "error": f"backup_method_error: {e}"}

    def _select_best_mask_simple(self, masks: List[Any], image_shape: Tuple, method: str) -> Optional[Any]:
        """シンプルなマスク選択"""
        if not masks:
            return None
        
        height, width = image_shape[:2]
        
        best_mask = None
        best_score = -1.0
        
        for mask in masks:
            if hasattr(mask, 'mask') and hasattr(mask, 'composite_score'):
                mask_area = np.sum(mask.mask > 0) / (width * height)
                
                if method == "size_priority":
                    score = mask_area * 0.7 + mask.composite_score * 0.3
                elif method == "fullbody_priority":
                    score = mask.composite_score
                else:  # balanced
                    score = mask_area * 0.4 + mask.composite_score * 0.6
                
                if score > best_score:
                    best_score = score
                    best_mask = mask
        
        return best_mask

    def _extract_character_from_mask(self, image: np.ndarray, mask_obj: Any) -> np.ndarray:
        """マスクからキャラクターを抽出"""
        if hasattr(mask_obj, 'mask'):
            mask = mask_obj.mask
        else:
            mask = mask_obj
        
        # マスクが2次元でない場合は変換
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # マスクを正規化
        if mask.max() > 1:
            mask = (mask > 0).astype(np.uint8) * 255
        else:
            mask = (mask * 255).astype(np.uint8)
        
        # マスクを3チャンネルに変換
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_normalized = mask_3ch.astype(np.float32) / 255.0
        
        # 背景を黒に設定
        background = np.zeros_like(image, dtype=np.uint8)
        
        # マスク適用
        result = (image.astype(np.float32) * mask_normalized + 
                 background.astype(np.float32) * (1.0 - mask_normalized))
        
        return result.astype(np.uint8)

    def _crop_to_content(self, image: np.ndarray, padding: int = 10) -> np.ndarray:
        """コンテンツ部分のみにクロップ"""
        # グレースケール変換
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 非ゼロピクセルの境界を検出
        rows = np.any(gray > 0, axis=1)
        cols = np.any(gray > 0, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return image
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # パディングを追加
        rmin = max(0, rmin - padding)
        rmax = min(image.shape[0] - 1, rmax + padding)
        cmin = max(0, cmin - padding)
        cmax = min(image.shape[1] - 1, cmax + padding)
        
        return image[rmin:rmax+1, cmin:cmax+1]

    def _detect_face_simple(self, image: np.ndarray) -> bool:
        """簡易顔検出"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            return len(faces) > 0
        except:
            return False

    def _evaluate_result_quality(self, result: Dict[str, Any], original_image: np.ndarray) -> float:
        """結果の品質を評価"""
        if not result.get("success", False):
            return 0.0
        
        base_score = 0.5
        
        # 顔検出ボーナス
        if result.get("face_detected", False):
            base_score += 0.3
        
        # サイズ品質
        size_str = result.get("size", "0x0")
        try:
            w, h = map(int, size_str.split('x'))
            size_score = min(1.0, (w * h) / 50000)  # 50000ピクセルを基準
            base_score += size_score * 0.2
        except:
            pass
        
        # 色調品質（色調保持手法のみ）
        if "color_quality" in result:
            base_score += result["color_quality"] * 0.1
        
        # マスク数によるペナルティ（多すぎると不安定）
        mask_count = result.get("mask_count", 0)
        if mask_count > 100:
            base_score *= 0.9
        elif mask_count < 5:
            base_score *= 0.95
        
        return min(1.0, base_score)


def test_robust_extractor():
    """ロバスト抽出システムのテスト"""
    extractor = RobustCharacterExtractor()
    
    # テスト画像
    test_images = [
        "kana08_0001.jpg",  # 前回F評価
        "kana08_0003.jpg",  # 前回A評価
        "kana08_0000_cover.jpg"  # 腕のみ抽出問題
    ]
    
    input_dir = Path("/mnt/c/AItools/lora/train/yado/org/kana08")
    output_dir = Path("/tmp/robust_test")
    output_dir.mkdir(exist_ok=True)
    
    print("🚀 ロバスト抽出システム テスト")
    
    for filename in test_images:
        input_path = input_dir / filename
        output_path = output_dir / filename
        
        if input_path.exists():
            print(f"\n📸 テスト: {filename}")
            result = extractor.extract_character_robust(input_path, output_path, verbose=True)
            
            if result["success"]:
                print(f"✅ 成功: {result}")
            else:
                print(f"❌ 失敗: {result}")
        else:
            print(f"⚠️ ファイル不存在: {filename}")


if __name__ == "__main__":
    test_robust_extractor()