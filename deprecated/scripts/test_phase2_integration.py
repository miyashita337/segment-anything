#!/usr/bin/env python3
"""
Phase 2 Integration Test - Phase 2改善システム統合テスト
境界認識強化、手足保護、キャラクター優先順位学習の統合動作確認
"""

import numpy as np
import cv2

import json
import logging
import time

# 既存システムのインポート
from features.extraction.robust_extractor import RobustCharacterExtractor

# Phase 2システムのインポート
from features.processing.advanced_boundary_detector import AdvancedBoundaryDetector
from features.processing.character_priority_learning import CharacterPriorityLearning
from features.processing.limb_protection_system import LimbProtectionSystem
from features.processing.preprocessing.color_preserving_enhancer import ColorPreservingEnhancer
from pathlib import Path
from typing import Any, Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Phase2IntegratedSystem:
    """Phase 2統合改善システム"""

    def __init__(self):
        """システム初期化"""
        # Phase 2新システム
        self.boundary_detector = AdvancedBoundaryDetector(
            enable_panel_detection=True,
            enable_multi_stage_edge=True,
            enable_boundary_completion=True,
        )

        self.limb_protector = LimbProtectionSystem(
            enable_pose_estimation=True, enable_limb_completion=True, protection_margin=15
        )

        self.character_priority = CharacterPriorityLearning(
            enable_face_detection=True, enable_position_analysis=True, enable_size_priority=True
        )

        # 既存システム
        self.robust_extractor = RobustCharacterExtractor()
        self.color_enhancer = ColorPreservingEnhancer()

        logger.info("Phase 2統合システム初期化完了")

    def process_image_integrated(self, image_path: Path, output_path: Path) -> Dict[str, Any]:
        """
        統合処理によるキャラクター抽出

        Args:
            image_path: 入力画像パス
            output_path: 出力画像パス

        Returns:
            処理結果の詳細情報
        """
        start_time = time.time()
        logger.info(f"統合処理開始: {image_path.name}")

        # 画像読み込み
        image = cv2.imread(str(image_path))
        if image is None:
            return {"error": f"画像の読み込みに失敗: {image_path}"}

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = {
            "input_path": str(image_path),
            "output_path": str(output_path),
            "image_shape": image_rgb.shape,
            "processing_steps": [],
            "phase2_analysis": {},
            "final_result": {},
            "processing_time": 0.0,
            "success": False,
        }

        try:
            # ステップ1: 高度境界強化
            step1_start = time.time()
            (
                boundary_enhanced,
                boundary_analysis,
            ) = self.boundary_detector.enhance_boundaries_advanced(image_rgb)
            step1_time = time.time() - step1_start

            result["processing_steps"].append(
                {"step": "boundary_enhancement", "time": step1_time, "analysis": boundary_analysis}
            )

            # ステップ2: 色保持強化
            step2_start = time.time()
            color_enhanced = self.color_enhancer.enhance_image_boundaries(boundary_enhanced)
            step2_time = time.time() - step2_start

            result["processing_steps"].append({"step": "color_enhancement", "time": step2_time})

            # ステップ3: ロバスト抽出（複数候補取得）
            step3_start = time.time()
            extraction_result = self.robust_extractor.extract_character_robust(
                image_path, output_path, verbose=False
            )
            step3_time = time.time() - step3_start

            result["processing_steps"].append(
                {
                    "step": "robust_extraction",
                    "time": step3_time,
                    "methods_used": extraction_result.get("methods_used", []),
                    "best_method": extraction_result.get("best_method", "unknown"),
                }
            )

            # ステップ4: キャラクター候補分析（ダミー候補で代替）
            step4_start = time.time()
            dummy_candidates = self._create_dummy_candidates(image_rgb, extraction_result)
            (
                prioritized_candidates,
                priority_analysis,
            ) = self.character_priority.prioritize_characters(color_enhanced, dummy_candidates)
            step4_time = time.time() - step4_start

            result["processing_steps"].append(
                {
                    "step": "character_prioritization",
                    "time": step4_time,
                    "candidates_analyzed": len(dummy_candidates),
                    "primary_character": priority_analysis.get("primary_character"),
                }
            )

            # ステップ5: 手足保護処理
            step5_start = time.time()
            if extraction_result.get("success", False):
                # 抽出されたマスクを読み込み
                if output_path.exists():
                    extracted_image = cv2.imread(str(output_path))
                    if extracted_image is not None:
                        extracted_rgb = cv2.cvtColor(extracted_image, cv2.COLOR_BGR2RGB)

                        # 簡易マスク生成
                        gray = cv2.cvtColor(extracted_rgb, cv2.COLOR_RGB2GRAY)
                        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

                        # 手足保護適用
                        protected_mask, limb_analysis = self.limb_protector.protect_limbs_in_mask(
                            color_enhanced, mask
                        )

                        # 保護されたマスクで最終画像を作成
                        final_image = self._apply_protected_mask(color_enhanced, protected_mask)

                        # 最終結果保存
                        final_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(output_path), final_bgr)

                        result["processing_steps"].append(
                            {
                                "step": "limb_protection",
                                "time": time.time() - step5_start,
                                "protection_applied": limb_analysis.get(
                                    "protection_applied", False
                                ),
                                "protection_quality": limb_analysis.get("protection_quality", 0.0),
                            }
                        )

            # Phase 2分析結果統合
            result["phase2_analysis"] = {
                "boundary_enhancement": boundary_analysis,
                "character_priority": priority_analysis,
                "limb_protection": result["processing_steps"][-1]
                if len(result["processing_steps"]) >= 5
                else {},
            }

            # 最終結果
            result["final_result"] = {
                "extraction_success": extraction_result.get("success", False),
                "quality_score": extraction_result.get("quality_score", 0.0),
                "output_exists": output_path.exists(),
                "phase2_enhancements": {
                    "boundary_quality": boundary_analysis.get("enhancement_quality", 0.0),
                    "priority_score": priority_analysis.get("primary_character", {}).get(
                        "priority_score", 0.0
                    ),
                    "limb_protection": result["processing_steps"][-1].get("protection_quality", 0.0)
                    if len(result["processing_steps"]) >= 5
                    else 0.0,
                },
            }

            result["success"] = True

        except Exception as e:
            logger.error(f"統合処理エラー: {e}")
            result["error"] = str(e)

        result["processing_time"] = time.time() - start_time
        logger.info(f"統合処理完了: {result['processing_time']:.2f}秒")

        return result

    def _create_dummy_candidates(
        self, image: np.ndarray, extraction_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """抽出結果からダミー候補を生成"""
        height, width = image.shape[:2]

        # 基本候補（中央）
        candidates = [
            {
                "bbox": (width // 4, height // 4, width // 2, height // 2),
                "mask": np.ones((height // 2, width // 2), dtype=np.uint8) * 255,
                "confidence": extraction_result.get("quality_score", 0.7),
                "area": (width // 2) * (height // 2),
                "center": (width // 2, height // 2),
            }
        ]

        # 追加候補（異なる位置・サイズ）
        if extraction_result.get("success", False):
            candidates.extend(
                [
                    {
                        "bbox": (width // 6, height // 6, width // 3, height // 3),
                        "mask": np.ones((height // 3, width // 3), dtype=np.uint8) * 255,
                        "confidence": 0.6,
                        "area": (width // 3) * (height // 3),
                        "center": (width // 3, height // 3),
                    },
                    {
                        "bbox": (width // 2, height // 3, width // 4, height // 4),
                        "mask": np.ones((height // 4, width // 4), dtype=np.uint8) * 255,
                        "confidence": 0.5,
                        "area": (width // 4) * (height // 4),
                        "center": (5 * width // 8, 5 * height // 12),
                    },
                ]
            )

        return candidates

    def _apply_protected_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """保護されたマスクを画像に適用"""
        if len(mask.shape) == 3:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        else:
            mask_gray = mask.copy()

        # マスクを3チャンネルに変換
        mask_3ch = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2RGB)

        # マスクによる抽出
        result = cv2.bitwise_and(image, mask_3ch)

        # 背景を透明化（白背景）
        background = np.ones_like(image) * 255
        mask_inv = cv2.bitwise_not(mask_gray)
        mask_inv_3ch = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2RGB)
        background_part = cv2.bitwise_and(background, mask_inv_3ch)

        final_result = cv2.add(result, background_part)
        return final_result


def run_integration_test():
    """統合テスト実行"""
    print("🧪 Phase 2統合システムテスト開始")

    # システム初期化
    integrated_system = Phase2IntegratedSystem()

    # テスト画像
    test_images = [
        Path("/mnt/c/AItools/lora/train/yado/org/kaname08/kaname08_0001.jpg"),
        Path("/mnt/c/AItools/lora/train/yado/org/kaname08/kaname08_0002.jpg"),
        Path("/mnt/c/AItools/lora/train/yado/org/kaname08/kaname08_0009.jpg"),
    ]

    test_results = []

    for i, test_image in enumerate(test_images):
        if not test_image.exists():
            print(f"⚠️ テスト画像が見つかりません: {test_image}")
            continue

        print(f"\\n📸 テスト {i+1}/3: {test_image.name}")

        # 出力パス
        output_path = Path(f"/tmp/phase2_integration_test_{i+1}.jpg")

        # 統合処理実行
        result = integrated_system.process_image_integrated(test_image, output_path)

        if result["success"]:
            print(f"✅ 処理成功 ({result['processing_time']:.2f}秒)")

            # Phase 2改善効果
            enhancements = result["final_result"]["phase2_enhancements"]
            print(f"   境界強化品質: {enhancements['boundary_quality']:.3f}")
            print(f"   キャラクター優先度: {enhancements['priority_score']:.3f}")
            print(f"   手足保護品質: {enhancements['limb_protection']:.3f}")

            # 処理ステップ時間
            for step in result["processing_steps"]:
                print(f"   {step['step']}: {step['time']:.3f}秒")

        else:
            print(f"❌ 処理失敗: {result.get('error', '不明なエラー')}")

        test_results.append(result)

    # 統合テスト結果サマリー
    print("\\n📊 統合テスト結果サマリー:")
    successful_tests = [r for r in test_results if r["success"]]
    print(
        f"成功率: {len(successful_tests)}/{len(test_results)} ({len(successful_tests)/len(test_results)*100:.1f}%)"
    )

    if successful_tests:
        avg_time = sum(r["processing_time"] for r in successful_tests) / len(successful_tests)
        print(f"平均処理時間: {avg_time:.2f}秒")

        # Phase 2改善効果の平均
        avg_boundary = sum(
            r["final_result"]["phase2_enhancements"]["boundary_quality"] for r in successful_tests
        ) / len(successful_tests)
        avg_priority = sum(
            r["final_result"]["phase2_enhancements"]["priority_score"] for r in successful_tests
        ) / len(successful_tests)
        avg_limb = sum(
            r["final_result"]["phase2_enhancements"]["limb_protection"] for r in successful_tests
        ) / len(successful_tests)

        print(f"平均改善効果:")
        print(f"  境界強化: {avg_boundary:.3f}")
        print(f"  キャラクター優先: {avg_priority:.3f}")
        print(f"  手足保護: {avg_limb:.3f}")

    # 結果をJSONで保存
    results_path = Path("/tmp/phase2_integration_test_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\\n💾 詳細結果保存: {results_path}")
    print("🎉 Phase 2統合テスト完了")


if __name__ == "__main__":
    run_integration_test()
