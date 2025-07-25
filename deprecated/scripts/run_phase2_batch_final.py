#!/usr/bin/env python3
"""
Phase 2 Final Batch Execution - Phase 2改善システム最終バッチ実行
境界認識強化 + 手足保護 + キャラクター優先順位学習の統合バッチ処理
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import cv2

import json
import logging
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
# 品質保護システム
from features.extraction.quality_guard_system import QualityGuardSystem
# 既存のロバスト抽出システム
from features.extraction.robust_extractor import RobustCharacterExtractor
# Phase 2新システム
from features.processing.advanced_boundary_detector import AdvancedBoundaryDetector
from features.processing.character_priority_learning import CharacterPriorityLearning
from features.processing.limb_protection_system import LimbProtectionSystem
from typing import Any, Dict, List

# 通知システム
# from features.common.notification.global_pushover import GlobalPushover

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Phase2FinalBatchProcessor:
    """Phase 2最終バッチ処理システム"""
    
    def __init__(self):
        """システム初期化"""
        # Phase 2改善システム
        self.boundary_detector = AdvancedBoundaryDetector(
            enable_panel_detection=True,
            enable_multi_stage_edge=True, 
            enable_boundary_completion=True
        )
        
        self.limb_protector = LimbProtectionSystem(
            enable_pose_estimation=True,
            enable_limb_completion=True,
            protection_margin=15
        )
        
        self.character_priority = CharacterPriorityLearning(
            enable_face_detection=True,
            enable_position_analysis=True,
            enable_size_priority=True
        )
        
        # 既存システム
        self.robust_extractor = RobustCharacterExtractor()
        
        # 品質保護システム
        self.quality_guard = QualityGuardSystem()
        
        # 通知システム
        self.notifier = None  # 暫定的に無効化
        
        # 処理統計
        self.processing_stats = {
            "total_images": 0,
            "processed_images": 0,
            "successful_extractions": 0,
            "phase2_enhancements": 0,
            "protected_files": 0,
            "processing_times": [],
            "start_time": time.time()
        }
        
        logger.info("Phase 2最終バッチ処理システム初期化完了")

    def process_single_image_phase2(self, 
                                   image_path: Path, 
                                   output_path: Path) -> Dict[str, Any]:
        """
        Phase 2統合処理による単一画像の処理
        
        Args:
            image_path: 入力画像パス
            output_path: 出力画像パス
            
        Returns:
            処理結果
        """
        start_time = time.time()
        filename = image_path.name
        
        logger.info(f"🚀 Phase 2処理開始: {filename}")
        
        result = {
            "filename": filename,
            "input_path": str(image_path),
            "output_path": str(output_path),
            "success": False,
            "phase2_applied": False,
            "protected": False,
            "processing_time": 0.0,
            "quality_score": 0.0,
            "phase2_analysis": {}
        }
        
        try:
            # 1. 品質保護システムチェック
            should_skip, protected_record = self.quality_guard.should_skip_processing(filename)
            
            if should_skip and protected_record:
                logger.info(f"✅ 品質保護: {filename} (評価={protected_record.rating}, スコア={protected_record.quality_score:.3f})")
                result.update({
                    "success": True,
                    "protected": True,
                    "quality_score": protected_record.quality_score,
                    "protection_reason": f"既存の{protected_record.rating}評価を保護"
                })
                self.processing_stats["protected_files"] += 1
                return result
            
            # 2. 画像読み込み
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"画像読み込み失敗: {image_path}")
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 3. Phase 2境界強化
            boundary_enhanced, boundary_analysis = self.boundary_detector.enhance_boundaries_advanced(
                image_rgb
            )
            
            # 境界強化品質チェック
            boundary_quality = boundary_analysis.get("enhancement_quality", 0.0)
            if boundary_quality > 0.5:  # 境界強化が有効な場合のみ適用
                processed_image = boundary_enhanced
                result["phase2_analysis"]["boundary_enhancement"] = boundary_analysis
                logger.info(f"   🎯 境界強化適用: 品質={boundary_quality:.3f}")
            else:
                processed_image = image_rgb
                logger.info(f"   ⚠️ 境界強化スキップ: 品質不足={boundary_quality:.3f}")
            
            # 4. ロバスト抽出実行
            extraction_result = self.robust_extractor.extract_character_robust(
                image_path, output_path, verbose=False
            )
            
            # 5. Phase 2後処理（抽出成功時のみ）
            if extraction_result.get("success", False) and output_path.exists():
                # 抽出結果を読み込み
                extracted_image = cv2.imread(str(output_path))
                if extracted_image is not None:
                    extracted_rgb = cv2.cvtColor(extracted_image, cv2.COLOR_BGR2RGB)
                    
                    # 簡易マスク生成
                    gray = cv2.cvtColor(extracted_rgb, cv2.COLOR_RGB2GRAY)
                    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
                    
                    # 手足保護処理
                    protected_mask, limb_analysis = self.limb_protector.protect_limbs_in_mask(
                        processed_image, mask
                    )
                    
                    # 保護が適用された場合、結果を更新
                    if limb_analysis.get("protection_applied", False):
                        # 保護されたマスクで最終画像を作成
                        final_image = self._apply_mask_to_image(processed_image, protected_mask)
                        
                        # 結果保存
                        final_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(output_path), final_bgr)
                        
                        result["phase2_analysis"]["limb_protection"] = limb_analysis
                        result["phase2_applied"] = True
                        self.processing_stats["phase2_enhancements"] += 1
                        
                        logger.info(f"   🦴 手足保護適用: 品質={limb_analysis.get('protection_quality', 0.0):.3f}")
            
            # 結果更新
            result.update({
                "success": extraction_result.get("success", False),
                "quality_score": extraction_result.get("quality_score", 0.0),
                "processing_time": time.time() - start_time,
                "method_used": extraction_result.get("best_method", "unknown")
            })
            
            if result["success"]:
                self.processing_stats["successful_extractions"] += 1
                logger.info(f"✅ 処理成功: {filename} (品質={result['quality_score']:.3f}, {result['processing_time']:.1f}秒)")
            else:
                logger.warning(f"❌ 処理失敗: {filename}")
        
        except Exception as e:
            logger.error(f"💥 処理エラー: {filename} - {e}")
            result["error"] = str(e)
        
        result["processing_time"] = time.time() - start_time
        self.processing_stats["processing_times"].append(result["processing_time"])
        
        return result

    def _apply_mask_to_image(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """マスクを画像に適用"""
        if len(mask.shape) == 3:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        else:
            mask_gray = mask.copy()
        
        # マスクを3チャンネルに変換
        mask_3ch = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2RGB) / 255.0
        
        # 背景を白にして適用
        background = np.ones_like(image) * 255
        result = image * mask_3ch + background * (1 - mask_3ch)
        
        return result.astype(np.uint8)

    def run_batch_processing(self):
        """バッチ処理実行"""
        logger.info("🎬 Phase 2最終バッチ処理開始")
        
        # 入力・出力ディレクトリ
        input_dir = Path("/mnt/c/AItools/lora/train/yado/org/kana08")
        output_dir = Path("/mnt/c/AItools/lora/train/yado/clipped_boundingbox/kana08_improvement_phase2_final")
        
        # 出力ディレクトリ作成
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 入力画像ファイル取得
        image_files = list(input_dir.glob("*.jpg"))
        image_files.sort()
        
        self.processing_stats["total_images"] = len(image_files)
        logger.info(f"📁 処理対象: {len(image_files)}画像")
        logger.info(f"📂 出力先: {output_dir}")
        
        # バッチ処理実行
        results = []
        
        for i, image_path in enumerate(image_files, 1):
            logger.info(f"📸 [{i}/{len(image_files)}] {image_path.name}")
            
            output_path = output_dir / image_path.name
            
            # 単一画像処理
            result = self.process_single_image_phase2(image_path, output_path)
            results.append(result)
            
            self.processing_stats["processed_images"] += 1
            
            # 進捗表示
            progress = (i / len(image_files)) * 100
            logger.info(f"📊 進捗: {progress:.1f}% ({i}/{len(image_files)})")
        
        # バッチ処理完了
        self._finalize_batch_processing(results, output_dir)
        
        return results

    def _finalize_batch_processing(self, results: List[Dict[str, Any]], output_dir: Path):
        """バッチ処理完了処理"""
        end_time = time.time()
        total_time = end_time - self.processing_stats["start_time"]
        
        # 統計計算
        successful_count = len([r for r in results if r["success"]])
        phase2_enhanced_count = len([r for r in results if r["phase2_applied"]])
        protected_count = len([r for r in results if r["protected"]])
        
        avg_processing_time = np.mean(self.processing_stats["processing_times"]) if self.processing_stats["processing_times"] else 0
        avg_quality = np.mean([r["quality_score"] for r in results if r["success"]]) if successful_count > 0 else 0
        
        # 結果サマリー
        summary = {
            "batch_info": {
                "total_images": self.processing_stats["total_images"],
                "processed_images": self.processing_stats["processed_images"],
                "successful_extractions": successful_count,
                "success_rate": successful_count / self.processing_stats["total_images"] * 100,
                "total_processing_time": total_time,
                "average_processing_time": avg_processing_time,
                "average_quality_score": avg_quality
            },
            "phase2_improvements": {
                "phase2_enhanced_count": phase2_enhanced_count,
                "enhancement_rate": phase2_enhanced_count / self.processing_stats["total_images"] * 100,
                "protected_files": protected_count,
                "protection_rate": protected_count / self.processing_stats["total_images"] * 100
            },
            "detailed_results": results
        }
        
        # 結果保存
        results_path = output_dir / "phase2_batch_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        # ログ出力
        logger.info("🎉 Phase 2最終バッチ処理完了")
        logger.info(f"📊 処理結果サマリー:")
        logger.info(f"   総画像数: {self.processing_stats['total_images']}")
        logger.info(f"   成功数: {successful_count} ({successful_count/self.processing_stats['total_images']*100:.1f}%)")
        logger.info(f"   Phase 2強化: {phase2_enhanced_count} ({phase2_enhanced_count/self.processing_stats['total_images']*100:.1f}%)")
        logger.info(f"   品質保護: {protected_count} ({protected_count/self.processing_stats['total_images']*100:.1f}%)")
        logger.info(f"   平均品質スコア: {avg_quality:.3f}")
        logger.info(f"   平均処理時間: {avg_processing_time:.1f}秒")
        logger.info(f"   総処理時間: {total_time/60:.1f}分")
        logger.info(f"💾 詳細結果: {results_path}")
        
        # Pushover通知
        if self.notifier:
            self._send_completion_notification(summary, output_dir)

    def _send_completion_notification(self, summary: Dict[str, Any], output_dir: Path):
        """完了通知送信"""
        batch_info = summary["batch_info"]
        phase2_info = summary["phase2_improvements"]
        
        # 通知メッセージ作成
        message = f"""🎉 Phase 2最終バッチ処理完了

📊 処理結果:
• 総画像数: {batch_info['total_images']}
• 成功率: {batch_info['success_rate']:.1f}% ({batch_info['successful_extractions']}/{batch_info['total_images']})
• 平均品質: {batch_info['average_quality_score']:.3f}

🚀 Phase 2改善効果:
• 強化適用: {phase2_info['enhancement_rate']:.1f}% ({phase2_info['phase2_enhanced_count']}/{batch_info['total_images']})
• 品質保護: {phase2_info['protection_rate']:.1f}% ({phase2_info['protected_files']}/{batch_info['total_images']})

⏱️ 処理時間:
• 総処理時間: {batch_info['total_processing_time']/60:.1f}分
• 平均処理時間: {batch_info['average_processing_time']:.1f}秒/枚

📂 出力先: {output_dir.name}"""
        
        try:
            # サンプル画像パス取得
            sample_images = []
            for result in summary["detailed_results"][:3]:
                if result["success"] and Path(result["output_path"]).exists():
                    sample_images.append(result["output_path"])
            
            # 通知送信
            self.notifier.send_notification(
                message=message,
                title="Phase 2バッチ処理完了"
            )
            
            logger.info("📱 Pushover通知送信完了")
            
        except Exception as e:
            logger.error(f"📱 Pushover通知送信失敗: {e}")


def main():
    """メイン実行"""
    processor = Phase2FinalBatchProcessor()
    
    try:
        results = processor.run_batch_processing()
        return len([r for r in results if r["success"]])
    
    except KeyboardInterrupt:
        logger.info("🛑 ユーザーによる処理中断")
        return 0
    
    except Exception as e:
        logger.error(f"💥 バッチ処理エラー: {e}")
        return 0


if __name__ == "__main__":
    success_count = main()
    print(f"\n🎯 最終結果: {success_count}枚の成功")