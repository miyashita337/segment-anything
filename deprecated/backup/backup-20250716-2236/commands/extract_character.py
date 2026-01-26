#!/usr/bin/env python3
"""
Character Extraction Command
Main command for extracting characters from manga images using SAM + YOLO
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2

from hooks.start import get_performance_monitor, get_sam_model, get_yolo_model
from utils.difficult_pose import (
    DifficultPoseProcessor,
    detect_difficult_pose,
    get_difficult_pose_config,
    process_with_retry,
)
from utils.learned_quality_assessment import LearnedQualityAssessment, assess_image_quality
from utils.postprocessing import (
    calculate_mask_quality_metrics,
    crop_to_content,
    enhance_character_mask,
    extract_character_from_image,
    save_character_result,
)
from utils.preprocessing import preprocess_image_pipeline
from utils.text_detection import TextDetector


def extract_character_from_path(
    image_path: str,
    output_path: Optional[str] = None,
    enhance_contrast: bool = False,
    filter_text: bool = True,
    save_mask: bool = False,
    save_transparent: bool = False,
    min_yolo_score: float = 0.1,
    verbose: bool = True,
    difficult_pose: bool = False,
    low_threshold: bool = False,
    auto_retry: bool = False,
    high_quality: bool = False,
    manga_mode: bool = False,
    effect_removal: bool = False,
    panel_split: bool = False,
    multi_character_criteria: str = "balanced",
    adaptive_learning: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    画像パスからキャラクターを抽出 (Phase 2対応版)

    Args:
        image_path: 入力画像パス
        output_path: 出力パス（None の場合は自動生成）
        enhance_contrast: コントラスト強化
        filter_text: テキスト領域フィルタリング
        save_mask: マスクを保存
        save_transparent: 透明背景版を保存
        min_yolo_score: YOLO最小スコア
        verbose: 詳細出力
        difficult_pose: 複雑ポーズモード
        low_threshold: 低閾値モード（YOLO 0.02）
        auto_retry: 自動リトライモード
        high_quality: 高品質SAM処理
        manga_mode: 漫画前処理モード (Phase 2)
        effect_removal: エフェクト線除去を有効化 (Phase 2)
        panel_split: マルチコマ分割を有効化 (Phase 2)
        multi_character_criteria: 複数キャラクター選択基準 ('balanced', 'size_priority', 'fullbody_priority', 'central_priority', 'confidence_priority')
        adaptive_learning: 適応学習モード（281評価データに基づく最適手法選択）

    Returns:
        抽出結果の辞書
    """
    result = {
        "success": False,
        "input_path": image_path,
        "output_path": None,
        "processing_time": 0.0,
        "mask_quality": {},
        "error": None,
        "adaptive_learning_info": None,
    }

    start_time = time.time()

    # 自動リトライモードの場合は process_with_retry を使用
    if auto_retry:
        if verbose:
            print(f"🔄 自動リトライモードでキャラクター抽出開始: {image_path}")

        def extract_function(img_path, **config):
            # 元の処理ロジックを呼び出し（リトライ用）
            return extract_character_from_path(
                img_path,
                output_path,
                enhance_contrast,
                filter_text,
                save_mask,
                save_transparent,
                config.get("min_yolo_score", min_yolo_score),
                verbose=False,  # リトライ中は詳細出力を抑制
                difficult_pose=False,
                low_threshold=False,
                auto_retry=False,  # 無限ループ防止
                high_quality=config.get("enable_enhanced_processing", high_quality),
                manga_mode=config.get("enable_manga_preprocessing", manga_mode),
                effect_removal=config.get("enable_effect_removal", effect_removal),
                panel_split=config.get("enable_panel_split", panel_split),
                multi_character_criteria=multi_character_criteria,
                adaptive_learning=adaptive_learning,
                **{
                    k: v
                    for k, v in config.items()
                    if k
                    not in [
                        "min_yolo_score",
                        "enable_enhanced_processing",
                        "enable_manga_preprocessing",
                        "enable_effect_removal",
                        "enable_panel_split",
                    ]
                },
            )

        return process_with_retry(image_path, extract_function, max_retries=4)

    try:
        # Phase 3: 適応学習モード - 281評価データに基づく最適手法選択
        if adaptive_learning:
            if verbose:
                print(f"🧠 適応学習モード: 281評価データに基づく最適手法選択を実行中...")

            try:
                # 品質評価システムで画像特性を分析し最適手法を予測
                quality_prediction = assess_image_quality(image_path)
                result["adaptive_learning_info"] = {
                    "predicted_quality": quality_prediction.predicted_quality,
                    "confidence": quality_prediction.confidence,
                    "recommended_method": quality_prediction.recommended_method,
                    "fallback_method": quality_prediction.fallback_method,
                    "reasoning": quality_prediction.reasoning,
                    "image_characteristics": quality_prediction.image_characteristics,
                }

                # 推奨手法をmulti_character_criteriaに適用
                multi_character_criteria = quality_prediction.recommended_method

                # ImageCharacteristicsオブジェクトを作成
                from utils.learned_quality_assessment import ImageCharacteristics

                img_chars_dict = quality_prediction.image_characteristics
                img_chars = (
                    ImageCharacteristics(**img_chars_dict)
                    if isinstance(img_chars_dict, dict)
                    else img_chars_dict
                )

                # 画像特性に基づく最適化パラメータ取得
                assessor = LearnedQualityAssessment()
                optimized_params = assessor.get_method_parameters(
                    quality_prediction.recommended_method, img_chars
                )

                # 最適化パラメータを適用
                if optimized_params.get("score_threshold"):
                    min_yolo_score = optimized_params["score_threshold"]

                # 境界問題がある場合は漫画前処理を強制有効化（一時的に無効化）
                if img_chars.has_boundary_complexity:
                    # manga_mode = True
                    # effect_removal = True
                    if verbose:
                        print(f"   🎨 境界問題検出: 漫画前処理を強制有効化（無効化中）")

                if verbose:
                    print(f"   📊 推奨手法: {quality_prediction.recommended_method}")
                    print(f"   🎯 予測品質: {quality_prediction.predicted_quality:.3f}")
                    print(f"   🔧 信頼度: {quality_prediction.confidence:.3f}")
                    print(f"   📝 理由: {quality_prediction.reasoning}")
                    print(f"   ⚙️  最適YOLO閾値: {min_yolo_score}")

                    # 画像特性の詳細表示
                    if img_chars.has_complex_pose:
                        print(f"   🤸 複雑姿勢検出")
                    if img_chars.has_multiple_characters:
                        print(f"   👥 複数キャラクター")
                    if img_chars.has_screentone_issues:
                        print(f"   📰 スクリーントーン境界問題")
                    if img_chars.has_mosaic_issues:
                        print(f"   🔲 モザイク境界問題")

            except Exception as e:
                if verbose:
                    print(f"⚠️ 適応学習エラー、デフォルト手法で継続: {e}")
                # エラー時はデフォルト手法を継続使用
                result["adaptive_learning_info"] = {"error": str(e), "fallback_to_default": True}

        # Get models
        sam_model = get_sam_model()
        yolo_model = get_yolo_model()
        performance_monitor = get_performance_monitor()

        if not sam_model or not yolo_model:
            if verbose:
                print("🔄 モデル未初期化、自動初期化を実行中...")

            # 直接初期化関数を呼び出し
            try:
                from hooks.start import start

                start()

                # 再度モデル取得を試行
                sam_model = get_sam_model()
                yolo_model = get_yolo_model()
                performance_monitor = get_performance_monitor()

                if verbose:
                    print("✅ モデル自動初期化完了")

                if not sam_model or not yolo_model:
                    raise RuntimeError("Auto initialization failed. Models still not available.")

            except ImportError:
                # start関数がない場合のフォールバック
                raise RuntimeError("Models not initialized. Please run: python3 hooks/start.py")
            except Exception as e:
                raise RuntimeError(f"Failed to auto-initialize models: {e}")

        # 複雑ポーズ判定と設定調整 (Phase 2対応版)
        if difficult_pose or low_threshold or manga_mode:
            processor = DifficultPoseProcessor()

            if difficult_pose:
                # 複雑ポーズモード: 自動判定による設定
                complexity_info = processor.detect_pose_complexity(image_path)
                recommended_config = processor.get_recommended_config(complexity_info)

                if verbose:
                    print(
                        f"🔍 ポーズ複雑度: {complexity_info['complexity']} (スコア: {complexity_info['score']:.1f})"
                    )
                    print(f"🔧 推奨設定適用: {recommended_config['description']}")

                # 推奨設定を適用
                min_yolo_score = min(min_yolo_score, recommended_config["min_yolo_score"])
                if "sam_points_per_side" in recommended_config:
                    # SAM設定をkwargsに追加
                    kwargs.update(
                        {
                            "sam_points_per_side": recommended_config["sam_points_per_side"],
                            "sam_pred_iou_thresh": recommended_config["sam_pred_iou_thresh"],
                            "sam_stability_score_thresh": recommended_config[
                                "sam_stability_score_thresh"
                            ],
                        }
                    )

            if low_threshold:
                min_yolo_score = 0.02
                if verbose:
                    print(f"🔧 低閾値モード: YOLO閾値を{min_yolo_score}に設定")

            # Phase 2: 漫画前処理モード
            if manga_mode or effect_removal or panel_split:
                if verbose:
                    print(f"🎨 漫画前処理モード有効")
                    print(f"   エフェクト線除去: {'✅' if effect_removal else '❌'}")
                    print(f"   マルチコマ分割: {'✅' if panel_split else '❌'}")

                # 前処理を適用
                processed_image_path = processor.preprocess_for_difficult_pose(
                    image_path,
                    enable_manga_preprocessing=True,
                    enable_effect_removal=effect_removal,
                    enable_panel_split=panel_split,
                )

                # 処理済み画像を使用
                image_path = processed_image_path

        if high_quality:
            # 高品質SAM設定
            kwargs.update(
                {
                    "sam_points_per_side": kwargs.get("sam_points_per_side", 64),
                    "sam_pred_iou_thresh": kwargs.get("sam_pred_iou_thresh", 0.88),
                    "sam_stability_score_thresh": kwargs.get("sam_stability_score_thresh", 0.92),
                }
            )
            if verbose:
                print(f"🔧 高品質モード: SAM密度 {kwargs.get('sam_points_per_side', 64)} ポイント/サイド")

        if verbose:
            print(f"🎯 キャラクター抽出開始: {image_path}")
            print(f"📊 YOLO閾値: {min_yolo_score}")

        # Step 1: Image preprocessing
        performance_monitor.start_stage("Image Preprocessing")
        bgr_image, rgb_image, scale = preprocess_image_pipeline(
            image_path, enhance_contrast=enhance_contrast
        )

        if rgb_image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        performance_monitor.end_stage()

        # Step 2: SAM mask generation
        performance_monitor.start_stage("SAM Mask Generation")

        # 高品質/複雑ポーズモードでSAM設定を動的に適用
        if any(key.startswith("sam_") for key in kwargs.keys()) or high_quality:
            # SAMGeneratorを一時的に再構築
            try:
                from segment_anything import SamAutomaticMaskGenerator

                sam_params = {
                    "model": sam_model.sam,
                    "points_per_side": kwargs.get("sam_points_per_side", 32),
                    "pred_iou_thresh": kwargs.get("sam_pred_iou_thresh", 0.8),
                    "stability_score_thresh": kwargs.get("sam_stability_score_thresh", 0.85),
                    "crop_n_layers": 1,
                    "crop_n_points_downscale_factor": 2,
                    "min_mask_region_area": 100,
                }

                if verbose:
                    print(f"🔧 カスタムSAM設定適用:")
                    print(f"   ポイント密度: {sam_params['points_per_side']}")
                    print(f"   IoU閾値: {sam_params['pred_iou_thresh']}")
                    print(f"   安定性閾値: {sam_params['stability_score_thresh']}")

                # 一時的なマスクジェネレータで処理
                temp_generator = SamAutomaticMaskGenerator(**sam_params)
                all_masks = temp_generator.generate(rgb_image)

            except Exception as e:
                if verbose:
                    print(f"⚠️ カスタムSAM設定失敗、デフォルト設定で継続: {e}")
                all_masks = sam_model.generate_masks(rgb_image)
        else:
            all_masks = sam_model.generate_masks(rgb_image)

        if not all_masks:
            raise ValueError("No masks generated by SAM")

        character_masks = sam_model.filter_character_masks(all_masks)

        if verbose:
            print(f"📊 生成マスク: {len(all_masks)} → キャラクター候補: {len(character_masks)}")

        performance_monitor.end_stage()

        # Step 3: YOLO scoring
        performance_monitor.start_stage("YOLO Scoring")
        scored_masks = yolo_model.score_masks_with_detections(character_masks, bgr_image)

        best_mask = yolo_model.get_best_character_mask(
            scored_masks, bgr_image, min_yolo_score=min_yolo_score
        )

        if best_mask is None:
            raise ValueError(f"No good character masks found (min YOLO score: {min_yolo_score})")

        if verbose:
            print(
                f"🎯 最適マスク選択: YOLO score={best_mask['yolo_score']:.3f}, "
                f"combined score={best_mask['combined_score']:.3f}"
            )
            if adaptive_learning:
                print(f"   🧠 推奨手法: {multi_character_criteria} (適応学習)")
            else:
                print(f"   🔧 選択基準: {multi_character_criteria}")

        performance_monitor.end_stage()

        # Step 4: Text filtering (optional)
        if filter_text:
            performance_monitor.start_stage("Text Filtering")
            text_detector = TextDetector(use_easyocr=True)

            text_density = text_detector.calculate_text_density_score(bgr_image, best_mask["bbox"])

            if text_density > 0.5:
                if verbose:
                    print(f"⚠️ 高テキスト密度検出: {text_density:.3f} - 処理続行")

            # Add text density to result
            best_mask["text_density"] = text_density
            performance_monitor.end_stage()

        # Step 5: Mask refinement
        performance_monitor.start_stage("Mask Refinement")
        raw_mask = sam_model.mask_to_binary(best_mask)

        # 適応学習による境界問題対応 + 複雑ポーズ用の強化マスク処理
        boundary_complexity = False
        if (
            adaptive_learning
            and result["adaptive_learning_info"]
            and "image_characteristics" in result["adaptive_learning_info"]
        ):
            img_chars_dict = result["adaptive_learning_info"]["image_characteristics"]
            boundary_complexity = img_chars_dict.get("has_boundary_complexity", False)

        use_enhanced_processing = (
            difficult_pose or low_threshold or high_quality or boundary_complexity
        )

        if use_enhanced_processing:
            if verbose:
                enhancement_reason = []
                if difficult_pose:
                    enhancement_reason.append("複雑ポーズ")
                if low_threshold:
                    enhancement_reason.append("低閾値")
                if high_quality:
                    enhancement_reason.append("高品質")
                if boundary_complexity:
                    enhancement_reason.append("境界問題対応")

                print(f"🔧 マスク強化処理を適用: {'+'.join(enhancement_reason)}")

            # DifficultPoseProcessorを使用した強化処理
            if "processor" not in locals():
                processor = DifficultPoseProcessor()

            enhanced_mask = processor.enhance_mask_for_complex_pose(raw_mask, bgr_image)
        else:
            enhanced_mask = enhance_character_mask(
                raw_mask, remove_small_area=100, smooth_kernel=3, fill_holes=True
            )

        # Calculate mask quality metrics
        quality_metrics = calculate_mask_quality_metrics(enhanced_mask)
        result["mask_quality"] = quality_metrics

        if verbose:
            print(
                f"📐 マスク品質: coverage={quality_metrics['coverage_ratio']:.3f}, "
                f"compactness={quality_metrics['compactness']:.3f}"
            )

        performance_monitor.end_stage()

        # Step 6: Character extraction
        performance_monitor.start_stage("Character Extraction")
        character_image = extract_character_from_image(
            bgr_image, enhanced_mask, background_color=(0, 0, 0)  # Black background
        )

        # Crop to content
        cropped_character, cropped_mask, crop_bbox = crop_to_content(
            character_image, enhanced_mask, padding=10
        )

        performance_monitor.end_stage()

        # Step 7: Save results
        performance_monitor.start_stage("Saving Results")

        # Generate output path if not provided
        if output_path is None:
            input_path = Path(image_path)
            output_dir = input_path.parent / "character_output"
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / input_path.stem

        # Save results
        save_success = save_character_result(
            cropped_character,
            cropped_mask,
            str(output_path),
            save_mask=save_mask,
            save_transparent=save_transparent,
        )

        if not save_success:
            raise RuntimeError("Failed to save results")

        result["output_path"] = str(output_path)
        performance_monitor.end_stage()

        # Success
        result["success"] = True
        result["processing_time"] = time.time() - start_time

        # 適応学習結果のログ記録
        if adaptive_learning and result["adaptive_learning_info"]:
            try:
                assessor = LearnedQualityAssessment()
                # 実際の品質を計算（マスク品質メトリクスから推定）
                actual_quality = (
                    quality_metrics["coverage_ratio"] * 2 + quality_metrics["compactness"] * 2 + 1.0
                )  # 1-5スケール推定

                # 予測結果をログに記録（将来の学習更新用）
                assessor.log_prediction_result(
                    image_path,
                    type("QualityPrediction", (), result["adaptive_learning_info"])(),
                    actual_quality=actual_quality,
                )

                result["adaptive_learning_info"]["estimated_actual_quality"] = actual_quality

            except Exception as e:
                if verbose:
                    print(f"⚠️ 適応学習ログ記録エラー: {e}")

        if verbose:
            print(f"✅ キャラクター抽出完了: {result['processing_time']:.2f}秒")
            print(f"   出力: {result['output_path']}")

            # 適応学習結果のサマリ表示
            if (
                adaptive_learning
                and result["adaptive_learning_info"]
                and not result["adaptive_learning_info"].get("error")
            ):
                adaptive_info = result["adaptive_learning_info"]
                print(f"   🧠 適応学習結果:")
                print(f"      手法: {adaptive_info['recommended_method']}")
                print(f"      予測品質: {adaptive_info['predicted_quality']:.3f}")
                if "estimated_actual_quality" in adaptive_info:
                    print(f"      実際品質: {adaptive_info['estimated_actual_quality']:.3f}")
                    prediction_error = abs(
                        adaptive_info["predicted_quality"]
                        - adaptive_info["estimated_actual_quality"]
                    )
                    print(f"      予測精度: ±{prediction_error:.3f}")

        return result

    except Exception as e:
        result["error"] = str(e)
        result["processing_time"] = time.time() - start_time

        if verbose:
            print(f"❌ 抽出失敗: {e}")

        return result


def batch_extract_characters(input_dir: str, output_dir: str, **extract_kwargs) -> Dict[str, Any]:
    """
    ディレクトリ内の全画像に対してバッチ処理 (TDR安全対策版)

    Args:
        input_dir: 入力ディレクトリ
        output_dir: 出力ディレクトリ
        **extract_kwargs: extract_character_from_path の引数

    Returns:
        バッチ処理結果
    """
    import torch

    import gc

    def gpu_memory_cleanup():
        """GPU メモリクリーンアップ (TDR対策)"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
        except Exception as e:
            print(f"⚠️ GPU メモリクリーンアップ失敗: {e}")

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        return {"success": False, "error": f"Input directory not found: {input_dir}"}

    # 画像ファイルを取得
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_files = []

    for ext in image_extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))

    if not image_files:
        return {"success": False, "error": f"No image files found in {input_dir}"}

    # 出力ディレクトリ作成
    output_path.mkdir(parents=True, exist_ok=True)

    # バッチ処理実行
    results = []
    successful = 0

    print(f"🚀 バッチ処理開始: {len(image_files)} 画像")
    print(f"🛡️ TDR安全対策: GPU メモリクリーンアップ有効")

    for i, image_file in enumerate(image_files, 1):
        print(f"\n📁 処理中 [{i}/{len(image_files)}]: {image_file.name}")

        try:
            # 出力パス生成
            output_file = output_path / image_file.stem

            # 抽出実行
            batch_kwargs = extract_kwargs.copy()
            batch_kwargs["verbose"] = False

            result = extract_character_from_path(
                str(image_file), output_path=str(output_file), **batch_kwargs
            )

            result["filename"] = image_file.name
            results.append(result)

            if result["success"]:
                successful += 1
                print(f"✅ 成功: {image_file.name}")
            else:
                print(f"❌ 失敗: {image_file.name} - {result.get('error', 'Unknown error')}")

            # 5枚ごとにGPU メモリクリーンアップ (TDR対策)
            if i % 5 == 0:
                print(f"🧹 GPU メモリクリーンアップ実行 ({i}/{len(image_files)})")
                gpu_memory_cleanup()

        except KeyboardInterrupt:
            print("\n⏹️ ユーザーによる処理中断")
            gpu_memory_cleanup()
            break
        except Exception as e:
            print(f"❌ 画像処理エラー: {image_file.name} - {e}")
            # エラーでもバッチ処理は継続
            error_result = {
                "success": False,
                "error": str(e),
                "filename": image_file.name,
                "processing_time": 0.0,
            }
            results.append(error_result)

            # エラー時もGPU クリーンアップ
            gpu_memory_cleanup()

    # 最終GPU メモリクリーンアップ
    print("\n🧹 最終GPU メモリクリーンアップ...")
    gpu_memory_cleanup()

    # 結果サマリ
    batch_result = {
        "success": True,
        "total_files": len(image_files),
        "successful": successful,
        "failed": len(image_files) - successful,
        "success_rate": successful / len(image_files) if len(image_files) > 0 else 0,
        "results": results,
    }

    print(f"\n📊 バッチ処理完了:")
    print(f"   成功: {successful}/{len(image_files)} ({batch_result['success_rate']:.1%})")
    print(f"   🛡️ TDR対策: 安全にGPU処理完了")

    # Pushover通知送信
    try:
        from utils.notification import send_batch_notification

        print("\n📱 通知送信中...")
        notification_sent = send_batch_notification(
            successful=successful,
            total=len(image_files),
            failed=len(image_files) - successful,
            total_time=batch_result["total_time"],
        )

        if notification_sent:
            print("✅ Pushover通知送信完了")
        else:
            print("⚠️ Pushover通知送信失敗またはスキップ")
    except ImportError:
        print("⚠️ 通知モジュールが見つかりません")
    except Exception as e:
        print(f"⚠️ 通知送信エラー: {e}")

    return batch_result


def main():
    """Main function for command line interface"""
    parser = argparse.ArgumentParser(description="Character Extraction using SAM + YOLO")

    parser.add_argument("input", help="Input image path or directory")
    parser.add_argument("-o", "--output", help="Output path (auto-generated if not specified)")
    parser.add_argument("--batch", action="store_true", help="Batch processing mode")
    parser.add_argument("--enhance-contrast", action="store_true", help="Enhance image contrast")
    parser.add_argument(
        "--filter-text", action="store_true", default=True, help="Filter text regions"
    )
    parser.add_argument("--save-mask", action="store_true", default=False, help="Save mask files")
    parser.add_argument(
        "--save-transparent", action="store_true", default=False, help="Save transparent background"
    )
    parser.add_argument(
        "--min-yolo-score", type=float, default=0.1, help="Minimum YOLO score threshold"
    )
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")

    # 複雑ポーズ・ダイナミック構図対応オプション
    parser.add_argument(
        "--difficult-pose", action="store_true", help="Enable difficult pose processing mode"
    )
    parser.add_argument(
        "--low-threshold", action="store_true", help="Use low threshold settings (YOLO score 0.02)"
    )
    parser.add_argument(
        "--auto-retry", action="store_true", help="Enable automatic retry with progressive settings"
    )
    parser.add_argument(
        "--high-quality", action="store_true", help="Enable high-quality SAM processing"
    )

    # Phase 2: 漫画前処理オプション
    parser.add_argument(
        "--manga-mode", action="store_true", help="Enable manga-specific preprocessing (Phase 2)"
    )
    parser.add_argument(
        "--effect-removal", action="store_true", help="Enable effect line removal (Phase 2)"
    )
    parser.add_argument(
        "--panel-split", action="store_true", help="Enable multi-panel splitting (Phase 2)"
    )

    # 複数キャラクター選択基準オプション
    parser.add_argument(
        "--multi-character-criteria",
        choices=[
            "balanced",
            "size_priority",
            "fullbody_priority",
            "central_priority",
            "confidence_priority",
        ],
        default="balanced",
        help="Character selection criteria for multiple characters (default: balanced)",
    )

    # Phase 3: 適応学習モード（281評価データに基づく最適手法選択）
    parser.add_argument(
        "--adaptive-learning",
        action="store_true",
        help="Enable adaptive learning mode based on 281 evaluation records (Phase 3)",
    )

    args = parser.parse_args()

    # Extract common arguments (Phase 3対応版)
    extract_args = {
        "enhance_contrast": args.enhance_contrast,
        "filter_text": args.filter_text,
        "save_mask": args.save_mask,
        "save_transparent": args.save_transparent,
        "min_yolo_score": args.min_yolo_score,
        "verbose": args.verbose,
        "difficult_pose": args.difficult_pose,
        "low_threshold": args.low_threshold,
        "auto_retry": args.auto_retry,
        "high_quality": args.high_quality,
        "manga_mode": args.manga_mode,
        "effect_removal": args.effect_removal,
        "panel_split": args.panel_split,
        "multi_character_criteria": args.multi_character_criteria,
        "adaptive_learning": args.adaptive_learning,
    }

    # 複雑ポーズモード用の設定調整
    if args.low_threshold:
        extract_args["min_yolo_score"] = 0.02
        print("🔧 低閾値モード: YOLO閾値を0.02に設定")

    if args.high_quality:
        print("🔧 高品質モード: SAM高密度処理を有効化")

    # Phase 2: 漫画前処理モードの設定
    if args.manga_mode or args.effect_removal or args.panel_split:
        print("🎨 Phase 2: 漫画前処理モード有効")
        if args.effect_removal:
            print("   📝 エフェクト線除去: 有効")
        if args.panel_split:
            print("   📊 マルチコマ分割: 有効")

    # Phase 3: 適応学習モード
    if args.adaptive_learning:
        print("🧠 Phase 3: 適応学習モード有効")
        print("   📊 281評価データに基づく最適手法自動選択")
        print("   🎯 境界問題自動検出・対応")
        print("   ⚙️  パラメータ最適化")

    if args.batch:
        # Batch processing
        output_dir = args.output or f"{args.input}_character_output"
        result = batch_extract_characters(args.input, output_dir, **extract_args)
    else:
        # Single file processing
        result = extract_character_from_path(args.input, args.output, **extract_args)

    # Exit with appropriate code
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
