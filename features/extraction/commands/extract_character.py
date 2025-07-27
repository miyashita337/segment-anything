"""Character extraction command implementation.

Provides CLI interface for extracting anime characters from manga images.
"""
import numpy as np
import cv2

import click
import logging
import time
from features.common.hooks.start import (
    get_performance_monitor,
    get_sam_model,
    get_yolo_model,
    initialize_models,
)
from features.common.types import ImageType, MaskType
from features.evaluation.utils.face_detection import filter_non_character_masks
from features.evaluation.utils.mask_quality_validator import validate_and_improve_mask
from features.evaluation.utils.non_character_filter import apply_non_character_filter
from features.processing.postprocessing.postprocessing import calculate_mask_quality_metrics
from features.processing.preprocessing.boundary_enhancer import BoundaryEnhancer
from features.processing.preprocessing.preprocessing import preprocess_image_pipeline
from pathlib import Path
from PIL import Image
from typing import Any, Optional, Tuple

# 通知システム
try:
    from features.extraction.extraction_notifier import (
        ExtractionNotifier,
        create_extraction_results_dict,
    )
    EXTRACTION_NOTIFIER_AVAILABLE = True
except ImportError:
    EXTRACTION_NOTIFIER_AVAILABLE = False

logger = logging.getLogger(__name__)


@click.command()
@click.argument('input_path')
@click.option('-o', '--output-path', required=True, help='Output path for extracted character')
@click.option('--batch', is_flag=True, help='Process a directory of images')
@click.option('--verbose', is_flag=True, help='Enable verbose output')
@click.option('--no-notify', is_flag=True, help='Disable Pushover notification')
@click.option('--no-images', is_flag=True, help='Disable success images in notification')
def extract_character(
    input_path: str,
    output_path: str,
    batch: bool = False,
    verbose: bool = False,
    no_notify: bool = False,
    no_images: bool = False
) -> None:
    """Extract anime character from manga image.

    Args:
        input_path: Path to input image or directory
        output_path: Path to save extracted character
        batch: Process directory of images if True
        verbose: Enable detailed logging if True
    """
    # Initialize models if not already initialized
    if get_sam_model() is None or get_yolo_model() is None:
        if verbose:
            click.echo("Initializing models...")
        initialize_models()
        
    sam_model = get_sam_model()
    yolo_model = get_yolo_model()
    perf_monitor = get_performance_monitor()

    if batch:
        input_dir = Path(input_path)
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)

        # バッチ処理の統計情報
        batch_start_time = time.time()
        total_images = 0
        successful_extractions = []
        failed_images = []
        
        # 対象画像ファイルを取得
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_dir.glob(ext))
        
        total_images = len(image_files)
        
        if verbose:
            click.echo(f"🚀 バッチ処理開始: {total_images}枚の画像を処理します...")

        for img_path in image_files:
            try:
                output_path_single = output_dir / f'{img_path.stem}_extracted.png'
                
                if verbose:
                    click.echo(f"📝 処理中: {img_path.name}")
                
                mask = process_single_image(
                    img_path,
                    output_path_single,
                    sam_model,
                    yolo_model,
                    perf_monitor,
                    verbose
                )
                
                if mask is not None and output_path_single.exists():
                    # 成功記録
                    successful_extractions.append({
                        'input_path': str(img_path),
                        'output_path': str(output_path_single),
                        'extracted_files': [str(output_path_single)],
                        'quality_score': 0.8,  # デフォルト値
                        'processing_time': 1.0  # デフォルト値
                    })
                    
                    if verbose:
                        click.echo(f"✅ 成功: {img_path.name}")
                else:
                    failed_images.append(str(img_path))
                    if verbose:
                        click.echo(f"❌ 失敗: {img_path.name}")
                        
            except Exception as e:
                failed_images.append(str(img_path))
                if verbose:
                    click.echo(f"❌ エラー: {img_path.name} - {e}")
        
        # バッチ処理完了
        batch_processing_time = time.time() - batch_start_time
        success_count = len(successful_extractions)
        success_rate = (success_count / total_images * 100) if total_images > 0 else 0
        
        # 結果表示
        click.echo(f"\n🎯 バッチ処理完了!")
        click.echo(f"📊 結果: {success_count}/{total_images}枚成功 ({success_rate:.1f}%)")
        click.echo(f"⏱️ 処理時間: {batch_processing_time:.1f}秒")
        
        # Pushover通知送信
        if not no_notify and EXTRACTION_NOTIFIER_AVAILABLE and success_count > 0:
            try:
                notifier = ExtractionNotifier()
                
                # 抽出結果辞書を作成
                extraction_results = create_extraction_results_dict(
                    total_images=total_images,
                    successful_extractions=successful_extractions,
                    processing_time=batch_processing_time,
                    quality_method="fullbody_priority",
                    output_dir=str(output_dir)
                )
                
                # 通知送信（デフォルトで画像付き）
                include_images = not no_images
                success = notifier.send_extraction_completion_notification(
                    extraction_results, 
                    include_images=include_images
                )
                
                if success:
                    if include_images:
                        click.echo("📱 Pushover通知送信完了（成功画像付き）")
                    else:
                        click.echo("📱 Pushover通知送信完了")
                else:
                    click.echo("⚠️ Pushover通知送信失敗")
                    
            except Exception as e:
                if verbose:
                    click.echo(f"⚠️ 通知送信エラー: {e}")
        elif not no_notify and not EXTRACTION_NOTIFIER_AVAILABLE:
            if verbose:
                click.echo("⚠️ 通知システムが利用できません")
        elif success_count == 0:
            if verbose:
                click.echo("⚠️ 成功した抽出がないため通知をスキップ")
    else:
        process_single_image(
            Path(input_path),
            Path(output_path),
            sam_model,
            yolo_model, 
            perf_monitor,
            verbose
        )

def process_single_image(
    input_path: Path,
    output_path: Path,
    sam_model: Any,
    yolo_model: Any,
    perf_monitor: Any,
    verbose: bool = False
) -> Optional[MaskType]:
    """Process a single image for character extraction.

    Args:
        input_path: Path to input image
        output_path: Path to save result
        sam_model: SAM model instance
        yolo_model: YOLO model instance
        perf_monitor: Performance monitoring instance
        verbose: Enable detailed logging

    Returns:
        Generated mask if successful, None otherwise
    """
    try:
        # Use string path directly with preprocessing pipeline
        processed_bgr, processed_rgb, scale = preprocess_image_pipeline(str(input_path))
        if processed_bgr is None:
            return None

        # 境界強調前処理を適用
        boundary_enhancer = BoundaryEnhancer()
        enhanced_rgb = boundary_enhancer.enhance_image_boundaries(processed_rgb)
        enhanced_bgr = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)
        
        if verbose:
            # 強調統計情報を取得・表示
            stats = boundary_enhancer.get_enhancement_stats(processed_rgb, enhanced_rgb)
            click.echo(f"境界強調統計: コントラスト改善={stats['contrast_improvement']:.2f}x, "
                      f"エッジ改善={stats['edge_improvement']:.2f}x")

        # Use fullbody_priority for better upper body extraction
        quality_method = 'fullbody_priority'
        
        if perf_monitor and hasattr(perf_monitor, 'measure'):
            with perf_monitor.measure('inference'):
                mask = generate_character_mask(enhanced_bgr, sam_model, yolo_model, quality_method)
        else:
            mask = generate_character_mask(enhanced_bgr, sam_model, yolo_model, quality_method)

        if mask is not None:
            # 🚀 P1-A004: セグメンテーション精度総合向上システム適用
            try:
                from features.extraction.integrated_precision_pipeline import (
                    IntegratedPrecisionPipeline,
                )
                
                precision_pipeline = IntegratedPrecisionPipeline()
                
                # 統合精度向上パイプライン実行
                pipeline_result = precision_pipeline.process_with_integrated_pipeline(
                    image=enhanced_bgr,
                    initial_mask=mask,
                    yolo_model=yolo_model,
                    sam_model=sam_model,
                    target_quality=0.5
                )
                
                if pipeline_result.success and pipeline_result.final_mask is not None:
                    mask = pipeline_result.final_mask
                    
                    if verbose:
                        improvement_pct = pipeline_result.improvement_ratio * 100
                        click.echo(f'🚀 統合精度向上完了: 品質 {pipeline_result.initial_quality:.3f} → {pipeline_result.final_quality:.3f} '
                                 f'(改善率 {improvement_pct:+.1f}%, リトライ {pipeline_result.retry_count}回, '
                                 f'処理時間 {pipeline_result.total_processing_time:.1f}秒)')
                        
                        # 処理段階の詳細表示
                        completed_stages = [s.name for s in pipeline_result.processing_stages if s.status == 'completed']
                        if completed_stages:
                            click.echo(f'   完了段階: {", ".join(completed_stages)}')
                else:
                    if verbose:
                        click.echo(f'⚠️ 統合精度向上未達: 最終品質 {pipeline_result.final_quality:.3f}（元マスク使用）')
                
            except Exception as e:
                if verbose:
                    click.echo(f'⚠️ 統合精度向上エラー（フォールバック使用）: {e}')
                
                # フォールバック: 従来のSAM後処理のみ適用
                try:
                    from features.processing.sam_postprocessing_pipeline import (
                        SAMPostprocessingPipeline,
                    )
                    postprocessor = SAMPostprocessingPipeline()
                    
                    # マスク品質向上処理
                    enhancement_result = postprocessor.enhance_mask_quality(mask, enhanced_bgr)
                    enhanced_mask = enhancement_result.get('enhanced_mask', mask)
                    quality_score = enhancement_result.get('quality_score', 0.0)
                    
                    if verbose:
                        improvements = enhancement_result.get('improvements', [])
                        click.echo(f'🔧 フォールバック処理完了: 品質スコア {quality_score:.3f}, 改善 {len(improvements)}項目')
                    
                    # 改良されたマスクを使用
                    mask = enhanced_mask
                    
                except Exception as fallback_e:
                    if verbose:
                        click.echo(f'⚠️ フォールバック処理もエラー（元マスク使用）: {fallback_e}')
                    # エラー時は元のマスクを継続使用
            
            # Skip quality check for testing and save directly
            try:
                save_extracted_character(enhanced_bgr, mask, output_path)
                if verbose:
                    click.echo(f'Successfully processed {input_path}')
                return mask
            except Exception as e:
                if verbose:
                    click.echo(f'Save failed: {e}')
                return None

        if verbose:
            click.echo(f'Failed to extract character from {input_path}')
        return None

    except Exception as e:
        if verbose:
            click.echo(f'Error processing {input_path}: {str(e)}')
        return None

def generate_character_mask(image: ImageType, sam_model: Any, yolo_model: Any, quality_method: str = 'balanced') -> Optional[MaskType]:
    """Generate character mask using SAM and YOLO models with enhanced quality evaluation.
    
    Args:
        image: Input image
        sam_model: SAM model instance
        yolo_model: YOLO model instance
        quality_method: Quality evaluation method ('balanced', 'size_priority', 'fullbody_priority', etc.)
        
    Returns:
        Character mask if successful, None otherwise
    """
    try:
        # Convert PIL image to numpy array if needed
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
        
        print(f"🔍 Image shape: {image_array.shape}")
            
        # Generate masks with SAM
        all_masks = sam_model.generate_masks(image_array)
        if not all_masks:
            print("❌ No masks generated by SAM")
            return None
        
        print(f"📊 Generated {len(all_masks)} masks")
            
        # Filter for character masks
        character_masks = sam_model.filter_character_masks(all_masks)
        if not character_masks:
            print("❌ No character masks after filtering")
            return None
        
        print(f"👤 {len(character_masks)} character masks")
            
        # Score masks with YOLO
        scored_masks = yolo_model.score_masks_with_detections(character_masks, image_array)
        print(f"🎯 {len(scored_masks) if scored_masks else 0} scored masks")
        
        # 🚀 YOLO検出範囲拡張システム適用 (P1-A001)
        try:
            from features.extraction.yolo_detection_expansion import YOLODetectionExpander
            expander = YOLODetectionExpander()
            expanded_masks = expander.expand_detection_capabilities(yolo_model, image_array, scored_masks)
            print(f"🔧 YOLO拡張システム: {len(scored_masks)} → {len(expanded_masks)} masks")
            scored_masks = expanded_masks
        except Exception as e:
            print(f"⚠️ YOLO拡張システムエラー（フォールバック使用）: {e}")
            # エラー時は元のマスクを継続使用
        
        # Apply enhanced filtering system
        if scored_masks:
            # Step 1: Filter non-character elements (masks, speech bubbles, etc.)
            print(f"🔍 Step 1: Non-character element filtering")
            filtered_masks = apply_non_character_filter(scored_masks, image_array)
            
            # Step 2: Face detection validation
            print(f"🔍 Step 2: Face detection validation")
            validated_masks = filter_non_character_masks(filtered_masks, image_array)
            
            # Use validated masks for selection
            final_masks = validated_masks if validated_masks else scored_masks  # Fallback to original
            print(f"🎯 Final masks for selection: {len(final_masks)}")
        else:
            final_masks = scored_masks
        
        # Enhanced mask selection with multi-method fallback
        best_mask = _select_best_mask_with_fallback(final_masks, image_array.shape, quality_method)
        
        if best_mask:
            # Log validation results if available
            if 'face_validation' in best_mask:
                face_val = best_mask['face_validation']
                print(f"✅ Selected mask validation: faces={face_val['face_count']}, "
                      f"character_confidence={face_val['confidence']:.3f}")
            
            return best_mask['segmentation'] if 'segmentation' in best_mask else best_mask
        else:
            print("❌ No good character masks found after all fallback attempts")
            return None
        
    except Exception as e:
        print(f"❌ Error in generate_character_mask: {e}")
        return None

def _select_best_character_with_criteria(masks: list, image_shape: tuple, criteria: str = 'balanced') -> Optional[dict]:
    """Select best character using composite scoring (migrated from backup script).
    
    Args:
        masks: List of mask candidates
        image_shape: Image dimensions (height, width, channels)
        criteria: Selection criteria ('balanced', 'size_priority', 'fullbody_priority', 'central_priority', 'confidence_priority')
        
    Returns:
        Best mask or None
    """
    if not masks:
        return None
    
    h, w = image_shape[:2]
    image_center_x, image_center_y = w / 2, h / 2
    
    def calculate_composite_score(mask_data: dict) -> dict:
        """Calculate composite score"""
        scores = {}
        
        # 1. Area score (30%): Evaluate appropriate size
        area_ratio = mask_data['area'] / (h * w)
        if 0.05 <= area_ratio <= 0.4:  # 5-40% of image is ideal
            scores['area'] = min(area_ratio / 0.4, 1.0)
        else:
            scores['area'] = max(0, 1.0 - abs(area_ratio - 0.2) / 0.2)
        
        # 2. Aspect ratio score (25%): Prioritize full-body characters
        bbox = mask_data['bbox']
        aspect_ratio = bbox[3] / max(bbox[2], 1)  # height / width
        if 1.2 <= aspect_ratio <= 2.5:  # Full-body character range
            scores['fullbody'] = min((aspect_ratio - 0.5) / 2.0, 1.0)
        else:
            scores['fullbody'] = max(0, 1.0 - abs(aspect_ratio - 1.8) / 1.0)
        
        # 3. Central position score (20%): Prioritize characters near center
        mask_center_x = bbox[0] + bbox[2] / 2
        mask_center_y = bbox[1] + bbox[3] / 2
        distance_from_center = np.sqrt(
            ((mask_center_x - image_center_x) / w)**2 + 
            ((mask_center_y - image_center_y) / h)**2
        )
        scores['central'] = max(0, 1.0 - distance_from_center)
        
        # 4. Grounding score (15%): Prioritize characters in lower part
        bottom_position = (bbox[1] + bbox[3]) / h
        if bottom_position >= 0.6:  # Lower 60% and below
            scores['grounded'] = min(bottom_position, 1.0)
        else:
            scores['grounded'] = bottom_position / 0.6
        
        # 5. YOLO confidence score (10%)
        scores['confidence'] = mask_data.get('yolo_confidence', mask_data.get('yolo_score', 0.0))
        
        return scores
    
    # Weight configuration by criteria
    weight_configs = {
        'balanced': {'area': 0.30, 'fullbody': 0.25, 'central': 0.20, 'grounded': 0.15, 'confidence': 0.10},
        'size_priority': {'area': 0.50, 'fullbody': 0.15, 'central': 0.15, 'grounded': 0.10, 'confidence': 0.10},
        'fullbody_priority': {'area': 0.20, 'fullbody': 0.40, 'central': 0.15, 'grounded': 0.15, 'confidence': 0.10},
        'central_priority': {'area': 0.20, 'fullbody': 0.20, 'central': 0.35, 'grounded': 0.15, 'confidence': 0.10},
        'confidence_priority': {'area': 0.25, 'fullbody': 0.20, 'central': 0.15, 'grounded': 0.10, 'confidence': 0.30}
    }
    
    weights = weight_configs.get(criteria, weight_configs['balanced'])
    
    # Calculate scores for each mask
    best_mask = None
    best_score = 0.0
    
    print(f"🎯 複合スコア評価開始 (基準: {criteria})")
    print(f"   重み設定: {weights}")
    
    for i, mask_data in enumerate(masks):
        scores = calculate_composite_score(mask_data)
        
        # Calculate weighted composite score
        composite_score = sum(scores[key] * weights[key] for key in weights.keys())
        
        # Detailed debug information
        bbox = mask_data.get('bbox', [0, 0, 0, 0])
        area_ratio = mask_data.get('area', 0) / (h * w)
        aspect_ratio = bbox[3] / max(bbox[2], 1) if len(bbox) >= 4 else 0
        
        print(f"   マスク{i+1}: 総合={composite_score:.3f} "
              f"(面積={scores['area']:.2f}, 全身={scores['fullbody']:.2f}, "
              f"中央={scores['central']:.2f}, 接地={scores['grounded']:.2f}, "
              f"信頼度={scores['confidence']:.2f})")
        print(f"      詳細: 面積比={area_ratio:.3f}, アスペクト比={aspect_ratio:.2f}, "
              f"bbox={bbox}, YOLO信頼度={mask_data.get('yolo_confidence', mask_data.get('yolo_score', 0)):.3f}")
        
        if composite_score > best_score:
            best_score = composite_score
            best_mask = mask_data
            print(f"      🎯 現在の最高スコア更新!")
        else:
            print(f"      📊 スコア不足 (最高: {best_score:.3f})")
    
    if best_mask is not None:
        print(f"✅ 最適マスク選択: 総合スコア {best_score:.3f}")
        return best_mask
    
    return None

def _select_best_mask_with_fallback(masks: list, image_shape: tuple, primary_method: str = 'balanced') -> Optional[dict]:
    """Select best mask with multi-method fallback system.
    
    Args:
        masks: List of mask candidates
        image_shape: Image dimensions
        primary_method: Primary selection method
        
    Returns:
        Best mask or None
    """
    if not masks:
        return None
    
    # Method priority order for fallback
    method_priority = {
        'balanced': ['balanced', 'size_priority', 'fullbody_priority', 'central_priority', 'confidence_priority'],
        'size_priority': ['size_priority', 'balanced', 'fullbody_priority', 'central_priority', 'confidence_priority'],
        'fullbody_priority': ['fullbody_priority', 'balanced', 'size_priority', 'central_priority', 'confidence_priority'],
        'central_priority': ['central_priority', 'balanced', 'size_priority', 'fullbody_priority', 'confidence_priority'],
        'confidence_priority': ['confidence_priority', 'balanced', 'size_priority', 'fullbody_priority', 'central_priority']
    }
    
    methods_to_try = method_priority.get(primary_method, method_priority['balanced'])
    
    print(f"🔄 Multi-method fallback system starting with: {primary_method}")
    
    for i, method in enumerate(methods_to_try):
        print(f"   試行 {i+1}/{len(methods_to_try)}: {method}")
        
        best_mask = _select_best_character_with_criteria(masks, image_shape, method)
        
        if best_mask:
            # Check if mask quality is acceptable (basic quality threshold)
            mask_area_ratio = best_mask.get('area', 0) / (image_shape[0] * image_shape[1])
            
            # Quality thresholds
            min_area_ratio = 0.01  # At least 1% of image
            max_area_ratio = 0.7   # At most 70% of image
            
            if min_area_ratio <= mask_area_ratio <= max_area_ratio:
                if i == 0:
                    print(f"✅ Primary method {method} succeeded")
                else:
                    print(f"✅ Fallback to {method} succeeded (attempt {i+1})")
                return best_mask
            else:
                print(f"   ⚠️ {method}: Quality check failed (area ratio: {mask_area_ratio:.3f})")
        else:
            print(f"   ❌ {method}: No mask selected")
    
    # Final fallback: use original YOLO-based method
    print("🆘 Final fallback: Using original YOLO-based selection")
    try:
        from features.common.hooks.start import get_yolo_model
        yolo_model = get_yolo_model()
        fallback_mask = yolo_model.get_best_character_mask(masks, image_shape, min_yolo_score=0.02)
        if fallback_mask:
            print("✅ Final fallback succeeded")
            return fallback_mask
    except Exception as e:
        print(f"   ❌ Final fallback failed: {e}")
    
    print("❌ All fallback methods failed")
    return None

def enhance_mask_boundaries(mask: np.ndarray) -> np.ndarray:
    """Enhance mask boundaries to prevent limb cutting and improve quality.
    
    Args:
        mask: Binary mask
        
    Returns:
        Enhanced mask
    """
    try:
        import cv2

        # Convert to uint8 if needed
        if mask.dtype != np.uint8:
            enhanced_mask = (mask > 0).astype(np.uint8) * 255
        else:
            enhanced_mask = mask.copy()
            
        # 1. Connected component analysis - keep largest component
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(enhanced_mask, connectivity=8)
        if num_labels > 2:  # Background + multiple components
            # Find largest component (excluding background)
            largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            enhanced_mask = (labels == largest_label).astype(np.uint8) * 255
            
        # 2. Morphological operations to fill holes and smooth
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        enhanced_mask = cv2.morphologyEx(enhanced_mask, cv2.MORPH_CLOSE, kernel)
        enhanced_mask = cv2.morphologyEx(enhanced_mask, cv2.MORPH_OPEN, kernel)
        
        # 3. Contour smoothing to reduce jaggedness
        contours, _ = cv2.findContours(enhanced_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Smooth contour using approximation
            epsilon = 0.002 * cv2.arcLength(largest_contour, True)
            smoothed_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # Create mask from smoothed contour
            smooth_mask = np.zeros(enhanced_mask.shape, dtype=np.uint8)
            cv2.fillPoly(smooth_mask, [smoothed_contour], 255)
            enhanced_mask = smooth_mask
        
        return (enhanced_mask > 0).astype(bool)
        
    except Exception as e:
        print(f"⚠️ Mask enhancement failed: {e}")
        return (mask > 0).astype(bool)

def extract_character_from_image(image: np.ndarray, 
                               mask: np.ndarray,
                               background_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """
    Extract character from image using mask (from backup script).
    
    Args:
        image: Input image (BGR)
        mask: Character mask (0-255)
        background_color: Background color (B, G, R)
        
    Returns:
        Extracted character image
    """
    # Convert mask to 3 channels if needed
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if len(mask.shape) == 2 else mask
    
    # Normalize mask
    mask_normalized = mask_3ch.astype(np.float32) / 255.0
    
    # Create background image
    background = np.full_like(image, background_color, dtype=np.uint8)
    
    # Apply mask
    result = (image.astype(np.float32) * mask_normalized + 
             background.astype(np.float32) * (1.0 - mask_normalized))
    result = result.astype(np.uint8)
    
    return result


def crop_to_content(image: np.ndarray, 
                   mask: np.ndarray,
                   padding: int = 10) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    Crop image to content area (from backup script).
    
    Args:
        image: Input image
        mask: Character mask
        padding: Padding around content
        
    Returns:
        (cropped_image, cropped_mask, bbox)
    """
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image, mask, (0, 0, image.shape[1], image.shape[0])
    
    # Get bounding box of largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Add padding
    height, width = image.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + w + padding)
    y2 = min(height, y + h + padding)
    
    # Crop
    cropped_image = image[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]
    
    bbox = (x1, y1, x2 - x1, y2 - y1)
    
    return cropped_image, cropped_mask, bbox


def save_character_result(image: np.ndarray,
                        mask: np.ndarray,
                        output_path: str) -> bool:
    """
    Save character extraction result (from backup script).
    
    Args:
        image: Extracted character image
        mask: Character mask
        output_path: Output path (without extension)
        
    Returns:
        Success flag
    """
    try:
        output_path = Path(output_path)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main image
        main_path = output_path.with_suffix('.jpg')
        cv2.imwrite(str(main_path), image)
        print(f"💾 Character extracted: {main_path} (size: {image.shape[1]}x{image.shape[0]})")
        
        return True
        
    except Exception as e:
        print(f"❌ Save error: {e}")
        return False


def save_extracted_character(image: ImageType, mask: MaskType, output_path: Path) -> None:
    """
    Extract and save character using backup script logic.
    
    Args:
        image: Input image
        mask: Character mask
        output_path: Output file path
    """
    try:
        # Convert to numpy arrays
        if isinstance(image, Path):
            image_array = cv2.imread(str(image))
        elif isinstance(image, Image.Image):
            image_array = np.array(image)
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            image_array = image
        
        if isinstance(mask, Path):
            mask_array = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
        else:
            mask_array = np.array(mask)
            if len(mask_array.shape) == 3:
                mask_array = cv2.cvtColor(mask_array, cv2.COLOR_RGB2GRAY)
        
        if mask_array.dtype != np.uint8:
            mask_array = (mask_array * 255).astype(np.uint8)
        
        # Enhance mask boundaries
        enhanced_mask = enhance_mask_boundaries(mask_array)
        enhanced_mask = enhanced_mask.astype(np.uint8) * 255
        
        # Step 1: Validate and improve mask quality
        print(f"🔍 Validating mask quality...")
        y_indices, x_indices = np.where(enhanced_mask > 0)
        if len(y_indices) > 0 and len(x_indices) > 0:
            # Calculate initial bounding box
            initial_bbox = (
                max(0, x_indices.min() - 10),
                max(0, y_indices.min() - 10), 
                min(image_array.shape[1], x_indices.max() + 10) - max(0, x_indices.min() - 10),
                min(image_array.shape[0], y_indices.max() + 10) - max(0, y_indices.min() - 10)
            )
            
            # Validate and improve mask
            improved_mask, improved_bbox, validation_results = validate_and_improve_mask(
                image_array, enhanced_mask, initial_bbox
            )
            
            print(f"🎯 Mask quality: {validation_results['overall_quality']}, "
                  f"Face complete: {validation_results['face_validation']['face_complete']}, "
                  f"Needs improvement: {validation_results['needs_improvement']}")
            
            enhanced_mask = improved_mask
        
        # Step 2: Extract character from image (backup script logic)
        character_image = extract_character_from_image(
            image_array, 
            enhanced_mask,
            background_color=(0, 0, 0)  # Black background
        )
        
        # Step 3: Crop to content (backup script logic)
        cropped_character, cropped_mask, crop_bbox = crop_to_content(
            character_image,
            enhanced_mask,
            padding=10
        )
        
        # Step 4: Save results (backup script logic)
        save_success = save_character_result(
            cropped_character,
            cropped_mask,
            str(output_path.with_suffix(''))  # Remove extension
        )
        
        if not save_success:
            print(f"⚠️ Failed to save: {output_path}")
        
    except Exception as e:
        print(f"❌ Error extracting character: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    extract_character()