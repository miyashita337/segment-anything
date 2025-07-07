#!/usr/bin/env python3
"""
Character Extraction Command
Main command for extracting characters from manga images using SAM + YOLO
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from hooks.start import get_sam_model, get_yolo_model, get_performance_monitor
from utils.difficult_pose import (
    DifficultPoseProcessor, 
    detect_difficult_pose, 
    get_difficult_pose_config,
    process_with_retry
)
from utils.preprocessing import preprocess_image_pipeline
from utils.postprocessing import (
    enhance_character_mask, 
    extract_character_from_image, 
    crop_to_content,
    save_character_result,
    calculate_mask_quality_metrics
)
from utils.text_detection import TextDetector

# Phase 4: 統合システムインポート
from utils.phase4_integration import Phase4IntegratedExtractor

# Phase 4.1: 選択的ハイブリッドシステムインポート
from utils.phase41_integrated_system import Phase41IntegratedSystem
from utils.multi_character_handler import SelectionCriteria


def extract_character_from_path(image_path: str,
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
                               # Phase 4: 新しいオプション
                               enable_phase4: bool = False,
                               enable_mask_inversion_detection: bool = False,
                               enable_adaptive_range: bool = False,
                               enable_quality_prediction: bool = False,
                               # Phase 4.1: 選択的ハイブリッドオプション
                               enable_phase41: bool = False,
                               multi_character_criteria: str = "balanced",
                               **kwargs) -> Dict[str, Any]:
    """
    画像パスからキャラクターを抽出 (Phase 4対応版)
    
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
        enable_phase4: Phase 4統合システムを有効化
        enable_mask_inversion_detection: マスク逆転検出・修正
        enable_adaptive_range: 適応的範囲調整
        enable_quality_prediction: 品質予測・フィードバック
        
    Returns:
        抽出結果の辞書
    """
    result = {
        'success': False,
        'input_path': image_path,
        'output_path': None,
        'processing_time': 0.0,
        'mask_quality': {},
        'error': None
    }
    
    start_time = time.time()
    
    # 自動リトライモードの場合は process_with_retry を使用
    if auto_retry:
        if verbose:
            print(f"🔄 自動リトライモードでキャラクター抽出開始: {image_path}")
        
        def extract_function(img_path, **config):
            # 元の処理ロジックを呼び出し（リトライ用）
            return extract_character_from_path(
                img_path, output_path, enhance_contrast, filter_text,
                save_mask, save_transparent, config.get('min_yolo_score', min_yolo_score),
                verbose=False,  # リトライ中は詳細出力を抑制
                difficult_pose=False, low_threshold=False, auto_retry=False,  # 無限ループ防止
                high_quality=config.get('enable_enhanced_processing', high_quality),
                manga_mode=config.get('enable_manga_preprocessing', manga_mode),
                effect_removal=config.get('enable_effect_removal', effect_removal),
                panel_split=config.get('enable_panel_split', panel_split),
                **{k: v for k, v in config.items() if k not in [
                    'min_yolo_score', 'enable_enhanced_processing', 'enable_manga_preprocessing',
                    'enable_effect_removal', 'enable_panel_split'
                ]}
            )
        
        return process_with_retry(image_path, extract_function, max_retries=4)
    
    try:
        # Get models
        sam_model = get_sam_model()
        yolo_model = get_yolo_model()
        performance_monitor = get_performance_monitor()
        
        if not sam_model or not yolo_model:
            raise RuntimeError("Models not initialized. Run start hook first.")
        
        # 複雑ポーズ判定と設定調整 (Phase 2対応版)
        if difficult_pose or low_threshold or manga_mode:
            processor = DifficultPoseProcessor()
            
            if difficult_pose:
                # 複雑ポーズモード: 自動判定による設定
                complexity_info = processor.detect_pose_complexity(image_path)
                recommended_config = processor.get_recommended_config(complexity_info)
                
                if verbose:
                    print(f"🔍 ポーズ複雑度: {complexity_info['complexity']} (スコア: {complexity_info['score']:.1f})")
                    print(f"🔧 推奨設定適用: {recommended_config['description']}")
                
                # 推奨設定を適用
                min_yolo_score = min(min_yolo_score, recommended_config['min_yolo_score'])
                if 'sam_points_per_side' in recommended_config:
                    # SAM設定をkwargsに追加
                    kwargs.update({
                        'sam_points_per_side': recommended_config['sam_points_per_side'],
                        'sam_pred_iou_thresh': recommended_config['sam_pred_iou_thresh'],
                        'sam_stability_score_thresh': recommended_config['sam_stability_score_thresh']
                    })
            
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
                    enable_panel_split=panel_split
                )
                
                # 処理済み画像を使用
                image_path = processed_image_path
        
        if high_quality:
            # 高品質SAM設定
            kwargs.update({
                'sam_points_per_side': kwargs.get('sam_points_per_side', 64),
                'sam_pred_iou_thresh': kwargs.get('sam_pred_iou_thresh', 0.88),
                'sam_stability_score_thresh': kwargs.get('sam_stability_score_thresh', 0.92)
            })
            if verbose:
                print(f"🔧 高品質モード: SAM密度 {kwargs.get('sam_points_per_side', 64)} ポイント/サイド")
        
        if verbose:
            print(f"🎯 キャラクター抽出開始: {image_path}")
            print(f"📊 YOLO閾値: {min_yolo_score}")
        
        # Step 1: Image preprocessing
        performance_monitor.start_stage("Image Preprocessing")
        bgr_image, rgb_image, scale = preprocess_image_pipeline(
            image_path, 
            enhance_contrast=enhance_contrast
        )
        
        if rgb_image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        performance_monitor.end_stage()
        
        # Phase 4.1: 選択的ハイブリッドシステムによる処理分岐
        if enable_phase41:
            if verbose:
                print("🌟 Phase 4.1選択的ハイブリッドシステム実行中...")
            
            # Phase 4.1統合システム初期化
            try:
                criteria = SelectionCriteria(multi_character_criteria)
            except ValueError:
                criteria = SelectionCriteria.BALANCED
                if verbose:
                    print(f"⚠️ 無効な選択基準: {multi_character_criteria}, デフォルト(balanced)を使用")
            
            phase41_system = Phase41IntegratedSystem(
                sam_model=sam_model,
                yolo_model=yolo_model,
                multi_character_criteria=criteria,
                enable_detailed_logging=verbose
            )
            
            # YOLO検出の実行
            yolo_detections = yolo_model.detect_persons(bgr_image)
            if not yolo_detections or len(yolo_detections) == 0:
                if verbose:
                    print(f"⚠️ YOLO検出なし (閾値: {min_yolo_score}), 低閾値で再試行...")
                # 低閾値で再試行
                yolo_detections = yolo_model.detect_persons(bgr_image, score_threshold=0.02)
                if not yolo_detections:
                    raise ValueError(f"YOLO検出結果なし")
            
            # YOLO結果フォーマット
            formatted_detections = []
            for det in yolo_detections:
                if det['score'] >= min_yolo_score:
                    formatted_detections.append({
                        "bbox": det['bbox'],
                        "confidence": det['score']
                    })
            
            if verbose:
                print(f"📊 YOLO検出: {len(formatted_detections)}個のキャラクター")
            
            # Phase 4.1統合処理
            phase41_result = phase41_system.extract_character(bgr_image, formatted_detections)
            
            if not phase41_result.success:
                raise RuntimeError(f"Phase 4.1処理失敗: {phase41_result.error_message}")
            
            # 結果の変換
            enhanced_mask = phase41_result.final_mask
            character_image = extract_character_from_image(
                bgr_image, 
                enhanced_mask,
                background_color=(0, 0, 0)
            )
            
            # Phase 4.1専用結果構築
            result = {
                'success': True,
                'image_path': image_path,
                'output_path': output_path,
                'mask': enhanced_mask,
                'character_image': character_image,
                'bbox': phase41_result.final_bbox,
                'phase41_result': phase41_result,
                'processing_engine': phase41_result.selected_engine.value,
                'quality_score': phase41_result.quality_score,
                'yolo_detections': phase41_result.yolo_detections,
                'final_character_count': phase41_result.final_character_count,
                'complexity_level': phase41_result.complexity_analysis.level.value if phase41_result.complexity_analysis else "unknown",
                'multi_character_selection': phase41_result.multi_character_analysis.success if phase41_result.multi_character_analysis else False,
                'adjustments_made': phase41_result.adjustments_made,
                'processing_time': phase41_result.processing_time,
                'warnings': phase41_result.warnings or []
            }
            
            if verbose:
                print(f"✅ Phase 4.1処理完了:")
                print(f"   選択エンジン: {result['processing_engine']}")
                print(f"   品質スコア: {result['quality_score']:.3f}")
                print(f"   複雑度: {result['complexity_level']}")
                if result['multi_character_selection']:
                    print(f"   複数キャラクター選択: 成功")
                if result['adjustments_made']:
                    print(f"   適用された調整: {', '.join(result['adjustments_made'])}")
                if result['warnings']:
                    print(f"   警告: {'; '.join(result['warnings'])}")
        
        # Phase 4: 統合システムによる処理分岐
        elif enable_phase4 or enable_mask_inversion_detection or enable_adaptive_range or enable_quality_prediction:
            if verbose:
                print("🚀 Phase 4統合システム実行中...")
            
            # Phase 4統合システム初期化
            phase4_extractor = Phase4IntegratedExtractor(
                enable_mask_inversion_detection=enable_mask_inversion_detection or enable_phase4,
                enable_adaptive_range=enable_adaptive_range or enable_phase4,
                enable_quality_prediction=enable_quality_prediction or enable_phase4,
                max_iterations=3
            )
            
            # YOLO検出の実行（Phase 4で必要）
            yolo_detections = yolo_model.detect_persons(bgr_image)
            if not yolo_detections or len(yolo_detections) == 0:
                raise ValueError(f"No YOLO detections found (min score: {min_yolo_score})")
            
            # 最高スコアの検出を使用
            best_detection = max(yolo_detections, key=lambda x: x.get('confidence', 0))
            yolo_bbox = (
                int(best_detection['bbox'][0]),
                int(best_detection['bbox'][1]), 
                int(best_detection['bbox'][2]),
                int(best_detection['bbox'][3])
            )
            yolo_confidence = best_detection.get('confidence', 0.0)
            
            if verbose:
                print(f"🎯 YOLO検出: bbox={yolo_bbox}, confidence={yolo_confidence:.3f}")
            
            # Phase 4パラメータ設定
            phase4_params = {
                'min_yolo_score': min_yolo_score,
                'high_quality': high_quality,
                'manga_mode': manga_mode,
                'effect_removal': effect_removal,
                'expansion_factor': kwargs.get('expansion_factor', 1.1),
                'difficult_pose': difficult_pose,
                'low_threshold': low_threshold,
                'auto_retry': auto_retry
            }
            
            # Phase 4統合処理実行
            phase4_result = phase4_extractor.extract_with_phase4_enhancements(
                rgb_image, yolo_bbox, yolo_confidence, sam_model, phase4_params
            )
            
            if phase4_result.success and phase4_result.final_mask is not None:
                if verbose:
                    print(f"✅ Phase 4処理成功")
                    print(f"   品質スコア: {phase4_result.quality_metrics.confidence_score:.3f}")
                    print(f"   実行調整: {phase4_result.adjustments_made}")
                    print(f"   処理時間: {phase4_result.processing_stats['processing_time']:.3f}秒")
                
                # Phase 4結果を使用（適切な形式に変換）
                raw_mask = phase4_result.final_mask
                if raw_mask.dtype == bool:
                    enhanced_mask = raw_mask.astype(np.uint8) * 255
                else:
                    enhanced_mask = raw_mask
                result['phase4_stats'] = phase4_result.processing_stats
                result['phase4_adjustments'] = phase4_result.adjustments_made
                result['mask_quality'] = {
                    'coverage_ratio': phase4_result.quality_metrics.confidence_score,
                    'compactness': phase4_result.quality_metrics.edge_consistency,
                    'phase4_enabled': True
                }
                
                # Phase 4処理後は通常の後処理へジャンプ
                performance_monitor.start_stage("Character Extraction")
                character_image = extract_character_from_image(
                    bgr_image, 
                    enhanced_mask,
                    background_color=(0, 0, 0)
                )
                
                # Crop to content
                cropped_character, cropped_mask, crop_bbox = crop_to_content(
                    character_image,
                    enhanced_mask,
                    padding=10
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
                    save_transparent=save_transparent
                )
                
                if not save_success:
                    raise RuntimeError("Failed to save results")
                
                result['output_path'] = str(output_path)
                performance_monitor.end_stage()
                
                # Success
                result['success'] = True
                result['processing_time'] = time.time() - start_time
                
                if verbose:
                    print(f"✅ Phase 4キャラクター抽出完了: {result['processing_time']:.2f}秒")
                    print(f"   出力: {result['output_path']}")
                
                return result
            else:
                if verbose:
                    print(f"⚠️ Phase 4処理失敗、従来処理にフォールバック")
        
        # Step 2: SAM mask generation (従来処理)
        performance_monitor.start_stage("SAM Mask Generation")
        
        # 高品質/複雑ポーズモードでSAM設定を動的に適用
        if any(key.startswith('sam_') for key in kwargs.keys()) or high_quality:
            # SAMGeneratorを一時的に再構築
            try:
                from segment_anything import SamAutomaticMaskGenerator
                
                sam_params = {
                    'model': sam_model.sam,
                    'points_per_side': kwargs.get('sam_points_per_side', 32),
                    'pred_iou_thresh': kwargs.get('sam_pred_iou_thresh', 0.8),
                    'stability_score_thresh': kwargs.get('sam_stability_score_thresh', 0.85),
                    'crop_n_layers': 1,
                    'crop_n_points_downscale_factor': 2,
                    'min_mask_region_area': 100,
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
            scored_masks, 
            bgr_image, 
            min_yolo_score=min_yolo_score
        )
        
        if best_mask is None:
            raise ValueError(f"No good character masks found (min YOLO score: {min_yolo_score})")
        
        if verbose:
            print(f"🎯 最適マスク選択: YOLO score={best_mask['yolo_score']:.3f}, "
                  f"combined score={best_mask['combined_score']:.3f}")
        
        performance_monitor.end_stage()
        
        # Step 4: Text filtering (optional)
        if filter_text:
            performance_monitor.start_stage("Text Filtering")
            text_detector = TextDetector(use_easyocr=True)
            
            text_density = text_detector.calculate_text_density_score(
                bgr_image, 
                best_mask['bbox']
            )
            
            if text_density > 0.5:
                if verbose:
                    print(f"⚠️ 高テキスト密度検出: {text_density:.3f} - 処理続行")
            
            # Add text density to result
            best_mask['text_density'] = text_density
            performance_monitor.end_stage()
        
        # Step 5: Mask refinement
        performance_monitor.start_stage("Mask Refinement")
        raw_mask = sam_model.mask_to_binary(best_mask)
        
        # 複雑ポーズ用の強化マスク処理を適用
        if difficult_pose or low_threshold or high_quality:
            if verbose:
                print("🔧 複雑ポーズ用マスク強化処理を適用")
            
            # DifficultPoseProcessorを使用した強化処理
            if 'processor' not in locals():
                processor = DifficultPoseProcessor()
            
            enhanced_mask = processor.enhance_mask_for_complex_pose(raw_mask, bgr_image)
        else:
            enhanced_mask = enhance_character_mask(
                raw_mask,
                remove_small_area=100,
                smooth_kernel=3,
                fill_holes=True
            )
        
        # Calculate mask quality metrics
        quality_metrics = calculate_mask_quality_metrics(enhanced_mask)
        result['mask_quality'] = quality_metrics
        
        if verbose:
            print(f"📐 マスク品質: coverage={quality_metrics['coverage_ratio']:.3f}, "
                  f"compactness={quality_metrics['compactness']:.3f}")
        
        performance_monitor.end_stage()
        
        # Step 6: Character extraction
        performance_monitor.start_stage("Character Extraction")
        character_image = extract_character_from_image(
            bgr_image, 
            enhanced_mask,
            background_color=(0, 0, 0)  # Black background
        )
        
        # Crop to content
        cropped_character, cropped_mask, crop_bbox = crop_to_content(
            character_image,
            enhanced_mask,
            padding=10
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
            save_transparent=save_transparent
        )
        
        if not save_success:
            raise RuntimeError("Failed to save results")
        
        result['output_path'] = str(output_path)
        performance_monitor.end_stage()
        
        # Success
        result['success'] = True
        result['processing_time'] = time.time() - start_time
        
        if verbose:
            print(f"✅ キャラクター抽出完了: {result['processing_time']:.2f}秒")
            print(f"   出力: {result['output_path']}")
        
        return result
        
    except Exception as e:
        result['error'] = str(e)
        result['processing_time'] = time.time() - start_time
        
        if verbose:
            print(f"❌ 抽出失敗: {e}")
        
        return result


def batch_extract_characters(input_dir: str,
                           output_dir: str,
                           **extract_kwargs) -> Dict[str, Any]:
    """
    ディレクトリ内の全画像に対してバッチ処理
    
    Args:
        input_dir: 入力ディレクトリ
        output_dir: 出力ディレクトリ
        **extract_kwargs: extract_character_from_path の引数
        
    Returns:
        バッチ処理結果
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        return {'success': False, 'error': f'Input directory not found: {input_dir}'}
    
    # 画像ファイルを取得
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(input_path.glob(f'*{ext}'))
        image_files.extend(input_path.glob(f'*{ext.upper()}'))
    
    if not image_files:
        return {'success': False, 'error': f'No image files found in {input_dir}'}
    
    # 出力ディレクトリ作成
    output_path.mkdir(parents=True, exist_ok=True)
    
    # バッチ処理実行
    results = []
    successful = 0
    
    print(f"🚀 バッチ処理開始: {len(image_files)} 画像")
    
    for i, image_file in enumerate(image_files, 1):
        print(f"\n📁 処理中 [{i}/{len(image_files)}]: {image_file.name}")
        
        # 出力パス生成
        output_file = output_path / image_file.stem
        
        # 抽出実行
        # verboseはバッチ処理では抑制
        batch_kwargs = extract_kwargs.copy()
        batch_kwargs['verbose'] = False
        result = extract_character_from_path(
            str(image_file),
            output_path=str(output_file),
            **batch_kwargs
        )
        
        result['filename'] = image_file.name
        results.append(result)
        
        if result['success']:
            successful += 1
            print(f"✅ 成功: {image_file.name}")
        else:
            print(f"❌ 失敗: {image_file.name} - {result['error']}")
    
    # 結果サマリ
    batch_result = {
        'success': True,
        'total_files': len(image_files),
        'successful': successful,
        'failed': len(image_files) - successful,
        'success_rate': successful / len(image_files),
        'results': results
    }
    
    print(f"\n📊 バッチ処理完了:")
    print(f"   成功: {successful}/{len(image_files)} ({batch_result['success_rate']:.1%})")
    
    # Pushover通知送信
    try:
        from utils.notification import send_batch_notification
        print("\n📱 通知送信中...")
        notification_sent = send_batch_notification(
            successful=successful,
            total=len(image_files),
            failed=len(image_files) - successful,
            total_time=batch_result['total_time']
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
    
    parser.add_argument('input', help='Input image path or directory')
    parser.add_argument('-o', '--output', help='Output path (auto-generated if not specified)')
    parser.add_argument('--batch', action='store_true', help='Batch processing mode')
    parser.add_argument('--enhance-contrast', action='store_true', help='Enhance image contrast')
    parser.add_argument('--filter-text', action='store_true', default=True, help='Filter text regions')
    parser.add_argument('--save-mask', action='store_true', default=False, help='Save mask files')
    parser.add_argument('--save-transparent', action='store_true', default=False, help='Save transparent background')
    parser.add_argument('--min-yolo-score', type=float, default=0.1, help='Minimum YOLO score threshold')
    parser.add_argument('--verbose', action='store_true', default=True, help='Verbose output')
    
    # 複雑ポーズ・ダイナミック構図対応オプション
    parser.add_argument('--difficult-pose', action='store_true', help='Enable difficult pose processing mode')
    parser.add_argument('--low-threshold', action='store_true', help='Use low threshold settings (YOLO score 0.02)')
    parser.add_argument('--auto-retry', action='store_true', help='Enable automatic retry with progressive settings')
    parser.add_argument('--high-quality', action='store_true', help='Enable high-quality SAM processing')
    
    # Phase 2: 漫画前処理オプション
    parser.add_argument('--manga-mode', action='store_true', help='Enable manga-specific preprocessing (Phase 2)')
    parser.add_argument('--effect-removal', action='store_true', help='Enable effect line removal (Phase 2)')
    parser.add_argument('--panel-split', action='store_true', help='Enable multi-panel splitting (Phase 2)')
    
    # Phase 4: 統合システムオプション
    parser.add_argument('--phase4', action='store_true', help='Enable Phase 4 integrated system (all enhancements)')
    parser.add_argument('--mask-inversion-detection', action='store_true', help='Enable mask inversion detection/correction')
    parser.add_argument('--adaptive-range', action='store_true', help='Enable adaptive extraction range adjustment')
    parser.add_argument('--quality-prediction', action='store_true', help='Enable quality prediction and feedback')
    
    # Phase 4.1: 選択的ハイブリッドシステム
    parser.add_argument('--phase41', action='store_true', help='Enable Phase 4.1 selective hybrid system (best of 0.0.3 and 0.0.4)')
    parser.add_argument('--multi-character-criteria', choices=['balanced', 'size_priority', 'fullbody_priority', 'central_priority', 'confidence_priority'],
                       default='balanced', help='Multi-character selection criteria for Phase 4.1')
    
    args = parser.parse_args()
    
    # Extract common arguments (Phase 4対応版)
    extract_args = {
        'enhance_contrast': args.enhance_contrast,
        'filter_text': args.filter_text,
        'save_mask': args.save_mask,
        'save_transparent': args.save_transparent,
        'min_yolo_score': args.min_yolo_score,
        'verbose': args.verbose,
        'difficult_pose': args.difficult_pose,
        'low_threshold': args.low_threshold,
        'auto_retry': args.auto_retry,
        'high_quality': args.high_quality,
        'manga_mode': args.manga_mode,
        'effect_removal': args.effect_removal,
        'panel_split': args.panel_split,
        # Phase 4オプション
        'enable_phase4': args.phase4,
        'enable_mask_inversion_detection': args.mask_inversion_detection,
        'enable_adaptive_range': args.adaptive_range,
        'enable_quality_prediction': args.quality_prediction,
        # Phase 4.1オプション
        'enable_phase41': args.phase41 if hasattr(args, 'phase41') else False,
        'multi_character_criteria': args.multi_character_criteria if hasattr(args, 'multi_character_criteria') else "balanced"
    }
    
    # 複雑ポーズモード用の設定調整
    if args.low_threshold:
        extract_args['min_yolo_score'] = 0.02
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
    
    # Phase 4.1: 選択的ハイブリッドシステムの設定
    if args.phase41:
        print("🌟 Phase 4.1: 選択的ハイブリッドシステム有効")
        print(f"   🎯 複数キャラクター選択基準: {args.multi_character_criteria}")
        print("   ✨ 0.0.3と0.0.4のいいとこどり処理")
    
    # Phase 4: 統合システムの設定
    elif args.phase4 or args.mask_inversion_detection or args.adaptive_range or args.quality_prediction:
        print("🚀 Phase 4: 統合システム有効")
        if args.phase4:
            print("   🔧 フル統合モード: 有効")
        if args.mask_inversion_detection:
            print("   🔄 マスク逆転検出: 有効")
        if args.adaptive_range:
            print("   📐 適応的範囲調整: 有効")
        if args.quality_prediction:
            print("   🎯 品質予測システム: 有効")
    
    if args.batch:
        # Batch processing
        output_dir = args.output or f"{args.input}_character_output"
        result = batch_extract_characters(args.input, output_dir, **extract_args)
    else:
        # Single file processing
        result = extract_character_from_path(args.input, args.output, **extract_args)
    
    # Exit with appropriate code
    sys.exit(0 if result['success'] else 1)


if __name__ == "__main__":
    main()