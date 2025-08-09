"""Character extraction command implementation.

Provides CLI interface for extracting anime characters from manga images.
"""
import numpy as np
import cv2
import random
import torch

import click
import logging
import time
from features.common.hooks.start import (
    get_performance_monitor,
    get_sam_model,
    get_yolo_model,
    initialize_models,
)
from features.common.memory_optimizer import BatchMemoryManager, optimize_for_large_dataset
from features.common.retry_handler import image_retry_handler
from features.common.stable_batch_processor import StableBatchProcessor
from features.common.custom_types import ImageType, MaskType
from features.processing.sam_optimization_config import SAMOptimizationConfig, create_optimized_sam_generator
from features.evaluation.utils.face_detection import filter_non_character_masks
from features.evaluation.utils.mask_quality_validator import validate_and_improve_mask
from features.evaluation.utils.non_character_filter import apply_non_character_filter
from features.processing.postprocessing.auto_mask_correction import create_auto_mask_corrector
from features.processing.postprocessing.postprocessing import calculate_mask_quality_metrics
from features.processing.preprocessing.boundary_enhancer import BoundaryEnhancer
from features.processing.preprocessing.preprocessing import preprocess_image_pipeline
from pathlib import Path
from PIL import Image
from typing import Any, Optional, Tuple

# 統一通知システム（global_pushover使用）
try:
    from features.common.notification.global_pushover import (
        notify_success,
        notify_error,
        notify_process_complete,
        notify_warning
    )
    PUSHOVER_AVAILABLE = True
except ImportError:
    PUSHOVER_AVAILABLE = False
    print("Warning: Pushover notification not available.")

# Google Sheets自動更新
try:
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent.parent / "tools"))
    from status_update_hook import create_hook, update_extraction_status
    GOOGLE_SHEETS_HOOK_AVAILABLE = True
except ImportError:
    GOOGLE_SHEETS_HOOK_AVAILABLE = False

logger = logging.getLogger(__name__)


@click.command()
@click.argument('input_path')
@click.option('-o', '--output-path', required=True, help='Output path for extracted character')
@click.option('--batch', is_flag=True, help='Process a directory of images')
@click.option('--verbose', is_flag=True, help='Enable verbose output')
@click.option('--no-notify', is_flag=True, help='Disable Pushover notification')
@click.option('--no-images', is_flag=True, help='Disable success images in notification')
@click.option('--max-files', type=int, help='P1-018: Maximum number of files to process in batch mode')
@click.option('--resume', is_flag=True, help='P1-019: Resume from checkpoint if available')
@click.option('--sam-optimization-profile', 
              type=click.Choice(['original', 'p1_020_optimized', 'p1_020_balanced', 'p1_020_aggressive']),
              default='p1_020_optimized',
              help='P1-020: SAM optimization profile for 93% speed improvement')
@click.option('--enable-advanced-pipeline', is_flag=True, 
              help='Enable 3-stage improvement pipeline (experimental, use with caution)')
@click.option('--mode', 
              type=click.Choice(['standard', 'reproduce-auto']),
              default='standard',
              help='Processing mode: standard (normal) or reproduce-auto (workflow compatibility)')
def extract_character(
    input_path: str,
    output_path: str,
    batch: bool = False,
    verbose: bool = False,
    mode: str = 'standard',
    no_notify: bool = False,
    no_images: bool = False,
    max_files: Optional[int] = None,
    resume: bool = False,
    sam_optimization_profile: str = 'p1_020_optimized',
    enable_advanced_pipeline: bool = False
) -> None:
    """Extract anime character from manga image.

    Args:
        input_path: Path to input image or directory
        output_path: Path to save extracted character
        batch: Process directory of images if True
        verbose: Enable detailed logging if True
    """
    # 🚀 性能最適化: 決定論的実行を無効化（QCC-011性能テスト用）
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = False  # 性能最適化有効（5-10倍高速化）
        torch.backends.cudnn.benchmark = True       # GPU最適化有効
    
    # Initialize models if not already initialized
    if get_sam_model() is None or get_yolo_model() is None:
        if verbose:
            click.echo("Initializing models...")
        initialize_models()
        
    sam_model = get_sam_model()
    yolo_model = get_yolo_model()
    perf_monitor = get_performance_monitor()

    # reproduce-autoモード: ワークフロー互換性のための特別処理
    if mode == 'reproduce-auto':
        if verbose:
            click.echo("🔄 reproduce-auto モード実行中...")
        
        # バッチ処理を強制有効化
        batch = True
        
        # 入力・出力ディレクトリ設定の調整
        input_dir = Path(input_path)
        output_dir = Path(output_path)
        
        # ワークフロー互換性: 失敗・成功ディレクトリ作成
        failed_dir = output_dir.parent / "failed"
        success_dir = output_dir.parent / "success"
        failed_dir.mkdir(exist_ok=True)
        success_dir.mkdir(exist_ok=True)
        
        if verbose:
            click.echo(f"📂 入力: {input_dir}")
            click.echo(f"📂 出力: {output_dir}")
            click.echo(f"📂 失敗: {failed_dir}")
            click.echo(f"📂 成功: {success_dir}")

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
        
        # P1-018: max_files制限適用
        original_count = len(image_files)
        if max_files and max_files > 0:
            if max_files < len(image_files):
                image_files = image_files[:max_files]
                if verbose:
                    click.echo(f"🔢 P1-018バッチサイズ制御: {original_count}枚 → {max_files}枚に制限")
        
        total_images = len(image_files)
        
        if verbose:
            click.echo(f"🚀 バッチ処理開始: {total_images}枚の画像を処理します...")

        # P1-019: StableBatchProcessor初期化
        checkpoint_dir = output_dir / ".checkpoint"
        stable_processor = StableBatchProcessor(
            checkpoint_dir=str(checkpoint_dir),
            micro_batch_size=3  # P1-019: マイクロバッチサイズ
        )
        
        if verbose:
            click.echo("🔄 P1-019安定バッチ処理システム初期化完了")

        # Google Sheets: 抽出開始状態更新
        if GOOGLE_SHEETS_HOOK_AVAILABLE:
            try:
                dataset_name = input_dir.name
                update_extraction_status("PH2-002", "start", 
                                        dataset_name=dataset_name, 
                                        total_images=total_images)
                if verbose:
                    click.echo("📊 Google Sheets: 抽出開始状態更新完了")
            except Exception as e:
                logger.warning(f"Google Sheets更新エラー: {e}")

        # 単一ファイル処理関数の定義
        def process_single_file(file_path_str: str) -> Tuple[bool, str]:
            """単一ファイル処理（P1-019用）"""
            img_path = Path(file_path_str)
            output_path_single = output_dir / f'extracted_{img_path.stem}.jpg'
            
            try:
                if verbose:
                    click.echo(f"📝 処理中: {img_path.name}")
                
                # 直接処理（メモリ管理問題を回避）
                mask = process_single_image(
                    img_path,
                    output_path_single,
                    sam_model,
                    yolo_model,
                    perf_monitor,
                    verbose,
                    sam_optimization_profile
                )
                
                if mask is not None and output_path_single.exists():
                    # 成功記録（グローバルリストは関数内からアクセス制限があるため戻り値で処理）
                    return True, f"成功: {img_path.name}"
                else:
                    return False, f"抽出失敗: {img_path.name}"
                    
            except Exception as e:
                return False, f"エラー: {img_path.name} - {str(e)}"

        # P1-019: StableBatchProcessorでバッチ処理実行
        file_paths = [str(f) for f in image_files]
        batch_result = stable_processor.process_with_checkpoint(
            files=file_paths,
            process_function=process_single_file,
            output_dir=str(output_dir),
            resume=resume
        )
        
        if verbose:
            click.echo(f"🎯 P1-019バッチ処理完了: {batch_result.get('message', 'Unknown')}")
            stats = batch_result.get('stats', {})
            if stats:
                click.echo(f"📊 統計: 成功 {stats.get('success_count', 0)}件, "
                          f"エラー {stats.get('error_count', 0)}件, "
                          f"リトライ {stats.get('retry_count', 0)}回")

        # バッチ処理結果の統計処理
        batch_processing_time = time.time() - batch_start_time
        
        # StableBatchProcessor結果から成功・失敗を判定
        batch_stats = batch_result.get('stats', {})
        success_count = batch_stats.get('success_count', 0)
        error_count = batch_stats.get('error_count', 0)
        total_processed = success_count + error_count
        success_rate = (success_count / total_processed * 100) if total_processed > 0 else 0
        
        # reproduce-autoモード: 失敗・成功ディレクトリに画像をコピー
        if mode == 'reproduce-auto':
            import shutil
            
            if verbose:
                click.echo("📂 reproduce-autoモード: 結果分類処理中...")
            
            try:
                # 抽出済み画像ファイルを確認
                extracted_files = list(output_dir.glob("extracted_*.jpg"))
                extracted_stems = {f.stem.replace('extracted_', '') for f in extracted_files}
                
                for image_file in image_files:
                    original_path = Path(image_file)
                    stem = original_path.stem
                    
                    if stem in extracted_stems:
                        # 成功: 成功ディレクトリに元画像をコピー
                        success_dst = success_dir / original_path.name
                        if not success_dst.exists():
                            shutil.copy2(original_path, success_dst)
                    else:
                        # 失敗: 失敗ディレクトリに元画像をコピー
                        failed_dst = failed_dir / original_path.name
                        if not failed_dst.exists():
                            shutil.copy2(original_path, failed_dst)
                
                if verbose:
                    success_files = len(list(success_dir.glob("*.jpg")))
                    failed_files = len(list(failed_dir.glob("*.jpg")))
                    click.echo(f"📂 分類完了: 成功 {success_files}枚, 失敗 {failed_files}枚")
                    
            except Exception as e:
                click.echo(f"⚠️ reproduce-autoモード 分類処理エラー: {e}")
        
        # 下位互換のため元の変数も設定
        successful_extractions = ['success'] * success_count  # プレースホルダー
        failed_images = ['failed'] * error_count  # プレースホルダー
        
        # 結果表示
        click.echo(f"\n🎯 バッチ処理完了!")
        click.echo(f"📊 結果: {success_count}/{total_images}枚成功 ({success_rate:.1f}%)")
        click.echo(f"⏱️ 処理時間: {batch_processing_time:.1f}秒")
        
        # メモリ統計表示
        if verbose:
            try:
                from features.common.memory_optimizer import BatchMemoryManager
                temp_memory_manager = BatchMemoryManager()
                memory_stats = temp_memory_manager.get_memory_stats()
                current_memory = memory_stats['current_memory']
                click.echo(f"💾 メモリ統計: RAM {current_memory['ram_mb']:.1f}MB")
                if 'gpu_allocated_mb' in current_memory:
                    click.echo(f"    GPU {current_memory['gpu_allocated_mb']:.1f}MB使用")
                opt_stats = memory_stats['optimization_stats']
                click.echo(f"    最適化実行: GC {opt_stats['gc_calls']}回, "
                          f"キャッシュクリア {opt_stats['cache_clears']}回")
            except Exception as e:
                click.echo(f"💾 メモリ統計取得エラー: {e}")
        
        # Google Sheets: 抽出完了状態更新
        if GOOGLE_SHEETS_HOOK_AVAILABLE:
            try:
                update_extraction_status("PH2-002", "complete", 
                                        dataset_name=dataset_name, 
                                        total_images=total_images,
                                        success_count=success_count)
                if verbose:
                    click.echo("📊 Google Sheets: 抽出完了状態更新完了")
            except Exception as e:
                logger.warning(f"Google Sheets更新エラー: {e}")
        
        # Pushover通知送信（統一システム使用）
        if not no_notify and PUSHOVER_AVAILABLE and success_count > 0:
            try:
                # 成功率計算
                success_rate = (success_count / total_images * 100) if total_images > 0 else 0
                
                # 通知送信
                notify_process_complete(
                    title=f"キャラクター抽出完了: {output_dir.name}",
                    successful=success_count,
                    total=total_images,
                    duration=batch_processing_time
                )
                click.echo("📱 Pushover通知送信完了")
                    
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
            verbose,
            sam_optimization_profile,
            enable_advanced_pipeline
        )

@optimize_for_large_dataset
@image_retry_handler.retry
def process_single_image(
    input_path: Path,
    output_path: Path,
    sam_model: Any,
    yolo_model: Any,
    perf_monitor: Any,
    verbose: bool = False,
    sam_optimization_profile: str = 'p1_020_optimized',
    enable_advanced_pipeline: bool = False
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

        # QC成功版対応: 境界強調前処理を無効化
        enhanced_rgb = processed_rgb
        enhanced_bgr = processed_bgr
        
        if verbose:
            click.echo("🎯 QC成功版対応: 境界強調処理無効化")

        # ユーザー要望: fullbody_priorityを使用
        quality_method = 'fullbody_priority'
        
        if perf_monitor and hasattr(perf_monitor, 'measure'):
            with perf_monitor.measure('inference'):
                mask = generate_character_mask(enhanced_bgr, sam_model, yolo_model, quality_method, sam_optimization_profile)
        else:
            mask = generate_character_mask(enhanced_bgr, sam_model, yolo_model, quality_method, sam_optimization_profile)

        if mask is not None:
            # 📊 品質監視システム + 条件付き3段階処理
            quality_score = 0.0
            final_mask = mask  # デフォルトは元マスク使用
            
            try:
                from features.processing.postprocessing.postprocessing import (
                    calculate_mask_quality_metrics,
                )
                
                # 品質スコア計算（処理には影響させない）
                metrics = calculate_mask_quality_metrics(mask)
                quality_score = (
                    metrics.get('compactness', 0.0) * 0.4 +
                    metrics.get('fill_ratio', 0.0) * 0.3 +
                    metrics.get('coverage_ratio', 0.0) * 0.3
                )
                
                # 🚀 条件付き3段階処理（実験的機能）
                if enable_advanced_pipeline:
                    if verbose:
                        click.echo('🎆 高度パイプライン有効化（実験的）')
                    
                    try:
                        from features.extraction.integrated_precision_pipeline import (
                            IntegratedPrecisionPipeline,
                        )
                        
                        precision_pipeline = IntegratedPrecisionPipeline()
                        
                        # 3段階改善パイプライン実行（目標値を緩い設定）
                        pipeline_result = precision_pipeline.process_with_integrated_pipeline(
                            image=enhanced_bgr,
                            initial_mask=mask,
                            yolo_model=yolo_model,
                            sam_model=sam_model,
                            target_quality=0.25  # 50%から25%に緩和
                        )
                        
                        if pipeline_result.success and pipeline_result.final_mask is not None:
                            final_mask = pipeline_result.final_mask
                            quality_score = pipeline_result.final_quality  # 更新された品質スコア
                            
                            if verbose:
                                improvement_pct = pipeline_result.improvement_ratio * 100
                                click.echo(f'🚀 3段階改善完了: {pipeline_result.initial_quality:.3f} → {pipeline_result.final_quality:.3f} '
                                         f'({improvement_pct:+.1f}%, リトライ {pipeline_result.retry_count}回)')
                        else:
                            if verbose:
                                click.echo(f'⚠️ 3段階改善未達（元マスク使用）: {pipeline_result.final_quality:.3f}')
                    
                    except Exception as pipeline_e:
                        if verbose:
                            click.echo(f'⚠️ 3段階パイプラインエラー（QC成功版にフォールバック）: {pipeline_e}')
                        # エラー時は元マスクを使用
                
                # 品質スコアに基づく情報表示（処理制御なし）
                if quality_score < 0.05:  # 5%未満で警告
                    if verbose:
                        status = '🚀 改善済み' if enable_advanced_pipeline else '⚠️ 極端に低い'
                        click.echo(f'⚠️ 品質警告: スコア {quality_score:.3f} ({status})')
                elif quality_score < 0.10:  # 10%未満で統計フラグ
                    if verbose:
                        click.echo(f'📊 品質監視: スコア {quality_score:.3f} （要統計監視）')
                elif quality_score < 0.15:  # 15%未満で改善候補
                    if verbose:
                        click.echo(f'🔧 品質分析: スコア {quality_score:.3f} （改善検討候補）')
                elif verbose:
                    click.echo(f'✅ 品質確認: スコア {quality_score:.3f}')
                
                # 統計データ蓄積（将来の改善基盤）
                try:
                    import json
                    from datetime import datetime
                    from pathlib import Path
                    
                    stats_file = Path(output_path).parent / "quality_statistics.jsonl"
                    with open(stats_file, 'a', encoding='utf-8') as f:
                        stats_entry = {
                            'timestamp': datetime.now().isoformat(),
                            'image_path': str(input_path),
                            'quality_score': quality_score,
                            'metrics': metrics,
                            'advanced_pipeline_used': enable_advanced_pipeline,
                            'processing_success': True
                        }
                        f.write(json.dumps(stats_entry, ensure_ascii=False) + '\n')
                except Exception as stats_e:
                    if verbose:
                        click.echo(f'📊 統計記録エラー（処理継続）: {stats_e}')
                
            except Exception as quality_e:
                if verbose:
                    click.echo(f'📊 品質計算エラー（処理継続）: {quality_e}')
                # 品質計算失敗でも処理は継続
            
            # 全画像を通す方針: 品質に関係なく必ず保存実行
            try:
                save_extracted_character(enhanced_bgr, final_mask, output_path)
                if verbose:
                    pipeline_status = '🚀 高度' if enable_advanced_pipeline else '🎯 QC成功'
                    click.echo(f'✅ 抽出完了: {input_path.name} ({pipeline_status}版, 品質: {quality_score:.3f})')
                return final_mask
            except Exception as e:
                if verbose:
                    click.echo(f'❌ 保存失敗: {e}')
                return None

        if verbose:
            click.echo(f'Failed to extract character from {input_path}')
        return None

    except Exception as e:
        if verbose:
            click.echo(f'Error processing {input_path}: {str(e)}')
        return None

@image_retry_handler.retry
def generate_character_mask(image: ImageType, sam_model: Any, yolo_model: Any, quality_method: str = 'balanced', sam_optimization_profile: str = 'p1_020_optimized') -> Optional[MaskType]:
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
            
        # Hybrid方式: YOLOでperson検出してSAMにbboxプロンプトを渡す
        hybrid_masks = []
        try:
            # YOLOで人物検出
            persons = yolo_model.detect_persons(image_array)
            if persons:
                print(f"🎯 Hybrid方式: {len(persons)}人検出 → SAM bbox prompt")
                
                # 最大面積のpersonを選択（複数検出時）
                if len(persons) > 1:
                    persons.sort(key=lambda x: x['area'], reverse=True)
                    print(f"   最大面積person選択: {persons[0]['area']}px")
                
                # 最大のbboxをSAMプロンプトに使用
                best_person = persons[0]
                bbox = best_person['bbox']  # [x, y, w, h]
                
                # SAM hybrid生成
                hybrid_masks = sam_model.generate_masks_with_bbox_prompt(image_array, bbox)
                
                if hybrid_masks:
                    print(f"✅ Hybrid成功: {len(hybrid_masks)}マスク生成")
                    all_masks = hybrid_masks
                else:
                    print("⚠️ Hybrid失敗 → Grid方式にフォールバック")
                    all_masks = sam_model.generate_masks(image_array)
            else:
                print("⚠️ YOLO検出なし → Grid方式を使用")
                all_masks = sam_model.generate_masks(image_array)
                
        except Exception as e:
            print(f"⚠️ Hybridエラー → Grid方式にフォールバック: {e}")
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
            
        # QC成功版対応: 直接SAMスコアを使用（YOLOスコアリング無効化）
        # SAMマスクに直接スコアを付与
        scored_masks = []
        for i, mask in enumerate(character_masks):
            mask_with_score = mask.copy()
            # SAMのstability_scoreをyolo_scoreとして使用
            stability_score = mask.get('stability_score', 0.0)
            mask_with_score.update({
                'yolo_confidence': stability_score,
                'yolo_score': stability_score,
                'combined_score': stability_score
            })
            scored_masks.append(mask_with_score)
        
        print(f"🎯 QC成功版対応: 直接SAMスコア使用 - {len(scored_masks)} masks")
        
        # QC成功版対応: フィルタリングシステムを無効化
        final_masks = scored_masks
        print(f"🎯 QC成功版対応: フィルタリング無効化 - {len(final_masks)} masks直接選択")
        
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

def _select_best_mask_qc_method(masks: list) -> Optional[dict]:
    """最大面積のマスクを選択（ユーザー要望対応）
    
    複数キャラクター検出時は最大面積のキャラクターを抽出する仕様。
    信頼度ではなく面積ベースでの選択に修正。
    
    Args:
        masks: List of mask candidates
        
    Returns:
        Best mask (largest area) or None
    """
    if not masks:
        return None
    
    # Extract areas from all masks
    areas = []
    for mask_data in masks:
        area = mask_data.get('area', 0)
        areas.append(area)
    
    if not areas:
        print("❌ マスクの面積情報が見つかりません")
        return None
    
    # 最大面積のマスクを選択
    best_index = np.argmax(areas)
    best_mask = masks[best_index]
    best_area = areas[best_index]
    
    # 信頼度も参考として表示
    confidence = best_mask.get('yolo_confidence', best_mask.get('yolo_score', 0.0))
    
    print(f"🎯 最大面積選択: 面積 {best_area}px のマスクを選択")
    print(f"   選択: マスク{best_index + 1}/{len(masks)} (信頼度: {confidence:.3f})")
    
    # 他の候補の情報も表示（デバッグ用）
    if len(masks) > 1:
        print(f"   他候補: ", end="")
        for i, area in enumerate(areas):
            if i != best_index:
                print(f"[{i+1}]{area}px ", end="")
        print()
    
    return best_mask

def _select_best_mask_with_fallback(masks: list, image_shape: tuple, primary_method: str = 'balanced') -> Optional[dict]:
    """QC成功版マスク選択（複雑フォールバックシステム削除）
    
    P1-B004教訓により複雑なフォールバックシステムを削除。
    QC成功版の単純手法が100%成功のため、それを直接使用。
    
    Args:
        masks: List of mask candidates
        image_shape: Image dimensions (unused in QC method)
        primary_method: Method name (unused in QC method)
        
    Returns:
        Best mask or None
    """
    if not masks:
        return None
    
    print(f"🎯 QC成功版マスク選択開始（複雑フォールバック削除済み）")
    
    # QC成功版アルゴリズムを直接使用
    best_mask = _select_best_mask_qc_method(masks)
    
    if best_mask:
        # 基本的な品質チェック（面積のみ）
        mask_area_ratio = best_mask.get('area', 0) / (image_shape[0] * image_shape[1])
        
        # 極端に小さい・大きいマスクのみ除外（緩い設定）
        min_area_ratio = 0.001  # 0.1%以上（QC成功版に合わせて緩和）
        max_area_ratio = 0.8    # 80%以下
        
        if min_area_ratio <= mask_area_ratio <= max_area_ratio:
            print(f"✅ QC成功版選択完了: 面積比 {mask_area_ratio:.3f}")
            return best_mask
        else:
            print(f"⚠️ 極端な面積のため除外: {mask_area_ratio:.3f} (範囲: {min_area_ratio}-{max_area_ratio})")
            return None
    else:
        print("❌ QC成功版でも選択できませんでした")
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
        main_path = output_path.with_suffix('.png')
        cv2.imwrite(str(main_path), image)
        print(f"💾 Character extracted: {main_path} (size: {image.shape[1]}x{image.shape[0]})")
        
        return True
        
    except Exception as e:
        print(f"❌ Save error: {e}")
        return False


def save_extracted_character(image: ImageType, mask: MaskType, output_path: Path) -> None:
    """
    QC成功版保存方式: シンプルな白背景マスク適用（100%成功実績）
    
    P1-B004教訓: 複雑な境界強化・検証・黒背景処理は品質劣化の原因。
    QC成功版の単純な白背景処理が100%成功を実現。
    
    Args:
        image: Input image (BGR format)
        mask: Character mask (boolean or uint8)
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
        
        # QC成功版のシンプル処理: 白背景+マスク適用
        # Convert BGR to RGB for consistent processing
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        
        # Convert mask to boolean (QC成功版と同じ)
        best_mask = mask_array > 0
        
        # QC成功版: 白背景+マスク適用
        masked_image = np.ones_like(image_rgb) * 255  # 白背景
        masked_image[best_mask] = image_rgb[best_mask]
        
        # QC成功版: シンプルクロップ
        y_indices, x_indices = np.where(best_mask)
        if len(y_indices) > 0 and len(x_indices) > 0:
            x_min, x_max = x_indices.min(), x_indices.max()
            y_min, y_max = y_indices.min(), y_indices.max()
            cropped_image = masked_image[y_min:y_max+1, x_min:x_max+1]
        else:
            cropped_image = masked_image
        
        # QC成功版: 単純保存
        output_image = Image.fromarray(cropped_image.astype(np.uint8))
        output_image.save(str(output_path))
        
        print(f"✅ QC成功版保存完了: {output_path.name} (サイズ: {cropped_image.shape[1]}x{cropped_image.shape[0]})")
        
    except Exception as e:
        print(f"❌ QC成功版保存エラー: {e}")
        import traceback
        traceback.print_exc()


def generate_character_mask_qc_style(image_bgr: np.ndarray, 
                                    image_rgb: np.ndarray,
                                    sam_model: Any, 
                                    yolo_model: Any,
                                    verbose: bool = False) -> Optional[np.ndarray]:
    """
    QC成功版の完全再現: YOLO検出→最大面積選択→SAM処理→最高スコア選択
    
    Args:
        image_bgr: Input image (BGR format)
        image_rgb: Input image (RGB format) 
        sam_model: SAM model wrapper
        yolo_model: YOLO model wrapper
        verbose: Enable detailed logging
        
    Returns:
        Character mask or None
    """
    try:
        # QC成功版ステップ1: YOLO検出（confidence=0.07相当）
        persons = yolo_model.detect_persons(image_rgb)
        
        if verbose:
            print(f"🎯 QC成功版YOLO検出: {len(persons)} persons detected")
        
        if len(persons) == 0:
            if verbose:
                print("⚠️ 検出なし: 画像全体を使用")
            # 画像全体を使用
            h, w = image_rgb.shape[:2]
            box = np.array([0, 0, w, h])
        else:
            # QC成功版ステップ2: 最大面積ボックス選択
            areas = [person['area'] for person in persons]
            max_idx = np.argmax(areas)
            max_person = persons[max_idx]
            
            # bbox_xyxy format [x1, y1, x2, y2]
            box = np.array(max_person['bbox_xyxy'])
            
            if verbose:
                print(f"🎯 QC成功版最大面積選択: {max_person['area']}px, confidence={max_person['confidence']:.3f}")
        
        # QC成功版ステップ3: SAM全マスク生成（現在の実装に合わせる）
        all_masks = sam_model.generate_masks(image_rgb)
        
        if verbose:
            print(f"🎯 QC成功版SAM処理: {len(all_masks)} masks generated")
        
        if not all_masks:
            return None
        
        # QC成功版ステップ4: YOLO検出領域に重なるマスクをフィルタリング
        filtered_masks = []
        box_x1, box_y1, box_x2, box_y2 = box
        
        for mask_data in all_masks:
            mask_bbox = mask_data['bbox']  # [x, y, w, h]
            mask_x1, mask_y1, mask_w, mask_h = mask_bbox
            mask_x2 = mask_x1 + mask_w
            mask_y2 = mask_y1 + mask_h
            
            # YOLO検出領域との重複チェック
            overlap_x1 = max(box_x1, mask_x1)
            overlap_y1 = max(box_y1, mask_y1)
            overlap_x2 = min(box_x2, mask_x2)
            overlap_y2 = min(box_y2, mask_y2)
            
            if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                mask_area = mask_w * mask_h
                
                # 50%以上重複するマスクを採用
                if overlap_area / mask_area >= 0.5:
                    filtered_masks.append(mask_data)
        
        if not filtered_masks:
            # フォールバック: 最大面積マスク
            filtered_masks = [max(all_masks, key=lambda x: x['area'])]
        
        # QC成功版ステップ5: 最高スコアマスク選択
        best_mask_data = max(filtered_masks, key=lambda x: x['stability_score'])
        best_score = best_mask_data['stability_score']
        
        if verbose:
            print(f"🎯 QC成功版最終選択: {len(filtered_masks)}件から選択, score={best_score:.3f}")
        
        return best_mask_data['segmentation']
        
    except Exception as e:
        print(f"❌ QC成功版処理エラー: {e}")
        return None


if __name__ == '__main__':
    extract_character()