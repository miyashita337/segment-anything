"""Batch processing functionality for character extraction.

Handles batch operations, resume functionality, and progress tracking.
"""
import time
from datetime import datetime
from features.common.stable_batch_processor import StableBatchProcessor
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_command import BaseExtractionCommand, ExtractionConfig

try:
    from features.common.notification.global_pushover import (
    notify_success,
    notify_error,
    notify_process_complete
)
    PUSHOVER_AVAILABLE = True
    NOTIFIER_AVAILABLE = True
except ImportError:
    PUSHOVER_AVAILABLE = False
    NOTIFIER_AVAILABLE = False

try:
    from tools.status_update_hook import update_extraction_status
    SHEETS_AVAILABLE = True
except ImportError:
    SHEETS_AVAILABLE = False


class BatchProcessor(BaseExtractionCommand):
    """Handles batch processing of multiple images."""
    
    def __init__(self, config: ExtractionConfig):
        super().__init__(config)
        self.successful_extractions = []
        self.failed_images = []
        self.batch_start_time = None
        
    def execute(self) -> Dict[str, Any]:
        """Execute batch processing.
        
        Returns:
            Dict with processing results
        """
        if not self.validate_config():
            return {"success": False, "error": "Configuration validation failed"}
        
        input_dir = Path(self.config.input_path)
        output_dir = Path(self.config.output_path)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Get image files
        image_files = self._get_image_files(input_dir)
        
        # Apply max_files limit
        if self.config.max_files and self.config.max_files > 0:
            original_count = len(image_files)
            if self.config.max_files < len(image_files):
                image_files = image_files[:self.config.max_files]
                if self.config.verbose:
                    self.logger.info(f"Batch size limited: {original_count} → {self.config.max_files}")
        
        # Initialize batch processor
        checkpoint_dir = output_dir / ".checkpoint"
        stable_processor = StableBatchProcessor(
            checkpoint_dir=str(checkpoint_dir),
            micro_batch_size=3
        )
        
        # Start batch processing
        self.batch_start_time = time.time()
        
        # Update Google Sheets
        if SHEETS_AVAILABLE:
            try:
                dataset_name = input_dir.name
                update_extraction_status("P1-021", "start", 
                                        dataset_name=dataset_name, 
                                        total_images=len(image_files))
                if self.config.verbose:
                    self.logger.info("Google Sheets: Batch started")
            except Exception as e:
                self.logger.warning(f"Google Sheets update failed: {e}")
        
        # Process images
        results = self._process_batch(image_files, output_dir, stable_processor)
        
        # Send notification
        if not self.config.no_notify and NOTIFIER_AVAILABLE:
            self._send_completion_notification(results)
        
        return results
    
    def _get_image_files(self, input_dir: Path) -> List[Path]:
        """Get list of image files to process."""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_dir.glob(ext))
        return sorted(image_files)
    
    def _process_batch(self, image_files: List[Path], output_dir: Path, 
                      stable_processor: StableBatchProcessor) -> Dict[str, Any]:
        """Process batch of images."""
        total_images = len(image_files)
        processed = 0
        
        for image_file in image_files:
            try:
                # Import here to avoid circular imports
                from .single_processor import SingleProcessor

                # Create config for single processing
                single_config = ExtractionConfig(
                    input_path=str(image_file),
                    output_path=str(output_dir / f"{image_file.stem}_extracted.png"),
                    verbose=self.config.verbose,
                    sam_optimization_profile=self.config.sam_optimization_profile,
                    enable_quality_monitoring=self.config.enable_quality_monitoring,
                    quality_threshold=self.config.quality_threshold
                )
                
                # Process single image
                single_processor = SingleProcessor(single_config)
                result = single_processor.execute()
                
                if result.get("success", False):
                    self.successful_extractions.append(str(image_file))
                else:
                    self.failed_images.append(str(image_file))
                
                processed += 1
                
                if self.config.verbose:
                    progress = (processed / total_images) * 100
                    self.logger.info(f"Progress: {processed}/{total_images} ({progress:.1f}%)")
                
            except Exception as e:
                self.logger.error(f"Failed to process {image_file}: {e}")
                self.failed_images.append(str(image_file))
                processed += 1
        
        # Calculate final statistics
        processing_time = time.time() - self.batch_start_time
        success_rate = len(self.successful_extractions) / total_images * 100 if total_images > 0 else 0
        
        return {
            "success": True,
            "total_images": total_images,
            "successful": len(self.successful_extractions),
            "failed": len(self.failed_images),
            "success_rate": success_rate,
            "processing_time": processing_time,
            "successful_files": self.successful_extractions,
            "failed_files": self.failed_images
        }
    
    def _send_completion_notification(self, results: Dict[str, Any]):
        """Send completion notification."""
        try:
            # 統一通知システムを使用（インスタンス化不要）
            message = f"""バッチ処理完了 - P1-021
成功: {results['successful']}/{results['total_images']} ({results['success_rate']:.1f}%)
処理時間: {results['processing_time']:.1f}秒"""
            
            # Pushover通知送信
            if NOTIFIER_AVAILABLE:
                notify_process_complete(
                    title="キャラクター抽出完了",
                    successful=results['successful'],
                    total=results['total_images'],
                    failed=results['total_images'] - results['successful'],
                    duration=results['processing_time']
                )
        except Exception as e:
            self.logger.warning(f"Notification failed: {e}")