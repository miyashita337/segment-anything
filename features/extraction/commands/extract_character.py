import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np
from tqdm import tqdm

from core.automation.hooks.start import get_performance_monitor, get_sam_model, get_yolo_model
from features.processing.utils.postprocessing import calculate_mask_quality_metrics
from features.processing.utils.preprocessing import preprocess_image_pipeline

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
    multi_character_criteria: str = 'balanced',
    adaptive_learning: bool = False,
    **kwargs
) -> Dict[str, Any]:
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f'Failed to load image: {image_path}')
            
        image = preprocess_image_pipeline(image, enhance_contrast)
        
        yolo_model = get_yolo_model()
        sam_predictor = get_sam_model()
        
        yolo_detection = yolo_model(image, min_yolo_score)
        if not yolo_detection:
            raise ValueError('No characters detected by YOLO')
            
        if manga_mode:
            mask = _extract_with_solid_fill(image, yolo_detection, sam_predictor)
        else:
            mask = _extract_without_solid_fill(image, yolo_detection, sam_predictor)
            
        metrics = calculate_mask_quality_metrics(mask)
        
        if output_path:
            cv2.imwrite(output_path, mask)
            
        return {
            'success': True,
            'mask': mask,
            'metrics': metrics
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def _extract_with_solid_fill(image, yolo_detection, sam_predictor):
    mask = sam_predictor.generate(image, yolo_detection)
    coverage = _calculate_vertical_coverage(mask, yolo_detection)
    if coverage < 0.8:
        mask = _extract_without_solid_fill(image, yolo_detection, sam_predictor)
    return mask

def _extract_without_solid_fill(image, yolo_detection, sam_predictor):
    return sam_predictor.generate(image, yolo_detection, solid_fill=False)

def _calculate_vertical_coverage(mask, yolo_detection) -> float:
    mask_height = np.sum(mask.any(axis=1))
    bbox_height = yolo_detection['bbox'][3] - yolo_detection['bbox'][1]
    return mask_height / bbox_height

def _calculate_iou(mask, yolo_detection) -> float:
    bbox = yolo_detection['bbox']
    bbox_mask = np.zeros_like(mask)
    bbox_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
    intersection = np.logical_and(mask, bbox_mask).sum()
    union = np.logical_or(mask, bbox_mask).sum()
    return intersection / union

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input image path or directory')
    parser.add_argument('-o', '--output', help='Output path')
    parser.add_argument('--batch', action='store_true', help='Batch process directory')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if args.batch:
        input_dir = Path(args.input)
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)
        
        for img_path in tqdm(list(input_dir.glob('*.jpg'))):
            out_path = output_dir / img_path.name
            result = extract_character_from_path(str(img_path), str(out_path))
            if args.verbose:
                print(f'{img_path.name}: {"Success" if result["success"] else result["error"]}')
    else:
        result = extract_character_from_path(args.input, args.output)
        if args.verbose:
            print(f'Result: {"Success" if result["success"] else result["error"]}')

if __name__ == '__main__':
    main()