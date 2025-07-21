"""Character extraction command implementation.

Provides CLI interface for extracting anime characters from manga images.
"""
from pathlib import Path
from typing import Optional

import click
import numpy as np
from PIL import Image

from features.common.hooks.start import get_performance_monitor, get_sam_model, get_yolo_model
from features.processing.postprocessing.postprocessing import calculate_mask_quality_metrics
from features.processing.preprocessing.preprocessing import preprocess_image_pipeline
from features.common.types import ImageType, MaskType

@click.command()
@click.argument('input_path')
@click.option('-o', '--output-path', required=True, help='Output path for extracted character')
@click.option('--batch', is_flag=True, help='Process a directory of images')
@click.option('--verbose', is_flag=True, help='Enable verbose output')
def extract_character(
    input_path: str,
    output_path: str,
    batch: bool = False,
    verbose: bool = False
) -> None:
    """Extract anime character from manga image.

    Args:
        input_path: Path to input image or directory
        output_path: Path to save extracted character
        batch: Process directory of images if True
        verbose: Enable detailed logging if True
    """
    sam_model = get_sam_model()
    yolo_model = get_yolo_model()
    perf_monitor = get_performance_monitor()

    if batch:
        input_dir = Path(input_path)
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)

        for img_path in input_dir.glob('*.jpg'):
            process_single_image(
                img_path,
                output_dir / f'{img_path.stem}_extracted.png',
                sam_model,
                yolo_model,
                perf_monitor,
                verbose
            )
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
        image = Image.open(input_path)
        image = preprocess_image_pipeline(image)

        with perf_monitor.measure('inference'):
            mask = generate_character_mask(image, sam_model, yolo_model)

        if mask is not None:
            quality_metrics = calculate_mask_quality_metrics(mask)
            if quality_metrics['quality_score'] > 0.8:
                save_extracted_character(image, mask, output_path)
                if verbose:
                    click.echo(f'Successfully processed {input_path}')
                return mask

        if verbose:
            click.echo(f'Failed to extract character from {input_path}')
        return None

    except Exception as e:
        if verbose:
            click.echo(f'Error processing {input_path}: {str(e)}')
        return None

if __name__ == '__main__':
    extract_character()