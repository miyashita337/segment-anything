"""Refactored character extraction command using Command Pattern.

P1-021: Modular architecture implementation with backward compatibility.
"""
import click
from typing import Optional

# Import new modular components
from .base_command import ExtractionConfig
from .batch_processor import BatchProcessor
from .single_processor import SingleProcessor


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
@click.option('--enable-dashboard', is_flag=True, help='P1-B002: Enable realtime quality dashboard')
@click.option('--dashboard-port', type=int, default=8080, help='P1-B002: Dashboard server port')
@click.option('--enable-backup', is_flag=True, help='PH2-007: Enable automatic backup of results')
@click.option('--backup-retention-days', type=int, default=7, help='PH2-007: Backup retention period in days')
@click.option('--enable-quality-monitoring', is_flag=True, default=True, help='P1-B001: Enable integrated quality monitoring')
@click.option('--quality-threshold', type=float, default=0.7, help='P1-B001: Quality threshold for success detection')
def extract_character_modular(
    input_path: str,
    output_path: str,
    batch: bool = False,
    verbose: bool = False,
    no_notify: bool = False,
    no_images: bool = False,
    max_files: Optional[int] = None,
    resume: bool = False,
    sam_optimization_profile: str = 'p1_020_optimized',
    enable_dashboard: bool = False,
    dashboard_port: int = 8080,
    enable_backup: bool = False,
    backup_retention_days: int = 7,
    enable_quality_monitoring: bool = True,
    quality_threshold: float = 0.7
) -> None:
    """P1-021: Modular character extraction using Command Pattern.
    
    This refactored version uses the new modular architecture while maintaining
    full backward compatibility with the original CLI interface.
    """
    # Create configuration object
    config = ExtractionConfig(
        input_path=input_path,
        output_path=output_path,
        batch=batch,
        verbose=verbose,
        no_notify=no_notify,
        no_images=no_images,
        max_files=max_files,
        resume=resume,
        sam_optimization_profile=sam_optimization_profile,
        enable_dashboard=enable_dashboard,
        dashboard_port=dashboard_port,
        enable_backup=enable_backup,
        backup_retention_days=backup_retention_days,
        enable_quality_monitoring=enable_quality_monitoring,
        quality_threshold=quality_threshold
    )
    
    # Select appropriate processor based on batch mode
    if batch:
        processor = BatchProcessor(config)
        if verbose:
            click.echo(f"🚀 P1-021 Modular Batch Processing: {input_path} → {output_path}")
    else:
        processor = SingleProcessor(config)
        if verbose:
            click.echo(f"🎯 P1-021 Modular Single Processing: {input_path} → {output_path}")
    
    # Execute processing
    try:
        result = processor.execute()
        
        if result.get("success", False):
            if verbose:
                if batch:
                    success_rate = result.get("success_rate", 0)
                    total = result.get("total_images", 0)
                    successful = result.get("successful", 0)
                    click.echo(f"✅ Batch completed: {successful}/{total} ({success_rate:.1f}%)")
                else:
                    processing_time = result.get("processing_time", 0)
                    click.echo(f"✅ Extraction completed in {processing_time:.2f}s")
        else:
            error_msg = result.get("error", "Unknown error")
            click.echo(f"❌ Extraction failed: {error_msg}")
            exit(1)
            
    except Exception as e:
        click.echo(f"❌ P1-021 Processing failed: {e}")
        exit(1)


# For backward compatibility, provide the original function name as an alias
def extract_character_legacy(*args, **kwargs):
    """Legacy compatibility wrapper."""
    # Import and call the original function for full backward compatibility
    from .extract_character import extract_character as original_extract_character
    return original_extract_character(*args, **kwargs)


# Main entry point - can switch between modular and legacy
def extract_character(*args, **kwargs):
    """Main entry point with architecture switching capability.
    
    P1-021: Uses modular architecture by default, but can fallback to legacy.
    """
    # Environment variable to control architecture
    import os
    use_modular = os.getenv('P1_021_USE_MODULAR_ARCHITECTURE', 'true').lower() == 'true'
    
    if use_modular:
        return extract_character_modular(*args, **kwargs)
    else:
        return extract_character_legacy(*args, **kwargs)


if __name__ == "__main__":
    # CLI entry point
    extract_character_modular()