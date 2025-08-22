"""Test system requirements and dependencies."""
import pytest
import pkg_resources
import torch
import cv2
import numpy as np
from PIL import Image

def test_python_version():
    """Test Python version compatibility."""
    import sys
    assert sys.version_info >= (3, 8)

def test_core_dependencies():
    """Test core library versions."""
    deps = {
        'torch': '1.7.0',
        'opencv-python': '4.5.0',
        'numpy': '1.19.0',
        'Pillow': '8.0.0'
    }
    
    for package, min_version in deps.items():
        version = pkg_resources.get_distribution(package).version
        assert pkg_resources.parse_version(version) >= pkg_resources.parse_version(min_version)

def test_cuda_availability():
    """Test CUDA availability."""
    assert torch.cuda.is_available(), "CUDA not available"

def test_gpu_memory():
    """Test GPU memory availability."""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        min_memory = 8 * (1024 ** 3)  # 8GB
        assert gpu_memory >= min_memory

def test_image_processing():
    """Test basic image processing capabilities."""
    # Create test image
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    # Test OpenCV
    cv2.rectangle(img, (0, 0), (100, 100), (255, 255, 255), -1)
    assert img[50, 50].all() == 255
    
    # Test PIL
    pil_img = Image.fromarray(img)
    assert pil_img.size == (1920, 1080)