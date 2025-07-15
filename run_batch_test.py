#!/usr/bin/env python3
"""
Test script to run batch extraction directly
"""
import sys
import os

# Set working directory and path
os.chdir('/mnt/c/AItools/segment-anything')
sys.path.insert(0, '/mnt/c/AItools/segment-anything')

# Import the main function
from batch_extract import main

# Override sys.argv to simulate command line arguments
sys.argv = [
    'batch_extract.py',
    '/mnt/c/AItools/lora/train/yadokugaeru/org/kaname04',
    '/mnt/c/AItools/lora/train/yadokugaeru/clipped_boundingbox/kaname04'
]

if __name__ == "__main__":
    main()