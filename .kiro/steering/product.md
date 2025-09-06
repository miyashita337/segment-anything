# Product Overview

## Purpose
This project is a specialized anime/manga character extraction system that combines Meta's Segment Anything Model (SAM) with YOLOv8 for automated character detection and extraction. The primary goal is to generate high-quality training datasets for LoRA (Low-Rank Adaptation) machine learning models.

## Core Functionality
- **Character Detection**: Uses YOLOv8 for initial character detection in manga/anime images
- **Precise Segmentation**: Leverages SAM for accurate character boundary extraction
- **Quality Evaluation**: Implements A/B evaluation system with statistical quality metrics (SCI, PLA, PLE)
- **Batch Processing**: Supports large-scale image processing with progress tracking
- **Dashboard Generation**: Creates web-based quality dashboards for result visualization

## Target Users
- LoRA training dataset creators
- Anime/manga content processors
- Computer vision researchers working with character extraction

## Key Features
- **Extraction Success Rate**: Currently achieving 80% success rate
- **Quality Metrics**: A/B evaluation rate of 80%, SCI value of 0.853
- **Automated Pipeline**: End-to-end processing from input images to extracted characters
- **Progress Tracking**: Google Sheets integration for real-time progress monitoring
- **Notification System**: Pushover integration for batch completion alerts

## Version Status
Current version: v0.9.24 (development)
- Development fork of Facebook's Segment Anything
- Specialized for anime character extraction
- Not recommended for production use (still in development phase)