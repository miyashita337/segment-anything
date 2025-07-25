#!/usr/bin/env python3
"""
GPT-4O設計による新評価システム
座標+内容統合評価でキャラクター抽出の真の成功率を測定
"""

from .base import EvaluatorBase, EvaluationResult, EvaluationConfig
from .spatial import IoUEvaluator
from .content import ContentEvaluator
from .matcher import RegionMatcher, MultiCharacterMatcher
from .orchestrator import EvaluationOrchestrator

__version__ = "1.0.0"
__author__ = "Claude + GPT-4O Collaboration"

__all__ = [
    'EvaluatorBase',
    'EvaluationResult', 
    'EvaluationConfig',
    'IoUEvaluator',
    'ContentEvaluator',
    'RegionMatcher',
    'MultiCharacterMatcher',
    'EvaluationOrchestrator'
]