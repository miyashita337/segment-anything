#!/usr/bin/env python3
"""自動品質監視開始スクリプト"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from tools.core.automated_quality_testing import AutomatedQualityTesting

if __name__ == "__main__":
    testing_system = AutomatedQualityTesting()
    testing_system.run_continuous_monitoring(interval_hours=24)
