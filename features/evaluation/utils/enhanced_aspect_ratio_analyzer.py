#!/usr/bin/env python3
"""
P1-007: Enhanced Aspect Ratio Analyzer
アスペクト比判定の改善システム - 動的閾値計算と多指標統合

SOLID Principles Applied:
- Single Responsibility: 各クラスは単一の責任を持つ
- Open/Closed: 新しい Character Style の追加が容易
- Liskov Substitution: Interface での型安全性確保
- Interface Segregation: 小さな interface に分割
- Dependency Inversion: 抽象に依存、具象に依存しない
"""

import numpy as np
import cv2

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)


class CharacterStyle(Enum):
    """Character style categories"""

    REALISTIC = "realistic"
    ANIME = "anime"
    CHIBI = "chibi"
    DEFORMED = "deformed"
    UNKNOWN = "unknown"


@dataclass
class AspectRatioAnalysis:
    """Aspect ratio analysis results"""

    base_ratio: float
    adjusted_ratio: float
    style_category: CharacterStyle
    confidence_score: float
    threshold_range: Tuple[float, float]
    reasoning: str
    quality_indicators: Dict[str, float]


@dataclass
class DynamicThresholds:
    """Dynamic threshold configuration"""

    ideal_range: Tuple[float, float]
    acceptable_range: Tuple[float, float]
    extended_range: Tuple[float, float]
    style_factor: float
    size_factor: float


class ThresholdCalculatorProtocol(Protocol):
    """Protocol for threshold calculation strategies"""

    def calculate_thresholds(self, image_properties: Dict[str, Any]) -> DynamicThresholds:
        """Calculate dynamic thresholds based on image properties"""
        ...


class StyleDetectorProtocol(Protocol):
    """Protocol for character style detection"""

    def detect_style(
        self, image: np.ndarray, mask_data: Dict[str, Any]
    ) -> Tuple[CharacterStyle, float]:
        """Detect character style and confidence"""
        ...


class BasicThresholdCalculator:
    """
    Basic implementation of threshold calculation
    Follows YAGNI principle - implements only what's needed
    """

    def __init__(self):
        """Initialize with default settings"""
        self.base_thresholds = {
            CharacterStyle.REALISTIC: (1.4, 2.2),
            CharacterStyle.ANIME: (1.2, 2.5),
            CharacterStyle.CHIBI: (1.0, 1.8),
            CharacterStyle.DEFORMED: (0.8, 2.0),
            CharacterStyle.UNKNOWN: (1.2, 2.5),
        }

    def calculate_thresholds(self, image_properties: Dict[str, Any]) -> DynamicThresholds:
        """
        Calculate dynamic thresholds based on image properties

        Args:
            image_properties: Dict containing style, size, etc.

        Returns:
            DynamicThresholds: Calculated threshold configuration
        """
        style = image_properties.get("style", CharacterStyle.UNKNOWN)
        image_size = image_properties.get("image_size", (512, 512))

        # Base thresholds from character style
        base_min, base_max = self.base_thresholds[style]

        # Size adaptation factor - different for different image sizes
        height, width = image_size
        total_pixels = height * width

        # Base size factor on total image area
        if total_pixels < 300000:  # Small images (< ~548x548)
            size_factor = 0.9
        elif total_pixels > 800000:  # Large images (> ~894x894)
            size_factor = 1.1
        else:
            size_factor = 1.0

        # Style-specific adjustment factor
        style_factors = {
            CharacterStyle.REALISTIC: 1.0,
            CharacterStyle.ANIME: 1.1,
            CharacterStyle.CHIBI: 0.9,
            CharacterStyle.DEFORMED: 0.8,
            CharacterStyle.UNKNOWN: 1.0,
        }
        style_factor = style_factors[style]

        # Calculate adaptive ranges
        ideal_min = base_min * style_factor
        ideal_max = base_max * style_factor

        acceptable_min = ideal_min * 0.85
        acceptable_max = ideal_max * 1.15

        extended_min = acceptable_min * 0.8
        extended_max = acceptable_max * 1.25

        return DynamicThresholds(
            ideal_range=(ideal_min, ideal_max),
            acceptable_range=(acceptable_min, acceptable_max),
            extended_range=(extended_min, extended_max),
            style_factor=style_factor,
            size_factor=size_factor,
        )


class HeuristicStyleDetector:
    """
    Heuristic-based character style detection
    Uses simple image analysis to classify character style
    """

    def __init__(self):
        """Initialize style detector"""
        self.style_indicators = {
            "edge_density_threshold": 0.15,
            "color_saturation_threshold": 0.6,
            "size_ratio_threshold": 0.3,
        }

    def detect_style(
        self, image: np.ndarray, mask_data: Dict[str, Any]
    ) -> Tuple[CharacterStyle, float]:
        """
        Detect character style based on image characteristics

        Args:
            image: Input image (BGR)
            mask_data: Mask information including bbox

        Returns:
            Tuple[CharacterStyle, float]: Detected style and confidence
        """
        try:
            bbox = mask_data.get("bbox", [0, 0, image.shape[1], image.shape[0]])
            x, y, w, h = bbox

            # Extract region of interest with bounds checking
            y_start = max(0, y)
            y_end = min(image.shape[0], y + h)
            x_start = max(0, x)
            x_end = min(image.shape[1], x + w)

            roi = image[y_start:y_end, x_start:x_end]

            if roi.size == 0 or (y_end - y_start) < 10 or (x_end - x_start) < 10:
                return CharacterStyle.UNKNOWN, 0.1

            # Analyze characteristics
            edge_density = self._calculate_edge_density(roi)
            color_saturation = self._calculate_color_saturation(roi)
            size_ratio = (w * h) / (image.shape[0] * image.shape[1])

            # Classification logic
            style, confidence = self._classify_style(edge_density, color_saturation, size_ratio)

            return style, confidence

        except Exception as e:
            logger.warning(f"Style detection failed: {e}")
            return CharacterStyle.UNKNOWN, 0.1

    def _calculate_edge_density(self, roi: np.ndarray) -> float:
        """Calculate edge density in ROI"""
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi

        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.sum(edges > 0)
        total_pixels = gray.shape[0] * gray.shape[1]

        return edge_pixels / max(total_pixels, 1)

    def _calculate_color_saturation(self, roi: np.ndarray) -> float:
        """Calculate average color saturation"""
        if len(roi.shape) != 3:
            return 0.0

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].astype(np.float32) / 255.0

        return np.mean(saturation)

    def _classify_style(
        self, edge_density: float, color_saturation: float, size_ratio: float
    ) -> Tuple[CharacterStyle, float]:
        """Classify character style based on metrics"""

        # Chibi characteristics: high edge density, high saturation, small size
        if edge_density > 0.2 and color_saturation > 0.7 and size_ratio < 0.2:
            return CharacterStyle.CHIBI, 0.8

        # Anime characteristics: medium-high edge density, high saturation
        elif edge_density > 0.15 and color_saturation > 0.6:
            return CharacterStyle.ANIME, 0.7

        # Realistic characteristics: lower edge density, medium saturation
        elif edge_density < 0.1 and color_saturation < 0.5:
            return CharacterStyle.REALISTIC, 0.6

        # Deformed characteristics: very high edge density or very high saturation
        elif edge_density > 0.25 or color_saturation > 0.8:
            return CharacterStyle.DEFORMED, 0.5

        else:
            return CharacterStyle.UNKNOWN, 0.3


class EnhancedAspectRatioAnalyzer:
    """
    Enhanced aspect ratio analysis system

    Key Improvements:
    - Dynamic threshold calculation based on character style
    - Multi-factor analysis beyond simple aspect ratio
    - Style-aware processing for different character types
    - Backward compatibility with existing systems
    """

    def __init__(
        self,
        threshold_calculator: Optional[ThresholdCalculatorProtocol] = None,
        style_detector: Optional[StyleDetectorProtocol] = None,
    ):
        """
        Initialize analyzer with dependency injection

        Args:
            threshold_calculator: Strategy for threshold calculation
            style_detector: Strategy for style detection
        """
        self.threshold_calculator = threshold_calculator or BasicThresholdCalculator()
        self.style_detector = style_detector or HeuristicStyleDetector()

        # Quality evaluation weights
        self.quality_weights = {
            "aspect_ratio_fit": 0.4,  # How well aspect ratio fits expected range
            "style_consistency": 0.25,  # Consistency with detected style
            "proportional_balance": 0.2,  # Body proportions analysis
            "edge_quality": 0.15,  # Edge distribution quality
        }

    def analyze_character_proportions(
        self,
        image: np.ndarray,
        mask_data: Dict[str, Any],
        original_aspect_ratio: Optional[float] = None,
    ) -> AspectRatioAnalysis:
        """
        Comprehensive aspect ratio analysis

        Args:
            image: Input image (BGR)
            mask_data: Mask data including bbox, area, etc.
            original_aspect_ratio: Previously calculated aspect ratio (for compatibility)

        Returns:
            AspectRatioAnalysis: Detailed analysis results
        """
        try:
            # Extract basic measurements
            bbox = mask_data.get("bbox", [0, 0, 100, 100])
            x, y, w, h = bbox

            # Calculate base aspect ratio
            base_ratio = h / max(w, 1) if original_aspect_ratio is None else original_aspect_ratio

            # Detect character style
            style, style_confidence = self.style_detector.detect_style(image, mask_data)

            # Calculate dynamic thresholds
            image_properties = {
                "style": style,
                "image_size": (image.shape[0], image.shape[1]),
                "mask_size": (w, h),
                "style_confidence": style_confidence,
            }
            thresholds = self.threshold_calculator.calculate_thresholds(image_properties)

            # Perform multi-factor analysis
            quality_indicators = self._analyze_quality_indicators(
                image, mask_data, base_ratio, thresholds
            )

            # Calculate adjusted aspect ratio with style-aware corrections
            adjusted_ratio = self._calculate_adjusted_ratio(base_ratio, style, quality_indicators)

            # Generate reasoning
            reasoning = self._generate_reasoning(
                base_ratio, adjusted_ratio, style, thresholds, quality_indicators
            )

            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(
                quality_indicators, style_confidence
            )

            return AspectRatioAnalysis(
                base_ratio=base_ratio,
                adjusted_ratio=adjusted_ratio,
                style_category=style,
                confidence_score=overall_confidence,
                threshold_range=thresholds.ideal_range,
                reasoning=reasoning,
                quality_indicators=quality_indicators,
            )

        except Exception as e:
            logger.error(f"Aspect ratio analysis failed: {e}")
            # Fallback to basic analysis
            return self._create_fallback_analysis(original_aspect_ratio or 1.5)

    def _analyze_quality_indicators(
        self,
        image: np.ndarray,
        mask_data: Dict[str, Any],
        aspect_ratio: float,
        thresholds: DynamicThresholds,
    ) -> Dict[str, float]:
        """Analyze multiple quality indicators"""

        indicators = {}

        # 1. Aspect ratio fit score
        ideal_min, ideal_max = thresholds.ideal_range
        if ideal_min <= aspect_ratio <= ideal_max:
            indicators["aspect_ratio_fit"] = 1.0
        else:
            # Distance from ideal range
            if aspect_ratio < ideal_min:
                distance = ideal_min - aspect_ratio
                indicators["aspect_ratio_fit"] = max(0, 1.0 - distance / ideal_min)
            else:
                distance = aspect_ratio - ideal_max
                indicators["aspect_ratio_fit"] = max(0, 1.0 - distance / ideal_max)

        # 2. Style consistency (already calculated in style detection)
        indicators["style_consistency"] = 0.7  # Default reasonable value

        # 3. Proportional balance analysis
        indicators["proportional_balance"] = self._analyze_proportional_balance(image, mask_data)

        # 4. Edge quality analysis
        indicators["edge_quality"] = self._analyze_edge_quality(image, mask_data)

        return indicators

    def _analyze_proportional_balance(self, image: np.ndarray, mask_data: Dict[str, Any]) -> float:
        """Analyze body proportional balance"""
        try:
            bbox = mask_data.get("bbox", [0, 0, 100, 100])
            x, y, w, h = bbox

            # Extract ROI
            roi = image[
                max(0, y) : min(image.shape[0], y + h), max(0, x) : min(image.shape[1], x + w)
            ]

            if roi.size == 0:
                return 0.5

            # Analyze vertical distribution
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
            vertical_profile = np.mean(gray, axis=1)

            # Check for typical body structure (head, torso, legs)
            profile_peaks = self._find_significant_peaks(vertical_profile)

            # Good proportional balance should have 2-3 main regions
            if 2 <= len(profile_peaks) <= 3:
                return min(0.8 + 0.1 * len(profile_peaks), 1.0)
            else:
                return max(0.3, 0.8 - 0.1 * abs(len(profile_peaks) - 2.5))

        except Exception as e:
            logger.debug(f"Proportional balance analysis failed: {e}")
            return 0.5

    def _analyze_edge_quality(self, image: np.ndarray, mask_data: Dict[str, Any]) -> float:
        """Analyze edge distribution quality"""
        try:
            bbox = mask_data.get("bbox", [0, 0, 100, 100])
            x, y, w, h = bbox

            # Extract ROI
            roi = image[
                max(0, y) : min(image.shape[0], y + h), max(0, x) : min(image.shape[1], x + w)
            ]

            if roi.size == 0:
                return 0.5

            # Edge detection
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
            edges = cv2.Canny(gray, 50, 150)

            # Analyze edge distribution
            edge_density = np.sum(edges > 0) / edges.size

            # Good edge quality: moderate density, well distributed
            if 0.05 <= edge_density <= 0.25:
                return min(1.0, edge_density / 0.15)
            else:
                return max(0.2, 1.0 - abs(edge_density - 0.15) / 0.15)

        except Exception as e:
            logger.debug(f"Edge quality analysis failed: {e}")
            return 0.5

    def _find_significant_peaks(self, profile: np.ndarray, min_height: float = 0.1) -> List[int]:
        """Find significant peaks in vertical profile"""
        if len(profile) < 3:
            return []

        # Smooth profile
        kernel = np.ones(3) / 3
        if len(profile) >= 3:
            smoothed = np.convolve(profile, kernel, mode="same")
        else:
            smoothed = profile

        # Find peaks
        peaks = []
        threshold = np.mean(smoothed) + min_height * np.std(smoothed)

        for i in range(1, len(smoothed) - 1):
            if (
                smoothed[i] > smoothed[i - 1]
                and smoothed[i] > smoothed[i + 1]
                and smoothed[i] > threshold
            ):
                peaks.append(i)

        return peaks

    def _calculate_adjusted_ratio(
        self, base_ratio: float, style: CharacterStyle, quality_indicators: Dict[str, float]
    ) -> float:
        """Calculate style-aware adjusted aspect ratio"""

        # Style-specific adjustment factors
        style_adjustments = {
            CharacterStyle.REALISTIC: 1.0,
            CharacterStyle.ANIME: 1.05,
            CharacterStyle.CHIBI: 0.9,
            CharacterStyle.DEFORMED: 0.95,
            CharacterStyle.UNKNOWN: 1.0,
        }

        style_factor = style_adjustments[style]

        # Quality-based adjustment
        quality_factor = quality_indicators.get("proportional_balance", 0.5)
        adjustment = 1.0 + (quality_factor - 0.5) * 0.1

        adjusted_ratio = base_ratio * style_factor * adjustment

        # Sanity check: don't deviate too much from original
        max_adjustment = 0.3
        if abs(adjusted_ratio - base_ratio) > max_adjustment:
            if adjusted_ratio > base_ratio:
                adjusted_ratio = base_ratio + max_adjustment
            else:
                adjusted_ratio = base_ratio - max_adjustment

        return adjusted_ratio

    def _generate_reasoning(
        self,
        base_ratio: float,
        adjusted_ratio: float,
        style: CharacterStyle,
        thresholds: DynamicThresholds,
        quality_indicators: Dict[str, float],
    ) -> str:
        """Generate human-readable reasoning for the analysis"""

        reasoning_parts = []

        # Style detection
        reasoning_parts.append(f"Detected {style.value} character style")

        # Aspect ratio evaluation
        ideal_min, ideal_max = thresholds.ideal_range
        if ideal_min <= adjusted_ratio <= ideal_max:
            reasoning_parts.append(f"aspect ratio {adjusted_ratio:.2f} fits ideal range")
        else:
            reasoning_parts.append(
                f"aspect ratio {adjusted_ratio:.2f} outside ideal range "
                f"({ideal_min:.2f}-{ideal_max:.2f})"
            )

        # Quality indicators
        strong_indicators = [k for k, v in quality_indicators.items() if v > 0.7]
        weak_indicators = [k for k, v in quality_indicators.items() if v < 0.4]

        if strong_indicators:
            reasoning_parts.append(f"strong {', '.join(strong_indicators)}")
        if weak_indicators:
            reasoning_parts.append(f"weak {', '.join(weak_indicators)}")

        return "; ".join(reasoning_parts)

    def _calculate_overall_confidence(
        self, quality_indicators: Dict[str, float], style_confidence: float
    ) -> float:
        """Calculate overall analysis confidence"""

        # Weighted average of quality indicators
        weighted_quality = sum(
            quality_indicators.get(indicator, 0.5) * weight
            for indicator, weight in self.quality_weights.items()
        )

        # Combine with style detection confidence
        overall_confidence = 0.7 * weighted_quality + 0.3 * style_confidence

        return min(1.0, max(0.1, overall_confidence))

    def _create_fallback_analysis(self, aspect_ratio: float) -> AspectRatioAnalysis:
        """Create fallback analysis when main analysis fails"""
        return AspectRatioAnalysis(
            base_ratio=aspect_ratio,
            adjusted_ratio=aspect_ratio,
            style_category=CharacterStyle.UNKNOWN,
            confidence_score=0.3,
            threshold_range=(1.2, 2.5),
            reasoning="Fallback analysis due to processing error",
            quality_indicators={"fallback": 0.3},
        )

    def calculate_fullbody_score(self, analysis: AspectRatioAnalysis) -> float:
        """
        Calculate fullbody score based on enhanced analysis
        Compatible with existing scoring system

        Args:
            analysis: Enhanced aspect ratio analysis results

        Returns:
            float: Fullbody score (0.0-1.0)
        """

        # Use adjusted aspect ratio for scoring
        aspect_ratio = analysis.adjusted_ratio
        ideal_min, ideal_max = analysis.threshold_range

        # Calculate base score based on fit to ideal range
        if ideal_min <= aspect_ratio <= ideal_max:
            # Perfect fit
            range_center = (ideal_min + ideal_max) / 2
            distance_from_center = abs(aspect_ratio - range_center)
            range_width = ideal_max - ideal_min
            base_score = 1.0 - (distance_from_center / (range_width / 2)) * 0.2
        else:
            # Outside ideal range - calculate penalty
            if aspect_ratio < ideal_min:
                distance = ideal_min - aspect_ratio
                base_score = max(0, 1.0 - distance / ideal_min)
            else:
                distance = aspect_ratio - ideal_max
                base_score = max(0, 1.0 - distance / ideal_max)

        # Apply quality modifiers
        quality_modifier = 1.0
        for indicator, value in analysis.quality_indicators.items():
            weight = self.quality_weights.get(indicator, 0.1)
            quality_modifier += (value - 0.5) * weight * 0.2

        # Apply confidence modifier
        confidence_modifier = 0.5 + 0.5 * analysis.confidence_score

        final_score = base_score * quality_modifier * confidence_modifier

        return min(1.0, max(0.0, final_score))


# Convenience function for backward compatibility
def evaluate_enhanced_aspect_ratio(
    image: np.ndarray, mask_data: Dict[str, Any], original_aspect_ratio: Optional[float] = None
) -> Tuple[float, AspectRatioAnalysis]:
    """
    Convenience function for enhanced aspect ratio evaluation

    Args:
        image: Input image (BGR)
        mask_data: Mask data dictionary
        original_aspect_ratio: Previously calculated aspect ratio

    Returns:
        Tuple[float, AspectRatioAnalysis]: (fullbody_score, detailed_analysis)
    """

    analyzer = EnhancedAspectRatioAnalyzer()
    analysis = analyzer.analyze_character_proportions(image, mask_data, original_aspect_ratio)
    fullbody_score = analyzer.calculate_fullbody_score(analysis)

    return fullbody_score, analysis


if __name__ == "__main__":
    # Simple test
    print("=== P1-007 Enhanced Aspect Ratio Analyzer Test ===")

    # Create test image and mask data
    test_image = np.zeros((400, 200, 3), dtype=np.uint8)
    test_mask_data = {"bbox": [50, 50, 100, 300], "area": 30000}  # x, y, w, h

    # Test the analyzer
    fullbody_score, analysis = evaluate_enhanced_aspect_ratio(test_image, test_mask_data)

    print(f"Base aspect ratio: {analysis.base_ratio:.3f}")
    print(f"Adjusted aspect ratio: {analysis.adjusted_ratio:.3f}")
    print(f"Character style: {analysis.style_category.value}")
    print(f"Confidence: {analysis.confidence_score:.3f}")
    print(f"Fullbody score: {fullbody_score:.3f}")
    print(f"Reasoning: {analysis.reasoning}")
    print(f"Quality indicators: {analysis.quality_indicators}")
    print("✅ Basic test completed")

