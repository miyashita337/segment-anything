#!/usr/bin/env python3
"""
P1-007: Enhanced Aspect Ratio Analyzer Unit Tests
アスペクト比判定改善システムの単体テスト

テスト項目:
- 基本的なアスペクト比計算
- 動的閾値計算
- キャラクタースタイル検出
- 品質指標分析
- 既存システムとの互換性
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from features.evaluation.utils.enhanced_aspect_ratio_analyzer import (
    EnhancedAspectRatioAnalyzer,
    BasicThresholdCalculator,
    HeuristicStyleDetector,
    CharacterStyle,
    AspectRatioAnalysis,
    DynamicThresholds,
    evaluate_enhanced_aspect_ratio
)


class TestCharacterStyle:
    """Test CharacterStyle enum"""
    
    def test_character_style_enum_values(self):
        """Test that all expected character styles are available"""
        expected_styles = {'realistic', 'anime', 'chibi', 'deformed', 'unknown'}
        actual_styles = {style.value for style in CharacterStyle}
        assert actual_styles == expected_styles
    
    def test_character_style_string_representation(self):
        """Test string representation of character styles"""
        assert CharacterStyle.ANIME.value == "anime"
        assert CharacterStyle.CHIBI.value == "chibi"
        assert CharacterStyle.REALISTIC.value == "realistic"


class TestBasicThresholdCalculator:
    """Test BasicThresholdCalculator class"""
    
    def setUp(self):
        """Set up test instance"""
        self.calculator = BasicThresholdCalculator()
    
    def test_threshold_calculator_initialization(self):
        """Test threshold calculator initialization"""
        calculator = BasicThresholdCalculator()
        assert calculator.base_thresholds is not None
        assert CharacterStyle.ANIME in calculator.base_thresholds
        assert CharacterStyle.CHIBI in calculator.base_thresholds
    
    def test_basic_threshold_calculation(self):
        """Test basic threshold calculation"""
        calculator = BasicThresholdCalculator()
        
        # Test with anime style
        image_properties = {
            'style': CharacterStyle.ANIME,
            'image_size': (512, 512)
        }
        
        thresholds = calculator.calculate_thresholds(image_properties)
        
        assert isinstance(thresholds, DynamicThresholds)
        assert thresholds.ideal_range[0] > 0
        assert thresholds.ideal_range[1] > thresholds.ideal_range[0]
        assert thresholds.acceptable_range[0] <= thresholds.ideal_range[0]
        assert thresholds.acceptable_range[1] >= thresholds.ideal_range[1]
    
    def test_style_specific_thresholds(self):
        """Test that different styles produce different thresholds"""
        calculator = BasicThresholdCalculator()
        
        anime_props = {'style': CharacterStyle.ANIME, 'image_size': (512, 512)}
        chibi_props = {'style': CharacterStyle.CHIBI, 'image_size': (512, 512)}
        
        anime_thresholds = calculator.calculate_thresholds(anime_props)
        chibi_thresholds = calculator.calculate_thresholds(chibi_props)
        
        # Chibi should have lower thresholds (shorter, wider characters)
        assert chibi_thresholds.ideal_range[1] < anime_thresholds.ideal_range[1]
    
    def test_size_adaptation(self):
        """Test size-based threshold adaptation"""
        calculator = BasicThresholdCalculator()
        
        small_props = {'style': CharacterStyle.ANIME, 'image_size': (256, 256)}
        large_props = {'style': CharacterStyle.ANIME, 'image_size': (1024, 1024)}
        
        small_thresholds = calculator.calculate_thresholds(small_props)
        large_thresholds = calculator.calculate_thresholds(large_props)
        
        # Size factor should affect thresholds
        assert small_thresholds.size_factor != large_thresholds.size_factor


class TestHeuristicStyleDetector:
    """Test HeuristicStyleDetector class"""
    
    def test_style_detector_initialization(self):
        """Test style detector initialization"""
        detector = HeuristicStyleDetector()
        assert detector.style_indicators is not None
        assert 'edge_density_threshold' in detector.style_indicators
    
    def test_style_detection_with_valid_input(self):
        """Test style detection with valid inputs"""
        detector = HeuristicStyleDetector()
        
        # Create test image
        test_image = np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)
        test_mask_data = {'bbox': [10, 10, 80, 180]}
        
        style, confidence = detector.detect_style(test_image, test_mask_data)
        
        assert isinstance(style, CharacterStyle)
        assert 0.0 <= confidence <= 1.0
    
    def test_style_detection_with_empty_roi(self):
        """Test style detection with invalid ROI"""
        detector = HeuristicStyleDetector()
        
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_mask_data = {'bbox': [95, 95, 20, 20]}  # Mostly out of bounds
        
        style, confidence = detector.detect_style(test_image, test_mask_data)
        
        assert style == CharacterStyle.UNKNOWN
        assert confidence <= 0.2
    
    def test_edge_density_calculation(self):
        """Test edge density calculation"""
        detector = HeuristicStyleDetector()
        
        # Create image with known edge patterns
        edge_image = np.zeros((100, 100), dtype=np.uint8)
        edge_image[40:60, :] = 255  # Horizontal line
        edge_image[:, 40:60] = 255  # Vertical line
        
        edge_density = detector._calculate_edge_density(edge_image)
        
        assert 0.0 <= edge_density <= 1.0
        assert edge_density > 0.02  # Should detect some edges (lowered threshold)
    
    def test_color_saturation_calculation(self):
        """Test color saturation calculation"""
        detector = HeuristicStyleDetector()
        
        # Create saturated color image
        saturated_image = np.zeros((50, 50, 3), dtype=np.uint8)
        saturated_image[:, :, 2] = 255  # Pure red
        
        saturation = detector._calculate_color_saturation(saturated_image)
        
        assert 0.0 <= saturation <= 1.0
        assert saturation > 0.8  # Pure red should be highly saturated


class TestEnhancedAspectRatioAnalyzer:
    """Test EnhancedAspectRatioAnalyzer class"""
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization"""
        analyzer = EnhancedAspectRatioAnalyzer()
        assert analyzer.threshold_calculator is not None
        assert analyzer.style_detector is not None
        assert analyzer.quality_weights is not None
    
    def test_analyzer_with_custom_components(self):
        """Test analyzer initialization with custom components"""
        custom_calculator = BasicThresholdCalculator()
        custom_detector = HeuristicStyleDetector()
        
        analyzer = EnhancedAspectRatioAnalyzer(
            threshold_calculator=custom_calculator,
            style_detector=custom_detector
        )
        
        assert analyzer.threshold_calculator is custom_calculator
        assert analyzer.style_detector is custom_detector
    
    def test_character_proportions_analysis(self):
        """Test comprehensive character proportions analysis"""
        analyzer = EnhancedAspectRatioAnalyzer()
        
        # Create test data
        test_image = np.random.randint(0, 255, (400, 200, 3), dtype=np.uint8)
        test_mask_data = {
            'bbox': [50, 50, 100, 300],  # x, y, w, h -> aspect ratio 3.0
            'area': 30000
        }
        
        analysis = analyzer.analyze_character_proportions(test_image, test_mask_data)
        
        assert isinstance(analysis, AspectRatioAnalysis)
        assert analysis.base_ratio > 0
        assert analysis.adjusted_ratio > 0
        assert isinstance(analysis.style_category, CharacterStyle)
        assert 0.0 <= analysis.confidence_score <= 1.0
        assert len(analysis.threshold_range) == 2
        assert analysis.reasoning is not None
        assert analysis.quality_indicators is not None
    
    def test_character_proportions_with_original_ratio(self):
        """Test analysis with provided original aspect ratio"""
        analyzer = EnhancedAspectRatioAnalyzer()
        
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        test_mask_data = {'bbox': [50, 50, 100, 100]}
        original_ratio = 2.5
        
        analysis = analyzer.analyze_character_proportions(
            test_image, test_mask_data, original_ratio
        )
        
        assert analysis.base_ratio == original_ratio
    
    def test_quality_indicators_analysis(self):
        """Test quality indicators analysis"""
        analyzer = EnhancedAspectRatioAnalyzer()
        
        test_image = np.random.randint(0, 255, (300, 150, 3), dtype=np.uint8)
        test_mask_data = {'bbox': [25, 25, 100, 250]}
        
        # Mock thresholds
        from features.evaluation.utils.enhanced_aspect_ratio_analyzer import DynamicThresholds
        mock_thresholds = DynamicThresholds(
            ideal_range=(1.5, 2.5),
            acceptable_range=(1.2, 2.8),
            extended_range=(1.0, 3.0),
            style_factor=1.0,
            size_factor=1.0
        )
        
        indicators = analyzer._analyze_quality_indicators(
            test_image, test_mask_data, 2.0, mock_thresholds
        )
        
        assert 'aspect_ratio_fit' in indicators
        assert 'style_consistency' in indicators
        assert 'proportional_balance' in indicators
        assert 'edge_quality' in indicators
        
        for indicator, value in indicators.items():
            assert 0.0 <= value <= 1.0, f"Indicator {indicator} out of range: {value}"
    
    def test_proportional_balance_analysis(self):
        """Test proportional balance analysis"""
        analyzer = EnhancedAspectRatioAnalyzer()
        
        # Create test image with clear vertical structure
        test_image = np.zeros((300, 100, 3), dtype=np.uint8)
        # Add some structure (head, body, legs regions)
        test_image[10:50, :] = [100, 100, 100]    # Head region
        test_image[100:200, :] = [150, 150, 150]  # Body region  
        test_image[220:280, :] = [120, 120, 120]  # Legs region
        
        test_mask_data = {'bbox': [0, 0, 100, 300]}
        
        balance_score = analyzer._analyze_proportional_balance(test_image, test_mask_data)
        
        assert 0.0 <= balance_score <= 1.0
        # With clear structure, should score reasonably well
        assert balance_score >= 0.3
    
    def test_edge_quality_analysis(self):
        """Test edge quality analysis"""
        analyzer = EnhancedAspectRatioAnalyzer()
        
        # Create image with good edge characteristics
        test_image = np.zeros((200, 100, 3), dtype=np.uint8)
        test_image[50:150, 25:75] = [255, 255, 255]  # White rectangle for clear edges
        
        test_mask_data = {'bbox': [20, 40, 60, 120]}
        
        edge_score = analyzer._analyze_edge_quality(test_image, test_mask_data)
        
        assert 0.0 <= edge_score <= 1.0
    
    def test_adjusted_ratio_calculation(self):
        """Test adjusted aspect ratio calculation"""
        analyzer = EnhancedAspectRatioAnalyzer()
        
        base_ratio = 2.0
        style = CharacterStyle.ANIME
        quality_indicators = {
            'proportional_balance': 0.8,
            'edge_quality': 0.7
        }
        
        adjusted_ratio = analyzer._calculate_adjusted_ratio(base_ratio, style, quality_indicators)
        
        assert adjusted_ratio > 0
        # Should be reasonably close to base ratio
        assert abs(adjusted_ratio - base_ratio) <= 0.5
    
    def test_fullbody_score_calculation(self):
        """Test fullbody score calculation"""
        analyzer = EnhancedAspectRatioAnalyzer()
        
        # Create mock analysis with good characteristics
        from features.evaluation.utils.enhanced_aspect_ratio_analyzer import AspectRatioAnalysis
        mock_analysis = AspectRatioAnalysis(
            base_ratio=1.8,
            adjusted_ratio=1.9,
            style_category=CharacterStyle.ANIME,
            confidence_score=0.8,
            threshold_range=(1.2, 2.5),
            reasoning="Test analysis",
            quality_indicators={
                'aspect_ratio_fit': 0.9,
                'style_consistency': 0.8,
                'proportional_balance': 0.7,
                'edge_quality': 0.6
            }
        )
        
        fullbody_score = analyzer.calculate_fullbody_score(mock_analysis)
        
        assert 0.0 <= fullbody_score <= 1.0
        # With good characteristics, should score well
        assert fullbody_score >= 0.5
    
    def test_fallback_analysis_creation(self):
        """Test fallback analysis creation"""
        analyzer = EnhancedAspectRatioAnalyzer()
        
        fallback_analysis = analyzer._create_fallback_analysis(1.5)
        
        assert isinstance(fallback_analysis, AspectRatioAnalysis)
        assert fallback_analysis.base_ratio == 1.5
        assert fallback_analysis.adjusted_ratio == 1.5
        assert fallback_analysis.style_category == CharacterStyle.UNKNOWN
        assert fallback_analysis.confidence_score <= 0.5
        assert "Fallback" in fallback_analysis.reasoning or "fallback" in fallback_analysis.reasoning


class TestConvenienceFunction:
    """Test convenience function for backward compatibility"""
    
    def test_evaluate_enhanced_aspect_ratio_function(self):
        """Test the convenience function"""
        test_image = np.random.randint(0, 255, (300, 150, 3), dtype=np.uint8)
        test_mask_data = {
            'bbox': [25, 25, 100, 250],
            'area': 25000
        }
        
        fullbody_score, analysis = evaluate_enhanced_aspect_ratio(test_image, test_mask_data)
        
        assert 0.0 <= fullbody_score <= 1.0
        assert isinstance(analysis, AspectRatioAnalysis)
    
    def test_evaluate_with_original_aspect_ratio(self):
        """Test convenience function with original aspect ratio"""
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        test_mask_data = {'bbox': [50, 50, 100, 100]}
        original_ratio = 2.2
        
        fullbody_score, analysis = evaluate_enhanced_aspect_ratio(
            test_image, test_mask_data, original_ratio
        )
        
        assert analysis.base_ratio == original_ratio
        assert 0.0 <= fullbody_score <= 1.0


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_empty_image_handling(self):
        """Test handling of empty images"""
        empty_image = np.zeros((0, 0, 3), dtype=np.uint8)
        test_mask_data = {'bbox': [0, 0, 0, 0]}
        
        # Should not crash and return reasonable defaults
        fullbody_score, analysis = evaluate_enhanced_aspect_ratio(empty_image, test_mask_data)
        
        assert 0.0 <= fullbody_score <= 1.0
        assert isinstance(analysis, AspectRatioAnalysis)
    
    def test_invalid_bbox_handling(self):
        """Test handling of invalid bounding boxes"""
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        invalid_mask_data = {'bbox': [150, 150, 50, 50]}  # Out of bounds
        
        # Should not crash
        fullbody_score, analysis = evaluate_enhanced_aspect_ratio(test_image, invalid_mask_data)
        
        assert 0.0 <= fullbody_score <= 1.0
        assert isinstance(analysis, AspectRatioAnalysis)
    
    def test_missing_mask_data_handling(self):
        """Test handling of missing mask data"""
        test_image = np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)
        incomplete_mask_data = {}  # Missing bbox
        
        # Should use defaults and not crash
        fullbody_score, analysis = evaluate_enhanced_aspect_ratio(test_image, incomplete_mask_data)
        
        assert 0.0 <= fullbody_score <= 1.0
        assert isinstance(analysis, AspectRatioAnalysis)


class TestBackwardCompatibility:
    """Test backward compatibility with existing systems"""
    
    def test_score_range_compatibility(self):
        """Test that scores are in expected range for existing systems"""
        test_image = np.random.randint(0, 255, (400, 200, 3), dtype=np.uint8)
        test_mask_data = {'bbox': [50, 50, 100, 300]}
        
        fullbody_score, analysis = evaluate_enhanced_aspect_ratio(test_image, test_mask_data)
        
        # Score should be in range expected by existing quality assessment
        assert 0.0 <= fullbody_score <= 1.0
        assert isinstance(fullbody_score, float)
    
    def test_analysis_structure_completeness(self):
        """Test that analysis contains all expected fields"""
        test_image = np.zeros((300, 150, 3), dtype=np.uint8)
        test_mask_data = {'bbox': [25, 25, 100, 250]}
        
        fullbody_score, analysis = evaluate_enhanced_aspect_ratio(test_image, test_mask_data)
        
        # Check all required fields are present
        required_fields = ['base_ratio', 'adjusted_ratio', 'style_category', 
                          'confidence_score', 'threshold_range', 'reasoning', 
                          'quality_indicators']
        
        for field in required_fields:
            assert hasattr(analysis, field), f"Missing required field: {field}"
            assert getattr(analysis, field) is not None, f"Field {field} is None"


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])