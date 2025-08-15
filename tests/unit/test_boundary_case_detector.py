#!/usr/bin/env python3
"""
P1-012境界例自動検出システム 単体テスト
"""

import json
import numpy as np
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# プロジェクトパスの追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from features.evaluation.boundary_case_detector import (
    BoundaryCaseDetector,
    BoundaryCase,
    BoundaryType,
    BoundaryAnalysisResult
)


class TestBoundaryCaseDetector:
    """境界例検出システムテスト"""
    
    @pytest.fixture
    def detector(self):
        """テスト用検出器"""
        return BoundaryCaseDetector("TEST-P1-012")
        
    @pytest.fixture
    def sample_quality_metrics(self):
        """サンプル品質メトリクス"""
        return {
            'width': 800.0,
            'height': 600.0,
            'aspect_ratio': 1.33,
            'total_pixels': 480000.0,
            'brightness': 0.5,
            'contrast': 0.3,
            'sharpness': 0.4,
            'edge_density': 0.1,
            'noise_level': 0.3,
            'saturation': 0.6,
        }
        
    def test_init(self, detector):
        """初期化テスト"""
        assert detector.tracker_id == "TEST-P1-012"
        assert detector.boundary_cases == []
        assert detector.stats['total_processed'] == 0
        assert detector.stats['boundary_cases_found'] == 0
        
    def test_analyze_image_quality_normal(self, detector):
        """通常画像の品質分析テスト"""
        # モック画像データ作成
        mock_image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        
        with patch('cv2.imread', return_value=mock_image):
            with patch('cv2.cvtColor') as mock_cvtcolor:
                # グレースケール画像を返す
                mock_cvtcolor.return_value = np.random.randint(0, 255, (600, 800), dtype=np.uint8)
                
                with patch('cv2.Laplacian') as mock_laplacian:
                    mock_laplacian.return_value = np.random.rand(600, 800) * 100
                    
                    with patch('cv2.Canny') as mock_canny:
                        mock_canny.return_value = np.random.randint(0, 255, (600, 800), dtype=np.uint8)
                        
                        with patch('numpy.fft.fft2') as mock_fft:
                            mock_fft.return_value = np.random.rand(600, 800) + 1j * np.random.rand(600, 800)
                            
                            result = detector.analyze_image_quality(Path("test.jpg"))
                
                # 基本メトリクスの存在確認
                assert 'width' in result
                assert 'height' in result
                assert 'aspect_ratio' in result
                assert 'total_pixels' in result
                assert 'brightness' in result
                assert 'contrast' in result
                assert 'sharpness' in result
                assert 'edge_density' in result
                assert 'noise_level' in result
                assert 'saturation' in result
                
                # 値の妥当性確認
                assert result['width'] == 800.0
                assert result['height'] == 600.0
                assert result['aspect_ratio'] == pytest.approx(800/600, rel=1e-2)
                assert result['total_pixels'] == 480000.0
                
    def test_analyze_image_quality_error(self, detector):
        """画像読み込みエラーテスト"""
        with patch('cv2.imread', return_value=None):
            result = detector.analyze_image_quality(Path("nonexistent.jpg"))
            assert result == {'error': 1.0}
            
    def test_detect_pose_boundary_extreme_aspect(self, detector, sample_quality_metrics):
        """極端なアスペクト比の境界検出テスト"""
        # 極端に縦長
        sample_quality_metrics['aspect_ratio'] = 0.2
        is_boundary, reason, suggestions = detector.detect_pose_boundary(sample_quality_metrics)
        
        assert is_boundary is True
        assert "極端に縦長" in reason
        assert any("横向きポーズ" in s for s in suggestions)
        
        # 極端に横長
        sample_quality_metrics['aspect_ratio'] = 4.0
        is_boundary, reason, suggestions = detector.detect_pose_boundary(sample_quality_metrics)
        
        assert is_boundary is True
        assert "極端に横長" in reason
        assert any("縦向きポーズ" in s for s in suggestions)
        
    def test_detect_pose_boundary_high_edge_density(self, detector, sample_quality_metrics):
        """高エッジ密度の境界検出テスト"""
        sample_quality_metrics['edge_density'] = 0.2
        is_boundary, reason, suggestions = detector.detect_pose_boundary(sample_quality_metrics)
        
        assert is_boundary is True
        assert "高エッジ密度" in reason
        assert any("シンプルなポーズ" in s for s in suggestions)
        
    def test_detect_pose_boundary_normal(self, detector, sample_quality_metrics):
        """通常ポーズの境界検出テスト"""
        is_boundary, reason, suggestions = detector.detect_pose_boundary(sample_quality_metrics)
        
        assert is_boundary is False
        assert reason == ""
        assert suggestions == []
        
    def test_detect_size_boundary_small(self, detector, sample_quality_metrics):
        """小サイズ境界検出テスト"""
        sample_quality_metrics['total_pixels'] = 3000.0
        is_boundary, reason, suggestions = detector.detect_size_boundary(sample_quality_metrics)
        
        assert is_boundary is True
        assert "小サイズ画像" in reason
        assert any("高解像度" in s for s in suggestions)
        
    def test_detect_size_boundary_large(self, detector, sample_quality_metrics):
        """大サイズ境界検出テスト"""
        sample_quality_metrics['total_pixels'] = 600000.0
        is_boundary, reason, suggestions = detector.detect_size_boundary(sample_quality_metrics)
        
        assert is_boundary is True
        assert "大サイズ画像" in reason
        assert any("リサイズ" in s for s in suggestions)
        
    def test_detect_quality_boundary_low_brightness(self, detector, sample_quality_metrics):
        """低明度境界検出テスト"""
        sample_quality_metrics['brightness'] = 0.1
        is_boundary, reason, suggestions = detector.detect_quality_boundary(sample_quality_metrics)
        
        assert is_boundary is True
        assert "低明度" in reason
        assert any("明度の調整" in s for s in suggestions)
        
    def test_detect_quality_boundary_low_contrast(self, detector, sample_quality_metrics):
        """低コントラスト境界検出テスト"""
        sample_quality_metrics['contrast'] = 0.05
        is_boundary, reason, suggestions = detector.detect_quality_boundary(sample_quality_metrics)
        
        assert is_boundary is True
        assert "低コントラスト" in reason
        assert any("コントラスト強化" in s for s in suggestions)
        
    def test_calculate_boundary_confidence_pose(self, detector, sample_quality_metrics):
        """ポーズ境界信頼度計算テスト"""
        # 極端なアスペクト比
        sample_quality_metrics['aspect_ratio'] = 0.2
        sample_quality_metrics['edge_density'] = 0.1
        
        confidence = detector.calculate_boundary_confidence(
            sample_quality_metrics, 
            BoundaryType.POSE_BOUNDARY
        )
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.3  # 極端なケースなので高い信頼度
        
    def test_calculate_boundary_confidence_size(self, detector, sample_quality_metrics):
        """サイズ境界信頼度計算テスト"""
        # 小サイズ
        sample_quality_metrics['total_pixels'] = 2000.0
        
        confidence = detector.calculate_boundary_confidence(
            sample_quality_metrics, 
            BoundaryType.SIZE_BOUNDARY
        )
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # 小サイズなので高い信頼度
        
    def test_calculate_boundary_confidence_quality(self, detector, sample_quality_metrics):
        """品質境界信頼度計算テスト"""
        # 品質問題
        sample_quality_metrics['brightness'] = 0.1  # 低明度
        sample_quality_metrics['contrast'] = 0.05   # 低コントラスト
        sample_quality_metrics['sharpness'] = 0.05  # 低シャープネス
        
        confidence = detector.calculate_boundary_confidence(
            sample_quality_metrics, 
            BoundaryType.QUALITY_BOUNDARY
        )
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.3  # 複数の品質問題があるので高い信頼度
        
    @patch('features.evaluation.boundary_case_detector.BoundaryCaseDetector.analyze_image_quality')
    def test_process_image_boundary_case(self, mock_analyze, detector):
        """境界ケース画像処理テスト"""
        # 境界ケースとなる品質メトリクス
        mock_analyze.return_value = {
            'width': 800.0,
            'height': 600.0,
            'aspect_ratio': 0.2,  # 極端に縦長
            'total_pixels': 480000.0,
            'brightness': 0.5,
            'contrast': 0.3,
            'sharpness': 0.4,
            'edge_density': 0.05,
            'noise_level': 0.3,
            'saturation': 0.6,
        }
        
        result = detector.process_image(Path("test.jpg"))
        
        assert result is not None
        assert isinstance(result, BoundaryCase)
        assert result.case_type == BoundaryType.POSE_BOUNDARY
        assert result.confidence_score > 0.0
        assert "極端に縦長" in result.detection_reason
        assert len(result.improvement_suggestions) > 0
        
    @patch('features.evaluation.boundary_case_detector.BoundaryCaseDetector.analyze_image_quality')
    def test_process_image_normal_case(self, mock_analyze, detector):
        """通常画像処理テスト"""
        # 通常の品質メトリクス
        mock_analyze.return_value = {
            'width': 800.0,
            'height': 600.0,
            'aspect_ratio': 1.33,
            'total_pixels': 480000.0,
            'brightness': 0.5,
            'contrast': 0.3,
            'sharpness': 0.4,
            'edge_density': 0.05,
            'noise_level': 0.3,
            'saturation': 0.6,
        }
        
        result = detector.process_image(Path("test.jpg"))
        
        assert result is None  # 境界ケースではない
        
    @patch('features.evaluation.boundary_case_detector.BoundaryCaseDetector.analyze_image_quality')
    def test_process_image_error(self, mock_analyze, detector):
        """画像処理エラーテスト"""
        mock_analyze.return_value = {'error': 1.0}
        
        result = detector.process_image(Path("error.jpg"))
        
        assert result is None
        
    def test_calculate_improvement_priorities(self, detector):
        """改善優先度計算テスト"""
        # テスト用境界ケース作成
        boundary_cases = [
            BoundaryCase(
                image_path="test1.jpg",
                case_type=BoundaryType.POSE_BOUNDARY,
                confidence_score=0.8,
                quality_metrics={},
                detection_reason="テスト理由1",
                improvement_suggestions=["提案1", "提案2"],
                technical_details={}
            ),
            BoundaryCase(
                image_path="test2.jpg",
                case_type=BoundaryType.POSE_BOUNDARY,
                confidence_score=0.6,
                quality_metrics={},
                detection_reason="テスト理由2",
                improvement_suggestions=["提案1", "提案3"],
                technical_details={}
            ),
            BoundaryCase(
                image_path="test3.jpg",
                case_type=BoundaryType.QUALITY_BOUNDARY,
                confidence_score=0.9,
                quality_metrics={},
                detection_reason="テスト理由3",
                improvement_suggestions=["提案4"],
                technical_details={}
            ),
        ]
        
        priorities = detector._calculate_improvement_priorities(boundary_cases)
        
        assert len(priorities) == 2  # 2つのカテゴリ
        
        # 最優先は POSE_BOUNDARY（2件 × 0.7平均 = 1.4）
        assert priorities[0]['case_type'] == 'pose_boundary'
        assert priorities[0]['count'] == 2
        assert priorities[0]['average_confidence'] == 0.7
        assert priorities[0]['priority_score'] == 1.4
        
        # 次は QUALITY_BOUNDARY（1件 × 0.9平均 = 0.9）
        assert priorities[1]['case_type'] == 'quality_boundary'
        assert priorities[1]['count'] == 1
        assert priorities[1]['average_confidence'] == 0.9
        assert priorities[1]['priority_score'] == 0.9
        
    def test_calculate_improvement_priorities_empty(self, detector):
        """空リストの改善優先度計算テスト"""
        priorities = detector._calculate_improvement_priorities([])
        assert priorities == []


@pytest.mark.integration
class TestBoundaryCaseDetectorIntegration:
    """境界例検出システム統合テスト"""
    
    @pytest.fixture
    def detector(self):
        """統合テスト用検出器"""
        return BoundaryCaseDetector("TEST-INTEGRATION")
        
    def test_full_workflow_with_mock_images(self, detector):
        """モック画像での完全ワークフローテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # テスト画像ファイル作成
            test_images = ["test1.jpg", "test2.jpg", "test3.jpg"]
            for img_name in test_images:
                (temp_path / img_name).touch()
                
            # 品質分析をモック
            with patch.object(detector, 'analyze_image_quality') as mock_analyze:
                # 異なる品質メトリクスを返すように設定
                mock_analyze.side_effect = [
                    {  # 境界ケース: 極端なアスペクト比
                        'width': 200.0, 'height': 800.0, 'aspect_ratio': 0.25,
                        'total_pixels': 160000.0, 'brightness': 0.5, 'contrast': 0.3,
                        'sharpness': 0.4, 'edge_density': 0.05, 'noise_level': 0.3,
                        'saturation': 0.6,
                    },
                    {  # 通常ケース
                        'width': 800.0, 'height': 600.0, 'aspect_ratio': 1.33,
                        'total_pixels': 480000.0, 'brightness': 0.5, 'contrast': 0.3,
                        'sharpness': 0.4, 'edge_density': 0.05, 'noise_level': 0.3,
                        'saturation': 0.6,
                    },
                    {  # 境界ケース: 低品質
                        'width': 800.0, 'height': 600.0, 'aspect_ratio': 1.33,
                        'total_pixels': 480000.0, 'brightness': 0.1, 'contrast': 0.05,
                        'sharpness': 0.05, 'edge_density': 0.05, 'noise_level': 0.8,
                        'saturation': 0.6,
                    },
                ]
                
                # 処理実行
                result = detector.process_directory(temp_path)
                
                # 結果検証
                assert result.total_images == 3
                assert len(result.boundary_cases) == 2  # 2つの境界ケース
                assert result.summary_statistics['total_processed'] == 3
                assert result.summary_statistics['boundary_cases_found'] == 2
                
                # 境界ケースの詳細検証
                case_types = [case.case_type for case in result.boundary_cases]
                assert BoundaryType.POSE_BOUNDARY in case_types
                assert BoundaryType.QUALITY_BOUNDARY in case_types
                
                # 改善優先度の存在確認
                assert len(result.improvement_priorities) > 0
                
    def test_save_results(self, detector):
        """結果保存テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # テスト用結果作成
            boundary_case = BoundaryCase(
                image_path="test.jpg",
                case_type=BoundaryType.POSE_BOUNDARY,
                confidence_score=0.8,
                quality_metrics={'aspect_ratio': 0.2},
                detection_reason="テスト境界ケース",
                improvement_suggestions=["改善提案"],
                technical_details={'version': '1.0'}
            )
            
            result = BoundaryAnalysisResult(
                total_images=1,
                boundary_cases=[boundary_case],
                summary_statistics={
                    'total_processed': 1, 
                    'boundary_cases_found': 1,
                    'case_type_counts': {'pose_boundary': 1}
                },
                improvement_priorities=[]
            )
            
            # 保存実行
            result_file = detector.save_results(result, output_dir)
            
            # ファイル存在確認
            assert result_file.exists()
            assert result_file.name == "TEST-INTEGRATION_boundary_analysis.json"
            
            # JSON内容確認
            with open(result_file, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
                
            assert saved_data['tracker_id'] == "TEST-INTEGRATION"
            assert saved_data['total_images'] == 1
            assert saved_data['boundary_cases_count'] == 1
            assert len(saved_data['boundary_cases']) == 1
            
            # サマリーファイル確認
            summary_file = output_dir / "TEST-INTEGRATION_boundary_summary.md"
            assert summary_file.exists()
            
            summary_content = summary_file.read_text(encoding='utf-8')
            assert "境界例自動検出レポート" in summary_content
            assert "TEST-INTEGRATION" in summary_content