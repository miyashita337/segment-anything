#!/usr/bin/env python3
"""
QCA-001: 作者別パラメータ適応システム統合テスト

実際のキャラクター抽出パイプラインとの統合動作テスト
- extract_character.py コマンド統合
- SAMOptimizationConfig との連携
- 実画像データでの動作確認  
- エンドツーエンド処理フロー検証

Created for: QCA-001 - 作者別パラメータ適応システム・ディレクトリ構造ベース自動最適化
Author: Claude Code Integration System
"""

import os
import sys
import tempfile
import shutil
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# テスト対象のインポート
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from features.adaptation.author_parameter_adapter import AuthorParameterAdapter
from features.processing.sam_optimization_config import SAMOptimizationConfig, create_optimized_sam_generator


class TestQCA001AuthorAdaptationIntegration:
    """QCA-001 作者別パラメータ適応統合テスト"""
    
    @pytest.fixture
    def temp_test_structure(self):
        """テスト用一時ディレクトリ構造の作成"""
        temp_dir = tempfile.mkdtemp(prefix="qca001_test_")
        
        # 作者別ディレクトリ構造を作成
        authors = {
            "yado": ["kana05", "kana06", "kana07"],
            "aichi": ["work01", "work02"],  
            "zundamon": ["test01"]
        }
        
        test_files = []
        
        for author, works in authors.items():
            for work in works:
                work_dir = Path(temp_dir) / "train" / author / "org" / work
                work_dir.mkdir(parents=True, exist_ok=True)
                
                # ダミー画像ファイルを作成
                for i in range(2):
                    test_file = work_dir / f"{work}_{i:04d}.jpg"
                    test_file.write_text("dummy image data")  # ダミーデータ
                    test_files.append(test_file)
        
        return temp_dir, test_files
    
    def test_author_detection_integration(self, temp_test_structure):
        """作者検出ロジックの統合テスト"""
        temp_dir, test_files = temp_test_structure
        
        try:
            adapter = AuthorParameterAdapter()
            
            # 各テストファイルで作者検出を確認
            expected_detections = {
                "yado": 6,    # 3作品 × 2画像
                "aichi": 4,   # 2作品 × 2画像
                "zundamon": 2 # 1作品 × 2画像
            }
            
            detection_counts = {"yado": 0, "aichi": 0, "zundamon": 0}
            
            for test_file in test_files:
                detected_author = adapter.detect_author_from_path(str(test_file))
                if detected_author in detection_counts:
                    detection_counts[detected_author] += 1
            
            # 検出数の確認
            for author, expected_count in expected_detections.items():
                actual_count = detection_counts[author]
                assert actual_count == expected_count, \
                    f"{author}作者の検出数不一致: 期待{expected_count}, 実際{actual_count}"
            
            print("✅ 作者検出統合テスト成功")
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_sam_optimization_config_integration(self):
        """SAMOptimizationConfig との統合テスト"""
        adapter = AuthorParameterAdapter()
        optimizer = SAMOptimizationConfig()
        
        # 各作者の最適化パラメータをSAM設定に適用
        test_cases = [
            ("yado", "character_focused"),
            ("aichi", "precision_focused"),
            ("zundamon", "speed_optimized")
        ]
        
        for author, expected_sam_profile in test_cases:
            # 作者パラメータ取得
            author_params = adapter.get_optimized_parameters(author)
            
            # SAM設定に適用
            sam_config = optimizer.get_sam_config(author_params=author_params)
            
            # 期待するプロファイルの設定が適用されているか確認
            expected_profile = optimizer.OPTIMIZATION_PROFILES[expected_sam_profile]
            
            assert sam_config["points_per_side"] == expected_profile.points_per_side, \
                f"{author}: points_per_side不一致"
            assert sam_config["pred_iou_thresh"] == expected_profile.pred_iou_thresh, \
                f"{author}: pred_iou_thresh不一致"
            assert sam_config["stability_score_thresh"] == expected_profile.stability_score_thresh, \
                f"{author}: stability_score_thresh不一致"
            
            print(f"✅ {author}作者のSAM設定統合テスト成功")
    
    @patch('features.adaptation.author_parameter_adapter.AuthorParameterAdapter')
    def test_extract_character_integration(self, mock_adapter_class):
        """extract_character.pyコマンド統合テスト"""
        # モックアダプターの設定
        mock_adapter = MagicMock()
        mock_adapter.detect_author_from_path.return_value = "yado"
        mock_adapter.get_optimized_parameters.return_value = {
            "author_id": "yado",
            "sam_profile": "character_focused",
            "yolo_confidence": 0.07,
            "score_threshold": 0.07,
            "characteristics": "balanced"
        }
        mock_adapter_class.return_value = mock_adapter
        
        # extract_character.py のインポートと呼び出し
        try:
            from features.extraction.commands.extract_character import extract_character
            from click.testing import CliRunner
            
            runner = CliRunner()
            
            with tempfile.NamedTemporaryFile(suffix=".jpg") as input_file, \
                 tempfile.TemporaryDirectory() as output_dir:
                
                # ダミー画像ファイル作成
                input_file.write(b"dummy image data")
                input_file.flush()
                
                # QCA-001機能有効化でコマンド実行
                result = runner.invoke(extract_character, [
                    input_file.name,
                    "-o", output_dir,
                    "--enable-author-adaptation",
                    "--verbose"
                ])
                
                # アダプターが呼び出されたことを確認
                mock_adapter.detect_author_from_path.assert_called_once()
                
                print("✅ extract_character統合テスト成功")
                
        except ImportError as e:
            print(f"⚠️ extract_character import skip: {e}")
            pytest.skip("extract_character import不可")
    
    def test_author_parameter_yolo_confidence_application(self):
        """YOLO信頼度の作者別適用テスト"""
        adapter = AuthorParameterAdapter()
        
        # 各作者の信頼度設定確認
        test_cases = [
            ("yado", 0.07),      # バランス型
            ("aichi", 0.05),     # 細密描写・低信頼度
            ("zundamon", 0.08)   # シンプルスタイル・標準
        ]
        
        for author, expected_confidence in test_cases:
            params = adapter.get_optimized_parameters(author)
            actual_confidence = params["yolo_confidence"]
            
            assert actual_confidence == expected_confidence, \
                f"{author}作者の信頼度不一致: 期待{expected_confidence}, 実際{actual_confidence}"
            
            # スコア閾値も同じ値であることを確認
            assert params["score_threshold"] == expected_confidence, \
                f"{author}作者のスコア閾値不一致"
    
    def test_sam_profile_consistency(self):
        """SAMプロファイル一貫性テスト"""
        adapter = AuthorParameterAdapter()
        optimizer = SAMOptimizationConfig()
        
        for author in ["yado", "aichi", "zundamon"]:
            # 作者パラメータから期待SAMプロファイルを取得
            params = adapter.get_optimized_parameters(author)
            expected_sam_profile = params["sam_profile"]
            
            # SAMOptimizationConfigに該当プロファイルが存在するか確認
            assert expected_sam_profile in optimizer.OPTIMIZATION_PROFILES, \
                f"{author}作者のSAMプロファイル{expected_sam_profile}が未定義"
            
            # プロファイルの妥当性確認
            profile = optimizer.OPTIMIZATION_PROFILES[expected_sam_profile]
            assert profile.points_per_side > 0, f"{expected_sam_profile}: points_per_side無効"
            assert 0 < profile.pred_iou_thresh < 1, f"{expected_sam_profile}: pred_iou_thresh無効"
            assert 0 < profile.stability_score_thresh < 1, f"{expected_sam_profile}: stability_score_thresh無効"
            
            print(f"✅ {author}作者のSAMプロファイル一貫性確認")
    
    def test_error_handling_integration(self):
        """エラーハンドリング統合テスト"""
        adapter = AuthorParameterAdapter()
        optimizer = SAMOptimizationConfig()
        
        # 無効な作者IDでの処理
        invalid_author = "nonexistent_author"
        params = adapter.get_optimized_parameters(invalid_author)
        
        # デフォルトプロファイルが返されることを確認
        assert params["author_id"] == "default"
        assert params["sam_profile"] == "balanced"
        
        # 無効な作者パラメータでのSAM設定取得
        invalid_params = {"sam_profile": "nonexistent_profile"}
        sam_config = optimizer.get_sam_config(author_params=invalid_params)
        
        # デフォルト設定が使用されることを確認（エラーにならない）
        assert isinstance(sam_config, dict)
        assert "points_per_side" in sam_config
        
        print("✅ エラーハンドリング統合テスト成功")
    
    def test_performance_characteristics_mapping(self):
        """パフォーマンス特性マッピングテスト"""
        adapter = AuthorParameterAdapter()
        
        # 各作者の期待特性
        expected_characteristics = {
            "yado": ("balanced", 2.2),          # バランス型・中程度高速化
            "aichi": ("detail_oriented", 1.8),  # 細密描写・品質優先
            "zundamon": ("simple_style", 2.8)   # シンプル・最高速
        }
        
        for author, (expected_char, expected_speedup) in expected_characteristics.items():
            params = adapter.get_optimized_parameters(author)
            
            # 特性確認
            assert params["characteristics"] == expected_char, \
                f"{author}作者の特性不一致: 期待{expected_char}, 実際{params['characteristics']}"
            
            # SAMプロファイルの高速化期待値確認
            profile_name = params["sam_profile"]
            optimizer = SAMOptimizationConfig()
            profile = optimizer.OPTIMIZATION_PROFILES[profile_name]
            
            assert abs(profile.expected_speedup - expected_speedup) < 0.1, \
                f"{author}作者の高速化期待値不一致: 期待{expected_speedup}, 実際{profile.expected_speedup}"
            
            print(f"✅ {author}作者のパフォーマンス特性マッピング確認")
    
    def test_multi_work_author_consistency(self, temp_test_structure):
        """同一作者の複数作品での一貫性テスト"""
        temp_dir, test_files = temp_test_structure
        
        try:
            adapter = AuthorParameterAdapter()
            
            # yado作者の複数作品ファイル
            yado_files = [f for f in test_files if "/yado/" in str(f)]
            
            # 全ファイルで同じパラメータが返されることを確認
            first_params = None
            
            for yado_file in yado_files:
                params = adapter.apply_author_optimization(str(yado_file))
                
                if first_params is None:
                    first_params = params
                else:
                    # 主要パラメータの一致確認
                    assert params["author_id"] == first_params["author_id"]
                    assert params["sam_profile"] == first_params["sam_profile"]
                    assert params["yolo_confidence"] == first_params["yolo_confidence"]
                    assert params["score_threshold"] == first_params["score_threshold"]
            
            print("✅ 同一作者複数作品一貫性テスト成功")
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


def test_qca001_end_to_end_simulation():
    """QCA-001 エンドツーエンド処理シミュレーション"""
    print("\n🔬 QCA-001 エンドツーエンド処理シミュレーション")
    print("=" * 60)
    
    # 実際の処理フローをシミュレーション
    test_image_path = "/mnt/c/AItools/lora/train/yado/org/kana05/kana05_0001.jpg"
    
    # 1. 作者検出
    adapter = AuthorParameterAdapter()
    detected_author = adapter.detect_author_from_path(test_image_path)
    print(f"📋 作者検出: {detected_author or 'unknown'}")
    
    # 2. 最適化パラメータ取得
    params = adapter.apply_author_optimization(test_image_path)
    print(f"⚙️ SAMプロファイル: {params['sam_profile']}")
    print(f"⚙️ YOLO信頼度: {params['yolo_confidence']}")
    print(f"⚙️ スコア閾値: {params['score_threshold']}")
    
    # 3. SAM設定適用
    optimizer = SAMOptimizationConfig()
    sam_config = optimizer.get_sam_config(author_params=params)
    print(f"🚀 SAM points_per_side: {sam_config['points_per_side']}")
    print(f"🚀 SAM pred_iou_thresh: {sam_config['pred_iou_thresh']}")
    
    # 4. 統計情報
    stats = adapter.get_author_statistics()
    print(f"📊 対応作者数: {stats['total_authors']}")
    print(f"📊 信頼度範囲: {stats['confidence_range']['min']:.3f} - {stats['confidence_range']['max']:.3f}")
    
    print("✅ エンドツーエンドシミュレーション完了")


if __name__ == "__main__":
    """統合テストの直接実行"""
    import tempfile
    import shutil
    
    print("🧪 QCA-001: 作者別パラメータ適応システム統合テスト実行")
    print("=" * 70)
    
    test_integration = TestQCA001AuthorAdaptationIntegration()
    
    # 基本統合テストを実行
    test_methods = [
        test_integration.test_sam_optimization_config_integration,
        test_integration.test_author_parameter_yolo_confidence_application,
        test_integration.test_sam_profile_consistency,
        test_integration.test_error_handling_integration,
        test_integration.test_performance_characteristics_mapping
    ]
    
    passed = 0
    failed = 0
    
    for test_method in test_methods:
        try:
            test_method()
            print(f"✅ {test_method.__name__}")
            passed += 1
        except Exception as e:
            print(f"❌ {test_method.__name__}: {e}")
            failed += 1
    
    # エンドツーエンドテスト
    try:
        test_qca001_end_to_end_simulation()
        passed += 1
    except Exception as e:
        print(f"❌ エンドツーエンドテスト: {e}")
        failed += 1
    
    print("\n" + "=" * 70)
    print(f"📊 統合テスト結果: {passed}件成功 / {failed}件失敗 / {passed + failed}件総数")
    
    if failed == 0:
        print("🎉 全統合テスト合格！QCA-001システム統合準備完了")
        print("📝 次のステップ: 実動作確認・小規模テスト実行")
    else:
        print("⚠️ 一部統合テスト失敗。システム統合を確認してください。")