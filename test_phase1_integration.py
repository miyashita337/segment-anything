#!/usr/bin/env python3
"""
Phase 1 Integration Test - v0.3.3
Phase 1品質評価システム統合テスト

5つの新システムの統合動作テスト
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 新システムのインポート
from features.evaluation.utils.learning_data_collection import LearningDataCollectionPlanner
from features.evaluation.utils.evaluation_difference_analyzer import EvaluationDifferenceAnalyzer
from features.evaluation.utils.boundary_analysis import BoundaryAnalyzer
from features.evaluation.utils.human_structure_recognition import HumanStructureRecognizer
from features.evaluation.utils.foreground_background_analyzer import ForegroundBackgroundAnalyzer


class Phase1IntegrationTest:
    """Phase 1システム統合テスト"""
    
    def __init__(self):
        """初期化"""
        self.test_results = {}
        self.test_start_time = datetime.now()
        
        print("🚀 Phase 1品質評価システム統合テスト開始")
        print(f"テスト開始時刻: {self.test_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def create_test_image_and_mask(self) -> tuple:
        """テスト用画像・マスクの作成"""
        # テスト画像（アニメキャラクター風）
        test_image = np.zeros((200, 150, 3), dtype=np.uint8)
        
        # 背景（青空）
        test_image[:, :] = [135, 206, 235]  # SkyBlue
        
        # キャラクター（前景）
        # 頭部（肌色の円）
        cv2.circle(test_image, (75, 40), 25, [255, 220, 177], -1)
        
        # 髪（茶色）
        cv2.ellipse(test_image, (75, 35), (30, 20), 0, 0, 180, [139, 69, 19], -1)
        
        # 胴体（シャツ - 白）
        cv2.rectangle(test_image, (50, 65), (100, 120), [255, 255, 255], -1)
        
        # 脚部（ズボン - 青）
        cv2.rectangle(test_image, (55, 120), (95, 180), [0, 0, 139], -1)
        
        # マスク作成（キャラクター領域）
        test_mask = np.zeros((200, 150), dtype=np.uint8)
        cv2.circle(test_mask, (75, 40), 25, 255, -1)  # 頭
        cv2.ellipse(test_mask, (75, 35), (30, 20), 0, 0, 180, 255, -1)  # 髪
        cv2.rectangle(test_mask, (50, 65), (100, 120), 255, -1)  # 胴体
        cv2.rectangle(test_mask, (55, 120), (95, 180), 255, -1)  # 脚
        
        return test_image, test_mask
    
    def test_learning_data_collection(self) -> dict:
        """P1-009: 学習データ収集計画策定のテスト"""
        print("\n📊 [P1-009] 学習データ収集計画策定テスト...")
        
        try:
            planner = LearningDataCollectionPlanner()
            
            if planner.evaluation_data:
                analysis = planner.analyze_existing_data()
                gaps = planner.identify_data_gaps()
                strategy = planner.generate_collection_strategy()
                
                result = {
                    'status': 'success',
                    'data_count': len(planner.evaluation_data),
                    'success_rate': analysis.get('basic_stats', {}).get('success_rate', 0),
                    'identified_gaps': len([g for gaps_list in gaps.values() for g in gaps_list]),
                    'target_samples': strategy.get('sampling_strategy', {}).get('target_total_samples', 0)
                }
                print(f"  ✅ 成功: {result['data_count']}件分析、目標{result['target_samples']}件")
            else:
                result = {
                    'status': 'no_data',
                    'message': 'テストデータなし（正常動作）'
                }
                print(f"  ⚠️ 評価データなし（テスト環境では正常）")
            
        except Exception as e:
            result = {'status': 'error', 'error': str(e)}
            print(f"  ❌ エラー: {e}")
        
        return result
    
    def test_evaluation_difference_analyzer(self) -> dict:
        """P1-015: 評価差分の定量化のテスト"""
        print("\n📊 [P1-015] 評価差分の定量化テスト...")
        
        try:
            analyzer = EvaluationDifferenceAnalyzer()
            
            if analyzer.evaluation_data:
                correlations = analyzer.calculate_correlations()
                patterns = analyzer.analyze_difference_patterns()
                recommendations = analyzer.generate_improvement_recommendations()
                
                result = {
                    'status': 'success',
                    'sample_count': correlations.get('sample_count', 0),
                    'correlation': correlations.get('pearson_correlation', 0),
                    'recommendations_count': len(recommendations)
                }
                print(f"  ✅ 成功: {result['sample_count']}サンプル、相関{result['correlation']:.3f}")
            else:
                result = {
                    'status': 'no_data',
                    'message': 'テストデータなし（正常動作）'
                }
                print(f"  ⚠️ 評価データなし（テスト環境では正常）")
            
        except Exception as e:
            result = {'status': 'error', 'error': str(e)}
            print(f"  ❌ エラー: {e}")
        
        return result
    
    def test_boundary_analysis(self) -> dict:
        """P1-017: 境界線解析アルゴリズムのテスト"""
        print("\n📊 [P1-017] 境界線解析アルゴリズムテスト...")
        
        try:
            _, test_mask = self.create_test_image_and_mask()
            
            analyzer = BoundaryAnalyzer()
            quality_result = analyzer.calculate_boundary_quality_score(test_mask)
            
            result = {
                'status': 'success',
                'overall_score': quality_result['overall_score'],
                'quality_grade': quality_result['quality_grade'],
                'contour_count': quality_result['contour_count'],
                'boundary_pixels': quality_result['boundary_pixel_count']
            }
            print(f"  ✅ 成功: スコア{result['overall_score']:.3f}、グレード{result['quality_grade']}")
            
        except Exception as e:
            result = {'status': 'error', 'error': str(e)}
            print(f"  ❌ エラー: {e}")
        
        return result
    
    def test_human_structure_recognition(self) -> dict:
        """P1-019: 人体構造認識システムのテスト"""
        print("\n📊 [P1-019] 人体構造認識システムテスト...")
        
        try:
            _, test_mask = self.create_test_image_and_mask()
            
            recognizer = HumanStructureRecognizer()
            analysis_result = recognizer.analyze_mask_structure(test_mask)
            
            basic = analysis_result.get('basic_analysis', {})
            truncation = analysis_result.get('truncation_risk', {})
            overall = analysis_result.get('overall_assessment', {})
            
            result = {
                'status': 'success',
                'aspect_ratio': basic.get('aspect_ratio', 0),
                'detected_regions': len(analysis_result.get('body_regions', [])),
                'truncation_risk': truncation.get('overall_severity', 'unknown'),
                'overall_grade': overall.get('overall_grade', 'unknown')
            }
            print(f"  ✅ 成功: {result['detected_regions']}部位検出、グレード{result['overall_grade']}")
            
        except Exception as e:
            result = {'status': 'error', 'error': str(e)}
            print(f"  ❌ エラー: {e}")
        
        return result
    
    def test_foreground_background_analyzer(self) -> dict:
        """P1-021: 背景・前景分離精度測定のテスト"""
        print("\n📊 [P1-021] 背景・前景分離精度測定テスト...")
        
        try:
            test_image, test_mask = self.create_test_image_and_mask()
            
            analyzer = ForegroundBackgroundAnalyzer()
            analysis_result = analyzer.analyze_separation_quality(test_image, test_mask)
            
            separation_score = analysis_result.get('separation_score', {})
            contamination = analysis_result.get('contamination_analysis', {})
            assessment = analysis_result.get('overall_assessment', {})
            
            result = {
                'status': 'success',
                'overall_score': separation_score.get('overall_score', 0),
                'quality_grade': separation_score.get('quality_grade', 'unknown'),
                'contamination_level': contamination.get('contamination_level', 'unknown'),
                'extraction_reliability': assessment.get('extraction_reliability', 'unknown')
            }
            print(f"  ✅ 成功: スコア{result['overall_score']:.3f}、グレード{result['quality_grade']}")
            
        except Exception as e:
            result = {'status': 'error', 'error': str(e)}
            print(f"  ❌ エラー: {e}")
        
        return result
    
    def run_integration_test(self) -> dict:
        """統合テストの実行"""
        print("\n" + "="*60)
        print("🔄 Phase 1システム統合テスト実行中...")
        print("="*60)
        
        # 各システムのテスト実行
        self.test_results = {
            'learning_data_collection': self.test_learning_data_collection(),
            'evaluation_difference_analyzer': self.test_evaluation_difference_analyzer(),
            'boundary_analysis': self.test_boundary_analysis(),
            'human_structure_recognition': self.test_human_structure_recognition(),
            'foreground_background_analyzer': self.test_foreground_background_analyzer()
        }
        
        # 統合結果の評価
        success_count = sum(1 for result in self.test_results.values() 
                           if result.get('status') == 'success')
        total_count = len(self.test_results)
        
        integration_result = {
            'overall_status': 'success' if success_count == total_count else 'partial_success',
            'success_rate': success_count / total_count,
            'successful_systems': success_count,
            'total_systems': total_count,
            'test_duration': (datetime.now() - self.test_start_time).total_seconds()
        }
        
        return integration_result
    
    def print_test_summary(self, integration_result: dict):
        """テスト結果サマリーの出力"""
        print("\n" + "="*60)
        print("📋 Phase 1統合テスト結果サマリー")
        print("="*60)
        
        print(f"\n📊 統合テスト結果:")
        print(f"  成功システム: {integration_result['successful_systems']}/{integration_result['total_systems']}")
        print(f"  成功率: {integration_result['success_rate']:.1%}")
        print(f"  実行時間: {integration_result['test_duration']:.2f}秒")
        print(f"  総合ステータス: {integration_result['overall_status']}")
        
        print(f"\n🔍 個別システム結果:")
        for system_name, result in self.test_results.items():
            status_icon = "✅" if result.get('status') == 'success' else "⚠️" if result.get('status') == 'no_data' else "❌"
            print(f"  {status_icon} {system_name}: {result.get('status', 'unknown')}")
        
        print(f"\n🎯 v0.3.3リリース準備状況:")
        if integration_result['success_rate'] >= 0.8:
            print(f"  ✅ 統合テスト合格 - リリース可能")
            print(f"  📦 次ステップ: バッチ処理テスト & リリース作業")
        else:
            print(f"  ⚠️ 要修正 - 統合テスト不完全")
            print(f"  🔧 次ステップ: エラー修正 & 再テスト")


def main():
    """メイン実行関数"""
    tester = Phase1IntegrationTest()
    
    # 統合テスト実行
    integration_result = tester.run_integration_test()
    
    # 結果出力
    tester.print_test_summary(integration_result)
    
    print(f"\n✅ Phase 1統合テスト完了")
    return integration_result['overall_status'] == 'success'


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)