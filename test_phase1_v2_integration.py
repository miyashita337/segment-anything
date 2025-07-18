#!/usr/bin/env python3
"""
Phase 1 v2 統合テスト - v0.3.4準備
新規実装5システムの統合動作テスト

P1-018: 滑らかさ評価指標の実装
P1-020: 切断検出アルゴリズム
P1-022: 混入率定量化システム
P1-016: フィードバックループ構築
P1-010: 効率的サンプリングアルゴリズム
"""

import sys
import os
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 新システムのインポート
from features.evaluation.utils.smoothness_metrics import SmoothnessMetrics
from features.evaluation.utils.truncation_detector import TruncationDetector
from features.evaluation.utils.contamination_quantifier import ContaminationQuantifier
from features.evaluation.utils.feedback_loop_system import FeedbackLoopSystem
from features.evaluation.utils.efficient_sampling import EfficientSampling


class Phase1V2IntegrationTest:
    """Phase 1 v2システム統合テスト"""
    
    def __init__(self):
        """初期化"""
        self.test_results = {}
        self.test_start_time = datetime.now()
        
        print("🚀 Phase 1 v2 システム統合テスト開始")
        print(f"テスト開始時刻: {self.test_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("新規実装システム: P1-018, P1-020, P1-022, P1-016, P1-010")
    
    def create_comprehensive_test_data(self) -> Dict[str, Any]:
        """包括的テストデータ作成"""
        # テスト画像（アニメキャラクター風）
        test_image = np.zeros((300, 250, 3), dtype=np.uint8)
        
        # 背景（青空）
        test_image[:, :] = [135, 206, 235]
        
        # キャラクター（前景）
        # 頭部（肌色の円）
        cv2.circle(test_image, (125, 60), 35, [255, 220, 177], -1)
        
        # 髪（茶色）
        cv2.ellipse(test_image, (125, 50), (40, 25), 0, 0, 180, [139, 69, 19], -1)
        
        # 胴体（シャツ - 白）
        cv2.rectangle(test_image, (90, 95), (160, 180), [255, 255, 255], -1)
        
        # 脚部（ズボン - 青、一部切断）
        cv2.rectangle(test_image, (100, 180), (150, 280), [0, 0, 139], -1)
        
        # マスク作成（キャラクター領域）
        test_mask = np.zeros((300, 250), dtype=np.uint8)
        cv2.circle(test_mask, (125, 60), 35, 255, -1)  # 頭
        cv2.ellipse(test_mask, (125, 50), (40, 25), 0, 0, 180, 255, -1)  # 髪
        cv2.rectangle(test_mask, (90, 95), (160, 180), 255, -1)  # 胴体
        cv2.rectangle(test_mask, (100, 180), (150, 280), 255, -1)  # 脚
        
        # サンプリング用候補データ
        candidate_data = []
        for i in range(15):
            candidate = {
                'id': f'integration_test_{i:03d}',
                'image_path': f'/test/integration/image_{i:03d}.jpg',
                'quality_score': 0.3 + (i % 7) * 0.1,
                'confidence_score': 0.4 + (i % 6) * 0.1,
                'processing_time': 2.0 + (i % 4) * 1.5,
                'complexity_score': 0.2 + (i % 5) * 0.15,
                'characteristics': {
                    'scene_type': 'test_scene',
                    'character_count': 1 + (i % 3)
                }
            }
            candidate_data.append(candidate)
        
        # 評価・処理結果データ
        evaluation_data = {
            'user_rating': 0.72,
            'comments': 'Integration test evaluation',
            'aspects': {'quality': 0.75, 'completeness': 0.68, 'accuracy': 0.74}
        }
        
        processing_results = {
            'quality_score': 0.65,
            'processing_time': 4.1,
            'success': True,
            'quality_method': 'balanced',
            'enhancement_applied': ['contrast_enhancement', 'boundary_smoothing'],
            'yolo_score_threshold': 0.07,
            'confidence_score': 0.71,
            'boundary_quality': 0.68
        }
        
        return {
            'test_image': test_image,
            'test_mask': test_mask,
            'candidate_data': candidate_data,
            'evaluation_data': evaluation_data,
            'processing_results': processing_results
        }
    
    def test_smoothness_metrics(self, test_data: Dict) -> Dict[str, Any]:
        """P1-018: 滑らかさ評価指標のテスト"""
        print("\n📊 [P1-018] 滑らかさ評価指標テスト...")
        
        try:
            analyzer = SmoothnessMetrics()
            result = analyzer.analyze_boundary_smoothness(test_data['test_mask'])
            
            if 'error' not in result:
                overall = result.get('overall_assessment', {})
                return {
                    'status': 'success',
                    'overall_score': overall.get('overall_smoothness_score', 0),
                    'smoothness_grade': overall.get('smoothness_grade', 'F'),
                    'confidence': overall.get('confidence', 0),
                    'available_metrics': overall.get('available_metrics', 0)
                }
            else:
                return {'status': 'error', 'error': result['error']}
                
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def test_truncation_detector(self, test_data: Dict) -> Dict[str, Any]:
        """P1-020: 切断検出アルゴリズムのテスト"""
        print("\n📊 [P1-020] 切断検出アルゴリズムテスト...")
        
        try:
            detector = TruncationDetector()
            result = detector.detect_truncation(test_data['test_mask'], (300, 250))
            
            if 'error' not in result:
                overall = result.get('overall_assessment', {})
                return {
                    'status': 'success',
                    'truncation_score': overall.get('overall_truncation_score', 0),
                    'truncation_grade': overall.get('truncation_grade', 'F'),
                    'severity': overall.get('severity_assessment', 'unknown'),
                    'recovery_suggestions': len(result.get('recovery_suggestions', []))
                }
            else:
                return {'status': 'error', 'error': result['error']}
                
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def test_contamination_quantifier(self, test_data: Dict) -> Dict[str, Any]:
        """P1-022: 混入率定量化システムのテスト"""
        print("\n📊 [P1-022] 混入率定量化システムテスト...")
        
        try:
            quantifier = ContaminationQuantifier()
            result = quantifier.quantify_contamination(test_data['test_image'], test_data['test_mask'])
            
            if 'error' not in result:
                overall = result.get('overall_assessment', {})
                return {
                    'status': 'success',
                    'contamination_score': overall.get('overall_contamination_score', 0),
                    'contamination_grade': overall.get('contamination_grade', 'F'),
                    'confidence': overall.get('confidence', 0),
                    'available_metrics': overall.get('available_metrics', 0)
                }
            else:
                return {'status': 'error', 'error': result['error']}
                
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def test_feedback_loop_system(self, test_data: Dict) -> Dict[str, Any]:
        """P1-016: フィードバックループ構築のテスト"""
        print("\n📊 [P1-016] フィードバックループシステムテスト...")
        
        try:
            feedback_system = FeedbackLoopSystem()
            result = feedback_system.integrate_evaluation_feedback(
                test_data['evaluation_data'], 
                test_data['processing_results']
            )
            
            if 'error' not in result:
                integration = result.get('feedback_integration', {})
                learning = result.get('learning_effectiveness', {})
                return {
                    'status': 'success',
                    'integration_success': integration.get('integration_success', False),
                    'entry_id': integration.get('entry_id', 'N/A'),
                    'learning_effectiveness': learning.get('effectiveness', 'unknown'),
                    'recommendations_count': len(result.get('improvement_recommendations', []))
                }
            else:
                return {'status': 'error', 'error': result['error']}
                
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def test_efficient_sampling(self, test_data: Dict) -> Dict[str, Any]:
        """P1-010: 効率的サンプリングアルゴリズムのテスト"""
        print("\n📊 [P1-010] 効率的サンプリングアルゴリズムテスト...")
        
        try:
            sampler = EfficientSampling()
            result = sampler.generate_sampling_strategy(
                test_data['candidate_data'], 
                target_samples=6, 
                strategy='hybrid'
            )
            
            if 'error' not in result:
                return {
                    'status': 'success',
                    'strategy': result.get('sampling_strategy', 'unknown'),
                    'target_samples': result.get('target_samples', 0),
                    'actual_samples': result.get('actual_samples', 0),
                    'candidate_pool': result.get('candidate_pool_size', 0),
                    'optimization_suggestions': len(result.get('optimization_suggestions', []))
                }
            else:
                return {'status': 'error', 'error': result['error']}
                
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def run_integration_test(self) -> Dict[str, Any]:
        """統合テスト実行"""
        print("\n" + "="*70)
        print("🔄 Phase 1 v2 システム統合テスト実行中...")
        print("="*70)
        
        # テストデータ作成
        test_data = self.create_comprehensive_test_data()
        
        # 各システムのテスト実行
        self.test_results = {
            'smoothness_metrics': self.test_smoothness_metrics(test_data),
            'truncation_detector': self.test_truncation_detector(test_data),
            'contamination_quantifier': self.test_contamination_quantifier(test_data),
            'feedback_loop_system': self.test_feedback_loop_system(test_data),
            'efficient_sampling': self.test_efficient_sampling(test_data)
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
            'test_duration': (datetime.now() - self.test_start_time).total_seconds(),
            'v034_readiness': success_count >= 4  # 80%以上で準備完了
        }
        
        return integration_result
    
    def print_test_summary(self, integration_result: Dict):
        """テスト結果サマリー出力"""
        print("\n" + "="*70)
        print("📋 Phase 1 v2 統合テスト結果サマリー")
        print("="*70)
        
        print(f"\n📊 統合テスト結果:")
        print(f"  成功システム: {integration_result['successful_systems']}/{integration_result['total_systems']}")
        print(f"  成功率: {integration_result['success_rate']:.1%}")
        print(f"  実行時間: {integration_result['test_duration']:.2f}秒")
        print(f"  総合ステータス: {integration_result['overall_status']}")
        
        print(f"\n🔍 個別システム結果:")
        for system_name, result in self.test_results.items():
            status_icon = "✅" if result.get('status') == 'success' else "❌"
            status = result.get('status', 'unknown')
            print(f"  {status_icon} {system_name}: {status}")
            
            if result.get('status') == 'success':
                # システム固有の主要指標表示
                if system_name == 'smoothness_metrics':
                    print(f"     滑らかさスコア: {result.get('overall_score', 0):.3f} (グレード: {result.get('smoothness_grade', 'N/A')})")
                elif system_name == 'truncation_detector':
                    print(f"     切断スコア: {result.get('truncation_score', 0):.3f} (重要度: {result.get('severity', 'N/A')})")
                elif system_name == 'contamination_quantifier':
                    print(f"     混入スコア: {result.get('contamination_score', 0):.3f} (グレード: {result.get('contamination_grade', 'N/A')})")
                elif system_name == 'feedback_loop_system':
                    print(f"     フィードバック統合: {result.get('integration_success', False)} (効果: {result.get('learning_effectiveness', 'N/A')})")
                elif system_name == 'efficient_sampling':
                    print(f"     サンプリング: {result.get('actual_samples', 0)}/{result.get('target_samples', 0)}件 (戦略: {result.get('strategy', 'N/A')})")
            elif result.get('status') == 'error':
                print(f"     エラー: {result.get('error', 'Unknown error')}")
        
        print(f"\n🎯 v0.3.4リリース準備状況:")
        if integration_result['v034_readiness']:
            print(f"  ✅ 統合テスト合格 - v0.3.4リリース準備完了")
            print(f"  📦 次ステップ: kaname09バッチテスト実行")
        else:
            print(f"  ⚠️ 要修正 - 統合テスト不完全")
            print(f"  🔧 次ステップ: エラー修正 & 再テスト")
        
        print(f"\n🏆 Phase 1進捗更新:")
        print(f"  v0.3.3: 13/22タスク完了 (59%)")
        print(f"  v0.3.4: 18/22タスク完了 (82%) ← 新規5タスク追加")
        print(f"  残りタスク: 4タスクでPhase 1完了")


def main():
    """メイン実行関数"""
    tester = Phase1V2IntegrationTest()
    
    # 統合テスト実行
    integration_result = tester.run_integration_test()
    
    # 結果出力
    tester.print_test_summary(integration_result)
    
    print(f"\n✅ Phase 1 v2統合テスト完了")
    return integration_result['overall_status'] == 'success'


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)