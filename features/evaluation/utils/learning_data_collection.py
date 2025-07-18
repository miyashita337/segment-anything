#!/usr/bin/env python3
"""
Learning Data Collection Plan - P1-009
学習データ収集計画策定システム

既存のユーザー評価データを拡張し、効率的な学習データ収集戦略を確立
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import numpy as np
from collections import defaultdict, Counter


class LearningDataCollectionPlanner:
    """
    学習データ収集計画策定システム
    
    既存の評価データを分析し、効率的な追加データ収集戦略を策定
    """
    
    def __init__(self, evaluation_data_path: str = None):
        """
        初期化
        
        Args:
            evaluation_data_path: 既存評価データのパス
        """
        self.evaluation_data_path = evaluation_data_path or self._find_evaluation_data()
        self.evaluation_data = []
        self.analysis_results = {}
        
        if self.evaluation_data_path and os.path.exists(self.evaluation_data_path):
            self.load_evaluation_data()
    
    def _find_evaluation_data(self) -> Optional[str]:
        """既存の評価データファイルを検索"""
        possible_paths = [
            "features/evaluation/logs/kaname07_user_evaluation.jsonl",
            "logs/user_evaluation.jsonl",
            "evaluation_data.jsonl"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None
    
    def load_evaluation_data(self) -> bool:
        """評価データの読み込み"""
        try:
            with open(self.evaluation_data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.evaluation_data.append(json.loads(line))
            
            print(f"✅ 評価データ読み込み完了: {len(self.evaluation_data)}件")
            return True
            
        except Exception as e:
            print(f"❌ 評価データ読み込みエラー: {e}")
            return False
    
    def analyze_existing_data(self) -> Dict[str, Any]:
        """既存データの分析"""
        if not self.evaluation_data:
            return {"error": "評価データが見つかりません"}
        
        # 基本統計
        total_count = len(self.evaluation_data)
        success_count = sum(1 for item in self.evaluation_data if item.get('extraction_success', False))
        success_rate = success_count / total_count if total_count > 0 else 0
        
        # 評価分布
        rating_distribution = Counter()
        problem_distribution = Counter()
        region_distribution = Counter()
        
        for item in self.evaluation_data:
            # 評価分布
            rating = item.get('user_rating')
            if rating:
                rating_distribution[rating] += 1
            
            # 問題分布
            problem = item.get('actual_problem', 'unknown')
            problem_distribution[problem] += 1
            
            # 地域要求分布
            region = item.get('desired_region', 'unknown')
            if region != 'unknown' and region != 'success':
                region_distribution[region] += 1
        
        # 成功パターン分析
        success_patterns = []
        failure_patterns = []
        
        for item in self.evaluation_data:
            pattern = {
                'user_rating': item.get('user_rating'),
                'desired_region': item.get('desired_region'),
                'actual_problem': item.get('actual_problem'),
                'extraction_success': item.get('extraction_success', False)
            }
            
            if item.get('extraction_success', False) and item.get('user_rating') in ['A', 'B']:
                success_patterns.append(pattern)
            else:
                failure_patterns.append(pattern)
        
        self.analysis_results = {
            'basic_stats': {
                'total_count': total_count,
                'success_count': success_count,
                'success_rate': success_rate,
                'failure_count': total_count - success_count
            },
            'distributions': {
                'rating': dict(rating_distribution),
                'problems': dict(problem_distribution),
                'regions': dict(region_distribution)
            },
            'patterns': {
                'success_patterns': success_patterns,
                'failure_patterns': failure_patterns,
                'success_pattern_count': len(success_patterns),
                'failure_pattern_count': len(failure_patterns)
            }
        }
        
        return self.analysis_results
    
    def identify_data_gaps(self) -> Dict[str, List[str]]:
        """データギャップの特定"""
        if not self.analysis_results:
            self.analyze_existing_data()
        
        gaps = {
            'underrepresented_problems': [],
            'missing_regions': [],
            'rating_imbalance': [],
            'success_case_shortage': []
        }
        
        # 問題分布の偏り
        problem_dist = self.analysis_results['distributions']['problems']
        total_problems = sum(problem_dist.values())
        
        for problem, count in problem_dist.items():
            ratio = count / total_problems if total_problems > 0 else 0
            if ratio < 0.1:  # 10%未満は不足
                gaps['underrepresented_problems'].append(f"{problem} ({count}件, {ratio:.1%})")
        
        # 地域要求の不足
        region_dist = self.analysis_results['distributions']['regions']
        expected_regions = ['画面左側', '画面右側', '画面上部', '画面下部', '画面中央', '画面右上', '画面左下']
        
        for region in expected_regions:
            if region not in region_dist:
                gaps['missing_regions'].append(region)
        
        # 評価分布の偏り
        rating_dist = self.analysis_results['distributions']['rating']
        total_ratings = sum(rating_dist.values())
        
        for rating in ['A', 'B', 'C', 'D', 'E', 'F']:
            count = rating_dist.get(rating, 0)
            ratio = count / total_ratings if total_ratings > 0 else 0
            if rating in ['A', 'B'] and ratio < 0.2:  # 成功例が20%未満
                gaps['success_case_shortage'].append(f"{rating}評価 ({count}件, {ratio:.1%})")
            elif rating in ['D', 'E', 'F'] and ratio > 0.4:  # 失敗例が40%超
                gaps['rating_imbalance'].append(f"{rating}評価が過多 ({count}件, {ratio:.1%})")
        
        return gaps
    
    def generate_collection_strategy(self) -> Dict[str, Any]:
        """効率的データ収集戦略の生成"""
        gaps = self.identify_data_gaps()
        
        # 優先度付きデータ収集計画
        strategy = {
            'high_priority': [],
            'medium_priority': [],
            'low_priority': [],
            'sampling_strategy': {},
            'target_metrics': {}
        }
        
        # 高優先度: 成功事例不足
        if gaps['success_case_shortage']:
            strategy['high_priority'].extend([
                "A・B評価獲得事例の収集強化",
                "成功パターンの詳細分析",
                "高品質抽出の再現条件特定"
            ])
        
        # 中優先度: 地域・問題の偏り
        if gaps['missing_regions']:
            strategy['medium_priority'].extend([
                f"不足地域の収集: {', '.join(gaps['missing_regions'])}",
                "地域別キャラクター配置パターンの収集"
            ])
        
        if gaps['underrepresented_problems']:
            strategy['medium_priority'].extend([
                "少数問題パターンの集中収集",
                "境界例・難しいケースの特定"
            ])
        
        # サンプリング戦略
        current_success_rate = self.analysis_results['basic_stats']['success_rate']
        target_samples = max(100, len(self.evaluation_data) * 2)  # 現在の2倍または100件
        
        strategy['sampling_strategy'] = {
            'target_total_samples': target_samples,
            'success_case_ratio': 0.4,  # 40%は成功例
            'failure_case_ratio': 0.6,  # 60%は改善対象
            'regional_balance': True,
            'problem_type_balance': True
        }
        
        # 目標メトリクス
        strategy['target_metrics'] = {
            'success_rate_improvement': min(0.8, current_success_rate + 0.15),  # +15%改善
            'A_rating_ratio': 0.25,  # A評価25%
            'regional_coverage': 1.0,  # 全地域カバー
            'problem_type_coverage': 0.9  # 問題タイプ90%カバー
        }
        
        return strategy
    
    def create_collection_plan(self, dataset_name: str = None) -> Dict[str, Any]:
        """具体的な収集計画の作成"""
        strategy = self.generate_collection_strategy()
        
        plan = {
            'dataset_name': dataset_name or f"learning_data_{datetime.now().strftime('%Y%m%d')}",
            'collection_phases': [],
            'execution_steps': [],
            'success_criteria': {},
            'timeline': {}
        }
        
        # Phase 1: 成功事例収集
        plan['collection_phases'].append({
            'phase': 1,
            'name': "成功事例収集強化",
            'target_samples': strategy['sampling_strategy']['target_total_samples'] // 3,
            'focus': "A・B評価獲得可能な画像の特定と収集",
            'methods': [
                "既存成功パターンの類似画像検索",
                "高品質データセットからの選別収集",
                "複数品質手法での事前スクリーニング"
            ]
        })
        
        # Phase 2: 地域バランス改善
        plan['collection_phases'].append({
            'phase': 2,
            'name': "地域バランス改善",
            'target_samples': strategy['sampling_strategy']['target_total_samples'] // 3,
            'focus': "不足地域のキャラクター配置パターン収集",
            'methods': [
                "地域別キャラクター配置の意図的収集",
                "複数キャラクターシーンでの主人公特定",
                "背景・前景関係の多様化"
            ]
        })
        
        # Phase 3: 困難事例収集
        plan['collection_phases'].append({
            'phase': 3,
            'name': "困難事例・境界例収集",
            'target_samples': strategy['sampling_strategy']['target_total_samples'] // 3,
            'focus': "改善対象となる難しいケースの収集",
            'methods': [
                "複雑な姿勢・構図の収集",
                "部分隠蔽・重複キャラクターの収集",
                "エフェクト・背景が複雑な画像の収集"
            ]
        })
        
        # 実行ステップ
        plan['execution_steps'] = [
            "1. 候補データセットの特定（kaname08, kaname09等）",
            "2. 事前品質評価による優先度付け",
            "3. Phase別データ収集の実行",
            "4. 収集データの品質検証",
            "5. 学習データセットへの統合",
            "6. 効果測定とフィードバック"
        ]
        
        # 成功基準
        plan['success_criteria'] = {
            'minimum_sample_count': strategy['sampling_strategy']['target_total_samples'],
            'target_success_rate': strategy['target_metrics']['success_rate_improvement'],
            'regional_coverage_ratio': strategy['target_metrics']['regional_coverage'],
            'quality_distribution_balance': "A:B:C = 25:25:25, D:E:F = 8:8:9"
        }
        
        return plan
    
    def save_analysis_report(self, output_path: str = None) -> str:
        """分析レポートの保存"""
        if not output_path:
            output_path = f"learning_data_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'analysis_results': self.analysis_results,
            'data_gaps': self.identify_data_gaps(),
            'collection_strategy': self.generate_collection_strategy(),
            'collection_plan': self.create_collection_plan()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 分析レポート保存: {output_path}")
        return output_path
    
    def print_summary(self):
        """サマリーの出力"""
        if not self.analysis_results:
            self.analyze_existing_data()
        
        stats = self.analysis_results['basic_stats']
        gaps = self.identify_data_gaps()
        strategy = self.generate_collection_strategy()
        
        print("\n" + "="*60)
        print("📊 学習データ収集計画 - 分析サマリー")
        print("="*60)
        
        print(f"\n📈 現在のデータ状況:")
        print(f"  総件数: {stats['total_count']}件")
        print(f"  成功率: {stats['success_rate']:.1%} ({stats['success_count']}/{stats['total_count']})")
        print(f"  失敗件数: {stats['failure_count']}件")
        
        print(f"\n🎯 特定されたデータギャップ:")
        if gaps['success_case_shortage']:
            print(f"  成功事例不足: {', '.join(gaps['success_case_shortage'])}")
        if gaps['missing_regions']:
            print(f"  不足地域: {', '.join(gaps['missing_regions'])}")
        if gaps['underrepresented_problems']:
            print(f"  少数問題: {', '.join(gaps['underrepresented_problems'])}")
        
        print(f"\n🚀 推奨収集戦略:")
        target = strategy['sampling_strategy']['target_total_samples']
        current = stats['total_count']
        additional = target - current
        
        print(f"  追加収集目標: {additional}件 (現在{current}件 → 目標{target}件)")
        print(f"  成功率改善目標: {stats['success_rate']:.1%} → {strategy['target_metrics']['success_rate_improvement']:.1%}")
        print(f"  A評価目標比率: {strategy['target_metrics']['A_rating_ratio']:.1%}")


def main():
    """メイン実行関数"""
    print("🚀 学習データ収集計画策定システム開始")
    
    # 初期化
    planner = LearningDataCollectionPlanner()
    
    if not planner.evaluation_data:
        print("❌ 評価データが見つかりません")
        return
    
    # 分析実行
    print("\n📊 既存データ分析中...")
    planner.analyze_existing_data()
    
    # サマリー出力
    planner.print_summary()
    
    # レポート保存
    print("\n💾 詳細レポート保存中...")
    report_path = planner.save_analysis_report()
    
    print(f"\n✅ [P1-009] 学習データ収集計画策定完了")
    print(f"📄 詳細レポート: {report_path}")


if __name__ == "__main__":
    main()