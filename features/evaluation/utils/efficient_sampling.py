#!/usr/bin/env python3
"""
P1-010: 効率的サンプリングアルゴリズム
学習データ収集の効率性を最大化するサンプリング戦略システム

Features:
- Multi-criteria sampling strategies
- Uncertainty-based sampling
- Diversity-maximizing sampling
- Stratified sampling by quality metrics
- Active learning integration
- Cost-effectiveness optimization
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import random

# フォールバック実装用
HAS_SCIPY = True
HAS_SKLEARN = True

try:
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import entropy
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import pairwise_distances
    from sklearn.preprocessing import StandardScaler
except ImportError:
    HAS_SKLEARN = False


class EfficientSampling:
    """効率的サンプリングシステム"""
    
    def __init__(self):
        """初期化"""
        self.name = "EfficientSampling"
        self.version = "1.0.0"
        
        # サンプリングパラメータ
        self.sampling_params = {
            'uncertainty_weight': 0.3,      # 不確実性重み
            'diversity_weight': 0.3,        # 多様性重み
            'quality_weight': 0.2,          # 品質重み
            'cost_weight': 0.2,             # コスト重み
            'min_samples_per_stratum': 2,   # 層別最小サンプル数
            'max_samples_per_stratum': 10,  # 層別最大サンプル数
            'similarity_threshold': 0.8,    # 類似度閾値
            'exploration_ratio': 0.3        # 探索率
        }
        
        # サンプリング戦略
        self.strategies = {
            'uncertainty_based': self._uncertainty_based_sampling,
            'diversity_maximizing': self._diversity_maximizing_sampling,
            'stratified_quality': self._stratified_quality_sampling,
            'active_learning': self._active_learning_sampling,
            'cost_effective': self._cost_effective_sampling,
            'hybrid': self._hybrid_sampling
        }
    
    def generate_sampling_strategy(self, candidate_data: List[Dict], 
                                 target_samples: int,
                                 strategy: str = 'hybrid',
                                 constraints: Optional[Dict] = None) -> Dict[str, Any]:
        """
        効率的サンプリング戦略生成
        
        Args:
            candidate_data: 候補データリスト
            target_samples: 目標サンプル数
            strategy: サンプリング戦略
            constraints: 制約条件
            
        Returns:
            Dict: サンプリング戦略結果
        """
        if not candidate_data:
            return self._generate_error_result("No candidate data provided")
        
        if target_samples <= 0:
            return self._generate_error_result("Invalid target sample count")
        
        try:
            # 候補データ前処理
            processed_candidates = self._preprocess_candidates(candidate_data)
            
            # 制約条件適用
            if constraints:
                processed_candidates = self._apply_constraints(processed_candidates, constraints)
            
            # 特徴量抽出
            features = self._extract_features(processed_candidates)
            
            # 指定戦略でサンプリング実行
            if strategy in self.strategies:
                sampling_result = self.strategies[strategy](
                    processed_candidates, features, target_samples
                )
            else:
                return self._generate_error_result(f"Unknown strategy: {strategy}")
            
            # サンプリング効果分析
            effectiveness_analysis = self._analyze_sampling_effectiveness(
                sampling_result, processed_candidates, features
            )
            
            # 推奨サンプル最適化
            optimization_suggestions = self._generate_optimization_suggestions(
                sampling_result, effectiveness_analysis
            )
            
            return {
                'sampling_strategy': strategy,
                'target_samples': target_samples,
                'actual_samples': len(sampling_result['selected_samples']),
                'candidate_pool_size': len(processed_candidates),
                'selected_samples': sampling_result['selected_samples'],
                'sampling_rationale': sampling_result.get('rationale', {}),
                'effectiveness_analysis': effectiveness_analysis,
                'optimization_suggestions': optimization_suggestions,
                'processing_info': {
                    'timestamp': datetime.now().isoformat(),
                    'version': self.version
                }
            }
            
        except Exception as e:
            return self._generate_error_result(f"Sampling strategy generation failed: {str(e)}")
    
    def _preprocess_candidates(self, candidate_data: List[Dict]) -> List[Dict]:
        """候補データ前処理"""
        processed = []
        
        for i, candidate in enumerate(candidate_data):
            processed_candidate = {
                'id': candidate.get('id', f'candidate_{i}'),
                'image_path': candidate.get('image_path', ''),
                'quality_score': candidate.get('quality_score', 0.0),
                'confidence_score': candidate.get('confidence_score', 0.0),
                'processing_time': candidate.get('processing_time', 1.0),
                'complexity_score': candidate.get('complexity_score', 0.5),
                'previous_evaluation': candidate.get('previous_evaluation'),
                'characteristics': candidate.get('characteristics', {}),
                'metadata': candidate.get('metadata', {})
            }
            
            # 不確実性スコア計算
            processed_candidate['uncertainty_score'] = self._calculate_uncertainty(processed_candidate)
            
            # コストスコア計算
            processed_candidate['cost_score'] = self._calculate_cost_score(processed_candidate)
            
            processed.append(processed_candidate)
        
        return processed
    
    def _calculate_uncertainty(self, candidate: Dict) -> float:
        """不確実性スコア計算"""
        quality_score = candidate['quality_score']
        confidence_score = candidate['confidence_score']
        
        # 品質スコアの曖昧性（0.5付近が最も不確実）
        quality_uncertainty = 1.0 - 2 * abs(quality_score - 0.5)
        
        # 信頼度の低さ
        confidence_uncertainty = 1.0 - confidence_score
        
        # 統合不確実性
        uncertainty = (quality_uncertainty + confidence_uncertainty) / 2
        
        return min(max(uncertainty, 0.0), 1.0)
    
    def _calculate_cost_score(self, candidate: Dict) -> float:
        """コストスコア計算（低い方が良い）"""
        processing_time = candidate['processing_time']
        complexity = candidate['complexity_score']
        
        # 処理時間正規化（仮定：最大10秒）
        time_cost = min(processing_time / 10.0, 1.0)
        
        # 複雑性コスト
        complexity_cost = complexity
        
        # 統合コスト
        cost = (time_cost + complexity_cost) / 2
        
        return min(max(cost, 0.0), 1.0)
    
    def _apply_constraints(self, candidates: List[Dict], constraints: Dict) -> List[Dict]:
        """制約条件適用"""
        filtered = candidates.copy()
        
        # 品質スコア制約
        if 'min_quality' in constraints:
            min_quality = constraints['min_quality']
            filtered = [c for c in filtered if c['quality_score'] >= min_quality]
        
        if 'max_quality' in constraints:
            max_quality = constraints['max_quality']
            filtered = [c for c in filtered if c['quality_score'] <= max_quality]
        
        # 処理時間制約
        if 'max_processing_time' in constraints:
            max_time = constraints['max_processing_time']
            filtered = [c for c in filtered if c['processing_time'] <= max_time]
        
        # 特性制約
        if 'required_characteristics' in constraints:
            required = constraints['required_characteristics']
            filtered = [c for c in filtered 
                       if all(c['characteristics'].get(key) == value 
                             for key, value in required.items())]
        
        return filtered
    
    def _extract_features(self, candidates: List[Dict]) -> np.ndarray:
        """特徴量抽出"""
        features = []
        
        for candidate in candidates:
            feature_vector = [
                candidate['quality_score'],
                candidate['confidence_score'],
                candidate['uncertainty_score'],
                candidate['cost_score'],
                candidate['complexity_score'],
                candidate['processing_time'] / 10.0  # 正規化
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def _uncertainty_based_sampling(self, candidates: List[Dict], 
                                   features: np.ndarray, 
                                   target_samples: int) -> Dict[str, Any]:
        """不確実性ベースサンプリング"""
        # 不確実性スコアでソート
        sorted_indices = sorted(range(len(candidates)), 
                              key=lambda i: candidates[i]['uncertainty_score'], 
                              reverse=True)
        
        # 上位不確実性サンプル選択
        selected_indices = sorted_indices[:target_samples]
        selected_samples = [candidates[i] for i in selected_indices]
        
        return {
            'selected_samples': selected_samples,
            'rationale': {
                'strategy': 'uncertainty_based',
                'avg_uncertainty': np.mean([candidates[i]['uncertainty_score'] for i in selected_indices]),
                'selection_criterion': 'highest_uncertainty'
            }
        }
    
    def _diversity_maximizing_sampling(self, candidates: List[Dict],
                                     features: np.ndarray,
                                     target_samples: int) -> Dict[str, Any]:
        """多様性最大化サンプリング"""
        try:
            if HAS_SKLEARN and len(features) > target_samples:
                # K-meansクラスタリングベース多様性サンプリング
                n_clusters = min(target_samples, len(features))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(features)
                
                # 各クラスタから代表サンプル選択
                selected_indices = []
                for cluster_id in range(n_clusters):
                    cluster_indices = np.where(cluster_labels == cluster_id)[0]
                    if len(cluster_indices) > 0:
                        # クラスタ中心に最も近いサンプル選択
                        cluster_center = kmeans.cluster_centers_[cluster_id]
                        distances = [np.linalg.norm(features[i] - cluster_center) 
                                   for i in cluster_indices]
                        best_idx = cluster_indices[np.argmin(distances)]
                        selected_indices.append(best_idx)
                
                # 目標数に満たない場合は追加選択
                while len(selected_indices) < target_samples and len(selected_indices) < len(candidates):
                    remaining_indices = [i for i in range(len(candidates)) if i not in selected_indices]
                    if not remaining_indices:
                        break
                    
                    # 最も離れたサンプルを追加
                    if selected_indices:
                        selected_features = features[selected_indices]
                        distances = []
                        for idx in remaining_indices:
                            min_dist = min(np.linalg.norm(features[idx] - selected_features[j]) 
                                         for j in range(len(selected_features)))
                            distances.append(min_dist)
                        best_remaining = remaining_indices[np.argmax(distances)]
                        selected_indices.append(best_remaining)
                    else:
                        selected_indices.append(remaining_indices[0])
                
                rationale = {
                    'strategy': 'diversity_maximizing',
                    'method': 'kmeans_clustering',
                    'n_clusters': n_clusters
                }
                
            else:
                # フォールバック：簡易多様性サンプリング
                selected_indices = self._simple_diversity_sampling(features, target_samples)
                rationale = {
                    'strategy': 'diversity_maximizing',
                    'method': 'simple_diversity',
                    'fallback': True
                }
            
            selected_samples = [candidates[i] for i in selected_indices]
            
            return {
                'selected_samples': selected_samples,
                'rationale': rationale
            }
            
        except Exception as e:
            # エラー時はランダムサンプリング
            selected_indices = random.sample(range(len(candidates)), 
                                           min(target_samples, len(candidates)))
            selected_samples = [candidates[i] for i in selected_indices]
            
            return {
                'selected_samples': selected_samples,
                'rationale': {
                    'strategy': 'diversity_maximizing',
                    'method': 'random_fallback',
                    'error': str(e)
                }
            }
    
    def _simple_diversity_sampling(self, features: np.ndarray, target_samples: int) -> List[int]:
        """簡易多様性サンプリング（フォールバック）"""
        if len(features) <= target_samples:
            return list(range(len(features)))
        
        selected_indices = []
        
        # 最初のサンプルはランダム選択
        first_idx = random.randint(0, len(features) - 1)
        selected_indices.append(first_idx)
        
        # 残りは最も離れたサンプルを順次選択
        for _ in range(target_samples - 1):
            remaining_indices = [i for i in range(len(features)) if i not in selected_indices]
            if not remaining_indices:
                break
            
            max_min_distance = -1
            best_idx = remaining_indices[0]
            
            for candidate_idx in remaining_indices:
                min_distance = min(np.linalg.norm(features[candidate_idx] - features[selected_idx])
                                 for selected_idx in selected_indices)
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_idx = candidate_idx
            
            selected_indices.append(best_idx)
        
        return selected_indices
    
    def _stratified_quality_sampling(self, candidates: List[Dict],
                                   features: np.ndarray,
                                   target_samples: int) -> Dict[str, Any]:
        """品質層別サンプリング"""
        # 品質スコアによる層分け
        quality_scores = [c['quality_score'] for c in candidates]
        
        # 品質レンジごとの層定義
        strata = {
            'high_quality': [i for i, score in enumerate(quality_scores) if score >= 0.7],
            'medium_quality': [i for i, score in enumerate(quality_scores) if 0.3 <= score < 0.7],
            'low_quality': [i for i, score in enumerate(quality_scores) if score < 0.3]
        }
        
        # 各層からのサンプル数決定
        total_candidates = sum(len(stratum) for stratum in strata.values())
        selected_indices = []
        
        for stratum_name, stratum_indices in strata.items():
            if not stratum_indices:
                continue
                
            # 層サイズに比例した配分
            stratum_ratio = len(stratum_indices) / total_candidates
            stratum_samples = max(
                self.sampling_params['min_samples_per_stratum'],
                min(
                    int(target_samples * stratum_ratio),
                    self.sampling_params['max_samples_per_stratum'],
                    len(stratum_indices)
                )
            )
            
            # 層内でランダムサンプリング
            if stratum_samples < len(stratum_indices):
                sampled = random.sample(stratum_indices, stratum_samples)
            else:
                sampled = stratum_indices
            
            selected_indices.extend(sampled)
        
        # 目標数調整
        if len(selected_indices) > target_samples:
            selected_indices = random.sample(selected_indices, target_samples)
        elif len(selected_indices) < target_samples:
            # 不足分をランダム補完
            remaining = [i for i in range(len(candidates)) if i not in selected_indices]
            additional_needed = target_samples - len(selected_indices)
            if remaining and additional_needed > 0:
                additional = random.sample(remaining, min(additional_needed, len(remaining)))
                selected_indices.extend(additional)
        
        selected_samples = [candidates[i] for i in selected_indices]
        
        # 層別統計
        stratum_stats = {}
        for stratum_name, stratum_indices in strata.items():
            selected_in_stratum = [i for i in selected_indices if i in stratum_indices]
            stratum_stats[stratum_name] = {
                'total_candidates': len(stratum_indices),
                'selected_samples': len(selected_in_stratum),
                'selection_ratio': len(selected_in_stratum) / len(stratum_indices) if stratum_indices else 0
            }
        
        return {
            'selected_samples': selected_samples,
            'rationale': {
                'strategy': 'stratified_quality',
                'strata_definition': {name: len(indices) for name, indices in strata.items()},
                'stratum_statistics': stratum_stats
            }
        }
    
    def _active_learning_sampling(self, candidates: List[Dict],
                                features: np.ndarray,
                                target_samples: int) -> Dict[str, Any]:
        """アクティブラーニングサンプリング"""
        # 不確実性と多様性の組み合わせ
        uncertainty_scores = [c['uncertainty_score'] for c in candidates]
        
        # 各候補の総合スコア計算
        composite_scores = []
        for i, candidate in enumerate(candidates):
            uncertainty = uncertainty_scores[i]
            
            # 多様性スコア計算（他サンプルとの平均距離）
            if len(features) > 1:
                distances = [np.linalg.norm(features[i] - features[j]) 
                           for j in range(len(features)) if i != j]
                diversity = np.mean(distances) if distances else 0
                # 正規化
                diversity = min(diversity / np.max(pdist(features)) if len(features) > 1 else 0, 1.0)
            else:
                diversity = 0.5
            
            # 統合スコア
            composite_score = (self.sampling_params['uncertainty_weight'] * uncertainty +
                             self.sampling_params['diversity_weight'] * diversity)
            composite_scores.append(composite_score)
        
        # 上位スコアでサンプル選択
        sorted_indices = sorted(range(len(candidates)), 
                              key=lambda i: composite_scores[i], 
                              reverse=True)
        
        selected_indices = sorted_indices[:target_samples]
        selected_samples = [candidates[i] for i in selected_indices]
        
        return {
            'selected_samples': selected_samples,
            'rationale': {
                'strategy': 'active_learning',
                'uncertainty_weight': self.sampling_params['uncertainty_weight'],
                'diversity_weight': self.sampling_params['diversity_weight'],
                'avg_composite_score': np.mean([composite_scores[i] for i in selected_indices])
            }
        }
    
    def _cost_effective_sampling(self, candidates: List[Dict],
                                features: np.ndarray,
                                target_samples: int) -> Dict[str, Any]:
        """コスト効率サンプリング"""
        # コスト効率スコア計算（価値/コスト比）
        efficiency_scores = []
        
        for candidate in candidates:
            # 価値スコア（不確実性 + 品質のバランス）
            value = (candidate['uncertainty_score'] * 0.6 + 
                    (1.0 - abs(candidate['quality_score'] - 0.5) * 2) * 0.4)
            
            # コストスコア
            cost = candidate['cost_score']
            
            # 効率スコア（コストがゼロの場合は価値のみ）
            efficiency = value / (cost + 0.01)  # ゼロ除算回避
            efficiency_scores.append(efficiency)
        
        # 効率の高い順に選択
        sorted_indices = sorted(range(len(candidates)), 
                              key=lambda i: efficiency_scores[i], 
                              reverse=True)
        
        selected_indices = sorted_indices[:target_samples]
        selected_samples = [candidates[i] for i in selected_indices]
        
        return {
            'selected_samples': selected_samples,
            'rationale': {
                'strategy': 'cost_effective',
                'avg_efficiency_score': np.mean([efficiency_scores[i] for i in selected_indices]),
                'total_estimated_cost': sum(candidates[i]['cost_score'] for i in selected_indices)
            }
        }
    
    def _hybrid_sampling(self, candidates: List[Dict],
                        features: np.ndarray,
                        target_samples: int) -> Dict[str, Any]:
        """ハイブリッドサンプリング（複数戦略の組み合わせ）"""
        # 各戦略で小規模サンプリング実行
        strategies_samples = {
            'uncertainty': max(1, target_samples // 4),
            'diversity': max(1, target_samples // 4),
            'stratified': max(1, target_samples // 4),
            'cost_effective': max(1, target_samples // 4)
        }
        
        # 残りサンプル数
        remaining_samples = target_samples - sum(strategies_samples.values())
        if remaining_samples > 0:
            strategies_samples['uncertainty'] += remaining_samples
        
        all_selected_indices = set()
        strategy_results = {}
        
        # 各戦略実行
        for strategy_name, sample_count in strategies_samples.items():
            if strategy_name == 'uncertainty':
                result = self._uncertainty_based_sampling(candidates, features, sample_count)
            elif strategy_name == 'diversity':
                result = self._diversity_maximizing_sampling(candidates, features, sample_count)
            elif strategy_name == 'stratified':
                result = self._stratified_quality_sampling(candidates, features, sample_count)
            elif strategy_name == 'cost_effective':
                result = self._cost_effective_sampling(candidates, features, sample_count)
            
            strategy_results[strategy_name] = result
            
            # 選択されたサンプルのインデックス追加
            for sample in result['selected_samples']:
                sample_id = sample['id']
                for i, candidate in enumerate(candidates):
                    if candidate['id'] == sample_id:
                        all_selected_indices.add(i)
                        break
        
        # 重複除去と目標数調整
        selected_indices = list(all_selected_indices)
        
        if len(selected_indices) > target_samples:
            # 超過分をランダム除去
            selected_indices = random.sample(selected_indices, target_samples)
        elif len(selected_indices) < target_samples:
            # 不足分をランダム追加
            remaining_indices = [i for i in range(len(candidates)) if i not in selected_indices]
            additional_needed = target_samples - len(selected_indices)
            if remaining_indices and additional_needed > 0:
                additional = random.sample(remaining_indices, min(additional_needed, len(remaining_indices)))
                selected_indices.extend(additional)
        
        selected_samples = [candidates[i] for i in selected_indices]
        
        return {
            'selected_samples': selected_samples,
            'rationale': {
                'strategy': 'hybrid',
                'component_strategies': list(strategies_samples.keys()),
                'strategy_allocations': strategies_samples,
                'total_unique_selections': len(all_selected_indices)
            }
        }
    
    def _analyze_sampling_effectiveness(self, sampling_result: Dict,
                                      all_candidates: List[Dict],
                                      features: np.ndarray) -> Dict[str, Any]:
        """サンプリング効果分析"""
        try:
            selected_samples = sampling_result['selected_samples']
            
            if not selected_samples:
                return {'error': 'No samples selected for analysis'}
            
            # 選択サンプルの特性統計
            selected_qualities = [s['quality_score'] for s in selected_samples]
            selected_uncertainties = [s['uncertainty_score'] for s in selected_samples]
            selected_costs = [s['cost_score'] for s in selected_samples]
            
            # 全体との比較
            all_qualities = [c['quality_score'] for c in all_candidates]
            all_uncertainties = [c['uncertainty_score'] for c in all_candidates]
            all_costs = [c['cost_score'] for c in all_candidates]
            
            effectiveness_metrics = {
                'coverage_analysis': {
                    'quality_coverage': {
                        'selected_mean': np.mean(selected_qualities),
                        'selected_std': np.std(selected_qualities),
                        'population_mean': np.mean(all_qualities),
                        'population_std': np.std(all_qualities),
                        'coverage_ratio': np.std(selected_qualities) / np.std(all_qualities) if np.std(all_qualities) > 0 else 0
                    },
                    'uncertainty_coverage': {
                        'selected_mean': np.mean(selected_uncertainties),
                        'population_mean': np.mean(all_uncertainties),
                        'high_uncertainty_ratio': sum(1 for u in selected_uncertainties if u > 0.7) / len(selected_uncertainties)
                    }
                },
                'efficiency_analysis': {
                    'avg_cost': np.mean(selected_costs),
                    'total_cost': np.sum(selected_costs),
                    'cost_effectiveness': np.mean(selected_uncertainties) / (np.mean(selected_costs) + 0.01)
                },
                'diversity_analysis': self._calculate_diversity_metrics(selected_samples, features)
            }
            
            return effectiveness_metrics
            
        except Exception as e:
            return {'error': f'Effectiveness analysis failed: {str(e)}'}
    
    def _calculate_diversity_metrics(self, selected_samples: List[Dict], 
                                   all_features: np.ndarray) -> Dict[str, Any]:
        """多様性メトリクス計算"""
        try:
            if len(selected_samples) <= 1:
                return {'diversity_score': 0.0, 'note': 'insufficient_samples'}
            
            # 選択サンプルの特徴量抽出
            selected_features = []
            for sample in selected_samples:
                for i, candidate_id in enumerate([f"candidate_{j}" for j in range(len(all_features))]):
                    if sample['id'] == candidate_id or sample.get('image_path') == f"path_{i}":
                        if i < len(all_features):
                            selected_features.append(all_features[i])
                        break
            
            if len(selected_features) <= 1:
                return {'diversity_score': 0.0, 'note': 'feature_extraction_failed'}
            
            selected_features = np.array(selected_features)
            
            # 平均ペアワイズ距離
            if HAS_SCIPY:
                pairwise_distances = pdist(selected_features)
                avg_distance = np.mean(pairwise_distances) if len(pairwise_distances) > 0 else 0
                
                # 全体特徴量空間でのスケーリング
                all_pairwise = pdist(all_features)
                max_possible_distance = np.max(all_pairwise) if len(all_pairwise) > 0 else 1
                
                diversity_score = avg_distance / max_possible_distance if max_possible_distance > 0 else 0
            else:
                # フォールバック：簡易多様性計算
                distances = []
                for i in range(len(selected_features)):
                    for j in range(i + 1, len(selected_features)):
                        dist = np.linalg.norm(selected_features[i] - selected_features[j])
                        distances.append(dist)
                
                avg_distance = np.mean(distances) if distances else 0
                diversity_score = min(avg_distance / 2.0, 1.0)  # 正規化
            
            return {
                'diversity_score': float(diversity_score),
                'avg_pairwise_distance': float(avg_distance),
                'feature_dimensions': selected_features.shape[1] if len(selected_features) > 0 else 0
            }
            
        except Exception as e:
            return {'error': f'Diversity calculation failed: {str(e)}'}
    
    def _generate_optimization_suggestions(self, sampling_result: Dict,
                                         effectiveness_analysis: Dict) -> List[str]:
        """最適化提案生成"""
        suggestions = []
        
        # 効果分析に基づく提案
        if 'coverage_analysis' in effectiveness_analysis:
            coverage = effectiveness_analysis['coverage_analysis']
            
            # 品質カバレッジ
            quality_coverage = coverage.get('quality_coverage', {})
            if quality_coverage.get('coverage_ratio', 0) < 0.5:
                suggestions.append("increase_quality_diversity")
            
            # 不確実性カバレッジ
            uncertainty_coverage = coverage.get('uncertainty_coverage', {})
            if uncertainty_coverage.get('high_uncertainty_ratio', 0) < 0.3:
                suggestions.append("focus_on_uncertain_samples")
        
        # 効率性分析
        if 'efficiency_analysis' in effectiveness_analysis:
            efficiency = effectiveness_analysis['efficiency_analysis']
            if efficiency.get('cost_effectiveness', 0) < 0.5:
                suggestions.append("optimize_cost_effectiveness")
        
        # 多様性分析
        if 'diversity_analysis' in effectiveness_analysis:
            diversity = effectiveness_analysis['diversity_analysis']
            if diversity.get('diversity_score', 0) < 0.4:
                suggestions.append("increase_sample_diversity")
        
        # 戦略固有の提案
        strategy = sampling_result.get('rationale', {}).get('strategy')
        if strategy == 'uncertainty_based':
            suggestions.append("consider_adding_diversity_component")
        elif strategy == 'diversity_maximizing':
            suggestions.append("consider_adding_uncertainty_component")
        
        return suggestions
    
    def _generate_error_result(self, error_message: str) -> Dict[str, Any]:
        """エラー結果生成"""
        return {
            'error': error_message,
            'sampling_strategy': 'failed',
            'selected_samples': []
        }


def main():
    """テスト実行"""
    print("🚀 P1-010: 効率的サンプリングアルゴリズム テスト開始")
    
    # テスト用候補データ作成
    test_candidates = []
    for i in range(20):
        candidate = {
            'id': f'test_image_{i:03d}',
            'image_path': f'/test/path/image_{i:03d}.jpg',
            'quality_score': random.uniform(0.1, 0.9),
            'confidence_score': random.uniform(0.3, 0.95),
            'processing_time': random.uniform(1.0, 8.0),
            'complexity_score': random.uniform(0.2, 0.8),
            'characteristics': {
                'scene_type': random.choice(['indoor', 'outdoor']),
                'character_count': random.randint(1, 3)
            }
        }
        test_candidates.append(candidate)
    
    # サンプリング実行
    sampler = EfficientSampling()
    result = sampler.generate_sampling_strategy(
        test_candidates, 
        target_samples=8, 
        strategy='hybrid'
    )
    
    print("\n📊 効率的サンプリング結果:")
    if 'error' not in result:
        print(f"  戦略: {result['sampling_strategy']}")
        print(f"  目標サンプル数: {result['target_samples']}")
        print(f"  実選択数: {result['actual_samples']}")
        print(f"  候補プール: {result['candidate_pool_size']}件")
        
        # 選択サンプル例
        selected = result['selected_samples'][:3]  # 最初の3つ表示
        print(f"\n🔍 選択サンプル例:")
        for sample in selected:
            print(f"    {sample['id']}: 品質{sample['quality_score']:.2f}, 不確実性{sample['uncertainty_score']:.2f}")
        
        # 最適化提案
        suggestions = result.get('optimization_suggestions', [])
        if suggestions:
            print(f"\n💡 最適化提案: {', '.join(suggestions)}")
    else:
        print(f"  ❌ エラー: {result['error']}")
    
    print(f"\n✅ [P1-010] 効率的サンプリングアルゴリズム完了")


if __name__ == "__main__":
    main()