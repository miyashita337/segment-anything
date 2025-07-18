#!/usr/bin/env python3
"""
P1-016: フィードバックループ構築
評価差分データを活用した継続的品質改善システム

Features:
- Automatic feedback integration
- Performance trend analysis
- Adaptive parameter adjustment
- Quality prediction improvement
- Learning effectiveness measurement
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta
import json

# フォールバック実装用
HAS_SCIPY = True
HAS_SKLEARN = True

try:
    from scipy import stats
    from scipy.optimize import minimize_scalar
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
except ImportError:
    HAS_SKLEARN = False


class FeedbackLoopSystem:
    """フィードバックループシステム"""
    
    def __init__(self, feedback_data_path: Optional[str] = None):
        """初期化"""
        self.name = "FeedbackLoopSystem"
        self.version = "1.0.0"
        
        # フィードバックデータ保存パス
        if feedback_data_path:
            self.feedback_data_path = Path(feedback_data_path)
        else:
            self.feedback_data_path = Path("features/evaluation/logs/feedback_history.jsonl")
        
        # フィードバックパラメータ
        self.feedback_params = {
            'learning_rate': 0.1,           # 学習率
            'momentum': 0.9,                # モメンタム
            'adaptation_threshold': 0.05,   # 適応閾値
            'trend_window': 10,             # トレンド分析ウィンドウ
            'prediction_horizon': 5,        # 予測期間
            'confidence_threshold': 0.7     # 信頼度閾値
        }
        
        # 改善履歴
        self.improvement_history = []
        self.parameter_history = []
        
        # 初期化時にデータ読み込み
        self._load_feedback_history()
    
    def integrate_evaluation_feedback(self, evaluation_data: Dict[str, Any], 
                                    processing_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        評価フィードバックの統合処理
        
        Args:
            evaluation_data: ユーザー評価データ
            processing_results: 自動処理結果
            
        Returns:
            Dict: フィードバック統合結果
        """
        try:
            # フィードバックエントリ作成
            feedback_entry = self._create_feedback_entry(evaluation_data, processing_results)
            
            # パフォーマンストレンド分析
            trend_analysis = self._analyze_performance_trend(feedback_entry)
            
            # 適応的パラメータ調整
            parameter_adjustments = self._adjust_parameters_adaptively(feedback_entry, trend_analysis)
            
            # 品質予測改善
            prediction_improvements = self._improve_quality_prediction(feedback_entry)
            
            # 学習効果測定
            learning_effectiveness = self._measure_learning_effectiveness()
            
            # フィードバック保存
            self._save_feedback_entry(feedback_entry)
            
            # 改善推奨事項生成
            improvement_recommendations = self._generate_improvement_recommendations(
                trend_analysis, parameter_adjustments, prediction_improvements
            )
            
            return {
                'feedback_integration': {
                    'entry_id': feedback_entry['entry_id'],
                    'timestamp': feedback_entry['timestamp'],
                    'integration_success': True
                },
                'trend_analysis': trend_analysis,
                'parameter_adjustments': parameter_adjustments,
                'prediction_improvements': prediction_improvements,
                'learning_effectiveness': learning_effectiveness,
                'improvement_recommendations': improvement_recommendations,
                'processing_info': {
                    'version': self.version,
                    'feedback_count': len(self.improvement_history)
                }
            }
            
        except Exception as e:
            return self._generate_error_result(f"Feedback integration failed: {str(e)}")
    
    def _create_feedback_entry(self, evaluation_data: Dict, processing_results: Dict) -> Dict[str, Any]:
        """フィードバックエントリ作成"""
        entry_id = f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.improvement_history):04d}"
        
        # 評価差分計算
        auto_score = processing_results.get('quality_score', 0.0)
        user_score = evaluation_data.get('user_rating', 0.0)
        evaluation_diff = abs(auto_score - user_score)
        
        # 処理パラメータ抽出
        processing_params = self._extract_processing_parameters(processing_results)
        
        # 結果メトリクス抽出
        result_metrics = self._extract_result_metrics(processing_results)
        
        feedback_entry = {
            'entry_id': entry_id,
            'timestamp': datetime.now().isoformat(),
            'evaluation_data': {
                'user_rating': user_score,
                'user_comments': evaluation_data.get('comments', ''),
                'evaluation_aspects': evaluation_data.get('aspects', {})
            },
            'processing_results': {
                'auto_quality_score': auto_score,
                'processing_time': processing_results.get('processing_time', 0.0),
                'success': processing_results.get('success', False)
            },
            'evaluation_difference': {
                'score_difference': evaluation_diff,
                'agreement_level': self._calculate_agreement_level(evaluation_diff)
            },
            'processing_parameters': processing_params,
            'result_metrics': result_metrics
        }
        
        return feedback_entry
    
    def _extract_processing_parameters(self, processing_results: Dict) -> Dict[str, Any]:
        """処理パラメータ抽出"""
        return {
            'quality_method': processing_results.get('quality_method', 'unknown'),
            'enhancement_applied': processing_results.get('enhancement_applied', []),
            'yolo_score_threshold': processing_results.get('yolo_score_threshold', 0.0),
            'sam_parameters': processing_results.get('sam_parameters', {}),
            'preprocessing_steps': processing_results.get('preprocessing_steps', [])
        }
    
    def _extract_result_metrics(self, processing_results: Dict) -> Dict[str, Any]:
        """結果メトリクス抽出"""
        return {
            'extraction_success': processing_results.get('success', False),
            'quality_score': processing_results.get('quality_score', 0.0),
            'confidence_score': processing_results.get('confidence_score', 0.0),
            'boundary_quality': processing_results.get('boundary_quality', 0.0),
            'completeness_score': processing_results.get('completeness_score', 0.0)
        }
    
    def _calculate_agreement_level(self, score_diff: float) -> str:
        """評価一致度計算"""
        if score_diff <= 0.1:
            return "high_agreement"
        elif score_diff <= 0.2:
            return "moderate_agreement"
        elif score_diff <= 0.4:
            return "low_agreement"
        else:
            return "strong_disagreement"
    
    def _analyze_performance_trend(self, current_entry: Dict) -> Dict[str, Any]:
        """パフォーマンストレンド分析"""
        try:
            if len(self.improvement_history) < 2:
                return {'status': 'insufficient_data', 'trend': 'unknown'}
            
            # 最近のエントリから評価差分の推移を分析
            recent_entries = self.improvement_history[-self.feedback_params['trend_window']:]
            recent_diffs = [entry['evaluation_difference']['score_difference'] for entry in recent_entries]
            
            # トレンド分析
            if HAS_SCIPY:
                # 線形回帰によるトレンド
                x = np.arange(len(recent_diffs))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_diffs)
                
                trend_direction = "improving" if slope < 0 else "degrading" if slope > 0 else "stable"
                trend_strength = abs(r_value)
                
                # 統計的有意性
                is_significant = p_value < 0.05
                
                trend_analysis = {
                    'trend_direction': trend_direction,
                    'trend_strength': float(trend_strength),
                    'slope': float(slope),
                    'r_squared': float(r_value**2),
                    'p_value': float(p_value),
                    'is_statistically_significant': is_significant
                }
            else:
                # フォールバック：簡易トレンド分析
                first_half = recent_diffs[:len(recent_diffs)//2]
                second_half = recent_diffs[len(recent_diffs)//2:]
                
                first_avg = np.mean(first_half) if first_half else 0
                second_avg = np.mean(second_half) if second_half else 0
                
                if second_avg < first_avg - 0.02:
                    trend_direction = "improving"
                elif second_avg > first_avg + 0.02:
                    trend_direction = "degrading"
                else:
                    trend_direction = "stable"
                
                trend_analysis = {
                    'trend_direction': trend_direction,
                    'first_half_avg': float(first_avg),
                    'second_half_avg': float(second_avg),
                    'improvement': float(first_avg - second_avg)
                }
            
            # 現在の評価差分との比較
            current_diff = current_entry['evaluation_difference']['score_difference']
            recent_avg = np.mean(recent_diffs)
            
            trend_analysis.update({
                'current_vs_recent_avg': float(current_diff - recent_avg),
                'recent_average_difference': float(recent_avg),
                'data_points_analyzed': len(recent_diffs)
            })
            
            return trend_analysis
            
        except Exception as e:
            return {'error': f'Trend analysis failed: {str(e)}'}
    
    def _adjust_parameters_adaptively(self, feedback_entry: Dict, 
                                    trend_analysis: Dict) -> Dict[str, Any]:
        """適応的パラメータ調整"""
        try:
            adjustments = {}
            
            evaluation_diff = feedback_entry['evaluation_difference']['score_difference']
            agreement_level = feedback_entry['evaluation_difference']['agreement_level']
            
            # 評価差分に基づく調整
            if evaluation_diff > self.feedback_params['adaptation_threshold']:
                if agreement_level in ['low_agreement', 'strong_disagreement']:
                    # 大きな評価差分がある場合の調整
                    adjustments.update(self._generate_disagreement_adjustments(feedback_entry))
            
            # トレンドに基づく調整
            if 'trend_direction' in trend_analysis:
                if trend_analysis['trend_direction'] == 'degrading':
                    adjustments.update(self._generate_degradation_adjustments(trend_analysis))
                elif trend_analysis['trend_direction'] == 'improving':
                    adjustments.update(self._generate_improvement_adjustments(trend_analysis))
            
            # 品質手法固有の調整
            quality_method = feedback_entry['processing_parameters'].get('quality_method')
            if quality_method:
                method_adjustments = self._generate_method_specific_adjustments(
                    quality_method, feedback_entry
                )
                adjustments.update(method_adjustments)
            
            # 調整の実効性評価
            adjustment_confidence = self._evaluate_adjustment_confidence(adjustments, trend_analysis)
            
            return {
                'parameter_adjustments': adjustments,
                'adjustment_confidence': adjustment_confidence,
                'adjustment_rationale': self._generate_adjustment_rationale(adjustments, trend_analysis),
                'recommended_application': adjustment_confidence > self.feedback_params['confidence_threshold']
            }
            
        except Exception as e:
            return {'error': f'Parameter adjustment failed: {str(e)}'}
    
    def _generate_disagreement_adjustments(self, feedback_entry: Dict) -> Dict[str, Any]:
        """評価不一致時の調整"""
        adjustments = {}
        
        auto_score = feedback_entry['processing_results']['auto_quality_score']
        user_score = feedback_entry['evaluation_data']['user_rating']
        
        if auto_score > user_score:
            # 自動評価が過大評価している場合
            adjustments['quality_threshold'] = {'adjustment': -0.05, 'reason': 'auto_overestimation'}
            adjustments['confidence_scaling'] = {'adjustment': 0.9, 'reason': 'reduce_overconfidence'}
        else:
            # 自動評価が過小評価している場合
            adjustments['quality_threshold'] = {'adjustment': 0.03, 'reason': 'auto_underestimation'}
            adjustments['sensitivity_increase'] = {'adjustment': 1.1, 'reason': 'increase_sensitivity'}
        
        return adjustments
    
    def _generate_degradation_adjustments(self, trend_analysis: Dict) -> Dict[str, Any]:
        """性能劣化時の調整"""
        adjustments = {}
        
        # より保守的なパラメータ設定
        adjustments['learning_rate'] = {
            'adjustment': self.feedback_params['learning_rate'] * 0.8, 
            'reason': 'performance_degradation'
        }
        
        # 品質閾値の調整
        adjustments['quality_threshold_increase'] = {
            'adjustment': 0.02, 
            'reason': 'compensate_degradation'
        }
        
        return adjustments
    
    def _generate_improvement_adjustments(self, trend_analysis: Dict) -> Dict[str, Any]:
        """性能改善時の調整"""
        adjustments = {}
        
        # より積極的なパラメータ設定
        adjustments['learning_rate'] = {
            'adjustment': self.feedback_params['learning_rate'] * 1.1, 
            'reason': 'performance_improvement'
        }
        
        # 適応的閾値調整
        adjustments['adaptive_threshold'] = {
            'adjustment': -0.01, 
            'reason': 'leverage_improvement'
        }
        
        return adjustments
    
    def _generate_method_specific_adjustments(self, quality_method: str, 
                                            feedback_entry: Dict) -> Dict[str, Any]:
        """品質手法固有の調整"""
        adjustments = {}
        
        user_score = feedback_entry['evaluation_data']['user_rating']
        auto_score = feedback_entry['processing_results']['auto_quality_score']
        
        if quality_method == 'balanced':
            if user_score < auto_score - 0.2:
                adjustments['balanced_weight_adjustment'] = {
                    'adjustment': {'size_weight': -0.1, 'confidence_weight': 0.1},
                    'reason': 'balanced_method_recalibration'
                }
        
        elif quality_method == 'size_priority':
            if user_score > auto_score + 0.15:
                adjustments['size_priority_enhancement'] = {
                    'adjustment': {'size_threshold': 0.05},
                    'reason': 'size_priority_underperforming'
                }
        
        return adjustments
    
    def _evaluate_adjustment_confidence(self, adjustments: Dict, 
                                      trend_analysis: Dict) -> float:
        """調整信頼性評価"""
        confidence = 0.5  # ベース信頼性
        
        # データ量による信頼性調整
        data_points = trend_analysis.get('data_points_analyzed', 0)
        if data_points >= 10:
            confidence += 0.2
        elif data_points >= 5:
            confidence += 0.1
        
        # トレンドの統計的有意性
        if trend_analysis.get('is_statistically_significant', False):
            confidence += 0.2
        
        # 調整の一貫性
        if len(adjustments) > 0:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _generate_adjustment_rationale(self, adjustments: Dict, 
                                     trend_analysis: Dict) -> str:
        """調整根拠説明生成"""
        if not adjustments:
            return "no_adjustments_needed"
        
        trend_direction = trend_analysis.get('trend_direction', 'unknown')
        
        if trend_direction == 'degrading':
            return "performance_degradation_detected"
        elif trend_direction == 'improving':
            return "performance_improvement_leveraging"
        else:
            return "evaluation_disagreement_compensation"
    
    def _improve_quality_prediction(self, feedback_entry: Dict) -> Dict[str, Any]:
        """品質予測改善"""
        try:
            if len(self.improvement_history) < 5:
                return {'status': 'insufficient_data_for_prediction'}
            
            # 特徴量・ターゲット準備
            features, targets = self._prepare_prediction_data()
            
            if HAS_SKLEARN and len(features) > 0:
                # 機械学習ベース予測改善
                prediction_improvement = self._ml_based_prediction_improvement(features, targets)
            else:
                # フォールバック：統計ベース改善
                prediction_improvement = self._statistical_prediction_improvement(features, targets)
            
            # 現在エントリでの予測精度評価
            current_prediction_accuracy = self._evaluate_current_prediction(feedback_entry)
            
            prediction_improvement.update({
                'current_prediction_accuracy': current_prediction_accuracy,
                'prediction_improvement_available': prediction_improvement.get('model_r2', 0) > 0.3
            })
            
            return prediction_improvement
            
        except Exception as e:
            return {'error': f'Prediction improvement failed: {str(e)}'}
    
    def _prepare_prediction_data(self) -> Tuple[List[List[float]], List[float]]:
        """予測用データ準備"""
        features = []
        targets = []
        
        for entry in self.improvement_history:
            # 特徴量：処理パラメータと結果メトリクス
            feature_vector = [
                entry['processing_results'].get('auto_quality_score', 0.0),
                entry['processing_results'].get('processing_time', 0.0),
                len(entry['processing_parameters'].get('enhancement_applied', [])),
                entry['result_metrics'].get('confidence_score', 0.0),
                entry['result_metrics'].get('boundary_quality', 0.0)
            ]
            
            # ターゲット：ユーザー評価
            target = entry['evaluation_data'].get('user_rating', 0.0)
            
            features.append(feature_vector)
            targets.append(target)
        
        return features, targets
    
    def _ml_based_prediction_improvement(self, features: List[List[float]], 
                                       targets: List[float]) -> Dict[str, Any]:
        """機械学習ベース予測改善"""
        try:
            X = np.array(features)
            y = np.array(targets)
            
            # データ正規化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # リッジ回帰でオーバーフィッティング抑制
            model = Ridge(alpha=1.0)
            model.fit(X_scaled, y)
            
            # 予測精度評価
            predictions = model.predict(X_scaled)
            r2 = r2_score(y, predictions)
            mse = mean_squared_error(y, predictions)
            
            # 特徴量重要度
            feature_importance = np.abs(model.coef_)
            feature_names = ['auto_score', 'processing_time', 'enhancement_count', 
                           'confidence', 'boundary_quality']
            
            importance_ranking = sorted(zip(feature_names, feature_importance), 
                                      key=lambda x: x[1], reverse=True)
            
            return {
                'model_type': 'ridge_regression',
                'model_r2': float(r2),
                'model_mse': float(mse),
                'feature_importance': dict(importance_ranking),
                'prediction_model_available': True
            }
            
        except Exception as e:
            return {'error': f'ML prediction improvement failed: {str(e)}'}
    
    def _statistical_prediction_improvement(self, features: List[List[float]], 
                                          targets: List[float]) -> Dict[str, Any]:
        """統計ベース予測改善（フォールバック）"""
        try:
            if not features or not targets:
                return {'status': 'no_data'}
            
            # 自動スコアとユーザー評価の相関
            auto_scores = [f[0] for f in features]  # 最初の特徴量が自動スコア
            
            if HAS_SCIPY:
                correlation, p_value = stats.pearsonr(auto_scores, targets)
            else:
                # 簡易相関計算
                correlation = np.corrcoef(auto_scores, targets)[0, 1]
                p_value = 1.0  # フォールバック
            
            # 線形補正係数計算
            if len(auto_scores) > 1:
                slope = (np.mean(targets) - np.mean(auto_scores)) / np.std(auto_scores) if np.std(auto_scores) > 0 else 1.0
                intercept = np.mean(targets) - slope * np.mean(auto_scores)
            else:
                slope, intercept = 1.0, 0.0
            
            return {
                'model_type': 'statistical_correlation',
                'correlation': float(correlation),
                'p_value': float(p_value),
                'linear_correction': {'slope': float(slope), 'intercept': float(intercept)},
                'prediction_model_available': abs(correlation) > 0.3
            }
            
        except Exception as e:
            return {'error': f'Statistical prediction improvement failed: {str(e)}'}
    
    def _evaluate_current_prediction(self, feedback_entry: Dict) -> float:
        """現在予測精度評価"""
        auto_score = feedback_entry['processing_results']['auto_quality_score']
        user_score = feedback_entry['evaluation_data']['user_rating']
        
        # 予測誤差（低いほど良い）
        prediction_error = abs(auto_score - user_score)
        
        # 精度スコア（高いほど良い）
        accuracy = max(0, 1.0 - prediction_error)
        
        return accuracy
    
    def _measure_learning_effectiveness(self) -> Dict[str, Any]:
        """学習効果測定"""
        try:
            if len(self.improvement_history) < 10:
                return {'status': 'insufficient_data', 'effectiveness': 'unknown'}
            
            # 時系列での改善効果測定
            recent_period = self.improvement_history[-5:]
            earlier_period = self.improvement_history[-10:-5]
            
            # 評価差分の改善
            recent_diffs = [e['evaluation_difference']['score_difference'] for e in recent_period]
            earlier_diffs = [e['evaluation_difference']['score_difference'] for e in earlier_period]
            
            recent_avg = np.mean(recent_diffs)
            earlier_avg = np.mean(earlier_diffs)
            improvement = earlier_avg - recent_avg  # 正の値が改善
            
            # 一致度の改善
            recent_agreements = [e['evaluation_difference']['agreement_level'] for e in recent_period]
            high_agreement_ratio = sum(1 for a in recent_agreements if a in ['high_agreement', 'moderate_agreement']) / len(recent_agreements)
            
            # 学習効果評価
            if improvement > 0.02 and high_agreement_ratio > 0.6:
                effectiveness = "high"
            elif improvement > 0.01 or high_agreement_ratio > 0.4:
                effectiveness = "moderate"
            elif improvement >= 0:
                effectiveness = "low"
            else:
                effectiveness = "negative"
            
            return {
                'effectiveness': effectiveness,
                'improvement_score': float(improvement),
                'high_agreement_ratio': float(high_agreement_ratio),
                'recent_avg_difference': float(recent_avg),
                'earlier_avg_difference': float(earlier_avg),
                'feedback_entries_analyzed': len(self.improvement_history)
            }
            
        except Exception as e:
            return {'error': f'Learning effectiveness measurement failed: {str(e)}'}
    
    def _generate_improvement_recommendations(self, trend_analysis: Dict, 
                                            parameter_adjustments: Dict,
                                            prediction_improvements: Dict) -> List[str]:
        """改善推奨事項生成"""
        recommendations = []
        
        # トレンドベース推奨
        trend_direction = trend_analysis.get('trend_direction')
        if trend_direction == 'degrading':
            recommendations.append("investigate_performance_degradation")
            recommendations.append("review_recent_parameter_changes")
        elif trend_direction == 'improving':
            recommendations.append("maintain_current_improvements")
            recommendations.append("consider_parameter_optimization")
        
        # パラメータ調整推奨
        if parameter_adjustments.get('recommended_application', False):
            recommendations.append("apply_adaptive_parameter_adjustments")
        
        # 予測改善推奨
        if prediction_improvements.get('prediction_model_available', False):
            recommendations.append("integrate_improved_prediction_model")
        
        # データ収集推奨
        if len(self.improvement_history) < 20:
            recommendations.append("increase_feedback_data_collection")
        
        return recommendations
    
    def _load_feedback_history(self):
        """フィードバック履歴読み込み"""
        try:
            if self.feedback_data_path.exists():
                with open(self.feedback_data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            entry = json.loads(line.strip())
                            self.improvement_history.append(entry)
                            
                print(f"📚 フィードバック履歴読み込み: {len(self.improvement_history)}件")
            else:
                print(f"📝 新規フィードバック履歴を開始: {self.feedback_data_path}")
                # ディレクトリ作成
                self.feedback_data_path.parent.mkdir(parents=True, exist_ok=True)
                
        except Exception as e:
            print(f"⚠️ フィードバック履歴読み込みエラー: {e}")
            self.improvement_history = []
    
    def _save_feedback_entry(self, feedback_entry: Dict):
        """フィードバックエントリ保存"""
        try:
            # ディレクトリ作成
            self.feedback_data_path.parent.mkdir(parents=True, exist_ok=True)
            
            # JSONL形式で追記
            with open(self.feedback_data_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(feedback_entry, ensure_ascii=False) + '\n')
            
            # メモリ内履歴も更新
            self.improvement_history.append(feedback_entry)
            
        except Exception as e:
            print(f"⚠️ フィードバックエントリ保存エラー: {e}")
    
    def _generate_error_result(self, error_message: str) -> Dict[str, Any]:
        """エラー結果生成"""
        return {
            'error': error_message,
            'feedback_integration': {
                'integration_success': False
            }
        }


def main():
    """テスト実行"""
    print("🚀 P1-016: フィードバックループシステム テスト開始")
    
    # テスト用評価データ作成
    test_evaluation = {
        'user_rating': 0.75,
        'comments': 'Good extraction but some background noise',
        'aspects': {'quality': 0.8, 'completeness': 0.7}
    }
    
    test_processing = {
        'quality_score': 0.68,
        'processing_time': 3.2,
        'success': True,
        'quality_method': 'balanced',
        'enhancement_applied': ['contrast_enhancement'],
        'yolo_score_threshold': 0.05
    }
    
    # フィードバックループシステムテスト
    feedback_system = FeedbackLoopSystem()
    result = feedback_system.integrate_evaluation_feedback(test_evaluation, test_processing)
    
    print("\n📊 フィードバックループ統合結果:")
    if 'error' not in result:
        integration = result.get('feedback_integration', {})
        print(f"  統合成功: {integration.get('integration_success', False)}")
        print(f"  エントリID: {integration.get('entry_id', 'N/A')}")
        
        # 改善推奨事項
        recommendations = result.get('improvement_recommendations', [])
        if recommendations:
            print(f"  改善推奨: {', '.join(recommendations)}")
        
        # 学習効果
        learning = result.get('learning_effectiveness', {})
        if 'effectiveness' in learning:
            print(f"  学習効果: {learning['effectiveness']}")
    else:
        print(f"  ❌ エラー: {result['error']}")
    
    print(f"\n✅ [P1-016] フィードバックループシステム完了")


if __name__ == "__main__":
    main()