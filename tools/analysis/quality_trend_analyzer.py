#!/usr/bin/env python3
"""
T-004: 品質トレンド分析システム
時系列品質変化の追跡・予測・異常検知機能
"""

import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class QualityTrendAnalyzer:
    """品質トレンド分析システム"""
    
    def __init__(self, workspace_base: str = "/mnt/c/AItools/lora/train/yado/tracker-workspace"):
        """初期化"""
        self.workspace_base = Path(workspace_base)
        self.quality_data = []
        self.trend_model = None
        self.scaler = StandardScaler()
        
    def collect_quality_data(self) -> List[Dict[str, Any]]:
        """品質データの包括収集"""
        logger.info("品質データ収集開始")
        
        quality_records = []
        
        # 1. ワークスペース品質レポート収集
        for tracker_dir in self.workspace_base.glob("*/"):
            tracker_id = tracker_dir.name
            
            # 品質レポートファイル検索
            quality_files = []
            quality_files.extend(list(tracker_dir.glob("**/unified_quality_report_*.json")))
            quality_files.extend(list(tracker_dir.glob("**/quality_report_*.json")))
            quality_files.extend(list(tracker_dir.glob("quality/*.json")))
            quality_files.extend(list(tracker_dir.glob("**/dashboard_data.json")))
            
            for quality_file in quality_files:
                try:
                    record = self._parse_quality_file(quality_file, tracker_id)
                    if record:
                        quality_records.append(record)
                except Exception as e:
                    logger.warning(f"品質ファイル解析失敗: {quality_file} - {e}")
                    continue
        
        # 2. Google Sheets統計データ（将来拡張用）
        # TODO: Google Sheets APIから統計分析データを取得
        
        # 3. ログファイルから品質情報抽出
        quality_records.extend(self._extract_from_logs())
        
        self.quality_data = sorted(quality_records, key=lambda x: x['timestamp'])
        logger.info(f"品質データ収集完了: {len(self.quality_data)}件")
        
        return self.quality_data
    
    def _parse_quality_file(self, file_path: Path, tracker_id: str) -> Optional[Dict[str, Any]]:
        """品質ファイル解析"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # タイムスタンプ抽出
            timestamp = None
            if 'timestamp' in data:
                timestamp = pd.to_datetime(data['timestamp'])
            elif 'created_at' in data:
                timestamp = pd.to_datetime(data['created_at'])
            else:
                # ファイル名から推測
                import re
                match = re.search(r'(\d{8}_\d{6})', file_path.name)
                if match:
                    timestamp = pd.to_datetime(match.group(1), format='%Y%m%d_%H%M%S')
                else:
                    timestamp = pd.to_datetime(file_path.stat().st_mtime, unit='s')
            
            # 品質スコア抽出（複数パターンに対応）
            quality_score = None
            success_count = 0
            total_count = 0
            
            # パターン1: overall_quality_score
            if 'overall_quality_score' in data:
                quality_score = float(data['overall_quality_score'])
            
            # パターン2: quality_metrics.overall_score
            elif 'quality_metrics' in data:
                metrics = data['quality_metrics']
                if 'overall_score' in metrics:
                    quality_score = float(metrics['overall_score'])
                elif 'average_score' in metrics:
                    quality_score = float(metrics['average_score'])
            
            # パターン3: summary統計
            elif 'summary' in data:
                summary = data['summary']
                if 'average_score' in summary:
                    quality_score = float(summary['average_score'])
                if 'success_count' in summary and 'total_count' in summary:
                    success_count = int(summary['success_count'])
                    total_count = int(summary['total_count'])
                    if quality_score is None and total_count > 0:
                        quality_score = success_count / total_count
            
            # パターン4: 抽出結果統計
            elif 'extraction_results' in data:
                results = data['extraction_results']
                if isinstance(results, list):
                    total_count = len(results)
                    success_count = sum(1 for r in results if r.get('quality_score', 0) > 0.5)
                    if total_count > 0:
                        quality_score = success_count / total_count
            
            if quality_score is not None:
                return {
                    'tracker_id': tracker_id,
                    'timestamp': timestamp,
                    'quality_score': quality_score,
                    'success_count': success_count,
                    'total_count': total_count,
                    'file_path': str(file_path),
                    'source': 'quality_report'
                }
            
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            logger.warning(f"ファイル解析エラー {file_path}: {e}")
            
        return None
    
    def _extract_from_logs(self) -> List[Dict[str, Any]]:
        """ログファイルから品質情報抽出"""
        log_records = []
        
        # ログファイル検索
        log_patterns = [
            "*.log",
            "**/logs/*.log", 
            "**/extraction/*.log",
            "deprecated/untracked_files/logs_current/*.log"
        ]
        
        log_files = []
        for pattern in log_patterns:
            log_files.extend(self.workspace_base.parent.glob(pattern))
        
        for log_file in log_files:
            try:
                records = self._parse_log_file(log_file)
                log_records.extend(records)
            except Exception as e:
                logger.warning(f"ログ解析失敗: {log_file} - {e}")
                continue
        
        return log_records
    
    def _parse_log_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """ログファイル解析（品質情報抽出）"""
        records = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # 品質関連パターンを検索
            import re
            
            for line in lines:
                # パターン1: 成功率情報 "XX/YY件成功"
                match = re.search(r'(\d+)/(\d+)件成功', line)
                if match:
                    success = int(match.group(1))
                    total = int(match.group(2))
                    quality_score = success / total if total > 0 else 0
                    
                    # タイムスタンプ抽出
                    timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                    if timestamp_match:
                        timestamp = pd.to_datetime(timestamp_match.group(1))
                    else:
                        timestamp = pd.to_datetime(file_path.stat().st_mtime, unit='s')
                    
                    records.append({
                        'tracker_id': file_path.stem.replace('_extraction', '').replace('.log', ''),
                        'timestamp': timestamp,
                        'quality_score': quality_score,
                        'success_count': success,
                        'total_count': total,
                        'file_path': str(file_path),
                        'source': 'log_file'
                    })
                
                # パターン2: 品質スコア直接記載 "Quality: 0.XXX"
                match = re.search(r'Quality:\s*([0-9.]+)', line)
                if match:
                    quality_score = float(match.group(1))
                    
                    timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                    timestamp = pd.to_datetime(timestamp_match.group(1)) if timestamp_match else pd.to_datetime(file_path.stat().st_mtime, unit='s')
                    
                    records.append({
                        'tracker_id': file_path.stem,
                        'timestamp': timestamp,
                        'quality_score': quality_score,
                        'success_count': None,
                        'total_count': None,
                        'file_path': str(file_path),
                        'source': 'log_file'
                    })
        
        except Exception as e:
            logger.warning(f"ログファイル読み込みエラー: {file_path} - {e}")
        
        return records
    
    def analyze_trends(self) -> Dict[str, Any]:
        """包括的トレンド分析"""
        if not self.quality_data:
            self.collect_quality_data()
        
        if len(self.quality_data) < 2:
            return {
                'error': '分析に十分なデータがありません（最低2件必要）',
                'data_count': len(self.quality_data)
            }
        
        logger.info("トレンド分析開始")
        
        # データフレーム作成
        df = pd.DataFrame(self.quality_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # 基本統計
        basic_stats = {
            'total_records': len(df),
            'unique_trackers': df['tracker_id'].nunique(),
            'time_range': {
                'start': df['timestamp'].min().isoformat(),
                'end': df['timestamp'].max().isoformat(),
                'duration_days': (df['timestamp'].max() - df['timestamp'].min()).days
            },
            'quality_stats': {
                'mean': float(df['quality_score'].mean()),
                'std': float(df['quality_score'].std()),
                'min': float(df['quality_score'].min()),
                'max': float(df['quality_score'].max()),
                'median': float(df['quality_score'].median())
            }
        }
        
        # トレンド分析
        trend_analysis = self._calculate_trend(df)
        
        # トラッカー別分析
        tracker_analysis = self._analyze_by_tracker(df)
        
        # 時系列パターン分析
        temporal_analysis = self._analyze_temporal_patterns(df)
        
        # 異常検知
        anomalies = self._detect_anomalies(df)
        
        # 品質予測
        predictions = self._predict_quality_trend(df)
        
        analysis_result = {
            'analysis_timestamp': datetime.now().isoformat(),
            'basic_statistics': basic_stats,
            'trend_analysis': trend_analysis,
            'tracker_analysis': tracker_analysis,
            'temporal_patterns': temporal_analysis,
            'anomalies': anomalies,
            'predictions': predictions,
            'data_quality_assessment': self._assess_data_quality(df)
        }
        
        logger.info("トレンド分析完了")
        return analysis_result
    
    def _calculate_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """トレンド計算"""
        # 線形回帰によるトレンド分析
        df['days_since_start'] = (df['timestamp'] - df['timestamp'].min()).dt.days
        
        if len(df) >= 3:
            X = df[['days_since_start']].values
            y = df['quality_score'].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            y_pred = model.predict(X)
            
            return {
                'slope': float(model.coef_[0]),
                'intercept': float(model.intercept_),
                'r_squared': float(r2_score(y, y_pred)),
                'trend_direction': 'improving' if model.coef_[0] > 0.01 else 'declining' if model.coef_[0] < -0.01 else 'stable',
                'trend_strength': abs(float(model.coef_[0])),
                'prediction_accuracy': float(r2_score(y, y_pred))
            }
        else:
            # データ不足時は簡易分析
            if len(df) == 2:
                slope = (df.iloc[-1]['quality_score'] - df.iloc[0]['quality_score']) / max(1, df.iloc[-1]['days_since_start'])
                return {
                    'slope': float(slope),
                    'trend_direction': 'improving' if slope > 0 else 'declining',
                    'trend_strength': abs(float(slope)),
                    'note': 'Limited data - simple slope calculation'
                }
        
        return {'error': 'Insufficient data for trend analysis'}
    
    def _analyze_by_tracker(self, df: pd.DataFrame) -> Dict[str, Any]:
        """トラッカー別分析"""
        tracker_stats = {}
        
        for tracker_id in df['tracker_id'].unique():
            tracker_data = df[df['tracker_id'] == tracker_id]
            
            if len(tracker_data) >= 2:
                tracker_stats[tracker_id] = {
                    'record_count': len(tracker_data),
                    'avg_quality': float(tracker_data['quality_score'].mean()),
                    'quality_std': float(tracker_data['quality_score'].std()),
                    'latest_quality': float(tracker_data.iloc[-1]['quality_score']),
                    'improvement': float(tracker_data.iloc[-1]['quality_score'] - tracker_data.iloc[0]['quality_score']),
                    'time_span_days': (tracker_data['timestamp'].max() - tracker_data['timestamp'].min()).days
                }
        
        return {
            'tracker_statistics': tracker_stats,
            'best_performing': max(tracker_stats.items(), key=lambda x: x[1]['avg_quality'])[0] if tracker_stats else None,
            'most_improved': max(tracker_stats.items(), key=lambda x: x[1]['improvement'])[0] if tracker_stats else None
        }
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """時系列パターン分析"""
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_of_week
        df['month'] = df['timestamp'].dt.month
        
        patterns = {}
        
        # 時間帯パターン
        if len(df) >= 5:
            hourly_stats = df.groupby('hour')['quality_score'].agg(['mean', 'count']).reset_index()
            patterns['hourly'] = {
                'best_hours': hourly_stats.nlargest(3, 'mean')[['hour', 'mean']].to_dict('records'),
                'peak_activity_hours': hourly_stats.nlargest(3, 'count')[['hour', 'count']].to_dict('records')
            }
            
            # 曜日パターン
            weekly_stats = df.groupby('day_of_week')['quality_score'].agg(['mean', 'count']).reset_index()
            patterns['weekly'] = {
                'best_days': weekly_stats.nlargest(3, 'mean')[['day_of_week', 'mean']].to_dict('records'),
                'peak_activity_days': weekly_stats.nlargest(3, 'count')[['day_of_week', 'count']].to_dict('records')
            }
        
        return patterns
    
    def _detect_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """異常検知"""
        if len(df) < 5:
            return []
        
        # 統計的異常検知（Z-score）
        mean_score = df['quality_score'].mean()
        std_score = df['quality_score'].std()
        threshold = 2.0  # 2σを閾値
        
        anomalies = []
        
        for idx, row in df.iterrows():
            z_score = abs(row['quality_score'] - mean_score) / std_score if std_score > 0 else 0
            
            if z_score > threshold:
                anomalies.append({
                    'tracker_id': row['tracker_id'],
                    'timestamp': row['timestamp'].isoformat(),
                    'quality_score': float(row['quality_score']),
                    'z_score': float(z_score),
                    'anomaly_type': 'outlier_high' if row['quality_score'] > mean_score else 'outlier_low',
                    'severity': 'high' if z_score > 3.0 else 'medium'
                })
        
        return sorted(anomalies, key=lambda x: x['z_score'], reverse=True)
    
    def _predict_quality_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """品質トレンド予測"""
        if len(df) < 3:
            return {'error': 'Insufficient data for prediction'}
        
        try:
            # 時系列データ準備
            df['days_since_start'] = (df['timestamp'] - df['timestamp'].min()).dt.days
            
            X = df[['days_since_start']].values
            y = df['quality_score'].values
            
            # 線形回帰モデル学習
            model = LinearRegression()
            model.fit(X, y)
            
            # 未来予測（30日後まで）
            future_days = np.array([[df['days_since_start'].max() + i] for i in range(1, 31)])
            future_predictions = model.predict(future_days)
            
            # 予測結果
            predictions = []
            base_date = df['timestamp'].max()
            
            for i, pred in enumerate(future_predictions):
                future_date = base_date + timedelta(days=i+1)
                predictions.append({
                    'date': future_date.isoformat(),
                    'predicted_quality': max(0.0, min(1.0, float(pred))),  # 0-1の範囲に制限
                    'confidence': max(0.1, 1.0 - (i * 0.02))  # 時間が経つほど信頼度低下
                })
            
            return {
                'predictions': predictions[:7],  # 7日分のみ返却
                'model_accuracy': float(r2_score(y, model.predict(X))),
                'trend_slope': float(model.coef_[0]),
                'prediction_method': 'linear_regression'
            }
            
        except Exception as e:
            return {'error': f'Prediction failed: {str(e)}'}
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """データ品質評価"""
        # 品質スコアの有効範囲チェック
        valid_scores = ((df['quality_score'] >= 0) & (df['quality_score'] <= 1)).sum()
        quality_score_validity = float(valid_scores / len(df)) if len(df) > 0 else 0.0
        
        return {
            'completeness': {
                'total_records': len(df),
                'missing_quality_scores': int(df['quality_score'].isna().sum()),
                'missing_timestamps': int(df['timestamp'].isna().sum()),
                'completeness_rate': float((len(df) - df.isna().sum().sum()) / (len(df) * len(df.columns))) if len(df) > 0 else 0.0
            },
            'consistency': {
                'quality_score_range_valid': quality_score_validity,
                'timestamp_order_correct': bool(df['timestamp'].is_monotonic_increasing)
            },
            'coverage': {
                'time_span_days': int((df['timestamp'].max() - df['timestamp'].min()).days) if len(df) > 0 else 0,
                'unique_trackers': int(df['tracker_id'].nunique()),
                'average_records_per_tracker': float(len(df) / df['tracker_id'].nunique()) if df['tracker_id'].nunique() > 0 else 0.0
            }
        }
    
    def generate_trend_report(self, output_path: str = "/tmp/t004_quality_trend_report.json") -> str:
        """トレンド分析レポート生成"""
        logger.info("品質トレンド分析レポート生成開始")
        
        analysis_result = self.analyze_trends()
        
        # レポート拡張
        report = {
            'report_metadata': {
                'title': 'T-004: 品質トレンド分析レポート',
                'generated_at': datetime.now().isoformat(),
                'analyzer_version': '1.0.0',
                'data_sources': ['quality_reports', 'log_files', 'dashboard_data']
            },
            'executive_summary': self._generate_executive_summary(analysis_result),
            'detailed_analysis': analysis_result,
            'recommendations': self._generate_recommendations(analysis_result),
            'visualization_data': self._prepare_visualization_data()
        }
        
        # ファイル保存
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"品質トレンド分析レポート保存完了: {output_path}")
        return output_path
    
    def _generate_executive_summary(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """エグゼクティブサマリー生成"""
        if 'error' in analysis:
            return {
                'status': 'insufficient_data',
                'message': analysis['error'],
                'data_count': analysis.get('data_count', 0)
            }
        
        basic_stats = analysis['basic_statistics']
        trend = analysis.get('trend_analysis', {})
        
        summary = {
            'overall_status': 'healthy' if basic_stats['quality_stats']['mean'] > 0.7 else 'needs_attention',
            'data_coverage': {
                'total_records': basic_stats['total_records'],
                'time_span_days': basic_stats['time_range']['duration_days'],
                'tracker_count': basic_stats['unique_trackers']
            },
            'quality_overview': {
                'current_average': round(basic_stats['quality_stats']['mean'], 3),
                'quality_range': f"{basic_stats['quality_stats']['min']:.3f} - {basic_stats['quality_stats']['max']:.3f}",
                'stability': 'stable' if basic_stats['quality_stats']['std'] < 0.2 else 'variable'
            },
            'trend_summary': {
                'direction': trend.get('trend_direction', 'unknown'),
                'strength': trend.get('trend_strength', 0),
                'reliability': trend.get('prediction_accuracy', 0)
            }
        }
        
        # 主要な発見
        key_findings = []
        
        if trend.get('trend_direction') == 'improving':
            key_findings.append("品質が継続的に改善傾向にあります")
        elif trend.get('trend_direction') == 'declining':
            key_findings.append("品質の低下傾向が検出されました - 要注意")
        
        if analysis.get('anomalies'):
            key_findings.append(f"{len(analysis['anomalies'])}件の異常値を検出")
        
        summary['key_findings'] = key_findings
        
        return summary
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """推奨事項生成"""
        recommendations = []
        
        if 'error' in analysis:
            recommendations.append({
                'priority': 'high',
                'category': 'data_collection',
                'title': 'データ収集の改善',
                'description': '品質トレンド分析のためにより多くの品質データの収集が必要です',
                'action': '品質レポート生成の自動化とデータ蓄積システムの構築を検討してください'
            })
            return recommendations
        
        basic_stats = analysis['basic_statistics']
        trend = analysis.get('trend_analysis', {})
        anomalies = analysis.get('anomalies', [])
        
        # 品質レベルに基づく推奨事項
        if basic_stats['quality_stats']['mean'] < 0.5:
            recommendations.append({
                'priority': 'high',
                'category': 'quality_improvement',
                'title': '品質向上対策の実施',
                'description': f"平均品質スコア{basic_stats['quality_stats']['mean']:.3f}は改善が必要です",
                'action': 'SAM+YOLOパラメータの再調整、前処理手法の見直しを実施してください'
            })
        
        # トレンドに基づく推奨事項
        if trend.get('trend_direction') == 'declining':
            recommendations.append({
                'priority': 'high',
                'category': 'trend_monitoring',
                'title': '品質低下の原因調査',
                'description': '品質の低下傾向が検出されています',
                'action': '最近の設定変更や環境変化を調査し、品質低下の原因を特定してください'
            })
        
        # 異常値に基づく推奨事項
        if len(anomalies) > 0:
            high_severity_count = sum(1 for a in anomalies if a['severity'] == 'high')
            if high_severity_count > 0:
                recommendations.append({
                    'priority': 'medium',
                    'category': 'anomaly_investigation',
                    'title': '異常値の詳細調査',
                    'description': f'{high_severity_count}件の高重要度異常値を検出',
                    'action': '異常値が発生したトラッカーと時期を詳細調査してください'
                })
        
        # データ品質に基づる推奨事項
        data_quality = analysis.get('data_quality_assessment', {})
        if data_quality.get('completeness', {}).get('completeness_rate', 1.0) < 0.8:
            recommendations.append({
                'priority': 'medium',
                'category': 'data_quality',
                'title': 'データ品質の向上',
                'description': 'データの欠損率が高く分析精度に影響しています',
                'action': '品質レポート生成プロセスの見直しとデータ検証機能の強化を検討してください'
            })
        
        return recommendations
    
    def _prepare_visualization_data(self) -> Dict[str, Any]:
        """可視化用データ準備"""
        if not self.quality_data:
            return {}
        
        df = pd.DataFrame(self.quality_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 日次集計（日付をISOフォーマット文字列に変換）
        daily_agg = df.groupby(df['timestamp'].dt.date)['quality_score'].agg(['mean', 'std', 'count'])
        daily_aggregates = {}
        for date_key, row in daily_agg.iterrows():
            daily_aggregates[date_key.isoformat()] = {
                'mean': float(row['mean']),
                'std': float(row['std']) if not pd.isna(row['std']) else 0.0,
                'count': int(row['count'])
            }
        
        return {
            'time_series_data': [
                {
                    'timestamp': row['timestamp'].isoformat(),
                    'quality_score': float(row['quality_score']),
                    'tracker_id': row['tracker_id']
                }
                for _, row in df.iterrows()
            ],
            'tracker_summary': {
                tracker: {
                    'mean': float(stats['mean']),
                    'count': int(stats['count'])
                }
                for tracker, stats in df.groupby('tracker_id')['quality_score'].agg(['mean', 'count']).to_dict('index').items()
            },
            'daily_aggregates': daily_aggregates
        }


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='T-004: 品質トレンド分析システム')
    parser.add_argument('--workspace', default="/mnt/c/AItools/lora/train/yado/tracker-workspace",
                       help='ワークスペースディレクトリ')
    parser.add_argument('--output', default="/tmp/t004_quality_trend_report.json",
                       help='出力レポートファイル')
    parser.add_argument('--verbose', action='store_true', help='詳細ログ出力')
    
    args = parser.parse_args()
    
    # ログ設定
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 分析実行
    analyzer = QualityTrendAnalyzer(args.workspace)
    report_path = analyzer.generate_trend_report(args.output)
    
    print(f"🎯 T-004品質トレンド分析完了")
    print(f"📊 レポート: {report_path}")
    
    # 結果サマリー表示
    with open(report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)
    
    summary = report.get('executive_summary', {})
    if 'status' in summary:
        print(f"📈 ステータス: {summary['status']}")
        if summary['status'] == 'insufficient_data':
            print(f"⚠️  {summary['message']}")
        else:
            print(f"📊 品質平均: {summary['quality_overview']['current_average']}")
            print(f"📈 トレンド: {summary['trend_summary']['direction']}")
            
            if summary.get('key_findings'):
                print("🔍 主要な発見:")
                for finding in summary['key_findings']:
                    print(f"   - {finding}")


if __name__ == "__main__":
    main()