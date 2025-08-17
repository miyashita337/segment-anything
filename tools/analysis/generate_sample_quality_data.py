#!/usr/bin/env python3
"""
T-004: サンプル品質データ生成ツール
トレンド分析システムのテスト用データ生成
"""

import json
import random
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any


def generate_sample_quality_data(num_trackers: int = 5, days: int = 30, records_per_day: int = 2) -> List[Dict[str, Any]]:
    """サンプル品質データ生成"""
    
    quality_data = []
    start_date = datetime.now() - timedelta(days=days)
    
    tracker_names = [f"TEST-{i:03d}" for i in range(1, num_trackers + 1)]
    
    # トラッカー別の品質トレンド設定
    tracker_trends = {}
    for tracker in tracker_names:
        base_quality = random.uniform(0.4, 0.8)
        trend_slope = random.uniform(-0.01, 0.02)  # 日次変化率
        noise_level = random.uniform(0.05, 0.15)
        tracker_trends[tracker] = {
            'base': base_quality,
            'slope': trend_slope,
            'noise': noise_level
        }
    
    # データ生成
    for day in range(days):
        current_date = start_date + timedelta(days=day)
        
        for _ in range(records_per_day):
            # ランダムにトラッカー選択
            tracker = random.choice(tracker_names)
            trend = tracker_trends[tracker]
            
            # 時間経過による品質変化
            time_factor = day / days
            quality_score = trend['base'] + trend['slope'] * day
            
            # ノイズ追加
            quality_score += np.random.normal(0, trend['noise'])
            
            # 異常値を時々追加（5%の確率）
            if random.random() < 0.05:
                if random.random() < 0.5:
                    quality_score *= 0.5  # 異常に低い値
                else:
                    quality_score *= 1.3  # 異常に高い値
            
            # 0-1の範囲に制限
            quality_score = max(0.0, min(1.0, quality_score))
            
            # タイムスタンプにランダムな時刻を追加
            timestamp = current_date + timedelta(
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            
            # データレコード作成
            record = {
                'tracker_id': tracker,
                'timestamp': timestamp.isoformat(),
                'quality_score': quality_score,
                'success_count': int(quality_score * 100),
                'total_count': 100,
                'source': 'sample_generator'
            }
            
            quality_data.append(record)
    
    return sorted(quality_data, key=lambda x: x['timestamp'])


def create_sample_quality_reports(workspace_base: str = "/mnt/c/AItools/lora/train/yado/tracker-workspace"):
    """サンプル品質レポートファイル作成"""
    
    workspace_path = Path(workspace_base)
    
    # サンプルデータ生成
    quality_data = generate_sample_quality_data(num_trackers=5, days=30, records_per_day=2)
    
    # トラッカー別にグループ化
    tracker_groups = {}
    for record in quality_data:
        tracker_id = record['tracker_id']
        if tracker_id not in tracker_groups:
            tracker_groups[tracker_id] = []
        tracker_groups[tracker_id].append(record)
    
    # 各トラッカーのディレクトリに品質レポート作成
    created_files = []
    
    for tracker_id, records in tracker_groups.items():
        # トラッカーディレクトリ作成
        tracker_dir = workspace_path / tracker_id / "quality"
        tracker_dir.mkdir(parents=True, exist_ok=True)
        
        # 複数の品質レポートファイル作成
        for i, record in enumerate(records[:5]):  # 各トラッカー最大5ファイル
            timestamp_str = datetime.fromisoformat(record['timestamp']).strftime('%Y%m%d_%H%M%S')
            report_file = tracker_dir / f"quality_report_{timestamp_str}.json"
            
            # レポートデータ作成
            report_data = {
                'timestamp': record['timestamp'],
                'tracker_id': tracker_id,
                'overall_quality_score': record['quality_score'],
                'summary': {
                    'success_count': record['success_count'],
                    'total_count': record['total_count'],
                    'average_score': record['quality_score']
                },
                'quality_metrics': {
                    'overall_score': record['quality_score'],
                    'confidence': random.uniform(0.7, 0.95),
                    'coverage': random.uniform(0.6, 0.9)
                },
                'extraction_results': [
                    {
                        'image_id': f"img_{j:03d}",
                        'quality_score': random.uniform(0.3, 1.0),
                        'status': 'success' if random.random() > 0.3 else 'failed'
                    }
                    for j in range(10)
                ]
            }
            
            # ファイル保存
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            created_files.append(str(report_file))
    
    # 統合レポート作成
    summary_file = workspace_path / "sample_quality_summary.json"
    summary_data = {
        'generated_at': datetime.now().isoformat(),
        'total_records': len(quality_data),
        'trackers': list(tracker_groups.keys()),
        'files_created': created_files,
        'statistics': {
            'mean_quality': np.mean([r['quality_score'] for r in quality_data]),
            'std_quality': np.std([r['quality_score'] for r in quality_data]),
            'min_quality': min(r['quality_score'] for r in quality_data),
            'max_quality': max(r['quality_score'] for r in quality_data)
        }
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    
    return created_files, summary_file


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='T-004: サンプル品質データ生成')
    parser.add_argument('--workspace', default="/mnt/c/AItools/lora/train/yado/tracker-workspace",
                       help='ワークスペースディレクトリ')
    parser.add_argument('--trackers', type=int, default=5, help='トラッカー数')
    parser.add_argument('--days', type=int, default=30, help='データ生成日数')
    parser.add_argument('--records-per-day', type=int, default=2, help='1日あたりのレコード数')
    
    args = parser.parse_args()
    
    print("🔄 サンプル品質データ生成開始...")
    
    # サンプルレポート作成
    created_files, summary_file = create_sample_quality_reports(args.workspace)
    
    print(f"✅ サンプル品質データ生成完了")
    print(f"📊 作成ファイル数: {len(created_files)}")
    print(f"📋 サマリーファイル: {summary_file}")
    
    # 統計表示
    with open(summary_file, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    print(f"\n📈 生成データ統計:")
    print(f"   総レコード数: {summary['total_records']}")
    print(f"   トラッカー数: {len(summary['trackers'])}")
    print(f"   平均品質スコア: {summary['statistics']['mean_quality']:.3f}")
    print(f"   標準偏差: {summary['statistics']['std_quality']:.3f}")
    print(f"   範囲: {summary['statistics']['min_quality']:.3f} - {summary['statistics']['max_quality']:.3f}")
    
    print(f"\n💡 次のステップ:")
    print(f"   1. python3 tools/analysis/quality_trend_analyzer.py --verbose")
    print(f"   2. python3 tools/analysis/quality_trend_dashboard.py --verbose")
    print(f"   3. ブラウザで file:///tmp/t004_quality_trend_dashboard.html を開く")


if __name__ == "__main__":
    main()