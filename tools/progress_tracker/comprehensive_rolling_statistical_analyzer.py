#!/usr/bin/env python3
"""
包括的ローリング統計分析システム

全完了トラッカー(/release + 更新日付あり)を時系列順にソートし、
隣接ペアごとにウェルチのt検定を実行してGoogle Sheetsに更新。

新列構成対応:
X列: Current, Y列: Baseline, Z列: p値, AA列: Cohen's d, AB列: 改善率, AC列: 統計的有意性
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import re

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).parent.parent.parent))

from features.evaluation.statistical_quality_analyzer import StatisticalQualityAnalyzer
from tools.progress_tracker.sheets_client import GoogleSheetsClient
from tools.progress_tracker.config import get_default_config


class ComprehensiveRollingStatisticalAnalyzer:
    """包括的ローリング統計分析クラス"""
    
    def __init__(self):
        # QCC-022 StatisticalQualityAnalyzer初期化
        self.statistical_analyzer = StatisticalQualityAnalyzer()
        
        # Google Sheetsクライアント
        self.config = get_default_config()
        self.sheets_client = GoogleSheetsClient(self.config)
        
        print("🔬 包括的ローリング統計分析システム初期化完了")
    
    def extract_completed_trackers_chronological(self) -> List[Dict]:
        """
        完了トラッカーを時系列順で抽出
        
        Returns:
            List[Dict]: 時系列順の完了トラッカー情報
        """
        print("📊 完了トラッカー抽出開始（/release + 更新日付あり）\\n")
        
        # Google Sheets全データ取得
        all_values = self.sheets_client.get_sheet_values('A:E')
        if not all_values:
            raise ValueError("Google Sheetsからデータ取得失敗")
        
        completed_trackers = []
        
        for i, row in enumerate(all_values[1:], 2):  # ヘッダーをスキップして2行目から
            if row and len(row) > 4:
                tracker_id = row[0] if len(row) > 0 else ''
                status = row[2] if len(row) > 2 else ''
                update_date = row[4] if len(row) > 4 else ''
                
                # 条件: /release かつ 更新日付あり
                if '/release' in status and tracker_id and update_date and update_date.strip():
                    # 更新日付のソート用処理
                    date_for_sort = '0000-00-00'
                    parsed_date = None
                    
                    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y/%m/%d']:
                        try:
                            parsed_date = datetime.strptime(update_date, fmt)
                            date_for_sort = parsed_date.strftime('%Y-%m-%d %H:%M:%S')
                            break
                        except:
                            continue
                    
                    completed_trackers.append({
                        'tracker_id': tracker_id,
                        'status': status,
                        'update_date': update_date,
                        'parsed_date': parsed_date,
                        'date_for_sort': date_for_sort,
                        'row_number': i
                    })
        
        # 更新日付順でソート（新→旧、未設定は最後）
        completed_trackers.sort(key=lambda x: (x['date_for_sort'] != '0000-00-00', x['date_for_sort']), reverse=True)
        
        print(f"✅ 完了トラッカー抽出結果: {len(completed_trackers)}個")
        print("🗓️  更新日付順（新→旧）:")
        
        for i, tracker in enumerate(completed_trackers[:20], 1):  # 最初の20個表示
            date_display = tracker['update_date'] if tracker['update_date'] else '(未設定)'
            print(f"   {i:2}. {tracker['tracker_id']:15} | {date_display:19} | 行{tracker['row_number']}")
        
        if len(completed_trackers) > 20:
            print(f"   ... 他{len(completed_trackers)-20}個")
        
        return completed_trackers
    
    def verify_real_data_availability(self, tracker_id: str) -> Dict:
        """トラッカーの実データ利用可能性確認"""
        try:
            # QCC-022のload_extraction_resultsで読み込み試行
            metrics = self.statistical_analyzer.load_extraction_results(tracker_id)
            
            # 実データ確認（extraction_result.json）
            tracker_dir = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace") / tracker_id
            json_path = tracker_dir / "extraction_result.json"
            
            has_json = json_path.exists()
            generation_method = 'unknown'
            opencv_version = 'unknown'
            
            if has_json:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                generation_method = data.get('generation_method', 'unknown')
                opencv_version = data.get('opencv_version', 'unknown')
            
            # 品質スコア存在確認
            has_quality_scores = len(metrics.quality_scores) > 0
            sample_size = metrics.sample_size
            mean_quality = metrics.mean_quality_score
            
            # 実データ判定（opencv_analysis + 品質スコアあり）
            is_real_data = (generation_method == 'opencv_analysis' and has_quality_scores)
            
            return {
                'tracker_id': tracker_id,
                'available': is_real_data,
                'sample_size': sample_size,
                'mean_quality_score': mean_quality,
                'success_rate': metrics.success_rate,
                'generation_method': generation_method,
                'opencv_version': opencv_version,
                'has_json': has_json,
                'error': None
            }
            
        except Exception as e:
            return {
                'tracker_id': tracker_id,
                'available': False,
                'sample_size': 0,
                'mean_quality_score': 0.0,
                'success_rate': 0.0,
                'generation_method': 'error',
                'opencv_version': 'error',
                'has_json': False,
                'error': str(e)
            }
    
    def generate_adjacent_pairs(self, completed_trackers: List[Dict]) -> List[Tuple[Dict, Dict]]:
        """隣接ペア生成（時系列順）"""
        print("\\n🔗 隣接ペア生成中...")
        
        # 実データ利用可能性確認
        available_trackers = []
        unavailable_trackers = []
        
        for tracker_info in completed_trackers:
            availability = self.verify_real_data_availability(tracker_info['tracker_id'])
            
            if availability['available']:
                tracker_info.update(availability)
                available_trackers.append(tracker_info)
                print(f"   ✅ {availability['tracker_id']:15} | {availability['sample_size']:3}サンプル | 平均品質={availability['mean_quality_score']:.3f} | {availability['generation_method']}")
            else:
                unavailable_trackers.append({**tracker_info, **availability})
                error_short = availability['error'][:30] + "..." if availability['error'] and len(availability['error']) > 30 else availability['error']
                print(f"   ❌ {availability['tracker_id']:15} | エラー: {error_short}")
        
        print(f"\\n📊 実データ利用可能性:")
        print(f"   - 利用可能: {len(available_trackers)}個 ({len(available_trackers)/len(completed_trackers)*100:.1f}%)")
        print(f"   - 利用不可: {len(unavailable_trackers)}個 ({len(unavailable_trackers)/len(completed_trackers)*100:.1f}%)")
        
        if len(available_trackers) < 2:
            raise ValueError(f"統計比較に必要な最低2個のトラッカーが不足（利用可能: {len(available_trackers)}個）")
        
        # 隣接ペア生成（current, baseline）
        adjacent_pairs = []
        for i in range(len(available_trackers) - 1):
            current = available_trackers[i]    # より新しいトラッカー
            baseline = available_trackers[i + 1]  # より古いトラッカー（ベースライン）
            adjacent_pairs.append((current, baseline))
        
        print(f"\\n🎯 生成された隣接ペア: {len(adjacent_pairs)}ペア")
        for i, (current, baseline) in enumerate(adjacent_pairs[:10], 1):  # 最初の10ペアのみ表示
            print(f"   {i:2}. {current['tracker_id']:12} vs {baseline['tracker_id']:12} (ベースライン)")
        
        if len(adjacent_pairs) > 10:
            print(f"   ... 他{len(adjacent_pairs)-10}ペア")
        
        return adjacent_pairs
    
    def run_single_statistical_comparison(self, current: Dict, baseline: Dict) -> Dict:
        """単一ペアの統計比較実行"""
        try:
            # QCC-022のcompare_trackersメソッド使用（baseline, currentの順番に注意）
            t_test_result = self.statistical_analyzer.compare_trackers(
                baseline['tracker_id'], current['tracker_id'], 'quality_score'
            )
            
            # 改善効果分析も実行
            improvement_analysis = self.statistical_analyzer.analyze_improvement(
                baseline['tracker_id'], current['tracker_id']
            )
            
            # 統計的有意性テキスト生成
            if t_test_result.is_significant:
                if improvement_analysis['quality_comparison']['improvement_percent'] > 0:
                    significance_text = "✅有意改善"
                else:
                    significance_text = "🔴有意劣化"
            else:
                significance_text = "⚪有意差なし"
            
            return {
                'current_tracker': current['tracker_id'],
                'baseline_tracker': baseline['tracker_id'],
                'current_row': current['row_number'],
                'success': True,
                'p_value': t_test_result.p_value,
                'effect_size': t_test_result.effect_size,
                'is_significant': t_test_result.is_significant,
                'interpretation': t_test_result.interpretation,
                'improvement_percent': improvement_analysis['quality_comparison']['improvement_percent'],
                'current_mean': improvement_analysis['quality_comparison']['improved_mean'],
                'baseline_mean': improvement_analysis['quality_comparison']['baseline_mean'],
                'current_sample_size': improvement_analysis['sample_sizes']['improved'],
                'baseline_sample_size': improvement_analysis['sample_sizes']['baseline'],
                'significance_text': significance_text,
                'error': None
            }
            
        except Exception as e:
            return {
                'current_tracker': current['tracker_id'],
                'baseline_tracker': baseline['tracker_id'],
                'current_row': current['row_number'],
                'success': False,
                'p_value': None,
                'effect_size': None,
                'is_significant': False,
                'interpretation': 'エラー',
                'improvement_percent': 0.0,
                'current_mean': 0.0,
                'baseline_mean': 0.0,
                'current_sample_size': 0,
                'baseline_sample_size': 0,
                'significance_text': '❌エラー',
                'error': str(e)
            }
    
    def update_google_sheets_new_format(self, stats: Dict) -> bool:
        """
        Google Sheets新列構成での更新
        X列: Current, Y列: Baseline, Z列: p値, AA列: Cohen's d, AB列: 改善率, AC列: 統計的有意性
        """
        try:
            if not stats['success']:
                return False
            
            row_number = stats['current_row']
            
            # X列: Current トラッカーID
            self.sheets_client.update_sheet_values(f'X{row_number}', [[stats['current_tracker']]])
            
            # Y列: Baseline トラッカーID
            self.sheets_client.update_sheet_values(f'Y{row_number}', [[stats['baseline_tracker']]])
            
            # Z列: p値
            self.sheets_client.update_sheet_values(f'Z{row_number}', [[stats['p_value']]])
            
            # AA列: 効果サイズ（Cohen's d）
            self.sheets_client.update_sheet_values(f'AA{row_number}', [[stats['effect_size']]])
            
            # AB列: 改善率（%）
            self.sheets_client.update_sheet_values(f'AB{row_number}', [[stats['improvement_percent']]])
            
            # AC列: 統計的有意性
            self.sheets_client.update_sheet_values(f'AC{row_number}', [[stats['significance_text']]])
            
            return True
            
        except Exception as e:
            print(f"❌ Google Sheets更新エラー ({stats['current_tracker']}): {e}")
            return False
    
    def run_comprehensive_rolling_analysis(self) -> Dict:
        """包括的ローリング統計分析実行"""
        print("🚀 包括的ローリング統計分析開始\\n")
        
        # ステップ1: 完了トラッカー抽出・時系列ソート
        completed_trackers = self.extract_completed_trackers_chronological()
        
        # ステップ2: 隣接ペア生成
        adjacent_pairs = self.generate_adjacent_pairs(completed_trackers)
        
        # ステップ3: 各ペアで統計分析実行
        print(f"\\n📊 ローリング統計分析実行中（{len(adjacent_pairs)}ペア）...\\n")
        
        successful_analyses = 0
        failed_analyses = 0
        sheets_updates = 0
        
        analysis_results = []
        
        for i, (current, baseline) in enumerate(adjacent_pairs, 1):
            print(f"[{i}/{len(adjacent_pairs)}] 分析中: {current['tracker_id']} vs {baseline['tracker_id']}")
            
            # 統計比較実行
            stats = self.run_single_statistical_comparison(current, baseline)
            analysis_results.append(stats)
            
            if stats['success']:
                successful_analyses += 1
                
                # 結果表示
                sig_mark = stats['significance_text']
                print(f"   {sig_mark} p={stats['p_value']:.4f}, Cohen's d={stats['effect_size']:.3f}, 改善率={stats['improvement_percent']:.1f}%")
                print(f"   📊 品質: {stats['baseline_mean']:.3f}→{stats['current_mean']:.3f} (サンプル: {stats['baseline_sample_size']}→{stats['current_sample_size']})")
                
                # Google Sheets更新（新列構成）
                if self.update_google_sheets_new_format(stats):
                    sheets_updates += 1
                    print(f"   📝 Google Sheets更新完了（行{stats['current_row']}: X-AC列）")
                else:
                    print(f"   ⚠️  Google Sheets更新失敗")
                
            else:
                failed_analyses += 1
                print(f"   ❌ 分析失敗: {stats['error']}")
            
            print()
        
        # 結果サマリー
        print(f"🎯 包括的ローリング統計分析結果:")
        print(f"   - 総分析ペア数: {len(adjacent_pairs)}")
        print(f"   - 成功分析: {successful_analyses} ({successful_analyses/len(adjacent_pairs)*100:.1f}%)")
        print(f"   - 失敗分析: {failed_analyses} ({failed_analyses/len(adjacent_pairs)*100:.1f}%)")
        print(f"   - Google Sheets更新: {sheets_updates} ({sheets_updates/len(adjacent_pairs)*100:.1f}%)")
        
        # 統計的有意差の集計
        significant_improvements = [r for r in analysis_results if r['success'] and r['is_significant'] and r['improvement_percent'] > 0]
        significant_degradations = [r for r in analysis_results if r['success'] and r['is_significant'] and r['improvement_percent'] < 0]
        
        print(f"\\n📈 統計的有意差サマリー:")
        print(f"   - 有意改善: {len(significant_improvements)}ペア")
        print(f"   - 有意劣化: {len(significant_degradations)}ペア")
        print(f"   - 有意差なし: {successful_analyses - len(significant_improvements) - len(significant_degradations)}ペア")
        
        if significant_improvements:
            print(f"\\n🟢 有意な改善（上位5件）:")
            sorted_improvements = sorted(significant_improvements, key=lambda x: x['improvement_percent'], reverse=True)
            for result in sorted_improvements[:5]:
                print(f"   - {result['current_tracker']:12}: +{result['improvement_percent']:5.1f}% (p={result['p_value']:.4f}, d={result['effect_size']:.3f})")
        
        if significant_degradations:
            print(f"\\n🔴 有意な劣化（上位5件）:")
            sorted_degradations = sorted(significant_degradations, key=lambda x: x['improvement_percent'])
            for result in sorted_degradations[:5]:
                print(f"   - {result['current_tracker']:12}: {result['improvement_percent']:6.1f}% (p={result['p_value']:.4f}, d={result['effect_size']:.3f})")
        
        return {
            'success': True,
            'total_completed_trackers': len(completed_trackers),
            'analysis_pairs': len(adjacent_pairs),
            'successful_analyses': successful_analyses,
            'failed_analyses': failed_analyses,
            'sheets_updates': sheets_updates,
            'significant_improvements': len(significant_improvements),
            'significant_degradations': len(significant_degradations),
            'analysis_results': analysis_results
        }


def main():
    """メイン実行関数"""
    analyzer = ComprehensiveRollingStatisticalAnalyzer()
    
    try:
        result = analyzer.run_comprehensive_rolling_analysis()
        
        if result['success']:
            print("\\n✅ 包括的ローリング統計分析完全成功")
            print(f"🎯 {result['successful_analyses']}/{result['analysis_pairs']}ペア分析完了")
            print(f"📝 {result['sheets_updates']}/{result['analysis_pairs']}ペアGoogle Sheets更新完了（X-AC列）")
            print(f"📈 統計的有意差: 改善{result['significant_improvements']}件, 劣化{result['significant_degradations']}件")
        else:
            print(f"\\n❌ 包括的ローリング統計分析失敗: {result.get('error', '不明')}")
        
        return result
        
    except Exception as e:
        print(f"❌ 包括的ローリング統計分析システムエラー: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()