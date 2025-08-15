#!/usr/bin/env python3
"""
バッチ統計分析システム

完了済みトラッカーの時系列順統計分析を実行し、
Google SheetsのX-AB列に統計情報を一括更新する。

QCC-023: 効果サイズ計算システム (Cohen's d) の中核実装
"""

import json
import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).parent.parent.parent))

from tools.progress_tracker.sheets_client import GoogleSheetsClient
from tools.progress_tracker.config import get_default_config
from tools.progress_tracker.data_generator import ExtractionResultGenerator
from tools.progress_tracker.baseline_detector import BaselineDetector
from features.evaluation.statistical_quality_analyzer import StatisticalQualityAnalyzer
from tools.validation.statistical_validator import StatisticalValidator, TTestResult


class BatchStatisticalAnalyzer:
    """バッチ統計分析システム"""
    
    def __init__(self, workspace_base: str = "/mnt/c/AItools/lora/train/yado/tracker-workspace"):
        self.workspace_base = Path(workspace_base)
        
        # 各コンポーネント初期化
        self.data_generator = ExtractionResultGenerator(workspace_base)
        self.statistical_analyzer = StatisticalQualityAnalyzer(workspace_base)
        self.statistical_validator = StatisticalValidator(alpha=0.05)
        
        # Google Sheetsクライアント
        self.config = get_default_config()
        self.sheets_client = GoogleSheetsClient(self.config)
        
        # ログファイル設定
        self.log_file = Path("batch_statistical_analysis.log")
        
    def log_message(self, message: str):
        """ログメッセージ出力"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        print(log_entry)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + "\\n")
    
    def get_completed_trackers_sorted(self) -> List[Dict]:
        """
        完了済みトラッカーを更新日付順でソート取得
        
        Returns:
            List[Dict]: ソート済み完了トラッカーリスト
        """
        self.log_message("🔍 完了済みトラッカー取得開始...")
        
        try:
            all_values = self.sheets_client.get_sheet_values('A:AB')
            if not all_values:
                raise ValueError("Google Sheetsからデータ取得失敗")
            
            completed_trackers = []
            
            for i, row in enumerate(all_values[1:], 2):
                if row and len(row) > 2:
                    tracker_id = row[0] if len(row) > 0 else ''
                    status = row[2] if len(row) > 2 else ''
                    update_date = row[4] if len(row) > 4 else ''
                    
                    if '/release' in status:
                        parsed_date = None
                        date_for_sort = '0000-00-00'
                        
                        if update_date and update_date.strip():
                            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y/%m/%d']:
                                try:
                                    parsed_date = datetime.strptime(update_date, fmt)
                                    date_for_sort = parsed_date.strftime('%Y-%m-%d %H:%M:%S')
                                    break
                                except:
                                    continue
                        
                        completed_trackers.append({
                            'tracker_id': tracker_id,
                            'update_date': update_date if update_date else '(未設定)',
                            'date_for_sort': date_for_sort,
                            'row': i
                        })
            
            # ソート処理
            dated_trackers = [t for t in completed_trackers if t['date_for_sort'] != '0000-00-00']
            undated_trackers = [t for t in completed_trackers if t['date_for_sort'] == '0000-00-00']
            
            dated_trackers.sort(key=lambda x: x['date_for_sort'], reverse=True)
            
            # 結合（日付あり→日付なし）
            all_sorted = dated_trackers + undated_trackers
            
            self.log_message(f"✅ 完了済みトラッカー: {len(all_sorted)}件（日付あり: {len(dated_trackers)}、日付なし: {len(undated_trackers)}）")
            
            return all_sorted
            
        except Exception as e:
            self.log_message(f"❌ 完了済みトラッカー取得エラー: {e}")
            raise
    
    def generate_analysis_pairs(self, trackers: List[Dict]) -> List[Tuple[str, str]]:
        """
        統計分析対象ペア生成（時系列順隣接比較）
        
        Args:
            trackers: ソート済みトラッカーリスト
            
        Returns:
            List[Tuple[str, str]]: (現在トラッカー, ベースライントラッカー) のペアリスト
        """
        pairs = []
        
        for i in range(len(trackers) - 1):
            current = trackers[i]['tracker_id']
            baseline = trackers[i + 1]['tracker_id']
            pairs.append((current, baseline))
        
        self.log_message(f"📊 統計分析対象ペア: {len(pairs)}ペア生成")
        
        return pairs
    
    def detailed_significance_interpretation(self, p_value: float, effect_size: float) -> str:
        """
        詳細な統計的有意性判定
        
        Args:
            p_value: p値
            effect_size: 効果サイズ（Cohen's d）
            
        Returns:
            str: 詳細判定結果
        """
        # p値による基本判定
        if p_value < 0.001:
            significance_level = "極めて強い有意差（p<0.001）"
            confidence = "改善効果が非常に確実"
        elif p_value < 0.01:
            significance_level = "強い有意差（p<0.01）"
            confidence = "改善効果が確実"
        elif p_value < 0.05:
            significance_level = "有意差あり（p<0.05）"
            confidence = "改善効果が統計的に証明"
        elif p_value < 0.10:
            significance_level = "境界的有意性（p<0.10）"
            confidence = "改善の兆候あり、さらなる検証推奨"
        else:
            significance_level = f"有意差なし（p={p_value:.3f}）"
            confidence = "統計的改善効果は検出されず"
        
        # 効果サイズによる実用性評価
        abs_effect = abs(effect_size)
        if abs_effect >= 0.8:
            effect_desc = "大きい効果"
            practical = "実用的に意味のある改善"
        elif abs_effect >= 0.5:
            effect_desc = "中程度の効果"
            practical = "中程度の実用的改善"
        elif abs_effect >= 0.2:
            effect_desc = "小さい効果"
            practical = "限定的な改善効果"
        else:
            effect_desc = "効果なし"
            practical = "実用的効果は期待できない"
        
        return f"{significance_level}：{confidence}、{effect_desc}（d={effect_size:.3f}）、{practical}"
    
    def calculate_improvement_rate(self, baseline_tracker: str, improved_tracker: str) -> float:
        """
        改善率計算
        
        Args:
            baseline_tracker: ベースライントラッカー
            improved_tracker: 改善後トラッカー
            
        Returns:
            float: 改善率（％）
        """
        try:
            baseline_metrics = self.statistical_analyzer.load_extraction_results(baseline_tracker)
            improved_metrics = self.statistical_analyzer.load_extraction_results(improved_tracker)
            
            if baseline_metrics.mean_quality_score == 0:
                return 0.0
            
            improvement = (
                (improved_metrics.mean_quality_score - baseline_metrics.mean_quality_score) 
                / baseline_metrics.mean_quality_score * 100
            )
            
            return improvement
            
        except Exception as e:
            self.log_message(f"❌ {baseline_tracker} vs {improved_tracker}: 改善率計算エラー → {e}")
            return 0.0
    
    def find_tracker_row(self, tracker_id: str) -> Optional[int]:
        """
        Google Sheetsでトラッカーの行番号を特定
        
        Args:
            tracker_id: トラッカーID
            
        Returns:
            Optional[int]: 行番号（見つからない場合はNone）
        """
        try:
            all_values = self.sheets_client.get_sheet_values('A:A')
            
            for i, row in enumerate(all_values[1:], 2):
                if row and len(row) > 0 and row[0] == tracker_id:
                    return i
            
            return None
            
        except Exception as e:
            self.log_message(f"❌ {tracker_id}: Google Sheets行検索エラー → {e}")
            return None
    
    def analyze_single_pair(self, current_tracker: str, baseline_tracker: str) -> Dict:
        """
        単一ペアの統計分析実行
        
        Args:
            current_tracker: 現在のトラッカー
            baseline_tracker: ベースライントラッカー
            
        Returns:
            Dict: 分析結果
        """
        result = {
            'success': False,
            'current_tracker': current_tracker,
            'baseline_tracker': baseline_tracker,
            'error': None
        }
        
        try:
            self.log_message(f"📊 分析開始: {current_tracker} vs {baseline_tracker}")
            
            # 1. データ存在確認・生成
            results = self.data_generator.ensure_extraction_results_exist([baseline_tracker, current_tracker])
            
            if len(results) != 2:
                raise ValueError(f"データ生成失敗: {baseline_tracker}, {current_tracker}")
            
            # 2. 統計分析実行
            t_test_result = self.statistical_analyzer.compare_trackers(
                baseline_tracker, current_tracker, 'quality_score'
            )
            
            # 3. 改善率計算
            improvement = self.calculate_improvement_rate(baseline_tracker, current_tracker)
            
            # 4. 詳細判定
            detailed_significance = self.detailed_significance_interpretation(
                t_test_result.p_value, t_test_result.effect_size
            )
            
            result.update({
                'success': True,
                'p_value': t_test_result.p_value,
                'effect_size': t_test_result.effect_size,
                'improvement_rate': improvement,
                'detailed_significance': detailed_significance,
                't_test_result': t_test_result
            })
            
            self.log_message(f"✅ {current_tracker} vs {baseline_tracker}: 分析完了（p={t_test_result.p_value:.6f}, d={t_test_result.effect_size:.3f}, 改善={improvement:.1f}%）")
            
        except Exception as e:
            error_msg = str(e)
            result['error'] = error_msg
            self.log_message(f"❌ {current_tracker} vs {baseline_tracker}: 分析失敗 → {error_msg}")
        
        return result
    
    def update_google_sheets_row(self, tracker_id: str, analysis_result: Dict) -> bool:
        """
        Google SheetsのX-AB列を更新
        
        Args:
            tracker_id: トラッカーID
            analysis_result: 分析結果
            
        Returns:
            bool: 更新成功の可否
        """
        row_num = self.find_tracker_row(tracker_id)
        if row_num is None:
            self.log_message(f"❌ {tracker_id}: Google Sheets行未発見")
            return False
        
        try:
            if analysis_result['success']:
                # 正常な統計結果
                values = [[
                    round(analysis_result['p_value'], 6),                # X: p値
                    round(analysis_result['effect_size'], 3),           # Y: 効果サイズ
                    f"{analysis_result['improvement_rate']:.1f}%",      # Z: 改善率
                    analysis_result['baseline_tracker'],                # AA: ベースライン
                    analysis_result['detailed_significance']            # AB: 詳細統計判定
                ]]
            else:
                # エラー時の安全処理
                error_msg = analysis_result['error']
                values = [[
                    "ERROR",                                            # X: p値
                    "ERROR",                                            # Y: 効果サイズ
                    "ERROR",                                            # Z: 改善率
                    analysis_result['baseline_tracker'],               # AA: ベースライン
                    f"計算失敗: {error_msg[:50]}..."                     # AB: エラー情報
                ]]
            
            range_name = f"X{row_num}:AB{row_num}"
            self.sheets_client.update_sheet_values(range_name, values)
            
            self.log_message(f"✅ {tracker_id}: Google Sheets更新完了（行{row_num}）")
            return True
            
        except Exception as e:
            self.log_message(f"❌ {tracker_id}: Google Sheets更新失敗 → {e}")
            return False
    
    def run_batch_analysis(self, pairs: List[Tuple[str, str]], max_pairs: Optional[int] = None) -> Dict:
        """
        バッチ統計分析実行
        
        Args:
            pairs: 分析対象ペアリスト
            max_pairs: 最大分析ペア数（テスト用、Noneで全て）
            
        Returns:
            Dict: バッチ分析結果
        """
        if max_pairs:
            pairs = pairs[:max_pairs]
        
        self.log_message(f"🚀 バッチ統計分析開始: {len(pairs)}ペア")
        
        results = {
            'total_pairs': len(pairs),
            'successful_analyses': 0,
            'successful_updates': 0,
            'failed_analyses': 0,
            'failed_updates': 0,
            'pair_results': []
        }
        
        start_time = time.time()
        
        for i, (current, baseline) in enumerate(pairs, 1):
            self.log_message(f"\\n[{i}/{len(pairs)}] 処理中: {current} vs {baseline}")
            
            # 統計分析実行
            analysis_result = self.analyze_single_pair(current, baseline)
            results['pair_results'].append(analysis_result)
            
            if analysis_result['success']:
                results['successful_analyses'] += 1
                
                # Google Sheets更新
                if self.update_google_sheets_row(current, analysis_result):
                    results['successful_updates'] += 1
                else:
                    results['failed_updates'] += 1
            else:
                results['failed_analyses'] += 1
                
                # エラー時もGoogle Sheetsに記録
                if self.update_google_sheets_row(current, analysis_result):
                    results['successful_updates'] += 1
                else:
                    results['failed_updates'] += 1
            
            # 進捗表示
            self.log_message(f"[{i}/{len(pairs)}] 完了")
            
            # 少し休憩（Google Sheets API制限対策）
            time.sleep(0.1)
        
        elapsed_time = time.time() - start_time
        
        # 結果サマリー
        self.log_message(f"\\n📊 バッチ分析結果:")
        self.log_message(f"   - 総ペア数: {results['total_pairs']}")
        self.log_message(f"   - 分析成功: {results['successful_analyses']}")
        self.log_message(f"   - 分析失敗: {results['failed_analyses']}")
        self.log_message(f"   - 更新成功: {results['successful_updates']}")
        self.log_message(f"   - 更新失敗: {results['failed_updates']}")
        self.log_message(f"   - 実行時間: {elapsed_time:.1f}秒")
        self.log_message(f"   - 成功率: {results['successful_analyses']/results['total_pairs']*100:.1f}%")
        
        return results


def main():
    """メイン実行関数"""
    analyzer = BatchStatisticalAnalyzer()
    
    try:
        # 1. 完了済みトラッカー取得
        completed_trackers = analyzer.get_completed_trackers_sorted()
        
        # 2. 分析ペア生成
        pairs = analyzer.generate_analysis_pairs(completed_trackers)
        
        # 3. バッチ分析実行（テスト用に最初の5ペアのみ）
        analyzer.log_message("\\n🧪 テスト実行: 最初の5ペアで分析開始...")
        results = analyzer.run_batch_analysis(pairs, max_pairs=5)
        
        if results['successful_analyses'] > 0:
            analyzer.log_message("\\n✅ QCC-023バッチ統計分析テスト完了")
            analyzer.log_message("\\n🎯 全59ペア分析を実行するには、max_pairs=Noneで実行してください")
        else:
            analyzer.log_message("\\n❌ QCC-023バッチ統計分析テスト失敗")
        
        return results
        
    except Exception as e:
        analyzer.log_message(f"❌ バッチ分析システムエラー: {e}")
        raise


if __name__ == "__main__":
    main()