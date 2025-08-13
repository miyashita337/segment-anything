#!/usr/bin/env python3
"""
統計列統合システム

QCC-021 vs QCC-022の統計分析を実行し、
Google SheetsのX-AB列に統計情報を更新する。
"""

import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).parent.parent.parent))

from tools.progress_tracker.data_generator import ExtractionResultGenerator
from tools.progress_tracker.baseline_detector import BaselineDetector
from tools.progress_tracker.sheets_client import GoogleSheetsClient
from tools.progress_tracker.config import get_default_config
from features.evaluation.statistical_quality_analyzer import StatisticalQualityAnalyzer
from tools.validation.statistical_validator import TTestResult


class StatisticalIntegrationSystem:
    """統計列統合システム"""
    
    def __init__(self, workspace_base: str = "/mnt/c/AItools/lora/train/yado/tracker-workspace"):
        self.workspace_base = Path(workspace_base)
        
        # 各コンポーネント初期化
        self.data_generator = ExtractionResultGenerator(workspace_base)
        self.baseline_detector = BaselineDetector(workspace_base)
        self.statistical_analyzer = StatisticalQualityAnalyzer(workspace_base)
        
        # Google Sheetsクライアント
        self.config = get_default_config()
        self.sheets_client = GoogleSheetsClient(self.config)
    
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
            # 各トラッカーのメトリクス取得
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
            print(f"❌ 改善率計算エラー: {e}")
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
            all_values = self.sheets_client.get_sheet_values('A:A')  # A列のみ取得
            
            for i, row in enumerate(all_values[1:], 2):  # ヘッダースキップ
                if row and len(row) > 0 and row[0] == tracker_id:
                    return i
            
            print(f"❌ {tracker_id}: Google Sheetsに未発見")
            return None
            
        except Exception as e:
            print(f"❌ Google Sheets行検索エラー: {e}")
            return None
    
    def update_initial_tracker(self, tracker_id: str):
        """
        初回トラッカーの統計列更新（ベースライン設定）
        
        Args:
            tracker_id: 初回トラッカーID
        """
        row_num = self.find_tracker_row(tracker_id)
        if row_num is None:
            return
        
        try:
            values = [[
                "N/A",              # X: p値
                "N/A",              # Y: 効果サイズ
                "N/A",              # Z: 改善率
                "ベースライン",         # AA: ベースライン
                "初回トラッカー：統計比較の基準点" # AB: 統計的有意性
            ]]
            
            range_name = f"X{row_num}:AB{row_num}"
            self.sheets_client.update_sheet_values(range_name, values)
            
            print(f"✅ {tracker_id}: 初回トラッカーとして設定完了")
            
        except Exception as e:
            print(f"❌ {tracker_id}: 初回トラッカー設定エラー → {e}")
    
    def log_statistical_error(self, tracker_id: str, error_message: str):
        """
        統計エラーをログ出力
        
        Args:
            tracker_id: トラッカーID
            error_message: エラーメッセージ
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] STATISTICAL_ERROR {tracker_id}: {error_message}"
        
        print(f"🚨 {log_entry}")
        
        # ログファイル出力
        log_file = Path("statistical_integration_errors.log")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + "\n")
    
    def update_statistical_columns_enhanced(self, tracker_id: str) -> bool:
        """
        強化版統計列更新システム
        
        Args:
            tracker_id: 対象トラッカーID
            
        Returns:
            bool: 更新成功の可否
        """
        print(f"📊 {tracker_id}: 統計列更新開始...")
        
        try:
            # 1. ベースライン特定
            baseline = self.baseline_detector.determine_baseline_tracker(tracker_id)
            
            if baseline is None:
                # 初回トラッカー処理
                print(f"🎯 {tracker_id}: 初回トラッカーとして処理")
                self.update_initial_tracker(tracker_id)
                return True
            
            # 2. データ存在確認・生成
            print(f"🔍 データ存在確認・生成中...")
            results = self.data_generator.ensure_extraction_results_exist([baseline, tracker_id])
            
            if len(results) != 2:
                raise ValueError(f"データ生成失敗: {baseline}, {tracker_id}")
            
            # 3. 統計分析実行
            print(f"📈 統計分析実行中: {baseline} vs {tracker_id}")
            result = self.statistical_analyzer.compare_trackers(baseline, tracker_id, 'quality_score')
            improvement = self.calculate_improvement_rate(baseline, tracker_id)
            
            # 4. 詳細判定
            detailed_significance = self.detailed_significance_interpretation(
                result.p_value, result.effect_size
            )
            
            # 5. Google Sheets更新
            row_num = self.find_tracker_row(tracker_id)
            if row_num is None:
                raise ValueError(f"Google Sheetsで{tracker_id}が見つかりません")
            
            values = [[
                round(result.p_value, 6),        # X: p値（6桁精度）
                round(result.effect_size, 3),    # Y: 効果サイズ（3桁精度）
                f"{improvement:.1f}%",           # Z: 改善率
                baseline,                        # AA: ベースライン
                detailed_significance            # AB: 詳細統計判定
            ]]
            
            range_name = f"X{row_num}:AB{row_num}"
            self.sheets_client.update_sheet_values(range_name, values)
            
            print(f"✅ {tracker_id}: 統計列更新完了")
            print(f"📊 統計結果:")
            print(f"   - ベースライン: {baseline}")
            print(f"   - p値: {result.p_value:.6f}")
            print(f"   - 効果サイズ: {result.effect_size:.3f}")
            print(f"   - 改善率: {improvement:.1f}%")
            print(f"   - 判定: {detailed_significance}")
            
            return True
            
        except Exception as e:
            # エラー時の安全処理
            error_msg = str(e)
            self.log_statistical_error(tracker_id, error_msg)
            
            # エラー情報をGoogle Sheetsに記録
            row_num = self.find_tracker_row(tracker_id)
            if row_num:
                try:
                    error_values = [[
                        "ERROR",     # X: p値
                        "ERROR",     # Y: 効果サイズ
                        "ERROR",     # Z: 改善率
                        baseline if 'baseline' in locals() else "不明", # AA: ベースライン
                        f"計算失敗: {error_msg[:50]}..."  # AB: エラー情報
                    ]]
                    
                    range_name = f"X{row_num}:AB{row_num}"
                    self.sheets_client.update_sheet_values(range_name, error_values)
                    
                except Exception as e2:
                    print(f"❌ エラー情報記録失敗: {e2}")
            
            return False
    
    def batch_update_statistical_columns(self, tracker_ids: List[str]) -> Dict[str, bool]:
        """
        複数トラッカーの統計列バッチ更新
        
        Args:
            tracker_ids: トラッカーIDリスト
            
        Returns:
            Dict[str, bool]: トラッカーID -> 更新成功の可否
        """
        results = {}
        
        print(f"🚀 バッチ統計更新開始: {len(tracker_ids)}件")
        
        for i, tracker_id in enumerate(tracker_ids, 1):
            print(f"\n[{i}/{len(tracker_ids)}] 処理中: {tracker_id}")
            
            success = self.update_statistical_columns_enhanced(tracker_id)
            results[tracker_id] = success
            
            if success:
                print(f"✅ [{i}/{len(tracker_ids)}] {tracker_id}: 完了")
            else:
                print(f"❌ [{i}/{len(tracker_ids)}] {tracker_id}: 失敗")
        
        # 結果サマリー
        success_count = sum(results.values())
        print(f"\n📊 バッチ更新結果:")
        print(f"   - 成功: {success_count}/{len(tracker_ids)} ({success_count/len(tracker_ids)*100:.1f}%)")
        print(f"   - 失敗: {len(tracker_ids) - success_count}")
        
        return results


def main():
    """メイン実行関数"""
    integration_system = StatisticalIntegrationSystem()
    
    # QCC-022の統計分析・更新実行
    print("🎯 QCC-022統計分析開始...")
    
    success = integration_system.update_statistical_columns_enhanced('QCC-022')
    
    if success:
        print("\n✅ QCC-022統計分析・Google Sheets更新完了")
    else:
        print("\n❌ QCC-022統計分析失敗")
    
    return success


if __name__ == "__main__":
    main()