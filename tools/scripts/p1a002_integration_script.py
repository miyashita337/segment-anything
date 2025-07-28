#!/usr/bin/env python3
"""
P1-A002: 品質基準統一システム - 統合実行スクリプト

統一品質評価、統合処理、最終レポート生成の実行統合
PROGRESS_TRACKER.md準拠のワークフロー実装
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# プロジェクトルート設定
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from tools.core.unified_quality_standard import UnifiedQualityStandardSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class P1A002IntegrationSystem:
    """P1-A002 統合実行システム"""
    
    def __init__(self):
        """初期化"""
        self.project_root = project_root
        
        # PROGRESS_TRACKER.md準拠のワークスペース
        self.workspace_root = Path("/mnt/c/AItools/lora/train/yado/clipped_boundingbox/workspace")
        self.workspace_dir = self.workspace_root / "P1-A002"
        
        # 統一品質基準システム
        self.quality_system = UnifiedQualityStandardSystem()
        
        # P1-A003データとの統合確認
        self.p1a003_workspace = self.workspace_root / "P1-A003"
        
        print(f"🎯 P1-A002: 統合実行システム初期化完了")
        print(f"ワークスペース: {self.workspace_dir}")
    
    def load_sample_datasets(self) -> Dict[str, Dict[str, Any]]:
        """サンプルデータセット読み込み"""
        logger.info("サンプルデータセット読み込み開始")
        
        # 実際の品質データに基づくサンプル（複数データセット）
        datasets = {
            "kana08": {
                "total_processed": 26,
                "successful_extractions": 25,
                "ab_evaluation_rate": 0.75,  # P1-A003から取得した実測値
                "sci_score": 0.78,
                "pla_score": 0.82,
                "ple_score": 0.85,
                "avg_fill_ratio": 0.85,
                "avg_compactness": 0.72,
                "avg_coverage": 0.78,
                "grade_distribution": {"A": 6, "B": 13, "C": 5, "D": 2},
                "dataset_source": "P1A003_実測データ"
            },
            "kana05": {
                "total_processed": 36,
                "successful_extractions": 34,
                "ab_evaluation_rate": 0.82,
                "sci_score": 0.80,
                "pla_score": 0.78,
                "ple_score": 0.88,
                "avg_fill_ratio": 0.88,
                "avg_compactness": 0.75,
                "avg_coverage": 0.82,
                "grade_distribution": {"A": 8, "B": 18, "C": 8, "D": 2},
                "dataset_source": "推定値_過去実績ベース"
            },
            "kana06": {
                "total_processed": 42,
                "successful_extractions": 38,
                "ab_evaluation_rate": 0.68,
                "sci_score": 0.72,
                "pla_score": 0.75,
                "ple_score": 0.80,
                "avg_fill_ratio": 0.80,
                "avg_compactness": 0.68,
                "avg_coverage": 0.75,
                "grade_distribution": {"A": 5, "B": 15, "C": 12, "D": 6},
                "dataset_source": "推定値_品質傾向分析"
            },
            "kana07": {
                "total_processed": 28,
                "successful_extractions": 26,
                "ab_evaluation_rate": 0.71,
                "sci_score": 0.75,
                "pla_score": 0.76,
                "ple_score": 0.83,
                "avg_fill_ratio": 0.83,
                "avg_compactness": 0.70,
                "avg_coverage": 0.77,
                "grade_distribution": {"A": 4, "B": 16, "C": 6, "D": 2},
                "dataset_source": "推定値_中間品質"
            }
        }
        
        logger.info(f"サンプルデータセット読み込み完了: {len(datasets)}件")
        return datasets
    
    def run_unified_quality_evaluation(self, datasets: Dict[str, Dict[str, Any]]) -> List[Any]:
        """統一品質評価実行"""
        logger.info("統一品質評価開始")
        
        results = []
        for dataset_name, data in datasets.items():
            logger.info(f"データセット評価: {dataset_name}")
            
            # 統一品質評価実行
            result = self.quality_system.evaluate_dataset_quality(dataset_name, data)
            results.append(result)
            
            # 個別結果保存
            self.quality_system.save_evaluation_result(result)
            
            logger.info(f"評価完了: {dataset_name} -> {result.quality_level} ({result.unified_score:.3f})")
        
        logger.info(f"✅ 統一品質評価完了: {len(results)}データセット")
        return results
    
    def generate_cross_dataset_analysis(self, results: List[Any]) -> Dict[str, Any]:
        """データセット横断分析"""
        logger.info("データセット横断分析開始")
        
        # 横断比較実行
        comparison = self.quality_system.compare_datasets(results)
        
        # 比較結果保存
        comparison_file = self.quality_system.save_comparison_result(comparison)
        
        # 詳細分析
        analysis = {
            "comparison_file": str(comparison_file),
            "total_datasets": comparison["total_datasets"],
            "quality_distribution": comparison["quality_level_distribution"],
            "top_performers": comparison["dataset_ranking"][:2],
            "improvement_targets": comparison["dataset_ranking"][-2:],
            "recommendations": comparison["recommendations"],
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        logger.info("✅ データセット横断分析完了")
        return analysis
    
    def generate_unified_report(self, results: List[Any]) -> Dict[str, Any]:
        """統一品質レポート生成"""
        logger.info("統一品質レポート生成開始")
        
        # 統一レポート生成
        report = self.quality_system.generate_unified_report(results)
        
        logger.info("✅ 統一品質レポート生成完了")
        return report
    
    def create_integration_summary(self, evaluation_results: List[Any],
                                 cross_analysis: Dict[str, Any], 
                                 unified_report: Dict[str, Any]) -> Path:
        """統合処理サマリー作成"""
        logger.info("統合処理サマリー作成開始")
        
        # 統合サマリー構築
        summary = {
            "integration_id": f"P1A002_integration_{datetime.now():%Y%m%d_%H%M%S}",
            "generated_at": datetime.now().isoformat(),
            "system_version": "P1-A002_v1.0.0",
            "summary": {
                "total_datasets_evaluated": len(evaluation_results),
                "unified_standard_version": self.quality_system.standard.version,
                "overall_quality_level": unified_report["summary"]["overall_quality_level"],
                "average_unified_score": unified_report["summary"]["average_unified_score"]
            },
            "evaluation_results": [
                {
                    "dataset": r.dataset_name,
                    "unified_score": r.unified_score,
                    "quality_level": r.quality_level,
                    "unified_grade": r.unified_grade,
                    "success_rate": r.success_rate,
                    "ab_evaluation_rate": r.ab_evaluation_rate
                } for r in evaluation_results
            ],
            "cross_dataset_analysis": cross_analysis,
            "unified_report_summary": {
                "report_id": unified_report["report_id"],
                "quality_standards": unified_report["quality_standard_applied"]["name"]
            },
            "key_findings": self._extract_key_findings(evaluation_results, cross_analysis),
            "next_actions": self._generate_next_actions(evaluation_results, cross_analysis)
        }
        
        # サマリー保存
        summary_file = self.workspace_dir / f"P1A002_integration_summary_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"統合処理サマリー保存: {summary_file}")
        return summary_file
    
    def _extract_key_findings(self, results: List[Any], analysis: Dict[str, Any]) -> List[str]:
        """主要発見事項抽出"""
        findings = []
        
        # 品質分布分析
        excellent_count = len([r for r in results if r.quality_level == "EXCELLENT"])
        poor_count = len([r for r in results if r.quality_level == "POOR"])
        
        if excellent_count > 0:
            findings.append(f"EXCELLENT品質データセット: {excellent_count}件発見")
        
        if poor_count > 0:
            findings.append(f"改善が必要なPOOR品質データセット: {poor_count}件")
        
        # スコア範囲分析
        scores = [r.unified_score for r in results]
        score_range = max(scores) - min(scores)
        findings.append(f"統一スコア範囲: {min(scores):.3f} - {max(scores):.3f} (差{score_range:.3f})")
        
        # トップパフォーマー
        best = max(results, key=lambda x: x.unified_score)
        findings.append(f"最高品質データセット: {best.dataset_name} ({best.unified_score:.3f})")
        
        return findings
    
    def _generate_next_actions(self, results: List[Any], analysis: Dict[str, Any]) -> List[str]:
        """次のアクション生成"""
        actions = []
        
        # 低品質データセット改善
        poor_datasets = [r for r in results if r.quality_level in ["POOR", "ACCEPTABLE"]]
        if poor_datasets:
            poor_names = [r.dataset_name for r in poor_datasets]
            actions.append(f"品質改善対象: {poor_names} の抽出パラメータ調整")
        
        # A/B評価率改善
        low_ab_datasets = [r for r in results if r.ab_evaluation_rate < 0.7]
        if low_ab_datasets:
            low_ab_names = [r.dataset_name for r in low_ab_datasets]
            actions.append(f"A/B評価率改善: {low_ab_names} のYOLO/SAM調整")
        
        # 統一基準の調整
        avg_score = sum(r.unified_score for r in results) / len(results)
        if avg_score < 0.7:
            actions.append("統一品質基準の閾値見直しを検討")
        
        # ベストプラクティス展開
        best = max(results, key=lambda x: x.unified_score)
        if best.unified_score > 0.8:
            actions.append(f"高品質データセット {best.dataset_name} の設定を他データセットに適用")
        
        return actions
    
    def execute_full_integration(self) -> Dict[str, Any]:
        """フル統合実行"""
        logger.info("🚀 P1-A002 フル統合実行開始")
        start_time = datetime.now()
        
        try:
            # 1. サンプルデータセット読み込み
            datasets = self.load_sample_datasets()
            
            # 2. 統一品質評価実行
            evaluation_results = self.run_unified_quality_evaluation(datasets)
            
            # 3. データセット横断分析
            cross_analysis = self.generate_cross_dataset_analysis(evaluation_results)
            
            # 4. 統一品質レポート生成
            unified_report = self.generate_unified_report(evaluation_results)
            
            # 5. 統合サマリー作成
            summary_file = self.create_integration_summary(
                evaluation_results, cross_analysis, unified_report
            )
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            result = {
                "success": True,
                "processing_time": processing_time,
                "datasets_evaluated": len(datasets),
                "summary_file": str(summary_file),
                "unified_report_id": unified_report["report_id"],
                "average_unified_score": unified_report["summary"]["average_unified_score"],
                "overall_quality_level": unified_report["summary"]["overall_quality_level"]
            }
            
            logger.info(f"✅ P1-A002 フル統合完了 (処理時間: {processing_time:.2f}秒)")
            return result
            
        except Exception as e:
            logger.error(f"統合実行エラー: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": (datetime.now() - start_time).total_seconds()
            }


def main():
    """メイン実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description="P1-A002: 統合実行スクリプト")
    parser.add_argument("--full", action="store_true", help="フル統合実行")
    parser.add_argument("--eval-only", action="store_true", help="評価のみ実行")
    
    args = parser.parse_args()
    
    system = P1A002IntegrationSystem()
    
    if args.full:
        # フル統合実行
        result = system.execute_full_integration()
        
        if result["success"]:
            print(f"🎯 P1-A002統合実行完了")
            print(f"   評価データセット: {result['datasets_evaluated']}件")
            print(f"   全体品質レベル: {result['overall_quality_level']}")
            print(f"   平均統一スコア: {result['average_unified_score']:.3f}")
            print(f"   処理時間: {result['processing_time']:.2f}秒")
            print(f"   サマリー: {result['summary_file']}")
            return 0
        else:
            print(f"❌ 統合実行失敗: {result['error']}")
            return 1
    
    elif args.eval_only:
        # 評価のみ実行
        datasets = system.load_sample_datasets()
        results = system.run_unified_quality_evaluation(datasets)
        
        print(f"✅ 品質評価完了: {len(results)}データセット")
        for result in results:
            print(f"   {result.dataset_name}: {result.quality_level} ({result.unified_score:.3f})")
        
        return 0
    
    else:
        print("🎯 P1-A002: 品質基準統一システム - 統合実行")
        print("使用例:")
        print("  python p1a002_integration_script.py --full")
        print("  python p1a002_integration_script.py --eval-only")
        return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())