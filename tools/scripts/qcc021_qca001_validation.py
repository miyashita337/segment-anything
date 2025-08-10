#!/usr/bin/env python3
"""
QCC-021とQCA-001統合検証スクリプト
QCA-001の実際の抽出結果に対してサンプルサイズ妥当性を検証
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent.parent))

from features.analysis.sample_size_validator import SampleSizeValidator, TestType
# from tools.progress_tracker.cli import update_tracker_status  # 不要なimport削除

class QCA001ValidationIntegrator:
    """QCA-001の実際データでのQCC-021検証"""
    
    def __init__(self):
        self.validator = SampleSizeValidator()
        self.workspace_base = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace")
        
    def analyze_qca001_sample_adequacy(self) -> Dict[str, Any]:
        """
        QCA-001の実際の抽出結果でサンプルサイズ妥当性を検証
        
        Returns:
            検証結果の詳細レポート
        """
        print("🔍 QCA-001のサンプルサイズ妥当性検証開始...")
        
        # QCA-001のワークスペース確認
        qca001_workspace = self.workspace_base / "QCA-001"
        if not qca001_workspace.exists():
            raise FileNotFoundError(f"QCA-001ワークスペースが見つかりません: {qca001_workspace}")
        
        # 抽出画像数を確認
        extraction_dir = qca001_workspace / "extraction"
        if not extraction_dir.exists():
            raise FileNotFoundError(f"抽出ディレクトリが見つかりません: {extraction_dir}")
        
        # 画像ファイル数カウント
        image_files = list(extraction_dir.glob("*.jpg")) + list(extraction_dir.glob("*.png"))
        current_sample_size = len(image_files)
        
        print(f"📊 QCA-001の実際のサンプル数: {current_sample_size}枚")
        
        # QCA-001特化のテストシナリオ
        qca001_scenarios = [
            {
                'name': 'QCA-001作者別品質差検出（小効果）',
                'test_type': TestType.TWO_SAMPLE_T,
                'effect_size': 0.2,  # yado vs kiri の品質差
                'description': 'yado作者とkiri作者の品質差を統計的に検出'
            },
            {
                'name': 'QCA-001パラメータ最適化効果（中効果）',
                'test_type': TestType.PAIRED_T,
                'effect_size': 0.5,  # 最適化前後の改善
                'description': '作者別パラメータ適応による品質改善効果を検出'
            },
            {
                'name': 'QCA-001成功率改善検証（比率）',
                'test_type': TestType.PROPORTION,
                'effect_size': 0.25,  # 25%ポイント改善
                'description': 'A/B評価成功率の統計的改善を検出'
            },
            {
                'name': 'QCA-001品質スコア基準値比較',
                'test_type': TestType.ONE_SAMPLE_T,
                'effect_size': 0.3,  # 基準値からの差
                'description': '品質スコア0.7基準値との統計的差異を検出'
            }
        ]
        
        # サンプルサイズ妥当性検証実行
        validation = self.validator.validate_sample_adequacy(
            current_sample_size=current_sample_size,
            test_scenarios=qca001_scenarios
        )
        
        # 結果詳細分析
        analysis_results = self._analyze_validation_results(validation, current_sample_size)
        
        # QCA-001特化の改善提案生成
        qca001_recommendations = self._generate_qca001_recommendations(validation, current_sample_size)
        
        # 総合レポート作成
        comprehensive_report = {
            "qca001_sample_info": {
                "current_sample_size": current_sample_size,
                "workspace_path": str(qca001_workspace),
                "image_files": [f.name for f in image_files[:10]]  # 最初の10ファイル
            },
            "statistical_validation": {
                "overall_adequacy": validation.overall_adequacy,
                "recommended_n": validation.recommended_n,
                "current_power": validation.current_power,
                "precision_assessment": validation.precision_assessment
            },
            "detailed_requirements": [
                {
                    "scenario": req.test_type.value,
                    "current_n": req.current_n,
                    "required_n": req.required_n,
                    "is_adequate": req.is_adequate,
                    "confidence_width": req.confidence_width,
                    "precision_level": req.precision_level,
                    "effect_size": req.effect_size
                }
                for req in validation.sample_requirements
            ],
            "warnings_and_suggestions": {
                "statistical_warnings": validation.statistical_warnings,
                "improvement_suggestions": validation.improvement_suggestions,
                "qca001_specific_recommendations": qca001_recommendations
            },
            "analysis_results": analysis_results
        }
        
        return comprehensive_report
    
    def _analyze_validation_results(self, validation, current_sample_size: int) -> Dict[str, Any]:
        """検証結果の詳細分析"""
        
        # 統計的パワー分析
        power_analysis = {
            "current_power": validation.current_power,
            "power_interpretation": self._interpret_power(validation.current_power),
            "power_adequacy": "adequate" if validation.current_power >= 0.8 else "inadequate"
        }
        
        # 精度評価分析
        precision_analysis = {
            "precision_level": validation.precision_assessment,
            "confidence_interpretation": self._interpret_precision(validation.precision_assessment),
            "sample_efficiency": current_sample_size / validation.recommended_n if validation.recommended_n > 0 else 1.0
        }
        
        # 効果サイズ別の必要サンプル数
        effect_size_analysis = {}
        for req in validation.sample_requirements:
            effect_name = f"effect_size_{req.effect_size}"
            effect_size_analysis[effect_name] = {
                "required_samples": req.required_n,
                "shortage": max(0, req.required_n - current_sample_size),
                "adequacy_ratio": current_sample_size / req.required_n,
                "test_type": req.test_type.value
            }
        
        return {
            "power_analysis": power_analysis,
            "precision_analysis": precision_analysis,
            "effect_size_analysis": effect_size_analysis,
            "overall_assessment": self._generate_overall_assessment(validation, current_sample_size)
        }
    
    def _interpret_power(self, power: float) -> str:
        """検出力の解釈"""
        if power >= 0.9:
            return "非常に高い検出力（90%以上）"
        elif power >= 0.8:
            return "適切な検出力（80%以上）"
        elif power >= 0.6:
            return "やや低い検出力（60-80%）"
        else:
            return "不十分な検出力（60%未満）"
    
    def _interpret_precision(self, precision_level: str) -> str:
        """精度レベルの解釈"""
        interpretations = {
            "高精度": "信頼区間が狭く、高精度な推定が可能",
            "中精度": "中程度の精度、実用的な推定範囲",
            "低精度": "信頼区間が広く、推定精度が低い"
        }
        return interpretations.get(precision_level, "精度評価不明")
    
    def _generate_overall_assessment(self, validation, current_sample_size: int) -> str:
        """総合評価生成"""
        if validation.overall_adequacy:
            return f"✅ 現在のサンプル数{current_sample_size}は統計的に十分です"
        else:
            shortage = validation.recommended_n - current_sample_size
            return f"❌ 統計的妥当性には{shortage}サンプル追加が必要です（推奨: {validation.recommended_n}サンプル）"
    
    def _generate_qca001_recommendations(self, validation, current_sample_size: int) -> List[str]:
        """QCA-001特化の改善提案"""
        recommendations = []
        
        if not validation.overall_adequacy:
            shortage = validation.recommended_n - current_sample_size
            recommendations.append(
                f"QCA-001の統計的信頼性向上には追加{shortage}サンプル推奨"
            )
            recommendations.append(
                f"他作者（kiri, zundamon）からの画像追加でサンプル数拡張を検討"
            )
        
        if validation.current_power < 0.8:
            recommendations.append(
                "検出力向上のため、効果サイズの大きい作者ペアでの検証を推奨"
            )
        
        if validation.precision_assessment == "低精度":
            recommendations.append(
                "推定精度向上のため、品質スコア分散を考慮したサンプリング戦略を推奨"
            )
        
        # QCA-001固有の提案
        if current_sample_size < 30:
            recommendations.append(
                "中心極限定理の適用には30サンプル以上を強く推奨（現在の機械学習評価における業界標準）"
            )
        
        return recommendations
    
    def save_validation_report(self, report: Dict[str, Any], tracker_id: str = "QCC-021") -> str:
        """検証レポート保存"""
        
        # レポート保存先
        report_dir = self.workspace_base / tracker_id / "quality"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON形式で詳細レポート保存
        json_path = report_dir / "qca001_sample_validation.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # テキスト形式でサマリー保存
        txt_path = report_dir / "qca001_sample_validation_summary.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(self._generate_text_summary(report))
        
        print(f"✅ QCC-021検証レポート保存完了:")
        print(f"   - JSON詳細: {json_path}")
        print(f"   - テキスト要約: {txt_path}")
        
        return str(json_path)
    
    def _generate_text_summary(self, report: Dict[str, Any]) -> str:
        """テキスト要約生成"""
        lines = [
            "=" * 80,
            "QCC-021: QCA-001サンプルサイズ妥当性検証レポート",
            "=" * 80,
            "",
            f"📊 サンプル数: {report['qca001_sample_info']['current_sample_size']}枚",
            f"📈 統計的妥当性: {'✅ 適切' if report['statistical_validation']['overall_adequacy'] else '❌ 不適切'}",
            f"🎯 推奨サンプル数: {report['statistical_validation']['recommended_n']}",
            f"⚡ 現在の検出力: {report['statistical_validation']['current_power']:.3f}",
            f"🔍 精度評価: {report['statistical_validation']['precision_assessment']}",
            "",
            "⚠️ 統計的警告:",
        ]
        
        for warning in report['warnings_and_suggestions']['statistical_warnings']:
            lines.append(f"  - {warning}")
        
        lines.extend([
            "",
            "💡 改善提案:",
        ])
        
        for suggestion in report['warnings_and_suggestions']['improvement_suggestions']:
            lines.append(f"  - {suggestion}")
        
        lines.extend([
            "",
            "🎯 QCA-001特化推奨事項:",
        ])
        
        for rec in report['warnings_and_suggestions']['qca001_specific_recommendations']:
            lines.append(f"  - {rec}")
        
        lines.extend([
            "",
            "📋 詳細要件:",
        ])
        
        for req in report['detailed_requirements']:
            status = "✅" if req['is_adequate'] else "❌"
            lines.append(
                f"  {status} {req['scenario']}: {req['current_n']}/{req['required_n']} "
                f"(精度: {req['precision_level']})"
            )
        
        lines.extend([
            "",
            f"🔬 総合評価: {report['analysis_results']['overall_assessment']}",
            "",
            "=" * 80
        ])
        
        return "\n".join(lines)


def main():
    """メイン実行関数"""
    print("🚀 QCC-021 × QCA-001 統合検証システム開始")
    
    try:
        integrator = QCA001ValidationIntegrator()
        
        # QCA-001のサンプルサイズ妥当性検証
        validation_report = integrator.analyze_qca001_sample_adequacy()
        
        # レポート保存
        report_path = integrator.save_validation_report(validation_report)
        
        # 結果サマリー表示
        print("\n" + "=" * 60)
        print("🎯 QCC-021検証結果サマリー")
        print("=" * 60)
        
        sample_info = validation_report['qca001_sample_info']
        stat_validation = validation_report['statistical_validation']
        
        print(f"📊 QCA-001現在のサンプル数: {sample_info['current_sample_size']}枚")
        print(f"📈 統計的妥当性: {'✅ 適切' if stat_validation['overall_adequacy'] else '❌ 不適切'}")
        print(f"🎯 推奨サンプル数: {stat_validation['recommended_n']}")
        print(f"⚡ 現在の検出力: {stat_validation['current_power']:.3f}")
        print(f"🔍 精度評価: {stat_validation['precision_assessment']}")
        
        warnings = validation_report['warnings_and_suggestions']['statistical_warnings']
        if warnings:
            print(f"\n⚠️ 統計的警告 ({len(warnings)}件):")
            for warning in warnings[:3]:  # 最初の3件
                print(f"  - {warning}")
        
        recommendations = validation_report['warnings_and_suggestions']['qca001_specific_recommendations']
        if recommendations:
            print(f"\n💡 QCA-001特化推奨事項 ({len(recommendations)}件):")
            for rec in recommendations[:3]:  # 最初の3件
                print(f"  - {rec}")
        
        print(f"\n📄 詳細レポート: {report_path}")
        print("✅ QCC-021検証完了")
        
        return 0
        
    except Exception as e:
        print(f"❌ QCC-021検証エラー: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())