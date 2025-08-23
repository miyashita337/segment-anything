#!/usr/bin/env python3
"""
QCC-021実用版: 機械学習コンテキストでの実用的サンプルサイズ検証
理論値ではなく実用的基準での妥当性評価
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from features.analysis.sample_size_validator import SampleSizeValidator, TestType

class PracticalSampleSizeValidator:
    """機械学習実用版サンプルサイズ検証"""
    
    def __init__(self):
        # 機械学習向けの実用的パラメータ
        self.validator = SampleSizeValidator(
            default_power=0.70,  # 70%検出力（統計学的80%より緩和）
            default_alpha=0.10   # 10%有意水準（統計学的5%より緩和）
        )
    
    def validate_practical_adequacy(self, current_sample_size: int):
        """実用的妥当性検証"""
        print(f"🔍 機械学習実用版サンプルサイズ検証（現在: {current_sample_size}サンプル）")
        
        # 機械学習実用シナリオ
        practical_scenarios = [
            {
                'name': '機械学習品質比較（実用効果）',
                'test_type': TestType.TWO_SAMPLE_T,
                'effect_size': 0.6,  # より大きな効果サイズ
                'description': '実用的な品質差を検出'
            },
            {
                'name': '改善効果検証（中効果）',
                'test_type': TestType.PAIRED_T,
                'effect_size': 0.5,
                'description': 'パラメータ最適化効果検証'
            },
            {
                'name': '成功率改善（実用的）',
                'test_type': TestType.PROPORTION,
                'effect_size': 0.3,  # 30%ポイント改善
                'description': '実用的な成功率向上検証'
            }
        ]
        
        validation = self.validator.validate_sample_adequacy(
            current_sample_size=current_sample_size,
            test_scenarios=practical_scenarios
        )
        
        print(f"\n📊 実用版検証結果:")
        print(f"・統計的妥当性: {'✅ 適切' if validation.overall_adequacy else '⚠️ 要改善'}")
        print(f"・推奨サンプル数: {validation.recommended_n}")
        print(f"・現在の検出力: {validation.current_power:.3f}")
        print(f"・精度評価: {validation.precision_assessment}")
        
        if validation.statistical_warnings:
            print(f"\n⚠️ 実用的警告:")
            for warning in validation.statistical_warnings:
                print(f"  - {warning}")
        
        # 機械学習実用基準での評価
        ml_adequacy = self._evaluate_ml_adequacy(current_sample_size)
        print(f"\n🎯 機械学習実用基準評価:")
        print(f"・最小基準（10サンプル）: {'✅ 達成' if current_sample_size >= 10 else '❌ 不足'}")
        print(f"・推奨基準（30サンプル）: {'✅ 達成' if current_sample_size >= 30 else '⚠️ 不足'}")
        print(f"・理想基準（50サンプル）: {'✅ 達成' if current_sample_size >= 50 else '📈 向上推奨'}")
        
        return validation, ml_adequacy
    
    def _evaluate_ml_adequacy(self, n: int) -> dict:
        """機械学習実用基準での評価"""
        if n >= 50:
            return {
                'level': 'excellent',
                'description': '機械学習プロジェクトとして理想的なサンプルサイズ',
                'confidence': '高信頼性'
            }
        elif n >= 30:
            return {
                'level': 'good', 
                'description': '機械学習プロジェクトとして十分なサンプルサイズ',
                'confidence': '中信頼性'
            }
        elif n >= 20:
            return {
                'level': 'acceptable',
                'description': '機械学習プロジェクトとして最低限のサンプルサイズ',
                'confidence': '実用レベル'
            }
        elif n >= 10:
            return {
                'level': 'minimal',
                'description': 'プロトタイプ・初期検証レベル',
                'confidence': '限定的'
            }
        else:
            return {
                'level': 'insufficient',
                'description': 'サンプル数不足・追加データ収集必須',
                'confidence': '不十分'
            }

def main():
    """実行"""
    validator = PracticalSampleSizeValidator()
    
    print("=" * 60)
    print("🎯 QCC-021実用版: 機械学習コンテキスト妥当性検証")
    print("=" * 60)
    
    # QCA-001の14サンプルで検証
    current_n = 14
    validation, ml_adequacy = validator.validate_practical_adequacy(current_n)
    
    print(f"\n💡 実用的推奨事項:")
    
    if current_n < 30:
        shortage_30 = 30 - current_n
        print(f"  1. 【推奨】{shortage_30}サンプル追加で機械学習実用基準達成（合計30サンプル）")
        print(f"     - kiri作者から{shortage_30//2}サンプル")
        print(f"     - zundamon作者から{shortage_30//2}サンプル")
    
    if current_n < 50:
        shortage_50 = 50 - current_n
        print(f"  2. 【理想】{shortage_50}サンプル追加で高信頼性達成（合計50サンプル）")
        print(f"     - 複数作者・複数データセットからの収集")
    
    print(f"\n📋 現在のレベル: {ml_adequacy['level'].upper()}")
    print(f"📝 評価: {ml_adequacy['description']}")
    print(f"🔒 信頼性: {ml_adequacy['confidence']}")
    
    print(f"\n🎯 結論:")
    if ml_adequacy['level'] in ['good', 'excellent']:
        print("   ✅ 現在のサンプル数で実用的な分析が可能です")
    elif ml_adequacy['level'] == 'acceptable':
        print("   ⚠️ 最低限の分析は可能ですが、追加データ推奨")
    else:
        print("   📈 追加データ収集を強く推奨します")

if __name__ == "__main__":
    main()