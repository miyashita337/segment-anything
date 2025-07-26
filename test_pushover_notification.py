#!/usr/bin/env python3
"""
Pushover通知機能テスト
統合品質チェッカーの通知機能のテスト用スクリプト
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# プロジェクトルート追加
sys.path.append(str(Path(__file__).parent))

from tools.unified_quality_checker import UnifiedQualityChecker, UnifiedQualityReport, QualityMetric

def create_test_report():
    """テスト用品質レポート作成"""
    
    # テスト用メトリクス
    evaluation_metrics = [
        QualityMetric(
            name="Largest-Character Accuracy",
            value=0.615,
            threshold=0.8,
            status="failed",
            category="evaluation",
            notes="16/26 成功",
            improvement_suggestions=["YOLO閾値調整", "SAM後処理改良"]
        ),
        QualityMetric(
            name="A/B評価率",
            value=0.0625,
            threshold=0.7,
            status="failed", 
            category="evaluation",
            notes="1/16 A/B評価",
            improvement_suggestions=["品質判定基準見直し"]
        ),
        QualityMetric(
            name="FPS",
            value=2.514,
            threshold=0.2,
            status="passed",
            category="evaluation",
            notes="平均処理時間: 0.40秒"
        )
    ]
    
    mask_metrics = [
        QualityMetric(
            name="平均カバレッジ率",
            value=0.081,
            threshold=0.15,
            status="failed",
            category="mask",
            notes="5枚のサンプル分析"
        )
    ]
    
    objective_metrics = [
        QualityMetric(
            name="PLA (Pixel-Level Accuracy)",
            value=0.839,
            threshold=0.75,
            status="passed",
            category="objective",
            notes="SAMスコア推定"
        )
    ]
    
    # テストレポート作成
    test_report = UnifiedQualityReport(
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        dataset_name="kana08_test",
        total_images=26,
        evaluation_metrics=evaluation_metrics,
        mask_metrics=mask_metrics,
        objective_metrics=objective_metrics,
        overall_score=0.2,  # 20%
        passed_metrics=2,
        total_metrics=5,
        status="FAIL",
        priority_improvements=[
            "YOLO閾値調整",
            "SAM後処理改良", 
            "品質判定基準見直し",
            "検出範囲拡張"
        ],
        technical_recommendations=[
            "アニメキャラクター特化YOLO閾値最適化",
            "SAMマスク後処理パイプライン改良"
        ]
    )
    
    return test_report

def test_notification():
    """通知機能テスト"""
    print("🧪 Pushover通知機能テスト開始")
    
    # テストレポート作成
    test_report = create_test_report()
    
    # 統合品質チェッカー初期化
    checker = UnifiedQualityChecker()
    
    # 通知送信テスト
    print("📱 通知送信テスト実行中...")
    
    try:
        success = checker.send_completion_notification(test_report, "test_results.json")
        
        if success:
            print("✅ 通知送信成功！")
            print("📱 スマートフォンで通知を確認してください")
        else:
            print("⚠️ 通知送信失敗または設定未完了")
            print("💡 config/pushover.json を設定してください")
            
    except Exception as e:
        print(f"❌ 通知エラー: {e}")
        
    print("\n🔍 期待される通知内容:")
    print("=" * 50)
    print("タイトル: 品質チェック完了: kana08_test")
    print()
    print("メッセージ:")
    print("❌ kana08_testデータセット品質チェック完了")
    print()
    print("✅ 成功: 16/26画像 (61.5%)")
    print("📈 総合スコア: 20.0%")
    print("🎯 合格指標: 2/5項目")
    print()
    print("ステータス: FAIL")
    print()
    print("🔧 主要改善提案:")
    print("• YOLO閾値調整")
    print("• SAM後処理改良")
    print("• 品質判定基準見直し")
    print()
    print("⚙️ 技術推奨:")
    print("• アニメキャラクター特化YOLO閾値最適化")
    print("• SAMマスク後処理パイプライン改良")
    print("=" * 50)

if __name__ == "__main__":
    test_notification()