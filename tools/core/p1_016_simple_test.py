#!/usr/bin/env python3
"""
P1-016簡易テスト: フィードバックループシステム機能確認

ボトルネック特定・最適化推奨事項・学習機能の検証
"""

import logging
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from features.processing.feedback_loop_system import create_feedback_loop_system

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_p1016_simple_test():
    """P1-016簡易機能テスト"""
    logger.info("🧪 P1-016簡易機能テスト開始")

    # フィードバックループシステム作成
    system = create_feedback_loop_system("P1-016-SIMPLE")
    system.start_feedback_processing()

    try:
        # テスト画像パス
        test_images = [
            "/mnt/c/AItools/lora/train/yado/org/kana05/kana05_0001.jpg",
            "/mnt/c/AItools/lora/train/yado/org/kana05/kana05_0002.jpg",
            "/mnt/c/AItools/lora/train/yado/org/kana05/kana05_0003.jpg",
        ]

        # 各画像でセッションテスト
        for i, image_path in enumerate(test_images, 1):
            if not Path(image_path).exists():
                logger.warning(f"⚠️ テスト画像{i}が存在しません: {image_path}")
                continue

            logger.info(f"📊 テスト{i}: {Path(image_path).name}")

            # セッション作成・分析
            session_id = system.create_processing_session(image_path)

            if session_id:
                # 監視開始
                start_metrics = system.start_processing_monitoring(session_id)

                # 模擬処理時間記録
                system.record_processing_stage(session_id, "yolo_inference", 25.0 + i * 5)
                system.record_processing_stage(session_id, "sam_inference", 300.0 + i * 30)
                system.record_processing_stage(session_id, "postprocessing", 20.0 + i * 3)

                # セッション完了
                system.complete_processing_session(
                    session_id, start_metrics, success=True, actual_quality_score=2.0 + i * 0.3
                )

                logger.info(f"✅ テスト{i}完了")
            else:
                logger.error(f"❌ テスト{i}セッション作成失敗")

        # パフォーマンス分析結果
        logger.info("📈 パフォーマンス分析結果:")
        analysis = system.get_performance_analysis()

        print(f"📊 処理統計:")
        for key, value in analysis["processing_stats"].items():
            print(f"   {key}: {value}")

        print(f"🔍 ボトルネック分析:")
        bottleneck = analysis["bottleneck_analysis"]
        print(f"   主要ボトルネック: {bottleneck.get('primary_bottleneck', 'unknown')}")
        print(f"   ボトルネック時間: {bottleneck.get('bottleneck_time', 0):.1f}秒")
        print(f"   ボトルネック割合: {bottleneck.get('bottleneck_percentage', 0):.1f}%")

        # 最適化推奨事項
        recommendations = system.get_optimization_recommendations()
        print(f"\n💡 最適化推奨事項 ({len(recommendations)}件):")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec['title']} (優先度: {rec['priority']})")
            print(f"      {rec['description']}")
            if rec.get("actions"):
                for action in rec["actions"][:2]:  # 上位2アクション
                    print(f"      - {action}")

        # 学習状況確認
        print(f"\n🧠 学習状況:")
        opt_report = analysis.get("optimization_report", {})
        print(f"   最適化実行回数: {opt_report.get('total_optimizations', 0)}")
        print(f"   学習パターン数: {opt_report.get('learned_patterns', 0)}")
        print(f"   学習パターンキー: {opt_report.get('learned_pattern_keys', [])}")

        logger.info("✅ P1-016簡易機能テスト完了")
        return True

    except Exception as e:
        logger.error(f"❌ テスト実行エラー: {e}")
        return False

    finally:
        system.stop_feedback_processing()


if __name__ == "__main__":
    success = run_p1016_simple_test()
    if not success:
        sys.exit(1)
