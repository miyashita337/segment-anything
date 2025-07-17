#!/usr/bin/env python3
"""
kaname07データセット専用バッチ処理スクリプト
v0.1.0適応学習システムによる高精度抽出
"""

import sys
import os
sys.path.append('.')

from utils.notification import send_batch_notification

def main():
    # モデル初期化
    print("🔄 モデル初期化中...")
    from hooks.start import start
    start()
    print("✅ モデル初期化完了")
    
    # バッチ処理実行
    print("🚀 kaname07バッチ処理開始...")
    from commands.extract_character import batch_extract_characters
    
    input_dir = "/mnt/c/AItools/lora/train/yadokugaeru/org/kaname07"
    output_dir = "/mnt/c/AItools/lora/train/yadokugaeru/clipped_boundingbox/kaname07"
    
    # v0.1.0最適化設定（適応学習システム有効）
    extract_args = {
        'enhance_contrast': False,
        'filter_text': True,
        'save_mask': False,
        'save_transparent': False,
        'min_yolo_score': 0.1,  # 適応学習で自動調整される
        'verbose': False,
        'adaptive_learning': True,  # 適応学習システム有効
        'high_quality': True,       # 高品質SAM処理
        'manga_mode': True,         # 漫画モード有効
        'effect_removal': True,     # エフェクト除去有効
        'difficult_pose': True,     # 困難姿勢対応
        'multi_character_criteria': 'size_priority'  # 適応学習推奨手法
    }
    
    result = batch_extract_characters(input_dir, output_dir, **extract_args)
    
    print(f"\n📊 kaname07最終結果:")
    print(f"   成功: {result['successful']}/{result['total']} ({result['success_rate']:.1f}%)")
    print(f"   失敗: {result['failed']}")
    print(f"   処理時間: {result['total_time']:.2f}秒")
    
    # 期待値との比較
    expected_success_rate = 100  # v0.1.0実績
    if result['success_rate'] >= expected_success_rate * 0.8:
        print(f"✅ 期待値達成: {result['success_rate']:.1f}% >= {expected_success_rate * 0.8}%")
    else:
        print(f"⚠️ 期待値未達: {result['success_rate']:.1f}% < {expected_success_rate * 0.8}%")
    
    # 適応学習統計表示
    if result.get('adaptive_learning_stats'):
        stats = result['adaptive_learning_stats']
        print(f"\n🧠 適応学習統計:")
        print(f"   推奨手法: {stats.get('most_recommended_method', 'N/A')}")
        print(f"   平均予測品質: {stats.get('avg_predicted_quality', 'N/A'):.3f}")
        print(f"   平均実際品質: {stats.get('avg_actual_quality', 'N/A'):.3f}")
        print(f"   予測精度: ±{stats.get('prediction_accuracy', 'N/A'):.3f}")
    
    # Pushover通知送信
    print("\n📱 通知送信中...")
    notification_sent = send_batch_notification(
        successful=result['successful'],
        total=result['total'],
        failed=result['failed'],
        total_time=result['total_time']
    )
    
    if notification_sent:
        print("✅ Pushover通知送信完了")
    else:
        print("⚠️ Pushover通知送信失敗またはスキップ")
    
    # 結果判定
    if result['success_rate'] >= 80:
        print(f"\n🎉 kaname07バッチ処理成功!")
        print(f"   v0.1.0適応学習システムによる高精度抽出完了")
        sys.exit(0)
    else:
        print(f"\n🚨 kaname07バッチ処理で品質問題発生")
        print(f"   成功率{result['success_rate']:.1f}%は期待値を下回る")
        sys.exit(1)

if __name__ == "__main__":
    main()