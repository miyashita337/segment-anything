#!/usr/bin/env python3
"""
P1-010 自動リトライ機能のデモスクリプト
失敗時に最大3回まで自動再実行を行う
"""

import logging
import random
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from features.common.retry_handler import RetryConfig, RetryHandler

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def simulate_unstable_function(success_rate: float = 0.3):
    """不安定な関数をシミュレート（成功率30%）"""
    if random.random() < success_rate:
        return "処理成功！"
    else:
        raise RuntimeError("ランダムエラーが発生しました")


def demo_basic_retry():
    """基本的なリトライ機能のデモ"""
    print("\n=== 基本的なリトライ機能のデモ ===")
    
    # リトライハンドラーの設定
    config = RetryConfig(
        max_retries=3,
        initial_delay=1.0,
        exponential_backoff=True
    )
    handler = RetryHandler(config)
    
    # デコレータを使用
    @handler.retry
    def my_unstable_function():
        return simulate_unstable_function(0.4)  # 成功率40%
    
    # 実行
    try:
        result = my_unstable_function()
        print(f"✅ 結果: {result}")
    except Exception as e:
        print(f"❌ 最終的に失敗: {e}")
    
    # 統計情報の表示
    stats = handler.get_statistics()
    print(f"\n📊 リトライ統計:")
    for key, value in stats['retry_stats'].items():
        print(f"  {key}: {value}")


def demo_image_processing_retry():
    """画像処理用リトライ設定のデモ"""
    print("\n=== 画像処理用リトライ設定のデモ ===")
    
    from features.common.retry_handler import image_retry_handler
    
    @image_retry_handler.retry
    def process_image(image_path: str):
        # 60%の確率で成功
        if random.random() < 0.6:
            return f"画像処理成功: {image_path}"
        else:
            raise ValueError("画像処理エラー")
    
    # テスト実行
    test_images = ["image1.jpg", "image2.jpg", "image3.jpg"]
    
    for img in test_images:
        print(f"\n処理中: {img}")
        try:
            result = process_image(img)
            print(f"✅ {result}")
        except Exception as e:
            print(f"❌ 失敗: {e}")
    
    # 統計情報
    stats = image_retry_handler.get_statistics()
    print(f"\n📊 画像処理リトライ統計:")
    for key, value in stats['retry_stats'].items():
        print(f"  {key}: {value}")


def demo_fallback():
    """フォールバック機能のデモ"""
    print("\n=== フォールバック機能のデモ ===")
    
    handler = RetryHandler(RetryConfig(max_retries=2))
    
    def primary_function():
        """必ず失敗する関数"""
        raise RuntimeError("プライマリ処理が失敗")
    
    def fallback_function():
        """フォールバック関数"""
        return "フォールバック処理で成功"
    
    # フォールバック付きリトライ
    retry_with_fallback = handler.retry_with_fallback(
        primary_function,
        fallback_function
    )
    
    result = retry_with_fallback()
    print(f"✅ 結果: {result}")


def main():
    """メイン実行関数"""
    print("🚀 P1-010 自動リトライ機能デモ")
    
    # 各種デモを実行
    demo_basic_retry()
    demo_image_processing_retry()
    demo_fallback()
    
    print("\n✅ デモ完了！")


if __name__ == "__main__":
    main()