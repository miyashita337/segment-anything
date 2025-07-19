#!/usr/bin/env python3
"""
Phase A: GPT-4O推奨ボックス拡張機能のデモ
実際の画像を使用した動作確認デモ
"""

import os
import sys
from pathlib import Path

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

def find_test_image():
    """テスト用画像を探す"""
    test_dirs = [
        "test_small",
        "assets", 
        "examples",
        "test_images"
    ]
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    
    for test_dir in test_dirs:
        test_path = Path(test_dir)
        if test_path.exists():
            for ext in image_extensions:
                images = list(test_path.glob(f'*{ext}'))
                images.extend(test_path.glob(f'*{ext.upper()}'))
                if images:
                    return str(images[0])
    
    return None


def demo_box_expansion():
    """ボックス拡張機能のデモ実行"""
    print("🎪 Phase A: GPT-4O推奨ボックス拡張機能デモ")
    print("=" * 50)
    
    # テスト画像を探す
    test_image = find_test_image()
    
    if not test_image:
        print("⚠️ テスト用画像が見つかりません")
        print("📁 以下のディレクトリに画像を配置してください:")
        print("   - test_small/")
        print("   - assets/")
        print("   - examples/")
        return False
    
    print(f"🖼️ テスト画像: {test_image}")
    
    try:
        from features.extraction.commands.extract_character import extract_character_from_path
        
        # 通常の処理（ボックス拡張なし）
        print("\n📊 通常処理（ボックス拡張なし）")
        print("-" * 30)
        
        result_normal = extract_character_from_path(
            test_image,
            verbose=True,
            use_box_expansion=False
        )
        
        print(f"結果: {'成功' if result_normal['success'] else '失敗'}")
        if result_normal['success']:
            print(f"処理時間: {result_normal['processing_time']:.2f}秒")
        else:
            print(f"エラー: {result_normal.get('error', 'Unknown')}")
        
        # ボックス拡張処理（balanced戦略）
        print("\n🎯 GPT-4O推奨ボックス拡張（balanced戦略）")
        print("-" * 40)
        
        result_expanded = extract_character_from_path(
            test_image,
            verbose=True,
            use_box_expansion=True,
            expansion_strategy='balanced'
        )
        
        print(f"結果: {'成功' if result_expanded['success'] else '失敗'}")
        if result_expanded['success']:
            print(f"処理時間: {result_expanded['processing_time']:.2f}秒")
        else:
            print(f"エラー: {result_expanded.get('error', 'Unknown')}")
        
        # 結果比較
        print("\n📈 結果比較")
        print("-" * 20)
        
        if result_normal['success'] and result_expanded['success']:
            normal_quality = result_normal.get('mask_quality', {})
            expanded_quality = result_expanded.get('mask_quality', {})
            
            print(f"通常処理の品質:")
            print(f"   カバレッジ: {normal_quality.get('coverage_ratio', 0):.3f}")
            print(f"   コンパクト性: {normal_quality.get('compactness', 0):.3f}")
            
            print(f"拡張処理の品質:")
            print(f"   カバレッジ: {expanded_quality.get('coverage_ratio', 0):.3f}")
            print(f"   コンパクト性: {expanded_quality.get('compactness', 0):.3f}")
            
            # 改善度を計算
            coverage_improvement = expanded_quality.get('coverage_ratio', 0) - normal_quality.get('coverage_ratio', 0)
            compactness_improvement = expanded_quality.get('compactness', 0) - normal_quality.get('compactness', 0)
            
            print(f"改善度:")
            print(f"   カバレッジ: {coverage_improvement:+.3f}")
            print(f"   コンパクト性: {compactness_improvement:+.3f}")
            
            if coverage_improvement > 0 or compactness_improvement > 0:
                print("✅ GPT-4O推奨ボックス拡張により品質が向上しました！")
            else:
                print("📊 品質変化は軽微です")
        
        print("\n🎉 デモ完了!")
        print("📝 実際の使用方法:")
        print("   python features/extraction/commands/extract_character.py <image> --use-box-expansion")
        print("   python features/extraction/commands/extract_character.py <image> --use-box-expansion --expansion-strategy aggressive")
        
        return True
        
    except Exception as e:
        print(f"❌ デモエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_command_line():
    """コマンドライン版のデモ"""
    print("\n🖥️ コマンドライン版デモ")
    print("-" * 30)
    
    test_image = find_test_image()
    if not test_image:
        print("⚠️ テスト用画像が見つかりません")
        return
    
    print(f"💡 実行例（ボックス拡張有効）:")
    print(f"   python3 features/extraction/commands/extract_character.py '{test_image}' --use-box-expansion")
    print(f"   python3 features/extraction/commands/extract_character.py '{test_image}' --use-box-expansion --expansion-strategy aggressive")
    
    print(f"\n💡 バッチ処理例:")
    print(f"   python3 features/extraction/commands/extract_character.py test_small/ --batch --use-box-expansion")


def main():
    """メインデモ実行"""
    print("🚀 Phase A: GPT-4O推奨ボックス拡張機能デモ開始")
    
    # デモ実行
    success = demo_box_expansion()
    
    # コマンドライン例の表示
    demo_command_line()
    
    if success:
        print("\n🎊 Phase A実装とデモが完了しました!")
        print("📊 GPT-4O推奨の顔検出ボックス拡張（水平2.5-3倍、垂直4倍）が利用可能です")
    else:
        print("\n⚠️ デモ実行中にエラーが発生しました")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)