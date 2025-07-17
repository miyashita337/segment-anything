#!/usr/bin/env python3
"""
Phase 3: クイックインタラクティブ抽出コマンド
コマンドラインからシードポイントやバウンディングボックスを指定して抽出
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Interactive Character Extraction")
    
    parser.add_argument('input', help='Input image path')
    parser.add_argument('-o', '--output', help='Output path (auto-generated if not specified)')
    
    # インタラクティブ方式の選択
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--points', nargs='+', metavar='x,y[,type]', 
                           help='Seed points (format: x1,y1[,pos] x2,y2[,neg] ...)')
    mode_group.add_argument('--region', metavar='x,y,w,h',
                           help='Bounding box region (format: x,y,width,height)')
    
    parser.add_argument('--verbose', action='store_true', default=True, help='Verbose output')
    
    args = parser.parse_args()
    
    try:
        # モデル初期化
        if args.verbose:
            print("🔄 モデル初期化中...")
        
        from hooks.start import start
        start()
        
        if args.verbose:
            print("✅ モデル初期化完了")
        
        # 処理方式に応じて実行
        if args.points:
            result = extract_with_points(args.input, args.points, args.output, args.verbose)
        elif args.region:
            result = extract_with_region(args.input, args.region, args.output, args.verbose)
        else:
            print("❌ --points または --region を指定してください")
            return 1
        
        # 結果表示
        if result['success']:
            print(f"✅ 抽出成功: {result['output_path']}")
            return 0
        else:
            print(f"❌ 抽出失敗: {result['error']}")
            return 1
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        return 1


def extract_with_points(image_path: str, points_args: list, output_path: str = None, verbose: bool = True) -> dict:
    """シードポイントでの抽出"""
    from utils.interactive_core import quick_extract_with_points
    
    # ポイント引数を解析
    points = []
    for point_arg in points_args:
        parts = point_arg.split(',')
        if len(parts) < 2:
            raise ValueError(f"Invalid point format: {point_arg}")
        
        x, y = int(parts[0]), int(parts[1])
        is_positive = True  # デフォルトは正のポイント
        
        if len(parts) >= 3:
            point_type = parts[2].lower()
            if point_type in ['neg', 'negative', 'n', '0', 'false']:
                is_positive = False
        
        points.append((x, y, is_positive))
        
        if verbose:
            point_type_str = "正" if is_positive else "負"
            print(f"🎯 {point_type_str}のシードポイント: ({x}, {y})")
    
    if verbose:
        print(f"📸 画像: {image_path}")
        print(f"🔧 方式: シードポイント ({len(points)}個)")
    
    return quick_extract_with_points(image_path, points, output_path)


def extract_with_region(image_path: str, region_arg: str, output_path: str = None, verbose: bool = True) -> dict:
    """バウンディングボックスでの抽出"""
    from utils.interactive_core import quick_extract_with_region
    
    # 領域引数を解析
    parts = region_arg.split(',')
    if len(parts) != 4:
        raise ValueError(f"Invalid region format: {region_arg} (expected: x,y,width,height)")
    
    x, y, w, h = map(int, parts)
    region = (x, y, w, h)
    
    if verbose:
        print(f"📸 画像: {image_path}")
        print(f"🔧 方式: バウンディングボックス")
        print(f"📦 領域: x={x}, y={y}, w={w}, h={h}")
    
    return quick_extract_with_region(image_path, region, output_path)


if __name__ == "__main__":
    sys.exit(main())