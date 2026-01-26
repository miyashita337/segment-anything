#!/usr/bin/env python3
"""
日本語フォント問題修正スクリプト
matplotlibで日本語が正しく表示されるようフォント設定を修正
"""

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

import os
from pathlib import Path


def check_available_fonts():
    """利用可能なフォント一覧を確認"""
    print("🔍 利用可能なフォント確認中...")

    # システムフォント一覧取得
    font_list = [f.name for f in fm.fontManager.ttflist]

    # 日本語対応可能なフォント候補
    japanese_font_candidates = [
        "DejaVu Sans",
        "Hiragino Sans",
        "Yu Gothic",
        "Meiryo",
        "MS Gothic",
        "Takao Gothic",
        "IPAexGothic",
        "IPAPGothic",
        "VL PGothic",
        "Noto Sans CJK JP",
        "Liberation Sans",
    ]

    print("\n📋 日本語対応フォント候補の確認:")
    available_fonts = []
    for font in japanese_font_candidates:
        if font in font_list:
            print(f"  ✅ {font} - 利用可能")
            available_fonts.append(font)
        else:
            print(f"  ❌ {font} - 利用不可")

    if available_fonts:
        print(f"\n🎯 推奨フォント: {available_fonts[0]}")
        return available_fonts[0]
    else:
        print("\n⚠️ 日本語対応フォントが見つかりません")
        return None


def test_japanese_display(font_name=None):
    """日本語表示テスト"""
    print(f"\n🧪 日本語表示テスト (フォント: {font_name})")

    if font_name:
        plt.rcParams["font.family"] = [font_name]

    # テスト用チャート作成
    fig, ax = plt.subplots(figsize=(8, 6))

    # 日本語テキスト
    test_texts = ["品質分析ダッシュボード", "平均スコア", "中央値", "標準偏差", "QI-001 品質指標"]

    y_positions = range(len(test_texts))
    ax.barh(y_positions, [1, 2, 3, 4, 5])
    ax.set_yticks(y_positions)
    ax.set_yticklabels(test_texts)
    ax.set_xlabel("値")
    ax.set_title("日本語フォントテスト - QI-001")

    # テスト画像保存
    test_path = Path("japanese_font_test.png")
    plt.savefig(test_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✅ テスト画像保存: {test_path}")
    return test_path


def install_font_packages():
    """日本語フォントパッケージのインストール指示"""
    print("\n📦 日本語フォントインストール手順:")
    print("Ubuntu/Debian系:")
    print("  sudo apt update")
    print("  sudo apt install fonts-noto-cjk fonts-liberation")
    print("  sudo apt install fonts-takao-gothic fonts-ipafont-gothic")

    print("\nWSL2の場合:")
    print("  上記コマンド実行後、matplotlibのフォントキャッシュをクリア:")
    print("  rm -rf ~/.cache/matplotlib")


def fix_quality_dashboard_fonts():
    """品質ダッシュボード生成スクリプトのフォント設定修正"""

    # 最適なフォントを検出
    best_font = check_available_fonts()

    if best_font:
        print(f"\n🔧 {best_font} を使用してフォント設定を修正します")

        # フォント設定を適用
        plt.rcParams["font.family"] = [best_font, "DejaVu Sans", "Liberation Sans"]
        plt.rcParams["font.sans-serif"] = [best_font, "DejaVu Sans", "Liberation Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        # テスト実行
        test_path = test_japanese_display(best_font)

        print(f"\n✅ フォント設定完了")
        print(f"📊 {test_path} でフォント表示を確認してください")

        return True
    else:
        install_font_packages()
        return False


def regenerate_dashboard():
    """ダッシュボードの再生成"""
    print("\n🔄 品質ダッシュボードを再生成中...")

    try:
        # QI-001の統計ファイルを探す
        workspace_base = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace")
        qi001_stats = workspace_base / "QI-001" / "quality_statistics.jsonl"

        if qi001_stats.exists():
            # ダッシュボード生成スクリプト実行
            import subprocess

            cmd = [
                "python3",
                "/mnt/c/AItools/segment-anything/tools/quality_dashboard_generator.py",
                "--stats-file",
                str(qi001_stats),
                "--output-dir",
                str(workspace_base / "QI-001" / "dashboard"),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ ダッシュボード再生成完了")
                print("🌐 https://100.123.241.106/tracker/QI-001 で確認してください")
            else:
                print(f"❌ ダッシュボード生成エラー: {result.stderr}")
        else:
            print(f"⚠️ 統計ファイルが見つかりません: {qi001_stats}")

    except Exception as e:
        print(f"❌ エラー: {e}")


def main():
    """メイン処理"""
    print("🎯 日本語フォント問題修正スクリプト")
    print("=" * 50)

    # フォント設定修正
    if fix_quality_dashboard_fonts():
        # ダッシュボード再生成
        regenerate_dashboard()
    else:
        print("\n⚠️ 日本語フォントをインストール後、再度実行してください")


if __name__ == "__main__":
    main()
