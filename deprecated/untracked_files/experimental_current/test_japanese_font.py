from features.common.environment_manager import (
    get_path,
    get_test_image_path,
    is_ci_environment,
    setup_test_env,
)

#!/usr/bin/env python3
"""
日本語フォント表示テスト - 修正版
"""

import numpy as np
# フォントキャッシュをクリア
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# matplotlibのフォントキャッシュを再構築
fm.fontManager.__init__()

# 利用可能なフォント確認
print("🔍 利用可能な日本語フォント:")
for font in fm.fontManager.ttflist:
    if 'CJK' in font.name or 'Takao' in font.name or 'IPA' in font.name:
        print(f"  ✅ {font.name} - {font.fname}")

# 日本語フォント設定
plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'Takao Gothic', 'IPAexGothic', 'IPAPGothic']
plt.rcParams['axes.unicode_minus'] = False

print(f"\n🎯 使用フォント: {plt.rcParams['font.family']}")

# テストチャート作成
fig, ax = plt.subplots(figsize=(10, 8))

# 日本語テキスト
categories = ['品質分析', '平均スコア', '中央値', '標準偏差', 'QI-001指標']
values = [1.2, 0.85, 0.92, 0.31, 1.5]

bars = ax.bar(categories, values, color=['#3498db', '#27ae60', '#f39c12', '#e74c3c', '#9b59b6'])

# 日本語ラベル設定
ax.set_title('品質分析ダッシュボード - QUAL-009 品質指標「良いとこ取り」戦略', fontsize=16, pad=20)
ax.set_xlabel('評価項目', fontsize=14)
ax.set_ylabel('品質スコア', fontsize=14)

# グリッド追加
ax.grid(True, alpha=0.3)

# 値をバーの上に表示
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax.annotate(f'{value:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=12)

plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# 保存
output_path = get_path("data", Path(get_path("data", Path("/mnt/c/AItools/segment-anything/japanese_font_test_fixed.png").relative_to("/mnt/c/AItools/"))).relative_to("/mnt/c/AItools/"))
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"✅ テスト画像保存: {output_path}")

# フォント情報表示
print(f"\n📝 実際に使用されたフォント:")
print(f"  Title: {ax.title.get_fontname()}")
print(f"  X-label: {ax.xaxis.label.get_fontname()}")
print(f"  Y-label: {ax.yaxis.label.get_fontname()}")

plt.close()