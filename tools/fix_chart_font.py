#!/usr/bin/env python3
"""
文字化け修正：英語ラベルでレーダーチャート再生成
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# フォント設定（英語のみ）
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

def create_english_radar_chart(output_path: Path):
    """英語ラベルでレーダーチャート作成"""
    
    # サンプルデータ（PH2-002相当）
    metrics_names = [
        'Character Acc',
        'A/B Rate', 
        'FPS',
        'C+ Rate',
        'SCI', 
        'PLA',
        'PLE'
    ]
    
    values = [1.0, 0.0, 0.667, 0.111, 0.4, 0.0, 0.0]  # PH2-002データ
    
    # 角度設定
    angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # 円を閉じる
    values_closed = values + [values[0]]
    
    # レーダーチャート作成
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # データプロット
    ax.plot(angles, values_closed, 'o-', linewidth=2, label='Current', color='#3498db')
    ax.fill(angles, values_closed, alpha=0.25, color='#3498db')
    
    # 軸設定
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_names, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    ax.set_title('Quality Metrics Radar Chart', fontsize=16, pad=20)
    
    # 保存
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"英語レーダーチャート生成完了: {output_path}")

def main():
    output_path = Path("/mnt/c/AItools/lora/train/yado/clipped_boundingbox/workspace/PH2-002/dashboard/radar_chart.png")
    create_english_radar_chart(output_path)

if __name__ == "__main__":
    main()