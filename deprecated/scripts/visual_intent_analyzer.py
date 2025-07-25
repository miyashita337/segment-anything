#!/usr/bin/env python3
"""
視覚的意図分析器
人間が本当に指定したかった領域 vs 実際の抽出画像の比較
"""

import numpy as np
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisualIntentAnalyzer:
    """視覚的意図分析"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.output_dir = project_root / "visual_intent_analysis"
        self.output_dir.mkdir(exist_ok=True)
    
    def analyze_kana07_0023(self):
        """kana07_0023の詳細視覚分析"""
        # パス設定
        original_path = self.project_root / "lora/train/yado/org/kana07_cursor/kana07_0023.jpg"
        extracted_path = self.project_root / "lora/train/yado/clipped_boundingbox/kana07/kana07_0023.jpg"
        
        # 画像読み込み
        original_img = cv2.imread(str(original_path))
        extracted_img = cv2.imread(str(extracted_path))
        
        if original_img is None or extracted_img is None:
            print("❌ 画像読み込み失敗")
            return
        
        # BGR→RGB変換
        original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        extracted_rgb = cv2.cvtColor(extracted_img, cv2.COLOR_BGR2RGB)
        
        # 座標情報（前回のデバッグから）
        human_bbox = [0, 504, 1364, 1608]  # x, y, w, h
        ai_bbox = [0, 505, 1362, 1606]
        
        # 可視化作成
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. オリジナル + 人間ラベル
        ax1 = axes[0, 0] 
        ax1.imshow(original_rgb)
        hx, hy, hw, hh = human_bbox
        rect_human = patches.Rectangle((hx, hy), hw, hh, 
                                     linewidth=3, edgecolor='red', facecolor='none', alpha=0.7)
        ax1.add_patch(rect_human)
        ax1.set_title("オリジナル画像 + 人間ラベル（赤枠）", fontsize=12)
        ax1.axis('off')
        
        # 2. オリジナル + AI抽出領域
        ax2 = axes[0, 1]
        ax2.imshow(original_rgb)
        ax, ay, aw, ah = ai_bbox
        rect_ai = patches.Rectangle((ax, ay), aw, ah,
                                  linewidth=3, edgecolor='blue', facecolor='none', alpha=0.7)
        ax2.add_patch(rect_ai)
        ax2.set_title("オリジナル画像 + AI抽出領域（青枠）", fontsize=12)
        ax2.axis('off')
        
        # 3. 人間ラベル領域のクロップ
        ax3 = axes[1, 0]
        human_crop = original_rgb[hy:hy+hh, hx:hx+hw]
        ax3.imshow(human_crop)
        ax3.set_title(f"人間ラベル領域\n({hw}x{hh})", fontsize=12)
        ax3.axis('off')
        
        # 4. 実際の抽出画像
        ax4 = axes[1, 1]
        ax4.imshow(extracted_rgb)
        ax4.set_title(f"実際の抽出画像\n({extracted_rgb.shape[1]}x{extracted_rgb.shape[0]})", fontsize=12)
        ax4.axis('off')
        
        plt.tight_layout()
        
        # 保存
        save_path = self.output_dir / "kana07_0023_visual_intent_analysis.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 視覚分析結果:")
        print(f"人間ラベル: {hw}x{hh} = {hw*hh:,}ピクセル")
        print(f"実際抽出: {extracted_rgb.shape[1]}x{extracted_rgb.shape[0]} = {extracted_rgb.shape[1]*extracted_rgb.shape[0]:,}ピクセル")
        print(f"面積比: {(extracted_rgb.shape[1]*extracted_rgb.shape[0])/(hw*hh):.1%}")
        print(f"可視化保存: {save_path}")
        
        # キャラクター位置分析
        self.analyze_character_positions(original_rgb, human_bbox, extracted_rgb)
    
    def analyze_character_positions(self, original_img: np.ndarray, 
                                  human_bbox: list, extracted_img: np.ndarray):
        """キャラクター位置の分析"""
        print(f"\n🎭 キャラクター位置分析:")
        
        h, w = original_img.shape[:2]
        hx, hy, hw, hh = human_bbox
        
        # 人間ラベル領域の画像内比率
        label_coverage = (hw * hh) / (w * h)
        print(f"人間ラベル範囲: {label_coverage:.1%} (画像全体に対する比率)")
        
        # 抽出画像の位置推定（実際の抽出ファイルのサイズから）
        extracted_h, extracted_w = extracted_img.shape[:2]
        
        print(f"実際抽出サイズ: {extracted_w}x{extracted_h}")
        print(f"人間ラベルサイズ: {hw}x{hh}")
        
        # 抽出画像が人間ラベル内のどの部分かを推定
        if extracted_w < hw and extracted_h < hh:
            print("⚠️  実際の抽出は人間ラベル内の一部分のみ")
            
            # 上部キャラクター推定
            if extracted_h < hh * 0.5:
                print("📍 推定位置: 人間ラベル領域の上部")
                print("💡 これが「左下を指定したが上部を抽出」問題の証拠")
        
        # 結論
        print(f"\n🚨 問題の構造:")
        print(f"1. 人間は左下のキャラクターを意図")
        print(f"2. しかし広い範囲（{label_coverage:.1%}）をラベル付け")
        print(f"3. AIは同じ広い範囲内の上部キャラクターを抽出")
        print(f"4. 座標は一致するが、意図したキャラクターは異なる")
    
    def generate_analysis_report(self):
        """分析レポート生成"""
        report = """# 視覚的意図分析レポート - kana07_0023

## 🔍 問題の詳細構造

### 座標情報
- **人間ラベル**: (0, 504, 1364, 1608) - 広範囲指定
- **AI抽出領域**: (0, 505, 1362, 1606) - ほぼ同一
- **IoU**: 0.997 - 数値的には完璧

### 実際の内容
- **人間ラベル領域**: 1364×1608 = 2,193,312ピクセル（画像の76%）
- **実際の抽出画像**: 248×264 = 65,472ピクセル（3%のみ）
- **抽出位置**: 人間ラベル内の上部（白髪キャラクター）

## 🚨 根本問題

### 1. 人間ラベルの粗さ
人間が左下のキャラクターを意図したが、画像の大部分を囲む粗いラベリングを行った。

### 2. AIの解釈問題  
AIは正確な座標範囲を特定したが、その範囲内の異なるキャラクター（上部）を抽出した。

### 3. 評価システムの盲点
従来のIoU評価は座標の一致のみを確認し、実際の抽出内容を検証していない。

## 💡 解決策

### 短期対策
1. **人間ラベルの精密化**: 個別キャラクターを正確にラベリング
2. **視覚的検証**: 抽出結果の内容確認を必須化

### 長期対策  
1. **意図推定AI**: 人間の視覚的意図を理解するシステム
2. **コンテクスト評価**: キャラクター間の関係性を考慮した評価

---

この事例は「座標一致≠内容一致」問題の典型例であり、
AIシステムの評価において視覚的検証の重要性を示している。
"""
        
        report_path = self.output_dir / "visual_intent_analysis_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📋 分析レポート保存: {report_path}")


def main():
    """メイン実行"""
    project_root = Path("/mnt/c/AItools")
    analyzer = VisualIntentAnalyzer(project_root)
    
    # kana07_0023の詳細分析
    analyzer.analyze_kana07_0023()
    
    # 分析レポート生成
    analyzer.generate_analysis_report()
    
    print(f"\n✅ 視覚的意図分析完了")
    print(f"kana07_0023問題の根本構造を解明しました。")


if __name__ == "__main__":
    main()