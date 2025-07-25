#!/usr/bin/env python3
"""
視覚的検証システム
人間ラベルとAI抽出結果を視覚的に比較し、真の成功率を算出
"""

import numpy as np
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """検証結果"""
    image_id: str
    human_bbox: Tuple[int, int, int, int]
    ai_bbox: Tuple[int, int, int, int]
    reported_iou: float
    visual_match: bool  # 視覚的に正しいキャラクターを抽出しているか
    actual_character_extracted: str  # 実際に抽出されたキャラクターの説明
    expected_character: str  # 期待されたキャラクターの説明
    issue_type: Optional[str]  # 問題の種類


class VisualVerificationSystem:
    """視覚的検証システム"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_file = project_root / "lora/train/yado/integrated_benchmark/integrated_improvement_results_20250724_030756.json"
        self.labels_file = project_root / "segment-anything/extracted_labels.json"
        self.output_dir = project_root / "visual_verification_results"
        self.output_dir.mkdir(exist_ok=True)
        
        # データ読み込み
        self.load_data()
        
    def load_data(self):
        """データ読み込み"""
        # AI抽出結果
        with open(self.results_file, 'r', encoding='utf-8') as f:
            self.ai_results = json.load(f)
            
        # 人間ラベル
        with open(self.labels_file, 'r', encoding='utf-8') as f:
            labels_data = json.load(f)
            
        self.human_labels = {}
        for item in labels_data:
            if item.get('red_boxes'):
                image_id = item['filename'].rsplit('.', 1)[0]
                self.human_labels[image_id] = item
                
    def visualize_comparison(self, image_id: str, save: bool = True) -> Optional[VerificationResult]:
        """単一画像の視覚的比較"""
        # AI結果検索
        ai_result = None
        for result in self.ai_results:
            if result['image_id'] == image_id:
                ai_result = result
                break
                
        if not ai_result:
            logger.warning(f"AI結果が見つかりません: {image_id}")
            return None
            
        # 画像パス取得
        image_path = Path(ai_result['image_path'])
        if not image_path.exists():
            logger.warning(f"画像が見つかりません: {image_path}")
            return None
            
        # 抽出結果画像パス
        extracted_paths = [
            self.project_root / f"lora/train/yado/clipped_boundingbox/kana07/{image_id}.jpg",
            self.project_root / f"lora/train/yado/clipped_boundingbox/kana05/{image_id}.jpg",
            self.project_root / f"lora/train/yado/clipped_boundingbox/kana08/{image_id}.jpg"
        ]
        
        extracted_path = None
        for path in extracted_paths:
            if path.exists():
                extracted_path = path
                break
                
        # 画像読み込み
        original_img = cv2.imread(str(image_path))
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        extracted_img = None
        if extracted_path:
            extracted_img = cv2.imread(str(extracted_path))
            extracted_img = cv2.cvtColor(extracted_img, cv2.COLOR_BGR2RGB)
            
        # 可視化
        fig, axes = plt.subplots(1, 3 if extracted_img is not None else 2, figsize=(18, 6))
        
        # 1. 人間ラベル表示
        ax1 = axes[0]
        ax1.imshow(original_img)
        
        human_bbox = ai_result['human_bbox']
        rect_human = patches.Rectangle((human_bbox[0], human_bbox[1]), 
                                     human_bbox[2], human_bbox[3],
                                     linewidth=3, edgecolor='red', facecolor='none')
        ax1.add_patch(rect_human)
        ax1.set_title(f"人間ラベル（赤枠）\n座標: {human_bbox}", fontsize=12)
        ax1.axis('off')
        
        # 2. AI抽出結果表示
        ax2 = axes[1]
        ax2.imshow(original_img)
        
        if ai_result['final_bbox']:
            ai_bbox = ai_result['final_bbox']
            rect_ai = patches.Rectangle((ai_bbox[0], ai_bbox[1]), 
                                      ai_bbox[2], ai_bbox[3],
                                      linewidth=3, edgecolor='blue', facecolor='none')
            ax2.add_patch(rect_ai)
            ax2.set_title(f"AI抽出結果（青枠）\n座標: {ai_bbox}\nIoU: {ai_result['iou_score']:.3f}", fontsize=12)
        else:
            ax2.set_title("AI抽出失敗", fontsize=12)
        ax2.axis('off')
        
        # 3. 実際の抽出画像
        if extracted_img is not None:
            ax3 = axes[2]
            ax3.imshow(extracted_img)
            ax3.set_title("実際の抽出画像", fontsize=12)
            ax3.axis('off')
            
        plt.suptitle(f"{image_id} - 視覚的検証", fontsize=16)
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / f"{image_id}_verification.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"検証画像保存: {save_path}")
            
        plt.close()
        
        # 視覚的一致の判定（ここでは手動確認が必要）
        return VerificationResult(
            image_id=image_id,
            human_bbox=tuple(human_bbox),
            ai_bbox=tuple(ai_result['final_bbox']) if ai_result['final_bbox'] else (0, 0, 0, 0),
            reported_iou=ai_result['iou_score'],
            visual_match=None,  # 手動確認が必要
            actual_character_extracted="要手動確認",
            expected_character="要手動確認",
            issue_type=None
        )
        
    def verify_high_iou_cases(self, threshold: float = 0.9):
        """高IoUケースの検証"""
        high_iou_cases = []
        
        for result in self.ai_results:
            if result['iou_score'] >= threshold and result['extraction_success']:
                high_iou_cases.append(result['image_id'])
                
        logger.info(f"高IoU（>={threshold}）ケース数: {len(high_iou_cases)}")
        
        # 上位10件を詳細検証
        for i, image_id in enumerate(high_iou_cases[:10]):
            logger.info(f"検証中 [{i+1}/10]: {image_id}")
            self.visualize_comparison(image_id)
            
        return high_iou_cases
        
    def analyze_coordinate_system(self):
        """座標系の分析"""
        logger.info("座標系分析開始")
        
        # kana07_0023の詳細分析
        target_id = "kana07_0023"
        
        for result in self.ai_results:
            if result['image_id'] == target_id:
                logger.info(f"\n{target_id}の座標分析:")
                logger.info(f"人間ラベル: {result['human_bbox']}")
                logger.info(f"AI抽出: {result['final_bbox']}")
                logger.info(f"報告IoU: {result['iou_score']}")
                
                # 画像サイズ確認
                image_path = Path(result['image_path'])
                if image_path.exists():
                    img = cv2.imread(str(image_path))
                    h, w = img.shape[:2]
                    logger.info(f"画像サイズ: {w}x{h}")
                    
                    # 座標の妥当性確認
                    hx, hy, hw, hh = result['human_bbox']
                    logger.info(f"人間ラベル領域: 左上({hx},{hy}) 右下({hx+hw},{hy+hh})")
                    
                    if result['final_bbox']:
                        ax, ay, aw, ah = result['final_bbox']
                        logger.info(f"AI抽出領域: 左上({ax},{ay}) 右下({ax+aw},{ay+ah})")
                        
                break
                
    def generate_verification_report(self):
        """検証レポート生成"""
        # 高IoUケースの検証
        high_iou_cases = self.verify_high_iou_cases(0.9)
        
        # 座標系分析
        self.analyze_coordinate_system()
        
        # レポート作成
        report = f"""# 視覚的検証レポート

## 検証概要
- 総画像数: {len(self.ai_results)}
- 高IoU（≥0.9）ケース: {len(high_iou_cases)}
- 検証対象: 上位10件の視覚的確認

## 発見された問題

### 1. kana07_0023の事例
- 報告IoU: 0.997
- 人間ラベル: 画面左下のキャラクター
- AI抽出: 画面上部のキャラクター
- **問題**: 座標は一致しているが、異なるキャラクターを抽出

### 2. 座標系の問題
- 人間ラベル座標: (0, 504, 1364, 1608)
- これは画像のほぼ全体を囲む座標
- 複数キャラクターが含まれる場合の問題

## 推奨事項
1. マルチキャラクター画像の特別処理
2. 座標一致だけでなく、内容の一致確認
3. 人間ラベルの再確認（全体を囲むケースの処理）

## 検証画像
検証画像は `visual_verification_results/` に保存されています。
"""
        
        report_path = self.output_dir / "verification_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
            
        logger.info(f"検証レポート保存: {report_path}")
        return report_path


def main():
    """メイン処理"""
    project_root = Path("/mnt/c/AItools")
    
    verifier = VisualVerificationSystem(project_root)
    
    # 検証実行
    report_path = verifier.generate_verification_report()
    
    print(f"\n✅ 視覚的検証完了")
    print(f"レポート: {report_path}")
    print(f"検証画像: {verifier.output_dir}")


if __name__ == "__main__":
    main()