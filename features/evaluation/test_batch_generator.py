#!/usr/bin/env python3
"""
テストバッチ生成システム
Claudeの自己評価に基づくベスト5・ワースト5抽出機能
"""

import numpy as np
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TestBatchItem:
    """テストバッチ項目"""
    image_id: str
    image_path: str
    rank: int  # 1-10のランク
    category: str  # "best" or "worst"
    claude_score: float  # Claudeの自己評価スコア
    ground_truth_bbox: Tuple[int, int, int, int]  # 正解bbox
    predicted_bbox: Optional[Tuple[int, int, int, int]]  # 予測bbox
    iou_score: float
    confidence: float
    quality_grade: str
    issues: List[str]  # 発見された問題点
    notes: str


@dataclass
class TestBatchSummary:
    """テストバッチサマリー"""
    timestamp: str
    phase: str
    total_items: int
    best_items: List[TestBatchItem]
    worst_items: List[TestBatchItem]
    score_range: Tuple[float, float]  # (min, max)
    avg_score_best: float
    avg_score_worst: float
    key_insights: List[str]


class TestBatchGenerator:
    """テストバッチ生成器"""
    
    def __init__(self, project_root: Path):
        """
        初期化
        
        Args:
            project_root: プロジェクトルートディレクトリ
        """
        self.project_root = project_root
        # セキュリティ原則: プロジェクトルート直下への画像出力禁止
        self.output_dir = Path("/mnt/c/AItools/lora/train/yado/test_batches")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 人間ラベルデータ読み込み
        self.labels_file = project_root / "extracted_labels.json"
        self.ground_truth_labels = self.load_ground_truth_labels()
        
    def load_ground_truth_labels(self) -> Dict[str, Any]:
        """人間ラベルデータ読み込み"""
        try:
            if not self.labels_file.exists():
                logger.warning(f"ラベルファイルが見つかりません: {self.labels_file}")
                return {}
            
            with open(self.labels_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"人間ラベルデータ読み込み完了: {len(data)}ファイル")
            return data
            
        except Exception as e:
            logger.error(f"ラベルデータ読み込みエラー: {e}")
            return {}
    
    def calculate_claude_score(self, result: Dict[str, Any]) -> float:
        """
        Claudeの自己評価スコア計算
        
        Args:
            result: ベンチマーク結果
            
        Returns:
            0.0-1.0の自己評価スコア
        """
        try:
            if not isinstance(result, dict):
                logger.error(f"calculate_claude_score: 予期しないデータ型 {type(result)}")
                return 0.0
            # 複合スコア計算
            weights = {
                'iou_weight': 0.4,        # IoU重要度
                'confidence_weight': 0.25, # 信頼度重要度
                'quality_weight': 0.20,   # 品質グレード重要度
                'stability_weight': 0.15  # 処理安定性重要度
            }
            
            # IoUスコア (0.0-1.0)
            iou_score = result.get('iou_score', 0.0)
            
            # 信頼度スコア (0.0-1.0)
            confidence_score = result.get('confidence_score', 0.0)
            
            # 品質グレードスコア
            grade_mapping = {'A': 1.0, 'B': 0.8, 'C': 0.6, 'D': 0.4, 'E': 0.2, 'F': 0.0}
            quality_score = grade_mapping.get(result.get('quality_grade', 'F'), 0.0)
            
            # 処理安定性スコア（処理時間の逆数ベース）
            processing_time = result.get('processing_time', 10.0)
            stability_score = min(1.0, 5.0 / max(processing_time, 1.0))  # 5秒以下で1.0
            
            # 重み付き総合スコア
            claude_score = (
                iou_score * weights['iou_weight'] +
                confidence_score * weights['confidence_weight'] +
                quality_score * weights['quality_weight'] +
                stability_score * weights['stability_weight']
            )
            
            return min(1.0, max(0.0, claude_score))
            
        except Exception as e:
            logger.error(f"Claudeスコア計算エラー: {e}")
            return 0.0
    
    def analyze_issues(self, result: Dict[str, Any]) -> List[str]:
        """
        問題点分析
        
        Args:
            result: ベンチマーク結果
            
        Returns:
            発見された問題点のリスト
        """
        issues = []
        
        try:
            if not isinstance(result, dict):
                logger.error(f"analyze_issues: 予期しないデータ型 {type(result)}")
                return ["データ形式エラー"]
            iou_score = result.get('iou_score', 0.0)
            confidence = result.get('confidence_score', 0.0)
            processing_time = result.get('processing_time', 0.0)
            character_count = result.get('character_count', 0)
            
            # IoU関連の問題
            if iou_score < 0.1:
                issues.append("完全なミス抽出（IoU < 0.1）")
            elif iou_score < 0.3:
                issues.append("大幅な位置ずれ（IoU < 0.3）")
            elif iou_score < 0.5:
                issues.append("部分的な位置ずれ（IoU < 0.5）")
            
            # 信頼度関連の問題
            if confidence < 0.2:
                issues.append("極低信頼度検出（< 0.2）")
            elif confidence < 0.4:
                issues.append("低信頼度検出（< 0.4）")
            
            # 処理性能関連の問題
            if processing_time > 15.0:
                issues.append("処理時間過長（> 15秒）")
            elif processing_time > 10.0:
                issues.append("処理時間長い（> 10秒）")
            
            # キャラクター検出関連の問題
            if character_count == 0:
                issues.append("キャラクター未検出")
            elif character_count > 5:
                issues.append(f"過剰検出（{character_count}体検出）")
            
            # 予測bbox関連の問題
            pred_bbox = result.get('prediction_bbox')
            if pred_bbox is None:
                issues.append("予測bbox生成失敗")
            elif pred_bbox == (0, 0, 0, 0):
                issues.append("無効なbbox座標")
            
            # 品質グレード関連の警告
            quality_grade = result.get('quality_grade', 'F')
            if quality_grade == 'F':
                issues.append("品質評価F（要大幅改善）")
            elif quality_grade in ['D', 'E']:
                issues.append(f"低品質評価（{quality_grade}評価）")
            
            return issues
            
        except Exception as e:
            logger.error(f"問題分析エラー: {e}")
            return ["分析エラー"]
    
    def generate_test_batch(self, benchmark_results: List[Dict[str, Any]], 
                          phase: str = "phase0") -> TestBatchSummary:
        """
        テストバッチ生成
        
        Args:
            benchmark_results: ベンチマーク結果リスト
            phase: 現在のPhase
            
        Returns:
            テストバッチサマリー
        """
        try:
            logger.info("テストバッチ生成開始")
            
            # Claudeスコア計算
            scored_results = []
            for i, result in enumerate(benchmark_results):
                if i == 0:  # デバッグ用
                    logger.info(f"結果データ型: {type(result)}, 内容サンプル: {str(result)[:200]}")
                
                try:
                    claude_score = self.calculate_claude_score(result)
                    issues = self.analyze_issues(result)
                    
                    scored_result = {
                        **result,
                        'claude_score': claude_score,
                        'issues': issues
                    }
                    scored_results.append(scored_result)
                except Exception as e:
                    logger.error(f"結果処理エラー (項目{i}): {e}")
                    continue
            
            # スコア順ソート
            scored_results.sort(key=lambda x: x['claude_score'], reverse=True)
            
            # ベスト5・ワースト5選択
            best_5 = scored_results[:5]
            worst_5 = scored_results[-5:]
            
            # TestBatchItem作成
            best_items = []
            for i, result in enumerate(best_5, 1):
                try:
                    item = self.create_test_batch_item(result, i, "best")
                    best_items.append(item)
                except Exception as e:
                    logger.error(f"ベストアイテム作成エラー (項目{i}): {e}")
                    continue
            
            worst_items = []
            for i, result in enumerate(worst_5, 1):
                try:
                    item = self.create_test_batch_item(result, i + 5, "worst")
                    worst_items.append(item)
                except Exception as e:
                    logger.error(f"ワーストアイテム作成エラー (項目{i}): {e}")
                    continue
            
            # 統計計算
            all_scores = [r['claude_score'] for r in scored_results]
            score_range = (min(all_scores), max(all_scores))
            avg_score_best = np.mean([item.claude_score for item in best_items])
            avg_score_worst = np.mean([item.claude_score for item in worst_items])
            
            # 主要な洞察抽出
            key_insights = self.extract_key_insights(best_items, worst_items, scored_results)
            
            # サマリー作成
            summary = TestBatchSummary(
                timestamp=time.strftime("%Y%m%d_%H%M%S"),
                phase=phase,
                total_items=10,
                best_items=best_items,
                worst_items=worst_items,
                score_range=score_range,
                avg_score_best=avg_score_best,
                avg_score_worst=avg_score_worst,
                key_insights=key_insights
            )
            
            # 出力ファイル生成
            self.save_test_batch(summary)
            self.create_visual_batch(summary)
            
            logger.info("テストバッチ生成完了")
            return summary
            
        except Exception as e:
            logger.error(f"テストバッチ生成エラー: {e}")
            raise
    
    def create_test_batch_item(self, result: Dict[str, Any], 
                              rank: int, category: str) -> TestBatchItem:
        """TestBatchItem作成"""
        
        # Ground truthデータ取得
        image_id = result['image_id']
        
        # ground_truth_bboxが結果に含まれている場合はそれを使用
        if 'ground_truth_bbox' in result and result['ground_truth_bbox']:
            gt_bbox = tuple(result['ground_truth_bbox'])
        else:
            # ラベルファイルから取得
            gt_data = self.ground_truth_labels.get(image_id, {})
            if isinstance(gt_data, dict):
                gt_bbox = tuple(gt_data.get('red_box_coords', [0, 0, 0, 0]))
            else:
                gt_bbox = (0, 0, 0, 0)
        
        return TestBatchItem(
            image_id=image_id,
            image_path=result.get('image_path', ''),
            rank=rank,
            category=category,
            claude_score=result['claude_score'],
            ground_truth_bbox=gt_bbox,
            predicted_bbox=tuple(result['prediction_bbox']) if result.get('prediction_bbox') else None,
            iou_score=result.get('iou_score', 0.0),
            confidence=result.get('confidence_score', 0.0),
            quality_grade=result.get('quality_grade', 'F'),
            issues=result.get('issues', []),
            notes=f"Claude自己評価: {result['claude_score']:.3f}, "
                  f"問題数: {len(result.get('issues', []))}"
        )
    
    def extract_key_insights(self, best_items: List[TestBatchItem], 
                           worst_items: List[TestBatchItem],
                           all_results: List[Dict[str, Any]]) -> List[str]:
        """主要洞察の抽出"""
        insights = []
        
        try:
            # ベスト5の共通特徴
            best_avg_iou = np.mean([item.iou_score for item in best_items])
            best_avg_confidence = np.mean([item.confidence for item in best_items])
            insights.append(f"ベスト5平均: IoU {best_avg_iou:.3f}, 信頼度 {best_avg_confidence:.3f}")
            
            # ワースト5の共通問題
            worst_avg_iou = np.mean([item.iou_score for item in worst_items])
            worst_avg_confidence = np.mean([item.confidence for item in worst_items])
            insights.append(f"ワースト5平均: IoU {worst_avg_iou:.3f}, 信頼度 {worst_avg_confidence:.3f}")
            
            # 最も頻繁な問題の特定
            all_issues = []
            for result in all_results:
                all_issues.extend(result.get('issues', []))
            
            if all_issues:
                from collections import Counter
                issue_counts = Counter(all_issues)
                top_issue = issue_counts.most_common(1)[0]
                insights.append(f"最頻問題: {top_issue[0]} ({top_issue[1]}件)")
            
            # スコア分布
            scores = [r['claude_score'] for r in all_results]
            high_score_count = sum(1 for s in scores if s >= 0.7)
            insights.append(f"高スコア(≥0.7): {high_score_count}/{len(scores)}件 ({high_score_count/len(scores):.1%})")
            
            # 品質グレード分析
            grade_counts = {}
            for result in all_results:
                grade = result.get('quality_grade', 'F')
                grade_counts[grade] = grade_counts.get(grade, 0) + 1
            
            ab_count = grade_counts.get('A', 0) + grade_counts.get('B', 0)
            insights.append(f"A/B評価率: {ab_count}/{len(all_results)}件 ({ab_count/len(all_results):.1%})")
            
        except Exception as e:
            logger.error(f"洞察抽出エラー: {e}")
            insights.append("洞察抽出でエラーが発生しました")
        
        return insights
    
    def save_test_batch(self, summary: TestBatchSummary):
        """テストバッチJSON保存"""
        try:
            # JSON保存
            json_file = self.output_dir / f"test_batch_{summary.phase}_{summary.timestamp}.json"
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(summary), f, indent=2, ensure_ascii=False)
            
            # 最新版としてもコピー
            latest_file = self.output_dir / f"latest_test_batch_{summary.phase}.json"
            shutil.copy2(json_file, latest_file)
            
            logger.info(f"テストバッチJSON保存: {json_file}")
            
        except Exception as e:
            logger.error(f"テストバッチ保存エラー: {e}")
    
    def create_visual_batch(self, summary: TestBatchSummary):
        """可視化テストバッチ作成"""
        try:
            timestamp = summary.timestamp
            
            # 出力ディレクトリ作成
            visual_dir = self.output_dir / f"visual_batch_{summary.phase}_{timestamp}"
            visual_dir.mkdir(exist_ok=True)
            
            # ベスト5画像コピー・アノテーション
            best_dir = visual_dir / "best_5"
            best_dir.mkdir(exist_ok=True)
            
            for item in summary.best_items:
                self.create_annotated_image(item, best_dir)
            
            # ワースト5画像コピー・アノテーション
            worst_dir = visual_dir / "worst_5"
            worst_dir.mkdir(exist_ok=True)
            
            for item in summary.worst_items:
                self.create_annotated_image(item, worst_dir)
            
            # サマリー画像作成
            self.create_summary_image(summary, visual_dir)
            
            # README作成
            self.create_batch_readme(summary, visual_dir)
            
            logger.info(f"可視化テストバッチ作成: {visual_dir}")
            
        except Exception as e:
            logger.error(f"可視化バッチ作成エラー: {e}")
    
    def create_annotated_image(self, item: TestBatchItem, output_dir: Path):
        """アノテーション付き画像作成"""
        try:
            # 元画像読み込み
            if not Path(item.image_path).exists():
                # test_smallディレクトリから検索
                possible_paths = [
                    self.project_root / "test_small" / f"{item.image_id}.png",
                    self.project_root / "test_small" / f"{item.image_id}.jpg",
                    self.project_root / "test_small" / f"{item.image_id}.jpeg"
                ]
                
                image_path = None
                for path in possible_paths:
                    if path.exists():
                        image_path = path
                        break
                
                if image_path is None:
                    logger.warning(f"画像ファイルが見つかりません: {item.image_id}")
                    return
            else:
                image_path = Path(item.image_path)
            
            # 画像読み込み
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"画像読み込み失敗: {image_path}")
                return
            
            # BGR -> RGB変換
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # matplotlib図作成
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(image_rgb)
            
            # Ground Truth bbox (緑色)
            gt_bbox = item.ground_truth_bbox
            if gt_bbox != (0, 0, 0, 0):
                gt_rect = patches.Rectangle(
                    (gt_bbox[0], gt_bbox[1]), gt_bbox[2], gt_bbox[3],
                    linewidth=3, edgecolor='green', facecolor='none',
                    label='Ground Truth'
                )
                ax.add_patch(gt_rect)
            
            # Predicted bbox (赤色)
            if item.predicted_bbox:
                pred_bbox = item.predicted_bbox
                pred_rect = patches.Rectangle(
                    (pred_bbox[0], pred_bbox[1]), pred_bbox[2], pred_bbox[3],
                    linewidth=3, edgecolor='red', facecolor='none',
                    label='Prediction'
                )
                ax.add_patch(pred_rect)
            
            # タイトル・情報追加
            title = (f"Rank {item.rank} ({item.category.upper()}) - {item.image_id}\n"
                    f"Claude Score: {item.claude_score:.3f}, IoU: {item.iou_score:.3f}, "
                    f"Grade: {item.quality_grade}")
            ax.set_title(title, fontsize=12, fontweight='bold')
            
            # 問題点テキスト追加
            if item.issues:
                issues_text = "Issues: " + "; ".join(item.issues[:3])  # 最大3件表示
                ax.text(10, image_rgb.shape[0] - 30, issues_text, 
                       fontsize=10, color='red', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            # 凡例追加
            ax.legend(loc='upper right')
            ax.axis('off')
            
            # 保存
            output_file = output_dir / f"{item.rank:02d}_{item.image_id}_score{item.claude_score:.3f}.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"アノテーション画像作成エラー ({item.image_id}): {e}")
    
    def create_summary_image(self, summary: TestBatchSummary, output_dir: Path):
        """サマリー画像作成"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Test Batch Summary - {summary.phase.upper()} ({summary.timestamp})', 
                        fontsize=16, fontweight='bold')
            
            # 1. スコア分布比較
            ax1 = axes[0, 0]
            best_scores = [item.claude_score for item in summary.best_items]
            worst_scores = [item.claude_score for item in summary.worst_items]
            
            x_pos = np.arange(5)
            width = 0.35
            
            bars1 = ax1.bar(x_pos - width/2, best_scores, width, label='Best 5', color='green', alpha=0.7)
            bars2 = ax1.bar(x_pos + width/2, worst_scores, width, label='Worst 5', color='red', alpha=0.7)
            
            ax1.set_title('Claude Score Comparison')
            ax1.set_xlabel('Rank')
            ax1.set_ylabel('Claude Score')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels([f'#{i+1}' for i in range(5)])
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # バーに数値表示
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8)
            
            for bar in bars2:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8)
            
            # 2. IoU vs Confidence散布図
            ax2 = axes[0, 1]
            
            best_iou = [item.iou_score for item in summary.best_items]
            best_conf = [item.confidence for item in summary.best_items]
            worst_iou = [item.iou_score for item in summary.worst_items]
            worst_conf = [item.confidence for item in summary.worst_items]
            
            ax2.scatter(best_iou, best_conf, c='green', s=100, alpha=0.7, label='Best 5')
            ax2.scatter(worst_iou, worst_conf, c='red', s=100, alpha=0.7, label='Worst 5')
            
            ax2.set_title('IoU vs Confidence')
            ax2.set_xlabel('IoU Score')
            ax2.set_ylabel('Confidence')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. 品質グレード分布
            ax3 = axes[1, 0]
            
            all_items = summary.best_items + summary.worst_items
            grades = [item.quality_grade for item in all_items]
            grade_counts = {grade: grades.count(grade) for grade in ['A', 'B', 'C', 'D', 'E', 'F']}
            
            colors = ['#2ecc71', '#27ae60', '#f39c12', '#e67e22', '#e74c3c', '#c0392b']
            wedges, texts, autotexts = ax3.pie(
                list(grade_counts.values()), 
                labels=list(grade_counts.keys()),
                colors=colors,
                autopct='%1.0f%%',
                startangle=90
            )
            ax3.set_title('Quality Grade Distribution')
            
            # 4. 主要洞察テキスト
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            insights_text = "Key Insights:\n\n"
            for i, insight in enumerate(summary.key_insights, 1):
                insights_text += f"{i}. {insight}\n\n"
            
            insights_text += f"\nScore Range: {summary.score_range[0]:.3f} - {summary.score_range[1]:.3f}\n"
            insights_text += f"Best 5 Avg: {summary.avg_score_best:.3f}\n"
            insights_text += f"Worst 5 Avg: {summary.avg_score_worst:.3f}"
            
            ax4.text(0.05, 0.95, insights_text, transform=ax4.transAxes,
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            
            # 保存
            summary_file = output_dir / "test_batch_summary.png"
            plt.savefig(summary_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"サマリー画像作成: {summary_file}")
            
        except Exception as e:
            logger.error(f"サマリー画像作成エラー: {e}")
    
    def create_batch_readme(self, summary: TestBatchSummary, output_dir: Path):
        """テストバッチREADME作成"""
        try:
            readme_content = f"""# Test Batch Report - {summary.phase.upper()}

**生成日時**: {summary.timestamp}  
**Phase**: {summary.phase}  
**総アイテム数**: {summary.total_items}

---

## 📊 概要

### スコア統計
- **スコア範囲**: {summary.score_range[0]:.3f} - {summary.score_range[1]:.3f}
- **ベスト5平均**: {summary.avg_score_best:.3f}
- **ワースト5平均**: {summary.avg_score_worst:.3f}

### 主要洞察
"""
            
            for i, insight in enumerate(summary.key_insights, 1):
                readme_content += f"{i}. {insight}\n"
            
            readme_content += f"""

---

## 🏆 ベスト5（Claudeが高評価）

"""
            
            for item in summary.best_items:
                readme_content += f"""### #{item.rank} - {item.image_id}
- **Claudeスコア**: {item.claude_score:.3f}
- **IoU**: {item.iou_score:.3f}
- **信頼度**: {item.confidence:.3f}
- **品質グレード**: {item.quality_grade}
- **問題点**: {', '.join(item.issues) if item.issues else 'なし'}

"""
            
            readme_content += f"""---

## 💥 ワースト5（Claudeが低評価）

"""
            
            for item in summary.worst_items:
                readme_content += f"""### #{item.rank} - {item.image_id}
- **Claudeスコア**: {item.claude_score:.3f}
- **IoU**: {item.iou_score:.3f}
- **信頼度**: {item.confidence:.3f}
- **品質グレード**: {item.quality_grade}
- **問題点**: {', '.join(item.issues) if item.issues else 'なし'}

"""
            
            readme_content += f"""---

## 📁 ファイル構成

```
{output_dir.name}/
├── best_5/          # ベスト5画像（アノテーション付き）
├── worst_5/         # ワースト5画像（アノテーション付き）
├── test_batch_summary.png  # 統計サマリー画像
└── README.md        # このファイル
```

---

## 🎯 人間評価との比較観点

### チェックポイント
1. **ベスト5**: Claudeが高評価した画像は実際に良い抽出結果か？
2. **ワースト5**: Claudeが低評価した画像は実際に問題があるか？
3. **認識の乖離**: 人間の評価とClaude評価で大きく異なるケースは？

### 改善フィードバック
- ベスト5で問題があるケース → 評価基準の調整が必要
- ワースト5で良いケース → スコア算出方法の改善が必要
- 一貫して問題があるパターン → システム自体の改善が必要

---

*Generated by Test Batch Generator v1.0*
"""
            
            readme_file = output_dir / "README.md"
            with open(readme_file, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            
            logger.info(f"README作成: {readme_file}")
            
        except Exception as e:
            logger.error(f"README作成エラー: {e}")


def main():
    """メイン処理（テスト用）"""
    logging.basicConfig(level=logging.INFO)
    
    # サンプルデータでテスト
    project_root = Path("/mnt/c/AItools/segment-anything")
    generator = TestBatchGenerator(project_root)
    
    # サンプル結果データ
    sample_results = []
    for i in range(20):
        result = {
            'image_id': f'kana08_{i:04d}',
            'image_path': f'/test_small/kana08_{i:04d}.png',
            'largest_char_predicted': np.random.random() > 0.4,
            'iou_score': np.random.uniform(0.0, 1.0),
            'confidence_score': np.random.uniform(0.1, 0.9),
            'processing_time': np.random.uniform(3.0, 12.0),
            'character_count': np.random.randint(1, 6),
            'area_largest_ratio': np.random.uniform(0.2, 0.8),
            'quality_grade': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F'], 
                                            p=[0.05, 0.15, 0.25, 0.25, 0.20, 0.10]),
            'prediction_bbox': (
                int(np.random.uniform(50, 200)),
                int(np.random.uniform(50, 200)),
                int(np.random.uniform(100, 300)),
                int(np.random.uniform(150, 400))
            ) if np.random.random() > 0.2 else None
        }
        sample_results.append(result)
    
    # テストバッチ生成
    summary = generator.generate_test_batch(sample_results, "phase0")
    
    print(f"\n📊 テストバッチ生成完了")
    print(f"ベスト5平均スコア: {summary.avg_score_best:.3f}")
    print(f"ワースト5平均スコア: {summary.avg_score_worst:.3f}")
    print(f"出力ディレクトリ: {generator.output_dir}")


if __name__ == "__main__":
    main()