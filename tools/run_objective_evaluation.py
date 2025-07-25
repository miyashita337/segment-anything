#!/usr/bin/env python3
"""
客観的評価システム実行ツール
既存のsegment-anythingシステムと統合した評価実行

Usage:
    python tools/run_objective_evaluation.py --batch results_batch/
    python tools/run_objective_evaluation.py --batch results_batch/ --config config/evaluation.json
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import cv2
from PIL import Image

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from features.evaluation.objective_evaluation_system import ObjectiveEvaluationSystem
# 通知システム（利用可能な場合のみ）
try:
    from features.common.notification.notification import NotificationManager
except ImportError:
    NotificationManager = None

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchResultsLoader:
    """バッチ処理結果の読み込み"""
    
    def __init__(self, batch_path: str):
        self.batch_path = Path(batch_path)
        self.logger = logging.getLogger(f"{__name__}.BatchResultsLoader")
    
    def load_extraction_results(self) -> List[Dict]:
        """抽出結果の読み込み"""
        if not self.batch_path.exists():
            raise FileNotFoundError(f"バッチディレクトリが見つかりません: {self.batch_path}")
        
        results = []
        
        # 抽出済み画像ファイルを検索
        image_files = list(self.batch_path.glob("*.jpg")) + \
                     list(self.batch_path.glob("*.png")) + \
                     list(self.batch_path.glob("*.webp"))
        
        self.logger.info(f"画像ファイル発見: {len(image_files)}件")
        
        for image_file in image_files:
            try:
                # 抽出画像の読み込み
                extracted_image = self._load_image(image_file)
                
                # 対応するマスクファイルを検索
                mask_file = self._find_mask_file(image_file)
                predicted_mask = self._load_mask(mask_file) if mask_file else None
                
                # 正解データを検索（存在する場合）
                ground_truth_file = self._find_ground_truth_file(image_file)
                ground_truth_mask = self._load_mask(ground_truth_file) if ground_truth_file else None
                
                result = {
                    'image_path': str(image_file),
                    'extracted_image': extracted_image,
                    'predicted_mask': predicted_mask,
                    'ground_truth_mask': ground_truth_mask
                }
                
                results.append(result)
                
            except Exception as e:
                self.logger.warning(f"画像読み込みエラー {image_file}: {e}")
                continue
        
        self.logger.info(f"読み込み完了: {len(results)}件の結果")
        return results
    
    def _load_image(self, image_path: Path) -> np.ndarray:
        """画像ファイルの読み込み"""
        image = cv2.imread(str(image_path))
        if image is None:
            # PILで再試行
            pil_image = Image.open(image_path)
            image = np.array(pil_image.convert('RGB'))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image
    
    def _load_mask(self, mask_path: Optional[Path]) -> Optional[np.ndarray]:
        """マスクファイルの読み込み"""
        if mask_path is None or not mask_path.exists():
            return None
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        return mask
    
    def _find_mask_file(self, image_file: Path) -> Optional[Path]:
        """対応するマスクファイルを検索"""
        base_name = image_file.stem
        
        # 一般的なマスクファイル命名パターン
        mask_patterns = [
            f"{base_name}_mask.png",
            f"{base_name}_mask.jpg",
            f"mask_{base_name}.png",
            f"{base_name}.mask.png"
        ]
        
        for pattern in mask_patterns:
            mask_file = self.batch_path / pattern
            if mask_file.exists():
                return mask_file
        
        return None
    
    def _find_ground_truth_file(self, image_file: Path) -> Optional[Path]:
        """正解マスクファイルを検索"""
        base_name = image_file.stem
        
        # 正解データ命名パターン
        gt_patterns = [
            f"{base_name}_gt.png",
            f"{base_name}_ground_truth.png",
            f"gt_{base_name}.png"
        ]
        
        # 別ディレクトリも検索
        possible_dirs = [
            self.batch_path,
            self.batch_path / "ground_truth",
            self.batch_path / "gt",
            self.batch_path.parent / "ground_truth"
        ]
        
        for directory in possible_dirs:
            if not directory.exists():
                continue
                
            for pattern in gt_patterns:
                gt_file = directory / pattern
                if gt_file.exists():
                    return gt_file
        
        return None


class EvaluationReportManager:
    """評価レポート管理"""
    
    def __init__(self, output_dir: str = "evaluation_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.EvaluationReportManager")
    
    def save_comprehensive_report(self, report, batch_path: str):
        """包括的レポートの保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_name = Path(batch_path).name
        
        # JSONレポート
        json_path = self.output_dir / f"evaluation_{batch_name}_{timestamp}.json"
        
        # テキストレポート
        txt_path = self.output_dir / f"evaluation_{batch_name}_{timestamp}.txt"
        
        # CSVサマリー（後で分析用）
        csv_path = self.output_dir / f"evaluation_{batch_name}_{timestamp}_summary.csv"
        
        try:
            # ObjectiveEvaluationSystemのsave_reportメソッドを使用
            evaluator = ObjectiveEvaluationSystem()
            evaluator.save_report(report, str(json_path))
            
            # CSVサマリーの作成
            self._create_csv_summary(report, csv_path)
            
            self.logger.info(f"レポート保存完了: {json_path}")
            return json_path
            
        except Exception as e:
            self.logger.error(f"レポート保存エラー: {e}")
            raise
    
    def _create_csv_summary(self, report, csv_path: Path):
        """CSV形式のサマリー作成"""
        try:
            import csv
            
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # ヘッダー
                writer.writerow([
                    'timestamp', 'batch_size', 'pla_mean', 'pla_std', 'sci_mean', 'sci_std',
                    'ple_score', 'overall_quality', 'phase_a1_progress', 'phase_a2_progress',
                    'alerts_count'
                ])
                
                # データ行
                writer.writerow([
                    report.timestamp.isoformat(),
                    report.batch_size,
                    round(report.pla_statistics.mean, 4) if report.pla_statistics else 0,
                    round(report.pla_statistics.std, 4) if report.pla_statistics else 0,
                    round(report.sci_statistics.mean, 4) if report.sci_statistics else 0,
                    round(report.sci_statistics.std, 4) if report.sci_statistics else 0,
                    round(report.ple_result.ple_score, 4),
                    round(report.overall_quality_score, 4),
                    round(report.milestone_progress.get('phase_a1', 0), 4),
                    round(report.milestone_progress.get('phase_a2', 0), 4),
                    len(report.alerts)
                ])
                
        except Exception as e:
            self.logger.warning(f"CSV作成エラー: {e}")


def setup_notification_if_available():
    """通知システム設定（利用可能な場合）"""
    if NotificationManager is None:
        logger.info("通知システム未実装 - 通知は無効")
        return None
        
    try:
        config_path = Path("config/pushover.json")
        if config_path.exists():
            return NotificationManager(str(config_path))
        else:
            logger.info("通知設定ファイル未発見 - 通知は無効")
            return None
    except Exception as e:
        logger.warning(f"通知システム初期化失敗: {e}")
        return None


def send_completion_notification(notification_manager, report, batch_path: str):
    """完了通知送信"""
    if notification_manager is None:
        return
    
    try:
        batch_name = Path(batch_path).name
        message = f"""
📊 客観的評価完了: {batch_name}

🎯 結果サマリー:
  画像数: {report.batch_size}
  総合品質: {report.overall_quality_score:.3f} ({report.overall_quality_level})
  
  PLA: {report.pla_statistics.mean:.3f} ± {report.pla_statistics.std:.3f}
  SCI: {report.sci_statistics.mean:.3f} ± {report.sci_statistics.std:.3f}
  PLE: {report.ple_result.ple_score:.3f} ({report.ple_result.learning_status})

📈 マイルストーン進捗:
  Phase A1: {report.milestone_progress.get('phase_a1', 0):.1%}
  Phase A2: {report.milestone_progress.get('phase_a2', 0):.1%}

{'⚠️ アラート: ' + str(len(report.alerts)) + '件' if report.alerts else '✅ アラートなし'}
        """.strip()
        
        notification_manager.send_message("客観的評価完了", message)
        
    except Exception as e:
        logger.warning(f"通知送信エラー: {e}")


def main():
    parser = argparse.ArgumentParser(description="客観的評価システム実行ツール")
    parser.add_argument("--batch", required=True, help="バッチ処理結果ディレクトリ")
    parser.add_argument("--config", help="評価システム設定ファイル")
    parser.add_argument("--output", help="レポート出力ディレクトリ", default="evaluation_reports")
    parser.add_argument("--notify", action="store_true", help="完了通知を送信")
    parser.add_argument("--verbose", "-v", action="store_true", help="詳細ログ出力")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        logger.info("=" * 60)
        logger.info("📊 客観的評価システム実行開始")
        logger.info("=" * 60)
        
        # 1. バッチ結果の読み込み
        logger.info(f"🔍 バッチ結果読み込み: {args.batch}")
        loader = BatchResultsLoader(args.batch)
        extraction_results = loader.load_extraction_results()
        
        if not extraction_results:
            logger.error("❌ 評価対象の画像が見つかりませんでした")
            return 1
        
        # 2. 客観的評価システム初期化
        logger.info("🧮 客観的評価システム初期化")
        evaluator = ObjectiveEvaluationSystem(args.config)
        
        # 3. バッチ評価実行
        logger.info(f"⚡ 評価実行開始: {len(extraction_results)}画像")
        report = evaluator.evaluate_batch(extraction_results)
        
        # 4. レポート保存
        logger.info("💾 レポート保存")
        report_manager = EvaluationReportManager(args.output)
        report_path = report_manager.save_comprehensive_report(report, args.batch)
        
        # 5. 結果表示
        print()
        print(evaluator.generate_detailed_report(report))
        
        # 6. 通知送信（オプション）
        if args.notify:
            logger.info("📬 完了通知送信")
            notification_manager = setup_notification_if_available()
            send_completion_notification(notification_manager, report, args.batch)
        
        logger.info("=" * 60)
        logger.info("✅ 客観的評価システム実行完了")
        logger.info(f"📋 レポートファイル: {report_path}")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ 実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)