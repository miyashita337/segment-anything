#!/usr/bin/env python3
"""
正解マスクデータ検証ツール
ユーザーが作成した正解マスク（ground truth）の品質と整合性を確認

Usage:
    python tools/validate_evaluation_data.py --directory /path/to/masks/
    python tools/validate_evaluation_data.py --directory /path/to/masks/ --fix-issues
    python tools/validate_evaluation_data.py --check-all
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import cv2
from PIL import Image

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """検証結果データクラス"""
    file_path: str
    is_valid: bool
    issues: List[str]
    recommendations: List[str]
    quality_score: float


@dataclass
class OverallValidationReport:
    """全体検証レポート"""
    total_files: int
    valid_files: int
    invalid_files: int
    file_results: List[ValidationResult]
    summary_issues: List[str]
    summary_recommendations: List[str]
    overall_quality: float


class GroundTruthValidator:
    """正解マスクデータ検証システム"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.GroundTruthValidator")
        
        # 検証基準
        self.validation_criteria = {
            'min_white_ratio': 0.01,    # 最小白色領域比率（1%）
            'max_white_ratio': 0.95,    # 最大白色領域比率（95%）
            'min_resolution': (50, 50), # 最小解像度
            'max_resolution': (4000, 4000), # 最大解像度
            'required_channels': 1,      # グレースケール必須
            'valid_extensions': ['.png', '.jpg', '.jpeg'],
            'naming_patterns': ['_gt.png', '_ground_truth.png', 'gt_']
        }
    
    def validate_directory(self, directory: str, fix_issues: bool = False) -> OverallValidationReport:
        """ディレクトリ内の正解マスクを一括検証"""
        directory_path = Path(directory)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"ディレクトリが見つかりません: {directory}")
        
        self.logger.info(f"🔍 検証開始: {directory}")
        
        # 正解マスクファイルを検索
        mask_files = self._find_ground_truth_files(directory_path)
        
        if not mask_files:
            self.logger.warning("⚠️ 正解マスクファイルが見つかりませんでした")
            return OverallValidationReport(
                total_files=0,
                valid_files=0,
                invalid_files=0,
                file_results=[],
                summary_issues=["正解マスクファイルが見つかりません"],
                summary_recommendations=["_gt.png形式でマスクファイルを作成してください"],
                overall_quality=0.0
            )
        
        self.logger.info(f"📁 検証対象: {len(mask_files)}ファイル")
        
        # 各ファイルの検証
        validation_results = []
        for mask_file in mask_files:
            try:
                result = self._validate_single_file(mask_file, fix_issues)
                validation_results.append(result)
                
                status = "✅" if result.is_valid else "❌"
                self.logger.info(f"{status} {mask_file.name}: {result.quality_score:.3f}")
                
            except Exception as e:
                self.logger.error(f"❌ 検証エラー {mask_file.name}: {e}")
                validation_results.append(ValidationResult(
                    file_path=str(mask_file),
                    is_valid=False,
                    issues=[f"検証エラー: {e}"],
                    recommendations=["ファイルの破損を確認してください"],
                    quality_score=0.0
                ))
        
        # 全体レポート生成
        report = self._generate_overall_report(validation_results)
        
        self.logger.info(f"✅ 検証完了: {report.valid_files}/{report.total_files}ファイル有効")
        
        return report
    
    def _find_ground_truth_files(self, directory: Path) -> List[Path]:
        """正解マスクファイルを検索"""
        mask_files = []
        
        # 検索パターン
        patterns = ['*_gt.png', '*_ground_truth.png', 'gt_*.png']
        
        for pattern in patterns:
            mask_files.extend(list(directory.glob(pattern)))
        
        # 重複除去とソート
        mask_files = sorted(list(set(mask_files)))
        
        return mask_files
    
    def _validate_single_file(self, file_path: Path, fix_issues: bool = False) -> ValidationResult:
        """単一ファイルの検証"""
        issues = []
        recommendations = []
        quality_score = 1.0
        
        try:
            # 1. ファイル存在確認
            if not file_path.exists():
                issues.append("ファイルが存在しません")
                quality_score = 0.0
                return ValidationResult(str(file_path), False, issues, recommendations, quality_score)
            
            # 2. ファイル名検証
            name_valid, name_issues, name_recs = self._validate_filename(file_path)
            if not name_valid:
                issues.extend(name_issues)
                recommendations.extend(name_recs)
                quality_score -= 0.1
            
            # 3. 画像読み込み
            try:
                mask = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    # PILで再試行
                    pil_image = Image.open(file_path)
                    mask = np.array(pil_image.convert('L'))
            except Exception as e:
                issues.append(f"画像読み込みエラー: {e}")
                quality_score = 0.0
                return ValidationResult(str(file_path), False, issues, recommendations, quality_score)
            
            # 4. 画像形式検証
            format_valid, format_issues, format_recs, format_score = self._validate_image_format(mask, file_path)
            if not format_valid:
                issues.extend(format_issues)
                recommendations.extend(format_recs)
                quality_score -= format_score
            
            # 5. 画像内容検証
            content_valid, content_issues, content_recs, content_score = self._validate_image_content(mask)
            if not content_valid:
                issues.extend(content_issues)
                recommendations.extend(content_recs)
                quality_score -= content_score
            
            # 6. 対応する元画像確認
            original_valid, original_issues, original_recs = self._validate_original_image(file_path)
            if not original_valid:
                issues.extend(original_issues)
                recommendations.extend(original_recs)
                quality_score -= 0.1
            
            # 7. 修正処理（オプション）
            if fix_issues and issues:
                self._attempt_fixes(file_path, mask, issues)
            
            # 最終スコア調整
            quality_score = max(0.0, min(1.0, quality_score))
            is_valid = quality_score >= 0.7 and len(issues) == 0
            
            return ValidationResult(
                file_path=str(file_path),
                is_valid=is_valid,
                issues=issues,
                recommendations=recommendations,
                quality_score=quality_score
            )
            
        except Exception as e:
            return ValidationResult(
                file_path=str(file_path),
                is_valid=False,
                issues=[f"予期しないエラー: {e}"],
                recommendations=["技術サポートに連絡してください"],
                quality_score=0.0
            )
    
    def _validate_filename(self, file_path: Path) -> Tuple[bool, List[str], List[str]]:
        """ファイル名検証"""
        issues = []
        recommendations = []
        
        # 拡張子確認
        if file_path.suffix.lower() not in self.validation_criteria['valid_extensions']:
            issues.append(f"非対応拡張子: {file_path.suffix}")
            recommendations.append("PNG形式(.png)を推奨")
        
        # 命名規則確認
        filename = file_path.name.lower()
        pattern_found = any(pattern in filename for pattern in self.validation_criteria['naming_patterns'])
        
        if not pattern_found:
            issues.append("標準命名規則に従っていません")
            recommendations.append("ファイル名を '{元画像名}_gt.png' に変更してください")
        
        return len(issues) == 0, issues, recommendations
    
    def _validate_image_format(self, mask: np.ndarray, file_path: Path) -> Tuple[bool, List[str], List[str], float]:
        """画像形式検証"""
        issues = []
        recommendations = []
        score_penalty = 0.0
        
        # 解像度確認
        height, width = mask.shape[:2]
        min_h, min_w = self.validation_criteria['min_resolution']
        max_h, max_w = self.validation_criteria['max_resolution']
        
        if height < min_h or width < min_w:
            issues.append(f"解像度が低すぎます: {width}x{height}")
            recommendations.append(f"最小{min_w}x{min_h}以上にしてください")
            score_penalty += 0.3
        
        if height > max_h or width > max_w:
            issues.append(f"解像度が高すぎます: {width}x{height}")
            recommendations.append(f"最大{max_w}x{max_h}以下にしてください")
            score_penalty += 0.1
        
        # チャンネル数確認
        if len(mask.shape) != 2:
            issues.append(f"グレースケールではありません: {len(mask.shape)}チャンネル")
            recommendations.append("グレースケール画像で保存してください")
            score_penalty += 0.2
        
        return len(issues) == 0, issues, recommendations, score_penalty
    
    def _validate_image_content(self, mask: np.ndarray) -> Tuple[bool, List[str], List[str], float]:
        """画像内容検証"""
        issues = []
        recommendations = []
        score_penalty = 0.0
        
        # バイナリマスク確認
        unique_values = np.unique(mask)
        
        # 完全バイナリ（0と255のみ）でない場合は警告
        if not (len(unique_values) == 2 and 0 in unique_values and 255 in unique_values):
            if len(unique_values) > 10:
                issues.append("バイナリマスクではありません（多階調値検出）")
                recommendations.append("白(255)と黒(0)のみの画像にしてください")
                score_penalty += 0.3
            else:
                # ほぼバイナリの場合は軽微な警告
                recommendations.append("可能な限り純粋な白(255)と黒(0)で描画してください")
                score_penalty += 0.1
        
        # 白色領域比率確認
        white_pixels = np.sum(mask > 127)  # 閾値127以上を白とみなす
        total_pixels = mask.size
        white_ratio = white_pixels / total_pixels
        
        min_ratio = self.validation_criteria['min_white_ratio']
        max_ratio = self.validation_criteria['max_white_ratio']
        
        if white_ratio < min_ratio:
            issues.append(f"白色領域が少なすぎます: {white_ratio:.3f}")
            recommendations.append("キャラクター部分を白色で描画してください")
            score_penalty += 0.4
        
        if white_ratio > max_ratio:
            issues.append(f"白色領域が多すぎます: {white_ratio:.3f}")
            recommendations.append("背景は黒色で描画してください")
            score_penalty += 0.2
        
        # ノイズ確認（小さな白色領域の検出）
        if white_ratio > 0:
            contours, _ = cv2.findContours(
                (mask > 127).astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if len(contours) > 5:  # 多数の独立領域がある場合
                issues.append(f"多数の独立した白色領域: {len(contours)}個")
                recommendations.append("ノイズを除去し、キャラクター部分のみを描画してください")
                score_penalty += 0.1
        
        return len(issues) == 0, issues, recommendations, score_penalty
    
    def _validate_original_image(self, mask_path: Path) -> Tuple[bool, List[str], List[str]]:
        """対応する元画像の確認"""
        issues = []
        recommendations = []
        
        # 元画像ファイル名を推定
        mask_name = mask_path.stem
        
        # _gt, _ground_truth等を除去して元ファイル名を復元
        for pattern in ['_gt', '_ground_truth']:
            if mask_name.endswith(pattern):
                original_name = mask_name[:-len(pattern)]
                break
        else:
            if mask_name.startswith('gt_'):
                original_name = mask_name[3:]
            else:
                original_name = mask_name
        
        # 元画像ファイルを検索
        original_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        original_found = False
        
        for ext in original_extensions:
            original_path = mask_path.parent / f"{original_name}{ext}"
            if original_path.exists():
                original_found = True
                break
        
        if not original_found:
            issues.append(f"対応する元画像が見つかりません: {original_name}")
            recommendations.append("元画像ファイルが同じディレクトリにあることを確認してください")
        
        return original_found, issues, recommendations
    
    def _attempt_fixes(self, file_path: Path, mask: np.ndarray, issues: List[str]):
        """自動修正の試行"""
        try:
            # バイナリ化処理
            if "バイナリマスクではありません" in str(issues):
                _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                backup_path = file_path.with_suffix(f'.backup{file_path.suffix}')
                # バックアップ作成
                cv2.imwrite(str(backup_path), mask)
                # 修正版保存
                cv2.imwrite(str(file_path), binary_mask)
                self.logger.info(f"🔧 自動修正完了: {file_path.name} (バックアップ: {backup_path.name})")
        
        except Exception as e:
            self.logger.warning(f"⚠️ 自動修正失敗 {file_path.name}: {e}")
    
    def _generate_overall_report(self, validation_results: List[ValidationResult]) -> OverallValidationReport:
        """全体レポート生成"""
        total_files = len(validation_results)
        valid_files = sum(1 for r in validation_results if r.is_valid)
        invalid_files = total_files - valid_files
        
        # 共通問題の集計
        all_issues = []
        all_recommendations = []
        
        for result in validation_results:
            all_issues.extend(result.issues)
            all_recommendations.extend(result.recommendations)
        
        # 頻出問題をサマリーに
        summary_issues = list(set(all_issues))[:5]  # 上位5件
        summary_recommendations = list(set(all_recommendations))[:5]  # 上位5件
        
        # 全体品質スコア
        if total_files > 0:
            overall_quality = sum(r.quality_score for r in validation_results) / total_files
        else:
            overall_quality = 0.0
        
        return OverallValidationReport(
            total_files=total_files,
            valid_files=valid_files,
            invalid_files=invalid_files,
            file_results=validation_results,
            summary_issues=summary_issues,
            summary_recommendations=summary_recommendations,
            overall_quality=overall_quality
        )


class ValidationReportGenerator:
    """検証レポート生成"""
    
    def __init__(self, output_dir: str = "validation_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.ValidationReportGenerator")
    
    def generate_report(self, report: OverallValidationReport, directory: str) -> str:
        """詳細レポートの生成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        directory_name = Path(directory).name
        
        # テキストレポート
        report_path = self.output_dir / f"validation_{directory_name}_{timestamp}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(self._generate_report_content(report, directory))
        
        self.logger.info(f"📋 レポート生成完了: {report_path}")
        return str(report_path)
    
    def _generate_report_content(self, report: OverallValidationReport, directory: str) -> str:
        """レポート内容生成"""
        content = f"""
=============================================================
🔍 正解マスクデータ検証レポート
=============================================================

📅 検証日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📁 検証ディレクトリ: {directory}

📊 検証結果サマリー:
  総ファイル数: {report.total_files}
  ✅ 有効ファイル: {report.valid_files}
  ❌ 無効ファイル: {report.invalid_files}
  📈 全体品質スコア: {report.overall_quality:.3f} / 1.000

🎯 品質評価:
  {self._get_quality_assessment(report.overall_quality)}

"""
        
        # 主要問題
        if report.summary_issues:
            content += "⚠️ 主要な問題:\n"
            for issue in report.summary_issues:
                content += f"  • {issue}\n"
            content += "\n"
        
        # 推奨改善策
        if report.summary_recommendations:
            content += "💡 推奨改善策:\n"
            for rec in report.summary_recommendations:
                content += f"  • {rec}\n"
            content += "\n"
        
        # 個別ファイル結果
        content += "📋 個別ファイル検証結果:\n"
        content += "=" * 60 + "\n"
        
        for result in report.file_results:
            status = "✅ 有効" if result.is_valid else "❌ 無効"
            filename = Path(result.file_path).name
            
            content += f"{status} | {filename} | スコア: {result.quality_score:.3f}\n"
            
            if result.issues:
                for issue in result.issues:
                    content += f"    ⚠️ {issue}\n"
            
            if result.recommendations:
                for rec in result.recommendations:
                    content += f"    💡 {rec}\n"
            
            content += "\n"
        
        # 次のステップ
        content += self._generate_next_steps(report)
        
        return content
    
    def _get_quality_assessment(self, score: float) -> str:
        """品質評価メッセージ"""
        if score >= 0.9:
            return "🎉 優秀 - 高品質なマスクデータです"
        elif score >= 0.7:
            return "✅ 良好 - 軽微な改善で使用可能です"
        elif score >= 0.5:
            return "⚠️ 要改善 - いくつかの問題があります"
        else:
            return "❌ 品質不足 - 大幅な修正が必要です"
    
    def _generate_next_steps(self, report: OverallValidationReport) -> str:
        """次のステップ提案"""
        content = "🚀 次のステップ:\n"
        content += "=" * 30 + "\n"
        
        if report.overall_quality >= 0.7:
            content += "✅ 品質基準をクリアしています\n"
            content += "  • PLA評価システムで客観的評価を実行してください\n"
            content += "  • コマンド例: python tools/run_objective_evaluation.py --batch [ディレクトリ]\n\n"
        else:
            content += "⚠️ 品質改善が必要です\n"
            content += "  • 上記の推奨改善策を実施してください\n"
            content += "  • 修正後、再度検証を実行してください\n"
            content += "  • コマンド例: python tools/validate_evaluation_data.py --directory [ディレクトリ] --fix-issues\n\n"
        
        content += "📈 進捗管理:\n"
        content += "  • PROGRESS_TRACKER.mdで作業進捗を更新\n"
        content += "  • 目標: 15枚の正解マスク完成\n"
        content += f"  • 現在: {report.valid_files}枚完了済み\n\n"
        
        return content


def main():
    parser = argparse.ArgumentParser(description="正解マスクデータ検証ツール")
    parser.add_argument("--directory", "-d", help="検証対象ディレクトリ")
    parser.add_argument("--check-all", action="store_true", help="全ての既知ディレクトリを検証")
    parser.add_argument("--fix-issues", action="store_true", help="可能な問題を自動修正")
    parser.add_argument("--output", help="レポート出力ディレクトリ", default="validation_reports")
    parser.add_argument("--verbose", "-v", action="store_true", help="詳細ログ出力")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        validator = GroundTruthValidator()
        report_generator = ValidationReportGenerator(args.output)
        
        # 検証対象ディレクトリの決定
        if args.check_all:
            # 既知のディレクトリを全て検証
            directories = [
                "/mnt/c/AItools/lora/train/yado/org/kana08_cursor_fix/",
                "/mnt/c/AItools/lora/train/yado/org/kana07_cursor_fix/",
                "/mnt/c/AItools/lora/train/yado/org/kana05_cursor_fix/"
            ]
            directories = [d for d in directories if Path(d).exists()]
        elif args.directory:
            directories = [args.directory]
        else:
            print("❌ エラー: --directory または --check-all を指定してください")
            return 1
        
        all_reports = []
        
        for directory in directories:
            print(f"\n{'='*60}")
            print(f"🔍 検証開始: {directory}")
            print(f"{'='*60}")
            
            # 検証実行
            report = validator.validate_directory(directory, args.fix_issues)
            all_reports.append((directory, report))
            
            # 結果表示
            print(f"\n📊 検証結果:")
            print(f"  総ファイル数: {report.total_files}")
            print(f"  ✅ 有効: {report.valid_files}")
            print(f"  ❌ 無効: {report.invalid_files}")
            print(f"  📈 品質スコア: {report.overall_quality:.3f}")
            
            if report.summary_issues:
                print(f"\n⚠️ 主要な問題:")
                for issue in report.summary_issues[:3]:
                    print(f"  • {issue}")
            
            # レポート生成
            report_path = report_generator.generate_report(report, directory)
            print(f"\n📋 詳細レポート: {report_path}")
        
        # 全体サマリー
        if len(all_reports) > 1:
            print(f"\n{'='*60}")
            print(f"📈 全体サマリー")
            print(f"{'='*60}")
            
            total_files = sum(r[1].total_files for r in all_reports)
            total_valid = sum(r[1].valid_files for r in all_reports)
            avg_quality = sum(r[1].overall_quality for r in all_reports) / len(all_reports)
            
            print(f"  総検証ファイル数: {total_files}")
            print(f"  総有効ファイル数: {total_valid}")
            print(f"  平均品質スコア: {avg_quality:.3f}")
            print(f"  完了率: {total_valid/15*100:.1f}% (目標15枚)")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ 実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)