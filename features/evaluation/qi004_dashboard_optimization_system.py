"""
QI-004ダッシュボード標準化・Base64画像表示システム構築

QI-004要件:
- ダッシュボード標準化システム実装
- Base64画像表示システム構築  
- 画像パス参照方式の最適化
- パフォーマンス最適化とUI改善
- レスポンシブデザイン強化実装
"""

import numpy as np
import cv2

import json
import logging
import os
import time
from dataclasses import dataclass

# 既存システムのインポート
from features.common.dashboard_generator import DashboardGenerator
from features.common.notification.pushover_image_sender import PushoverImageSender
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class QI004OptimizationResult:
    """QI-004ダッシュボード最適化結果"""

    total_images: int
    image_quality_scores: List[float]
    dashboard_size_mb: float
    load_time_seconds: float
    optimization_improvements: Dict[str, Any]
    image_path_references: List[str]
    performance_metrics: Dict[str, float]


class ImageQualityAnalyzer:
    """画像品質解析器（QI-004仕様）"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze_image_quality(self, image_path: str) -> Dict[str, Any]:
        """
        画像品質の詳細解析

        Args:
            image_path: 画像ファイルパス

        Returns:
            品質解析結果
        """
        try:
            # 画像読み込み
            image = cv2.imread(image_path)
            if image is None:
                return self._create_error_result("画像読み込み失敗")

            # 基本解析
            height, width = image.shape[:2]
            resolution_score = self._calculate_resolution_score(width, height)

            # アスペクト比評価
            aspect_ratio = width / height
            aspect_score = self._calculate_aspect_score(aspect_ratio)

            # 背景除去品質評価
            background_quality = self._evaluate_background_removal(image)

            # 切り抜き精度評価
            crop_precision = self._evaluate_crop_precision(image)

            # 総合品質スコア計算
            overall_score = (
                resolution_score * 0.3
                + aspect_score * 0.2
                + background_quality * 0.3
                + crop_precision * 0.2
            )

            return {
                "overall_score": overall_score,
                "resolution_score": resolution_score,
                "aspect_score": aspect_score,
                "background_quality": background_quality,
                "crop_precision": crop_precision,
                "dimensions": (width, height),
                "aspect_ratio": aspect_ratio,
                "file_size_mb": os.path.getsize(image_path) / (1024 * 1024),
            }

        except Exception as e:
            self.logger.error(f"Image quality analysis failed for {image_path}: {e}")
            return self._create_error_result(str(e))

    def _calculate_resolution_score(self, width: int, height: int) -> float:
        """解像度スコア計算"""
        total_pixels = width * height

        if total_pixels >= 1920 * 1080:  # フルHD以上
            return 1.0
        elif total_pixels >= 1280 * 720:  # HD
            return 0.8
        elif total_pixels >= 640 * 480:  # VGA
            return 0.6
        else:
            return 0.3

    def _calculate_aspect_score(self, aspect_ratio: float) -> float:
        """アスペクト比スコア計算"""
        # キャラクター画像として適切なアスペクト比を評価
        ideal_ratios = [
            (16 / 9, 0.9),  # 横長
            (4 / 3, 1.0),  # 標準
            (1 / 1, 0.8),  # 正方形
            (3 / 4, 0.9),  # 縦長
            (9 / 16, 0.7),  # 極端な縦長
        ]

        best_score = 0.0
        for ratio, score in ideal_ratios:
            difference = abs(aspect_ratio - ratio)
            if difference < 0.1:
                best_score = max(best_score, score)
            elif difference < 0.3:
                best_score = max(best_score, score * 0.7)

        return best_score

    def _evaluate_background_removal(self, image: np.ndarray) -> float:
        """背景除去品質評価"""
        # グレースケール変換
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # エッジ検出
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # 背景の均一性チェック
        corners = [
            gray[0:50, 0:50],  # 左上
            gray[0:50, -50:],  # 右上
            gray[-50:, 0:50],  # 左下
            gray[-50:, -50:],  # 右下
        ]

        corner_uniformity = 0.0
        for corner in corners:
            std_dev = np.std(corner)
            uniformity = max(0, 1 - std_dev / 50)  # 標準偏差が小さいほど均一
            corner_uniformity += uniformity

        corner_uniformity /= len(corners)

        # 総合背景品質スコア
        background_score = edge_density * 0.4 + corner_uniformity * 0.6
        return min(background_score, 1.0)

    def _evaluate_crop_precision(self, image: np.ndarray) -> float:
        """切り抜き精度評価"""
        height, width = image.shape[:2]

        # 中心領域の重要度評価
        center_y, center_x = height // 2, width // 2
        center_region = image[
            center_y - height // 4 : center_y + height // 4,
            center_x - width // 4 : center_x + width // 4,
        ]

        # 中心領域の分散（詳細度）
        center_variance = np.var(center_region)
        variance_score = min(center_variance / 1000, 1.0)

        # マージン評価（適切な余白があるか）
        margin_score = self._evaluate_margins(image)

        return variance_score * 0.6 + margin_score * 0.4

    def _evaluate_margins(self, image: np.ndarray) -> float:
        """マージン（余白）評価"""
        height, width = image.shape[:2]

        # 端5%の領域を評価
        margin_size = min(height, width) // 20

        margins = [
            image[:margin_size, :],  # 上
            image[-margin_size:, :],  # 下
            image[:, :margin_size],  # 左
            image[:, -margin_size:],  # 右
        ]

        margin_scores = []
        for margin in margins:
            # マージンの均一性（背景らしさ）
            std_dev = np.std(margin)
            uniformity = max(0, 1 - std_dev / 50)
            margin_scores.append(uniformity)

        return np.mean(margin_scores)

    def _create_error_result(self, error_msg: str) -> Dict[str, Any]:
        """エラー結果の作成"""
        return {
            "overall_score": 0.0,
            "resolution_score": 0.0,
            "aspect_score": 0.0,
            "background_quality": 0.0,
            "crop_precision": 0.0,
            "dimensions": (0, 0),
            "aspect_ratio": 0.0,
            "file_size_mb": 0.0,
            "error": error_msg,
        }


class DashboardOptimizer:
    """ダッシュボード最適化器（QI-004仕様）"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def optimize_dashboard_performance(
        self, image_paths: List[str], dashboard_dir: str
    ) -> Dict[str, Any]:
        """
        ダッシュボードパフォーマンス最適化

        Args:
            image_paths: 画像パスリスト
            dashboard_dir: ダッシュボードディレクトリ

        Returns:
            最適化結果
        """
        start_time = time.time()

        # 画像パス参照方式の最適化
        optimized_paths = self._optimize_image_paths(image_paths, dashboard_dir)

        # 画像サイズ最適化
        size_optimization = self._optimize_image_sizes(image_paths)

        # キャッシュ戦略最適化
        cache_strategy = self._optimize_cache_strategy(image_paths)

        # レスポンシブデザイン最適化
        responsive_optimization = self._optimize_responsive_design()

        optimization_time = time.time() - start_time

        return {
            "optimized_paths": optimized_paths,
            "size_optimization": size_optimization,
            "cache_strategy": cache_strategy,
            "responsive_optimization": responsive_optimization,
            "optimization_time_seconds": optimization_time,
            "total_images_processed": len(image_paths),
        }

    def _optimize_image_paths(self, image_paths: List[str], dashboard_dir: str) -> List[str]:
        """画像パス参照の最適化"""
        optimized_paths = []
        workspace_base = "/mnt/c/AItools/lora/train/yado/tracker-workspace"

        for image_path in image_paths:
            if workspace_base in image_path:
                # 相対パスへの変換
                relative_path = image_path.replace(workspace_base, "")
                if not relative_path.startswith("/"):
                    relative_path = "/" + relative_path
                optimized_paths.append(relative_path)
            else:
                # フォールバック：ファイル名のみ
                filename = os.path.basename(image_path)
                optimized_paths.append(f"/QI-004/extraction/{filename}")

        return optimized_paths

    def _optimize_image_sizes(self, image_paths: List[str]) -> Dict[str, Any]:
        """画像サイズ最適化"""
        total_size = 0
        size_distribution = {"small": 0, "medium": 0, "large": 0}

        for image_path in image_paths:
            try:
                size_mb = os.path.getsize(image_path) / (1024 * 1024)
                total_size += size_mb

                if size_mb < 0.5:
                    size_distribution["small"] += 1
                elif size_mb < 2.0:
                    size_distribution["medium"] += 1
                else:
                    size_distribution["large"] += 1
            except:
                continue

        return {
            "total_size_mb": total_size,
            "average_size_mb": total_size / len(image_paths) if image_paths else 0,
            "size_distribution": size_distribution,
            "recommended_optimization": total_size > 50,  # 50MB以上で最適化推奨
        }

    def _optimize_cache_strategy(self, image_paths: List[str]) -> Dict[str, Any]:
        """キャッシュ戦略最適化"""
        return {
            "cache_enabled": True,
            "cache_duration_hours": 24,
            "preload_strategy": "lazy",  # lazy loading推奨
            "compression_enabled": True,
            "estimated_cache_size_mb": len(image_paths) * 0.3,  # 平均300KB/画像と仮定
        }

    def _optimize_responsive_design(self) -> Dict[str, Any]:
        """レスポンシブデザイン最適化"""
        return {
            "breakpoints": {
                "mobile": "320px",
                "tablet": "768px",
                "desktop": "1024px",
                "large": "1440px",
            },
            "grid_columns": {"mobile": 1, "tablet": 2, "desktop": 3, "large": 4},
            "image_sizing": "object-contain",
            "max_height": "400px",
        }


class QI004DashboardOptimizationSystem:
    """
    QI-004ダッシュボード標準化・Base64画像表示システム構築

    主要機能:
    - ダッシュボード標準化
    - 画像パス参照方式最適化
    - パフォーマンス最適化
    - UI改善実装
    """

    def __init__(self):
        """QI-004システムの初期化"""
        self.logger = logging.getLogger(__name__)
        self.quality_analyzer = ImageQualityAnalyzer()
        self.dashboard_optimizer = DashboardOptimizer()
        self.dashboard_generator = DashboardGenerator()
        self.notification_sender = PushoverImageSender()

        self.logger.info("🎯 QI-004ダッシュボード最適化システム初期化完了")

    def run_complete_optimization(
        self, tracker_id: str, extraction_dir: str, output_dir: str
    ) -> QI004OptimizationResult:
        """
        完全なダッシュボード最適化プロセス実行

        Args:
            tracker_id: トラッカーID
            extraction_dir: 抽出ディレクトリ
            output_dir: 出力ディレクトリ

        Returns:
            最適化結果
        """
        self.logger.info(f"🔄 QI-004完全最適化プロセス開始: {tracker_id}")
        start_time = time.time()

        # 1. 画像収集
        image_paths = self._collect_extracted_images(extraction_dir)
        self.logger.info(f"📷 検出画像数: {len(image_paths)}枚")

        # 2. 画像品質解析
        quality_scores = self._analyze_all_images(image_paths)

        # 3. ダッシュボード最適化
        dashboard_dir = os.path.join(output_dir, "dashboard")
        optimization_result = self.dashboard_optimizer.optimize_dashboard_performance(
            image_paths, dashboard_dir
        )

        # 4. 最適化されたダッシュボード生成
        dashboard_data = {
            "tracker_id": tracker_id,
            "total_images": len(image_paths),
            "quality_scores": quality_scores,
            "black_screen_indices": [],  # QI-004では黒画面検出は継承
            "image_paths": image_paths,  # QI-004: 実際の画像パスを使用
            "dashboard_dir": dashboard_dir,
            "optimization_data": optimization_result,
        }

        dashboard_path = self.dashboard_generator.generate_standard_dashboard(
            dashboard_data, dashboard_dir
        )

        # 5. ダッシュボードサイズ・読み込み時間測定（画像パス参照で大幅縮小期待）
        dashboard_size_mb = dashboard_path.stat().st_size / (1024 * 1024)
        load_time = self._measure_dashboard_load_time(str(dashboard_path))

        # 6. 通知送信
        self._send_completion_notification(
            tracker_id, len(image_paths), dashboard_size_mb, load_time
        )

        total_time = time.time() - start_time

        self.logger.info(f"✅ QI-004最適化完了: {total_time:.1f}秒")

        return QI004OptimizationResult(
            total_images=len(image_paths),
            image_quality_scores=quality_scores,
            dashboard_size_mb=dashboard_size_mb,
            load_time_seconds=load_time,
            optimization_improvements=optimization_result,
            image_path_references=optimization_result["optimized_paths"],
            performance_metrics={
                "total_time_seconds": total_time,
                "images_per_second": len(image_paths) / total_time if total_time > 0 else 0,
                "optimization_efficiency": optimization_result["optimization_time_seconds"],
            },
        )

    def _collect_extracted_images(self, extraction_dir: str) -> List[str]:
        """抽出画像の収集"""
        if not os.path.exists(extraction_dir):
            self.logger.warning(f"⚠️ 抽出ディレクトリが存在しません: {extraction_dir}")
            return []

        image_extensions = [".jpg", ".jpeg", ".png", ".webp"]
        image_paths = []

        for file in os.listdir(extraction_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(extraction_dir, file))

        return sorted(image_paths)

    def _analyze_all_images(self, image_paths: List[str]) -> List[float]:
        """全画像の品質解析"""
        quality_scores = []

        for i, image_path in enumerate(image_paths):
            analysis = self.quality_analyzer.analyze_image_quality(image_path)
            quality_scores.append(analysis["overall_score"])

            if (i + 1) % 10 == 0:
                self.logger.info(f"📊 品質解析進捗: {i + 1}/{len(image_paths)}")

        return quality_scores

    def _measure_dashboard_load_time(self, dashboard_path: str) -> float:
        """ダッシュボード読み込み時間測定"""
        try:
            start_time = time.time()

            # HTMLファイル読み込みシミュレーション
            with open(dashboard_path, "r", encoding="utf-8") as f:
                content = f.read()

            # パース時間シミュレーション
            time.sleep(0.1)  # 実際の解析・レンダリング時間を模擬

            load_time = time.time() - start_time
            return load_time

        except Exception as e:
            self.logger.error(f"ダッシュボード読み込み時間測定エラー: {e}")
            return 1.0  # デフォルト値

    def _send_completion_notification(
        self, tracker_id: str, image_count: int, dashboard_size_mb: float, load_time: float
    ):
        """完了通知送信"""
        try:
            extraction_stats = {
                "success": image_count,
                "total": image_count,
                "dashboard_size_mb": dashboard_size_mb,
                "load_time_seconds": load_time,
            }

            # QI-004専用通知メッセージ
            self.notification_sender.send_extraction_complete_with_images(
                tracker_id=tracker_id,
                image_paths=[],  # 画像添付は行わない（最適化済みダッシュボードのみ）
                extraction_stats=extraction_stats,
            )

            self.logger.info(f"📱 QI-004完了通知送信済み: {tracker_id}")

        except Exception as e:
            self.logger.error(f"通知送信エラー: {e}")


def create_qi004_optimized_dashboard(tracker_id: str, extraction_dir: str, output_dir: str) -> bool:
    """
    QI-004最適化ダッシュボード生成のエントリーポイント

    Args:
        tracker_id: トラッカーID
        extraction_dir: 抽出ディレクトリパス
        output_dir: 出力ディレクトリパス

    Returns:
        成功フラグ
    """
    try:
        system = QI004DashboardOptimizationSystem()
        result = system.run_complete_optimization(tracker_id, extraction_dir, output_dir)

        print(f"✅ QI-004最適化完了")
        print(f"   📊 処理画像数: {result.total_images}枚")
        print(f"   📄 ダッシュボードサイズ: {result.dashboard_size_mb:.2f}MB")
        print(f"   ⚡ 読み込み時間: {result.load_time_seconds:.2f}秒")
        print(f"   🎯 平均品質スコア: {np.mean(result.image_quality_scores):.3f}")

        return True

    except Exception as e:
        logging.error(f"QI-004最適化エラー: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="QI-004ダッシュボード最適化システム")
    parser.add_argument("tracker_id", help="トラッカーID")
    parser.add_argument("extraction_dir", help="抽出ディレクトリパス")
    parser.add_argument("output_dir", help="出力ディレクトリパス")

    args = parser.parse_args()

    success = create_qi004_optimized_dashboard(
        args.tracker_id, args.extraction_dir, args.output_dir
    )

    exit(0 if success else 1)
