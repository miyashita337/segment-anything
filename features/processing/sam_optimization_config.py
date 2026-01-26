#!/usr/bin/env python3
"""
P1-020: SAM推論最適化実装
P1-016フィードバックループで特定されたボトルネック（85.5%、6分/画像）を解消
処理時間50-70%短縮を実現する最適化設定システム
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class SAMOptimizationProfile:
    """SAM最適化プロファイル"""

    name: str
    points_per_side: int
    crop_n_layers: int
    pred_iou_thresh: float
    stability_score_thresh: float
    box_nms_thresh: float
    crop_n_points_downscale_factor: int
    min_mask_region_area: int
    description: str
    expected_speedup: float  # 期待される高速化倍率


class SAMOptimizationConfig:
    """
    SAM推論最適化設定管理クラス
    P1-016で特定されたボトルネックを解消する最適化設定を提供

    QCA-001: 作者別パラメータ適応システム対応
    各作者の絵柄特性に合わせたSAMプロファイルを提供
    """

    # P1-016フィードバックループが推奨した最適化設定
    OPTIMIZATION_PROFILES = {
        "original": SAMOptimizationProfile(
            name="original",
            points_per_side=32,
            crop_n_layers=1,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.95,
            box_nms_thresh=0.7,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=100,
            description="オリジナル設定（品質重視）",
            expected_speedup=1.0,
        ),
        "p1_020_optimized": SAMOptimizationProfile(
            name="p1_020_optimized",
            points_per_side=16,  # 32 → 16 (50%削減)
            crop_n_layers=0,  # 1 → 0 (クロップ無効化)
            pred_iou_thresh=0.8,  # 0.86 → 0.8 (緩和)
            stability_score_thresh=0.88,  # 0.95 → 0.88 (緩和)
            box_nms_thresh=0.7,  # 維持
            crop_n_points_downscale_factor=2,  # 1 → 2 (ダウンスケール強化)
            min_mask_region_area=200,  # 100 → 200 (小領域除外強化)
            description="P1-020最適化（6分→2.5分目標）",
            expected_speedup=2.4,  # 58%短縮
        ),
        "p1_020_aggressive": SAMOptimizationProfile(
            name="p1_020_aggressive",
            points_per_side=12,  # さらに削減
            crop_n_layers=0,  # クロップ無効
            pred_iou_thresh=0.75,  # さらに緩和
            stability_score_thresh=0.85,  # さらに緩和
            box_nms_thresh=0.65,  # NMS緩和
            crop_n_points_downscale_factor=2,
            min_mask_region_area=300,  # 小領域除外強化
            description="P1-020アグレッシブ（3倍高速化目標）",
            expected_speedup=3.0,  # 67%短縮
        ),
        "p1_020_balanced": SAMOptimizationProfile(
            name="p1_020_balanced",
            points_per_side=20,  # 32 → 20 (品質とのバランス)
            crop_n_layers=0,  # クロップ無効
            pred_iou_thresh=0.82,  # 0.86 → 0.82 (軽度緩和)
            stability_score_thresh=0.90,  # 0.95 → 0.90 (軽度緩和)
            box_nms_thresh=0.7,  # 維持
            crop_n_points_downscale_factor=1,
            min_mask_region_area=150,  # 軽度強化
            description="P1-020バランス（品質維持・2倍高速化）",
            expected_speedup=2.0,  # 50%短縮
        ),
        # QCA-001: 作者別SAMプロファイル拡張
        "character_focused": SAMOptimizationProfile(
            name="character_focused",
            points_per_side=18,  # yado作者用: キャラクター重視
            crop_n_layers=0,
            pred_iou_thresh=0.8,
            stability_score_thresh=0.88,
            box_nms_thresh=0.7,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=120,
            description="QCA-001: yado作者用キャラクター重視プロファイル",
            expected_speedup=2.2,
        ),
        "precision_focused": SAMOptimizationProfile(
            name="precision_focused",
            points_per_side=24,  # aichi作者用: 細密描写特化
            crop_n_layers=1,  # 高品質のためクロップ有効
            pred_iou_thresh=0.85,  # 高精度設定
            stability_score_thresh=0.92,
            box_nms_thresh=0.75,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=100,  # 細かい部分も検出
            description="QCA-001: aichi作者用細密描写特化プロファイル",
            expected_speedup=1.8,  # 品質優先のためやや遅い
        ),
        "speed_optimized": SAMOptimizationProfile(
            name="speed_optimized",
            points_per_side=14,  # zundamon作者用: 高速処理重視
            crop_n_layers=0,
            pred_iou_thresh=0.78,
            stability_score_thresh=0.85,
            box_nms_thresh=0.65,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=200,
            description="QCA-001: zundamon作者用高速処理プロファイル",
            expected_speedup=2.8,  # 最高速度
        ),
    }

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.current_profile = "p1_020_optimized"  # デフォルトは最適化版
        self.performance_history = []

    def get_sam_config(
        self, profile_name: Optional[str] = None, author_params: Optional[dict] = None
    ) -> Dict[str, Any]:
        """指定プロファイルのSAM設定を取得

        Args:
            profile_name: 使用するプロファイル名
            author_params: QCA-001 作者別パラメータ（優先適用）
        """
        if profile_name is None:
            profile_name = self.current_profile

        # QCA-001: 作者別パラメータからSAMプロファイルを適用
        if author_params and "sam_profile" in author_params:
            author_sam_profile = author_params["sam_profile"]
            if author_sam_profile in self.OPTIMIZATION_PROFILES:
                profile_name = author_sam_profile
                self.logger.info(f"🎯 QCA-001: 作者別SAMプロファイル適用 = {author_sam_profile}")

        if profile_name not in self.OPTIMIZATION_PROFILES:
            self.logger.warning(f"不明なプロファイル: {profile_name}, デフォルトを使用")
            profile_name = "p1_020_optimized"

        profile = self.OPTIMIZATION_PROFILES[profile_name]

        config = {
            "points_per_side": profile.points_per_side,
            "pred_iou_thresh": profile.pred_iou_thresh,
            "stability_score_thresh": profile.stability_score_thresh,
            "box_nms_thresh": profile.box_nms_thresh,
            "crop_n_layers": profile.crop_n_layers,
            "crop_n_points_downscale_factor": profile.crop_n_points_downscale_factor,
            "min_mask_region_area": profile.min_mask_region_area,
        }

        self.logger.info(f"🚀 SAM設定適用: {profile.description}")
        self.logger.info(f"📊 期待高速化: {profile.expected_speedup:.1f}倍")

        return config

    def set_profile(self, profile_name: str) -> bool:
        """使用プロファイルを設定"""
        if profile_name not in self.OPTIMIZATION_PROFILES:
            self.logger.error(f"不明なプロファイル: {profile_name}")
            return False

        self.current_profile = profile_name
        profile = self.OPTIMIZATION_PROFILES[profile_name]
        self.logger.info(f"✅ プロファイル変更: {profile.description}")
        return True

    def compare_profiles(self) -> str:
        """プロファイル比較表を生成"""
        comparison = "🔍 SAM最適化プロファイル比較\n"
        comparison += "=" * 80 + "\n"
        comparison += f"{'プロファイル':<20} {'points':<8} {'crop':<6} {'iou':<6} {'期待高速化':<10} {'説明'}\n"
        comparison += "-" * 80 + "\n"

        for name, profile in self.OPTIMIZATION_PROFILES.items():
            comparison += f"{profile.name:<20} {profile.points_per_side:<8} "
            comparison += f"{profile.crop_n_layers:<6} {profile.pred_iou_thresh:<6.2f} "
            comparison += f"{profile.expected_speedup:<10.1f}x {profile.description}\n"

        return comparison

    def record_performance(
        self, profile_name: str, processing_time_seconds: float, quality_score: float, success: bool
    ) -> None:
        """パフォーマンス記録"""
        record = {
            "timestamp": time.time(),
            "profile": profile_name,
            "processing_time": processing_time_seconds,
            "quality_score": quality_score,
            "success": success,
            "speedup_vs_original": None,
        }

        # オリジナルとの比較計算
        original_times = [
            r["processing_time"]
            for r in self.performance_history
            if r["profile"] == "original" and r["success"]
        ]
        if original_times:
            avg_original_time = sum(original_times) / len(original_times)
            record["speedup_vs_original"] = avg_original_time / processing_time_seconds

        self.performance_history.append(record)

        # ログ出力
        self.logger.info(f"📊 パフォーマンス記録: {profile_name}")
        self.logger.info(f"   処理時間: {processing_time_seconds:.1f}秒")
        self.logger.info(f"   品質スコア: {quality_score:.3f}")
        if record["speedup_vs_original"]:
            self.logger.info(f"   高速化: {record['speedup_vs_original']:.1f}倍")

    def get_performance_summary(self) -> Dict[str, Any]:
        """パフォーマンスサマリー取得"""
        if not self.performance_history:
            return {"status": "記録なし"}

        summary = {}
        for profile_name in self.OPTIMIZATION_PROFILES.keys():
            profile_records = [r for r in self.performance_history if r["profile"] == profile_name]
            if not profile_records:
                continue

            successful_records = [r for r in profile_records if r["success"]]
            if not successful_records:
                continue

            avg_time = sum(r["processing_time"] for r in successful_records) / len(
                successful_records
            )
            avg_quality = sum(r["quality_score"] for r in successful_records) / len(
                successful_records
            )
            success_rate = len(successful_records) / len(profile_records) * 100

            summary[profile_name] = {
                "avg_processing_time": avg_time,
                "avg_quality_score": avg_quality,
                "success_rate": success_rate,
                "sample_count": len(successful_records),
            }

        return summary

    def recommend_profile(self, target_speedup: float = 2.0, min_quality_score: float = 2.0) -> str:
        """条件に基づく推奨プロファイル"""

        # パフォーマンス履歴がある場合は実績ベース
        summary = self.get_performance_summary()
        if summary and len(summary) > 1:
            best_profile = None
            best_score = 0

            for profile_name, stats in summary.items():
                # 品質条件を満たし、目標高速化に近いプロファイルを選択
                if stats["avg_quality_score"] >= min_quality_score:
                    profile = self.OPTIMIZATION_PROFILES[profile_name]
                    speedup_diff = abs(profile.expected_speedup - target_speedup)
                    score = stats["success_rate"] / (1 + speedup_diff)  # 成功率と高速化の兼ね合い

                    if score > best_score:
                        best_score = score
                        best_profile = profile_name

            if best_profile:
                return best_profile

        # 実績がない場合は期待値ベース推奨
        if target_speedup <= 1.5:
            return "original"
        elif target_speedup <= 2.0:
            return "p1_020_balanced"
        elif target_speedup <= 2.5:
            return "p1_020_optimized"
        else:
            return "p1_020_aggressive"

    def save_performance_history(self, file_path: str) -> None:
        """パフォーマンス履歴保存"""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.performance_history, f, indent=2, ensure_ascii=False)
            self.logger.info(f"📊 パフォーマンス履歴保存: {file_path}")
        except Exception as e:
            self.logger.error(f"履歴保存エラー: {str(e)}")

    def load_performance_history(self, file_path: str) -> bool:
        """パフォーマンス履歴読み込み"""
        try:
            if Path(file_path).exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    self.performance_history = json.load(f)
                self.logger.info(f"📊 パフォーマンス履歴読み込み: {len(self.performance_history)} 件")
                return True
        except Exception as e:
            self.logger.error(f"履歴読み込みエラー: {str(e)}")
        return False


# P1-016フィードバックループシステム統合用ヘルパー
def create_optimized_sam_generator(
    sam_model,
    optimization_config: SAMOptimizationConfig,
    profile_name: Optional[str] = None,
    author_params: Optional[dict] = None,
):
    """最適化されたSAMマスクジェネレーター作成

    Args:
        sam_model: SAMモデルインスタンス
        optimization_config: SAM最適化設定
        profile_name: 使用するプロファイル名
        author_params: QCA-001 作者別パラメータ
    """
    from segment_anything import SamAutomaticMaskGenerator

    config = optimization_config.get_sam_config(profile_name, author_params)

    return SamAutomaticMaskGenerator(model=sam_model, **config)


if __name__ == "__main__":
    # 使用例・テスト
    print("=== P1-020 SAM推論最適化システム ===")

    optimizer = SAMOptimizationConfig()

    # プロファイル比較表示
    print(optimizer.compare_profiles())

    # 各プロファイル設定表示
    for profile_name in ["original", "p1_020_optimized", "p1_020_aggressive"]:
        print(f"\n📋 {profile_name} 設定:")
        config = optimizer.get_sam_config(profile_name)
        for key, value in config.items():
            print(f"  {key}: {value}")

    # 推奨プロファイル
    print(f"\n🎯 推奨プロファイル（2倍高速化）: {optimizer.recommend_profile(2.0)}")
    print(f"🎯 推奨プロファイル（3倍高速化）: {optimizer.recommend_profile(3.0)}")
