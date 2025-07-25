#!/usr/bin/env python3
"""
アニメ画像前処理システム
顔検出率向上のための特化前処理パイプライン
"""

import numpy as np
import cv2

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class AnimeImagePreprocessor:
    """アニメ画像前処理システム"""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.AnimeImagePreprocessor")

        # CLAHE（適応的ヒストグラム平均化）初期化
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # アニメ画像用に調整

    def enhance_for_face_detection(
        self, image: np.ndarray, lightweight_mode: bool = False
    ) -> np.ndarray:
        """顔検出用総合前処理"""
        try:
            if lightweight_mode:
                # 軽量モード: 必要最小限の処理のみ（1-2秒目標）
                return self._lightweight_preprocessing(image)
            else:
                # 高品質モード: 全5段階処理（従来版）
                return self._full_preprocessing(image)

        except Exception as e:
            self.logger.warning(f"前処理エラー、元画像を返却: {e}")
            return image

    def _lightweight_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """軽量前処理（GPT-4O提案版: 1-2秒目標）"""
        try:
            # 1. 高速ノイズ除去（fastNlMeansの簡易版）
            denoised = self._fast_denoise(image)

            # 2. 適応的ヒストグラム平均化（CLAHE）
            clahe_enhanced = self._apply_clahe(denoised)

            self.logger.debug("軽量前処理完了")
            return clahe_enhanced

        except Exception as e:
            self.logger.warning(f"軽量前処理エラー: {e}")
            return image

    def _full_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """高品質前処理（従来の5段階処理）"""
        # 1. 基本的なノイズ除去
        denoised = self._denoise_image(image)

        # 2. コントラスト強化
        contrast_enhanced = self._enhance_contrast(denoised)

        # 3. ヒストグラム平均化
        histogram_equalized = self._histogram_equalization(contrast_enhanced)

        # 4. 適応的ヒストグラム平均化（CLAHE）
        clahe_enhanced = self._apply_clahe(histogram_equalized)

        # 5. エッジ保持平滑化
        edge_preserved = self._edge_preserving_smoothing(clahe_enhanced)

        self.logger.debug("高品質前処理完了")
        return edge_preserved

    def _fast_denoise(self, image: np.ndarray) -> np.ndarray:
        """高速ノイズ除去（軽量版）"""
        if len(image.shape) == 3:
            # カラー画像 - より軽量なパラメータ
            denoised = cv2.fastNlMeansDenoisingColored(
                image, None, h=5, hColor=5, templateWindowSize=5, searchWindowSize=15
            )
        else:
            # グレースケール画像
            denoised = cv2.fastNlMeansDenoising(
                image, None, h=5, templateWindowSize=5, searchWindowSize=15
            )

        return denoised

    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """ノイズ除去"""
        # Non-local Means Denoisingを使用（アニメ画像に適している）
        if len(image.shape) == 3:
            # カラー画像
            denoised = cv2.fastNlMeansDenoisingColored(
                image, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21
            )
        else:
            # グレースケール画像
            denoised = cv2.fastNlMeansDenoising(
                image, None, h=10, templateWindowSize=7, searchWindowSize=21
            )

        return denoised

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """コントラスト強化"""
        # LAB色空間でL（明度）チャンネルのみを強化
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)

            # L チャンネルのコントラスト強化
            alpha = 1.2  # コントラスト強化係数
            beta = 10  # 明度調整
            l_enhanced = cv2.convertScaleAbs(l_channel, alpha=alpha, beta=beta)

            # LAB色空間を再構築
            lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
            enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        else:
            # グレースケールの場合
            alpha = 1.2
            beta = 10
            enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        return enhanced

    def _histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """ヒストグラム平均化"""
        if len(image.shape) == 3:
            # カラー画像の場合、YUV色空間でY（輝度）チャンネルを処理
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            # グレースケールの場合
            equalized = cv2.equalizeHist(image)

        return equalized

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """適応的ヒストグラム平均化（CLAHE）適用"""
        if len(image.shape) == 3:
            # カラー画像の場合、LAB色空間でLチャンネルを処理
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)

            # CLAHEをLチャンネルに適用
            l_clahe = self.clahe.apply(l_channel)

            # LAB色空間を再構築
            lab_clahe = cv2.merge([l_clahe, a_channel, b_channel])
            clahe_result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        else:
            # グレースケールの場合
            clahe_result = self.clahe.apply(image)

        return clahe_result

    def _edge_preserving_smoothing(self, image: np.ndarray) -> np.ndarray:
        """エッジ保持平滑化"""
        # アニメ画像の特徴を保持しながら平滑化
        smoothed = cv2.edgePreservingFilter(
            image,
            flags=cv2.RECURS_FILTER,  # 再帰フィルタ使用
            sigma_s=50,  # ネイバーフッドサイズ
            sigma_r=0.4,  # 異なる色の平均化具合
        )

        return smoothed

    def create_multi_scale_versions(
        self, image: np.ndarray, lightweight_mode: bool = False
    ) -> list:
        """マルチスケール版画像生成"""
        if lightweight_mode:
            # 軽量モード: 3スケールのみ（GPT-4O提案）
            scales = [0.75, 1.0, 1.25]
        else:
            # 高品質モード: 5スケール（従来版）
            scales = [0.5, 0.75, 1.0, 1.25, 1.5]

        multi_scale_images = []
        height, width = image.shape[:2]

        for scale in scales:
            new_width = int(width * scale)
            new_height = int(height * scale)

            if scale == 1.0:
                scaled_image = image.copy()
            else:
                scaled_image = cv2.resize(
                    image,
                    (new_width, new_height),
                    interpolation=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA,
                )

            multi_scale_images.append(
                {"scale": scale, "image": scaled_image, "size": (new_width, new_height)}
            )

        return multi_scale_images

    def detect_optimal_brightness(self, image: np.ndarray) -> Tuple[float, bool]:
        """最適明度検出と調整要否判定"""
        # グレースケール変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 明度統計
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)

        # 調整要否判定
        needs_adjustment = False

        if mean_brightness < 80:  # 暗すぎる
            needs_adjustment = True
            self.logger.debug(f"画像が暗すぎます: 平均明度 {mean_brightness:.1f}")
        elif mean_brightness > 200:  # 明るすぎる
            needs_adjustment = True
            self.logger.debug(f"画像が明るすぎます: 平均明度 {mean_brightness:.1f}")
        elif std_brightness < 30:  # コントラストが低い
            needs_adjustment = True
            self.logger.debug(f"コントラストが低すぎます: 標準偏差 {std_brightness:.1f}")

        return mean_brightness, needs_adjustment

    def adaptive_brightness_adjustment(self, image: np.ndarray) -> np.ndarray:
        """適応的明度調整"""
        mean_brightness, needs_adjustment = self.detect_optimal_brightness(image)

        if not needs_adjustment:
            return image

        # 目標明度
        target_brightness = 128

        # 調整係数計算
        if mean_brightness < 80:
            # 暗すぎる場合
            beta = target_brightness - mean_brightness
            alpha = 1.1
        elif mean_brightness > 200:
            # 明るすぎる場合
            beta = target_brightness - mean_brightness
            alpha = 0.9
        else:
            # コントラストが低い場合
            beta = 0
            alpha = 1.3

        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        self.logger.debug(f"明度調整: alpha={alpha:.2f}, beta={beta:.1f}")
        return adjusted


def main():
    """テスト実行関数"""
    import argparse

    parser = argparse.ArgumentParser(description="アニメ画像前処理テスト")
    parser.add_argument("--image", "-i", required=True, help="テスト画像パス")
    parser.add_argument("--output", "-o", help="出力画像パス")

    args = parser.parse_args()

    # 画像読み込み
    image = cv2.imread(args.image)
    if image is None:
        print(f"画像読み込み失敗: {args.image}")
        return 1

    print(f"🎨 アニメ画像前処理テスト: {args.image}")
    print("=" * 60)

    # 前処理実行
    preprocessor = AnimeImagePreprocessor()

    # 元画像の統計
    original_brightness, needs_adjustment = preprocessor.detect_optimal_brightness(image)
    print(f"元画像統計:")
    print(f"  平均明度: {original_brightness:.1f}")
    print(f"  調整必要: {'Yes' if needs_adjustment else 'No'}")

    # 前処理実行
    enhanced_image = preprocessor.enhance_for_face_detection(image)

    # 処理後の統計
    enhanced_brightness, _ = preprocessor.detect_optimal_brightness(enhanced_image)
    print(f"\n処理後統計:")
    print(f"  平均明度: {enhanced_brightness:.1f}")
    print(f"  明度改善: {enhanced_brightness - original_brightness:+.1f}")

    # マルチスケール版生成
    multi_scale_versions = preprocessor.create_multi_scale_versions(enhanced_image)
    print(f"\nマルチスケール版: {len(multi_scale_versions)}種類生成")
    for version in multi_scale_versions:
        print(f"  スケール {version['scale']}: {version['size']}")

    # 出力保存
    if args.output:
        cv2.imwrite(args.output, enhanced_image)
        print(f"\n💾 前処理結果保存: {args.output}")

    print("✅ アニメ画像前処理テスト完了")
    return 0


if __name__ == "__main__":
    exit(main())
