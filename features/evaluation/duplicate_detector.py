#!/usr/bin/env python3
"""
P1-B003: 画像重複判定機能
AIによる重複判定システム - ハッシュ値と視覚的類似度のハイブリッド判定
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from PIL import Image
from typing import Any, Dict, List, Optional, Set, Tuple

# 既存システム利用
try:
    from evaluation.content import ContentEvaluator

    CONTENT_EVALUATOR_AVAILABLE = True
except ImportError:
    CONTENT_EVALUATOR_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DuplicateInfo:
    """重複画像情報"""

    file_path: str
    file_hash: str
    file_size: int
    visual_hash: Optional[str] = None


@dataclass
class DuplicateGroup:
    """重複グループ"""

    group_id: str
    primary_image: str  # 代表画像（最高品質・最小ファイルサイズ）
    duplicate_images: List[str]
    similarity_scores: Dict[str, float]  # ペア間類似度
    hash_type: str  # "exact", "visual", "hybrid"
    confidence: float


@dataclass
class DuplicateDetectionResult:
    """重複検出結果"""

    total_images: int
    total_duplicates: int
    duplicate_groups: List[DuplicateGroup]
    processing_time: float
    hash_cache_hits: int
    visual_comparisons: int


class ImageDuplicateDetector:
    """
    P1-B003 メイン実装: 画像重複判定システム

    機能:
    - ファイルハッシュによる完全一致検出
    - 視覚的類似度による近似重複検出
    - ハイブリッド判定による高精度重複検出
    - 重複グループ可視化レポート生成
    """

    def __init__(
        self,
        visual_threshold: float = 0.85,
        enable_visual_detection: bool = True,
        cache_dir: Optional[Path] = None,
    ):
        """
        Args:
            visual_threshold: 視覚的類似度の閾値 (0.0-1.0)
            enable_visual_detection: 視覚的類似度検出の有効化
            cache_dir: ハッシュキャッシュディレクトリ
        """
        self.visual_threshold = visual_threshold
        self.enable_visual_detection = enable_visual_detection

        # キャッシュシステム
        self.cache_dir = cache_dir or Path.cwd() / ".duplicate_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.hash_cache_file = self.cache_dir / "image_hashes.json"

        # データ構造
        self.image_hashes: Dict[str, DuplicateInfo] = {}
        self.visual_cache: Dict[Tuple[str, str], float] = {}

        # 統計
        self.stats = {
            "hash_cache_hits": 0,
            "hash_calculations": 0,
            "visual_comparisons": 0,
            "exact_duplicates": 0,
            "visual_duplicates": 0,
        }

        # 視覚的類似度評価器
        if self.enable_visual_detection and CONTENT_EVALUATOR_AVAILABLE:
            try:
                self.content_evaluator = ContentEvaluator(
                    backbone="clip_ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu"
                )
                logger.info("視覚的類似度評価器初期化完了")
            except Exception as e:
                logger.warning(f"視覚的類似度評価器初期化失敗: {e}")
                self.content_evaluator = None
        else:
            self.content_evaluator = None

        self._load_hash_cache()

    def _calculate_file_hash(self, file_path: Path) -> str:
        """
        ファイルハッシュ計算 (SHA256)
        大ファイル対応チャンク読み込み
        """
        try:
            hasher = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            self.stats["hash_calculations"] += 1
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"ハッシュ計算エラー {file_path}: {e}")
            return ""

    def _calculate_visual_hash(self, image_path: Path) -> Optional[str]:
        """
        視覚的ハッシュ計算 (pHash風)
        低解像度変換 + DCT によるロバストハッシュ
        """
        try:
            # 画像読み込み・グレースケール変換
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None

            # 32x32にリサイズ（高速化）
            img = cv2.resize(img, (32, 32))

            # DCT変換
            img_float = np.float32(img)
            dct = cv2.dct(img_float)

            # 左上8x8の低周波成分のみ使用
            dct_low = dct[0:8, 0:8]

            # 中央値でバイナリ化
            median = np.median(dct_low)
            hash_bits = dct_low > median

            # ビット列をハッシュ文字列に変換
            hash_str = "".join(["1" if bit else "0" for bit in hash_bits.flatten()])
            return hash_str

        except Exception as e:
            logger.error(f"視覚的ハッシュ計算エラー {image_path}: {e}")
            return None

    def _calculate_visual_similarity(self, img1_path: Path, img2_path: Path) -> float:
        """
        AIベース視覚的類似度計算 (CLIP使用)

        Returns:
            類似度スコア (0.0-1.0)
        """
        if not self.content_evaluator:
            return 0.0

        cache_key = (str(img1_path), str(img2_path))
        reverse_key = (str(img2_path), str(img1_path))

        # キャッシュ確認
        if cache_key in self.visual_cache:
            return self.visual_cache[cache_key]
        if reverse_key in self.visual_cache:
            return self.visual_cache[reverse_key]

        try:
            # 画像読み込み
            img1 = np.array(Image.open(img1_path).convert("RGB"))
            img2 = np.array(Image.open(img2_path).convert("RGB"))

            # CLIP類似度計算
            similarity = self.content_evaluator.evaluate_crop_similarity(img1, img2)

            # キャッシュ保存
            self.visual_cache[cache_key] = similarity
            self.stats["visual_comparisons"] += 1

            return similarity

        except Exception as e:
            logger.error(f"視覚的類似度計算エラー {img1_path} vs {img2_path}: {e}")
            return 0.0

    def _load_hash_cache(self):
        """ハッシュキャッシュ読み込み"""
        if not self.hash_cache_file.exists():
            return

        try:
            with open(self.hash_cache_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
                self.image_hashes = {
                    path: DuplicateInfo(**info) for path, info in cache_data.items()
                }
            logger.info(f"ハッシュキャッシュ読み込み: {len(self.image_hashes)}件")
        except Exception as e:
            logger.error(f"ハッシュキャッシュ読み込みエラー: {e}")

    def _save_hash_cache(self):
        """ハッシュキャッシュ保存"""
        try:
            cache_data = {path: asdict(info) for path, info in self.image_hashes.items()}
            with open(self.hash_cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"ハッシュキャッシュ保存エラー: {e}")

    def _get_image_info(self, image_path: Path) -> DuplicateInfo:
        """画像情報取得（キャッシュ活用）"""
        path_str = str(image_path)

        # キャッシュ確認
        if path_str in self.image_hashes:
            cached_info = self.image_hashes[path_str]
            # ファイル更新時刻チェック（簡略化）
            try:
                current_mtime = image_path.stat().st_mtime
                # キャッシュされた情報のファイルサイズが同じなら有効とみなす
                if cached_info.file_size == image_path.stat().st_size:
                    self.stats["hash_cache_hits"] += 1
                    return cached_info
            except (OSError, AttributeError):
                # ファイルアクセスエラーの場合は再計算
                pass

        # 新規計算
        file_hash = self._calculate_file_hash(image_path)
        visual_hash = (
            self._calculate_visual_hash(image_path) if self.enable_visual_detection else None
        )

        info = DuplicateInfo(
            file_path=path_str,
            file_hash=file_hash,
            file_size=image_path.stat().st_size,
            visual_hash=visual_hash,
        )

        # キャッシュ更新
        self.image_hashes[path_str] = info
        return info

    def detect_duplicates(
        self,
        image_dir: Path,
        image_extensions: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"},
    ) -> DuplicateDetectionResult:
        """
        重複画像検出メイン処理

        Args:
            image_dir: 検索対象ディレクトリ
            image_extensions: 対象画像拡張子

        Returns:
            重複検出結果
        """
        start_time = time.time()

        logger.info(f"重複検出開始: {image_dir}")

        # 1. 画像ファイル収集
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(image_dir.glob(f"**/*{ext}")))
            image_files.extend(list(image_dir.glob(f"**/*{ext.upper()}")))

        logger.info(f"検出対象画像: {len(image_files)}件")

        # 2. ハッシュ計算
        image_infos = []
        for img_path in image_files:
            try:
                info = self._get_image_info(img_path)
                if info.file_hash:  # ハッシュ計算成功のみ
                    image_infos.append(info)
            except Exception as e:
                logger.error(f"画像情報取得エラー {img_path}: {e}")

        # 3. 完全一致重複検出（ファイルハッシュ）
        hash_groups = self._group_by_hash(image_infos)
        duplicate_groups = []

        for file_hash, group_images in hash_groups.items():
            if len(group_images) > 1:
                # 完全一致グループ
                primary = min(group_images, key=lambda x: (x.file_size, x.file_path))
                duplicates = [img.file_path for img in group_images if img != primary]

                group = DuplicateGroup(
                    group_id=f"exact_{file_hash[:8]}",
                    primary_image=primary.file_path,
                    duplicate_images=duplicates,
                    similarity_scores={},
                    hash_type="exact",
                    confidence=1.0,
                )
                duplicate_groups.append(group)
                self.stats["exact_duplicates"] += len(duplicates)

        # 4. 視覚的類似重複検出
        if self.enable_visual_detection and self.content_evaluator:
            visual_groups = self._detect_visual_duplicates(image_infos, hash_groups)
            duplicate_groups.extend(visual_groups)

        # 5. キャッシュ保存
        self._save_hash_cache()

        processing_time = time.time() - start_time

        result = DuplicateDetectionResult(
            total_images=len(image_files),
            total_duplicates=sum(len(g.duplicate_images) for g in duplicate_groups),
            duplicate_groups=duplicate_groups,
            processing_time=processing_time,
            hash_cache_hits=self.stats["hash_cache_hits"],
            visual_comparisons=self.stats["visual_comparisons"],
        )

        logger.info(f"重複検出完了: {result.total_duplicates}件の重複を{len(duplicate_groups)}グループで検出")
        return result

    def _group_by_hash(self, image_infos: List[DuplicateInfo]) -> Dict[str, List[DuplicateInfo]]:
        """ファイルハッシュによるグループ化"""
        hash_groups = {}
        for info in image_infos:
            if info.file_hash not in hash_groups:
                hash_groups[info.file_hash] = []
            hash_groups[info.file_hash].append(info)
        return hash_groups

    def _detect_visual_duplicates(
        self, image_infos: List[DuplicateInfo], hash_groups: Dict[str, List[DuplicateInfo]]
    ) -> List[DuplicateGroup]:
        """視覚的類似重複検出（完全一致以外）"""
        visual_groups = []

        # 完全一致でない画像のみ対象
        unique_images = [
            info for hash_val, group in hash_groups.items() if len(group) == 1 for info in group
        ]

        logger.info(f"視覚的類似度検査対象: {len(unique_images)}件")

        # 全ペア比較（効率化要検討）
        processed_pairs = set()

        for i, img1 in enumerate(unique_images):
            for j, img2 in enumerate(unique_images[i + 1 :], i + 1):
                pair_key = tuple(sorted([img1.file_path, img2.file_path]))
                if pair_key in processed_pairs:
                    continue

                processed_pairs.add(pair_key)

                # 視覚的類似度計算
                similarity = self._calculate_visual_similarity(
                    Path(img1.file_path), Path(img2.file_path)
                )

                if similarity >= self.visual_threshold:
                    # 類似グループ作成
                    primary = img1 if img1.file_size <= img2.file_size else img2
                    duplicate = img2 if primary == img1 else img1

                    group = DuplicateGroup(
                        group_id=f"visual_{i}_{j}",
                        primary_image=primary.file_path,
                        duplicate_images=[duplicate.file_path],
                        similarity_scores={
                            f"{primary.file_path}:{duplicate.file_path}": similarity
                        },
                        hash_type="visual",
                        confidence=similarity,
                    )
                    visual_groups.append(group)
                    self.stats["visual_duplicates"] += 1

                    logger.debug(
                        f"視覚的重複検出: {similarity:.3f} - {Path(img1.file_path).name} vs {Path(img2.file_path).name}"
                    )

        return visual_groups

    def generate_report(self, result: DuplicateDetectionResult, output_path: Path):
        """重複検出レポート生成"""
        try:
            report_data = {
                "summary": {
                    "total_images": result.total_images,
                    "total_duplicates": result.total_duplicates,
                    "duplicate_groups": len(result.duplicate_groups),
                    "processing_time": f"{result.processing_time:.2f}s",
                    "hash_cache_hits": result.hash_cache_hits,
                    "visual_comparisons": result.visual_comparisons,
                },
                "groups": [],
            }

            for group in result.duplicate_groups:
                group_data = {
                    "group_id": group.group_id,
                    "type": group.hash_type,
                    "confidence": group.confidence,
                    "primary_image": group.primary_image,
                    "duplicate_count": len(group.duplicate_images),
                    "duplicates": group.duplicate_images,
                    "similarity_scores": group.similarity_scores,
                }
                report_data["groups"].append(group_data)

            # JSON保存
            json_path = output_path / "duplicate_report.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)

            # Markdown保存
            md_path = output_path / "duplicate_report.md"
            self._generate_markdown_report(result, md_path)

            logger.info(f"重複検出レポート生成完了: {json_path}")
            return json_path

        except Exception as e:
            logger.error(f"レポート生成エラー: {e}")
            return None

    def _generate_markdown_report(self, result: DuplicateDetectionResult, output_path: Path):
        """Markdownレポート生成"""
        lines = [
            "# 画像重複検出レポート",
            f"**生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## サマリー",
            f"- **総画像数**: {result.total_images}件",
            f"- **重複画像数**: {result.total_duplicates}件",
            f"- **重複グループ数**: {len(result.duplicate_groups)}グループ",
            f"- **処理時間**: {result.processing_time:.2f}秒",
            f"- **キャッシュヒット数**: {result.hash_cache_hits}件",
            f"- **視覚的比較数**: {result.visual_comparisons}件",
            "",
            "## 重複グループ詳細",
            "",
        ]

        for i, group in enumerate(result.duplicate_groups, 1):
            lines.extend(
                [
                    f"### グループ {i}: {group.group_id}",
                    f"- **判定種別**: {group.hash_type}",
                    f"- **信頼度**: {group.confidence:.3f}",
                    f"- **代表画像**: `{Path(group.primary_image).name}`",
                    f"- **重複数**: {len(group.duplicate_images)}件",
                    "",
                    "**重複画像一覧**:",
                ]
            )

            for dup_path in group.duplicate_images:
                lines.append(f"- `{Path(dup_path).name}`")

            if group.similarity_scores:
                lines.append("\n**類似度スコア**:")
                for pair, score in group.similarity_scores.items():
                    lines.append(f"- {pair}: {score:.3f}")

            lines.append("")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def create_duplicate_detector(
    visual_threshold: float = 0.85,
    enable_visual_detection: bool = True,
    cache_dir: Optional[Path] = None,
) -> ImageDuplicateDetector:
    """重複検出器ファクトリー関数"""
    return ImageDuplicateDetector(
        visual_threshold=visual_threshold,
        enable_visual_detection=enable_visual_detection,
        cache_dir=cache_dir,
    )


if __name__ == "__main__":
    # CLI実行テスト
    import sys

    if len(sys.argv) < 2:
        print("使用法: python duplicate_detector.py <image_directory>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    if not input_dir.exists():
        print(f"エラー: ディレクトリが存在しません: {input_dir}")
        sys.exit(1)

    # 基本テスト実行
    detector = create_duplicate_detector()
    result = detector.detect_duplicates(input_dir)

    output_path = Path.cwd() / "duplicate_detection_results"
    output_path.mkdir(exist_ok=True)

    detector.generate_report(result, output_path)
    print(f"重複検出完了: {result.total_duplicates}件の重複を検出")
