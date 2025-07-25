#!/usr/bin/env python3
"""
Phase 1 データ拡張システム
疑似ラベル生成 + 人手修正による3-5倍データ拡張
"""

import numpy as np
import cv2

import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
from typing import Any, Dict, List, Optional, Tuple

# albumentationsインポート（オプション）
try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("WARNING: albumentations not available, using basic augmentation")

logger = logging.getLogger(__name__)


@dataclass
class PseudoLabel:
    """疑似ラベル"""
    image_id: str
    source_image_path: str
    augmented_image_path: str
    original_bbox: Tuple[int, int, int, int]
    transformed_bbox: Tuple[int, int, int, int]
    confidence_score: float
    augmentation_type: str
    requires_manual_review: bool
    notes: str


@dataclass
class DataExpansionResult:
    """データ拡張結果"""
    original_count: int
    generated_count: int
    total_count: int
    expansion_ratio: float
    pseudo_labels: List[PseudoLabel]
    augmentation_stats: Dict[str, int]
    quality_distribution: Dict[str, int]


class DataExpansionSystem:
    """データ拡張システム"""
    
    def __init__(self, project_root: Path):
        """
        初期化
        
        Args:
            project_root: プロジェクトルートディレクトリ
        """
        self.project_root = project_root
        # セキュリティ原則: プロジェクトルート直下への画像出力禁止
        self.output_dir = Path("/mnt/c/AItools/lora/train/yado/expanded")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 拡張画像・ラベル用ディレクトリ
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.review_dir = self.output_dir / "manual_review"
        
        for dir_path in [self.images_dir, self.labels_dir, self.review_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # 元データ読み込み
        self.labels_file = project_root / "extracted_labels.json"
        self.original_labels = self.load_original_labels()
        
        # 拡張設定
        self.target_expansion_ratio = 4.0  # 4倍拡張
        self.manual_review_threshold = 0.6  # 信頼度0.6以下は手動レビュー
        
        # データ拡張パイプライン定義
        self.augmentation_pipeline = self.setup_augmentation_pipeline()
        
    def load_original_labels(self) -> Dict[str, Any]:
        """元ラベルデータ読み込み"""
        try:
            if not self.labels_file.exists():
                logger.warning(f"ラベルファイルが見つかりません: {self.labels_file}")
                return {}
            
            with open(self.labels_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # リスト形式の場合は辞書形式に変換
            if isinstance(data, list):
                converted_data = {}
                for item in data:
                    filename = item.get('filename', '')
                    if filename:
                        # ファイル名から拡張子を除去してIDとする
                        image_id = filename.rsplit('.', 1)[0]
                        # 最初の赤枠座標を使用
                        if item.get('red_boxes') and len(item['red_boxes']) > 0:
                            first_box = item['red_boxes'][0]
                            bbox_data = first_box.get('bbox', {})
                            converted_data[image_id] = {
                                'red_box_coords': [
                                    bbox_data.get('x', 0),
                                    bbox_data.get('y', 0),
                                    bbox_data.get('width', 0),
                                    bbox_data.get('height', 0)
                                ]
                            }
                data = converted_data
            
            logger.info(f"元ラベルデータ読み込み: {len(data)}ファイル")
            return data
            
        except Exception as e:
            logger.error(f"ラベルデータ読み込みエラー: {e}")
            return {}
    
    def setup_augmentation_pipeline(self) -> Dict[str, Any]:
        """データ拡張パイプライン設定"""
        
        if ALBUMENTATIONS_AVAILABLE:
            return self.setup_albumentations_pipeline()
        else:
            return self.setup_basic_pipeline()
    
    def setup_albumentations_pipeline(self) -> Dict[str, Any]:
        """Albumentationsを使用したデータ拡張パイプライン"""
        
        # 1. 軽微な変換（高信頼度）
        light_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=5, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.GaussNoise(var_limit=(10, 30), p=0.3),
        ], bbox_params=A.BboxParams(format='xywh', label_fields=['class_labels']))
        
        # 2. 中程度変換（中信頼度）
        medium_transform = A.Compose([
            A.OneOf([
                A.HorizontalFlip(p=1.0),
                A.Rotate(limit=10, p=1.0),
                A.RandomScale(scale_limit=0.1, p=1.0),
            ], p=0.8),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=1.0),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            ], p=0.6),
            A.GaussNoise(var_limit=(10, 50), p=0.4),
        ], bbox_params=A.BboxParams(format='xywh', label_fields=['class_labels']))
        
        # 3. 強い変換（低信頼度・要レビュー）
        strong_transform = A.Compose([
            A.OneOf([
                A.HorizontalFlip(p=1.0),
                A.Rotate(limit=15, p=1.0),
                A.RandomScale(scale_limit=0.2, p=1.0),
                A.ElasticTransform(alpha=50, sigma=5, p=1.0),
            ], p=0.9),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=25, val_shift_limit=20, p=1.0),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
            ], p=0.8),
            A.OneOf([
                A.GaussNoise(var_limit=(20, 80), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                A.MultiplicativeNoise(multiplier=[0.9, 1.1], per_channel=True, p=1.0),
            ], p=0.5),
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.3),
        ], bbox_params=A.BboxParams(format='xywh', label_fields=['class_labels']))
        
        return {
            'light': light_transform,
            'medium': medium_transform,
            'strong': strong_transform
        }
    
    def setup_basic_pipeline(self) -> Dict[str, Any]:
        """基本的なデータ拡張パイプライン（albumentations不使用）"""
        return {
            'light': self.basic_light_transform,
            'medium': self.basic_medium_transform,
            'strong': self.basic_strong_transform
        }
    
    def basic_light_transform(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """基本的な軽微変換"""
        aug_image = image.copy()
        x, y, w, h = bbox
        
        # 水平反転（50%確率）
        if np.random.random() < 0.5:
            aug_image = cv2.flip(aug_image, 1)
            x = aug_image.shape[1] - x - w
        
        # 軽微な明度調整
        brightness = np.random.uniform(0.9, 1.1)
        aug_image = np.clip(aug_image * brightness, 0, 255).astype(np.uint8)
        
        return aug_image, (x, y, w, h)
    
    def basic_medium_transform(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """基本的な中程度変換"""
        aug_image = image.copy()
        x, y, w, h = bbox
        
        # 水平反転
        if np.random.random() < 0.5:
            aug_image = cv2.flip(aug_image, 1)
            x = aug_image.shape[1] - x - w
        
        # 明度・コントラスト調整
        brightness = np.random.uniform(0.8, 1.2)
        contrast = np.random.uniform(0.8, 1.2)
        aug_image = np.clip(aug_image * contrast + (brightness - 1) * 127, 0, 255).astype(np.uint8)
        
        # ノイズ追加
        noise = np.random.normal(0, 25, aug_image.shape).astype(np.uint8)
        aug_image = np.clip(aug_image.astype(int) + noise, 0, 255).astype(np.uint8)
        
        return aug_image, (x, y, w, h)
    
    def basic_strong_transform(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """基本的な強い変換"""
        aug_image = image.copy()
        x, y, w, h = bbox
        
        # 水平反転
        if np.random.random() < 0.5:
            aug_image = cv2.flip(aug_image, 1)
            x = aug_image.shape[1] - x - w
        
        # 強い明度・コントラスト調整
        brightness = np.random.uniform(0.6, 1.4)
        contrast = np.random.uniform(0.6, 1.4)
        aug_image = np.clip(aug_image * contrast + (brightness - 1) * 127, 0, 255).astype(np.uint8)
        
        # 強いノイズ追加
        noise = np.random.normal(0, 40, aug_image.shape).astype(np.uint8)
        aug_image = np.clip(aug_image.astype(int) + noise, 0, 255).astype(np.uint8)
        
        # 軽微なブラー
        if np.random.random() < 0.3:
            aug_image = cv2.GaussianBlur(aug_image, (3, 3), 0)
        
        return aug_image, (x, y, w, h)
    
    def calculate_confidence_score(self, augmentation_type: str, 
                                 bbox_change_ratio: float,
                                 intensity_factor: float) -> float:
        """
        疑似ラベル信頼度スコア計算
        
        Args:
            augmentation_type: 変換タイプ
            bbox_change_ratio: バウンディングボックス変化率
            intensity_factor: 変換強度係数
            
        Returns:
            0.0-1.0の信頼度スコア
        """
        # ベース信頼度
        base_confidence = {
            'light': 0.9,
            'medium': 0.7,
            'strong': 0.5
        }.get(augmentation_type, 0.5)
        
        # bbox変化によるペナルティ
        bbox_penalty = min(bbox_change_ratio * 0.5, 0.3)
        
        # 強度によるペナルティ  
        intensity_penalty = min(intensity_factor * 0.2, 0.2)
        
        # 最終信頼度
        confidence = base_confidence - bbox_penalty - intensity_penalty
        return max(0.0, min(1.0, confidence))
    
    def apply_augmentation(self, image: np.ndarray, bbox: Tuple[int, int, int, int],
                          aug_type: str) -> Tuple[np.ndarray, Tuple[int, int, int, int], float]:
        """
        データ拡張適用
        
        Args:
            image: 入力画像
            bbox: バウンディングボックス [x, y, w, h]
            aug_type: 拡張タイプ
            
        Returns:
            (拡張画像, 変換後bbox, 信頼度スコア)
        """
        try:
            x, y, w, h = bbox
            
            if ALBUMENTATIONS_AVAILABLE:
                # Albumentations使用
                bboxes = [[x, y, w, h]]
                class_labels = ['character']
                
                # 変換適用
                transform = self.augmentation_pipeline[aug_type]
                transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                
                # 結果取得
                aug_image = transformed['image']
                transformed_bboxes = transformed['bboxes']
                
                if not transformed_bboxes:
                    # 変換後にbboxが消失した場合
                    return aug_image, (0, 0, 0, 0), 0.0
                
                new_bbox = tuple(map(int, transformed_bboxes[0]))
            else:
                # 基本変換使用
                transform_func = self.augmentation_pipeline[aug_type]
                aug_image, new_bbox = transform_func(image, bbox)
            
            # bbox変化率計算
            original_area = w * h
            new_area = new_bbox[2] * new_bbox[3]
            area_change_ratio = abs(new_area - original_area) / max(original_area, 1)
            
            # 強度係数計算（画像変化の程度）
            intensity_factor = self.calculate_image_change_intensity(image, aug_image)
            
            # 信頼度計算
            confidence = self.calculate_confidence_score(aug_type, area_change_ratio, intensity_factor)
            
            return aug_image, new_bbox, confidence
            
        except Exception as e:
            logger.error(f"データ拡張適用エラー: {e}")
            return image, bbox, 0.0
    
    def calculate_image_change_intensity(self, original: np.ndarray, augmented: np.ndarray) -> float:
        """画像変化強度計算"""
        try:
            # MSE計算
            mse = np.mean((original.astype(float) - augmented.astype(float)) ** 2)
            # 正規化（0-1範囲）
            intensity = min(mse / 1000.0, 1.0)
            return intensity
        except:
            return 0.5  # デフォルト値
    
    def generate_pseudo_labels(self) -> DataExpansionResult:
        """疑似ラベル生成メイン処理"""
        try:
            logger.info("データ拡張開始")
            start_time = time.time()
            
            original_count = len(self.original_labels)
            target_generated = int(original_count * (self.target_expansion_ratio - 1))
            
            pseudo_labels = []
            augmentation_stats = {'light': 0, 'medium': 0, 'strong': 0}
            quality_distribution = {'high': 0, 'medium': 0, 'low': 0}
            
            generated_count = 0
            
            # 各画像に対して複数の拡張を生成
            augmentations_per_image = max(1, target_generated // original_count)
            
            for image_id, label_data in self.original_labels.items():
                if generated_count >= target_generated:
                    break
                
                # 元画像読み込み
                image_path = self.find_image_file(image_id)
                if not image_path:
                    logger.warning(f"画像ファイルが見つかりません: {image_id}")
                    continue
                
                image = cv2.imread(str(image_path))
                if image is None:
                    logger.warning(f"画像読み込み失敗: {image_path}")
                    continue
                
                original_bbox = tuple(label_data['red_box_coords'])
                
                # 複数の拡張を生成
                for aug_idx in range(augmentations_per_image):
                    if generated_count >= target_generated:
                        break
                    
                    # 拡張タイプ選択（段階的に強度を上げる）
                    if aug_idx == 0:
                        aug_type = 'light'
                    elif aug_idx <= 2:
                        aug_type = 'medium'
                    else:
                        aug_type = 'strong'
                    
                    # データ拡張適用
                    aug_image, transformed_bbox, confidence = self.apply_augmentation(
                        image, original_bbox, aug_type
                    )
                    
                    # 無効な結果はスキップ
                    if transformed_bbox == (0, 0, 0, 0) or confidence < 0.1:
                        continue
                    
                    # ファイル保存
                    aug_image_id = f"{image_id}_aug_{aug_type}_{aug_idx:02d}"
                    aug_image_path = self.images_dir / f"{aug_image_id}.png"
                    
                    cv2.imwrite(str(aug_image_path), aug_image)
                    
                    # 品質分類
                    if confidence >= 0.8:
                        quality = 'high'
                    elif confidence >= 0.6:
                        quality = 'medium'
                    else:
                        quality = 'low'
                    
                    # 疑似ラベル作成
                    pseudo_label = PseudoLabel(
                        image_id=aug_image_id,
                        source_image_path=str(image_path),
                        augmented_image_path=str(aug_image_path),
                        original_bbox=original_bbox,
                        transformed_bbox=transformed_bbox,
                        confidence_score=confidence,
                        augmentation_type=aug_type,
                        requires_manual_review=confidence < self.manual_review_threshold,
                        notes=f"Generated from {image_id} using {aug_type} augmentation"
                    )
                    
                    pseudo_labels.append(pseudo_label)
                    augmentation_stats[aug_type] += 1
                    quality_distribution[quality] += 1
                    generated_count += 1
                    
                    # 手動レビュー対象は別ディレクトリにもコピー
                    if pseudo_label.requires_manual_review:
                        review_path = self.review_dir / f"{aug_image_id}.png"
                        shutil.copy2(aug_image_path, review_path)
                
                # 進捗表示
                if (len(pseudo_labels) % 50) == 0:
                    logger.info(f"データ拡張進行中: {len(pseudo_labels)}/{target_generated} 生成済み")
            
            # ラベルファイル保存
            self.save_pseudo_labels(pseudo_labels)
            
            # 結果作成
            result = DataExpansionResult(
                original_count=original_count,
                generated_count=len(pseudo_labels),
                total_count=original_count + len(pseudo_labels),
                expansion_ratio=(original_count + len(pseudo_labels)) / original_count,
                pseudo_labels=pseudo_labels,
                augmentation_stats=augmentation_stats,
                quality_distribution=quality_distribution
            )
            
            total_time = time.time() - start_time
            logger.info(f"データ拡張完了: {len(pseudo_labels)}件生成 ({total_time:.1f}秒)")
            
            return result
            
        except Exception as e:
            logger.error(f"データ拡張エラー: {e}")
            raise
    
    def find_image_file(self, image_id: str) -> Optional[Path]:
        """画像ファイル検索"""
        possible_extensions = ['.png', '.jpg', '.jpeg']
        search_dirs = [
            # ユーザー提供の画像ファイル場所（優先）
            Path("/mnt/c/AItools/lora/train/yado/org/kana05_cursor"),
            Path("/mnt/c/AItools/lora/train/yado/org/kana07_cursor"), 
            Path("/mnt/c/AItools/lora/train/yado/org/kana08_cursor"),
            # フォールバック検索パス
            self.project_root / "test_small",
            self.project_root,
        ]
        
        for directory in search_dirs:
            for ext in possible_extensions:
                image_path = directory / f"{image_id}{ext}"
                if image_path.exists():
                    return image_path
        
        return None
    
    def save_pseudo_labels(self, pseudo_labels: List[PseudoLabel]):
        """疑似ラベル保存"""
        try:
            # JSON形式で保存
            labels_data = {
                pl.image_id: {
                    'red_box_coords': list(pl.transformed_bbox),
                    'confidence_score': pl.confidence_score,
                    'augmentation_type': pl.augmentation_type,
                    'requires_manual_review': pl.requires_manual_review,
                    'source_image_id': pl.image_id.split('_aug_')[0],
                    'original_bbox': list(pl.original_bbox),
                    'notes': pl.notes
                }
                for pl in pseudo_labels
            }
            
            # 詳細ラベルファイル（numpy配列をリストに変換）
            detailed_file = self.labels_dir / "pseudo_labels_detailed.json"
            detailed_data = []
            for pl in pseudo_labels:
                pl_dict = asdict(pl)
                # numpy配列をリストに変換
                if hasattr(pl_dict.get('mask'), 'tolist'):
                    pl_dict['mask'] = None  # マスクは巨大なため保存しない
                detailed_data.append(pl_dict)
            
            with open(detailed_file, 'w', encoding='utf-8') as f:
                json.dump(detailed_data, f, indent=2, ensure_ascii=False)
            
            # 簡易ラベルファイル（既存形式互換）
            simple_file = self.labels_dir / "pseudo_labels_simple.json"
            with open(simple_file, 'w', encoding='utf-8') as f:
                json.dump(labels_data, f, indent=2, ensure_ascii=False)
            
            # 統合ラベルファイル（元データ + 疑似ラベル）
            combined_data = {**self.original_labels, **labels_data}
            combined_file = self.labels_dir / "combined_labels.json"
            with open(combined_file, 'w', encoding='utf-8') as f:
                json.dump(combined_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"疑似ラベル保存: {len(pseudo_labels)}件")
            
        except Exception as e:
            logger.error(f"疑似ラベル保存エラー: {e}")
    
    def create_expansion_report(self, result: DataExpansionResult) -> str:
        """データ拡張レポート生成"""
        
        manual_review_count = sum(1 for pl in result.pseudo_labels if pl.requires_manual_review)
        
        report = f"""# Phase 1 データ拡張レポート

**実行日時**: {time.strftime('%Y-%m-%d %H:%M:%S')}

---

## 📊 拡張結果サマリー

### データ数
- **元データ**: {result.original_count}件
- **生成データ**: {result.generated_count}件
- **総データ数**: {result.total_count}件
- **拡張倍率**: {result.expansion_ratio:.1f}倍

### 品質分布
"""
        
        for quality, count in result.quality_distribution.items():
            percentage = (count / result.generated_count) * 100 if result.generated_count > 0 else 0
            report += f"- **{quality.upper()}品質**: {count}件 ({percentage:.1f}%)\n"
        
        report += f"""

### 拡張手法分布
"""
        
        for aug_type, count in result.augmentation_stats.items():
            percentage = (count / result.generated_count) * 100 if result.generated_count > 0 else 0
            report += f"- **{aug_type.upper()}変換**: {count}件 ({percentage:.1f}%)\n"
        
        report += f"""

---

## 🔍 品質管理

### 手動レビュー対象
- **要レビュー**: {manual_review_count}件
- **レビュー率**: {manual_review_count/result.generated_count*100:.1f}%
- **レビュー基準**: 信頼度 < {self.manual_review_threshold}

### 信頼度統計
"""
        
        confidences = [pl.confidence_score for pl in result.pseudo_labels]
        if confidences:
            report += f"""- **平均信頼度**: {np.mean(confidences):.3f}
- **信頼度範囲**: {min(confidences):.3f} - {max(confidences):.3f}
- **標準偏差**: {np.std(confidences):.3f}
"""
        
        report += f"""

---

## 📁 出力ファイル

### ディレクトリ構成
```
expanded_dataset/
├── images/              # 拡張画像 ({result.generated_count}件)
├── labels/              # ラベルファイル
│   ├── pseudo_labels_detailed.json
│   ├── pseudo_labels_simple.json  
│   └── combined_labels.json
└── manual_review/       # 手動レビュー対象 ({manual_review_count}件)
```

### 重要ファイル
- **combined_labels.json**: 元データ + 疑似ラベル統合版（Phase 1学習用）
- **manual_review/**: 信頼度の低い画像（人手確認推奨）

---

## 🚀 次のステップ

### 即座に実行
1. **手動レビュー**: `manual_review/`の{manual_review_count}件を確認
2. **品質確認**: 低信頼度サンプルの目視確認
3. **Phase 1学習準備**: `combined_labels.json`を使用してコマ検出ネット学習開始

### Phase 1学習設定
- **学習データ**: {result.total_count}件
- **検証分割**: Stratified 5-fold CV推奨
- **データローダー**: `combined_labels.json`使用

---

*Generated by Data Expansion System v1.0*
"""
        
        return report


def main():
    """メイン処理（テスト用）"""
    logging.basicConfig(level=logging.INFO)
    
    project_root = Path("/mnt/c/AItools/segment-anything")
    
    # データ拡張システム初期化
    expansion_system = DataExpansionSystem(project_root)
    
    # 疑似ラベル生成実行
    result = expansion_system.generate_pseudo_labels()
    
    # レポート生成
    report = expansion_system.create_expansion_report(result)
    
    # レポート保存
    report_file = expansion_system.output_dir / f"expansion_report_{time.strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\n" + "="*60)
    print("📊 Phase 1 データ拡張完了")
    print("="*60)
    print(f"元データ: {result.original_count}件")
    print(f"生成データ: {result.generated_count}件")
    print(f"拡張倍率: {result.expansion_ratio:.1f}倍")
    print(f"手動レビュー対象: {sum(1 for pl in result.pseudo_labels if pl.requires_manual_review)}件")
    print(f"レポート: {report_file}")


if __name__ == "__main__":
    main()