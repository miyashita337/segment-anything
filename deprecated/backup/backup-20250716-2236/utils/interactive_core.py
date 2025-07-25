#!/usr/bin/env python3
"""
Phase 3: インタラクティブ補助機能（コア部分）
GUI部分を除いたコア機能のみ
"""

import numpy as np
import cv2

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


class InteractiveAssistant:
    """インタラクティブ補助機能のコアクラス（GUI非依存）"""
    
    def __init__(self):
        self.current_image = None
        self.processed_image = None
        self.seed_points = []  # [(x, y, is_positive), ...]
        self.selected_region = None  # (x, y, w, h)
        self.sam_model = None
        self.yolo_model = None
    
    def set_models(self, sam_model, yolo_model):
        """SAMとYOLOモデルを設定"""
        self.sam_model = sam_model
        self.yolo_model = yolo_model
    
    def load_image(self, image_path: str) -> bool:
        """画像を読み込み"""
        try:
            self.current_image = cv2.imread(image_path)
            if self.current_image is None:
                return False
            
            # プレビュー用に画像をリサイズ
            height, width = self.current_image.shape[:2]
            if max(height, width) > 1024:
                scale = 1024 / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                self.processed_image = cv2.resize(self.current_image, (new_width, new_height))
            else:
                self.processed_image = self.current_image.copy()
            
            # 初期化
            self.seed_points = []
            self.selected_region = None
            
            return True
        except Exception as e:
            print(f"画像読み込みエラー: {e}")
            return False
    
    def add_seed_point(self, x: int, y: int, is_positive: bool = True):
        """シードポイントを追加"""
        self.seed_points.append((x, y, is_positive))
    
    def remove_last_seed_point(self):
        """最後のシードポイントを削除"""
        if self.seed_points:
            self.seed_points.pop()
    
    def clear_seed_points(self):
        """全シードポイントをクリア"""
        self.seed_points = []
    
    def set_region(self, x: int, y: int, w: int, h: int):
        """注目領域を設定"""
        self.selected_region = (x, y, w, h)
    
    def clear_region(self):
        """注目領域をクリア"""
        self.selected_region = None
    
    def generate_mask_with_seeds(self) -> Optional[np.ndarray]:
        """シードポイントを使用してSAMマスクを生成"""
        if not self.seed_points or self.sam_model is None:
            return None
        
        try:
            # シードポイントを正負に分離
            positive_points = []
            negative_points = []
            
            for x, y, is_positive in self.seed_points:
                if is_positive:
                    positive_points.append([x, y])
                else:
                    negative_points.append([x, y])
            
            # SAMにシードポイントを渡してマスク生成
            input_points = positive_points + negative_points if negative_points else positive_points
            input_labels = [1] * len(positive_points) + [0] * len(negative_points)
            
            if not input_points:
                return None
            
            # SAMの予測実行
            rgb_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            
            # SAMの prompt-based prediction を使用
            from segment_anything import SamPredictor
            
            if hasattr(self.sam_model, 'sam'):
                predictor = SamPredictor(self.sam_model.sam)
            else:
                predictor = SamPredictor(self.sam_model)
            
            predictor.set_image(rgb_image)
            
            masks, scores, _ = predictor.predict(
                point_coords=np.array(input_points),
                point_labels=np.array(input_labels),
                multimask_output=True
            )
            
            # 最も良いマスクを選択
            if len(masks) > 0:
                best_mask_idx = np.argmax(scores)
                return masks[best_mask_idx].astype(np.uint8) * 255
            
            return None
            
        except Exception as e:
            print(f"マスク生成エラー: {e}")
            return None
    
    def generate_mask_with_region(self) -> Optional[np.ndarray]:
        """指定領域を使用してSAMマスクを生成"""
        if self.selected_region is None or self.sam_model is None:
            return None
        
        try:
            x, y, w, h = self.selected_region
            
            # SAMのバウンディングボックス予測を使用
            rgb_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            
            from segment_anything import SamPredictor
            
            if hasattr(self.sam_model, 'sam'):
                predictor = SamPredictor(self.sam_model.sam)
            else:
                predictor = SamPredictor(self.sam_model)
            
            predictor.set_image(rgb_image)
            
            # バウンディングボックス形式: [x1, y1, x2, y2]
            box = np.array([x, y, x + w, y + h])
            
            masks, scores, _ = predictor.predict(
                box=box,
                multimask_output=True
            )
            
            # 最も良いマスクを選択
            if len(masks) > 0:
                best_mask_idx = np.argmax(scores)
                return masks[best_mask_idx].astype(np.uint8) * 255
            
            return None
            
        except Exception as e:
            print(f"領域マスク生成エラー: {e}")
            return None
    
    def extract_character_interactive(self, output_path: str = None) -> Dict[str, Any]:
        """インタラクティブにキャラクターを抽出"""
        result = {
            'success': False,
            'output_path': None,
            'method': 'interactive',
            'seed_points': self.seed_points.copy(),
            'selected_region': self.selected_region,
            'error': None
        }
        
        try:
            # マスク生成方法を決定
            mask = None
            
            if self.seed_points:
                mask = self.generate_mask_with_seeds()
                result['method'] = 'seed_points'
            elif self.selected_region:
                mask = self.generate_mask_with_region()
                result['method'] = 'bounding_box'
            else:
                result['error'] = "シードポイントまたは領域が指定されていません"
                return result
            
            if mask is None:
                result['error'] = "マスク生成に失敗しました"
                return result
            
            # キャラクター抽出
            from utils.postprocessing import (
                crop_to_content,
                extract_character_from_image,
                save_character_result,
            )
            
            character_image = extract_character_from_image(
                self.current_image,
                mask,
                background_color=(0, 0, 0)
            )
            
            # クロップ
            cropped_character, cropped_mask, crop_bbox = crop_to_content(
                character_image,
                mask,
                padding=10
            )
            
            # 保存
            if output_path is None:
                output_path = "/tmp/interactive_extraction"
            
            save_success = save_character_result(
                cropped_character,
                cropped_mask,
                output_path,
                save_mask=True,
                save_transparent=True
            )
            
            if save_success:
                result['success'] = True
                result['output_path'] = output_path
            else:
                result['error'] = "保存に失敗しました"
            
            return result
            
        except Exception as e:
            result['error'] = f"抽出エラー: {e}"
            return result


# 便利関数
def quick_extract_with_points(image_path: str, points: List[Tuple[int, int, bool]], output_path: str = None) -> Dict[str, Any]:
    """シードポイントでクイック抽出"""
    from hooks.start import get_sam_model, get_yolo_model
    
    assistant = InteractiveAssistant()
    sam_model = get_sam_model()
    yolo_model = get_yolo_model()
    
    if not sam_model or not yolo_model:
        return {'success': False, 'error': 'Models not initialized'}
    
    assistant.set_models(sam_model, yolo_model)
    
    if not assistant.load_image(image_path):
        return {'success': False, 'error': 'Failed to load image'}
    
    for x, y, is_positive in points:
        assistant.add_seed_point(x, y, is_positive)
    
    return assistant.extract_character_interactive(output_path)


def quick_extract_with_region(image_path: str, region: Tuple[int, int, int, int], output_path: str = None) -> Dict[str, Any]:
    """バウンディングボックスでクイック抽出"""
    from hooks.start import get_sam_model, get_yolo_model
    
    assistant = InteractiveAssistant()
    sam_model = get_sam_model()
    yolo_model = get_yolo_model()
    
    if not sam_model or not yolo_model:
        return {'success': False, 'error': 'Models not initialized'}
    
    assistant.set_models(sam_model, yolo_model)
    
    if not assistant.load_image(image_path):
        return {'success': False, 'error': 'Failed to load image'}
    
    x, y, w, h = region
    assistant.set_region(x, y, w, h)
    
    return assistant.extract_character_interactive(output_path)