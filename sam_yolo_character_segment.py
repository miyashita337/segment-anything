#!/usr/bin/env python3
"""
SAM + YOLOv8 漫画キャラクター切り出しパイプライン

このスクリプトは、Segment Anything Model (SAM) と YOLOv8 を組み合わせて
漫画画像からキャラクターを自動的に切り出すパイプラインを提供します。

対話モードとバッチモードの両方をサポートしており、
SAMで生成された複数のマスク候補から、YOLOv8の人物検出スコアを使用して
最適なマスクを選択します。

使用例:
    # 対話形式：1枚の画像を確認しながら最適化
    python sam_yolo_character_segment.py --mode interactive --input image.jpg
    
    # バッチ形式：自動で複数枚を一気に処理
    python sam_yolo_character_segment.py --mode batch --input_dir ./manga_images/ --output_dir ./results/
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import time
import glob
import urllib.request
import psutil
import gc

# SAM関連インポート
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from segment_anything.utils.amg import batched_mask_to_box

# YOLOv8関連インポート
from ultralytics import YOLO


class PerformanceMonitor:
    """
    処理パフォーマンスを監視するクラス
    """
    
    def __init__(self):
        self.start_time = None
        self.stage_times = {}
        self.memory_usage = {}
        self.current_stage = None
        
    def start_monitoring(self):
        """モニタリング開始"""
        self.start_time = time.time()
        self.log_system_info()
        
    def start_stage(self, stage_name: str):
        """処理段階の開始"""
        if self.current_stage:
            self.end_stage()
        
        self.current_stage = stage_name
        self.stage_times[stage_name] = time.time()
        
        # メモリ使用量記録
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        gpu_memory = self.get_gpu_memory() if torch.cuda.is_available() else 0
        
        print(f"🔄 開始: {stage_name} (RAM: {memory_mb:.1f}MB, GPU: {gpu_memory:.1f}MB)")
        
    def end_stage(self):
        """処理段階の終了"""
        if not self.current_stage:
            return
            
        elapsed = time.time() - self.stage_times[self.current_stage]
        
        # メモリ使用量記録
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        gpu_memory = self.get_gpu_memory() if torch.cuda.is_available() else 0
        
        print(f"✅ 完了: {self.current_stage} ({elapsed:.2f}秒, RAM: {memory_mb:.1f}MB, GPU: {gpu_memory:.1f}MB)")
        
        self.stage_times[self.current_stage] = elapsed
        self.current_stage = None
        
        # ガベージコレクション実行
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_gpu_memory(self):
        """GPU メモリ使用量を取得"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024 / 1024
            return 0
        except:
            return 0
    
    def log_system_info(self):
        """システム情報をログ出力"""
        print("=== システム情報 ===")
        print(f"CPU: {psutil.cpu_count()} コア")
        print(f"RAM: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f}GB")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU RAM: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.1f}GB")
        print("=" * 20)
    
    def print_summary(self):
        """処理時間サマリーを出力"""
        if not self.start_time:
            return
            
        total_time = time.time() - self.start_time
        
        print("\n=== パフォーマンス サマリー ===")
        print(f"総処理時間: {total_time:.2f}秒")
        
        for stage, duration in self.stage_times.items():
            if isinstance(duration, float):
                percentage = (duration / total_time) * 100
                print(f"  {stage}: {duration:.2f}秒 ({percentage:.1f}%)")
        print("=" * 30)


def setup_japanese_font():
    """
    matplotlib用の日本語フォントを設定
    """
    try:
        # Windows環境での日本語フォント候補
        font_candidates = [
            'Yu Gothic UI',
            'Meiryo',
            'MS Gothic',
            'Hiragino Sans',
            'Noto Sans CJK JP',
            'DejaVu Sans'  # フォールバック
        ]
        
        # 利用可能なフォントを検索
        available_fonts = [font.name for font in fm.fontManager.ttflist]
        
        selected_font = None
        for font_name in font_candidates:
            if font_name in available_fonts:
                selected_font = font_name
                break
        
        if selected_font:
            plt.rcParams['font.family'] = selected_font
            print(f"✅ 日本語フォント設定完了: {selected_font}")
        else:
            # フォールバック: Unicode対応
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.rcParams['axes.unicode_minus'] = False
            print("⚠️ 日本語フォントが見つかりません。英語表示にフォールバックします")
            
    except Exception as e:
        print(f"⚠️ フォント設定エラー: {e}")
        # 最小限の設定
        plt.rcParams['axes.unicode_minus'] = False


def is_color_image(image: np.ndarray, threshold: float = 0.01) -> bool:
    """
    画像がカラー画像かどうかを判定
    
    Args:
        image: 入力画像 (BGR/RGB)
        threshold: カラー判定の閾値
        
    Returns:
        True if カラー画像, False if グレースケール画像
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        return False
    
    # RGBチャンネル間の差分を計算
    r, g, b = cv2.split(image)
    
    # 各チャンネル間の標準偏差を計算
    diff_rg = np.std(r.astype(np.float32) - g.astype(np.float32))
    diff_rb = np.std(r.astype(np.float32) - b.astype(np.float32))
    diff_gb = np.std(g.astype(np.float32) - b.astype(np.float32))
    
    # いずれかのチャンネル間差分が閾値を超えればカラー画像
    max_diff = max(diff_rg, diff_rb, diff_gb)
    is_color = max_diff > threshold
    
    print(f"カラー判定: 最大チャンネル差分={max_diff:.3f}, 閾値={threshold}, 結果={'カラー' if is_color else 'グレースケール'}")
    
    return is_color


def download_anime_yolo_model():
    """
    アニメ専用YOLOモデルをダウンロード
    """
    anime_model_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    anime_model_path = "yolov8n-anime.pt"
    
    if not os.path.exists(anime_model_path):
        try:
            print("🎌 アニメ専用YOLOモデルをダウンロード中...")
            urllib.request.urlretrieve(anime_model_url, anime_model_path)
            print(f"✅ アニメモデルダウンロード完了: {anime_model_path}")
            return anime_model_path
        except Exception as e:
            print(f"❌ アニメモデルダウンロード失敗: {e}")
            return None
    else:
        print(f"✅ アニメモデル既存: {anime_model_path}")
        return anime_model_path


def cleanup_old_preview_images(preview_dir: str, days_old: int = 7):
    """
    指定した日数より古いプレビュー画像を削除
    
    Args:
        preview_dir: プレビュー画像ディレクトリ
        days_old: 削除対象の経過日数（デフォルト: 7日）
    """
    if not os.path.exists(preview_dir):
        return
        
    current_time = time.time()
    cutoff_time = current_time - (days_old * 24 * 60 * 60)  # 指定日数前の秒数
    
    # プレビュー画像ファイルパターン
    patterns = [
        os.path.join(preview_dir, "choice_preview_*.jpg"),
        os.path.join(preview_dir, "choice_preview_*.jpeg"),
        os.path.join(preview_dir, "choice_preview_*.png")
    ]
    
    deleted_count = 0
    for pattern in patterns:
        for file_path in glob.glob(pattern):
            try:
                # ファイルの作成日時をチェック
                file_mtime = os.path.getmtime(file_path)
                if file_mtime < cutoff_time:
                    os.remove(file_path)
                    deleted_count += 1
                    print(f"古いプレビュー画像を削除: {file_path}")
            except Exception as e:
                print(f"ファイル削除エラー: {file_path} - {e}")
    
    if deleted_count > 0:
        print(f"合計 {deleted_count} 個の古いプレビュー画像を削除しました")
    else:
        print("削除対象の古いプレビュー画像はありませんでした")


class ClickHandler:
    """
    matplotlibでのクリックイベントを処理するクラス
    右クリックでの複数選択機能を追加
    """
    
    def __init__(self, masks, image_shape, display_shape):
        self.masks = masks  # マスクのリスト
        self.image_shape = image_shape  # 元画像のサイズ (height, width)
        self.display_shape = display_shape  # 表示画像のサイズ (height, width)
        self.clicked_point = None
        self.selected_mask_idx = None
        self.selected_masks = []  # 複数選択されたマスクのインデックス
        self.is_multi_select = False  # 複数選択モードフラグ
        self.figure = None
        self.ax = None
        
    def on_click(self, event):
        """
        クリックイベントのハンドラー
        左クリック: 単一選択または確定
        右クリック: 複数選択モード
        """
        if event.inaxes is None:
            # マスク外クリックで確定
            if self.is_multi_select and self.selected_masks:
                print(f"マスク外クリックで確定: {len(self.selected_masks)}個のマスクを結合")
                plt.close()
            return
            
        # クリック座標を取得
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
            
        self.clicked_point = (x, y)
        
        # 表示座標を元画像座標に変換
        orig_x = int(x * self.image_shape[1] / self.display_shape[1])
        orig_y = int(y * self.image_shape[0] / self.display_shape[0])
        
        # 座標変換のズレを計算
        scale_x = self.image_shape[1] / self.display_shape[1]
        scale_y = self.image_shape[0] / self.display_shape[0]
        
        # ズレが大きい場合は警告
        if abs(scale_x - scale_y) > 0.1:
            print(f"警告: 座標変換でズレが発生している可能性があります (scale_x: {scale_x:.3f}, scale_y: {scale_y:.3f})")
        
        # 境界チェック
        orig_x = max(0, min(orig_x, self.image_shape[1] - 1))
        orig_y = max(0, min(orig_y, self.image_shape[0] - 1))
        
        # クリックされたマスクを判定
        clicked_mask_idx = self._determine_mask(orig_x, orig_y)
        
        if clicked_mask_idx is None:
            print(f"クリック座標({orig_x}, {orig_y})にマスクがありません")
            return
            
        print(f"クリック座標: 表示({x:.1f}, {y:.1f}) -> 元画像({orig_x}, {orig_y}) -> マスク{clicked_mask_idx}")
        
        # ボタンによる処理分岐
        if event.button == 1:  # 左クリック
            if self.is_multi_select:
                # マルチセレクトモードでの左クリックは確定
                if self.selected_masks:
                    print(f"左クリックで確定: {len(self.selected_masks)}個のマスクを結合")
                    plt.close()
                else:
                    print("選択されたマスクがありません")
            else:
                # 単一選択モード
                self.selected_mask_idx = clicked_mask_idx
                print(f"左クリックでマスク{clicked_mask_idx}を選択")
                plt.close()
                
        elif event.button == 3:  # 右クリック
            self.is_multi_select = True
            if clicked_mask_idx in self.selected_masks:
                # 既に選択されているマスクを右クリックした場合は選択解除
                self.selected_masks.remove(clicked_mask_idx)
                print(f"右クリックでマスク{clicked_mask_idx}を選択解除 (選択中: {self.selected_masks})")
            else:
                # 新しいマスクを追加
                self.selected_masks.append(clicked_mask_idx)
                print(f"右クリックでマスク{clicked_mask_idx}を追加 (選択中: {self.selected_masks})")
                
            # 表示を更新して選択状態を反映
            self._update_display()
        
    def _determine_mask(self, x, y):
        """
        クリック位置でのマスクを判定
        """
        candidate_masks = []
        
        # クリック位置に該当するマスクを探す
        for i, mask_data in enumerate(self.masks):
            mask = mask_data['segmentation']
            if y < mask.shape[0] and x < mask.shape[1] and mask[y, x]:
                # マスクのピクセル数を計算
                pixel_count = np.sum(mask)
                candidate_masks.append((i, pixel_count))
        
        if not candidate_masks:
            print("エラー: クリック位置にマスクが見つかりません。もう一度クリックしてください。")
            return None
        
        # ピクセル数が最大のマスクを選択
        selected_idx, max_pixels = max(candidate_masks, key=lambda x: x[1])
        
        print(f"選択されたマスク: {selected_idx} (ピクセル数: {max_pixels})")
        if len(candidate_masks) > 1:
            print(f"重複マスク数: {len(candidate_masks)}, 最大ピクセル数で選択")
        
        return selected_idx
    
    def _update_display(self):
        """
        選択されたマスクの表示を更新（境界線を太くする）
        """
        if not self.figure or not self.ax:
            return
            
        # 既存の境界線を削除
        for artist in self.ax.get_children():
            if hasattr(artist, 'get_label'):
                label = artist.get_label()
                if label and isinstance(label, str) and label.startswith('selected_'):
                    artist.remove()
        
        # 選択されたマスクに太い境界線を追加
        colors = [
            (1.0, 0.0, 0.0),   # 赤
            (0.0, 1.0, 0.0),   # 緑
            (0.0, 0.0, 1.0),   # 青
            (1.0, 1.0, 0.0),   # 黄
            (1.0, 0.0, 1.0),   # マゼンタ
            (0.0, 1.0, 1.0),   # シアン
            (1.0, 0.5, 0.0),   # オレンジ
            (0.5, 0.0, 1.0),   # 紫
            (0.0, 0.5, 0.0),   # 濃い緑
            (0.5, 0.5, 0.5)    # グレー
        ]
        
        for mask_idx in self.selected_masks:
            if mask_idx < len(self.masks):
                mask = self.masks[mask_idx]['segmentation']
                
                # マスクの境界線を抽出
                import cv2
                # 表示サイズにリサイズ
                resized_mask = cv2.resize(mask.astype(np.uint8), 
                                        (self.display_shape[1], self.display_shape[0]))
                
                # 境界線を見つける
                contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # 境界線を描画
                for contour in contours:
                    contour = contour.squeeze()
                    if len(contour.shape) == 2 and contour.shape[0] > 2:
                        color = colors[mask_idx % len(colors)]
                        self.ax.plot(contour[:, 0], contour[:, 1], 
                                   color=color, linewidth=4, alpha=0.8,
                                   label=f'selected_{mask_idx}')
        
        # 表示を更新
        self.figure.canvas.draw()
        
    def merge_selected_masks(self):
        """
        選択されたマスクを論理和（OR）でマージ
        """
        if not self.selected_masks:
            return None
            
        # 最初のマスクを基準にする
        first_mask = self.masks[self.selected_masks[0]]['segmentation']
        merged_mask = first_mask.copy()
        
        # 他のマスクを論理和で結合
        for mask_idx in self.selected_masks[1:]:
            mask = self.masks[mask_idx]['segmentation']
            merged_mask = np.logical_or(merged_mask, mask)
        
        return merged_mask


class SAMYOLOCharacterSegmentor:
    """
    SAMとYOLOv8を組み合わせたキャラクター切り出しクラス
    """
    
    def __init__(self, 
                 sam_checkpoint: str = "sam_vit_h_4b8939.pth",
                 model_type: str = "vit_h",
                 yolo_model: str = "yolov8n.pt",
                 score_threshold: float = 0.15,
                 device: Optional[str] = None,
                 use_anime_yolo: bool = False):
        """
        初期化
        
        Args:
            sam_checkpoint: SAMのチェックポイントファイルパス
            model_type: SAMモデルの種類（vit_h, vit_l, vit_b）
            yolo_model: YOLOv8モデルファイルパス
            score_threshold: YOLO人物検出スコアの閾値
            device: 計算デバイス（cuda/cpu）
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.score_threshold = score_threshold
        
        # パフォーマンス監視
        self.monitor = PerformanceMonitor()
        
        # 日本語フォント設定
        setup_japanese_font()
        
        # アニメモードの場合、より低い閾値を使用
        if use_anime_yolo:
            self.score_threshold = max(0.05, score_threshold * 0.7)  # 閾値を30%下げる
            print(f"🎌 アニメモード: スコア閾値を {score_threshold} → {self.score_threshold} に調整")
        
        # SAM初期化
        self.monitor.start_stage("SAMモデル読み込み")
        print(f"SAMモデルを読み込み中... ({model_type})")
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=self.device)
        self.monitor.end_stage()
        
        # SAM自動マスク生成器（人物全体をカバーするバランス調整）
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=12,  # 32→12: 適度に大きな領域を生成
            pred_iou_thresh=0.6, # 0.88→0.6: 程よく緩い閾値
            stability_score_thresh=0.7, # 0.95→0.7: 程よく緩い安定性
            crop_n_layers=0,     # クロップ処理を無効化
            crop_n_points_downscale_factor=1,
            min_mask_region_area=3000,  # 100→3000: 適度な最小領域
            box_nms_thresh=0.6,  # 適度なNMS閾値
        )
        
        # YOLOv8初期化
        self.monitor.start_stage("YOLOモデル読み込み")
        self.use_anime_yolo = use_anime_yolo
        
        if use_anime_yolo:
            # アニメ・マンガ専用モデルを試行
            anime_models = [
                # 実際に利用可能なモデル
                "yolov8x.pt",  # より大きなモデルでアニメ検出精度向上
                "yolov8l.pt",  # 大型モデル
                "yolov8m.pt",  # 中型モデル
                yolo_model     # フォールバック用標準モデル
            ]
            
            print("🎌 アニメモード: より大きなYOLOモデルで検出精度向上を試行します")
            
            model_loaded = False
            for anime_model in anime_models:
                try:
                    print(f"アニメ用YOLOモデルを試行中... ({anime_model})")
                    self.yolo = YOLO(anime_model)
                    self.current_yolo_model = anime_model
                    model_loaded = True
                    print(f"✅ アニメ用YOLOモデルを読み込み完了: {anime_model}")
                    break
                except Exception as e:
                    print(f"❌ {anime_model} の読み込み失敗: {e}")
                    continue
            
            if not model_loaded:
                print(f"⚠️ アニメ用モデルが見つかりません。標準モデルにフォールバック: {yolo_model}")
                self.yolo = YOLO(yolo_model)
                self.current_yolo_model = yolo_model
        else:
            print(f"YOLOv8モデルを読み込み中... ({yolo_model})")
            self.yolo = YOLO(yolo_model)
            self.current_yolo_model = yolo_model
        
        print(f"デバイス: {self.device}")
        self.monitor.end_stage()
    
    def generate_masks(self, image: np.ndarray) -> List[dict]:
        """
        SAMを使用してマスクを生成
        
        Args:
            image: 入力画像 (RGB)
            
        Returns:
            マスクのリスト
        """
        self.monitor.start_stage("SAMマスク生成")
        print("SAMでマスクを生成中...")
        masks = self.mask_generator.generate(image)
        print(f"生成されたマスク数: {len(masks)}")
        
        # マスクサイズの統計情報を表示
        if masks:
            mask_areas = [np.sum(mask['segmentation']) for mask in masks]
            print(f"マスク面積統計: 最小={min(mask_areas)}, 最大={max(mask_areas)}, 平均={int(np.mean(mask_areas))}")
        
        self.monitor.end_stage()
        return masks
    
    def filter_masks_with_yolo(self, image: np.ndarray, masks: List[dict]) -> List[Tuple[dict, float]]:
        """
        YOLOv8を使用してマスクを人物検出スコアでフィルタリング
        
        Args:
            image: 入力画像 (RGB)
            masks: SAMで生成されたマスク
            
        Returns:
            (マスク, 人物スコア)のタプルリスト
        """
        self.monitor.start_stage("YOLO人物検出フィルタリング")
        print("YOLOv8で人物検出スコアを計算中...")
        
        mask_scores = []
        
        print(f"YOLOv8で{len(masks)}個のマスクを評価開始...")
        
        for i, mask_data in enumerate(masks):
            try:
                mask = mask_data['segmentation']
                bbox = mask_data['bbox']  # x, y, w, h
                
                # バウンディングボックスの座標を取得
                x, y, w, h = bbox
                x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                
                # 画像の境界チェック
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image.shape[1], x2)
                y2 = min(image.shape[0], y2)
                
                # バウンディングボックス内の画像を切り出し
                cropped_image = image[y1:y2, x1:x2]
                
                if cropped_image.size == 0:
                    mask_scores.append((mask_data, 0.0))
                    continue
                    
                # YOLOv8で人物検出（アニメモードの場合、より緩い設定）
                if self.use_anime_yolo:
                    # アニメモード: より多くの候補を検出
                    results = self.yolo(cropped_image, verbose=False, conf=0.01, iou=0.7)
                else:
                    # 標準モード
                    results = self.yolo(cropped_image, verbose=False)
                
                # 人物クラスの最高スコアを取得（アニメモデル対応）
                person_score = 0.0
                for result in results:
                    if result.boxes is not None:
                        boxes = result.boxes
                        for box in boxes:
                            class_id = int(box.cls)
                            
                            # 標準YOLOの人物クラス（0）またはアニメモデルの文字クラスをチェック
                            if self.use_anime_yolo:
                                # アニメモデルの場合、より多くのクラスを人物として認識
                                if class_id in [0, 1, 2, 3]:  # person, character, face, body など
                                    person_score = max(person_score, float(box.conf))
                            else:
                                # 標準モデルの場合、person クラスのみ
                                if class_id == 0:  # person class
                                    person_score = max(person_score, float(box.conf))
                
                mask_scores.append((mask_data, person_score))
                
                if (i + 1) % 10 == 0:
                    print(f"YOLO処理済み: {i + 1}/{len(masks)} (現在のスコア: {person_score:.3f})")
                    
            except Exception as e:
                print(f"⚠️ マスク{i}の処理でエラー: {e}")
                mask_scores.append((mask_data, 0.0))
                continue
        
        # スコアでソート（降順）
        mask_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 閾値以上のマスクのみ返す
        filtered_masks = [(mask, score) for mask, score in mask_scores if score >= self.score_threshold]
        
        print(f"人物検出スコア >= {self.score_threshold} のマスク数: {len(filtered_masks)}")
        
        # スコア分布を表示
        if mask_scores:
            all_scores = [score for _, score in mask_scores]
            print(f"YOLO検出スコア統計: 最高={max(all_scores):.3f}, 最低={min(all_scores):.3f}, 平均={np.mean(all_scores):.3f}")
            
            # 上位10個のスコアを表示
            top_scores = sorted(all_scores, reverse=True)[:10]
            print(f"上位10スコア: {[f'{s:.3f}' for s in top_scores]}")
            
            # アニメモードの場合、使用モデル情報も表示
            if self.use_anime_yolo:
                print(f"🎌 アニメモード使用中: {self.current_yolo_model}")
        
        self.monitor.end_stage()
        return filtered_masks
    
    def extract_character(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        マスクを使用してキャラクターを切り出し
        
        Args:
            image: 入力画像 (RGB)
            mask: マスク
            
        Returns:
            マスクされた画像
        """
        masked_image = image.copy()
        masked_image[~mask] = 0  # マスク外を黒にする
        return masked_image
    
    def process_single_image(self, image_path: str, output_dir: str, interactive: bool = False, mask_choice: int = None) -> str:
        """
        単一画像を処理
        
        Args:
            image_path: 入力画像パス
            output_dir: 出力ディレクトリ
            interactive: 対話モード
            mask_choice: 手動でマスク番号を指定 (0-4)
            
        Returns:
            処理結果 ("success", "skip", "error")
        """
        try:
            # 画像読み込み
            image = cv2.imread(image_path)
            if image is None:
                print(f"エラー: 画像を読み込めませんでした - {image_path}")
                return "error"
            
            # カラー画像チェック
            if is_color_image(image, threshold=10.0):  # 閾値を高めに設定してカラー画像を確実に検出
                print(f"スキップ: カラー画像のため処理をスキップします - {image_path}")
                return "skip"
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # マスク生成
            masks = self.generate_masks(image)
            
            if not masks:
                print(f"スキップ: マスクが生成されませんでした - {image_path}")
                return "skip"
            
            # YOLO フィルタリング
            filtered_masks = self.filter_masks_with_yolo(image, masks)
            
            if not filtered_masks:
                print(f"スキップ: 人物検出スコアが閾値以上のマスクがありませんでした - {image_path}")
                return "skip"
            
            # 結果の処理
            if interactive:
                return self._process_interactive(image, filtered_masks, image_path, output_dir, mask_choice)
            else:
                return self._process_automatic(image, filtered_masks, image_path, output_dir)
                
        except Exception as e:
            print(f"エラー: {image_path} の処理中にエラーが発生しました - {str(e)}")
            return "error"
    
    def process_choice_mode(self, input_dir: str, output_dir: str) -> None:
        """
        choiceモードでディレクトリ内の全画像を処理
        
        Args:
            input_dir: 入力ディレクトリ
            output_dir: 出力ディレクトリ
        """
        self.monitor.start_monitoring()
        self.monitor.start_stage("初期化・ファイル検索")
        
        # 画像ファイルを再帰的に取得
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        input_path = Path(input_dir)
        for ext in image_extensions:
            image_files.extend(input_path.rglob(f"*{ext}"))
            image_files.extend(input_path.rglob(f"*{ext.upper()}"))
        
        # 重複ファイルを除去
        image_files = list(set(image_files))
        image_files.sort()  # ファイル名でソートして処理順序を一定にする
        
        if not image_files:
            print(f"エラー: 入力ディレクトリに画像ファイルが見つかりません: {input_dir}")
            return
        
        print(f"処理対象画像数: {len(image_files)} (重複除去後)")
        self.monitor.end_stage()
        
        # バッチ処理
        self.monitor.start_stage("画像バッチ処理")
        success_count = 0
        skip_count = 0
        error_count = 0
        
        for i, image_file in enumerate(image_files):
            image_start_time = time.time()
            print(f"\n進捗: {i+1}/{len(image_files)} - {image_file.relative_to(input_path)}")
            
            try:
                # 画像読み込み
                image = cv2.imread(str(image_file))
                if image is None:
                    print(f"エラー: 画像を読み込めませんでした - {image_file}")
                    error_count += 1
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                load_time = time.time() - image_start_time
                print(f"  📂 画像読み込み: {load_time:.2f}秒")
                
                # マスク生成
                masks = self.generate_masks(image)
                
                if not masks:
                    print(f"スキップ: マスクが生成されませんでした - {image_file}")
                    skip_count += 1
                    continue
                
                # YOLO フィルタリング
                filtered_masks = self.filter_masks_with_yolo(image, masks)
                
                if not filtered_masks:
                    print(f"スキップ: 人物検出スコアが閾値以上のマスクがありませんでした - {image_file}")
                    skip_count += 1
                    continue
                
                # 相対パスを計算して出力ディレクトリ構造を保持
                relative_path = image_file.relative_to(input_path)
                output_subdir = Path(output_dir) / relative_path.parent
                
                # choiceモードで処理
                choice_start = time.time()
                result = self._process_choice_mode(image, filtered_masks, str(image_file), str(output_subdir))
                choice_time = time.time() - choice_start
                
                total_image_time = time.time() - image_start_time
                print(f"  🎯 選択処理: {choice_time:.2f}秒, 画像合計: {total_image_time:.2f}秒")
                
                if result == "success":
                    success_count += 1
                elif result == "skip":
                    skip_count += 1
                else:
                    error_count += 1
                    
            except KeyboardInterrupt:
                print("\n処理が中断されました")
                break
            except Exception as e:
                print(f"エラー: {image_file} の処理中にエラーが発生しました - {str(e)}")
                error_count += 1
        
        self.monitor.end_stage()
        
        print(f"\n処理完了: {success_count}/{len(image_files)} 枚成功, {skip_count} 枚スキップ, {error_count} 枚エラー")
        
        # matplotlib リソースをクリーンアップ
        try:
            plt.close('all')  # 全ての図を閉じる
            print("✅ GUI リソースをクリーンアップしました")
        except Exception as e:
            print(f"⚠️ GUI クリーンアップエラー: {e}")
        
        # パフォーマンスサマリーを表示
        self.monitor.print_summary()
    
    def _process_interactive(self, image: np.ndarray, filtered_masks: List[Tuple[dict, float]], 
                           image_path: str, output_dir: str, mask_choice: int = None) -> str:
        """
        対話モードでの処理
        """
        print(f"\n=== 対話モード: {os.path.basename(image_path)} ===")
        
        # 上位5つのマスクを表示
        top_masks = filtered_masks[:5]
        
        if len(top_masks) == 0:
            print("表示するマスクがありません")
            return "skip"
        
        # マスクを表示
        fig, axes = plt.subplots(1, min(len(top_masks), 5), figsize=(15, 3))
        if len(top_masks) == 1:
            axes = [axes]
        
        for i, (mask_data, score) in enumerate(top_masks):
            mask = mask_data['segmentation']
            masked_image = self.extract_character(image, mask)
            
            ax = axes[i] if len(top_masks) > 1 else axes[0]
            ax.imshow(masked_image)
            ax.set_title(f"Mask {i}: Score {score:.3f}")
            ax.axis('off')
        
        plt.tight_layout()
        
        # 対話モードでは表示を無効化（WSLでは表示できない）
        # plt.show()
        
        # 代わりに画像を保存して確認用に出力
        preview_path = os.path.join(os.path.dirname(image_path), "preview_masks.png")
        plt.savefig(preview_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"マスクプレビューを保存しました: {preview_path}")
        
        # マスクの選択
        if mask_choice is not None:
            # 手動でマスク番号が指定された場合
            if 0 <= mask_choice < len(top_masks):
                selected_mask_data, selected_score = top_masks[mask_choice]
                selected_mask = selected_mask_data['segmentation']
                
                # 保存
                output_path = self._save_result(image, selected_mask, image_path, output_dir)
                print(f"保存完了: {output_path} (マスク{mask_choice}選択、スコア: {selected_score:.3f})")
                return "success"
            else:
                print(f"エラー: マスク番号{mask_choice}が範囲外です (0-{len(top_masks)-1})")
                return "error"
        
        # ユーザー選択（WSLでは自動的に最高スコアを選択）
        if os.environ.get('WSL_DISTRO_NAME') or not sys.stdin.isatty():
            # WSL環境または非対話環境では自動選択
            print("WSL環境または非対話環境のため、最高スコアのマスクを自動選択します")
            selected_mask_data, selected_score = top_masks[0]
            selected_mask = selected_mask_data['segmentation']
            
            # 保存
            output_path = self._save_result(image, selected_mask, image_path, output_dir)
            print(f"保存完了: {output_path} (スコア: {selected_score:.3f})")
            return "success"
        
        # 通常の対話モード
        while True:
            try:
                choice = input(f"選択するマスク番号 (0-{len(top_masks)-1}), 's'でスキップ, 'q'で終了: ")
                
                if choice.lower() == 's':
                    print("スキップしました")
                    return "skip"
                elif choice.lower() == 'q':
                    print("処理を終了します")
                    return "quit"
                
                mask_idx = int(choice)
                if 0 <= mask_idx < len(top_masks):
                    selected_mask_data, selected_score = top_masks[mask_idx]
                    selected_mask = selected_mask_data['segmentation']
                    
                    # 保存
                    output_path = self._save_result(image, selected_mask, image_path, output_dir)
                    print(f"保存完了: {output_path} (スコア: {selected_score:.3f})")
                    return "success"
                else:
                    print(f"無効な選択です。0-{len(top_masks)-1}の範囲で入力してください。")
                    
            except ValueError:
                print("無効な入力です。数字を入力してください。")
            except EOFError:
                print("入力が終了しました。最高スコアのマスクを自動選択します。")
                selected_mask_data, selected_score = top_masks[0]
                selected_mask = selected_mask_data['segmentation']
                
                # 保存
                output_path = self._save_result(image, selected_mask, image_path, output_dir)
                print(f"保存完了: {output_path} (スコア: {selected_score:.3f})")
                return "success"
    
    def _process_automatic(self, image: np.ndarray, filtered_masks: List[Tuple[dict, float]], 
                         image_path: str, output_dir: str) -> str:
        """
        自動モードでの処理（最高スコアのマスクを選択）
        """
        if not filtered_masks:
            return "skip"
        
        # 最高スコアのマスクを選択
        best_mask_data, best_score = filtered_masks[0]
        best_mask = best_mask_data['segmentation']
        
        # 保存
        output_path = self._save_result(image, best_mask, image_path, output_dir)
        print(f"保存完了: {output_path} (スコア: {best_score:.3f})")
        
        return "success"
    
    def _save_result(self, image: np.ndarray, mask: np.ndarray, image_path: str, output_dir: str) -> str:
        """
        結果を保存
        """
        # 出力ディレクトリ作成
        os.makedirs(output_dir, exist_ok=True)
        
        # キャラクター切り出し
        masked_image = self.extract_character(image, mask)
        
        # ファイル名生成
        input_filename = os.path.basename(image_path)
        name, ext = os.path.splitext(input_filename)
        output_filename = f"{name}_character{ext}"
        output_path = os.path.join(output_dir, output_filename)
        
        # 保存
        masked_bgr = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, masked_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        
        return output_path
    
    def _process_choice_mode(self, image: np.ndarray, filtered_masks: List[Tuple[dict, float]], 
                           image_path: str, output_dir: str) -> str:
        """
        choiceモードでの処理（10個のマスクを重ねて表示してクリック選択）
        左クリック: 単一選択, 右クリック: 複数選択
        """
        print(f"\n=== クリック選択モード: {os.path.basename(image_path)} ===")
        
        # 上位10個のマスクを取得
        top_masks = filtered_masks[:10]
        
        if len(top_masks) == 0:
            print(f"スキップ: マスクが0個です")
            return "skip"
        
        # 透明度を設定してマスクを重ねて表示
        max_retry = 3
        for retry in range(max_retry):
            try:
                selection_result = self._display_overlayed_masks(image, top_masks, image_path)
                
                if selection_result is not None:
                    # マルチセレクションの場合の処理
                    if isinstance(selection_result, dict) and selection_result.get("type") == "multi_select":
                        selected_masks = selection_result["masks"]
                        merged_mask = selection_result["merged_mask"]
                        
                        # 保存
                        output_path = self._save_result(image, merged_mask, image_path, output_dir)
                        print(f"保存完了: {output_path} (マスク{selected_masks}を結合、{len(selected_masks)}個)")
                        return "success"
                    
                    # シングルセレクションの場合の処理
                    elif isinstance(selection_result, int):
                        selected_idx = selection_result
                        selected_mask_data, selected_score = top_masks[selected_idx]
                        selected_mask = selected_mask_data['segmentation']
                        
                        # 保存
                        output_path = self._save_result(image, selected_mask, image_path, output_dir)
                        print(f"保存完了: {output_path} (マスク{selected_idx}選択、スコア: {selected_score:.3f})")
                        return "success"
                else:
                    if retry < max_retry - 1:
                        print(f"再試行 {retry + 1}/{max_retry}")
                    else:
                        print("最大試行回数に達しました。この画像をスキップします。")
                        return "skip"
                        
            except Exception as e:
                print(f"エラー: {str(e)}")
                if retry < max_retry - 1:
                    print(f"再試行 {retry + 1}/{max_retry}")
                else:
                    print("最大試行回数に達しました。この画像をスキップします。")
                    return "error"
        
        return "skip"
    
    def _display_overlayed_masks(self, image: np.ndarray, top_masks: List[Tuple[dict, float]], 
                               image_path: str) -> Optional[int]:
        """
        10個のマスクを透明度を変えて重ねて表示し、クリック選択を受け付ける
        左クリック: 単一選択, 右クリック: 複数選択
        戻り値: int (単一選択) または dict (複数選択) または None (スキップ)
        """
        # 表示用の画像を作成
        display_image = image.copy().astype(np.float32)
        
        # 10色の配色を定義（区別しやすい色）
        colors = [
            (1.0, 0.0, 0.0),   # 赤
            (0.0, 1.0, 0.0),   # 緑
            (0.0, 0.0, 1.0),   # 青
            (1.0, 1.0, 0.0),   # 黄
            (1.0, 0.0, 1.0),   # マゼンタ
            (0.0, 1.0, 1.0),   # シアン
            (1.0, 0.5, 0.0),   # オレンジ
            (0.5, 0.0, 1.0),   # 紫
            (0.0, 0.5, 0.0),   # 濃い緑
            (0.5, 0.5, 0.5)    # グレー
        ]
        alpha = 0.3  # 統一透明度
        
        # マスクを重ねて適用
        for i, (mask_data, score) in enumerate(top_masks):
            mask = mask_data['segmentation']
            color = colors[i % len(colors)]  # 10色を循環使用
            
            # マスクの部分に色を付ける
            for c in range(3):  # RGB
                display_image[:, :, c] = np.where(
                    mask,
                    display_image[:, :, c] * (1 - alpha) + color[c] * 255 * alpha,
                    display_image[:, :, c]
                )
        
        # uint8に変換
        display_image = np.clip(display_image, 0, 255).astype(np.uint8)
        
        # 画像サイズを調整（大きすぎる場合はリサイズ）
        max_size = 800
        orig_h, orig_w = image.shape[:2]
        
        if max(orig_h, orig_w) > max_size:
            if orig_h > orig_w:
                new_h = max_size
                new_w = int(orig_w * max_size / orig_h)
            else:
                new_w = max_size
                new_h = int(orig_h * max_size / orig_w)
            
            display_image = cv2.resize(display_image, (new_w, new_h))
            display_shape = (new_h, new_w)
        else:
            display_shape = (orig_h, orig_w)
        
        # プレビュー画像を保存
        preview_dir = "./character_boudingbox_preview"
        os.makedirs(preview_dir, exist_ok=True)
        
        # 古いプレビュー画像をクリーンアップ（初回のみ実行）
        if not hasattr(self, '_preview_cleaned'):
            cleanup_old_preview_images(preview_dir, days_old=7)
            self._preview_cleaned = True
        
        preview_path = os.path.join(preview_dir, f"choice_preview_{os.path.basename(image_path)}")
        cv2.imwrite(preview_path, cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR))
        print(f"プレビュー画像を保存しました: {preview_path}")
        # マスク情報を出力（フォント対応）
        mask_info = []
        color_names_jp = ["赤", "緑", "青", "黄", "マゼンタ", "シアン", "オレンジ", "紫", "濃緑", "グレー"]
        color_names_en = ["Red", "Green", "Blue", "Yellow", "Magenta", "Cyan", "Orange", "Purple", "DarkGreen", "Gray"]
        
        use_english = plt.rcParams.get('font.family') == ['DejaVu Sans']
        color_names = color_names_en if use_english else color_names_jp
        
        for i, (mask_data, score) in enumerate(top_masks):
            color_name = color_names[i % len(color_names)]
            if use_english:
                mask_info.append(f"{color_name}=Mask{i}(Score:{score:.3f})")
            else:
                mask_info.append(f"{color_name}=マスク{i}(スコア:{score:.3f})")
        print(", ".join(mask_info))
        
        # matplotlibを使用してクリック選択
        backends_to_try = ['TkAgg', 'Qt5Agg', 'Agg']
        
        for backend in backends_to_try:
            try:
                print(f"バックエンド '{backend}' を試行中...")
                import matplotlib
                matplotlib.use(backend)
                
                if backend == 'Agg':
                    # Aggバックエンドでは保存のみ対応
                    print("Aggバックエンドのため、キー入力モードを使用します")
                    break
                
                selected_idx = self._show_matplotlib_viewer_for_selection(display_image, top_masks, image_path, (orig_h, orig_w), display_shape)
                return selected_idx
                
            except Exception as e:
                print(f"バックエンド '{backend}' でエラー: {e}")
                continue
        
        print("全てのGUIバックエンドが失敗しました。キー入力モードに切り替えます")
        
        # フォールバック：キー入力で選択
        while True:
            try:
                max_idx = len(top_masks) - 1
                choice = input(f"選択するマスク番号 (0-{max_idx}), 's'でスキップ: ")
                
                if choice.lower() == 's':
                    print("スキップしました")
                    return None
                
                mask_idx = int(choice)
                if 0 <= mask_idx <= max_idx:
                    selected_mask_data, selected_score = top_masks[mask_idx]
                    print(f"マスク{mask_idx}を選択しました (スコア: {selected_score:.3f})")
                    return mask_idx
                else:
                    print(f"無効な選択です。0-{max_idx}の範囲で入力してください。")
                    
            except ValueError:
                print("無効な入力です。数字を入力してください。")
            except EOFError:
                print("入力が終了しました。スキップします。")
                return None
    
    def _show_matplotlib_viewer_for_selection(self, display_image: np.ndarray, top_masks: List[Tuple[dict, float]], 
                                            image_path: str, orig_shape: Tuple[int, int], display_shape: Tuple[int, int]) -> Optional[int]:
        """
        matplotlibのClickHandlerを使用してクリック選択を実装
        左クリック: 単一選択または確定
        右クリック: 複数選択モード
        """
        print("matplotlibウィンドウでマスクをクリックして選択してください...")
        
        # マスク情報を表示（フォント対応）
        color_names_jp = ["赤", "緑", "青", "黄", "マゼンタ", "シアン", "オレンジ", "紫", "濃緑", "グレー"]
        color_names_en = ["Red", "Green", "Blue", "Yellow", "Magenta", "Cyan", "Orange", "Purple", "DarkGreen", "Gray"]
        
        # フォント設定に応じて表示言語を選択
        use_english = plt.rcParams.get('font.family') == ['DejaVu Sans']
        color_names = color_names_en if use_english else color_names_jp
        
        for i, (mask_data, score) in enumerate(top_masks):
            color_name = color_names[i % len(color_names)]
            if use_english:
                print(f"{color_name}=Mask{i}(Score:{score:.3f})")
            else:
                print(f"{color_name}=マスク{i}(スコア:{score:.3f})")
        
        if use_english:
            print("\nControls:")
            print("- Left click: Single selection or confirm")
            print("- Right click: Multiple selection (add/remove)")
            print("- Click outside masks: Confirm multiple selection")
            print("- Click on colored mask areas")
        else:
            print("\n操作方法:")
            print("- 左クリック: 単一選択または確定")
            print("- 右クリック: 複数選択（追加/削除）")
            print("- マスク外をクリック: 複数選択の確定")
            print("- マスクの色がついた部分をクリックしてください")
        
        # 図を表示
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(display_image)
        
        # タイトルを更新して10マスク対応（フォント対応）
        title_parts = []
        for i, (mask_data, score) in enumerate(top_masks):
            color_name = color_names[i % len(color_names)]
            if use_english:
                title_parts.append(f"{color_name}=Mask{i}(Score:{score:.3f})")
            else:
                title_parts.append(f"{color_name}=マスク{i}(スコア:{score:.3f})")
        
        if use_english:
            title_text = f"Mask Selection: {os.path.basename(image_path)}\n{', '.join(title_parts)}\nLeft: Single/Confirm, Right: Multi-select"
        else:
            title_text = f"マスク選択: {os.path.basename(image_path)}\n{', '.join(title_parts)}\n左クリック: 単一選択/確定, 右クリック: 複数選択"
        
        ax.set_title(title_text, fontsize=10)
        ax.axis('off')
        
        # ClickHandlerを作成
        mask_list = [mask_data for mask_data, _ in top_masks]
        click_handler = ClickHandler(mask_list, orig_shape, display_shape)
        click_handler.figure = fig
        click_handler.ax = ax
        
        # クリックイベントを接続
        cid = fig.canvas.mpl_connect('button_press_event', click_handler.on_click)
        
        # 表示して待機
        try:
            print("画像内をクリックしてください...")
            plt.show()
            
            # 結果を処理
            if click_handler.is_multi_select and click_handler.selected_masks:
                # 複数選択モード
                print(f"複数選択: {len(click_handler.selected_masks)}個のマスクが選択されました")
                print(f"選択されたマスク: {click_handler.selected_masks}")
                
                # 複数マスクをマージ
                merged_mask = click_handler.merge_selected_masks()
                
                # 仮想的な結果を返す（複数選択の場合は特別な処理）
                return {"type": "multi_select", "masks": click_handler.selected_masks, "merged_mask": merged_mask}
                
            elif click_handler.selected_mask_idx is not None:
                # 単一選択モード
                selected_idx = click_handler.selected_mask_idx
                print(f"マスク{selected_idx}が選択されました (スコア: {top_masks[selected_idx][1]:.3f})")
                return selected_idx
            else:
                print("クリック位置にマスクが見つかりませんでした")
                return None
                
        except Exception as e:
            print(f"クリック選択でエラーが発生しました: {e}")
            return None
        finally:
            # イベントハンドラーを解除
            if 'cid' in locals():
                try:
                    fig.canvas.mpl_disconnect(cid)
                except:
                    pass
            # 図を確実に閉じる
            try:
                plt.close(fig)
                plt.close('all')
            except:
                pass
    
    def _determine_mask_from_click(self, x: int, y: int, top_masks: List[Tuple[dict, float]]) -> Optional[int]:
        """
        クリック位置からマスクを判定
        """
        candidate_masks = []
        
        # クリック位置に該当するマスクを探す
        for i, (mask_data, score) in enumerate(top_masks):
            mask = mask_data['segmentation']
            if y < mask.shape[0] and x < mask.shape[1] and mask[y, x]:
                # マスクのピクセル数を計算
                pixel_count = np.sum(mask)
                candidate_masks.append((i, pixel_count, score))
        
        if not candidate_masks:
            return None
        
        # ピクセル数が最大のマスクを選択
        selected_idx, max_pixels, score = max(candidate_masks, key=lambda x: x[1])
        
        print(f"選択されたマスク詳細: マスク{selected_idx} (スコア: {score:.3f}, ピクセル数: {max_pixels})")
        if len(candidate_masks) > 1:
            print(f"重複マスク数: {len(candidate_masks)}, 最大ピクセル数で選択")
        
        return selected_idx
    
    def _show_image_viewer_for_selection(self, display_image: np.ndarray, top_masks: List[Tuple[dict, float]], 
                                       image_path: str, orig_shape: Tuple[int, int], display_shape: Tuple[int, int]) -> Optional[int]:
        """
        OpenCVを使用して画像ビューワーを表示してクリック選択を受け付ける
        """
        class OpenCVImageViewer:
            def __init__(self, image, masks, image_name, orig_shape, display_shape):
                self.image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # OpenCV用にBGRに変換
                self.masks = [mask_data for mask_data, _ in masks]
                self.mask_scores = [score for _, score in masks]
                self.image_name = image_name
                self.orig_shape = orig_shape  # (height, width)
                self.display_shape = display_shape  # (height, width)
                self.selected_mask_idx = None
                
            def mouse_callback(self, event, x, y, flags, param):
                """マウスクリックイベントハンドラー"""
                if event == cv2.EVENT_LBUTTONDOWN:
                    # 表示座標を元画像座標に変換
                    orig_x = int(x * self.orig_shape[1] / self.display_shape[1])
                    orig_y = int(y * self.orig_shape[0] / self.display_shape[0])
                    
                    # 座標変換のズレを計算
                    scale_x = self.orig_shape[1] / self.display_shape[1]
                    scale_y = self.orig_shape[0] / self.display_shape[0]
                    
                    # ズレが大きい場合は警告
                    if abs(scale_x - scale_y) > 0.1:
                        print(f"警告: 座標変換でズレが発生している可能性があります (scale_x: {scale_x:.3f}, scale_y: {scale_y:.3f})")
                    
                    # 境界チェック
                    orig_x = max(0, min(orig_x, self.orig_shape[1] - 1))
                    orig_y = max(0, min(orig_y, self.orig_shape[0] - 1))
                    
                    print(f"クリック座標: 表示({x}, {y}) -> 元画像({orig_x}, {orig_y})")
                    
                    # マスクを判定
                    self.selected_mask_idx = self._determine_mask(orig_x, orig_y)
                    
                    if self.selected_mask_idx is not None:
                        print(f"マスク{self.selected_mask_idx}が選択されました！ウィンドウを閉じています...")
                        cv2.destroyAllWindows()
                    else:
                        print("マスクが見つかりません。マスクの色がついた部分をクリックしてください。")
                
            def _determine_mask(self, x, y):
                """クリック位置でのマスクを判定"""
                candidate_masks = []
                
                # クリック位置に該当するマスクを探す
                for i, mask_data in enumerate(self.masks):
                    mask = mask_data['segmentation']
                    if y < mask.shape[0] and x < mask.shape[1] and mask[y, x]:
                        # マスクのピクセル数を計算
                        pixel_count = np.sum(mask)
                        candidate_masks.append((i, pixel_count))
                
                if not candidate_masks:
                    return None
                
                # ピクセル数が最大のマスクを選択
                selected_idx, max_pixels = max(candidate_masks, key=lambda x: x[1])
                
                print(f"選択されたマスク: {selected_idx} (スコア: {self.mask_scores[selected_idx]:.3f}, ピクセル数: {max_pixels})")
                if len(candidate_masks) > 1:
                    print(f"重複マスク数: {len(candidate_masks)}, 最大ピクセル数で選択")
                
                return selected_idx
            
            def show(self):
                """OpenCVウィンドウで画像を表示"""
                window_name = f"マスク選択: {os.path.basename(self.image_name)}"
                
                # ウィンドウを作成
                cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
                cv2.setMouseCallback(window_name, self.mouse_callback)
                
                # 情報をコンソールに表示
                print(f"\n=== {window_name} ===")
                print(f"赤=マスク0(スコア:{self.mask_scores[0]:.3f})")
                print(f"緑=マスク1(スコア:{self.mask_scores[1]:.3f})")
                print(f"青=マスク2(スコア:{self.mask_scores[2]:.3f})")
                print("マスクをクリックして選択してください。")
                print("'s'キーでスキップ、'q'キーで終了")
                
                while True:
                    # 画像を表示
                    cv2.imshow(window_name, self.image)
                    
                    # キー入力を待機
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('s'):
                        # スキップ
                        print("スキップしました")
                        self.selected_mask_idx = None
                        break
                    elif key == ord('q') or key == 27:  # ESCキー
                        # 終了
                        print("処理を終了します")
                        self.selected_mask_idx = None
                        break
                    elif self.selected_mask_idx is not None:
                        # マスクが選択された
                        break
                
                cv2.destroyAllWindows()
                return self.selected_mask_idx
        
        # OpenCV画像ビューワーを作成して表示
        viewer = OpenCVImageViewer(display_image, top_masks, image_path, orig_shape, display_shape)
        return viewer.show()


def main():
    """
    メイン関数
    """
    parser = argparse.ArgumentParser(description="SAM + YOLOv8 漫画キャラクター切り出しパイプライン")
    
    # 必須引数
    parser.add_argument("--mode", required=True, choices=["interactive", "batch", "choice"], 
                      help="処理モード (interactive: 対話形式, batch: バッチ形式, choice: クリック選択形式)")
    
    # 入力関連
    parser.add_argument("--input", type=str, help="単一画像のパス（対話形式）")
    parser.add_argument("--input_dir", type=str, help="バッチ処理時の画像ディレクトリ")
    parser.add_argument("--output_dir", type=str, default="./results", 
                      help="切り出した画像の保存先 (デフォルト: ./results)")
    parser.add_argument("--mask_choice", type=int, help="手動でマスク番号を指定 (0-4)")
    
    # モデル関連
    parser.add_argument("--model-type", type=str, default="vit_h", 
                      choices=["vit_h", "vit_l", "vit_b"],
                      help="SAMモデルの種類 (デフォルト: vit_h)")
    parser.add_argument("--sam-checkpoint", type=str, default="sam_vit_h_4b8939.pth",
                      help="SAMのチェックポイントファイルのパス")
    parser.add_argument("--yolo-model", type=str, default="yolov8n.pt",
                      help="YOLOv8モデルのパス or モデル名")
    parser.add_argument("--score-threshold", type=float, default=0.15,
                      help="YOLOv8の人物スコア閾値 (デフォルト: 0.15)")
    parser.add_argument("--anime-mode", action="store_true",
                      help="アニメ・マンガ専用YOLOモデルを使用")
    
    args = parser.parse_args()
    
    # 引数チェック
    if args.mode == "interactive" and not args.input:
        print("エラー: --input を指定してください（対話形式）")
        sys.exit(1)
    
    if args.mode == "batch" and not args.input_dir:
        print("エラー: --input_dir を指定してください（バッチ形式）")
        sys.exit(1)
    
    if args.mode == "choice" and not args.input_dir:
        print("エラー: --input_dir を指定してください（クリック選択形式）")
        sys.exit(1)
    
    # SAMチェックポイントファイルの存在確認
    if not os.path.exists(args.sam_checkpoint):
        print(f"エラー: SAMチェックポイントファイルが見つかりません: {args.sam_checkpoint}")
        sys.exit(1)
    
    try:
        # セグメンター初期化
        segmentor = SAMYOLOCharacterSegmentor(
            sam_checkpoint=args.sam_checkpoint,
            model_type=args.model_type,
            yolo_model=args.yolo_model,
            score_threshold=args.score_threshold,
            use_anime_yolo=args.anime_mode
        )
        
        if args.mode == "interactive":
            # 対話モード
            if not os.path.exists(args.input):
                print(f"エラー: 入力画像が見つかりません: {args.input}")
                sys.exit(1)
            
            result = segmentor.process_single_image(args.input, args.output_dir, interactive=True, mask_choice=args.mask_choice)
            if result == "error":
                print("処理中にエラーが発生しました")
                sys.exit(1)
            elif result == "quit":
                print("処理を終了しました")
                sys.exit(0)
        
        elif args.mode == "batch":
            # バッチモード
            if not os.path.exists(args.input_dir):
                print(f"エラー: 入力ディレクトリが見つかりません: {args.input_dir}")
                sys.exit(1)
            
            # 画像ファイルを再帰的に取得
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_files = []
            
            input_path = Path(args.input_dir)
            for ext in image_extensions:
                image_files.extend(input_path.rglob(f"*{ext}"))
                image_files.extend(input_path.rglob(f"*{ext.upper()}"))
            
            # 重複ファイルを除去
            image_files = list(set(image_files))
            image_files.sort()  # ファイル名でソートして処理順序を一定にする
            
            if not image_files:
                print(f"エラー: 入力ディレクトリに画像ファイルが見つかりません: {args.input_dir}")
                sys.exit(1)
            
            print(f"処理対象画像数: {len(image_files)} (重複除去後)")
            
            # バッチ処理
            success_count = 0
            skip_count = 0
            for i, image_file in enumerate(image_files):
                print(f"\n進捗: {i+1}/{len(image_files)} - {image_file.relative_to(input_path)}")
                
                # 相対パスを計算して出力ディレクトリ構造を保持
                relative_path = image_file.relative_to(input_path)
                output_subdir = Path(args.output_dir) / relative_path.parent
                
                result = segmentor.process_single_image(str(image_file), str(output_subdir), interactive=False)
                if result == "success":
                    success_count += 1
                elif result == "skip":
                    skip_count += 1
            
            print(f"\n処理完了: {success_count}/{len(image_files)} 枚成功, {skip_count} 枚スキップ")
        
        elif args.mode == "choice":
            # クリック選択モード
            if not os.path.exists(args.input_dir):
                print(f"エラー: 入力ディレクトリが見つかりません: {args.input_dir}")
                sys.exit(1)
            
            segmentor.process_choice_mode(args.input_dir, args.output_dir)
            print("🎉 全ての画像の処理が完了しました！")
            sys.exit(0)  # 正常終了
    
    except KeyboardInterrupt:
        print("\n処理が中断されました")
        sys.exit(1)
    except Exception as e:
        print(f"エラー: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()