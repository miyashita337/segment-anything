#!/usr/bin/env python3
"""
Phase 3: インタラクティブ補助機能
手動介入によるキャラクター抽出支援システム
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import time
from pathlib import Path


class InteractiveAssistant:
    """インタラクティブ補助機能のコアクラス"""
    
    def __init__(self):
        self.current_image = None
        self.processed_image = None
        self.seed_points = []  # [(x, y, is_positive), ...]
        self.selected_region = None  # (x, y, w, h)
        self.preview_callback = None
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
            from utils.postprocessing import extract_character_from_image, crop_to_content, save_character_result
            
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


class InteractiveGUI:
    """インタラクティブ補助機能のGUIアプリケーション"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Phase 3: インタラクティブキャラクター抽出")
        self.root.geometry("1200x800")
        
        self.assistant = InteractiveAssistant()
        self.canvas = None
        self.image_tk = None
        self.scale_factor = 1.0
        
        # 描画状態
        self.drawing_region = False
        self.region_start = None
        self.current_rect = None
        
        self.setup_ui()
        self.initialize_models()
    
    def setup_ui(self):
        """UIセットアップ"""
        # メインフレーム
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左側: コントロールパネル
        control_frame = ttk.Frame(main_frame, width=250)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)
        
        # ファイル操作
        file_frame = ttk.LabelFrame(control_frame, text="ファイル操作")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_frame, text="画像を開く", command=self.open_image).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="抽出実行", command=self.extract_character).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="結果を保存", command=self.save_result).pack(fill=tk.X, pady=2)
        
        # シードポイント操作
        seed_frame = ttk.LabelFrame(control_frame, text="シードポイント")
        seed_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(seed_frame, text="左クリック: 正のポイント").pack(anchor=tk.W)
        ttk.Label(seed_frame, text="右クリック: 負のポイント").pack(anchor=tk.W)
        
        ttk.Button(seed_frame, text="最後のポイント削除", command=self.remove_last_point).pack(fill=tk.X, pady=2)
        ttk.Button(seed_frame, text="全ポイントクリア", command=self.clear_points).pack(fill=tk.X, pady=2)
        
        # 領域選択
        region_frame = ttk.LabelFrame(control_frame, text="領域選択")
        region_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(region_frame, text="Shift+ドラッグ: 領域選択").pack(anchor=tk.W)
        ttk.Button(region_frame, text="領域クリア", command=self.clear_region).pack(fill=tk.X, pady=2)
        
        # プレビュー操作
        preview_frame = ttk.LabelFrame(control_frame, text="プレビュー")
        preview_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(preview_frame, text="マスクプレビュー", command=self.preview_mask).pack(fill=tk.X, pady=2)
        ttk.Button(preview_frame, text="元画像表示", command=self.show_original).pack(fill=tk.X, pady=2)
        
        # ステータス
        self.status_var = tk.StringVar(value="画像を開いてください")
        status_frame = ttk.LabelFrame(control_frame, text="ステータス")
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(status_frame, textvariable=self.status_var, wraplength=200).pack(anchor=tk.W)
        
        # 右側: 画像表示エリア
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # スクロール可能なキャンバス
        canvas_frame = ttk.Frame(image_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg='gray90')
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        
        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # キャンバスイベント
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<Shift-Button-1>", self.on_shift_click)
        self.canvas.bind("<Shift-B1-Motion>", self.on_shift_drag)
        self.canvas.bind("<Shift-ButtonRelease-1>", self.on_shift_release)
    
    def initialize_models(self):
        """モデル初期化（別スレッドで実行）"""
        def init_models():
            try:
                self.status_var.set("モデル初期化中...")
                self.root.update()
                
                from hooks.start import get_sam_model, get_yolo_model
                sam_model = get_sam_model()
                yolo_model = get_yolo_model()
                
                if sam_model and yolo_model:
                    self.assistant.set_models(sam_model, yolo_model)
                    self.status_var.set("準備完了 - 画像を開いてください")
                else:
                    self.status_var.set("モデル初期化失敗")
                
            except Exception as e:
                self.status_var.set(f"初期化エラー: {e}")
        
        # 別スレッドで初期化
        threading.Thread(target=init_models, daemon=True).start()
    
    def open_image(self):
        """画像ファイルを開く"""
        file_path = filedialog.askopenfilename(
            title="画像ファイルを選択",
            filetypes=[
                ("画像ファイル", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("すべてのファイル", "*.*")
            ]
        )
        
        if file_path:
            if self.assistant.load_image(file_path):
                self.display_image()
                self.status_var.set(f"画像読み込み完了: {Path(file_path).name}")
            else:
                messagebox.showerror("エラー", "画像の読み込みに失敗しました")
    
    def display_image(self):
        """画像をキャンバスに表示"""
        if self.assistant.processed_image is None:
            return
        
        # OpenCV → PIL → Tkinter
        rgb_image = cv2.cvtColor(self.assistant.processed_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        self.image_tk = ImageTk.PhotoImage(pil_image)
        
        # キャンバスサイズ調整
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)
        
        # スケールファクター計算
        original_height, original_width = self.assistant.current_image.shape[:2]
        processed_height, processed_width = self.assistant.processed_image.shape[:2]
        self.scale_factor = original_width / processed_width
        
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.redraw_annotations()
    
    def redraw_annotations(self):
        """シードポイントと領域を再描画"""
        # 既存の注釈を削除
        self.canvas.delete("annotation")
        
        # シードポイントを描画
        for x, y, is_positive in self.assistant.seed_points:
            # スケール調整
            display_x = x / self.scale_factor
            display_y = y / self.scale_factor
            
            color = "green" if is_positive else "red"
            self.canvas.create_oval(
                display_x - 5, display_y - 5,
                display_x + 5, display_y + 5,
                outline=color, fill=color, width=2, tags="annotation"
            )
        
        # 選択領域を描画
        if self.assistant.selected_region:
            x, y, w, h = self.assistant.selected_region
            # スケール調整
            display_x = x / self.scale_factor
            display_y = y / self.scale_factor
            display_w = w / self.scale_factor
            display_h = h / self.scale_factor
            
            self.canvas.create_rectangle(
                display_x, display_y,
                display_x + display_w, display_y + display_h,
                outline="blue", width=2, tags="annotation"
            )
    
    def on_left_click(self, event):
        """左クリック: 正のシードポイント追加"""
        if self.assistant.current_image is None:
            return
        
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # 元画像座標に変換
        original_x = int(canvas_x * self.scale_factor)
        original_y = int(canvas_y * self.scale_factor)
        
        self.assistant.add_seed_point(original_x, original_y, True)
        self.redraw_annotations()
        self.status_var.set(f"正のポイント追加: ({original_x}, {original_y})")
    
    def on_right_click(self, event):
        """右クリック: 負のシードポイント追加"""
        if self.assistant.current_image is None:
            return
        
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # 元画像座標に変換
        original_x = int(canvas_x * self.scale_factor)
        original_y = int(canvas_y * self.scale_factor)
        
        self.assistant.add_seed_point(original_x, original_y, False)
        self.redraw_annotations()
        self.status_var.set(f"負のポイント追加: ({original_x}, {original_y})")
    
    def on_shift_click(self, event):
        """Shift+クリック: 領域選択開始"""
        if self.assistant.current_image is None:
            return
        
        self.drawing_region = True
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        self.region_start = (canvas_x, canvas_y)
        
        if self.current_rect:
            self.canvas.delete(self.current_rect)
    
    def on_shift_drag(self, event):
        """Shift+ドラッグ: 領域選択中"""
        if not self.drawing_region or not self.region_start:
            return
        
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        if self.current_rect:
            self.canvas.delete(self.current_rect)
        
        self.current_rect = self.canvas.create_rectangle(
            self.region_start[0], self.region_start[1],
            canvas_x, canvas_y,
            outline="blue", width=2, tags="temp"
        )
    
    def on_shift_release(self, event):
        """Shift+リリース: 領域選択完了"""
        if not self.drawing_region or not self.region_start:
            return
        
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # 元画像座標に変換
        x1 = int(min(self.region_start[0], canvas_x) * self.scale_factor)
        y1 = int(min(self.region_start[1], canvas_y) * self.scale_factor)
        x2 = int(max(self.region_start[0], canvas_x) * self.scale_factor)
        y2 = int(max(self.region_start[1], canvas_y) * self.scale_factor)
        
        w = x2 - x1
        h = y2 - y1
        
        if w > 10 and h > 10:  # 最小サイズチェック
            self.assistant.set_region(x1, y1, w, h)
            self.status_var.set(f"領域選択: ({x1}, {y1}, {w}, {h})")
        
        self.drawing_region = False
        self.region_start = None
        if self.current_rect:
            self.canvas.delete(self.current_rect)
            self.current_rect = None
        
        self.redraw_annotations()
    
    def remove_last_point(self):
        """最後のシードポイントを削除"""
        self.assistant.remove_last_seed_point()
        self.redraw_annotations()
        self.status_var.set("最後のポイントを削除しました")
    
    def clear_points(self):
        """全シードポイントをクリア"""
        self.assistant.clear_seed_points()
        self.redraw_annotations()
        self.status_var.set("全ポイントをクリアしました")
    
    def clear_region(self):
        """領域選択をクリア"""
        self.assistant.clear_region()
        self.redraw_annotations()
        self.status_var.set("領域選択をクリアしました")
    
    def preview_mask(self):
        """マスクプレビューを表示"""
        if self.assistant.current_image is None:
            return
        
        self.status_var.set("マスク生成中...")
        self.root.update()
        
        try:
            mask = None
            if self.assistant.seed_points:
                mask = self.assistant.generate_mask_with_seeds()
            elif self.assistant.selected_region:
                mask = self.assistant.generate_mask_with_region()
            
            if mask is not None:
                # マスクをオーバーレイして表示
                overlay = self.assistant.processed_image.copy()
                mask_resized = cv2.resize(mask, (overlay.shape[1], overlay.shape[0]))
                
                # マスク領域を半透明の緑で表示
                overlay[mask_resized > 0] = cv2.addWeighted(
                    overlay[mask_resized > 0], 0.7,
                    np.full_like(overlay[mask_resized > 0], [0, 255, 0]), 0.3,
                    0
                )
                
                # 表示更新
                rgb_overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                pil_overlay = Image.fromarray(rgb_overlay)
                self.image_tk = ImageTk.PhotoImage(pil_overlay)
                
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)
                self.redraw_annotations()
                
                self.status_var.set("マスクプレビュー表示中")
            else:
                self.status_var.set("マスク生成失敗")
                
        except Exception as e:
            self.status_var.set(f"プレビューエラー: {e}")
    
    def show_original(self):
        """元画像を表示"""
        self.display_image()
        self.status_var.set("元画像表示")
    
    def extract_character(self):
        """キャラクター抽出実行"""
        if self.assistant.current_image is None:
            messagebox.showwarning("警告", "画像が読み込まれていません")
            return
        
        if not self.assistant.seed_points and not self.assistant.selected_region:
            messagebox.showwarning("警告", "シードポイントまたは領域を指定してください")
            return
        
        self.status_var.set("抽出処理中...")
        self.root.update()
        
        try:
            result = self.assistant.extract_character_interactive()
            
            if result['success']:
                self.status_var.set(f"抽出完了: {result['output_path']}")
                messagebox.showinfo("成功", f"キャラクター抽出が完了しました\n{result['output_path']}")
            else:
                self.status_var.set("抽出失敗")
                messagebox.showerror("エラー", f"抽出失敗: {result['error']}")
                
        except Exception as e:
            self.status_var.set("抽出エラー")
            messagebox.showerror("エラー", f"抽出エラー: {e}")
    
    def save_result(self):
        """結果を保存"""
        output_path = filedialog.asksaveasfilename(
            title="保存先を選択",
            defaultextension=".jpg",
            filetypes=[
                ("JPEG画像", "*.jpg"),
                ("PNG画像", "*.png"),
                ("すべてのファイル", "*.*")
            ]
        )
        
        if output_path:
            try:
                result = self.assistant.extract_character_interactive(output_path)
                if result['success']:
                    messagebox.showinfo("成功", f"保存完了: {output_path}")
                else:
                    messagebox.showerror("エラー", f"保存失敗: {result['error']}")
            except Exception as e:
                messagebox.showerror("エラー", f"保存エラー: {e}")
    
    def run(self):
        """GUI実行"""
        self.root.mainloop()


def main():
    """メイン関数"""
    app = InteractiveGUI()
    app.run()


if __name__ == "__main__":
    main()