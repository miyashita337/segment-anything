#!/usr/bin/env python3
"""
ポーズランドマーク可視化テストスクリプト (Week 2)
MediaPipe Pose最適化の効果をビジュアルで確認

目的：
- ポーズ検出結果の視覚的確認
- ボーン描画による姿勢分析
- Week 2最適化の効果測定（38.9% → 80%目標）
"""

import numpy as np
import cv2

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from features.evaluation.enhanced_detection_systems import EnhancedPoseDetector, PoseDetectionResult

# MediaPipeインポート（利用可能な場合）
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️ MediaPipeが利用できません")


class PoseLandmarkVisualizer:
    """ポーズランドマーク可視化システム"""
    
    def __init__(self):
        self.pose_detector = EnhancedPoseDetector()
        
        # MediaPipe ポーズ接続定義（ボーン描画用）
        self.pose_connections = [
            # 顔部分
            (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6), (6, 8),
            
            # 肩から腕
            (9, 10),  # 口
            (11, 12),  # 肩
            (11, 13), (13, 15),  # 左腕
            (12, 14), (14, 16),  # 右腕
            
            # 上半身
            (11, 23), (12, 24),  # 肩から腰
            (23, 24),  # 腰
            
            # 脚
            (23, 25), (25, 27), (27, 29), (27, 31),  # 左脚
            (24, 26), (26, 28), (28, 30), (28, 32),  # 右脚
        ]
        
        # キーポイント名定義
        self.keypoint_names = [
            "鼻", "左目内側", "左目", "左目外側", "右目内側", "右目", "右目外側",
            "左耳", "右耳", "口左", "口右", "左肩", "右肩", "左肘", "右肘",
            "左手首", "右手首", "左小指", "右小指", "左人差し指", "右人差し指",
            "左親指", "右親指", "左股関節", "右股関節", "左膝", "右膝",
            "左足首", "右足首", "左踵", "右踵", "左足先", "右足先"
        ]
        
        # 部位別色分け
        self.colors = {
            'face': (0, 255, 255),      # 黄色 - 顔部分
            'arms': (0, 255, 0),        # 緑色 - 腕部分
            'torso': (255, 0, 0),       # 青色 - 上半身
            'legs': (255, 0, 255),      # マゼンタ - 下半身
            'hands': (0, 128, 255),     # オレンジ - 手部分
            'feet': (128, 0, 128),      # 紫色 - 足部分
        }
    
    def visualize_pose_landmarks(self, image_path: str, output_dir: str = "pose_analysis") -> Dict:
        """ポーズランドマーク可視化実行"""
        print(f"🎯 ポーズランドマーク可視化: {image_path}")
        
        # 出力ディレクトリ作成
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 画像読み込み
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 画像読み込み失敗: {image_path}")
            return {}
        
        # ポーズ検出実行
        print("🔍 ポーズ検出実行中...")
        pose_result = self.pose_detector.detect_pose_comprehensive(image)
        
        # 結果情報
        result_info = {
            'image_path': image_path,
            'timestamp': datetime.now().isoformat(),
            'detected': pose_result.detected,
            'pose_category': pose_result.pose_category,
            'visibility_score': pose_result.visibility_score,
            'completeness_score': pose_result.completeness_score,
            'confidence': pose_result.confidence,
            'keypoints_detected': pose_result.keypoints_detected,
            'week2_optimization': True
        }
        
        if not pose_result.detected:
            print("❌ ポーズ検出失敗")
            return result_info
        
        # 可視化画像作成
        vis_image = image.copy()
        
        # ランドマーク描画
        self._draw_landmarks(vis_image, pose_result.landmarks)
        
        # ボーン（接続線）描画
        self._draw_pose_connections(vis_image, pose_result.landmarks)
        
        # 情報テキスト描画
        self._draw_info_text(vis_image, pose_result)
        
        # 保存
        input_name = Path(image_path).stem
        output_file = output_path / f"{input_name}_pose_landmarks.jpg"
        cv2.imwrite(str(output_file), vis_image)
        
        print(f"✅ 可視化結果保存: {output_file}")
        print(f"   - 検出: {pose_result.detected}")
        print(f"   - カテゴリ: {pose_result.pose_category}")
        print(f"   - 可視性: {pose_result.visibility_score:.3f}")
        print(f"   - 信頼度: {pose_result.confidence:.3f}")
        print(f"   - キーポイント: {pose_result.keypoints_detected}/33")
        
        result_info['output_file'] = str(output_file)
        return result_info
    
    def _draw_landmarks(self, image: np.ndarray, landmarks) -> None:
        """ランドマーク点を描画"""
        if not landmarks or not landmarks.landmark:
            return
        
        h, w = image.shape[:2]
        
        for idx, landmark in enumerate(landmarks.landmark):
            if landmark.visibility < 0.2:  # Week 2最適化: 緩和された閾値
                continue
            
            # 座標変換
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            
            # 部位別色分け
            color = self._get_keypoint_color(idx)
            
            # 可視性に応じて円のサイズ調整
            radius = int(3 + landmark.visibility * 5)
            
            # ランドマーク点描画（外側）
            cv2.circle(image, (x, y), radius, color, -1)
            # ランドマーク点描画（内側・白）
            cv2.circle(image, (x, y), max(1, radius-2), (255, 255, 255), -1)
            
            # キーポイント番号表示（重要なポイントのみ）
            if idx in [11, 12, 13, 14, 15, 16, 23, 24]:  # 肩、肘、手首、腰
                cv2.putText(image, str(idx), (x+8, y-8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    def _draw_pose_connections(self, image: np.ndarray, landmarks) -> None:
        """ポーズ接続線（ボーン）を描画"""
        if not landmarks or not landmarks.landmark:
            return
        
        h, w = image.shape[:2]
        
        for start_idx, end_idx in self.pose_connections:
            if start_idx >= len(landmarks.landmark) or end_idx >= len(landmarks.landmark):
                continue
            
            start_landmark = landmarks.landmark[start_idx]
            end_landmark = landmarks.landmark[end_idx]
            
            # 両端点が可視の場合のみ描画
            if start_landmark.visibility < 0.2 or end_landmark.visibility < 0.2:
                continue
            
            # 座標変換
            start_x = int(start_landmark.x * w)
            start_y = int(start_landmark.y * h)
            end_x = int(end_landmark.x * w)
            end_y = int(end_landmark.y * h)
            
            # 接続部位に応じた色選択
            color = self._get_connection_color(start_idx, end_idx)
            
            # 可視性に応じて線の太さ調整
            thickness = max(1, int((start_landmark.visibility + end_landmark.visibility) * 2))
            
            # ボーン描画
            cv2.line(image, (start_x, start_y), (end_x, end_y), color, thickness)
    
    def _get_keypoint_color(self, keypoint_idx: int) -> Tuple[int, int, int]:
        """キーポイント別色分け"""
        if keypoint_idx <= 10:  # 顔部分
            return self.colors['face']
        elif keypoint_idx in [11, 12]:  # 肩
            return self.colors['torso']
        elif keypoint_idx in [13, 14, 15, 16]:  # 腕
            return self.colors['arms']
        elif keypoint_idx in [17, 18, 19, 20, 21, 22]:  # 手
            return self.colors['hands']
        elif keypoint_idx in [23, 24]:  # 腰
            return self.colors['torso']
        elif keypoint_idx in [25, 26, 27, 28]:  # 脚
            return self.colors['legs']
        else:  # 足
            return self.colors['feet']
    
    def _get_connection_color(self, start_idx: int, end_idx: int) -> Tuple[int, int, int]:
        """接続線の色選択"""
        # 顔部分
        if max(start_idx, end_idx) <= 10:
            return self.colors['face']
        # 腕部分
        elif any(idx in [11, 12, 13, 14, 15, 16] for idx in [start_idx, end_idx]):
            return self.colors['arms']
        # 上半身
        elif any(idx in [11, 12, 23, 24] for idx in [start_idx, end_idx]):
            return self.colors['torso']
        # 下半身
        else:
            return self.colors['legs']
    
    def _draw_info_text(self, image: np.ndarray, pose_result: PoseDetectionResult) -> None:
        """情報テキストを描画"""
        h, w = image.shape[:2]
        
        # 背景矩形
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (400, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # テキスト情報
        texts = [
            f"Week 2 MediaPipe Pose最適化結果",
            f"カテゴリ: {pose_result.pose_category}",
            f"可視性スコア: {pose_result.visibility_score:.3f}",
            f"完全性スコア: {pose_result.completeness_score:.3f}",
            f"総合信頼度: {pose_result.confidence:.3f}",
            f"検出キーポイント: {pose_result.keypoints_detected}/33"
        ]
        
        for i, text in enumerate(texts):
            y_pos = 30 + i * 18
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            font_scale = 0.6 if i == 0 else 0.5
            cv2.putText(image, text, (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)


def test_batch_images(input_dir: str, output_dir: str = "pose_analysis") -> Dict:
    """バッチ画像での可視化テスト"""
    print(f"📁 バッチポーズ可視化テスト: {input_dir}")
    
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ 入力ディレクトリが存在しません: {input_dir}")
        return {}
    
    # 画像ファイル取得
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"❌ 画像ファイルが見つかりません: {input_dir}")
        return {}
    
    print(f"🎯 {len(image_files)}個の画像を処理します")
    
    # 可視化実行
    visualizer = PoseLandmarkVisualizer()
    results = []
    
    detection_count = 0
    
    for i, image_file in enumerate(image_files[:10]):  # 最大10枚
        print(f"\n📸 [{i+1}/{min(len(image_files), 10)}] {image_file.name}")
        
        result = visualizer.visualize_pose_landmarks(str(image_file), output_dir)
        results.append(result)
        
        if result.get('detected', False):
            detection_count += 1
    
    # 統計情報
    total_processed = len(results)
    detection_rate = detection_count / total_processed if total_processed > 0 else 0
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'input_directory': input_dir,
        'output_directory': output_dir,
        'total_processed': total_processed,
        'detection_count': detection_count,
        'detection_rate': detection_rate,
        'week2_target_rate': 0.80,
        'target_achieved': detection_rate >= 0.80,
        'results': results
    }
    
    print(f"\n" + "="*60)
    print(f"📊 Week 2 MediaPipe Pose最適化 - バッチテスト結果")
    print(f"="*60)
    print(f"処理画像数: {total_processed}")
    print(f"検出成功数: {detection_count}")
    print(f"検出率: {detection_rate:.1%}")
    print(f"Week 2目標(80%): {'✅ 達成' if summary['target_achieved'] else '❌ 未達成'}")
    
    if detection_rate < 0.80:
        improvement_needed = 0.80 - detection_rate
        print(f"改善必要: +{improvement_needed:.1%}")
    
    # JSON保存
    output_path = Path(output_dir)
    report_file = output_path / f"pose_landmark_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"📄 詳細レポート保存: {report_file}")
    
    return summary


def main():
    """メイン処理"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ポーズランドマーク可視化テスト (Week 2)")
    parser.add_argument("--image", "-i", help="単一画像のテスト")
    parser.add_argument("--batch", "-b", 
                       default="/mnt/c/AItools/lora/train/yado/clipped_boundingbox/kana08_rev_merge",
                       help="バッチ処理ディレクトリ")
    parser.add_argument("--output", "-o", default="pose_analysis", help="出力ディレクトリ")
    
    args = parser.parse_args()
    
    if not MEDIAPIPE_AVAILABLE:
        print("❌ MediaPipeが必要です。インストールしてください：pip install mediapipe")
        return 1
    
    print("🎯 Week 2: MediaPipe Pose最適化 - ランドマーク可視化テスト")
    print("=" * 80)
    
    if args.image:
        # 単一画像テスト
        visualizer = PoseLandmarkVisualizer()
        result = visualizer.visualize_pose_landmarks(args.image, args.output)
        return 0 if result.get('detected', False) else 1
    else:
        # バッチテスト
        summary = test_batch_images(args.batch, args.output)
        return 0 if summary.get('target_achieved', False) else 1


if __name__ == "__main__":
    exit(main())