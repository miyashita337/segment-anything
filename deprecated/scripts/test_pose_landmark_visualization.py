#!/usr/bin/env python3
"""
ランドマーク可視化テストシステム（Week 2）
MediaPipe ポーズランドマークのボーン描画と姿勢分析
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
import cv2

from features.evaluation.enhanced_detection_systems import EnhancedPoseDetector, PoseDetectionResult

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PoseLandmarkVisualizer:
    """ポーズランドマーク可視化システム"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.PoseLandmarkVisualizer")
        self.pose_detector = EnhancedPoseDetector()
        
        # MediaPipe ポーズランドマーク接続定義（33点）
        self.pose_connections = [
            # 顔輪郭
            (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6), (6, 8),
            # 上半身
            (9, 10),  # 口
            (11, 12), # 肩
            (11, 13), (13, 15),  # 左腕
            (12, 14), (14, 16),  # 右腕
            (11, 23), (12, 24),  # 肩から腰
            (23, 24), # 腰
            # 下半身
            (23, 25), (25, 27), (27, 29), (27, 31),  # 左脚
            (24, 26), (26, 28), (28, 30), (28, 32),  # 右脚
        ]
        
        # キーポイント名称定義
        self.keypoint_names = {
            0: 'nose', 1: 'left_eye_inner', 2: 'left_eye', 3: 'left_eye_outer',
            4: 'right_eye_inner', 5: 'right_eye', 6: 'right_eye_outer',
            7: 'left_ear', 8: 'right_ear', 9: 'mouth_left', 10: 'mouth_right',
            11: 'left_shoulder', 12: 'right_shoulder', 13: 'left_elbow', 14: 'right_elbow',
            15: 'left_wrist', 16: 'right_wrist', 17: 'left_pinky', 18: 'right_pinky',
            19: 'left_index', 20: 'right_index', 21: 'left_thumb', 22: 'right_thumb',
            23: 'left_hip', 24: 'right_hip', 25: 'left_knee', 26: 'right_knee',
            27: 'left_ankle', 28: 'right_ankle', 29: 'left_heel', 30: 'right_heel',
            31: 'left_foot_index', 32: 'right_foot_index'
        }
        
        # Week 2最適化: 部分重要度色分け
        self.keypoint_colors = {
            # 上半身（重要）: 赤系
            11: (0, 0, 255), 12: (0, 0, 255),  # 肩
            13: (0, 50, 255), 14: (0, 50, 255),  # 肘
            15: (0, 100, 255), 16: (0, 100, 255),  # 手首
            # 頭部（重要）: 青系
            0: (255, 0, 0), 1: (255, 50, 0), 2: (255, 100, 0), 3: (255, 150, 0),
            4: (255, 150, 0), 5: (255, 100, 0), 6: (255, 50, 0), 7: (255, 0, 50), 8: (255, 0, 50),
            9: (200, 0, 100), 10: (200, 0, 100),
            # 下半身（補助）: 緑系
            23: (0, 255, 0), 24: (0, 255, 0),  # 腰
            25: (0, 200, 50), 26: (0, 200, 50),  # 膝
            27: (0, 150, 100), 28: (0, 150, 100),  # 足首
        }
    
    def visualize_pose_landmarks(self, image: np.ndarray, pose_result: PoseDetectionResult) -> np.ndarray:
        """ポーズランドマークの可視化"""
        if not pose_result.detected or not pose_result.landmarks:
            return self._draw_no_detection_message(image)
        
        # 画像をコピー
        output_image = image.copy()
        height, width = output_image.shape[:2]
        
        # ランドマーク座標を画像座標に変換
        landmark_points = []
        for i, landmark in enumerate(pose_result.landmarks.landmark):
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            visibility = landmark.visibility
            landmark_points.append((x, y, visibility))
        
        # ボーン（接続線）を描画
        self._draw_pose_connections(output_image, landmark_points)
        
        # キーポイントを描画
        self._draw_keypoints(output_image, landmark_points)
        
        # 分析情報をオーバーレイ
        self._draw_analysis_overlay(output_image, pose_result)
        
        return output_image
    
    def _draw_pose_connections(self, image: np.ndarray, landmark_points: List[Tuple[int, int, float]]):
        """ポーズ接続線の描画"""
        for connection in self.pose_connections:
            start_idx, end_idx = connection
            
            if start_idx < len(landmark_points) and end_idx < len(landmark_points):
                start_point = landmark_points[start_idx]
                end_point = landmark_points[end_idx]
                
                # 両端点の可視性チェック
                if start_point[2] > 0.2 and end_point[2] > 0.2:  # Week 2最適化: 緩和された閾値
                    # 可視性に応じた線の太さと透明度
                    visibility = min(start_point[2], end_point[2])
                    thickness = max(1, int(visibility * 4))
                    
                    # 線の色（接続の種類による）
                    if start_idx in [11, 12, 13, 14, 15, 16] or end_idx in [11, 12, 13, 14, 15, 16]:
                        color = (0, 0, 255)  # 上半身: 赤
                    elif start_idx in range(11) or end_idx in range(11):
                        color = (255, 0, 0)  # 頭部: 青
                    else:
                        color = (0, 255, 0)  # 下半身: 緑
                    
                    cv2.line(image, (start_point[0], start_point[1]), 
                            (end_point[0], end_point[1]), color, thickness)
    
    def _draw_keypoints(self, image: np.ndarray, landmark_points: List[Tuple[int, int, float]]):
        """キーポイントの描画"""
        for i, (x, y, visibility) in enumerate(landmark_points):
            if visibility > 0.2:  # Week 2最適化: 緩和された閾値
                # 可視性に応じた円のサイズ
                radius = max(2, int(visibility * 8))
                
                # キーポイント別の色
                color = self.keypoint_colors.get(i, (128, 128, 128))
                
                # キーポイント描画
                cv2.circle(image, (x, y), radius, color, -1)
                
                # Week 2最適化: 重要キーポイントにラベル表示
                if i in [11, 12, 13, 14, 15, 16]:  # 上半身主要部位
                    keypoint_name = self.keypoint_names.get(i, f"point_{i}")
                    cv2.putText(image, keypoint_name[:4], (x + 5, y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    def _draw_analysis_overlay(self, image: np.ndarray, pose_result: PoseDetectionResult):
        """分析情報オーバーレイ"""
        height, width = image.shape[:2]
        
        # 背景矩形
        overlay_height = 180
        cv2.rectangle(image, (10, 10), (width - 10, overlay_height), (0, 0, 0), -1)
        cv2.rectangle(image, (10, 10), (width - 10, overlay_height), (255, 255, 255), 2)
        
        # 分析結果テキスト
        texts = [
            f"Pose Category: {pose_result.pose_category}",
            f"Keypoints Detected: {pose_result.keypoints_detected}/33",
            f"Visibility Score: {pose_result.visibility_score:.3f}",
            f"Completeness Score: {pose_result.completeness_score:.3f}",
            f"Confidence: {pose_result.confidence:.3f}",
            f"Detection Status: {'SUCCESS' if pose_result.detected else 'FAILED'}"
        ]
        
        for i, text in enumerate(texts):
            y_pos = 30 + i * 25
            color = (0, 255, 0) if pose_result.detected else (0, 0, 255)
            cv2.putText(image, text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def _draw_no_detection_message(self, image: np.ndarray) -> np.ndarray:
        """検出失敗時のメッセージ描画"""
        output_image = image.copy()
        height, width = output_image.shape[:2]
        
        # 赤い背景矩形
        cv2.rectangle(output_image, (10, 10), (width - 10, 100), (0, 0, 255), -1)
        cv2.rectangle(output_image, (10, 10), (width - 10, 100), (255, 255, 255), 2)
        
        # エラーメッセージ
        cv2.putText(output_image, "NO POSE DETECTED", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(output_image, "Week 2 Optimization Needed", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return output_image


class PoseLandmarkVisualizationTest:
    """ランドマーク可視化テスト実行システム"""
    
    def __init__(self):
        self.visualizer = PoseLandmarkVisualizer()
        self.pose_detector = EnhancedPoseDetector()
        self.test_datasets = {
            "kana05": "/mnt/c/AItools/lora/train/yado/org/kana05_cursor_fix",
            "kana07": "/mnt/c/AItools/lora/train/yado/org/kana07_cursor_fix", 
            "kana08": "/mnt/c/AItools/lora/train/yado/org/kana08_cursor_fix"
        }
    
    def run_visualization_test(self, output_dir: str = "pose_analysis") -> Dict:
        """ランドマーク可視化テスト実行"""
        logger.info("🎨 ポーズランドマーク可視化テスト開始")
        
        # 出力ディレクトリ作成
        os.makedirs(output_dir, exist_ok=True)
        
        all_results = {}
        total_images = 0
        total_poses_detected = 0
        detection_details = []
        
        for dataset_name, dataset_path in self.test_datasets.items():
            if not os.path.exists(dataset_path):
                logger.warning(f"データセット未発見: {dataset_path}")
                continue
            
            logger.info(f"📂 可視化テスト実行: {dataset_name}")
            dataset_results = self.test_dataset_visualization(dataset_path, dataset_name, output_dir)
            all_results[dataset_name] = dataset_results
            
            total_images += dataset_results['image_count']
            total_poses_detected += dataset_results['poses_detected']
            detection_details.extend(dataset_results['detection_details'])
        
        # 総合統計計算
        overall_stats = {
            'total_images_processed': total_images,
            'total_poses_detected': total_poses_detected,
            'overall_pose_detection_rate': total_poses_detected / total_images if total_images > 0 else 0.0,
            'week2_target_achievement': total_poses_detected / total_images >= 0.8 if total_images > 0 else False,
            'detection_details': detection_details
        }
        
        # Week 2最適化効果レポート生成
        self.generate_week2_optimization_report(all_results, overall_stats, output_dir)
        
        return {
            'dataset_results': all_results,
            'overall_statistics': overall_stats,
            'test_completion_time': datetime.now().isoformat(),
            'output_directory': output_dir
        }
    
    def test_dataset_visualization(self, dataset_path: str, dataset_name: str, output_dir: str) -> Dict:
        """単一データセット可視化テスト"""
        image_files = [f for f in os.listdir(dataset_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png')) 
                      and not f.endswith('_gt.png')]
        
        results = {
            'dataset_name': dataset_name,
            'image_count': len(image_files),
            'poses_detected': 0,
            'detection_details': [],
            'visualization_files': []
        }
        
        dataset_output_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        for image_file in image_files:
            image_path = os.path.join(dataset_path, image_file)
            
            try:
                # 画像読み込み
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                # Week 2最適化ポーズ検出実行（効率モード）
                pose_result = self.pose_detector.detect_pose_comprehensive(image, efficient_mode=True)
                
                # ランドマーク可視化
                visualization_image = self.visualizer.visualize_pose_landmarks(image, pose_result)
                
                # 結果保存
                base_name = os.path.splitext(image_file)[0]
                output_filename = f"{base_name}_pose_analysis.jpg"
                output_path = os.path.join(dataset_output_dir, output_filename)
                cv2.imwrite(output_path, visualization_image)
                
                # 統計記録
                if pose_result.detected:
                    results['poses_detected'] += 1
                
                detection_detail = {
                    'image_file': image_file,
                    'detected': pose_result.detected,
                    'pose_category': pose_result.pose_category,
                    'keypoints_detected': pose_result.keypoints_detected,
                    'visibility_score': pose_result.visibility_score,
                    'completeness_score': pose_result.completeness_score,
                    'confidence': pose_result.confidence,
                    'output_file': output_filename
                }
                
                results['detection_details'].append(detection_detail)
                results['visualization_files'].append(output_path)
                
                logger.debug(f"  {image_file}: {'SUCCESS' if pose_result.detected else 'FAILED'} "
                           f"({pose_result.keypoints_detected} keypoints)")
            
            except Exception as e:
                logger.error(f"可視化エラー {image_file}: {e}")
        
        return results
    
    def generate_week2_optimization_report(self, all_results: Dict, overall_stats: Dict, output_dir: str):
        """Week 2最適化効果レポート生成"""
        
        print("\n" + "=" * 80)
        print("🎨 Week 2 ポーズランドマーク可視化テスト結果")
        print("=" * 80)
        
        print(f"\n📊 総合統計:")
        print(f"  処理画像数: {overall_stats['total_images_processed']}枚")
        print(f"  ポーズ検出数: {overall_stats['total_poses_detected']}件")
        print(f"  検出率: {overall_stats['overall_pose_detection_rate']:.1%}")
        print(f"  Week 2目標達成: {'✅ YES' if overall_stats['week2_target_achievement'] else '❌ NO'} (目標80%)")
        
        print(f"\n📈 データセット別詳細:")
        for dataset_name, results in all_results.items():
            detection_rate = (results['poses_detected'] / results['image_count']) if results['image_count'] > 0 else 0
            
            print(f"  {dataset_name}:")
            print(f"    画像数: {results['image_count']}枚")
            print(f"    ポーズ検出数: {results['poses_detected']}件")
            print(f"    検出率: {detection_rate:.1%}")
            print(f"    可視化ファイル: {len(results['visualization_files'])}件生成")
        
        # Week 2最適化効果分析
        detection_categories = {}
        keypoint_stats = {'min': 33, 'max': 0, 'total': 0, 'count': 0}
        
        for detail in overall_stats['detection_details']:
            if detail['detected']:
                category = detail['pose_category']
                if category not in detection_categories:
                    detection_categories[category] = 0
                detection_categories[category] += 1
                
                keypoints = detail['keypoints_detected']
                keypoint_stats['min'] = min(keypoint_stats['min'], keypoints)
                keypoint_stats['max'] = max(keypoint_stats['max'], keypoints)
                keypoint_stats['total'] += keypoints
                keypoint_stats['count'] += 1
        
        print(f"\n🎯 Week 2最適化効果:")
        print(f"  新ポーズカテゴリ検出:")
        for category, count in detection_categories.items():
            emoji = "🆕" if category in ['partial_pose', 'upper_body_only'] else "✅"
            print(f"    {emoji} {category}: {count}件")
        
        if keypoint_stats['count'] > 0:
            avg_keypoints = keypoint_stats['total'] / keypoint_stats['count']
            print(f"  キーポイント統計:")
            print(f"    平均検出数: {avg_keypoints:.1f}点")
            print(f"    最小検出数: {keypoint_stats['min']}点（Week 2目標: 3点以上）")
            print(f"    最大検出数: {keypoint_stats['max']}点")
        
        # 出力ディレクトリ情報
        print(f"\n💾 可視化結果:")
        print(f"  出力ディレクトリ: {output_dir}/")
        print(f"  総ファイル数: {sum(len(r['visualization_files']) for r in all_results.values())}件")
        
        target_rate = 0.80
        current_rate = overall_stats['overall_pose_detection_rate']
        
        if current_rate >= target_rate:
            print(f"\n🎉 Week 2目標達成! ポーズ検出率{current_rate:.1%} >= 目標80%")
        else:
            improvement_needed = target_rate - current_rate
            print(f"\n📋 追加改善必要: あと{improvement_needed:+.1%}の向上が必要")
        
        print("\n" + "=" * 80)
        
        # JSON形式でも保存
        report_data = {
            'overall_statistics': overall_stats,
            'dataset_results': all_results,
            'optimization_analysis': {
                'detection_categories': detection_categories,
                'keypoint_statistics': keypoint_stats,
                'week2_target_achievement': overall_stats['week2_target_achievement']
            }
        }
        
        report_file = os.path.join(output_dir, f"week2_pose_optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        print(f"📄 詳細レポート保存: {report_file}")


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ポーズランドマーク可視化テスト")
    parser.add_argument("--output", "-o", default="pose_analysis", help="出力ディレクトリ")
    
    args = parser.parse_args()
    
    print("🎨 Week 2 ポーズランドマーク可視化テスト開始")
    
    try:
        tester = PoseLandmarkVisualizationTest()
        results = tester.run_visualization_test(args.output)
        
        print(f"\n✅ Week 2 ポーズランドマーク可視化テスト完了")
        print(f"💾 結果保存: {args.output}/")
        
        return 0
    
    except Exception as e:
        logger.error(f"テスト実行エラー: {e}")
        return 1


if __name__ == "__main__":
    exit(main())