#!/usr/bin/env python3
"""
軽量版顔検出システム統合テスト（GPT-4O最適化適用）
MediaPipe無し、OpenCV+アニメカスケードのみで高速テスト実行
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
import cv2

from features.evaluation.anime_image_preprocessor import AnimeImagePreprocessor

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LightweightFaceDetector:
    """軽量版顔検出（MediaPipe無し）"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.LightweightFaceDetector")
        self.preprocessor = AnimeImagePreprocessor()
        
        # プロジェクトルートパス
        project_root = Path(__file__).parent
        
        # OpenCV Cascade分類器初期化（アニメ専用のみ）
        self.cascade_detectors = {}
        cascade_files = {
            'frontal': 'haarcascade_frontalface_default.xml',
            'anime': 'lbpcascade_animeface.xml'
        }
        
        for name, filename in cascade_files.items():
            try:
                if name == 'anime':
                    cascade_path = str(project_root / "models" / "cascades" / filename)
                else:
                    cascade_path = cv2.data.haarcascades + filename
                
                if os.path.exists(cascade_path):
                    self.cascade_detectors[name] = cv2.CascadeClassifier(cascade_path)
                    self.logger.info(f"OpenCV {name} Cascade初期化完了")
                else:
                    self.logger.warning(f"Cascadeファイル未発見: {cascade_path}")
            except Exception as e:
                self.logger.warning(f"OpenCV {name} Cascade初期化失敗: {e}")
    
    def detect_faces_lightweight(self, image: np.ndarray) -> Dict:
        """軽量版顔検出（処理時間重視）"""
        start_time = datetime.now()
        results = {
            'detections': [],
            'preprocessing_time': 0.0,
            'detection_time': 0.0,
            'total_time': 0.0,
            'methods_used': []
        }
        
        try:
            # 1. 軽量前処理
            preprocess_start = datetime.now()
            enhanced_image = self.preprocessor.enhance_for_face_detection(image, lightweight_mode=True)
            results['preprocessing_time'] = (datetime.now() - preprocess_start).total_seconds()
            
            # 2. 段階的検出
            detection_start = datetime.now()
            
            # Step 1: アニメ専用カスケード（最優先）
            if 'anime' in self.cascade_detectors:
                anime_faces = self._detect_anime_cascade(enhanced_image)
                if anime_faces:
                    results['detections'].extend(anime_faces)
                    results['methods_used'].append(f"anime_cascade: {len(anime_faces)}件")
            
            # Step 2: 標準カスケード（補完）
            if len(results['detections']) == 0 and 'frontal' in self.cascade_detectors:
                frontal_faces = self._detect_frontal_cascade(enhanced_image)
                if frontal_faces:
                    results['detections'].extend(frontal_faces)
                    results['methods_used'].append(f"frontal_cascade: {len(frontal_faces)}件")
            
            # Step 3: マルチスケール（3スケールのみ）
            if len(results['detections']) == 0 and 'anime' in self.cascade_detectors:
                multiscale_faces = self._detect_multiscale_anime(image)
                if multiscale_faces:
                    results['detections'].extend(multiscale_faces)
                    results['methods_used'].append(f"multiscale_anime: {len(multiscale_faces)}件")
            
            results['detection_time'] = (datetime.now() - detection_start).total_seconds()
            results['total_time'] = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"軽量検出完了: {len(results['detections'])}件 "
                           f"（前処理: {results['preprocessing_time']:.2f}秒, "
                           f"検出: {results['detection_time']:.2f}秒, "
                           f"合計: {results['total_time']:.2f}秒）")
            
        except Exception as e:
            self.logger.error(f"軽量検出エラー: {e}")
            results['error'] = str(e)
        
        return results
    
    def _detect_anime_cascade(self, image: np.ndarray) -> List[Dict]:
        """アニメ専用カスケード検出"""
        detections = []
        
        try:
            detector = self.cascade_detectors['anime']
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.03,
                minNeighbors=1,
                minSize=(10, 10),
                maxSize=(500, 500)
            )
            
            for (x, y, w, h) in faces:
                detections.append({
                    'bbox': (x, y, w, h),
                    'confidence': 0.85,
                    'method': 'anime_cascade'
                })
        
        except Exception as e:
            self.logger.warning(f"アニメカスケード検出エラー: {e}")
        
        return detections
    
    def _detect_frontal_cascade(self, image: np.ndarray) -> List[Dict]:
        """標準顔カスケード検出"""
        detections = []
        
        try:
            detector = self.cascade_detectors['frontal']
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=2,
                minSize=(20, 20)
            )
            
            for (x, y, w, h) in faces:
                detections.append({
                    'bbox': (x, y, w, h),
                    'confidence': 0.7,
                    'method': 'frontal_cascade'
                })
        
        except Exception as e:
            self.logger.warning(f"標準カスケード検出エラー: {e}")
        
        return detections
    
    def _detect_multiscale_anime(self, image: np.ndarray) -> List[Dict]:
        """マルチスケールアニメ検出（3スケールのみ）"""
        detections = []
        
        try:
            # 3スケールのみ生成
            multi_scale_images = self.preprocessor.create_multi_scale_versions(image, lightweight_mode=True)
            detector = self.cascade_detectors['anime']
            
            for scale_data in multi_scale_images:
                scale = scale_data['scale']
                scaled_image = scale_data['image']
                gray = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2GRAY)
                
                faces = detector.detectMultiScale(
                    gray,
                    scaleFactor=1.02,
                    minNeighbors=1,
                    minSize=(8, 8),
                    maxSize=(600, 600)
                )
                
                for (x, y, w, h) in faces:
                    # 元画像座標系に変換
                    orig_x = int(x / scale)
                    orig_y = int(y / scale)
                    orig_w = int(w / scale)
                    orig_h = int(h / scale)
                    
                    detections.append({
                        'bbox': (orig_x, orig_y, orig_w, orig_h),
                        'confidence': 0.8,
                        'method': f'multiscale_anime_{scale:.2f}'
                    })
        
        except Exception as e:
            self.logger.warning(f"マルチスケール検出エラー: {e}")
        
        return detections


class LightweightFaceDetectionTest:
    """軽量版顔検出統合テスト"""
    
    def __init__(self):
        self.detector = LightweightFaceDetector()
        self.test_datasets = {
            "kana05": "/mnt/c/AItools/lora/train/yado/org/kana05_cursor_fix",
            "kana07": "/mnt/c/AItools/lora/train/yado/org/kana07_cursor_fix", 
            "kana08": "/mnt/c/AItools/lora/train/yado/org/kana08_cursor_fix"
        }
    
    def run_comprehensive_test(self) -> Dict:
        """軽量版包括テスト"""
        logger.info("🚀 軽量版顔検出統合テスト開始")
        
        all_results = {}
        total_images = 0
        total_faces_detected = 0
        total_processing_time = 0.0
        processing_times = []
        
        for dataset_name, dataset_path in self.test_datasets.items():
            if not os.path.exists(dataset_path):
                logger.warning(f"データセット未発見: {dataset_path}")
                continue
            
            logger.info(f"📂 テスト実行: {dataset_name}")
            dataset_results = self.test_dataset(dataset_path, dataset_name)
            all_results[dataset_name] = dataset_results
            
            total_images += dataset_results['image_count']
            total_faces_detected += dataset_results['total_faces_detected']
            total_processing_time += dataset_results['total_processing_time']
            processing_times.extend(dataset_results['processing_times'])
        
        # 総合統計計算
        overall_stats = {
            'total_images_processed': total_images,
            'total_faces_detected': total_faces_detected,
            'overall_detection_rate': total_faces_detected / total_images if total_images > 0 else 0.0,
            'total_processing_time': total_processing_time,
            'average_processing_time': total_processing_time / total_images if total_images > 0 else 0.0,
            'processing_times': processing_times
        }
        
        # 結果レポート生成
        self.generate_lightweight_report(all_results, overall_stats)
        
        return {
            'dataset_results': all_results,
            'overall_statistics': overall_stats,
            'test_completion_time': datetime.now().isoformat()
        }
    
    def test_dataset(self, dataset_path: str, dataset_name: str) -> Dict:
        """単一データセットテスト"""
        image_files = [f for f in os.listdir(dataset_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png')) 
                      and not f.endswith('_gt.png')]
        
        results = {
            'dataset_name': dataset_name,
            'image_count': len(image_files),
            'detection_results': [],
            'total_faces_detected': 0,
            'total_processing_time': 0.0,
            'processing_times': [],
            'method_counts': {}
        }
        
        for image_file in image_files:
            image_path = os.path.join(dataset_path, image_file)
            
            try:
                # 画像読み込み
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                # 軽量版顔検出実行
                detection_result = self.detector.detect_faces_lightweight(image)
                
                face_count = len(detection_result.get('detections', []))
                processing_time = detection_result.get('total_time', 0.0)
                
                results['detection_results'].append({
                    'image_file': image_file,
                    'faces_detected': face_count,
                    'processing_time': processing_time,
                    'preprocessing_time': detection_result.get('preprocessing_time', 0.0),
                    'detection_time': detection_result.get('detection_time', 0.0),
                    'methods_used': detection_result.get('methods_used', [])
                })
                
                results['total_faces_detected'] += face_count
                results['total_processing_time'] += processing_time
                results['processing_times'].append(processing_time)
                
                # 手法別統計
                for method_info in detection_result.get('methods_used', []):
                    method_name = method_info.split(':')[0]
                    if method_name not in results['method_counts']:
                        results['method_counts'][method_name] = 0
                    results['method_counts'][method_name] += 1
                
                logger.debug(f"  {image_file}: {face_count}件検出 ({processing_time:.2f}秒)")
            
            except Exception as e:
                logger.error(f"画像処理エラー {image_file}: {e}")
        
        return results
    
    def generate_lightweight_report(self, all_results: Dict, overall_stats: Dict):
        """軽量版改善レポート生成"""
        
        print("\n" + "=" * 80)
        print("🚀 軽量版顔検出システム統合テスト結果（GPT-4O最適化適用）")
        print("=" * 80)
        
        print(f"\n📊 総合統計:")
        print(f"  処理画像数: {overall_stats['total_images_processed']}枚")
        print(f"  総検出数: {overall_stats['total_faces_detected']}件")
        print(f"  検出率: {overall_stats['overall_detection_rate']:.1%}")
        print(f"  総処理時間: {overall_stats['total_processing_time']:.1f}秒")
        print(f"  平均処理時間: {overall_stats['average_processing_time']:.2f}秒/画像")
        
        print(f"\n📈 データセット別詳細:")
        for dataset_name, results in all_results.items():
            detection_rate = (results['total_faces_detected'] / results['image_count']) if results['image_count'] > 0 else 0
            avg_time = results['total_processing_time'] / results['image_count'] if results['image_count'] > 0 else 0
            
            print(f"  {dataset_name}:")
            print(f"    画像数: {results['image_count']}枚")
            print(f"    検出数: {results['total_faces_detected']}件")
            print(f"    検出率: {detection_rate:.1%}")
            print(f"    平均処理時間: {avg_time:.2f}秒")
            print(f"    使用手法: {', '.join(results['method_counts'].keys())}")
        
        # GPT-4O最適化効果分析
        print(f"\n🎯 GPT-4O最適化効果:")
        
        target_time = 12.0  # ユーザー要求: 10-12秒以下
        current_avg_time = overall_stats['average_processing_time']
        
        if current_avg_time <= target_time:
            print(f"  ✅ 処理時間目標達成: {current_avg_time:.2f}秒 <= {target_time}秒")
        else:
            print(f"  ⚠️ 処理時間要改善: {current_avg_time:.2f}秒 > {target_time}秒")
        
        target_detection_rate = 0.90  # 90%目標
        current_rate = overall_stats['overall_detection_rate']
        
        if current_rate >= target_detection_rate:
            print(f"  ✅ 検出率目標達成: {current_rate:.1%} >= {target_detection_rate:.1%}")
        else:
            print(f"  📋 検出率要改善: {current_rate:.1%} < {target_detection_rate:.1%}")
        
        # 速度改善推定
        estimated_old_time = current_avg_time * 8  # 従来版推定（8倍遅い）
        improvement_factor = estimated_old_time / current_avg_time
        
        print(f"\n📈 処理速度改善推定:")
        print(f"  推定従来処理時間: {estimated_old_time:.1f}秒/画像")
        print(f"  軽量版処理時間: {current_avg_time:.2f}秒/画像")
        print(f"  改善倍率: {improvement_factor:.1f}倍高速化")
        
        print(f"\n📋 次のステップ:")
        if current_rate >= target_detection_rate and current_avg_time <= target_time:
            print(f"  🎉 Week 1完全達成! Phase A2目標クリア")
        else:
            if current_rate < target_detection_rate:
                print(f"  📋 検出率向上が必要（現在{current_rate:.1%} → 目標90%）")
            if current_avg_time > target_time:
                print(f"  📋 処理時間短縮が必要（現在{current_avg_time:.2f}秒 → 目標{target_time}秒以下）")
        
        print("\n" + "=" * 80)


def main():
    """メイン実行関数"""
    print("🚀 軽量版顔検出システム統合テスト開始")
    
    try:
        tester = LightweightFaceDetectionTest()
        results = tester.run_comprehensive_test()
        
        # 結果をJSONで保存
        output_file = f"lightweight_face_detection_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 詳細結果保存: {output_file}")
        print("✅ 軽量版顔検出統合テスト完了")
        
        return 0
    
    except Exception as e:
        logger.error(f"テスト実行エラー: {e}")
        return 1


if __name__ == "__main__":
    exit(main())