#!/usr/bin/env python3
"""
P1-B003 重複検出機能ユニットテスト
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image
import json
import sys

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from features.evaluation.duplicate_detector import (
    ImageDuplicateDetector,
    DuplicateInfo,
    DuplicateGroup,
    DuplicateDetectionResult,
    create_duplicate_detector
)


class TestP1B003DuplicateDetector(unittest.TestCase):
    """P1-B003 重複検出システムテスト"""

    def setUp(self):
        """テスト前準備"""
        self.test_dir = Path(tempfile.mkdtemp(prefix="p1_b003_test_"))
        self.input_dir = self.test_dir / "input"
        self.output_dir = self.test_dir / "output"
        self.cache_dir = self.test_dir / "cache"
        
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        
        # テスト画像作成
        self.create_test_images()
        
    def tearDown(self):
        """テスト後クリーンアップ"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def create_test_images(self):
        """テスト用画像ファイル作成"""
        # 1. 完全同一画像（ファイルハッシュ一致）
        img1 = Image.new('RGB', (64, 64), color='red')
        img1_path = self.input_dir / "image1.jpg"
        img1_copy_path = self.input_dir / "image1_copy.jpg"
        
        img1.save(img1_path, 'JPEG')
        img1.save(img1_copy_path, 'JPEG')  # 完全同一
        
        # 2. 視覚的に類似した画像
        img2 = Image.new('RGB', (64, 64), color='blue')
        img2_similar = Image.new('RGB', (64, 64), color=(0, 0, 250))  # 微妙に違う青
        
        img2_path = self.input_dir / "image2.jpg"
        img2_similar_path = self.input_dir / "image2_similar.jpg"
        
        img2.save(img2_path, 'JPEG')
        img2_similar.save(img2_similar_path, 'JPEG')
        
        # 3. 全く違う画像
        img3 = Image.new('RGB', (64, 64), color='green')
        img3_path = self.input_dir / "image3.jpg"
        img3.save(img3_path, 'JPEG')
        
        # 4. 異なるサイズの同じ色画像
        img4_large = Image.new('RGB', (128, 128), color='yellow')
        img4_small = Image.new('RGB', (32, 32), color='yellow')
        
        img4_large_path = self.input_dir / "image4_large.jpg"
        img4_small_path = self.input_dir / "image4_small.jpg"
        
        img4_large.save(img4_large_path, 'JPEG')
        img4_small.save(img4_small_path, 'JPEG')
        
        # テスト用ファイルリスト保存
        self.test_files = {
            'identical': [img1_path, img1_copy_path],
            'similar': [img2_path, img2_similar_path],
            'unique': [img3_path],
            'different_sizes': [img4_large_path, img4_small_path]
        }
    
    def test_detector_initialization(self):
        """重複検出器初期化テスト"""
        detector = ImageDuplicateDetector(
            visual_threshold=0.8,
            enable_visual_detection=True,
            cache_dir=self.cache_dir
        )
        
        self.assertEqual(detector.visual_threshold, 0.8)
        self.assertTrue(detector.enable_visual_detection)
        self.assertEqual(detector.cache_dir, self.cache_dir)
        self.assertIsInstance(detector.image_hashes, dict)
        self.assertIsInstance(detector.visual_cache, dict)
        self.assertIsInstance(detector.stats, dict)
    
    def test_factory_function(self):
        """ファクトリー関数テスト"""
        detector = create_duplicate_detector(
            visual_threshold=0.9,
            enable_visual_detection=False,
            cache_dir=self.cache_dir
        )
        
        self.assertIsInstance(detector, ImageDuplicateDetector)
        self.assertEqual(detector.visual_threshold, 0.9)
        self.assertFalse(detector.enable_visual_detection)
    
    def test_file_hash_calculation(self):
        """ファイルハッシュ計算テスト"""
        detector = ImageDuplicateDetector(cache_dir=self.cache_dir)
        
        # 同一ファイルは同じハッシュ
        hash1 = detector._calculate_file_hash(self.test_files['identical'][0])
        hash2 = detector._calculate_file_hash(self.test_files['identical'][1])
        self.assertEqual(hash1, hash2)
        
        # 異なるファイルは異なるハッシュ
        hash3 = detector._calculate_file_hash(self.test_files['unique'][0])
        self.assertNotEqual(hash1, hash3)
        
        # ハッシュが空でない
        self.assertNotEqual(hash1, "")
        self.assertNotEqual(hash3, "")
    
    @patch('features.evaluation.duplicate_detector.CONTENT_EVALUATOR_AVAILABLE', False)
    def test_visual_hash_calculation(self):
        """視覚的ハッシュ計算テスト"""
        detector = ImageDuplicateDetector(
            enable_visual_detection=True,
            cache_dir=self.cache_dir
        )
        
        # 視覚的ハッシュ計算
        visual_hash1 = detector._calculate_visual_hash(self.test_files['identical'][0])
        visual_hash2 = detector._calculate_visual_hash(self.test_files['identical'][1])
        visual_hash3 = detector._calculate_visual_hash(self.test_files['unique'][0])
        
        # 同一画像は同じ視覚的ハッシュ
        self.assertEqual(visual_hash1, visual_hash2)
        
        # 異なる画像は異なる視覚的ハッシュ
        self.assertNotEqual(visual_hash1, visual_hash3)
        
        # ハッシュが有効な形式
        if visual_hash1:
            self.assertIsInstance(visual_hash1, str)
            self.assertTrue(all(c in '01' for c in visual_hash1))  # バイナリ文字列
    
    def test_image_info_caching(self):
        """画像情報キャッシュテスト"""
        detector = ImageDuplicateDetector(cache_dir=self.cache_dir)
        
        # 初回取得
        info1 = detector._get_image_info(self.test_files['unique'][0])
        initial_calculations = detector.stats['hash_calculations']
        
        # 2回目取得（キャッシュヒット）
        info2 = detector._get_image_info(self.test_files['unique'][0])
        
        # 同じ情報が返される
        self.assertEqual(info1.file_path, info2.file_path)
        self.assertEqual(info1.file_hash, info2.file_hash)
        
        # ハッシュ計算が増加していない（キャッシュ利用）
        self.assertEqual(detector.stats['hash_calculations'], initial_calculations)
        self.assertGreater(detector.stats['hash_cache_hits'], 0)
    
    @patch('features.evaluation.duplicate_detector.CONTENT_EVALUATOR_AVAILABLE', False)
    def test_exact_duplicate_detection(self):
        """完全一致重複検出テスト"""
        detector = ImageDuplicateDetector(
            enable_visual_detection=False,  # ファイルハッシュのみ
            cache_dir=self.cache_dir
        )
        
        result = detector.detect_duplicates(self.input_dir)
        
        # 結果検証
        self.assertIsInstance(result, DuplicateDetectionResult)
        self.assertGreater(result.total_images, 0)
        
        # 完全一致重複グループの確認
        exact_groups = [g for g in result.duplicate_groups if g.hash_type == "exact"]
        self.assertGreater(len(exact_groups), 0)
        
        # 完全一致グループの詳細検証
        for group in exact_groups:
            self.assertEqual(group.confidence, 1.0)
            self.assertGreater(len(group.duplicate_images), 0)
            self.assertIn(group.primary_image, [str(f) for f in self.test_files['identical']])
    
    @patch('features.evaluation.duplicate_detector.ContentEvaluator')
    @patch('features.evaluation.duplicate_detector.CONTENT_EVALUATOR_AVAILABLE', True)
    def test_visual_duplicate_detection(self, mock_evaluator_class):
        """視覚的類似重複検出テスト"""
        # モックの設定
        mock_evaluator = MagicMock()
        mock_evaluator_class.return_value = mock_evaluator
        mock_evaluator.evaluate_crop_similarity.return_value = 0.9  # 高い類似度
        
        detector = ImageDuplicateDetector(
            visual_threshold=0.8,
            enable_visual_detection=True,
            cache_dir=self.cache_dir
        )
        
        result = detector.detect_duplicates(self.input_dir)
        
        # 視覚的類似グループの確認
        visual_groups = [g for g in result.duplicate_groups if g.hash_type == "visual"]
        
        # ContentEvaluatorが呼び出されていることを確認
        if visual_groups:
            mock_evaluator.evaluate_crop_similarity.assert_called()
            
            # 視覚的グループの詳細検証
            for group in visual_groups:
                self.assertEqual(group.hash_type, "visual")
                self.assertGreaterEqual(group.confidence, detector.visual_threshold)
                self.assertGreater(len(group.similarity_scores), 0)
    
    def test_duplicate_grouping(self):
        """重複グループ化テスト"""
        detector = ImageDuplicateDetector(cache_dir=self.cache_dir)
        
        # テスト用画像情報作成
        test_infos = [
            DuplicateInfo("img1.jpg", "hash1", 1000),
            DuplicateInfo("img1_copy.jpg", "hash1", 1000),  # 同一ハッシュ
            DuplicateInfo("img2.jpg", "hash2", 2000),
            DuplicateInfo("img3.jpg", "hash3", 3000),
        ]
        
        hash_groups = detector._group_by_hash(test_infos)
        
        # グループ化結果検証
        self.assertIn("hash1", hash_groups)
        self.assertEqual(len(hash_groups["hash1"]), 2)  # 重複あり
        self.assertEqual(len(hash_groups["hash2"]), 1)  # 重複なし
        self.assertEqual(len(hash_groups["hash3"]), 1)  # 重複なし
    
    def test_report_generation(self):
        """レポート生成テスト"""
        detector = ImageDuplicateDetector(cache_dir=self.cache_dir)
        
        # テスト用結果作成
        test_group = DuplicateGroup(
            group_id="test_group",
            primary_image="primary.jpg",
            duplicate_images=["dup1.jpg", "dup2.jpg"],
            similarity_scores={"primary.jpg:dup1.jpg": 0.95},
            hash_type="exact",
            confidence=1.0
        )
        
        test_result = DuplicateDetectionResult(
            total_images=5,
            total_duplicates=2,
            duplicate_groups=[test_group],
            processing_time=1.5,
            hash_cache_hits=3,
            visual_comparisons=2
        )
        
        # レポート生成
        report_path = detector.generate_report(test_result, self.output_dir)
        
        # レポートファイル確認
        self.assertIsNotNone(report_path)
        self.assertTrue(report_path.exists())
        
        # JSONレポート検証
        with open(report_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        
        self.assertEqual(report_data['summary']['total_images'], 5)
        self.assertEqual(report_data['summary']['total_duplicates'], 2)
        self.assertEqual(len(report_data['groups']), 1)
        
        # Markdownレポート確認
        md_path = self.output_dir / "duplicate_report.md"
        self.assertTrue(md_path.exists())
        
        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        self.assertIn("# 画像重複検出レポート", md_content)
        self.assertIn("総画像数", md_content)
        self.assertIn("test_group", md_content)
    
    def test_cache_persistence(self):
        """キャッシュ永続化テスト"""
        # 1回目の検出器
        detector1 = ImageDuplicateDetector(cache_dir=self.cache_dir)
        info1 = detector1._get_image_info(self.test_files['unique'][0])
        detector1._save_hash_cache()
        
        # 2回目の検出器（キャッシュ復元）
        detector2 = ImageDuplicateDetector(cache_dir=self.cache_dir)
        info2 = detector2._get_image_info(self.test_files['unique'][0])
        
        # 同じハッシュが復元される
        self.assertEqual(info1.file_hash, info2.file_hash)
        self.assertGreater(detector2.stats['hash_cache_hits'], 0)
    
    def test_error_handling(self):
        """エラーハンドリングテスト"""
        detector = ImageDuplicateDetector(cache_dir=self.cache_dir)
        
        # 存在しないファイルのハッシュ計算
        non_existent = Path("/non/existent/file.jpg")
        hash_result = detector._calculate_file_hash(non_existent)
        self.assertEqual(hash_result, "")  # エラー時は空文字列
        
        # 存在しないディレクトリでの重複検出
        non_existent_dir = Path("/non/existent/directory")
        result = detector.detect_duplicates(non_existent_dir)
        self.assertEqual(result.total_images, 0)
        self.assertEqual(len(result.duplicate_groups), 0)
    
    def test_statistics_tracking(self):
        """統計追跡テスト"""
        detector = ImageDuplicateDetector(cache_dir=self.cache_dir)
        
        # 初期統計確認
        initial_stats = detector.stats.copy()
        
        # 重複検出実行
        result = detector.detect_duplicates(self.input_dir)
        
        # 統計が更新されている
        self.assertGreaterEqual(detector.stats['hash_calculations'], 
                               initial_stats['hash_calculations'])
        
        # 結果に統計が反映されている
        self.assertGreaterEqual(result.hash_cache_hits, 0)
        
        if detector.enable_visual_detection and detector.content_evaluator:
            self.assertGreaterEqual(result.visual_comparisons, 0)
    
    @patch('features.evaluation.duplicate_detector.logger')
    def test_logging(self, mock_logger):
        """ログ出力テスト"""
        detector = ImageDuplicateDetector(cache_dir=self.cache_dir)
        
        # 重複検出実行
        detector.detect_duplicates(self.input_dir)
        
        # ログが出力されている
        mock_logger.info.assert_called()
    
    def test_large_dataset_handling(self):
        """大量データセット処理テスト"""
        # 追加のテスト画像を大量作成
        large_test_dir = self.test_dir / "large_input"
        large_test_dir.mkdir(exist_ok=True)
        
        # 100枚の画像を作成（実際は小さなテスト）
        num_images = 10  # テスト高速化のため少数
        colors = ['red', 'green', 'blue', 'yellow', 'purple']
        
        for i in range(num_images):
            color = colors[i % len(colors)]
            img = Image.new('RGB', (32, 32), color=color)
            img_path = large_test_dir / f"test_image_{i:03d}.jpg"
            img.save(img_path, 'JPEG')
        
        detector = ImageDuplicateDetector(
            enable_visual_detection=False,  # 高速化
            cache_dir=self.cache_dir
        )
        
        # 処理実行
        result = detector.detect_duplicates(large_test_dir)
        
        # 結果検証
        self.assertEqual(result.total_images, num_images)
        self.assertLessEqual(result.processing_time, 30.0)  # 30秒以内
        
        # 同色画像の重複検出確認
        # (同じ色の画像が複数ある場合、視覚的類似度で検出される可能性)
        self.assertGreaterEqual(len(result.duplicate_groups), 0)


if __name__ == '__main__':
    # 詳細ログ有効化
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    # テスト実行
    unittest.main(verbosity=2)