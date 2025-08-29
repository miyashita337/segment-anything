#!/usr/bin/env python3
"""TEST-001抽出機能単体テスト

Phase 2ステップ7: 単体テスト実装
- 個別ファイル検証
- ファイル属性確認
- エラーパターン分析
"""

import os
import unittest
from pathlib import Path
from PIL import Image


class TestTEST001Extraction(unittest.TestCase):
    """TEST-001抽出機能単体テスト"""
    
    def setUp(self):
        """テスト環境セットアップ"""
        self.tracker_id = "TEST-001"
        self.extraction_dir = Path(
            f"/mnt/c/AItools/lora/train/yado/tracker-workspace/{self.tracker_id}/extraction"
        )
        
    def test_file_format(self):
        """ファイル形式検証"""
        extracted_files = list(self.extraction_dir.glob("extracted_*.jpg"))
        
        for file in extracted_files:
            # JPEG形式確認
            self.assertTrue(
                file.suffix.lower() == '.jpg',
                f"{file.name}の拡張子が不正"
            )
            
            # 画像として開けることを確認
            try:
                img = Image.open(file)
                self.assertEqual(
                    img.format, 'JPEG',
                    f"{file.name}がJPEG形式ではありません"
                )
                img.close()
            except Exception as e:
                self.fail(f"{file.name}を画像として開けません: {e}")
                
    def test_file_dimensions(self):
        """画像サイズ検証"""
        extracted_files = list(self.extraction_dir.glob("extracted_*.jpg"))
        
        for file in extracted_files:
            img = Image.open(file)
            width, height = img.size
            
            # 最小サイズ確認（100x100以上）
            self.assertGreaterEqual(
                width, 100,
                f"{file.name}の幅{width}pxが最小値未満"
            )
            self.assertGreaterEqual(
                height, 100,
                f"{file.name}の高さ{height}pxが最小値未満"
            )
            
            # 最大サイズ確認（4000x4000以下）
            self.assertLessEqual(
                width, 4000,
                f"{file.name}の幅{width}pxが最大値超過"
            )
            self.assertLessEqual(
                height, 4000,
                f"{file.name}の高さ{height}pxが最大値超過"
            )
            
            img.close()
            
    def test_file_naming_convention(self):
        """ファイル命名規則検証"""
        extracted_files = list(self.extraction_dir.glob("extracted_*.jpg"))
        
        for file in extracted_files:
            # extracted_NNNN.jpg形式確認
            stem = file.stem
            self.assertTrue(
                stem.startswith("extracted_"),
                f"{file.name}が命名規則に従っていません"
            )
            
            # 番号部分の検証
            number_part = stem.replace("extracted_", "")
            self.assertTrue(
                number_part.isdigit() and len(number_part) == 4,
                f"{file.name}の番号形式が不正: {number_part}"
            )
            
    def test_specific_success_files(self):
        """特定成功ファイル詳細検証"""
        # ログから確認された成功ファイル
        success_files = [
            ("extracted_0001.jpg", 169996),  # 最初の成功
            ("extracted_0003.jpg", 783409),  # 最大サイズ
            ("extracted_0024.jpg", 50482),   # 最小サイズ近辺
        ]
        
        for filename, expected_size in success_files:
            file_path = self.extraction_dir / filename
            self.assertTrue(
                file_path.exists(),
                f"期待される成功ファイル{filename}が存在しません"
            )
            
            # サイズ誤差10%以内
            actual_size = file_path.stat().st_size
            size_diff = abs(actual_size - expected_size) / expected_size
            self.assertLessEqual(
                size_diff, 0.1,
                f"{filename}のサイズ{actual_size}が期待値{expected_size}と10%以上異なります"
            )
            
    def test_failure_patterns(self):
        """失敗パターン検証"""
        # 失敗が確認されているファイル
        failed_files = ["extracted_0002.jpg", "extracted_0027.jpg"]
        
        for filename in failed_files:
            file_path = self.extraction_dir / filename
            self.assertFalse(
                file_path.exists(),
                f"失敗すべき{filename}が存在しています"
            )


if __name__ == "__main__":
    unittest.main()