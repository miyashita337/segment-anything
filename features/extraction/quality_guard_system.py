#!/usr/bin/env python3
"""
Quality Guard System - 品質保護システム
A評価結果の保護とトレードオフ回避を行う品質管理システム
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class QualityRecord:
    """品質記録データクラス"""
    filename: str
    rating: str
    timestamp: str
    method: str
    size: str
    notes: str = ""
    
    @property
    def quality_score(self) -> float:
        """評価グレードを数値スコアに変換"""
        rating_scores = {
            'A': 0.9,
            'B': 0.7,
            'C': 0.5,
            'D': 0.3,
            'E': 0.2,
            'F': 0.1
        }
        return rating_scores.get(self.rating.upper(), 0.0)


class QualityGuardSystem:
    """品質保護・監視システム"""
    
    def __init__(self, 
                 quality_threshold: str = "B",
                 protection_enabled: bool = True):
        """
        Args:
            quality_threshold: 保護する品質レベル (A, B, C, D, E, F)
            protection_enabled: 保護機能の有効/無効
        """
        self.quality_threshold = quality_threshold
        self.protection_enabled = protection_enabled
        
        # 品質履歴を保存
        self.quality_history: Dict[str, List[QualityRecord]] = {}
        self.protected_files: Dict[str, QualityRecord] = {}
        
        # 既存の評価結果を読み込み
        self._load_existing_evaluations()
        
        logger.info(f"QualityGuardSystem初期化: threshold={quality_threshold}, "
                   f"protection={protection_enabled}")

    def _load_existing_evaluations(self):
        """既存の評価結果を読み込んで保護対象を設定"""
        evaluation_files = [
            "/mnt/c/AItools/image_evaluation_system/data/evaluation_progress_2025-07-22T17-44-20.json",  # Enhanced System
            "/mnt/c/AItools/image_evaluation_system/data/evaluation_progress_2025-07-22T18-47-36.json"   # Boundary Enhanced
        ]
        
        for eval_file in evaluation_files:
            eval_path = Path(eval_file)
            if eval_path.exists():
                try:
                    with open(eval_path, 'r', encoding='utf-8') as f:
                        eval_data = json.load(f)
                    
                    self._process_evaluation_data(eval_data, eval_path.name)
                    
                except Exception as e:
                    logger.warning(f"評価ファイル読み込みエラー {eval_file}: {e}")

    def _process_evaluation_data(self, eval_data: Dict[str, Any], source: str):
        """評価データを処理して保護対象を特定"""
        if "evaluationData" not in eval_data:
            return
        
        for item in eval_data["evaluationData"]:
            filename = item.get("filename", "")
            rating = item.get("folder2_rating", "")
            notes = item.get("notes", "")
            
            if filename and rating:
                record = QualityRecord(
                    filename=filename,
                    rating=rating,
                    timestamp=eval_data.get("timestamp", "unknown"),
                    method=self._extract_method_from_source(source),
                    size="unknown",
                    notes=notes
                )
                
                # 履歴に追加
                if filename not in self.quality_history:
                    self.quality_history[filename] = []
                self.quality_history[filename].append(record)
                
                # A評価またはB評価の場合は保護対象に追加
                if self._should_protect_rating(rating):
                    # 既存の保護対象がない、または品質がより良い場合
                    if (filename not in self.protected_files or 
                        self._is_better_rating(rating, self.protected_files[filename].rating)):
                        
                        self.protected_files[filename] = record
                        logger.info(f"保護対象追加: {filename} ({rating}評価)")

    def _extract_method_from_source(self, source: str) -> str:
        """ソースファイル名から手法名を抽出"""
        if "enhanced_system" in source:
            return "enhanced_system"
        elif "boundary_enhanced" in source:
            return "boundary_enhanced"
        else:
            return "unknown"

    def _should_protect_rating(self, rating: str) -> bool:
        """評価が保護対象かどうか判定"""
        if not rating or not self.protection_enabled:
            return False
        
        # A, B評価は保護
        rating_order = ["F", "E", "D", "C", "B", "A"]
        threshold_index = rating_order.index(self.quality_threshold) if self.quality_threshold in rating_order else 0
        rating_index = rating_order.index(rating) if rating in rating_order else 0
        
        return rating_index >= threshold_index

    def _is_better_rating(self, new_rating: str, current_rating: str) -> bool:
        """新しい評価の方が良いかどうか判定"""
        rating_order = ["F", "E", "D", "C", "B", "A"]
        new_index = rating_order.index(new_rating) if new_rating in rating_order else 0
        current_index = rating_order.index(current_rating) if current_rating in rating_order else 0
        
        return new_index > current_index

    def should_skip_processing(self, filename: str) -> Tuple[bool, Optional[QualityRecord]]:
        """
        ファイルの処理をスキップすべきかどうか判定
        
        Args:
            filename: 処理対象ファイル名
            
        Returns:
            (スキップするかどうか, 保護されている記録)
        """
        if not self.protection_enabled:
            return False, None
        
        if filename in self.protected_files:
            protected_record = self.protected_files[filename]
            logger.info(f"保護対象ファイル検出: {filename} "
                       f"({protected_record.rating}評価, {protected_record.method})")
            return True, protected_record
        
        return False, None

    def should_use_protected_result(self, filename: str, 
                                   new_result: Dict[str, Any]) -> Tuple[bool, Optional[Path]]:
        """
        新結果ではなく保護された結果を使用すべきかどうか判定
        
        Args:
            filename: ファイル名
            new_result: 新しい処理結果
            
        Returns:
            (保護結果を使用するか, 保護結果のパス)
        """
        if not self.protection_enabled or filename not in self.protected_files:
            return False, None
        
        protected_record = self.protected_files[filename]
        
        # A評価は絶対保護
        if protected_record.rating == "A":
            protected_path = self._get_protected_result_path(filename, protected_record)
            if protected_path and protected_path.exists():
                logger.info(f"A評価結果を保護使用: {filename}")
                return True, protected_path
        
        # B評価は新結果の品質が低い場合のみ保護
        elif protected_record.rating == "B":
            new_quality = new_result.get("quality_score", 0.0)
            
            # 新結果の品質が低い場合（0.7未満）は保護結果を使用
            if new_quality < 0.7:
                protected_path = self._get_protected_result_path(filename, protected_record)
                if protected_path and protected_path.exists():
                    logger.info(f"B評価結果を保護使用: {filename} "
                               f"(新品質={new_quality:.3f} < 0.7)")
                    return True, protected_path
        
        return False, None

    def _get_protected_result_path(self, filename: str, record: QualityRecord) -> Optional[Path]:
        """保護された結果のファイルパスを取得"""
        base_dirs = {
            "enhanced_system": "/mnt/c/AItools/lora/train/yado/clipped_boundingbox/kana08_enhanced_system_final",
            "boundary_enhanced": "/mnt/c/AItools/lora/train/yado/clipped_boundingbox/kana08_boundary_enhanced_full",
            "backup": "/mnt/c/AItools/lora/train/yado/clipped_boundingbox/kana08_claude_uni_13_9_backup_migrated"
        }
        
        method = record.method
        if method in base_dirs:
            return Path(base_dirs[method]) / filename
        
        # フォールバック: 全ディレクトリを検索
        for base_dir in base_dirs.values():
            candidate_path = Path(base_dir) / filename
            if candidate_path.exists():
                return candidate_path
        
        return None

    def register_new_result(self, filename: str, 
                           result: Dict[str, Any], 
                           method: str = "robust_system"):
        """新しい処理結果を登録"""
        if not result.get("success", False):
            return
        
        record = QualityRecord(
            filename=filename,
            rating="unknown",  # 評価は後で更新
            timestamp=datetime.now().isoformat(),
            method=method,
            size=result.get("size", "unknown"),
            notes=f"Quality: {result.get('quality_score', 0):.3f}"
        )
        
        if filename not in self.quality_history:
            self.quality_history[filename] = []
        self.quality_history[filename].append(record)
        
        logger.debug(f"新結果登録: {filename} ({method})")

    def get_protection_stats(self) -> Dict[str, Any]:
        """保護統計情報を取得"""
        total_files = len(self.quality_history)
        protected_count = len(self.protected_files)
        
        rating_breakdown = {}
        for record in self.protected_files.values():
            rating = record.rating
            rating_breakdown[rating] = rating_breakdown.get(rating, 0) + 1
        
        return {
            "total_files": total_files,
            "protected_count": protected_count,
            "protection_rate": protected_count / max(total_files, 1),
            "rating_breakdown": rating_breakdown,
            "protection_enabled": self.protection_enabled,
            "threshold": self.quality_threshold
        }

    def copy_protected_result(self, filename: str, target_path: Path) -> bool:
        """保護された結果を目標パスにコピー"""
        if filename not in self.protected_files:
            return False
        
        protected_record = self.protected_files[filename]
        source_path = self._get_protected_result_path(filename, protected_record)
        
        if not source_path or not source_path.exists():
            logger.warning(f"保護結果が見つかりません: {filename}")
            return False
        
        try:
            import shutil

            # 目標ディレクトリを作成
            target_path.parent.mkdir(parents=True, exist_ok=True)
            # ファイルをコピー
            shutil.copy2(source_path, target_path)
            logger.info(f"保護結果をコピー: {source_path} -> {target_path}")
            return True
        except Exception as e:
            logger.error(f"保護結果コピーエラー: {e}")
            return False


def test_quality_guard_system():
    """品質保護システムのテスト"""
    guard = QualityGuardSystem(quality_threshold="B", protection_enabled=True)
    
    print("🛡️ 品質保護システム テスト")
    
    # 統計情報表示
    stats = guard.get_protection_stats()
    print(f"📊 保護統計:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 保護対象ファイル表示
    print(f"\n🔒 保護対象ファイル ({len(guard.protected_files)}件):")
    for filename, record in guard.protected_files.items():
        print(f"  {filename}: {record.rating}評価 ({record.method})")
    
    # テスト対象ファイルでの判定確認
    test_files = ["kana08_0001.jpg", "kana08_0003.jpg", "kana08_0000_cover.jpg"]
    
    print(f"\n🧪 処理判定テスト:")
    for filename in test_files:
        should_skip, protected_record = guard.should_skip_processing(filename)
        print(f"  {filename}: スキップ={should_skip}")
        if protected_record:
            print(f"    保護記録: {protected_record.rating}評価 ({protected_record.method})")


if __name__ == "__main__":
    test_quality_guard_system()