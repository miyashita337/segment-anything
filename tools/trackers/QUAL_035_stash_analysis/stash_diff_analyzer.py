#!/usr/bin/env python3
"""
212ファイル復旧差分分析システム
stash@{0}の内容を詳細分析して適切な処理方針を決定
"""

import os
import re
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ChangeType(Enum):
    FORMAT_ONLY = "format_only"      # フォーマットのみ
    FUNCTIONAL = "functional"        # 機能的変更
    TRACKER_ID = "tracker_id"        # トラッカーID更新
    DOCUMENTATION = "documentation"  # ドキュメント更新
    MIXED = "mixed"                  # 複合変更


class RecommendedAction(Enum):
    INTEGRATE = "integrate"          # git統合推奨
    DISCARD = "discard"              # 削除推奨
    REVIEW = "review"                # 要レビュー
    PRESERVE = "preserve"            # 保留


@dataclass
class FileAnalysis:
    file_path: str
    lines_added: int
    lines_removed: int
    change_type: ChangeType
    recommended_action: RecommendedAction
    confidence: float  # 0.0-1.0
    reasons: List[str]
    sample_changes: List[str]


class StashDiffAnalyzer:
    """212ファイル差分分析システム"""
    
    def __init__(self):
        self.stash_ref = "stash@{0}"
        self.base_ref = "HEAD"
        
        # パターン定義
        self.format_patterns = [
            r'^\s*$',                    # 空行のみ
            r'^\s*(#.*)?$',              # コメントのみ
            r'^\s*""".*"""$',            # docstring
            r'^\s*from\s+\w+\s+import',  # import文
            r'^\s*import\s+\w+',         # import文
        ]
        
        self.tracker_id_patterns = [
            r'P1-[A-Z0-9]+-\d+',        # 旧トラッカーID
            r'QCC-\d+',                 # QCC形式
            r'QI-\d+',                  # QI形式
            r'QUAL-\d+',                # QUAL形式
            r'INTG-\d+',                # INTG形式
            r'TEST-\d+',                # TEST形式
            r'OPTM-\d+',                # OPTM形式
        ]
        
        self.critical_files = [
            'features/extraction/commands/extract_character.py',
            'features/common/memory_optimizer.py',
            'features/evaluation/integrated_quality_monitor.py',
            'tools/core/sam_yolo_character_segment.py',
        ]
    
    def analyze_stash_files(self) -> Dict[str, FileAnalysis]:
        """stash@{0}の全ファイルを分析"""
        print("📊 stash@{0} 差分分析開始...")
        
        # stash差分を取得
        try:
            result = subprocess.run([
                'git', 'stash', 'show', '--numstat', self.stash_ref
            ], capture_output=True, text=True, check=True)
            
            file_stats = self._parse_numstat(result.stdout)
            analyses = {}
            
            for file_path, (added, removed) in file_stats.items():
                analysis = self._analyze_file_changes(file_path, added, removed)
                analyses[file_path] = analysis
                
            print(f"✅ {len(analyses)}ファイルの分析完了")
            return analyses
            
        except subprocess.CalledProcessError as e:
            print(f"❌ stash分析エラー: {e}")
            return {}
    
    def _parse_numstat(self, numstat_output: str) -> Dict[str, Tuple[int, int]]:
        """numstat出力をパース"""
        file_stats = {}
        
        for line in numstat_output.strip().split('\n'):
            if not line:
                continue
                
            parts = line.split('\t')
            if len(parts) >= 3:
                added = int(parts[0]) if parts[0] != '-' else 0
                removed = int(parts[1]) if parts[1] != '-' else 0
                file_path = parts[2]
                file_stats[file_path] = (added, removed)
        
        return file_stats
    
    def _analyze_file_changes(self, file_path: str, added: int, removed: int) -> FileAnalysis:
        """個別ファイルの変更を詳細分析"""
        
        # ファイル差分内容を取得
        diff_content = self._get_file_diff(file_path)
        
        # 変更タイプの判定
        change_type = self._classify_change_type(file_path, diff_content)
        
        # 推奨アクションの決定
        recommended_action = self._determine_action(file_path, change_type, added, removed)
        
        # 信頼度の計算
        confidence = self._calculate_confidence(change_type, file_path, diff_content)
        
        # 理由の生成
        reasons = self._generate_reasons(change_type, file_path, added, removed)
        
        # サンプル変更の抽出
        sample_changes = self._extract_sample_changes(diff_content)
        
        return FileAnalysis(
            file_path=file_path,
            lines_added=added,
            lines_removed=removed,
            change_type=change_type,
            recommended_action=recommended_action,
            confidence=confidence,
            reasons=reasons,
            sample_changes=sample_changes
        )
    
    def _get_file_diff(self, file_path: str) -> str:
        """ファイルの差分内容を取得"""
        try:
            result = subprocess.run([
                'git', 'stash', 'show', '-p', self.stash_ref, '--', file_path
            ], capture_output=True, text=True)
            
            return result.stdout
            
        except subprocess.CalledProcessError:
            return ""
    
    def _classify_change_type(self, file_path: str, diff_content: str) -> ChangeType:
        """変更タイプの分類"""
        
        # ドキュメントファイル判定
        if file_path.endswith(('.md', '.txt', '.rst')):
            return ChangeType.DOCUMENTATION
        
        # トラッカーID変更判定
        tracker_changes = 0
        for pattern in self.tracker_id_patterns:
            if re.search(pattern, diff_content):
                tracker_changes += 1
        
        if tracker_changes > 0:
            return ChangeType.TRACKER_ID
        
        # フォーマット変更判定
        diff_lines = diff_content.split('\n')
        format_lines = 0
        functional_lines = 0
        
        for line in diff_lines:
            if line.startswith(('+', '-')) and not line.startswith(('+++', '---')):
                content = line[1:]  # +/- を除去
                
                if any(re.match(pattern, content) for pattern in self.format_patterns):
                    format_lines += 1
                elif content.strip():  # 空でない行
                    functional_lines += 1
        
        total_changes = format_lines + functional_lines
        if total_changes == 0:
            return ChangeType.FORMAT_ONLY
        
        format_ratio = format_lines / total_changes
        
        if format_ratio > 0.8:
            return ChangeType.FORMAT_ONLY
        elif format_ratio < 0.2:
            return ChangeType.FUNCTIONAL
        else:
            return ChangeType.MIXED
    
    def _determine_action(self, file_path: str, change_type: ChangeType, 
                         added: int, removed: int) -> RecommendedAction:
        """推奨アクションの決定"""
        
        # 重要ファイルは慎重に判定
        if file_path in self.critical_files:
            if change_type == ChangeType.FUNCTIONAL:
                return RecommendedAction.REVIEW
            elif change_type == ChangeType.FORMAT_ONLY:
                return RecommendedAction.DISCARD
        
        # 変更タイプ別の判定
        if change_type == ChangeType.FORMAT_ONLY:
            return RecommendedAction.DISCARD
        elif change_type == ChangeType.TRACKER_ID:
            # トラッカーID更新は統合推奨
            return RecommendedAction.INTEGRATE
        elif change_type == ChangeType.FUNCTIONAL:
            # 機能変更は要レビュー
            return RecommendedAction.REVIEW
        elif change_type == ChangeType.DOCUMENTATION:
            # ドキュメント更新は統合推奨
            return RecommendedAction.INTEGRATE
        else:
            return RecommendedAction.REVIEW
    
    def _calculate_confidence(self, change_type: ChangeType, 
                            file_path: str, diff_content: str) -> float:
        """判定信頼度の計算"""
        
        base_confidence = 0.7
        
        # 変更タイプ別の信頼度調整
        if change_type == ChangeType.FORMAT_ONLY:
            base_confidence = 0.9
        elif change_type == ChangeType.TRACKER_ID:
            base_confidence = 0.8
        elif change_type == ChangeType.FUNCTIONAL:
            base_confidence = 0.6
        
        # 重要ファイルは信頼度を下げる（慎重判定）
        if file_path in self.critical_files:
            base_confidence *= 0.8
        
        # 差分サイズによる調整
        diff_lines = len(diff_content.split('\n'))
        if diff_lines > 1000:
            base_confidence *= 0.9  # 大きな変更は信頼度を下げる
        
        return min(1.0, base_confidence)
    
    def _generate_reasons(self, change_type: ChangeType, file_path: str,
                         added: int, removed: int) -> List[str]:
        """判定理由の生成"""
        reasons = []
        
        # 変更タイプ別の理由
        if change_type == ChangeType.FORMAT_ONLY:
            reasons.append("フォーマット・インデント変更のみ")
        elif change_type == ChangeType.TRACKER_ID:
            reasons.append("トラッカーID標準化による参照更新")
        elif change_type == ChangeType.FUNCTIONAL:
            reasons.append("機能的な変更を含む")
        elif change_type == ChangeType.DOCUMENTATION:
            reasons.append("ドキュメント更新")
        
        # 変更規模
        if added + removed > 500:
            reasons.append(f"大規模変更 (+{added}/-{removed})")
        elif added + removed < 10:
            reasons.append(f"小規模変更 (+{added}/-{removed})")
        
        # ファイル特性
        if file_path in self.critical_files:
            reasons.append("重要システムファイル")
        
        if file_path.startswith('tests/'):
            reasons.append("テストファイル")
        
        return reasons
    
    def _extract_sample_changes(self, diff_content: str) -> List[str]:
        """サンプル変更の抽出"""
        samples = []
        diff_lines = diff_content.split('\n')
        
        # 追加・削除行から最大3つ抽出
        for line in diff_lines:
            if line.startswith(('+', '-')) and not line.startswith(('+++', '---')):
                if len(samples) < 3:
                    samples.append(line[:100])  # 最大100文字
        
        return samples
    
    def generate_summary_report(self, analyses: Dict[str, FileAnalysis]) -> Dict:
        """分析結果サマリーレポート生成"""
        
        # アクション別分類
        action_groups = {action: [] for action in RecommendedAction}
        change_type_counts = {ct: 0 for ct in ChangeType}
        
        total_added = 0
        total_removed = 0
        
        for analysis in analyses.values():
            action_groups[analysis.recommended_action].append(analysis)
            change_type_counts[analysis.change_type] += 1
            total_added += analysis.lines_added
            total_removed += analysis.lines_removed
        
        # 高信頼度の判定
        high_confidence = [a for a in analyses.values() if a.confidence > 0.8]
        
        summary = {
            "total_files": len(analyses),
            "total_changes": {
                "added": total_added,
                "removed": total_removed,
                "net": total_added - total_removed
            },
            "action_recommendations": {
                action.value: len(files) for action, files in action_groups.items()
            },
            "change_types": {
                ct.value: count for ct, count in change_type_counts.items()
            },
            "confidence_stats": {
                "high_confidence_count": len(high_confidence),
                "high_confidence_ratio": len(high_confidence) / len(analyses) if analyses else 0,
                "average_confidence": sum(a.confidence for a in analyses.values()) / len(analyses) if analyses else 0
            },
            "recommended_integration": [
                a.file_path for a in analyses.values() 
                if a.recommended_action == RecommendedAction.INTEGRATE and a.confidence > 0.7
            ],
            "safe_to_discard": [
                a.file_path for a in analyses.values()
                if a.recommended_action == RecommendedAction.DISCARD and a.confidence > 0.8
            ],
            "requires_review": [
                a.file_path for a in analyses.values()
                if a.recommended_action == RecommendedAction.REVIEW
            ]
        }
        
        return summary
    
    def save_analysis_results(self, analyses: Dict[str, FileAnalysis], 
                            output_dir: str = "tools/analysis"):
        """分析結果を保存"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 詳細結果をJSON保存
        detailed_results = {
            file_path: {
                "file_path": analysis.file_path,
                "lines_added": analysis.lines_added,
                "lines_removed": analysis.lines_removed,
                "change_type": analysis.change_type.value,
                "recommended_action": analysis.recommended_action.value,
                "confidence": analysis.confidence,
                "reasons": analysis.reasons,
                "sample_changes": analysis.sample_changes
            }
            for file_path, analysis in analyses.items()
        }
        
        with open(output_path / "stash_analysis_detailed.json", "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        # サマリーレポート保存
        summary = self.generate_summary_report(analyses)
        with open(output_path / "stash_analysis_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # レポート生成
        self._generate_text_report(analyses, summary, output_path / "stash_analysis_report.md")
        
        print(f"✅ 分析結果保存完了: {output_path}")
    
    def _generate_text_report(self, analyses: Dict[str, FileAnalysis], 
                            summary: Dict, output_file: Path):
        """テキストレポート生成"""
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("# 212ファイル復旧差分分析レポート\n\n")
            f.write(f"**分析日時**: {os.popen('date').read().strip()}\n")
            f.write(f"**対象**: stash@{{0}} vs HEAD\n\n")
            
            # サマリー
            f.write("## 📊 分析サマリー\n\n")
            f.write(f"- **総ファイル数**: {summary['total_files']}ファイル\n")
            f.write(f"- **変更行数**: +{summary['total_changes']['added']} / -{summary['total_changes']['removed']}\n")
            f.write(f"- **平均信頼度**: {summary['confidence_stats']['average_confidence']:.2f}\n\n")
            
            # 推奨アクション
            f.write("## 🎯 推奨アクション\n\n")
            for action, count in summary['action_recommendations'].items():
                f.write(f"- **{action}**: {count}ファイル\n")
            f.write("\n")
            
            # 統合推奨ファイル
            if summary['recommended_integration']:
                f.write("### ✅ 統合推奨ファイル\n\n")
                for file_path in summary['recommended_integration'][:10]:
                    f.write(f"- `{file_path}`\n")
                if len(summary['recommended_integration']) > 10:
                    f.write(f"- ... 他{len(summary['recommended_integration'])-10}ファイル\n")
                f.write("\n")
            
            # 削除推奨ファイル
            if summary['safe_to_discard']:
                f.write("### 🗑️ 削除推奨ファイル\n\n")
                for file_path in summary['safe_to_discard'][:10]:
                    f.write(f"- `{file_path}`\n")
                if len(summary['safe_to_discard']) > 10:
                    f.write(f"- ... 他{len(summary['safe_to_discard'])-10}ファイル\n")
                f.write("\n")
            
            # レビュー要求ファイル
            if summary['requires_review']:
                f.write("### 🔍 要レビューファイル\n\n")
                for file_path in summary['requires_review'][:10]:
                    analysis = analyses[file_path]
                    f.write(f"- `{file_path}` (信頼度: {analysis.confidence:.2f})\n")
                    f.write(f"  - 理由: {', '.join(analysis.reasons)}\n")
                if len(summary['requires_review']) > 10:
                    f.write(f"- ... 他{len(summary['requires_review'])-10}ファイル\n")
                f.write("\n")


def main():
    """メイン実行関数"""
    analyzer = StashDiffAnalyzer()
    
    # 分析実行
    analyses = analyzer.analyze_stash_files()
    
    if analyses:
        # 結果保存
        analyzer.save_analysis_results(analyses)
        
        # サマリー表示
        summary = analyzer.generate_summary_report(analyses)
        
        print("\n🎯 分析結果サマリー:")
        print(f"  総ファイル数: {summary['total_files']}")
        print(f"  統合推奨: {summary['action_recommendations']['integrate']}")
        print(f"  削除推奨: {summary['action_recommendations']['discard']}")
        print(f"  要レビュー: {summary['action_recommendations']['review']}")
        print(f"  平均信頼度: {summary['confidence_stats']['average_confidence']:.2f}")
        
        return True
    else:
        print("❌ 分析に失敗しました")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)