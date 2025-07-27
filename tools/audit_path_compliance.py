#!/usr/bin/env python3
"""
既存コードのパス設定準拠監査ツール
不正なパス設定を検出し、修正提案を行う
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import ast
import logging

# プロジェクトルートをパスに追加
sys.path.append('.')

logger = logging.getLogger(__name__)


@dataclass
class PathIssue:
    """パス設定問題の詳細"""
    file_path: str
    line_number: int
    line_content: str
    issue_type: str
    severity: str  # high, medium, low
    description: str
    suggested_fix: Optional[str] = None


class PathComplianceAuditor:
    """パス設定準拠監査ツール"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.issues: List[PathIssue] = []
        
        # 問題パターン定義
        self.problematic_patterns = {
            'relative_path': {
                'pattern': r'Path\(\"(?!/)([^"]*)\"\)',
                'severity': 'high',
                'description': '相対パスの使用（ハードコード）'
            },
            'hardcoded_output': {
                'pattern': r'(dashboard_output|results_batch|test_results|temp_output)',
                'severity': 'high', 
                'description': 'ハードコードされた出力ディレクトリ名'
            },
            'current_dir_relative': {
                'pattern': r'Path\(\"\.\/|Path\(\"\.\.\/',
                'severity': 'medium',
                'description': 'カレントディレクトリ基準の相対パス'
            },
            'mkdir_without_tracker': {
                'pattern': r'\.mkdir\(.*exist_ok=True.*\)',
                'severity': 'low',
                'description': 'トラッカーID管理外のディレクトリ作成'
            }
        }
        
        # 許可パターン（誤検出回避）
        self.allowed_patterns = {
            'absolute_workspace': r'/mnt/c/AItools/lora/train/yado/clipped_boundingbox/workspace',
            'config_files': r'Path\("config/',
            'temporary_test': r'Path\("test_',
            'model_files': r'Path\(".*\.(pth|pt|ckpt|safetensors)"',
        }
    
    def scan_file(self, file_path: Path) -> List[PathIssue]:
        """単一ファイルのスキャン"""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            return issues
        
        for line_num, line in enumerate(lines, 1):
            line_content = line.strip()
            
            # 空行やコメント行をスキップ
            if not line_content or line_content.startswith('#'):
                continue
            
            # 各問題パターンをチェック
            for issue_type, pattern_info in self.problematic_patterns.items():
                matches = re.finditer(pattern_info['pattern'], line_content)
                
                for match in matches:
                    # 許可パターンかチェック
                    if self._is_allowed_pattern(line_content):
                        continue
                    
                    # 問題として記録
                    issue = PathIssue(
                        file_path=str(file_path),
                        line_number=line_num,
                        line_content=line_content,
                        issue_type=issue_type,
                        severity=pattern_info['severity'],
                        description=pattern_info['description'],
                        suggested_fix=self._generate_fix_suggestion(
                            issue_type, line_content, match.group()
                        )
                    )
                    issues.append(issue)
        
        return issues
    
    def _is_allowed_pattern(self, line_content: str) -> bool:
        """許可パターンかどうかチェック"""
        for pattern in self.allowed_patterns.values():
            if re.search(pattern, line_content):
                return True
        return False
    
    def _generate_fix_suggestion(self, issue_type: str, line_content: str, matched_text: str) -> str:
        """修正提案生成"""
        suggestions = {
            'relative_path': f"""
# 修正前: {line_content}
# 修正後例:
from features.common.output_path_manager import OutputPathManager, OutputCategory
manager = OutputPathManager("{{tracker_id}}")
output_path = manager.get_output_path(OutputCategory.DASHBOARD, filename="report.html")
""",
            'hardcoded_output': f"""
# 修正前: {line_content}
# 修正後例:
from features.common.output_path_manager import ensure_compliant_output, OutputCategory
output_path = ensure_compliant_output("{{tracker_id}}", OutputCategory.DASHBOARD, "report.html")
""",
            'current_dir_relative': f"""
# 修正前: {line_content}
# 修正後例: 絶対パスまたはOutputPathManagerを使用
# WORKSPACE_BASE = Path("/mnt/c/AItools/lora/train/yado/clipped_boundingbox/workspace")
""",
            'mkdir_without_tracker': f"""
# 確認必要: このディレクトリ作成はトラッカー管理が必要か？
# 必要な場合: OutputPathManager.ensure_output_dir()を使用
"""
        }
        
        return suggestions.get(issue_type, "手動確認が必要です")
    
    def scan_project(self, 
                    include_patterns: List[str] = ["*.py"], 
                    exclude_dirs: List[str] = [".git", "__pycache__", "venv", "node_modules"]) -> List[PathIssue]:
        """プロジェクト全体のスキャン"""
        all_issues = []
        
        # 除外ディレクトリの準備
        exclude_paths = {self.project_root / exclude_dir for exclude_dir in exclude_dirs}
        
        # ファイルパターンごとにスキャン
        for pattern in include_patterns:
            for file_path in self.project_root.rglob(pattern):
                # 除外ディレクトリチェック
                if any(exclude_path in file_path.parents for exclude_path in exclude_paths):
                    continue
                
                # ファイルスキャン実行
                file_issues = self.scan_file(file_path)
                all_issues.extend(file_issues)
        
        self.issues = all_issues
        return all_issues
    
    def generate_report(self) -> Dict[str, Any]:
        """監査レポート生成"""
        # 重要度別集計
        severity_counts = {'high': 0, 'medium': 0, 'low': 0}
        issue_type_counts = {}
        file_counts = {}
        
        for issue in self.issues:
            severity_counts[issue.severity] += 1
            issue_type_counts[issue.issue_type] = issue_type_counts.get(issue.issue_type, 0) + 1
            file_counts[issue.file_path] = file_counts.get(issue.file_path, 0) + 1
        
        # 最も問題のあるファイル
        top_problematic_files = sorted(
            file_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        return {
            'summary': {
                'total_issues': len(self.issues),
                'severity_breakdown': severity_counts,
                'issue_type_breakdown': issue_type_counts,
                'files_affected': len(file_counts)
            },
            'top_problematic_files': top_problematic_files,
            'high_priority_issues': [
                issue for issue in self.issues if issue.severity == 'high'
            ],
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """改善推奨事項生成"""
        recommendations = []
        
        # 重要度高の問題があるか
        high_issues = [issue for issue in self.issues if issue.severity == 'high']
        if high_issues:
            recommendations.append(
                f"🚨 高優先度問題 {len(high_issues)} 件の即座対応が必要"
            )
        
        # 相対パス問題
        relative_path_issues = [issue for issue in self.issues if issue.issue_type == 'relative_path']
        if relative_path_issues:
            recommendations.append(
                f"📁 相対パス {len(relative_path_issues)} 件をOutputPathManagerで置換推奨"
            )
        
        # ハードコード問題
        hardcoded_issues = [issue for issue in self.issues if issue.issue_type == 'hardcoded_output']
        if hardcoded_issues:
            recommendations.append(
                f"🔧 ハードコード出力パス {len(hardcoded_issues)} 件の仕様準拠化が必要"
            )
        
        # 一般的推奨事項
        if self.issues:
            recommendations.extend([
                "📋 docs/guidelines/SPECIFICATION_COMPLIANCE_CHECKLIST.md に従った再実装推奨",
                "🔍 CI/CDパイプラインへの自動チェック統合検討",
                "📚 開発チームへの仕様書周知・教育実施"
            ])
        
        return recommendations
    
    def export_detailed_report(self, output_path: Path):
        """詳細レポートをファイル出力"""
        report = self.generate_report()
        
        # テキストレポート生成
        content = f"""# パス設定準拠監査レポート

**実行日時**: {Path().cwd()}  
**対象**: {self.project_root}  

## 📊 サマリー

- **総問題数**: {report['summary']['total_issues']}
- **影響ファイル数**: {report['summary']['files_affected']}
- **重要度別**:
  - 🚨 高: {report['summary']['severity_breakdown']['high']}
  - ⚠️ 中: {report['summary']['severity_breakdown']['medium']}
  - ℹ️ 低: {report['summary']['severity_breakdown']['low']}

## 🎯 改善推奨事項

"""
        
        for rec in report['recommendations']:
            content += f"- {rec}\n"
        
        content += "\n## 🚨 高優先度問題詳細\n\n"
        
        for issue in report['high_priority_issues']:
            content += f"""### {Path(issue.file_path).name}:{issue.line_number}
**問題**: {issue.description}  
**コード**: `{issue.line_content}`  
**修正提案**:
```python{issue.suggested_fix}```

---

"""
        
        # ファイル出力
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"詳細レポートを出力: {output_path}")
    
    def print_summary(self):
        """サマリーをコンソール出力"""
        report = self.generate_report()
        
        print("=" * 60)
        print("🔍 パス設定準拠監査結果")
        print("=" * 60)
        print(f"総問題数: {report['summary']['total_issues']}")
        print(f"影響ファイル数: {report['summary']['files_affected']}")
        print()
        
        print("重要度別:")
        for severity, count in report['summary']['severity_breakdown'].items():
            if count > 0:
                icon = {'high': '🚨', 'medium': '⚠️', 'low': 'ℹ️'}[severity]
                print(f"  {icon} {severity.capitalize()}: {count}")
        print()
        
        print("問題タイプ別:")
        for issue_type, count in report['summary']['issue_type_breakdown'].items():
            print(f"  - {issue_type}: {count}")
        print()
        
        if report['top_problematic_files']:
            print("最も問題のあるファイル（上位5件）:")
            for file_path, count in report['top_problematic_files'][:5]:
                print(f"  - {Path(file_path).name}: {count}件")
        print()
        
        print("🎯 推奨事項:")
        for rec in report['recommendations']:
            print(f"  {rec}")
        print("=" * 60)


def main():
    """メイン実行"""
    project_root = Path.cwd()
    auditor = PathComplianceAuditor(project_root)
    
    print("🔍 パス設定準拠監査を開始...")
    issues = auditor.scan_project()
    
    # コンソールサマリー
    auditor.print_summary()
    
    # 詳細レポート出力
    if issues:
        # OutputPathManager使用例
        try:
            from features.common.output_path_manager import OutputPathManager, OutputCategory
            manager = OutputPathManager("AUDIT")
            report_path = manager.get_output_path(
                OutputCategory.TEST_RESULT, 
                filename="path_compliance_audit_report.md"
            )
            manager.ensure_output_dir(OutputCategory.TEST_RESULT)
        except ImportError:
            # フォールバック
            report_path = Path("temp/path_compliance_audit_report.md")
        
        auditor.export_detailed_report(report_path)
    
    # 終了コード
    high_issues = [issue for issue in issues if issue.severity == 'high']
    return len(high_issues)  # 高優先度問題数を返す


if __name__ == "__main__":
    exit_code = main()
    sys.exit(min(exit_code, 99))  # 最大99で制限