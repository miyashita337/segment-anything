#!/usr/bin/env python3
"""
トラッカー完了検証スクリプト
4回目仕様違反の恒久対策として作成
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import logging

# config モジュールのパスを追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.workspace_config import WorkspaceConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TrackerCompletionValidator:
    """トラッカー完了検証クラス"""
    
    @property
    def WORKSPACE_BASE(self) -> Path:
        return WorkspaceConfig.get_workspace_root()
    
    REQUIRED_DIRECTORIES = [
        "extraction",
        "quality", 
        "dashboard",
        "tests"
    ]
    
    REQUIRED_FILES = [
        "dashboard/dashboard.html",
        "quality/unified_quality_report.json"
    ]
    
    def __init__(self, tracker_id: str):
        """
        初期化
        
        Args:
            tracker_id: 検証対象のトラッカーID
        """
        self.tracker_id = tracker_id
        self.workspace_path = self.WORKSPACE_BASE / tracker_id
        self.validation_results = {}
    
    def validate_workspace_structure(self) -> Dict[str, bool]:
        """ワークスペース構造検証"""
        results = {
            "workspace_exists": self.workspace_path.exists(),
            "required_directories": {},
            "required_files": {}
        }
        
        # ディレクトリ存在確認
        for directory in self.REQUIRED_DIRECTORIES:
            dir_path = self.workspace_path / directory
            results["required_directories"][directory] = dir_path.exists()
        
        # 必須ファイル存在確認
        for file_path in self.REQUIRED_FILES:
            full_path = self.workspace_path / file_path
            results["required_files"][file_path] = full_path.exists()
        
        return results
    
    def validate_extraction_results(self) -> Dict[str, any]:
        """抽出結果検証"""
        extraction_dir = self.workspace_path / "extraction"
        
        if not extraction_dir.exists():
            return {"status": "missing", "file_count": 0}
        
        # 抽出ファイル数確認
        image_files = list(extraction_dir.glob("*.jpg")) + list(extraction_dir.glob("*.png"))
        
        return {
            "status": "exists",
            "file_count": len(image_files),
            "has_results": len(image_files) > 0
        }
    
    def validate_quality_reports(self) -> Dict[str, any]:
        """品質レポート検証"""
        quality_dir = self.workspace_path / "quality"
        
        if not quality_dir.exists():
            return {"status": "missing", "reports": []}
        
        # JSONレポートファイル確認
        json_files = list(quality_dir.glob("*.json"))
        
        reports_info = []
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    reports_info.append({
                        "file": json_file.name,
                        "size": json_file.stat().st_size,
                        "valid_json": True,
                        "has_content": len(str(data)) > 10
                    })
            except Exception as e:
                reports_info.append({
                    "file": json_file.name,
                    "size": json_file.stat().st_size,
                    "valid_json": False,
                    "error": str(e)
                })
        
        return {
            "status": "exists",
            "reports": reports_info,
            "report_count": len(json_files)
        }
    
    def validate_dashboard(self) -> Dict[str, any]:
        """ダッシュボード検証"""
        dashboard_dir = self.workspace_path / "dashboard"
        dashboard_html = dashboard_dir / "dashboard.html"
        
        if not dashboard_dir.exists():
            return {"status": "missing", "html_exists": False}
        
        if not dashboard_html.exists():
            return {"status": "incomplete", "html_exists": False}
        
        # HTMLファイルサイズ確認
        html_size = dashboard_html.stat().st_size
        
        return {
            "status": "complete",
            "html_exists": True,
            "html_size": html_size,
            "html_path": str(dashboard_html),
            "accessible": html_size > 1000  # 1KB以上で有効とみなす
        }
    
    def run_comprehensive_validation(self) -> Dict[str, any]:
        """包括的検証実行"""
        logger.info(f"🔍 トラッカー完了検証開始: {self.tracker_id}")
        
        validation_results = {
            "tracker_id": self.tracker_id,
            "validation_timestamp": str(Path(__file__).stat().st_mtime),
            "workspace_structure": self.validate_workspace_structure(),
            "extraction_results": self.validate_extraction_results(),
            "quality_reports": self.validate_quality_reports(),
            "dashboard": self.validate_dashboard()
        }
        
        # 総合判定
        overall_status = self._calculate_overall_status(validation_results)
        validation_results["overall_status"] = overall_status
        
        self.validation_results = validation_results
        return validation_results
    
    def _calculate_overall_status(self, results: Dict) -> Dict[str, any]:
        """総合ステータス計算"""
        checks = {
            "workspace_exists": results["workspace_structure"]["workspace_exists"],
            "all_directories": all(results["workspace_structure"]["required_directories"].values()),
            "critical_files": all(results["workspace_structure"]["required_files"].values()),
            "has_extractions": results["extraction_results"]["has_results"] if results["extraction_results"]["status"] == "exists" else False,
            "has_quality_reports": results["quality_reports"]["report_count"] > 0 if results["quality_reports"]["status"] == "exists" else False,
            "dashboard_accessible": results["dashboard"]["accessible"] if results["dashboard"]["status"] == "complete" else False
        }
        
        passed_checks = sum(checks.values())
        total_checks = len(checks)
        completion_rate = passed_checks / total_checks
        
        if completion_rate >= 1.0:
            status = "COMPLETE"
        elif completion_rate >= 0.8:
            status = "MOSTLY_COMPLETE"
        elif completion_rate >= 0.5:
            status = "PARTIAL"
        else:
            status = "INCOMPLETE"
        
        return {
            "status": status,
            "completion_rate": completion_rate,
            "passed_checks": passed_checks,
            "total_checks": total_checks,
            "failed_checks": [check for check, result in checks.items() if not result]
        }
    
    def print_validation_report(self):
        """検証レポート出力"""
        if not self.validation_results:
            print("❌ 検証未実行")
            return
        
        results = self.validation_results
        overall = results["overall_status"]
        
        print(f"\n{'='*60}")
        print(f"🔍 トラッカー完了検証レポート: {self.tracker_id}")
        print(f"{'='*60}")
        
        # 総合ステータス
        status_emoji = {
            "COMPLETE": "✅",
            "MOSTLY_COMPLETE": "🟡", 
            "PARTIAL": "🟠",
            "INCOMPLETE": "❌"
        }
        
        print(f"\n📊 総合ステータス: {status_emoji.get(overall['status'], '❓')} {overall['status']}")
        print(f"📈 完了率: {overall['completion_rate']:.1%} ({overall['passed_checks']}/{overall['total_checks']})")
        
        if overall['failed_checks']:
            print(f"\n❌ 失敗チェック:")
            for check in overall['failed_checks']:
                print(f"   - {check}")
        
        # 詳細情報
        print(f"\n📁 ワークスペース: {self.workspace_path}")
        print(f"   存在: {'✅' if results['workspace_structure']['workspace_exists'] else '❌'}")
        
        print(f"\n📂 ディレクトリ構造:")
        for directory, exists in results["workspace_structure"]["required_directories"].items():
            print(f"   {directory}/: {'✅' if exists else '❌'}")
        
        print(f"\n📄 重要ファイル:")
        for file_path, exists in results["workspace_structure"]["required_files"].items():
            print(f"   {file_path}: {'✅' if exists else '❌'}")
        
        # 抽出結果
        extraction = results["extraction_results"]
        if extraction["status"] == "exists":
            print(f"\n🎯 抽出結果: {extraction['file_count']}ファイル")
        else:
            print(f"\n🎯 抽出結果: ❌ 未実行")
        
        # 品質レポート
        quality = results["quality_reports"]
        if quality["status"] == "exists":
            print(f"\n📊 品質レポート: {quality['report_count']}ファイル")
        else:
            print(f"\n📊 品質レポート: ❌ 未生成")
        
        # ダッシュボード
        dashboard = results["dashboard"]
        if dashboard["status"] == "complete":
            print(f"\n🌐 ダッシュボード: ✅ 利用可能")
            print(f"   パス: file://{dashboard['html_path']}")
        else:
            print(f"\n🌐 ダッシュボード: ❌ 未生成")
        
        print(f"\n{'='*60}")
        
        # 推奨アクション
        if overall['status'] != "COMPLETE":
            print(f"\n💡 推奨アクション:")
            if not results['workspace_structure']['workspace_exists']:
                print(f"   1. ワークスペース作成: mkdir -p {self.workspace_path}")
            if 'dashboard_accessible' in overall['failed_checks']:
                print(f"   2. 品質ワークフロー実行: ./tools/scripts/run_quality_workflow.sh {self.tracker_id}")
            if 'has_extractions' in overall['failed_checks']:
                print(f"   3. 抽出パイプライン実行確認")


def main():
    """メイン実行"""
    if len(sys.argv) < 2:
        print("使用法: python3 validate_tracker_completion.py <TRACKER_ID>")
        print("例: python3 validate_tracker_completion.py P1-005")
        sys.exit(1)
    
    tracker_id = sys.argv[1]
    
    validator = TrackerCompletionValidator(tracker_id)
    validation_results = validator.run_comprehensive_validation()
    validator.print_validation_report()
    
    # 完了していない場合は終了コード1
    if validation_results["overall_status"]["status"] != "COMPLETE":
        sys.exit(1)
    
    print(f"\n🎉 {tracker_id} 完了検証: 成功")


if __name__ == "__main__":
    main()