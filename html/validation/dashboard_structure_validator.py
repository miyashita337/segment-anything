#!/usr/bin/env python3
"""
ダッシュボード構造自動検証ツール

目的：
- メインダッシュボード入れ子表示防止
- 左ペイン消失問題の検出・防止
- integrated_dashboard_server.py構造整合性検証
- HTML品質保証の自動化

使用方法：
python3 html/validation/dashboard_structure_validator.py --tracker-id QUAL-044
python3 html/validation/dashboard_structure_validator.py --validate-all
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import Request, urlopen
from urllib.parse import urljoin
import base64


class DashboardStructureValidator:
    def __init__(self):
        self.base_url = "http://100.123.241.106:8088"
        self.auth_header = self._create_auth_header("admin", "secure_track_2025_q3_8f9a")
        self.validation_results = []
        
    def _create_auth_header(self, username: str, password: str) -> str:
        """Basic認証ヘッダー生成"""
        credentials = f"{username}:{password}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        return f"Basic {encoded_credentials}"
    
    def validate_main_dashboard(self) -> Dict[str, any]:
        """メインダッシュボード構造検証"""
        print("🔍 メインダッシュボード構造検証開始...")
        
        results = {
            "test_name": "main_dashboard_validation",
            "passed": True,
            "errors": [],
            "warnings": []
        }
        
        try:
            # メインダッシュボードHTML取得
            html_content = self._fetch_dashboard_html("/")
            
            # 入れ子表示検証
            iframe_sources = self._extract_iframe_sources(html_content)
            nested_main = [src for src in iframe_sources if src in ["/", ""]]
            
            if nested_main:
                results["passed"] = False
                results["errors"].append({
                    "type": "nested_main_dashboard",
                    "message": f"メインダッシュボード入れ子表示検出: {nested_main}",
                    "severity": "CRITICAL"
                })
            
            # 左ペイン構造検証
            if not self._has_sidebar_structure(html_content):
                results["passed"] = False
                results["errors"].append({
                    "type": "missing_sidebar",
                    "message": "左ペイン（sidebar）構造が見つかりません",
                    "severity": "CRITICAL"
                })
            
            # 右ペイン構造検証
            if not self._has_main_content_structure(html_content):
                results["passed"] = False
                results["errors"].append({
                    "type": "missing_main_content",
                    "message": "右ペイン（main-content）構造が見つかりません",
                    "severity": "CRITICAL"
                })
            
            # ダッシュボード一覧表示検証
            dashboard_count = self._count_dashboard_links(html_content)
            if dashboard_count == 0:
                results["warnings"].append({
                    "type": "no_dashboard_links",
                    "message": "ダッシュボードリンクが見つかりません",
                    "severity": "WARNING"
                })
            
            print(f"✅ メインダッシュボード検証完了 - {'合格' if results['passed'] else '不合格'}")
            
        except Exception as e:
            results["passed"] = False
            results["errors"].append({
                "type": "validation_exception",
                "message": f"検証中にエラーが発生: {str(e)}",
                "severity": "CRITICAL"
            })
            print(f"❌ メインダッシュボード検証エラー: {e}")
        
        return results
    
    def validate_tracker_dashboard(self, tracker_id: str) -> Dict[str, any]:
        """個別トラッカーダッシュボード検証"""
        print(f"🔍 トラッカーダッシュボード検証開始: {tracker_id}")
        
        results = {
            "test_name": f"tracker_dashboard_validation_{tracker_id}",
            "tracker_id": tracker_id,
            "passed": True,
            "errors": [],
            "warnings": []
        }
        
        try:
            # トラッカーダッシュボードHTML取得
            html_content = self._fetch_dashboard_html(f"/tracker/{tracker_id}")
            
            # 左ペイン表示確認
            if not self._has_sidebar_structure(html_content):
                results["passed"] = False
                results["errors"].append({
                    "type": "sidebar_disappeared",
                    "message": f"トラッカー{tracker_id}で左ペインが消失しています",
                    "severity": "CRITICAL"
                })
            
            # トラッカーコンテンツ存在確認
            if not self._has_tracker_content(html_content, tracker_id):
                results["passed"] = False
                results["errors"].append({
                    "type": "missing_tracker_content",
                    "message": f"トラッカー{tracker_id}のコンテンツが見つかりません",
                    "severity": "HIGH"
                })
            
            # 統計分析結果確認
            stats_sections = self._count_statistical_sections(html_content)
            if stats_sections == 0:
                results["warnings"].append({
                    "type": "missing_statistical_analysis",
                    "message": f"トラッカー{tracker_id}で統計分析結果が見つかりません",
                    "severity": "MEDIUM"
                })
            
            print(f"✅ トラッカー{tracker_id}検証完了 - {'合格' if results['passed'] else '不合格'}")
            
        except Exception as e:
            results["passed"] = False
            results["errors"].append({
                "type": "validation_exception",
                "message": f"検証中にエラーが発生: {str(e)}",
                "severity": "CRITICAL"
            })
            print(f"❌ トラッカー{tracker_id}検証エラー: {e}")
        
        return results
    
    def validate_server_integration(self) -> Dict[str, any]:
        """サーバー統合性検証"""
        print("🔍 サーバー統合性検証開始...")
        
        results = {
            "test_name": "server_integration_validation",
            "passed": True,
            "errors": [],
            "warnings": []
        }
        
        try:
            # integrated_dashboard_server.pyプロセス確認
            if not self._check_integrated_server_running():
                results["passed"] = False
                results["errors"].append({
                    "type": "server_not_running",
                    "message": "integrated_dashboard_server.pyが動作していません",
                    "severity": "CRITICAL"
                })
            
            # ポート8088使用確認
            if not self._check_port_8088():
                results["passed"] = False
                results["errors"].append({
                    "type": "port_not_available",
                    "message": "ポート8088が使用されていません",
                    "severity": "HIGH"
                })
            
            # 不適切な単純HTTPサーバー確認
            if self._check_simple_http_server():
                results["warnings"].append({
                    "type": "simple_http_server_detected",
                    "message": "python3 -m http.server等の単純サーバーが検出されました",
                    "severity": "MEDIUM"
                })
            
            print(f"✅ サーバー統合性検証完了 - {'合格' if results['passed'] else '不合格'}")
            
        except Exception as e:
            results["passed"] = False
            results["errors"].append({
                "type": "validation_exception",
                "message": f"検証中にエラーが発生: {str(e)}",
                "severity": "CRITICAL"
            })
            print(f"❌ サーバー統合性検証エラー: {e}")
        
        return results
    
    def _fetch_dashboard_html(self, path: str) -> str:
        """ダッシュボードHTML取得"""
        url = urljoin(self.base_url, path)
        request = Request(url)
        request.add_header("Authorization", self.auth_header)
        
        with urlopen(request, timeout=10) as response:
            return response.read().decode('utf-8')
    
    def _extract_iframe_sources(self, html_content: str) -> List[str]:
        """iframe src属性抽出"""
        iframe_pattern = r'<iframe[^>]*src=["\']([^"\']*)["\']'
        return re.findall(iframe_pattern, html_content, re.IGNORECASE)
    
    def _has_sidebar_structure(self, html_content: str) -> bool:
        """左ペイン構造存在確認"""
        sidebar_patterns = [
            r'class="[^"]*sidebar[^"]*"',
            r'id="sidebar"',
            r'<div[^>]*sidebar[^>]*>'
        ]
        return any(re.search(pattern, html_content, re.IGNORECASE) 
                  for pattern in sidebar_patterns)
    
    def _has_main_content_structure(self, html_content: str) -> bool:
        """右ペイン構造存在確認"""
        content_patterns = [
            r'class="[^"]*main-content[^"]*"',
            r'id="main-content"',
            r'<div[^>]*main-content[^>]*>'
        ]
        return any(re.search(pattern, html_content, re.IGNORECASE) 
                  for pattern in content_patterns)
    
    def _count_dashboard_links(self, html_content: str) -> int:
        """ダッシュボードリンク数カウント"""
        link_patterns = [
            r'/tracker/[A-Z]+-\d+',
            r'href="[^"]*tracker[^"]*"'
        ]
        count = 0
        for pattern in link_patterns:
            count += len(re.findall(pattern, html_content, re.IGNORECASE))
        return count
    
    def _has_tracker_content(self, html_content: str, tracker_id: str) -> bool:
        """トラッカーコンテンツ存在確認"""
        content_indicators = [
            tracker_id,
            "dashboard.html",
            "品質スコア",
            "統計分析"
        ]
        return any(indicator in html_content for indicator in content_indicators)
    
    def _count_statistical_sections(self, html_content: str) -> int:
        """統計分析セクション数カウント"""
        stats_patterns = [
            r'統計分析結果',
            r'p値',
            r'効果サイズ',
            r'改善率'
        ]
        count = 0
        for pattern in stats_patterns:
            count += len(re.findall(pattern, html_content, re.IGNORECASE))
        return count
    
    def _check_integrated_server_running(self) -> bool:
        """integrated_dashboard_server.py動作確認"""
        try:
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            return 'integrated_dashboard_server' in result.stdout
        except:
            return False
    
    def _check_port_8088(self) -> bool:
        """ポート8088使用確認"""
        try:
            result = subprocess.run(['ss', '-tulpn'], capture_output=True, text=True)
            return ':8088' in result.stdout
        except:
            return False
    
    def _check_simple_http_server(self) -> bool:
        """単純HTTPサーバー検出"""
        try:
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            return 'http.server' in result.stdout
        except:
            return False
    
    def run_full_validation(self, tracker_ids: Optional[List[str]] = None) -> Dict[str, any]:
        """完全検証実行"""
        print("🚀 ダッシュボード構造完全検証開始...")
        start_time = time.time()
        
        validation_summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "critical_errors": 0,
            "warnings": 0,
            "results": []
        }
        
        # 1. メインダッシュボード検証
        main_result = self.validate_main_dashboard()
        validation_summary["results"].append(main_result)
        validation_summary["total_tests"] += 1
        
        if main_result["passed"]:
            validation_summary["passed_tests"] += 1
        else:
            validation_summary["failed_tests"] += 1
        
        # 2. サーバー統合性検証
        server_result = self.validate_server_integration()
        validation_summary["results"].append(server_result)
        validation_summary["total_tests"] += 1
        
        if server_result["passed"]:
            validation_summary["passed_tests"] += 1
        else:
            validation_summary["failed_tests"] += 1
        
        # 3. 個別トラッカー検証
        if tracker_ids:
            for tracker_id in tracker_ids:
                tracker_result = self.validate_tracker_dashboard(tracker_id)
                validation_summary["results"].append(tracker_result)
                validation_summary["total_tests"] += 1
                
                if tracker_result["passed"]:
                    validation_summary["passed_tests"] += 1
                else:
                    validation_summary["failed_tests"] += 1
        
        # エラー・警告集計
        for result in validation_summary["results"]:
            for error in result["errors"]:
                if error["severity"] == "CRITICAL":
                    validation_summary["critical_errors"] += 1
            validation_summary["warnings"] += len(result["warnings"])
        
        duration = time.time() - start_time
        validation_summary["duration_seconds"] = round(duration, 2)
        
        # 結果出力
        self._print_validation_summary(validation_summary)
        
        return validation_summary
    
    def _print_validation_summary(self, summary: Dict[str, any]) -> None:
        """検証結果サマリー出力"""
        print("\n" + "="*60)
        print("📊 ダッシュボード構造検証結果サマリー")
        print("="*60)
        
        print(f"⏰ 実行時刻: {summary['timestamp']}")
        print(f"⏱️ 所要時間: {summary['duration_seconds']}秒")
        print()
        
        print(f"📋 テスト実行数: {summary['total_tests']}")
        print(f"✅ 合格テスト: {summary['passed_tests']}")
        print(f"❌ 失敗テスト: {summary['failed_tests']}")
        print(f"🚨 重大エラー: {summary['critical_errors']}")
        print(f"⚠️ 警告: {summary['warnings']}")
        print()
        
        # 個別結果表示
        for result in summary['results']:
            status_icon = "✅" if result['passed'] else "❌"
            print(f"{status_icon} {result['test_name']}")
            
            if result['errors']:
                for error in result['errors']:
                    severity_icon = "🚨" if error['severity'] == "CRITICAL" else "⚠️"
                    print(f"    {severity_icon} {error['message']}")
        
        print()
        
        # 総合判定
        overall_passed = summary['failed_tests'] == 0 and summary['critical_errors'] == 0
        if overall_passed:
            print("🎉 **総合判定: 合格** - ダッシュボード構造に問題はありません")
        else:
            print("❌ **総合判定: 不合格** - 修正が必要な問題が検出されました")
        
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="ダッシュボード構造自動検証ツール")
    parser.add_argument('--tracker-id', type=str, help='検証対象トラッカーID')
    parser.add_argument('--validate-all', action='store_true', help='全項目検証実行')
    parser.add_argument('--main-only', action='store_true', help='メインダッシュボードのみ検証')
    parser.add_argument('--output', type=str, help='結果出力JSONファイル')
    
    args = parser.parse_args()
    
    validator = DashboardStructureValidator()
    
    if args.main_only:
        result = validator.validate_main_dashboard()
        results = {"results": [result]}
    elif args.tracker_id:
        results = validator.run_full_validation([args.tracker_id])
    else:
        # デフォルト：メイン+サーバー統合性のみ
        results = validator.run_full_validation()
    
    # JSON出力（オプション）
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"📄 結果をJSONファイルに出力: {args.output}")
    
    # 終了コード設定
    if 'critical_errors' in results and results['critical_errors'] > 0:
        sys.exit(1)  # 重大エラーありで異常終了
    elif 'failed_tests' in results and results['failed_tests'] > 0:
        sys.exit(2)  # 失敗テストありで警告終了
    else:
        sys.exit(0)  # 正常終了


if __name__ == "__main__":
    main()