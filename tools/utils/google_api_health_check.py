#!/usr/bin/env python3
"""
Google API接続ヘルスチェックスクリプト
環境診断と自動修復機能
"""

import sys
import subprocess
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class GoogleAPIHealthChecker:
    """Google API接続ヘルスチェッカー"""
    
    REQUIRED_PACKAGES = [
        'google-auth',
        'google-auth-oauthlib', 
        'google-auth-httplib2',
        'google-api-python-client'
    ]
    
    AUTH_FILE_CANDIDATES = [
        'config/google_sheets_auth.json',
        'config/google-service-account.json',
        'google_sheets_auth.json'
    ]
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = {}
        
    def log(self, message: str, level: str = "INFO"):
        """ログ出力"""
        if self.verbose:
            emoji = {"INFO": "ℹ️", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌"}
            print(f"{emoji.get(level, '📋')} {message}")
    
    def check_python_environment(self) -> Dict:
        """Python環境チェック"""
        self.log("Python環境チェック中...")
        
        result = {
            "executable": sys.executable,
            "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "is_sam_env": "sam-env" in sys.executable,
            "is_serena_env": "serena" in sys.executable or "mcp" in sys.executable
        }
        
        if result["is_sam_env"]:
            self.log("sam-env環境で実行中", "SUCCESS")
        elif result["is_serena_env"]:
            self.log("Serena環境で実行中 - sam-envに切り替えることを推奨", "WARNING")
        else:
            self.log(f"不明な環境: {sys.executable}", "WARNING")
            
        return result
    
    def check_google_packages(self) -> Dict:
        """Google APIパッケージチェック"""
        self.log("Google APIパッケージチェック中...")
        
        installed_packages = {}
        missing_packages = []
        
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                                  capture_output=True, text=True, check=True)
            pip_output = result.stdout.lower()
            
            for package in self.REQUIRED_PACKAGES:
                if package.lower() in pip_output:
                    # バージョン抽出
                    for line in result.stdout.split('\n'):
                        if line.lower().startswith(package.lower()):
                            installed_packages[package] = line.split()[1] if len(line.split()) > 1 else "unknown"
                            break
                    else:
                        installed_packages[package] = "installed"
                    self.log(f"{package}: インストール済み", "SUCCESS")
                else:
                    missing_packages.append(package)
                    self.log(f"{package}: 未インストール", "ERROR")
                    
        except subprocess.CalledProcessError as e:
            self.log(f"pip list実行エラー: {e}", "ERROR")
            return {"error": str(e)}
            
        return {
            "installed": installed_packages,
            "missing": missing_packages,
            "all_required_installed": len(missing_packages) == 0
        }
    
    def check_import_capability(self) -> Dict:
        """インポート可能性チェック"""
        self.log("Google APIインポートテスト中...")
        
        import_results = {}
        
        test_imports = [
            ('google.auth', 'Google認証ライブラリ'),
            ('google.auth.transport.requests', 'Google認証トランスポート'),
            ('google.oauth2.service_account', 'サービスアカウント認証'),
            ('googleapiclient.discovery', 'Google API Client'),
        ]
        
        for module_name, description in test_imports:
            try:
                __import__(module_name)
                import_results[module_name] = True
                self.log(f"{description}: インポート成功", "SUCCESS")
            except ImportError as e:
                import_results[module_name] = False
                self.log(f"{description}: インポート失敗 - {e}", "ERROR")
                
        return {
            "results": import_results,
            "all_imports_successful": all(import_results.values())
        }
    
    def check_auth_files(self) -> Dict:
        """認証ファイルチェック"""
        self.log("認証ファイルチェック中...")
        
        found_files = []
        valid_files = []
        
        for file_path in self.AUTH_FILE_CANDIDATES:
            if os.path.exists(file_path):
                found_files.append(file_path)
                self.log(f"認証ファイル発見: {file_path}", "SUCCESS")
                
                # ファイル内容検証
                try:
                    with open(file_path, 'r') as f:
                        content = json.load(f)
                        if 'private_key' in content and 'client_email' in content:
                            valid_files.append(file_path)
                            self.log(f"認証ファイル有効: {file_path}", "SUCCESS")
                        else:
                            self.log(f"認証ファイル形式エラー: {file_path}", "WARNING")
                except (json.JSONDecodeError, Exception) as e:
                    self.log(f"認証ファイル読み込みエラー ({file_path}): {e}", "ERROR")
            else:
                self.log(f"認証ファイル未発見: {file_path}", "WARNING")
        
        return {
            "found_files": found_files,
            "valid_files": valid_files,
            "has_valid_auth": len(valid_files) > 0
        }
    
    def test_sheets_connection(self) -> Dict:
        """Google Sheets接続テスト"""
        self.log("Google Sheets接続テスト中...")
        
        try:
            # プロジェクト固有のインポート
            sys.path.append(str(Path.cwd()))
            from tools.progress_tracker.data_models import ProgressTrackerConfig
            from tools.progress_tracker.sheets_client import GoogleSheetsClient
            
            # 認証ファイル検索
            auth_file = None
            for candidate in self.AUTH_FILE_CANDIDATES:
                if os.path.exists(candidate):
                    auth_file = candidate
                    break
            
            if not auth_file:
                return {"success": False, "error": "認証ファイルが見つかりません"}
            
            # 接続テスト
            config = ProgressTrackerConfig(
                spreadsheet_id='1gNNPT2Z8qLLJxV-cqJE3vNFVoUvKqKF9SPVpKRqPJKI',
                sheet_name='シート1',
                auth_file_path=auth_file
            )
            
            client = GoogleSheetsClient(config)
            tasks = client.get_all_tasks()
            
            self.log(f"Google Sheets接続成功: {len(tasks)}件のタスク取得", "SUCCESS")
            return {
                "success": True,
                "task_count": len(tasks),
                "auth_file_used": auth_file
            }
            
        except Exception as e:
            self.log(f"Google Sheets接続エラー: {e}", "ERROR")
            return {"success": False, "error": str(e)}
    
    def auto_fix_missing_packages(self) -> bool:
        """不足パッケージの自動修復"""
        package_check = self.check_google_packages()
        
        if package_check.get("all_required_installed", False):
            self.log("全パッケージ既にインストール済み", "SUCCESS")
            return True
        
        missing = package_check.get("missing", [])
        if not missing:
            return True
            
        self.log(f"不足パッケージの自動インストール開始: {', '.join(missing)}")
        
        try:
            cmd = [sys.executable, '-m', 'pip', 'install'] + missing
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.log("パッケージインストール成功", "SUCCESS")
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"パッケージインストール失敗: {e}", "ERROR")
            return False
    
    def run_full_health_check(self, auto_fix: bool = False) -> Dict:
        """完全ヘルスチェック実行"""
        self.log("=== Google API接続ヘルスチェック開始 ===")
        
        results = {
            "timestamp": os.popen("date").read().strip(),
            "python_env": self.check_python_environment(),
            "google_packages": self.check_google_packages(),
            "import_capability": self.check_import_capability(),
            "auth_files": self.check_auth_files(),
            "sheets_connection": self.test_sheets_connection()
        }
        
        # 自動修復
        if auto_fix and not results["google_packages"].get("all_required_installed", False):
            self.log("自動修復モード: 不足パッケージをインストール中...")
            fix_success = self.auto_fix_missing_packages()
            if fix_success:
                # 修復後再チェック
                results["google_packages_after_fix"] = self.check_google_packages()
                results["import_capability_after_fix"] = self.check_import_capability()
                results["sheets_connection_after_fix"] = self.test_sheets_connection()
        
        # 総合評価
        all_good = (
            results["google_packages"].get("all_required_installed", False) and
            results["import_capability"].get("all_imports_successful", False) and
            results["auth_files"].get("has_valid_auth", False) and
            results["sheets_connection"].get("success", False)
        )
        
        results["overall_health"] = "HEALTHY" if all_good else "ISSUES_DETECTED"
        
        self.log("=== ヘルスチェック完了 ===")
        self.log(f"総合状態: {'✅ 正常' if all_good else '❌ 問題あり'}")
        
        return results
    
    def generate_health_report(self, results: Dict) -> str:
        """ヘルスチェックレポート生成"""
        report_lines = [
            "# Google API接続ヘルスチェックレポート",
            f"生成日時: {results.get('timestamp', 'N/A')}",
            "",
            f"## 総合状態: {results.get('overall_health', 'UNKNOWN')}",
            ""
        ]
        
        # 各項目の詳細
        if "python_env" in results:
            env = results["python_env"]
            report_lines.extend([
                "## Python環境",
                f"- 実行ファイル: {env.get('executable', 'N/A')}",
                f"- バージョン: {env.get('version', 'N/A')}",
                f"- sam-env環境: {'✅' if env.get('is_sam_env') else '❌'}",
                ""
            ])
        
        if "google_packages" in results:
            pkg = results["google_packages"]
            report_lines.extend([
                "## Google APIパッケージ",
                f"- 全パッケージインストール済み: {'✅' if pkg.get('all_required_installed') else '❌'}",
                f"- インストール済み: {len(pkg.get('installed', {}))}件",
                f"- 不足: {len(pkg.get('missing', []))}件",
                ""
            ])
            
            if pkg.get("missing"):
                report_lines.append(f"- 不足パッケージ: {', '.join(pkg['missing'])}")
                report_lines.append("")
        
        if "sheets_connection" in results:
            conn = results["sheets_connection"]
            report_lines.extend([
                "## Google Sheets接続",
                f"- 接続成功: {'✅' if conn.get('success') else '❌'}",
                f"- 取得タスク数: {conn.get('task_count', 'N/A')}件" if conn.get('success') else f"- エラー: {conn.get('error', 'N/A')}",
                ""
            ])
        
        return "\n".join(report_lines)


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Google API接続ヘルスチェック")
    parser.add_argument("--auto-fix", action="store_true", help="自動修復モードを有効にする")
    parser.add_argument("--quiet", action="store_true", help="詳細出力を抑制")
    parser.add_argument("--report-file", help="レポートファイル出力先")
    
    args = parser.parse_args()
    
    checker = GoogleAPIHealthChecker(verbose=not args.quiet)
    results = checker.run_full_health_check(auto_fix=args.auto_fix)
    
    if args.report_file:
        report = checker.generate_health_report(results)
        with open(args.report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"📄 レポートを保存しました: {args.report_file}")
    
    # 終了コード設定
    sys.exit(0 if results.get("overall_health") == "HEALTHY" else 1)


if __name__ == "__main__":
    main()