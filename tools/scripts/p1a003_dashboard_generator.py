#!/usr/bin/env python3
"""
P1-A003ダッシュボード生成
自動テスト強化システムの可視化ダッシュボード
"""

import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# プロジェクトルート設定
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


class P1A003Dashboard:
    """P1-A003ダッシュボード生成システム"""

    def __init__(self):
        """初期化"""
        # PROGRESS_TRACKER.md仕様準拠の正しいパス
        self.workspace = Path(
            "/mnt/c/AItools/lora/train/yado/clipped_boundingbox/workspace/P1-A003"
        )
        self.workspace.mkdir(parents=True, exist_ok=True)

        self.test_results_dir = project_root / "test_results" / "quality"
        self.baselines_dir = Path(
            "/mnt/c/AItools/lora/train/yado/clipped_boundingbox/workspace/baseline"
        )

        self.dashboard_data = {
            "generated_at": datetime.now().isoformat(),
            "dashboard_id": f"P1A003_dashboard_{datetime.now():%Y%m%d_%H%M%S}",
            "system_status": {},
            "quality_metrics": {},
            "test_history": [],
            "alerts": [],
            "recommendations": [],
        }

    def generate_dashboard(self) -> Dict[str, Any]:
        """ダッシュボード生成"""
        print("📊 P1-A003: 自動テスト強化システム ダッシュボード生成開始")
        print(f"ダッシュボードID: {self.dashboard_data['dashboard_id']}")
        print("=" * 70)

        # データ収集
        self._collect_system_status()
        self._collect_quality_metrics()
        self._collect_test_history()
        self._analyze_trends()
        self._generate_alerts()
        self._generate_recommendations()

        # ダッシュボードファイル生成
        self._generate_html_dashboard()
        self._generate_json_report()
        self._generate_markdown_summary()

        print("\n📊 ダッシュボード生成完了")
        return self.dashboard_data

    def _collect_system_status(self):
        """システム状態収集"""
        print("🔍 システム状態収集中...")

        status = {
            "baseline_exists": self.baselines_dir.exists(),
            "test_results_count": 0,
            "monitoring_active": False,
            "last_test_time": None,
            "system_health": "UNKNOWN",
        }

        # テスト結果数カウント
        if self.test_results_dir.exists():
            test_files = list(self.test_results_dir.glob("test_kana08_*.json"))
            status["test_results_count"] = len(test_files)

            # 最新テスト時刻
            if test_files:
                latest_file = max(test_files, key=lambda f: f.stat().st_mtime)
                latest_time = datetime.fromtimestamp(latest_file.stat().st_mtime)
                status["last_test_time"] = latest_time.isoformat()

                # システム健全性判定
                time_diff = datetime.now() - latest_time
                if time_diff < timedelta(hours=1):
                    status["system_health"] = "HEALTHY"
                elif time_diff < timedelta(hours=24):
                    status["system_health"] = "WARNING"
                else:
                    status["system_health"] = "CRITICAL"

        # 監視状態確認
        monitoring_config = self.workspace / "monitoring_config.json"
        if monitoring_config.exists():
            try:
                with open(monitoring_config, "r", encoding="utf-8") as f:
                    config = json.load(f)
                status["monitoring_active"] = config.get("enabled", False)
            except:
                pass

        self.dashboard_data["system_status"] = status
        print(f"  システム健全性: {status['system_health']}")
        print(f"  テスト結果数: {status['test_results_count']}")

    def _collect_quality_metrics(self):
        """品質メトリクス収集"""
        print("📈 品質メトリクス収集中...")

        metrics = {"baseline_metrics": {}, "latest_metrics": {}, "trend_analysis": {}}

        # ベースライン読み込み
        baseline_file = self.baselines_dir / "kana08_baseline.json"
        if baseline_file.exists():
            try:
                with open(baseline_file, "r", encoding="utf-8") as f:
                    baseline_data = json.load(f)
                metrics["baseline_metrics"] = {
                    "ab_evaluation_rate": baseline_data.get("ab_evaluation_rate", 0),
                    "sci_score": baseline_data.get("sci_score", 0),
                    "pla_score": baseline_data.get("pla_score", 0),
                    "ple_score": baseline_data.get("ple_score", 0),
                    "success_count": baseline_data.get("success_count", 0),
                    "total_processed": baseline_data.get("total_processed", 0),
                }
            except Exception as e:
                print(f"  ベースライン読み込みエラー: {e}")

        # 最新テスト結果読み込み
        if self.test_results_dir.exists():
            test_files = list(self.test_results_dir.glob("test_kana08_*.json"))
            if test_files:
                latest_file = max(test_files, key=lambda f: f.stat().st_mtime)
                try:
                    with open(latest_file, "r", encoding="utf-8") as f:
                        latest_data = json.load(f)

                    current = latest_data.get("current", {})
                    metrics["latest_metrics"] = {
                        "ab_evaluation_rate": current.get("ab_evaluation_rate", 0),
                        "sci_score": current.get("sci_score", 0),
                        "pla_score": current.get("pla_score", 0),
                        "ple_score": current.get("ple_score", 0),
                        "success_count": current.get("success_count", 0),
                        "total_processed": current.get("total_processed", 0),
                        "test_status": latest_data.get("status", "UNKNOWN"),
                        "degradation_detected": latest_data.get("degradation_detected", False),
                    }
                except Exception as e:
                    print(f"  最新テスト結果読み込みエラー: {e}")

        self.dashboard_data["quality_metrics"] = metrics

        baseline = metrics["baseline_metrics"]
        latest = metrics["latest_metrics"]
        print(f"  ベースライン A/B評価率: {baseline.get('ab_evaluation_rate', 0):.1f}%")
        print(f"  最新 A/B評価率: {latest.get('ab_evaluation_rate', 0):.1f}%")

    def _collect_test_history(self):
        """テスト履歴収集"""
        print("📋 テスト履歴収集中...")

        history = []

        if self.test_results_dir.exists():
            test_files = sorted(
                self.test_results_dir.glob("test_kana08_*.json"),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )

            for test_file in test_files[:10]:  # 最新10件
                try:
                    with open(test_file, "r", encoding="utf-8") as f:
                        test_data = json.load(f)

                    history.append(
                        {
                            "test_id": test_data.get("test_id", ""),
                            "timestamp": test_data.get("timestamp", ""),
                            "status": test_data.get("status", "UNKNOWN"),
                            "degradation_detected": test_data.get("degradation_detected", False),
                            "degradation_count": len(test_data.get("degradation_details", [])),
                            "ab_evaluation_rate": test_data.get("current", {}).get(
                                "ab_evaluation_rate", 0
                            ),
                        }
                    )
                except Exception as e:
                    print(f"  テスト履歴読み込みエラー ({test_file.name}): {e}")

        self.dashboard_data["test_history"] = history
        print(f"  テスト履歴: {len(history)}件")

    def _analyze_trends(self):
        """トレンド分析"""
        print("📊 トレンド分析中...")

        history = self.dashboard_data["test_history"]
        if len(history) < 2:
            return

        # A/B評価率のトレンド
        ab_rates = [
            test.get("ab_evaluation_rate", 0) for test in history if test.get("ab_evaluation_rate")
        ]
        if len(ab_rates) >= 2:
            trend = (
                "IMPROVING"
                if ab_rates[0] > ab_rates[-1]
                else "DECLINING"
                if ab_rates[0] < ab_rates[-1]
                else "STABLE"
            )
            avg_rate = sum(ab_rates) / len(ab_rates)

            self.dashboard_data["quality_metrics"]["trend_analysis"] = {
                "ab_evaluation_trend": trend,
                "average_ab_rate": avg_rate,
                "recent_degradations": sum(
                    1 for test in history[:5] if test.get("degradation_detected")
                ),
                "stability_score": (
                    5 - sum(1 for test in history[:5] if test.get("degradation_detected"))
                )
                * 20,
            }

            print(f"  A/B評価率トレンド: {trend}")
            print(
                f"  安定性スコア: {self.dashboard_data['quality_metrics']['trend_analysis']['stability_score']}/100"
            )

    def _generate_alerts(self):
        """アラート生成"""
        print("🚨 アラート分析中...")

        alerts = []
        system_status = self.dashboard_data["system_status"]
        quality_metrics = self.dashboard_data["quality_metrics"]

        # システム健全性アラート
        if system_status.get("system_health") == "CRITICAL":
            alerts.append(
                {
                    "level": "CRITICAL",
                    "type": "SYSTEM_HEALTH",
                    "message": "24時間以上テストが実行されていません",
                    "action": "自動テストシステムの動作確認が必要です",
                }
            )
        elif system_status.get("system_health") == "WARNING":
            alerts.append(
                {
                    "level": "WARNING",
                    "type": "SYSTEM_HEALTH",
                    "message": "1時間以上テストが実行されていません",
                    "action": "定期テストの実行状況を確認してください",
                }
            )

        # 品質劣化アラート
        latest = quality_metrics.get("latest_metrics", {})
        if latest.get("degradation_detected"):
            alerts.append(
                {
                    "level": "WARNING",
                    "type": "QUALITY_DEGRADATION",
                    "message": "最新テストで品質劣化が検出されました",
                    "action": "劣化詳細を確認し、対策を検討してください",
                }
            )

        # トレンドアラート
        trend_analysis = quality_metrics.get("trend_analysis", {})
        stability_score = trend_analysis.get("stability_score", 100)
        if stability_score < 60:
            alerts.append(
                {
                    "level": "WARNING",
                    "type": "STABILITY_TREND",
                    "message": f"安定性スコアが低下しています ({stability_score}/100)",
                    "action": "連続する品質劣化の原因調査が必要です",
                }
            )

        self.dashboard_data["alerts"] = alerts
        print(f"  アラート: {len(alerts)}件")

    def _generate_recommendations(self):
        """推奨事項生成"""
        print("💡 推奨事項生成中...")

        recommendations = []
        system_status = self.dashboard_data["system_status"]

        # 監視推奨
        if not system_status.get("monitoring_active"):
            recommendations.append(
                {
                    "priority": "HIGH",
                    "category": "MONITORING",
                    "title": "継続監視の有効化",
                    "description": "24時間継続監視を有効化して、品質劣化の早期検出を実現しましょう",
                    "action": "monitoring_config.jsonでenabledをtrueに設定",
                }
            )

        # テスト頻度推奨
        test_count = system_status.get("test_results_count", 0)
        if test_count < 5:
            recommendations.append(
                {
                    "priority": "MEDIUM",
                    "category": "TESTING",
                    "title": "テスト実行頻度の増加",
                    "description": "より多くのテストデータを蓄積して、トレンド分析の精度を向上させましょう",
                    "action": "定期的なquickテストの実行を検討",
                }
            )

        # 品質改善推奨
        latest = self.dashboard_data["quality_metrics"].get("latest_metrics", {})
        ab_rate = latest.get("ab_evaluation_rate", 0)
        if ab_rate < 80:
            recommendations.append(
                {
                    "priority": "HIGH",
                    "category": "QUALITY",
                    "title": "A/B評価率の改善",
                    "description": f"現在のA/B評価率 ({ab_rate:.1f}%) を80%以上に改善することを推奨します",
                    "action": "YOLO検出パラメータの調整またはSAM分割精度の向上",
                }
            )

        self.dashboard_data["recommendations"] = recommendations
        print(f"  推奨事項: {len(recommendations)}件")

    def _generate_html_dashboard(self):
        """HTMLダッシュボード生成"""
        dashboard_file = self.workspace / f"P1A003_Dashboard_{datetime.now():%Y%m%d_%H%M%S}.html"

        html_content = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>P1-A003: 自動テスト強化システム ダッシュボード</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; }}
        .card {{ background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric {{ display: inline-block; margin: 10px 20px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #667eea; }}
        .metric-label {{ color: #666; font-size: 0.9em; }}
        .status-healthy {{ color: #28a745; }}
        .status-warning {{ color: #ffc107; }}
        .status-critical {{ color: #dc3545; }}
        .alert {{ padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .alert-critical {{ background-color: #f8d7da; border-left: 4px solid #dc3545; }}
        .alert-warning {{ background-color: #fff3cd; border-left: 4px solid #ffc107; }}
        .recommendation {{ padding: 15px; margin: 10px 0; background-color: #e7f3ff; border-left: 4px solid #007bff; border-radius: 5px; }}
        .test-history {{ max-height: 300px; overflow-y: auto; }}
        .test-item {{ padding: 10px; border-bottom: 1px solid #eee; }}
        .test-pass {{ color: #28a745; }}
        .test-warning {{ color: #ffc107; }}
        .test-fail {{ color: #dc3545; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧪 P1-A003: 自動テスト強化システム</h1>
            <h2>品質劣化事前検出ダッシュボード</h2>
            <p>生成日時: {datetime.now():%Y-%m-%d %H:%M:%S}</p>
        </div>
        
        <div class="card">
            <h3>📊 システム状態</h3>
            <div class="metric">
                <div class="metric-value status-{self.dashboard_data['system_status'].get('system_health', 'unknown').lower()}">{self.dashboard_data['system_status'].get('system_health', 'UNKNOWN')}</div>
                <div class="metric-label">システム健全性</div>
            </div>
            <div class="metric">
                <div class="metric-value">{self.dashboard_data['system_status'].get('test_results_count', 0)}</div>
                <div class="metric-label">テスト結果数</div>
            </div>
            <div class="metric">
                <div class="metric-value">{'🟢' if self.dashboard_data['system_status'].get('monitoring_active') else '🔴'}</div>
                <div class="metric-label">継続監視</div>
            </div>
        </div>
        
        <div class="card">
            <h3>📈 品質メトリクス</h3>
            <div class="metric">
                <div class="metric-value">{self.dashboard_data['quality_metrics'].get('latest_metrics', {}).get('ab_evaluation_rate', 0):.1f}%</div>
                <div class="metric-label">A/B評価率</div>
            </div>
            <div class="metric">
                <div class="metric-value">{self.dashboard_data['quality_metrics'].get('latest_metrics', {}).get('sci_score', 0):.3f}</div>
                <div class="metric-label">SCI スコア</div>
            </div>
            <div class="metric">
                <div class="metric-value">{self.dashboard_data['quality_metrics'].get('latest_metrics', {}).get('pla_score', 0):.3f}</div>
                <div class="metric-label">PLA スコア</div>
            </div>
            <div class="metric">
                <div class="metric-value">{self.dashboard_data['quality_metrics'].get('trend_analysis', {}).get('stability_score', 0)}/100</div>
                <div class="metric-label">安定性スコア</div>
            </div>
        </div>"""

        # アラート表示
        if self.dashboard_data["alerts"]:
            html_content += """
        <div class="card">
            <h3>🚨 アラート</h3>"""
            for alert in self.dashboard_data["alerts"]:
                level_class = "alert-critical" if alert["level"] == "CRITICAL" else "alert-warning"
                html_content += f"""
            <div class="alert {level_class}">
                <strong>{alert["level"]}: {alert["type"]}</strong><br>
                {alert["message"]}<br>
                <small>対応: {alert["action"]}</small>
            </div>"""
            html_content += "\n        </div>"

        # 推奨事項表示
        if self.dashboard_data["recommendations"]:
            html_content += """
        <div class="card">
            <h3>💡 推奨事項</h3>"""
            for rec in self.dashboard_data["recommendations"]:
                html_content += f"""
            <div class="recommendation">
                <strong>{rec["title"]} ({rec["priority"]})</strong><br>
                {rec["description"]}<br>
                <small>アクション: {rec["action"]}</small>
            </div>"""
            html_content += "\n        </div>"

        # テスト履歴表示
        if self.dashboard_data["test_history"]:
            html_content += """
        <div class="card">
            <h3>📋 テスト履歴 (最新10件)</h3>
            <div class="test-history">"""
            for test in self.dashboard_data["test_history"]:
                status_class = {
                    "PASS": "test-pass",
                    "WARNING": "test-warning",
                    "FAIL": "test-fail",
                }.get(test["status"], "")

                degradation_icon = "⚠️" if test["degradation_detected"] else "✅"

                html_content += f"""
                <div class="test-item">
                    <span class="{status_class}">●</span> 
                    {test["test_id"]} - {test["timestamp"][:19]} 
                    {degradation_icon} A/B: {test["ab_evaluation_rate"]:.1f}%
                </div>"""
            html_content += "\n            </div>\n        </div>"

        html_content += """
    </div>
</body>
</html>"""

        with open(dashboard_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"  📄 HTMLダッシュボード: {dashboard_file}")
        return dashboard_file

    def _generate_json_report(self):
        """JSONレポート生成"""
        json_file = self.workspace / f"dashboard_data_{datetime.now():%Y%m%d_%H%M%S}.json"

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(self.dashboard_data, f, indent=2, ensure_ascii=False)

        print(f"  📄 JSONレポート: {json_file}")
        return json_file

    def _generate_markdown_summary(self):
        """Markdownサマリー生成"""
        md_file = self.workspace / f"P1A003_Summary_{datetime.now():%Y%m%d}.md"

        system_status = self.dashboard_data["system_status"]
        latest_metrics = self.dashboard_data["quality_metrics"].get("latest_metrics", {})

        content = f"""# P1-A003: 自動テスト強化システム サマリー

**生成日時**: {datetime.now():%Y-%m-%d %H:%M:%S}

## システム状態

- **健全性**: {system_status.get('system_health', 'UNKNOWN')}
- **テスト結果数**: {system_status.get('test_results_count', 0)}件
- **継続監視**: {'有効' if system_status.get('monitoring_active') else '無効'}
- **最終テスト**: {system_status.get('last_test_time', '未実行')[:19] if system_status.get('last_test_time') else '未実行'}

## 最新品質指標

- **A/B評価率**: {latest_metrics.get('ab_evaluation_rate', 0):.1f}%
- **SCI スコア**: {latest_metrics.get('sci_score', 0):.3f}
- **PLA スコア**: {latest_metrics.get('pla_score', 0):.3f}
- **PLE スコア**: {latest_metrics.get('ple_score', 0):.3f}
- **成功率**: {(latest_metrics.get('success_count', 0) / latest_metrics.get('total_processed', 1) * 100):.1f}%

## アラート状況

{chr(10).join(f"- **{alert['level']}**: {alert['message']}" for alert in self.dashboard_data['alerts']) or "アラートなし"}

## 推奨事項

{chr(10).join(f"- **{rec['title']}** ({rec['priority']}): {rec['description']}" for rec in self.dashboard_data['recommendations']) or "推奨事項なし"}

---
*P1-A003自動テスト強化システム - 自動生成レポート*
"""

        with open(md_file, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"  📄 Markdownサマリー: {md_file}")
        return md_file


def main():
    """メイン実行"""
    dashboard = P1A003Dashboard()
    result = dashboard.generate_dashboard()

    # 通知送信
    alerts_count = len(result["alerts"])
    recommendations_count = len(result["recommendations"])

    subprocess.run(
        [
            "windows-notify",
            "-t",
            "P1-A003ダッシュボード生成完了",
            "-m",
            f"アラート: {alerts_count}件\n推奨事項: {recommendations_count}件",
        ],
        check=False,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
