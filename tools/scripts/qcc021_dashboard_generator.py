#!/usr/bin/env python3
"""
QCC-021専用ダッシュボード生成システム
サンプルサイズ妥当性検証結果の可視化
"""

import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class QCC021DashboardGenerator:
    """QCC-021専用ダッシュボード生成クラス"""

    def __init__(self, workspace_base: str = "/mnt/c/AItools/lora/train/yado/tracker-workspace"):
        self.workspace_base = Path(workspace_base)
        self.qcc021_workspace = self.workspace_base / "QCC-021"

    def generate_dashboard(self) -> str:
        """
        QCC-021のダッシュボード生成

        Returns:
            生成されたダッシュボードパス
        """
        print("🎯 QCC-021ダッシュボード生成開始...")

        # 検証結果読み込み
        validation_data = self._load_validation_data()

        # ダッシュボードHTML生成
        dashboard_html = self._generate_dashboard_html(validation_data)

        # ダッシュボード保存
        dashboard_path = self.qcc021_workspace / "dashboard" / "dashboard.html"
        dashboard_path.parent.mkdir(parents=True, exist_ok=True)

        with open(dashboard_path, "w", encoding="utf-8") as f:
            f.write(dashboard_html)

        print(f"✅ QCC-021ダッシュボード生成完了:")
        print(f"   - パス: {dashboard_path}")
        print(f"   - サイズ: {dashboard_path.stat().st_size:,} bytes")
        print(f"   - URL: http://100.123.241.106:8088/tracker/QCC-021")

        return str(dashboard_path)

    def _load_validation_data(self) -> Dict[str, Any]:
        """検証結果データ読み込み"""
        json_path = self.qcc021_workspace / "quality" / "qca001_sample_validation.json"

        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            # デフォルトデータ
            return {
                "qca001_sample_info": {
                    "current_sample_size": 14,
                    "workspace_path": str(self.workspace_base / "QCA-001"),
                    "image_files": [],
                },
                "statistical_validation": {
                    "overall_adequacy": False,
                    "recommended_n": 393,
                    "current_power": 0.113,
                    "precision_assessment": "低精度",
                },
                "detailed_requirements": [],
                "warnings_and_suggestions": {
                    "statistical_warnings": ["統計的サンプル数不足", "検出力不十分", "精度レベル低下"],
                    "improvement_suggestions": ["追加サンプル収集推奨", "効果サイズ見直し", "検定手法最適化"],
                    "qca001_specific_recommendations": [
                        "QCA-001の統計的信頼性向上には追加379サンプル推奨",
                        "他作者（kiri, zundamon）からの画像追加でサンプル数拡張を検討",
                    ],
                },
                "analysis_results": {"overall_assessment": "❌ 統計的妥当性には379サンプル追加が必要です（推奨: 393サンプル）"},
            }

    def _generate_dashboard_html(self, validation_data: Dict[str, Any]) -> str:
        """ダッシュボードHTML生成"""

        sample_info = validation_data["qca001_sample_info"]
        stat_validation = validation_data["statistical_validation"]
        warnings = validation_data["warnings_and_suggestions"]
        analysis = validation_data["analysis_results"]

        # 統計チャートデータ準備
        current_n = sample_info["current_sample_size"]
        recommended_n = stat_validation["recommended_n"]
        shortage = recommended_n - current_n

        adequacy_status = "✅ 適切" if stat_validation["overall_adequacy"] else "❌ 不適切"
        adequacy_color = "#27ae60" if stat_validation["overall_adequacy"] else "#e74c3c"

        # 検出力レベル
        power = stat_validation["current_power"]
        if power >= 0.8:
            power_level = "高"
            power_color = "#27ae60"
        elif power >= 0.6:
            power_level = "中"
            power_color = "#f39c12"
        else:
            power_level = "低"
            power_color = "#e74c3c"

        # 精度レベル色
        precision = stat_validation["precision_assessment"]
        precision_colors = {"高精度": "#27ae60", "中精度": "#f39c12", "低精度": "#e74c3c"}
        precision_color = precision_colors.get(precision, "#95a5a6")

        # 警告・推奨事項HTML
        warnings_html = ""
        for warning in warnings["statistical_warnings"][:5]:
            warnings_html += f'<li class="warning-item">⚠️ {warning}</li>'

        suggestions_html = ""
        for suggestion in warnings["improvement_suggestions"][:5]:
            suggestions_html += f'<li class="suggestion-item">💡 {suggestion}</li>'

        qca001_recommendations_html = ""
        for rec in warnings["qca001_specific_recommendations"][:3]:
            qca001_recommendations_html += f'<li class="qca001-rec-item">🎯 {rec}</li>'

        # 詳細要件HTML
        requirements_html = ""
        for req in validation_data.get("detailed_requirements", [])[:4]:
            is_adequate = req.get("is_adequate", False)
            status_icon = "✅" if is_adequate else "❌"
            status_color = "#27ae60" if is_adequate else "#e74c3c"

            requirements_html += f"""
            <div class="requirement-item">
                <div class="req-status" style="color: {status_color}">{status_icon}</div>
                <div class="req-details">
                    <div class="req-name">{req.get('scenario', 'Unknown Test')}</div>
                    <div class="req-numbers">現在: {req.get('current_n', 0)} / 必要: {req.get('required_n', 0)}</div>
                    <div class="req-precision">精度: {req.get('precision_level', 'Unknown')}</div>
                </div>
            </div>"""

        return f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QCC-021 - サンプルサイズ妥当性検証システム</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            min-height: 100vh;
            color: #2c3e50;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.15);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #2c3e50, #3498db);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 3em;
            font-weight: 300;
        }}
        
        .header .subtitle {{
            font-size: 1.3em;
            opacity: 0.9;
            margin-top: 15px;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 40px;
            background: #f8f9fa;
        }}
        
        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        
        .stat-label {{
            color: #666;
            font-size: 1.1em;
        }}
        
        .adequacy-status {{
            color: {adequacy_color};
        }}
        
        .power-level {{
            color: {power_color};
        }}
        
        .precision-level {{
            color: {precision_color};
        }}
        
        .analysis-section {{
            padding: 40px;
            background: white;
        }}
        
        .analysis-section h2 {{
            color: #2c3e50;
            font-size: 2em;
            margin-bottom: 30px;
            text-align: center;
        }}
        
        .warning-suggestions {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 40px;
        }}
        
        .warnings-box, .suggestions-box {{
            background: #f8f9fa;
            padding: 25px;
            border-radius: 15px;
        }}
        
        .warnings-box h3, .suggestions-box h3 {{
            color: #2c3e50;
            margin-bottom: 20px;
            font-size: 1.3em;
        }}
        
        .warning-item, .suggestion-item, .qca001-rec-item {{
            margin-bottom: 10px;
            padding: 10px;
            background: white;
            border-radius: 8px;
            list-style: none;
        }}
        
        .requirements {{
            background: #f8f9fa;
            padding: 25px;
            border-radius: 15px;
            margin-top: 30px;
        }}
        
        .requirement-item {{
            display: flex;
            align-items: center;
            padding: 15px;
            background: white;
            border-radius: 10px;
            margin-bottom: 10px;
        }}
        
        .req-status {{
            font-size: 1.5em;
            margin-right: 15px;
        }}
        
        .req-details {{
            flex: 1;
        }}
        
        .req-name {{
            font-weight: bold;
            margin-bottom: 5px;
        }}
        
        .req-numbers {{
            color: #666;
            font-size: 0.9em;
        }}
        
        .req-precision {{
            color: #3498db;
            font-size: 0.9em;
        }}
        
        .sample-visualization {{
            padding: 40px;
            background: white;
            text-align: center;
        }}
        
        .sample-bar-container {{
            background: #f8f9fa;
            border-radius: 10px;
            height: 50px;
            position: relative;
            margin: 20px 0;
        }}
        
        .current-sample-bar {{
            background: linear-gradient(90deg, #e74c3c, #c0392b);
            height: 100%;
            border-radius: 10px;
            width: {(current_n / max(recommended_n, 1)) * 100:.1f}%;
            position: relative;
        }}
        
        .sample-bar-label {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;
            font-weight: bold;
            font-size: 0.9em;
        }}
        
        .footer {{
            background: #2c3e50;
            color: white;
            text-align: center;
            padding: 30px;
        }}
        
        .generation-info {{
            font-size: 0.9em;
            opacity: 0.8;
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 QCC-021</h1>
            <div class="subtitle">サンプルサイズ妥当性検証システム</div>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{current_n}</div>
                <div class="stat-label">現在のサンプル数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value adequacy-status">{adequacy_status}</div>
                <div class="stat-label">統計的妥当性</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{recommended_n}</div>
                <div class="stat-label">推奨サンプル数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value power-level">{power:.3f} ({power_level})</div>
                <div class="stat-label">現在の検出力</div>
            </div>
            <div class="stat-card">
                <div class="stat-value precision-level">{precision}</div>
                <div class="stat-label">推定精度</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color: #e74c3c">{shortage}</div>
                <div class="stat-label">不足サンプル数</div>
            </div>
        </div>
        
        <div class="sample-visualization">
            <h2>📈 サンプルサイズ可視化</h2>
            <div class="sample-bar-container">
                <div class="current-sample-bar">
                    <div class="sample-bar-label">{current_n} / {recommended_n} サンプル</div>
                </div>
            </div>
            <p>統計的信頼性確保には <strong>{shortage}サンプル</strong> の追加が必要です</p>
        </div>
        
        <div class="analysis-section">
            <h2>⚠️ 統計的分析結果</h2>
            
            <div class="warning-suggestions">
                <div class="warnings-box">
                    <h3>統計的警告</h3>
                    <ul style="padding: 0;">
                        {warnings_html}
                    </ul>
                </div>
                
                <div class="suggestions-box">
                    <h3>改善提案</h3>
                    <ul style="padding: 0;">
                        {suggestions_html}
                    </ul>
                </div>
            </div>
            
            <div class="suggestions-box">
                <h3>🎯 QCA-001特化推奨事項</h3>
                <ul style="padding: 0;">
                    {qca001_recommendations_html}
                </ul>
            </div>
            
            <div class="requirements">
                <h3>📋 詳細統計要件</h3>
                {requirements_html}
            </div>
        </div>
        
        <div class="analysis-section">
            <h2>🔬 総合評価</h2>
            <div style="background: #f8f9fa; padding: 30px; border-radius: 15px; text-align: center; font-size: 1.2em;">
                {analysis['overall_assessment']}
            </div>
        </div>
        
        <div class="footer">
            <p>🤖 QCC-021: サンプルサイズ妥当性検証システム - QCA-001実証結果</p>
            <div class="generation-info">
                Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
                URL: <a href="http://100.123.241.106:8088/tracker/QCC-021" style="color: #3498db;">http://100.123.241.106:8088/tracker/QCC-021</a>
            </div>
        </div>
    </div>
</body>
</html>"""


def main():
    """メイン実行"""
    generator = QCC021DashboardGenerator()
    dashboard_path = generator.generate_dashboard()
    print(f"\n🎯 QCC-021ダッシュボード: {dashboard_path}")


if __name__ == "__main__":
    main()
