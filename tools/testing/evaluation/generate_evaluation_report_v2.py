#!/usr/bin/env python3
"""
評価結果HTMLレポート生成スクリプト v2
A-F判定評価結果をビジュアルなHTMLレポートに変換

機能:
- 新しいA-F評価フォーマット対応
- 人間評価との比較分析
- 問題分類の可視化
- インタラクティブなHTMLレポート生成
"""

import base64
import json
import os
from datetime import datetime
from pathlib import Path


class EvaluationReportGeneratorV2:
    """A-F判定評価結果HTMLレポート生成クラス"""

    def __init__(
        self,
        json_path: str,
        human_eval_path: str = None,
        output_path: str = "evaluation_report_v2.html",
    ):
        """
        初期化

        Args:
            json_path: GPT-4O評価結果JSONファイルパス
            human_eval_path: 人間評価結果JSONファイルパス
            output_path: 出力HTMLファイルパス
        """
        self.json_path = json_path
        self.human_eval_path = human_eval_path
        self.output_path = output_path
        self.gpt_data = None
        self.human_data = None

    def load_evaluation_data(self):
        """評価データの読み込み"""
        print(f"📊 GPT-4O評価データ読み込み: {self.json_path}")

        with open(self.json_path, "r", encoding="utf-8") as f:
            self.gpt_data = json.load(f)

        print(f"✅ GPT-4Oデータ読み込み完了: {self.gpt_data['batch_info']['total_images']}枚")

        # 人間評価データの読み込み（オプション）
        if self.human_eval_path and os.path.exists(self.human_eval_path):
            print(f"📊 人間評価データ読み込み: {self.human_eval_path}")
            with open(self.human_eval_path, "r", encoding="utf-8") as f:
                self.human_data = json.load(f)
            print(f"✅ 人間評価データ読み込み完了")

    def image_to_base64(self, image_path):
        """画像をBase64エンコード"""
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()
            return base64.b64encode(image_data).decode("utf-8")
        except Exception as e:
            print(f"⚠️ 画像読み込みエラー: {image_path} - {e}")
            return None

    def get_human_evaluation(self, filename):
        """ファイル名から人間評価を取得"""
        if not self.human_data:
            return None

        for item in self.human_data.get("evaluationData", []):
            if item.get("filename") == filename:
                return {
                    "rating": item.get("folder2_rating"),
                    "issues": item.get("folder2_issues_list", []),
                    "notes": item.get("notes", ""),
                }
        return None

    def generate_html_report(self):
        """HTMLレポート生成"""
        print("🎨 A-F判定HTMLレポート生成開始")

        # 統計情報の準備
        batch_info = self.gpt_data["batch_info"]
        results = self.gpt_data["results"]

        # GPT-4O評価統計
        grade_distribution = {}
        issues_count = {}
        api_usage = {}

        for result in results:
            if result["success"] and not result.get("parse_error"):
                grade = result.get("grade", "Unknown")
                grade_distribution[grade] = grade_distribution.get(grade, 0) + 1

                # 問題分類の集計
                issues = result.get("issues", [])
                for issue in issues:
                    issues_count[issue] = issues_count.get(issue, 0) + 1

                # API使用統計
                api_used = result.get("api_used", "Unknown")
                api_usage[api_used] = api_usage.get(api_used, 0) + 1

        # 人間評価統計（利用可能な場合）
        human_stats = {}
        if self.human_data:
            for item in self.human_data.get("evaluationData", []):
                rating = item.get("folder2_rating")
                if rating:
                    human_stats[rating] = human_stats.get(rating, 0) + 1

        # グレード色定義
        grade_colors = {
            "A": "#27ae60",  # 緑
            "B": "#2ecc71",  # 明緑
            "C": "#f39c12",  # オレンジ
            "D": "#e67e22",  # 濃オレンジ
            "E": "#e74c3c",  # 赤
            "F": "#c0392b",  # 濃赤
        }

        # HTMLテンプレート
        html_content = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPT-4O vs 人間評価 比較レポート v0.3.5</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 3px solid #3498db;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .summary-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            font-size: 1.2em;
        }}
        .summary-card .value {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .comparison-section {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 40px;
        }}
        .evaluation-panel {{
            background: #fff;
            border: 2px solid #ecf0f1;
            border-radius: 10px;
            padding: 20px;
        }}
        .evaluation-panel h3 {{
            text-align: center;
            margin-bottom: 20px;
            color: #2c3e50;
        }}
        .grade-chart {{
            display: grid;
            gap: 10px;
        }}
        .grade-item {{
            display: flex;
            align-items: center;
            padding: 10px;
            border-radius: 5px;
            background: #f8f9fa;
        }}
        .grade-badge {{
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            color: white;
            margin-right: 15px;
        }}
        .grade-info {{
            flex: 1;
        }}
        .grade-count {{
            font-size: 1.2em;
            font-weight: bold;
        }}
        .grade-percentage {{
            color: #666;
            font-size: 0.9em;
        }}
        .issues-section {{
            margin-bottom: 40px;
        }}
        .issues-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
        }}
        .issue-card {{
            background: #fff;
            border-left: 4px solid #e74c3c;
            border-radius: 0 10px 10px 0;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .issue-title {{
            font-weight: bold;
            color: #e74c3c;
            margin-bottom: 5px;
        }}
        .issue-count {{
            font-size: 1.5em;
            color: #2c3e50;
        }}
        .results-section {{
            margin-top: 40px;
        }}
        .image-result {{
            display: grid;
            grid-template-columns: 300px 1fr 200px;
            gap: 20px;
            margin-bottom: 30px;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 10px;
            background-color: #fafafa;
        }}
        .image-container {{
            text-align: center;
        }}
        .image-container img {{
            max-width: 100%;
            max-height: 250px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .image-info {{
            padding: 10px 0;
        }}
        .evaluation-details {{
            padding: 10px;
        }}
        .grade-display {{
            text-align: center;
            padding: 20px;
        }}
        .grade-circle {{
            width: 80px;
            height: 80px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2em;
            font-weight: bold;
            color: white;
            margin: 0 auto 10px;
        }}
        .comparison-grades {{
            display: flex;
            flex-direction: column;
            gap: 15px;
        }}
        .grade-comparison {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
        }}
        .issues-list {{
            margin-top: 10px;
        }}
        .issue-tag {{
            display: inline-block;
            background: #e74c3c;
            color: white;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            margin: 2px;
        }}
        .comments {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
            margin-top: 10px;
            font-style: italic;
        }}
        .api-badge {{
            display: inline-block;
            background: #3498db;
            color: white;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.8em;
            font-weight: bold;
            margin-top: 10px;
        }}
        .accuracy-indicator {{
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            font-weight: bold;
            margin-top: 10px;
        }}
        .match {{
            background: #d4edda;
            color: #155724;
        }}
        .close {{
            background: #fff3cd;
            color: #856404;
        }}
        .different {{
            background: #f8d7da;
            color: #721c24;
        }}
        @media (max-width: 768px) {{
            .comparison-section {{
                grid-template-columns: 1fr;
            }}
            .image-result {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 GPT-4O vs 👤 人間評価 比較レポート v0.3.5</h1>
        
        <div class="summary">
            <div class="summary-card">
                <h3>📊 評価画像数</h3>
                <div class="value">{batch_info['total_images']}</div>
                <div>抽出成功画像</div>
            </div>
            <div class="summary-card">
                <h3>🤖 MCP評価成功率</h3>
                <div class="value">{batch_info['mcp_success_rate']:.1f}%</div>
                <div>({batch_info['mcp_success_count']}/{batch_info['total_images']})</div>
            </div>
            <div class="summary-card">
                <h3>⏱️ 処理時間</h3>
                <div class="value">{batch_info['total_time']:.1f}s</div>
                <div>平均: {batch_info['total_time']/batch_info['total_images']:.1f}s/枚</div>
            </div>
            <div class="summary-card">
                <h3>🎯 評価方式</h3>
                <div class="value">A-F</div>
                <div>人間評価準拠</div>
            </div>
        </div>
"""

        # 比較セクションの追加
        if human_stats:
            html_content += f"""
        <div class="comparison-section">
            <div class="evaluation-panel">
                <h3>🤖 GPT-4O評価分布</h3>
                <div class="grade-chart">
"""
            for grade in ["A", "B", "C", "D", "E", "F"]:
                count = grade_distribution.get(grade, 0)
                total = len(results)
                percentage = (count / total * 100) if total > 0 else 0
                color = grade_colors.get(grade, "#95a5a6")

                html_content += f"""
                    <div class="grade-item">
                        <div class="grade-badge" style="background-color: {color};">{grade}</div>
                        <div class="grade-info">
                            <div class="grade-count">{count}枚</div>
                            <div class="grade-percentage">{percentage:.1f}%</div>
                        </div>
                    </div>
"""

            html_content += """
                </div>
            </div>
            
            <div class="evaluation-panel">
                <h3>👤 人間評価分布</h3>
                <div class="grade-chart">
"""

            for grade in ["A", "B", "C", "D", "E", "F"]:
                count = human_stats.get(grade, 0)
                total = sum(human_stats.values()) if human_stats else 1
                percentage = (count / total * 100) if total > 0 else 0
                color = grade_colors.get(grade, "#95a5a6")

                html_content += f"""
                    <div class="grade-item">
                        <div class="grade-badge" style="background-color: {color};">{grade}</div>
                        <div class="grade-info">
                            <div class="grade-count">{count}枚</div>
                            <div class="grade-percentage">{percentage:.1f}%</div>
                        </div>
                    </div>
"""

            html_content += """
                </div>
            </div>
        </div>
"""

        # 問題分類セクション
        html_content += f"""
        <div class="issues-section">
            <h2>⚠️ 問題分類分析（GPT-4O検出）</h2>
            <div class="issues-grid">
"""

        for issue, count in sorted(issues_count.items(), key=lambda x: x[1], reverse=True):
            html_content += f"""
                <div class="issue-card">
                    <div class="issue-title">{issue}</div>
                    <div class="issue-count">{count}枚</div>
                </div>
"""

        html_content += """
            </div>
        </div>
        
        <div class="results-section">
            <h2>📸 個別画像評価比較</h2>
"""

        # 個別画像結果の追加
        for i, result in enumerate(results, 1):
            if not result["success"] or result.get("parse_error"):
                continue

            filename = Path(result["image_path"]).name
            image_base64 = self.image_to_base64(result["image_path"])
            image_src = f"data:image/jpeg;base64,{image_base64}" if image_base64 else ""

            gpt_grade = result.get("grade", "N/A")
            gpt_issues = result.get("issues", [])
            gpt_comments = result.get("comments", "")

            # 人間評価データの取得
            human_eval = self.get_human_evaluation(filename)
            human_grade = human_eval["rating"] if human_eval else "N/A"
            human_issues = human_eval["issues"] if human_eval else []
            human_notes = human_eval["notes"] if human_eval else ""

            # 評価一致度の判定
            accuracy_class = "different"
            accuracy_text = "評価相違"
            if human_grade == gpt_grade:
                accuracy_class = "match"
                accuracy_text = "完全一致"
            elif human_grade != "N/A" and gpt_grade != "N/A":
                # グレードの近似判定
                grade_values = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "F": 0}
                h_val = grade_values.get(human_grade, 0)
                g_val = grade_values.get(gpt_grade, 0)
                if abs(h_val - g_val) <= 1:
                    accuracy_class = "close"
                    accuracy_text = "近似一致"

            gpt_color = grade_colors.get(gpt_grade, "#95a5a6")
            human_color = grade_colors.get(human_grade, "#95a5a6")

            html_content += f"""
            <div class="image-result">
                <div class="image-container">
                    <img src="{image_src}" alt="{filename}" />
                    <div class="image-info">
                        <strong>{filename}</strong>
                        <div class="api-badge">{result.get('api_used', 'Unknown').upper()}</div>
                    </div>
                </div>
                <div class="evaluation-details">
                    <div class="comparison-grades">
                        <div class="grade-comparison">
                            <div>
                                <strong>🤖 GPT-4O:</strong>
                                <span class="grade-badge" style="background-color: {gpt_color}; margin-left: 10px; width: 30px; height: 30px; font-size: 1em;">{gpt_grade}</span>
                            </div>
                        </div>
                        <div class="grade-comparison">
                            <div>
                                <strong>👤 人間:</strong>
                                <span class="grade-badge" style="background-color: {human_color}; margin-left: 10px; width: 30px; height: 30px; font-size: 1em;">{human_grade}</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="accuracy-indicator {accuracy_class}">
                        {accuracy_text}
                    </div>
                    
                    <div class="issues-list">
                        <strong>🚨 GPT-4O検出問題:</strong><br>
"""

            for issue in gpt_issues:
                html_content += f'<span class="issue-tag">{issue}</span>'

            if not gpt_issues:
                html_content += '<span style="color: #27ae60;">問題なし</span>'

            html_content += f"""
                    </div>
                    
                    <div class="comments">
                        <strong>💭 GPT-4Oコメント:</strong><br>
                        {gpt_comments}
                    </div>
"""

            if human_notes:
                html_content += f"""
                    <div class="comments" style="border-left-color: #f39c12;">
                        <strong>📝 人間評価メモ:</strong><br>
                        {human_notes}
                    </div>
"""

            html_content += """
                </div>
            </div>
"""

        # HTML終了部分
        html_content += f"""
        </div>
        
        <div style="text-align: center; margin-top: 40px; padding: 20px; background: #f8f9fa; border-radius: 10px;">
            <h3>📊 レポート生成情報</h3>
            <p>生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>GPT-4O評価データ: {self.json_path}</p>
            <p>人間評価データ: {self.human_eval_path or 'なし'}</p>
            <p>評価システム: A-F判定（人間評価準拠）</p>
        </div>
    </div>
</body>
</html>
"""

        # HTMLファイル保存
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"✅ HTMLレポート生成完了: {self.output_path}")

        return self.output_path


def main():
    """メイン実行関数"""
    print("🎨 A-F判定評価結果HTMLレポート生成開始")

    # 入力ファイル確認
    gpt_json_path = "batch_evaluation_results_v2.json"
    human_json_path = (
        "/mnt/c/AItools/image_evaluation_system/data/evaluation_progress_2025-07-18T15-37-19.json"
    )

    if not os.path.exists(gpt_json_path):
        print(f"❌ GPT-4O評価結果ファイルが見つかりません: {gpt_json_path}")
        return

    # レポート生成
    generator = EvaluationReportGeneratorV2(gpt_json_path, human_json_path)
    generator.load_evaluation_data()
    output_path = generator.generate_html_report()

    print(f"\n🎯 比較レポート生成完了!")
    print(f"📁 ファイル: {output_path}")
    print(f"🌐 ブラウザで開いてください: file://{os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
