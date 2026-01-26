#!/usr/bin/env python3
"""
評価結果HTMLレポート生成スクリプト
GPT-4O評価結果をビジュアルなHTMLレポートに変換

機能:
- JSON評価データの読み込み
- 画像と評価結果の並列表示
- 品質統計の可視化
- インタラクティブなHTMLレポート生成
"""

import base64
import json
import os
import re
from datetime import datetime
from pathlib import Path


class EvaluationReportGenerator:
    """評価結果HTMLレポート生成クラス"""

    def __init__(self, json_path: str, output_path: str = "evaluation_report.html"):
        """
        初期化

        Args:
            json_path: 評価結果JSONファイルパス
            output_path: 出力HTMLファイルパス
        """
        self.json_path = json_path
        self.output_path = output_path
        self.data = None

    def load_evaluation_data(self):
        """評価データの読み込み"""
        print(f"📊 評価データ読み込み: {self.json_path}")

        with open(self.json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        print(f"✅ データ読み込み完了: {self.data['batch_info']['total_images']}枚")

    def extract_scores_from_response(self, result):
        """レスポンスからスコアを抽出"""
        # 既にパースされたデータがある場合
        if not result.get("parse_error", True):
            return {
                "completeness": result.get("completeness", 0),
                "boundary_quality": result.get("boundary_quality", 0),
                "background_removal": result.get("background_removal", 0),
                "overall_quality": result.get("overall_quality", 0),
                "comments": result.get("comments", ""),
            }

        # raw_responseからJSONを抽出
        raw_response = result.get("raw_response", "")
        if not raw_response:
            return None

        # JSONコードブロックを抽出
        json_match = re.search(r"```json\n(.*?)\n```", raw_response, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(1)
                parsed_data = json.loads(json_str)
                return {
                    "completeness": parsed_data.get("completeness", 0),
                    "boundary_quality": parsed_data.get("boundary_quality", 0),
                    "background_removal": parsed_data.get("background_removal", 0),
                    "overall_quality": parsed_data.get("overall_quality", 0),
                    "comments": parsed_data.get("comments", ""),
                }
            except json.JSONDecodeError:
                pass

        return None

    def image_to_base64(self, image_path):
        """画像をBase64エンコード"""
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()
            return base64.b64encode(image_data).decode("utf-8")
        except Exception as e:
            print(f"⚠️ 画像読み込みエラー: {image_path} - {e}")
            return None

    def generate_html_report(self):
        """HTMLレポート生成"""
        print("🎨 HTMLレポート生成開始")

        # 統計情報の準備
        batch_info = self.data["batch_info"]
        results = self.data["results"]

        # スコア抽出
        parsed_results = []
        for result in results:
            scores = self.extract_scores_from_response(result)
            if scores:
                parsed_results.append(
                    {
                        "image_path": result["image_path"],
                        "filename": Path(result["image_path"]).name,
                        "api_used": result.get("api_used", "unknown"),
                        "timestamp": result.get("timestamp", ""),
                        **scores,
                    }
                )

        # 統計計算
        if parsed_results:
            avg_completeness = sum(r["completeness"] for r in parsed_results) / len(parsed_results)
            avg_boundary = sum(r["boundary_quality"] for r in parsed_results) / len(parsed_results)
            avg_background = sum(r["background_removal"] for r in parsed_results) / len(
                parsed_results
            )
            avg_overall = sum(r["overall_quality"] for r in parsed_results) / len(parsed_results)
        else:
            avg_completeness = avg_boundary = avg_background = avg_overall = 0

        # API使用統計
        api_usage = {}
        for result in parsed_results:
            api = result["api_used"]
            api_usage[api] = api_usage.get(api, 0) + 1

        # HTMLテンプレート
        html_content = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>画像抽出品質評価レポート v0.3.5</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
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
        .scores-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 40px;
        }}
        .score-card {{
            background: #fff;
            border: 2px solid #ecf0f1;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            transition: transform 0.2s;
        }}
        .score-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }}
        .score-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #27ae60;
            margin: 10px 0;
        }}
        .results-section {{
            margin-top: 40px;
        }}
        .image-result {{
            display: grid;
            grid-template-columns: 300px 1fr;
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
        .scores-row {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin-bottom: 20px;
        }}
        .score-item {{
            text-align: center;
            padding: 10px;
            background: #e8f6f3;
            border-radius: 5px;
        }}
        .score-item .label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }}
        .score-item .score {{
            font-size: 1.5em;
            font-weight: bold;
            color: #27ae60;
        }}
        .comments {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
            margin-top: 10px;
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
        .progress-bar {{
            width: 100%;
            height: 20px;
            background: #ecf0f1;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #ff6b6b, #feca57, #48dbfb, #ff9ff3, #54a0ff);
            transition: width 0.3s ease;
        }}
        .timestamp {{
            color: #666;
            font-size: 0.8em;
            margin-top: 10px;
        }}
        @media (max-width: 768px) {{
            .image-result {{
                grid-template-columns: 1fr;
            }}
            .scores-row {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 画像抽出品質評価レポート v0.3.5</h1>
        
        <div class="summary">
            <div class="summary-card">
                <h3>📊 総合統計</h3>
                <div class="value">{batch_info['total_images']}</div>
                <div>評価画像数</div>
            </div>
            <div class="summary-card">
                <h3>✅ 成功率</h3>
                <div class="value">{batch_info['success_rate']:.1f}%</div>
                <div>({batch_info['success_count']}/{batch_info['total_images']})</div>
            </div>
            <div class="summary-card">
                <h3>⏱️ 処理時間</h3>
                <div class="value">{batch_info['total_time']:.1f}s</div>
                <div>平均: {batch_info['total_time']/batch_info['total_images']:.1f}s/枚</div>
            </div>
            <div class="summary-card">
                <h3>🤖 主要API</h3>
                <div class="value">GPT-4O</div>
                <div>{max(api_usage.values()) if api_usage else 0}枚処理</div>
            </div>
        </div>
        
        <div class="scores-grid">
            <div class="score-card">
                <h3>🎯 抽出完全性</h3>
                <div class="score-value">{avg_completeness:.1f}</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {avg_completeness*20}%"></div>
                </div>
                <div>手足切断なし</div>
            </div>
            <div class="score-card">
                <h3>🔍 境界品質</h3>
                <div class="score-value">{avg_boundary:.1f}</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {avg_boundary*20}%"></div>
                </div>
                <div>境界線の精度</div>
            </div>
            <div class="score-card">
                <h3>🎨 背景除去</h3>
                <div class="score-value">{avg_background:.1f}</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {avg_background*20}%"></div>
                </div>
                <div>背景残存なし</div>
            </div>
            <div class="score-card">
                <h3>⭐ 総合品質</h3>
                <div class="score-value">{avg_overall:.1f}</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {avg_overall*20}%"></div>
                </div>
                <div>全体評価</div>
            </div>
        </div>
        
        <div class="results-section">
            <h2>📸 個別画像評価結果</h2>
"""

        # 個別画像結果の追加
        for i, result in enumerate(parsed_results, 1):
            image_base64 = self.image_to_base64(result["image_path"])
            image_src = f"data:image/jpeg;base64,{image_base64}" if image_base64 else ""

            html_content += f"""
            <div class="image-result">
                <div class="image-container">
                    <img src="{image_src}" alt="{result['filename']}" />
                    <div class="image-info">
                        <strong>{result['filename']}</strong>
                        <div class="api-badge">{result['api_used'].upper()}</div>
                        <div class="timestamp">{result['timestamp']}</div>
                    </div>
                </div>
                <div class="evaluation-details">
                    <h3>評価スコア</h3>
                    <div class="scores-row">
                        <div class="score-item">
                            <div class="label">完全性</div>
                            <div class="score">{result['completeness']}</div>
                        </div>
                        <div class="score-item">
                            <div class="label">境界品質</div>
                            <div class="score">{result['boundary_quality']}</div>
                        </div>
                        <div class="score-item">
                            <div class="label">背景除去</div>
                            <div class="score">{result['background_removal']}</div>
                        </div>
                        <div class="score-item">
                            <div class="label">総合評価</div>
                            <div class="score">{result['overall_quality']}</div>
                        </div>
                    </div>
                    <div class="comments">
                        <strong>AI評価コメント:</strong><br>
                        {result['comments']}
                    </div>
                </div>
            </div>
"""

        # HTML終了部分
        html_content += f"""
        </div>
        
        <div style="text-align: center; margin-top: 40px; padding: 20px; background: #f8f9fa; border-radius: 10px;">
            <h3>📊 レポート生成情報</h3>
            <p>生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>評価データ: {self.json_path}</p>
            <p>システム: GPT-4O + Gemini フォールバック評価システム</p>
        </div>
    </div>
</body>
</html>
"""

        # HTMLファイル保存
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"✅ HTMLレポート生成完了: {self.output_path}")
        print(f"📋 評価済み画像: {len(parsed_results)}/{batch_info['total_images']}枚")

        return self.output_path


def main():
    """メイン実行関数"""
    print("🎨 評価結果HTMLレポート生成開始")

    # 入力ファイル確認
    json_path = "batch_evaluation_results.json"
    if not os.path.exists(json_path):
        print(f"❌ 評価結果ファイルが見つかりません: {json_path}")
        return

    # レポート生成
    generator = EvaluationReportGenerator(json_path)
    generator.load_evaluation_data()
    output_path = generator.generate_html_report()

    print(f"\n🎯 レポート生成完了!")
    print(f"📁 ファイル: {output_path}")
    print(f"🌐 ブラウザで開いてください: file://{os.path.abspath(output_path)}")

    # 統計サマリー
    batch_info = generator.data["batch_info"]
    print(f"\n📊 統計サマリー:")
    print(f"  総画像数: {batch_info['total_images']}枚")
    print(f"  成功率: {batch_info['success_rate']:.1f}%")
    print(f"  処理時間: {batch_info['total_time']:.1f}秒")


if __name__ == "__main__":
    main()
