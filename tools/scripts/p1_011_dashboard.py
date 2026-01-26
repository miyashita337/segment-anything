#!/usr/bin/env python3
"""
P1-011処理キュー管理システム ダッシュボード生成
実行結果・統計情報のHTMLダッシュボード作成
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def create_p1_011_dashboard(tracker_id: str = "P1-011"):
    """P1-011ダッシュボード作成"""
    workspace_dir = Path(f"/mnt/c/AItools/lora/train/yado/tracker-workspace/{tracker_id}")
    dashboard_dir = workspace_dir / "dashboard"
    dashboard_dir.mkdir(parents=True, exist_ok=True)

    # テスト結果の読み込み
    test_extraction_dir = workspace_dir / "test_extraction"
    test_result_file = test_extraction_dir / "test_result.json"

    test_data = {}
    if test_result_file.exists():
        with open(test_result_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)

    # 抽出ディレクトリの確認
    extraction_dir = workspace_dir / "extraction"
    extracted_files = []
    if extraction_dir.exists():
        extracted_files = (
            list(extraction_dir.glob("*.txt"))
            + list(extraction_dir.glob("*.png"))
            + list(extraction_dir.glob("*.jpg"))
        )

    # HTML生成
    html_content = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>P1-011処理キュー管理システム ダッシュボード</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .header p {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 30px;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            border-left: 5px solid #3498db;
            transition: transform 0.3s ease;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        
        .stat-label {{
            font-size: 1.1em;
            color: #7f8c8d;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .success {{
            border-left-color: #27ae60;
        }}
        
        .success .stat-value {{
            color: #27ae60;
        }}
        
        .warning {{
            border-left-color: #f39c12;
        }}
        
        .warning .stat-value {{
            color: #f39c12;
        }}
        
        .error {{
            border-left-color: #e74c3c;
        }}
        
        .error .stat-value {{
            color: #e74c3c;
        }}
        
        .section {{
            margin-bottom: 30px;
            padding: 25px;
            background: #f8f9fa;
            border-radius: 15px;
            border: 1px solid #e9ecef;
        }}
        
        .section h2 {{
            color: #2c3e50;
            margin-bottom: 20px;
            font-size: 1.8em;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        
        .config-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        
        .config-table th,
        .config-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        
        .config-table th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        
        .config-table tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        
        .progress-bar {{
            width: 100%;
            height: 30px;
            background-color: #ecf0f1;
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #27ae60 0%, #2ecc71 100%);
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }}
        
        .timestamp {{
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            color: #7f8c8d;
            font-style: italic;
        }}
        
        .feature-highlight {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
        }}
        
        .feature-highlight h3 {{
            margin-bottom: 10px;
            font-size: 1.5em;
        }}
        
        .badge {{
            display: inline-block;
            padding: 5px 15px;
            background: #3498db;
            color: white;
            border-radius: 20px;
            font-size: 0.9em;
            margin: 2px;
        }}
        
        .success-badge {{
            background: #27ae60;
        }}
        
        .file-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }}
        
        .file-item {{
            background: #ecf0f1;
            padding: 10px;
            border-radius: 8px;
            font-family: monospace;
            font-size: 0.9em;
            color: #2c3e50;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 P1-011 処理キュー管理システム</h1>
            <p>大量画像の効率的処理順序制御 - 実行結果ダッシュボード</p>
        </div>
        
        <div class="content">
            <div class="feature-highlight">
                <h3>🎯 P1-011の主要機能</h3>
                <p>
                    <span class="badge success-badge">優先度ベース処理</span>
                    <span class="badge success-badge">並列ワーカー</span>
                    <span class="badge success-badge">自動リトライ</span>
                    <span class="badge success-badge">メモリ監視</span>
                    <span class="badge success-badge">統計収集</span>
                    <span class="badge success-badge">品質評価</span>
                </p>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card success">
                    <div class="stat-value">{test_data.get('success_rate', 0):.1f}%</div>
                    <div class="stat-label">処理成功率</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-value">{test_data.get('total_tasks', 0)}</div>
                    <div class="stat-label">総タスク数</div>
                </div>
                
                <div class="stat-card success">
                    <div class="stat-value">{test_data.get('completed_tasks', 0)}</div>
                    <div class="stat-label">完了タスク</div>
                </div>
                
                <div class="stat-card error">
                    <div class="stat-value">{test_data.get('failed_tasks', 0)}</div>
                    <div class="stat-label">失敗タスク</div>
                </div>
                
                <div class="stat-card warning">
                    <div class="stat-value">{test_data.get('total_time_seconds', 0):.2f}s</div>
                    <div class="stat-label">処理時間</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-value">{len(extracted_files)}</div>
                    <div class="stat-label">出力ファイル</div>
                </div>
            </div>
            
            <div class="section">
                <h2>📊 処理成功率</h2>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {test_data.get('success_rate', 0)}%">
                        {test_data.get('success_rate', 0):.1f}%
                    </div>
                </div>
                <p>目標成功率: 80% | 実際: {test_data.get('success_rate', 0):.1f}% 
                {"✅ 目標達成" if test_data.get('success_rate', 0) >= 80 else "❌ 目標未達"}</p>
            </div>
            
            <div class="section">
                <h2>⚙️ キュー設定</h2>
                <table class="config-table">
                    <thead>
                        <tr>
                            <th>設定項目</th>
                            <th>値</th>
                            <th>説明</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>最大ワーカー数</td>
                            <td>{test_data.get('config', {}).get('max_workers', 'N/A')}</td>
                            <td>並列処理スレッド数</td>
                        </tr>
                        <tr>
                            <td>処理モード</td>
                            <td>{test_data.get('config', {}).get('processing_mode', 'N/A')}</td>
                            <td>処理方式（適応的/順次/並列）</td>
                        </tr>
                        <tr>
                            <td>自動優先度</td>
                            <td>{"有効" if test_data.get('config', {}).get('auto_priority') else "無効"}</td>
                            <td>ファイルサイズベース優先度自動設定</td>
                        </tr>
                        <tr>
                            <td>バッチサイズ</td>
                            <td>{test_data.get('config', {}).get('batch_size', 'N/A')}</td>
                            <td>一度に処理する最大タスク数</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2>📈 キュー統計情報</h2>
                <table class="config-table">
                    <thead>
                        <tr>
                            <th>統計項目</th>
                            <th>値</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>総タスク数</td>
                            <td>{test_data.get('queue_statistics', {}).get('total_tasks', 'N/A')}</td>
                        </tr>
                        <tr>
                            <td>完了タスク数</td>
                            <td>{test_data.get('queue_statistics', {}).get('completed_tasks', 'N/A')}</td>
                        </tr>
                        <tr>
                            <td>失敗タスク数</td>
                            <td>{test_data.get('queue_statistics', {}).get('failed_tasks', 'N/A')}</td>
                        </tr>
                        <tr>
                            <td>総処理時間</td>
                            <td>{test_data.get('queue_statistics', {}).get('processing_time_total', 'N/A')}s</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2>📁 出力ファイル一覧</h2>
                <p>抽出結果ファイル数: <strong>{len(extracted_files)}個</strong></p>
                <div class="file-grid">
                    {"".join([f'<div class="file-item">{f.name}</div>' for f in extracted_files[:20]])}
                    {f'<div class="file-item">... 他 {len(extracted_files) - 20} ファイル</div>' if len(extracted_files) > 20 else ''}
                </div>
            </div>
            
            <div class="section">
                <h2>🎯 P1-011実装成果</h2>
                <div class="stats-grid">
                    <div class="stat-card success">
                        <div class="stat-value">✅</div>
                        <div class="stat-label">キュー管理システム</div>
                    </div>
                    <div class="stat-card success">
                        <div class="stat-value">✅</div>
                        <div class="stat-label">優先度制御</div>
                    </div>
                    <div class="stat-card success">
                        <div class="stat-value">✅</div>
                        <div class="stat-label">並列ワーカー</div>
                    </div>
                    <div class="stat-card success">
                        <div class="stat-value">✅</div>
                        <div class="stat-label">統計収集</div>
                    </div>
                    <div class="stat-card success">
                        <div class="stat-value">✅</div>
                        <div class="stat-label">メモリ監視</div>
                    </div>
                    <div class="stat-card success">
                        <div class="stat-value">✅</div>
                        <div class="stat-label">自動リトライ</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="timestamp">
            <p>📅 ダッシュボード生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>🔧 ワークスペース: {workspace_dir}</p>
        </div>
    </div>
</body>
</html>
"""

    # ダッシュボード保存
    dashboard_file = dashboard_dir / "dashboard.html"
    with open(dashboard_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ P1-011ダッシュボード生成完了: {dashboard_file}")
    return dashboard_file


def main():
    """メイン実行"""
    import argparse

    parser = argparse.ArgumentParser(description="P1-011ダッシュボード生成")
    parser.add_argument("--tracker_id", type=str, default="P1-011", help="トラッカーID")

    args = parser.parse_args()

    try:
        dashboard_file = create_p1_011_dashboard(args.tracker_id)
        print(f"🎉 ダッシュボードURL: file://{dashboard_file}")
        return 0
    except Exception as e:
        print(f"❌ ダッシュボード生成エラー: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
