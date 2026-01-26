"""
統一ダッシュボード生成システム（DashboardGenerator主実装版）

integrated_quality_pipeline.pyから抽出したDashboardGeneratorを主実装として採用。
StandardDashboardGenerator互換APIを提供。
"""

import json
import logging
import requests
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union


class DashboardGenerator:
    """統一ダッシュボード生成クラス - DashboardGenerator主実装版"""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        workspace_dir: Optional[Union[str, Path]] = None,
        logger: Optional[logging.Logger] = None,
        tracker_id: str = "",
    ):
        """
        初期化

        Args:
            config: 設定辞書（統合パイプライン用、省略可能）
            workspace_dir: ワークスペースディレクトリ
            logger: ロガー（省略時はデフォルト）
            tracker_id: トラッカーID
        """
        self.config = config or {}
        self.workspace_dir = Path(workspace_dir) if workspace_dir else None
        self.logger = logger or self._setup_default_logger()
        self.tracker_id = tracker_id

        if self.workspace_dir:
            self.dashboard_dir = self.workspace_dir / "dashboard"
            self.dashboard_dir.mkdir(parents=True, exist_ok=True)

    def _setup_default_logger(self) -> logging.Logger:
        """デフォルトロガー設定"""
        logger = logging.getLogger("DashboardGenerator")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    # ===== StandardDashboardGenerator互換API =====

    def create_dashboard(
        self, tracker_id: str, workspace_dir: str, extraction_result_path: Optional[str] = None
    ) -> str:
        """
        標準ダッシュボード作成（StandardDashboardGenerator互換）

        Args:
            tracker_id: トラッカーID
            workspace_dir: ワークスペースディレクトリ
            extraction_result_path: 抽出結果JSONパス（オプション）

        Returns:
            生成されたダッシュボードファイルのパス
        """
        workspace_path = Path(workspace_dir)

        # 抽出結果JSONパス決定
        if extraction_result_path is None:
            extraction_result_path = workspace_path / "extraction_result.json"

        # 出力パス決定
        dashboard_dir = workspace_path / "dashboard"
        dashboard_dir.mkdir(parents=True, exist_ok=True)
        dashboard_file = dashboard_dir / "dashboard.html"

        # データ読み込み
        with open(extraction_result_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # HTML生成（標準版）
        html_content = self._generate_standard_html(tracker_id, data)

        # ダッシュボードディレクトリにextraction_result.jsonもコピー
        dashboard_extraction_result = dashboard_dir / "extraction_result.json"
        if not dashboard_extraction_result.exists():
            import shutil

            shutil.copy2(extraction_result_path, dashboard_extraction_result)
            self.logger.info(f"📋 dashboard/extraction_result.json作成: {dashboard_extraction_result}")

        # ファイル出力
        with open(dashboard_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        self.logger.info(f"🎨 標準ダッシュボード生成完了: {dashboard_file}")
        return str(dashboard_file)

    # ===== 統合パイプライン互換API =====

    def generate_dashboard(self, quality_data: Dict[str, Any]) -> str:
        """
        統合パイプライン用ダッシュボード生成（DashboardGenerator互換）

        Args:
            quality_data: 品質評価データ辞書

        Returns:
            生成されたダッシュボードファイルのパス
        """
        if not self.workspace_dir:
            raise ValueError("workspace_dirが設定されていません。統合パイプライン用には必須です。")

        dashboard_file = self.dashboard_dir / "dashboard.html"

        # HTML生成（統合版）
        html_content = self._generate_integrated_html(quality_data)

        with open(dashboard_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        self.logger.info(f"🎨 統合ダッシュボード生成完了: {dashboard_file}")
        return str(dashboard_file)

    def send_extraction_results_to_pushover(
        self, extraction_dir: Path, max_images: int = 10, demo_mode: bool = False
    ) -> bool:
        """抽出結果をPushoverに送信（指定枚数まで）"""
        try:
            from features.common.notification.global_pushover import send_pushover_notification

            PUSHOVER_AVAILABLE = True
        except ImportError:
            PUSHOVER_AVAILABLE = False

        if not PUSHOVER_AVAILABLE:
            self.logger.warning("Pushover機能が利用できません")
            return False

        try:
            # 抽出された画像ファイルを取得
            extracted_files = list(extraction_dir.glob("extracted_*.png")) + list(
                extraction_dir.glob("extracted_*.jpg")
            )

            if not extracted_files:
                self.logger.warning("送信可能な抽出結果画像が見つかりません")
                return False

            # ファイルサイズでソート（大きいものから）
            extracted_files.sort(key=lambda f: f.stat().st_size, reverse=True)

            # 指定枚数まで制限
            files_to_send = extracted_files[:max_images]

            self.logger.info(f"Pushover送信開始: {len(files_to_send)}枚")

            success_count = 0
            for i, image_file in enumerate(files_to_send, 1):
                try:
                    # ファイルサイズチェック（Pushoverは2.5MB制限）
                    file_size_mb = image_file.stat().st_size / (1024 * 1024)
                    if file_size_mb > 2.0:  # 2MB以上はスキップ
                        self.logger.warning(
                            f"ファイルサイズが大きすぎます: {image_file.name} ({file_size_mb:.1f}MB)"
                        )
                        continue

                    title = f"統合パイプライン結果 {i}/{len(files_to_send)}"
                    message = (
                        f"トラッカー: {self.tracker_id}\n"
                        f"ファイル: {image_file.name}\n"
                        f"サイズ: {file_size_mb:.2f}MB"
                    )

                    # Pushover画像送信
                    if demo_mode:
                        # デモモード: 実際には送信しない
                        success = True
                        self.logger.info(f"[DEMO] Pushover送信: {image_file.name}")
                    else:
                        success = self._send_image_to_pushover(str(image_file), title, message)

                    if success:
                        success_count += 1
                        self.logger.info(f"Pushover送信成功: {image_file.name}")
                    else:
                        self.logger.error(f"Pushover送信失敗: {image_file.name}")

                    # レート制限対策（0.5秒間隔）
                    import time

                    time.sleep(0.5)

                except Exception as e:
                    self.logger.error(f"画像送信エラー {image_file.name}: {str(e)}")
                    continue

            # 送信結果サマリー
            summary_title = f"統合パイプライン完了: {self.tracker_id}"
            summary_message = (
                f"抽出結果送信完了\n"
                f"成功: {success_count}/{len(files_to_send)}枚\n"
                f"総抽出数: {len(extracted_files)}枚"
            )

            if not demo_mode:
                send_pushover_notification(message=summary_message, title=summary_title, priority=0)
            else:
                self.logger.info(f"[DEMO] サマリー送信: {summary_title}")

            self.logger.info(f"Pushover送信完了: {success_count}/{len(files_to_send)}枚成功")
            return success_count > 0

        except Exception as e:
            self.logger.error(f"Pushover送信処理失敗: {str(e)}")
            return False

    def _send_image_to_pushover(self, image_path: str, title: str, message: str) -> bool:
        """個別画像をPushoverに送信"""
        try:
            # 直接設定ファイルを読み込み
            config_path = Path("config/pushover.json")
            if not config_path.exists():
                self.logger.error(f"Pushover設定ファイルが存在しません: {config_path}")
                return False

            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            if not config.get("api_token") or not config.get("user_key"):
                self.logger.error("Pushover設定にapi_tokenまたはuser_keyが不足しています")
                return False

            url = "https://api.pushover.net/1/messages.json"

            data = {
                "token": config["api_token"],
                "user": config["user_key"],
                "title": title,
                "message": message,
                "sound": "magic",
            }

            with open(image_path, "rb") as f:
                files = {"attachment": f}
                response = requests.post(url, data=data, files=files, timeout=30)

            if response.status_code == 200:
                self.logger.info(f"Pushover送信成功: {response.json()}")
                return True
            else:
                self.logger.error(f"Pushover API エラー: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            self.logger.error(f"個別画像送信失敗: {str(e)}")
            import traceback

            self.logger.error(traceback.format_exc())
            return False

    def _generate_standard_html(self, tracker_id: str, data: Dict[str, Any]) -> str:
        """標準ダッシュボードHTML生成（StandardDashboardGenerator準拠）"""

        # 基本統計（正しいキー名で取得）
        total = data.get("total_images", 0)
        successful = data.get("successful_extractions", 0)
        avg_quality = data.get("average_quality_score", 0.0)

        # 画像リスト（extraction_resultsとresults両方に対応）
        results = data.get("results", [])
        if not results and "extraction_results" in data:
            # extraction_resultsが辞書の場合は、resultsを探す
            extraction_data = data["extraction_results"]
            if isinstance(extraction_data, dict):
                results = extraction_data.get("results", [])
            else:
                results = extraction_data

        # 品質分布計算（正しいキー名で取得）
        quality_dist = {"高品質": 0, "中品質": 0, "低品質": 0, "要改善": 0}
        for r in results:
            if r.get("success"):
                score = r.get(
                    "quality_score", r.get("quality_metrics", {}).get("overall_score", 0.0)
                )
                if score >= 0.8:
                    quality_dist["高品質"] += 1
                elif score >= 0.6:
                    quality_dist["中品質"] += 1
                elif score >= 0.4:
                    quality_dist["低品質"] += 1
                else:
                    quality_dist["要改善"] += 1

        # 現在時刻（自然な表示）
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 統計分析結果セクション追加
        statistical_section = ""
        if "statistical_analysis" in data:
            stats = data["statistical_analysis"]
            statistical_section = f"""
        <div class="bg-white rounded-lg shadow-md p-6 mb-8">
            <h2 class="text-xl font-semibold text-gray-800 mb-4">📊 統計分析結果</h2>
            <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-7 gap-4">
                <div class="text-center">
                    <div class="text-sm text-gray-600 mb-1">Current(平均品質スコア)</div>
                    <p class="text-lg font-bold text-blue-600">{avg_quality:.3f}</p>
                </div>
                <div class="text-center">
                    <div class="text-sm text-gray-600 mb-1">BaseLine</div>
                    <p class="text-lg font-bold text-gray-600">{stats.get('baseline_score', 'N/A')}</p>
                </div>
                <div class="text-center">
                    <div class="text-sm text-gray-600 mb-1">p値</div>
                    <p class="text-lg font-bold text-indigo-600">{stats.get('p_value', 'N/A')}</p>
                </div>
                <div class="text-center">
                    <div class="text-sm text-gray-600 mb-1">効果サイズ、Cohen's d</div>
                    <p class="text-lg font-bold text-purple-600">{stats.get('effect_size', 'N/A')}</p>
                </div>
                <div class="text-center">
                    <div class="text-sm text-gray-600 mb-1">改善率</div>
                    <p class="text-lg font-bold text-green-600">{stats.get('improvement_rate', 'N/A')}</p>
                </div>
                <div class="text-center">
                    <div class="text-sm text-gray-600 mb-1">統計的有意性</div>
                    <p class="text-lg font-bold text-red-600">{stats.get('significance', 'N/A')}</p>
                </div>
                <div class="text-center">
                    <div class="text-sm text-gray-600 mb-1">信頼区間</div>
                    <p class="text-lg font-bold text-teal-600">{stats.get('confidence_interval', 'N/A')}</p>
                </div>
            </div>
        </div>"""

        # HTML生成
        html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{tracker_id} - 品質評価ダッシュボード</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <header class="bg-white rounded-lg shadow-md p-6 mb-8">
            <h1 class="text-3xl font-bold text-gray-800 mb-2">{tracker_id} 品質評価ダッシュボード</h1>
            <p class="text-gray-600">生成日時: {current_time}</p>
            <p class="text-sm text-gray-500 mt-2">🔗 統一ダッシュボードシステム v2.0 - DashboardGenerator</p>
        </header>

        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <div class="bg-white rounded-lg shadow-md p-6">
                <h3 class="text-lg font-semibold text-gray-700 mb-2">総画像数</h3>
                <p class="text-3xl font-bold text-blue-600">{total}</p>
            </div>
            <div class="bg-white rounded-lg shadow-md p-6">
                <h3 class="text-lg font-semibold text-gray-700 mb-2">平均品質スコア</h3>
                <p class="text-3xl font-bold text-green-600">{avg_quality:.3f}</p>
            </div>
            <div class="bg-white rounded-lg shadow-md p-6">
                <h3 class="text-lg font-semibold text-gray-700 mb-2">成功画像数</h3>
                <p class="text-3xl font-bold text-emerald-600">{successful}</p>
            </div>
            <div class="bg-white rounded-lg shadow-md p-6">
                <h3 class="text-lg font-semibold text-gray-700 mb-2">要改善数</h3>
                <p class="text-3xl font-bold text-red-600">{quality_dist["要改善"]}</p>
            </div>
        </div>
        {statistical_section}
        <div class="bg-white rounded-lg shadow-md p-6 mb-8">
            <h2 class="text-xl font-semibold text-gray-800 mb-4">品質分布</h2>
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div class="text-center">
                    <div class="bg-green-500 text-white px-2 py-1 rounded text-xs font-semibold inline-block mb-2">高品質</div>
                    <p class="text-2xl font-bold">{quality_dist['高品質']}</p>
                </div>
                <div class="text-center">
                    <div class="bg-yellow-500 text-white px-2 py-1 rounded text-xs font-semibold inline-block mb-2">中品質</div>
                    <p class="text-2xl font-bold">{quality_dist['中品質']}</p>
                </div>
                <div class="text-center">
                    <div class="bg-orange-500 text-white px-2 py-1 rounded text-xs font-semibold inline-block mb-2">低品質</div>
                    <p class="text-2xl font-bold">{quality_dist['低品質']}</p>
                </div>
                <div class="text-center">
                    <div class="bg-red-500 text-white px-2 py-1 rounded text-xs font-semibold inline-block mb-2">要改善</div>
                    <p class="text-2xl font-bold">{quality_dist['要改善']}</p>
                </div>
            </div>
        </div>

        <div class="bg-white rounded-lg shadow-md p-6">
            <h2 class="text-2xl font-bold text-gray-800 mb-6">🖼️ 抽出結果ギャラリー</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">"""

        # 画像カード生成（修正版）
        for r in results:
            if r.get("success"):
                # 両形式対応: image_name（チェックリスト仕様）とimage_path（実装形式）
                filename = r.get("image_name", r.get("image_path", "unknown.jpg"))

                # パス重複防止: extraction/ プレフィックスが既にある場合は除去
                if filename.startswith("extraction/"):
                    filename = filename.replace("extraction/", "", 1)

                # 両形式対応: quality_score（チェックリスト仕様）とquality_metrics.overall_score（実装形式）
                score = r.get(
                    "quality_score", r.get("quality_metrics", {}).get("overall_score", 0.0)
                )
                quality_label = self._get_quality_label(score)
                quality_class = self._get_quality_class(score)

                html += f"""
                <div class="border rounded-lg p-3 bg-gray-50">
                    <img src="/{tracker_id}/extraction/{filename}"
                         alt="{filename}"
                         class="w-full object-contain rounded"
                         onerror="this.parentElement.innerHTML='&lt;div class=&quot;p-4 text-center text-gray-500&quot;&gt;画像読み込みエラー&lt;br&gt;{filename}&lt;/div&gt;'">
                    <div class="mt-2 text-center">
                        <span class="{quality_class}">{quality_label}</span>
                        <p class="text-sm text-gray-600 mt-1">{filename}</p>
                        <p class="text-xs text-gray-500 mt-1">スコア: {score:.3f}</p>
                    </div>
                </div>"""

        html += """
            </div>
        </div>
    </div>
</body>
</html>"""

        return html

    def _generate_integrated_html(self, quality_data: Dict[str, Any]) -> str:
        """統合パイプライン用ダッシュボードHTML生成（DashboardGenerator準拠）"""
        tracker_id = quality_data.get("tracker_id", "unknown")
        success_rate = quality_data.get("success_rate", 0)
        total_images = quality_data.get("total_images", 0)
        successful_images = quality_data.get("successful_images", 0)

        return f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>統合品質ダッシュボード - {tracker_id}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header .subtitle {{
            margin-top: 10px;
            opacity: 0.9;
            font-size: 1.1em;
        }}
        .unified-badge {{
            margin-top: 10px;
            background: rgba(255,255,255,0.2);
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            display: inline-block;
        }}
        .metrics-section {{
            padding: 30px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 25px;
            text-align: center;
            border-left: 4px solid #667eea;
        }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #2c3e50;
            display: block;
            margin-bottom: 10px;
        }}
        .metric-label {{
            color: #7f8c8d;
            font-size: 1.1em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .success-rate {{
            color: #27ae60;
        }}
        .details-section {{
            background: #f8f9fa;
            padding: 30px;
        }}
        .details-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
        .detail-card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .detail-card h3 {{
            margin-top: 0;
            color: #2c3e50;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 10px;
        }}
        .status-indicator {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        .status-success {{
            background-color: #27ae60;
        }}
        .status-warning {{
            background-color: #f39c12;
        }}
        .status-error {{
            background-color: #e74c3c;
        }}
        .timestamp {{
            text-align: center;
            padding: 20px;
            color: #7f8c8d;
            border-top: 1px solid #ecf0f1;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>統合品質ダッシュボード</h1>
            <div class="subtitle">トラッカーID: {tracker_id}</div>
            <div class="unified-badge">🔗 統一ダッシュボードシステム v2.0 - DashboardGenerator</div>
        </div>

        <div class="metrics-section">
            <div class="metrics-grid">
                <div class="metric-card">
                    <span class="metric-value success-rate">{success_rate:.1f}%</span>
                    <div class="metric-label">抽出成功率</div>
                </div>
                <div class="metric-card">
                    <span class="metric-value">{total_images}</span>
                    <div class="metric-label">総画像数</div>
                </div>
                <div class="metric-card">
                    <span class="metric-value">{successful_images}</span>
                    <div class="metric-label">成功画像数</div>
                </div>
                <div class="metric-card">
                    <span class="metric-value">{total_images - successful_images}</span>
                    <div class="metric-label">失敗画像数</div>
                </div>
            </div>
        </div>

        <div class="details-section">
            <div class="details-grid">
                <div class="detail-card">
                    <h3>パイプライン実行状況</h3>
                    <p><span class="status-indicator status-success"></span>Phase 3: 品質確認 - 完了</p>
                    <p><span class="status-indicator status-success"></span>Phase 4: 抽出実行 - 完了</p>
                    <p><span class="status-indicator status-success"></span>Phase 5: 品質評価 - 完了</p>
                    <p><span class="status-indicator status-success"></span>Phase 6: ダッシュボード生成 - 完了</p>
                </div>
                <div class="detail-card">
                    <h3>品質指標詳細</h3>
                    <p>品質評価手法: balanced</p>
                    <p>YOLO検出閾値: 0.07</p>
                    <p>SAM最適化: 有効</p>
                    <p>後処理: 自動改善適用</p>
                </div>
                <div class="detail-card">
                    <h3>出力ディレクトリ</h3>
                    <p><strong>抽出結果:</strong><br>{self.workspace_dir}/extraction/</p>
                    <p><strong>品質レポート:</strong><br>{self.workspace_dir}/quality/</p>
                    <p><strong>ダッシュボード:</strong><br>{self.workspace_dir}/dashboard/</p>
                </div>
                <div class="detail-card">
                    <h3>システム情報</h3>
                    <p>実行環境: Python 3.x</p>
                    <p>CUDA: {"利用可能" if self._check_cuda() else "利用不可"}</p>
                    <p>パイプライン: Phase 3-6 統合版</p>
                    <p>バージョン: DashboardGenerator-2.0</p>
                </div>
            </div>
        </div>

        <div class="timestamp">
            生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>"""

    def _get_quality_label(self, score: float) -> str:
        """品質ラベル取得"""
        if score >= 0.8:
            return "高品質"
        elif score >= 0.6:
            return "中品質"
        elif score >= 0.4:
            return "低品質"
        else:
            return "要改善"

    def _get_quality_class(self, score: float) -> str:
        """品質クラス取得"""
        if score >= 0.8:
            return "bg-green-500 text-white px-2 py-1 rounded text-xs font-semibold"
        elif score >= 0.6:
            return "bg-yellow-500 text-white px-2 py-1 rounded text-xs font-semibold"
        elif score >= 0.4:
            return "bg-orange-500 text-white px-2 py-1 rounded text-xs font-semibold"
        else:
            return "bg-red-500 text-white px-2 py-1 rounded text-xs font-semibold"

    def _check_cuda(self) -> bool:
        """CUDA利用可能性チェック"""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False


# ===== 後方互換性のためのエイリアス =====
StandardDashboardGenerator = DashboardGenerator
