#!/usr/bin/env python3
"""
統合品質パイプライン (Phase 3-6)

安全性とロバスト性を重視した統合パイプラインシステム。
状態管理、レジューム機能、Web対応ダッシュボード生成を提供。

Created for: INTETETETETETETETET-010
Author: Claude Code Integration System
"""

import os
import sys
import json
import yaml
import argparse
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import subprocess

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 必要な機能は直接実装または subprocess で呼び出し
try:
    from features.common.notification.global_pushover import notify_process_complete, send_pushover_notification, notify_success
    PUSHOVER_AVAILABLE = True
except ImportError:
    PUSHOVER_AVAILABLE = False

# PHS-013: 復旧機能システム統合
try:
    from features.common.recovery_system import RecoveryManager, RecoveryState, AutoRecoverySystem
    RECOVERY_AVAILABLE = True
except ImportError:
    RECOVERY_AVAILABLE = False


@dataclass
class ValidationResult:
    """バリデーション結果"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


@dataclass
class PhaseResult:
    """フェーズ実行結果"""
    phase_name: str
    success: bool
    duration_seconds: float
    output_data: Dict[str, Any]
    errors: List[str]


@dataclass
class PipelineResult:
    """パイプライン実行結果"""
    tracker_id: str
    success: bool
    total_duration_seconds: float
    phase_results: List[PhaseResult]
    dashboard_url: Optional[str]


class ValidationEngine:
    """多層入力バリデーション"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def validate_input_paths(self, paths: List[str]) -> ValidationResult:
        """入力パス存在チェック"""
        errors = []
        warnings = []
        
        for path in paths:
            if not path or path.strip() == "":
                errors.append(
                    "❌ エラー: 入力パスが指定されていません\n"
                    "🔧 対処方法:\n"
                    "   1. 入力パスを明示的に指定してください\n"
                    "   2. 設定ファイルのpaths.default_inputを確認してください"
                )
                continue
            
            path_obj = Path(path)
            if not path_obj.exists():
                parent_dir = path_obj.parent
                errors.append(
                    f"❌ エラー: 入力ディレクトリが存在しません\n"
                    f"   パス: {path}\n\n"
                    f"🔧 対処方法:\n"
                    f"   1. パスの確認: ls {parent_dir}\n"
                    f"   2. 正しいパスの指定\n"
                    f"   3. 必要に応じてディレクトリ作成\n\n"
                    f"⚠️ 注意: 存在しないパスでの強制実行は品質保証違反です"
                )
                continue
                
            if not path_obj.is_dir():
                errors.append(f"エラー: {path} はディレクトリではありません")
                continue
                
            # 読み取り権限チェック
            if not os.access(path, os.R_OK):
                errors.append(f"エラー: {path} に読み取り権限がありません")
                continue
                
            # 空ディレクトリチェック
            try:
                files = list(path_obj.glob("*.jpg")) + list(path_obj.glob("*.png"))
                if not files:
                    warnings.append(f"警告: {path} に画像ファイルが見つかりません")
            except Exception as e:
                errors.append(f"エラー: {path} の読み込みに失敗: {str(e)}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def validate_configuration(self, config: Dict[str, Any]) -> ValidationResult:
        """設定ファイル整合性チェック"""
        errors = []
        warnings = []
        
        # 必須セクションチェック
        required_sections = ['pipeline', 'phases', 'paths']
        for section in required_sections:
            if section not in config:
                errors.append(f"エラー: 設定ファイルに必須セクション '{section}' がありません")
        
        # フェーズ定義チェック
        if 'phases' in config:
            required_phases = ['phase3', 'phase4', 'phase5', 'phase6']
            for phase in required_phases:
                if phase not in config['phases']:
                    errors.append(f"エラー: フェーズ '{phase}' の定義がありません")
        
        # パス設定チェック
        if 'paths' in config:
            paths_config = config['paths']
            if 'workspace_base' not in paths_config:
                errors.append("エラー: workspace_base パスが設定されていません")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def validate_dependencies(self) -> ValidationResult:
        """依存関係可用性チェック"""
        errors = []
        warnings = []
        
        # Python依存関係チェック
        required_modules = [
            'torch', 'cv2', 'ultralytics', 'segment_anything'
        ]
        
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                errors.append(f"エラー: 必須モジュール '{module}' がインストールされていません")
        
        # CUDAチェック
        try:
            import torch
            if not torch.cuda.is_available():
                warnings.append("警告: CUDA が利用できません（CPU処理となり処理時間が大幅に増加します）")
        except ImportError:
            pass
        
        # モデルファイルチェック
        model_files = [
            'sam_vit_h_4b8939.pth',
            'yolov8x.pt'
        ]
        
        for model_file in model_files:
            if not Path(model_file).exists():
                warnings.append(f"警告: モデルファイル '{model_file}' が見つかりません")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )


class StateManager:
    """パイプライン状態の永続化と復旧"""
    
    def __init__(self, tracker_id: str, workspace_dir: Path, logger: logging.Logger):
        self.tracker_id = tracker_id
        self.workspace_dir = workspace_dir
        self.state_file = workspace_dir / "pipeline_state.json"
        self.logger = logger
        
        # ワークスペースディレクトリ作成
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, phase: str, data: Dict[str, Any]) -> None:
        """チェックポイント保存"""
        try:
            state = {
                'tracker_id': self.tracker_id,
                'current_phase': phase,
                'timestamp': datetime.now().isoformat(),
                'phase_data': data
            }
            
            # 既存状態があれば読み込み
            if self.state_file.exists():
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    existing_state = json.load(f)
                if 'completed_phases' in existing_state:
                    state['completed_phases'] = existing_state['completed_phases']
                else:
                    state['completed_phases'] = []
            else:
                state['completed_phases'] = []
            
            # 現在のフェーズを完了リストに追加
            if phase not in state['completed_phases']:
                state['completed_phases'].append(phase)
            
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"チェックポイント保存完了: {phase}")
            
        except Exception as e:
            self.logger.error(f"チェックポイント保存失敗: {str(e)}")
            raise
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """チェックポイント復旧"""
        try:
            if not self.state_file.exists():
                return None
                
            with open(self.state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
                
            self.logger.info(f"チェックポイント復旧完了: {state.get('current_phase', 'unknown')}")
            return state
            
        except Exception as e:
            self.logger.error(f"チェックポイント復旧失敗: {str(e)}")
            return None
    
    def clear_checkpoint(self) -> None:
        """チェックポイントクリア（新規実行時）"""
        try:
            if self.state_file.exists():
                self.state_file.unlink()
                self.logger.info("チェックポイントクリア完了")
        except Exception as e:
            self.logger.error(f"チェックポイントクリア失敗: {str(e)}")


class DashboardGenerator:
    """Phase 6: Web対応ダッシュボード生成"""
    
    def __init__(self, config: Dict[str, Any], workspace_dir: Path, logger: logging.Logger, tracker_id: str = ""):
        self.config = config
        self.workspace_dir = workspace_dir
        self.dashboard_dir = workspace_dir / "dashboard"
        self.logger = logger
        self.tracker_id = tracker_id  # PHS-013: Pushover送信用
        
        # ダッシュボードディレクトリ作成
        self.dashboard_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_dashboard(self, quality_data: Dict[str, Any]) -> str:
        """HTMLダッシュボード生成"""
        try:
            dashboard_file = self.dashboard_dir / "dashboard.html"
            
            html_content = self._generate_html_content(quality_data)
            
            with open(dashboard_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"ダッシュボード生成完了: {dashboard_file}")
            return str(dashboard_file)
            
        except Exception as e:
            self.logger.error(f"ダッシュボード生成失敗: {str(e)}")
            raise
    
    def send_extraction_results_to_pushover(self, extraction_dir: Path, max_images: int = 10, demo_mode: bool = False) -> bool:
        """抽出結果をPushoverに送信（指定枚数まで）"""
        if not PUSHOVER_AVAILABLE:
            self.logger.warning("Pushover機能が利用できません")
            return False
        
        try:
            # 抽出された画像ファイルを取得
            extracted_files = list(extraction_dir.glob("extracted_*.png")) + \
                             list(extraction_dir.glob("extracted_*.jpg"))
            
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
                        self.logger.warning(f"ファイルサイズが大きすぎます: {image_file.name} ({file_size_mb:.1f}MB)")
                        continue
                    
                    title = f"統合パイプライン結果 {i}/{len(files_to_send)}"
                    message = (f"トラッカー: {self.tracker_id}\n"
                              f"ファイル: {image_file.name}\n"
                              f"サイズ: {file_size_mb:.2f}MB")
                    
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
            summary_message = (f"抽出結果送信完了\n"
                              f"成功: {success_count}/{len(files_to_send)}枚\n"
                              f"総抽出数: {len(extracted_files)}枚")
            
            if not demo_mode:
                from features.common.notification.global_pushover import send_pushover_notification
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
            import requests
            import json
            
            # 直接設定ファイルを読み込み
            config_path = Path("config/pushover.json")
            if not config_path.exists():
                self.logger.error(f"Pushover設定ファイルが存在しません: {config_path}")
                return False
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            if not config.get('api_token') or not config.get('user_key'):
                self.logger.error("Pushover設定にapi_tokenまたはuser_keyが不足しています")
                return False
            
            url = "https://api.pushover.net/1/messages.json"
            
            data = {
                'token': config['api_token'],
                'user': config['user_key'],
                'title': title,
                'message': message,
                'sound': 'magic'
            }
            
            with open(image_path, 'rb') as f:
                files = {'attachment': f}
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
    
    def _generate_html_content(self, quality_data: Dict[str, Any]) -> str:
        """HTML コンテンツ生成"""
        tracker_id = quality_data.get('tracker_id', 'unknown')
        success_rate = quality_data.get('success_rate', 0)
        total_images = quality_data.get('total_images', 0)
        successful_images = quality_data.get('successful_images', 0)
        
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
                    <p>バージョン: INTETETETETETETETET-010</p>
                </div>
            </div>
        </div>
        
        <div class="timestamp">
            生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>"""
    
    def _check_cuda(self) -> bool:
        """CUDA利用可能性チェック"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False


class IntegratedQualityPipeline:
    """Phase 3-6を統合した堅牢なパイプライン"""
    
    def __init__(self, config_path: str, tracker_id: str):
        self.config_path = config_path
        self.tracker_id = tracker_id
        self.config = self._load_config(config_path)
        
        # ロガー設定
        self.logger = self._setup_logging(tracker_id)
        
        # ワークスペース設定
        workspace_base = Path(self.config['paths']['workspace_base'])
        self.workspace_dir = workspace_base / tracker_id
        
        # コンポーネント初期化
        self.validator = ValidationEngine(self.logger)
        self.state_manager = StateManager(tracker_id, self.workspace_dir, self.logger)
        self.dashboard_generator = DashboardGenerator(self.config, self.workspace_dir, self.logger, tracker_id)
        
        # PHS-013: 復旧機能システム統合
        self.recovery_manager = None
        self.recovery_state = None
        if RECOVERY_AVAILABLE:
            self.recovery_manager = RecoveryManager(tracker_id, self.workspace_dir)
            self.logger.info("復旧機能システム統合完了")
        else:
            self.logger.warning("復旧機能システムは利用できません")
        
        self.logger.info(f"統合パイプライン初期化完了: {tracker_id}")
    
    def _setup_logging(self, tracker_id: str) -> logging.Logger:
        """ロギング設定"""
        # ログディレクトリ作成
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # ロガー作成
        logger = logging.getLogger(f"integrated_pipeline_{tracker_id}")
        logger.setLevel(logging.INFO)
        
        # 既存ハンドラーがあれば削除
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # ファイルハンドラー
        log_file = log_dir / f"{tracker_id}_integrated_pipeline.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # コンソールハンドラー
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # フォーマッター
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # ハンドラー追加
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"設定ファイル読み込み失敗: {str(e)}")
    
    def execute_pipeline(self, resume: bool = False) -> PipelineResult:
        """統合パイプライン実行"""
        start_time = datetime.now()
        phase_results = []
        
        try:
            self.logger.info(f"統合パイプライン開始: {self.tracker_id}")
            
            # レジューム処理
            completed_phases = []
            if resume:
                checkpoint = self.state_manager.load_checkpoint()
                if checkpoint:
                    completed_phases = checkpoint.get('completed_phases', [])
                    self.logger.info(f"レジューム実行: 完了済みフェーズ {completed_phases}")
            else:
                self.state_manager.clear_checkpoint()
            
            # Phase 3: 品質確認
            if 'phase3' not in completed_phases:
                phase_result = self._execute_phase3()
                phase_results.append(phase_result)
                if not phase_result.success:
                    return self._create_pipeline_result(start_time, False, phase_results)
            
            # Phase 4: 抽出実行
            if 'phase4' not in completed_phases:
                phase_result = self._execute_phase4()
                phase_results.append(phase_result)
                if not phase_result.success:
                    return self._create_pipeline_result(start_time, False, phase_results)
            
            # Phase 5: 品質評価
            if 'phase5' not in completed_phases:
                phase_result = self._execute_phase5()
                phase_results.append(phase_result)
                if not phase_result.success:
                    return self._create_pipeline_result(start_time, False, phase_results)
            
            # Phase 6: ダッシュボード生成
            if 'phase6' not in completed_phases:
                phase_result = self._execute_phase6()
                phase_results.append(phase_result)
                if not phase_result.success:
                    return self._create_pipeline_result(start_time, False, phase_results)
            
            # 全フェーズ完了
            total_duration = (datetime.now() - start_time).total_seconds()
            dashboard_url = str(self.workspace_dir / "dashboard" / "dashboard.html")
            
            self.logger.info(f"統合パイプライン完了: {self.tracker_id} ({total_duration:.1f}秒)")
            
            return PipelineResult(
                tracker_id=self.tracker_id,
                success=True,
                total_duration_seconds=total_duration,
                phase_results=phase_results,
                dashboard_url=dashboard_url
            )
            
        except Exception as e:
            self.logger.error(f"統合パイプライン失敗: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            total_duration = (datetime.now() - start_time).total_seconds()
            return PipelineResult(
                tracker_id=self.tracker_id,
                success=False,
                total_duration_seconds=total_duration,
                phase_results=phase_results,
                dashboard_url=None
            )

    def execute_pipeline_with_recovery(self, resume: bool = False, max_retries: int = 3) -> PipelineResult:
        """復旧機能付き統合パイプライン実行 (PHS-013)"""
        if not RECOVERY_AVAILABLE:
            self.logger.warning("復旧機能が利用できません、通常実行にフォールバック")
            return self.execute_pipeline(resume)
        
        start_time = datetime.now()
        
        # 復旧セッション初期化
        self.recovery_state = self.recovery_manager.initialize_recovery_session("pipeline_start")
        
        for attempt in range(max_retries + 1):
            try:
                self.logger.info(f"パイプライン実行試行 {attempt + 1}/{max_retries + 1}")
                
                # 通常のパイプライン実行
                result = self.execute_pipeline(resume)
                
                if result.success:
                    # 成功時はセッション終了
                    self.recovery_manager.cleanup_recovery_session()
                    self.logger.info("復旧機能付きパイプライン実行成功")
                    return result
                else:
                    # 失敗時は復旧処理
                    error_msg = f"パイプライン実行失敗 (試行 {attempt + 1})"
                    if attempt < max_retries:
                        can_recover = self.recovery_manager.handle_failure(
                            self.recovery_state, error_msg
                        )
                        if can_recover:
                            self.logger.info("復旧処理完了、再試行します")
                            resume = True  # 次回はレジューム実行
                            continue
                    
                    # 復旧不可能または最大試行回数到達
                    self.logger.error("復旧不可能、処理を終了します")
                    return result
                    
            except Exception as e:
                error_msg = f"パイプライン実行中エラー: {str(e)}"
                self.logger.error(error_msg)
                
                if attempt < max_retries and self.recovery_manager:
                    can_recover = self.recovery_manager.handle_failure(
                        self.recovery_state, error_msg
                    )
                    if can_recover:
                        self.logger.info("例外からの復旧処理完了、再試行します")
                        resume = True
                        continue
                
                # 最終的な失敗
                total_duration = (datetime.now() - start_time).total_seconds()
                return PipelineResult(
                    tracker_id=self.tracker_id,
                    success=False,
                    total_duration_seconds=total_duration,
                    phase_results=[],
                    dashboard_url=None
                )
        
        # ここには到達しないはずですが安全のため
        total_duration = (datetime.now() - start_time).total_seconds()
        return PipelineResult(
            tracker_id=self.tracker_id,
            success=False,
            total_duration_seconds=total_duration,
            phase_results=[],
            dashboard_url=None
        )
    
    def _execute_phase3(self) -> PhaseResult:
        """Phase 3: 品質確認実行"""
        phase_start = datetime.now()
        errors = []
        
        try:
            self.logger.info("Phase 3: 品質確認開始")
            
            # 入力パスバリデーション
            input_paths = [self.config['paths'].get('default_input', '')]
            validation_result = self.validator.validate_input_paths(input_paths)
            
            if not validation_result.is_valid:
                errors.extend(validation_result.errors)
                for error in validation_result.errors:
                    self.logger.error(error)
                
                return PhaseResult(
                    phase_name="phase3",
                    success=False,
                    duration_seconds=(datetime.now() - phase_start).total_seconds(),
                    output_data={},
                    errors=errors
                )
            
            # 設定ファイルバリデーション
            config_validation = self.validator.validate_configuration(self.config)
            if not config_validation.is_valid:
                errors.extend(config_validation.errors)
            
            # 依存関係バリデーション
            deps_validation = self.validator.validate_dependencies()
            if not deps_validation.is_valid:
                errors.extend(deps_validation.errors)
            
            if errors:
                return PhaseResult(
                    phase_name="phase3",
                    success=False,
                    duration_seconds=(datetime.now() - phase_start).total_seconds(),
                    output_data={},
                    errors=errors
                )
            
            # チェックポイント保存
            self.state_manager.save_checkpoint('phase3', {
                'validation_passed': True,
                'input_paths': input_paths
            })
            
            duration = (datetime.now() - phase_start).total_seconds()
            self.logger.info(f"Phase 3: 品質確認完了 ({duration:.1f}秒)")
            
            return PhaseResult(
                phase_name="phase3",
                success=True,
                duration_seconds=duration,
                output_data={'validation_passed': True},
                errors=[]
            )
            
        except Exception as e:
            errors.append(f"Phase 3 実行エラー: {str(e)}")
            self.logger.error(f"Phase 3 失敗: {str(e)}")
            
            return PhaseResult(
                phase_name="phase3",
                success=False,
                duration_seconds=(datetime.now() - phase_start).total_seconds(),
                output_data={},
                errors=errors
            )
    
    def _execute_phase4(self) -> PhaseResult:
        """Phase 4: 抽出実行"""
        phase_start = datetime.now()
        errors = []
        
        try:
            self.logger.info("Phase 4: 抽出実行開始")
            
            # 抽出ディレクトリ準備
            extraction_dir = self.workspace_dir / "extraction"
            extraction_dir.mkdir(parents=True, exist_ok=True)
            
            # 抽出パイプライン実行
            # PHS-013専用: kana08データセット使用
            if self.tracker_id == "PHS-013":
                input_path = "/mnt/c/AItools/lora/train/yado/org/kana08/"
            else:
                input_path = self.config['paths'].get('default_input', '')
            
            # 新アーキテクチャの抽出コマンド使用
            extraction_script = project_root / "features" / "extraction" / "commands" / "extract_character.py"
            
            cmd = [
                sys.executable, str(extraction_script),
                input_path,
                "-o", str(extraction_dir),
                "--batch"
            ]
            
            self.logger.info(f"抽出コマンド実行: {' '.join(cmd)}")
            
            # PYTHONPATHを設定して実行
            env = os.environ.copy()
            env['PYTHONPATH'] = str(project_root)
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1200,  # 20分タイムアウト
                cwd=str(project_root),
                env=env
            )
            
            if result.returncode != 0:
                error_msg = f"抽出処理失敗 (終了コード: {result.returncode})\n"
                error_msg += f"stderr: {result.stderr}\n"
                error_msg += f"stdout: {result.stdout}"
                errors.append(error_msg)
                self.logger.error(error_msg)
                
                return PhaseResult(
                    phase_name="phase4",
                    success=False,
                    duration_seconds=(datetime.now() - phase_start).total_seconds(),
                    output_data={},
                    errors=errors
                )
            
            # 抽出結果確認
            extracted_files = list(extraction_dir.glob("*.jpg")) + list(extraction_dir.glob("*.png"))
            
            # チェックポイント保存
            self.state_manager.save_checkpoint('phase4', {
                'extraction_completed': True,
                'extracted_files_count': len(extracted_files),
                'extraction_dir': str(extraction_dir)
            })
            
            duration = (datetime.now() - phase_start).total_seconds()
            self.logger.info(f"Phase 4: 抽出実行完了 ({duration:.1f}秒, {len(extracted_files)}ファイル)")
            
            return PhaseResult(
                phase_name="phase4",
                success=True,
                duration_seconds=duration,
                output_data={
                    'extracted_files_count': len(extracted_files),
                    'extraction_dir': str(extraction_dir)
                },
                errors=[]
            )
            
        except Exception as e:
            errors.append(f"Phase 4 実行エラー: {str(e)}")
            self.logger.error(f"Phase 4 失敗: {str(e)}")
            
            return PhaseResult(
                phase_name="phase4",
                success=False,
                duration_seconds=(datetime.now() - phase_start).total_seconds(),
                output_data={},
                errors=errors
            )
    
    def _execute_phase5(self) -> PhaseResult:
        """Phase 5: 品質評価・レポート生成"""
        phase_start = datetime.now()
        errors = []
        
        try:
            self.logger.info("Phase 5: 品質評価開始")
            
            # 品質レポートディレクトリ準備
            quality_dir = self.workspace_dir / "quality"
            quality_dir.mkdir(parents=True, exist_ok=True)
            
            # 抽出結果から品質評価データ生成
            extraction_dir = self.workspace_dir / "extraction"
            extracted_files = list(extraction_dir.glob("*.jpg")) + list(extraction_dir.glob("*.png"))
            
            # 入力画像数確認
            input_path = Path(self.config['paths'].get('default_input', ''))
            input_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
            
            total_images = len(input_files)
            successful_images = len(extracted_files)
            success_rate = (successful_images / total_images * 100) if total_images > 0 else 0
            
            # 品質データ作成
            quality_data = {
                'tracker_id': self.tracker_id,
                'total_images': total_images,
                'successful_images': successful_images,
                'failed_images': total_images - successful_images,
                'success_rate': success_rate,
                'evaluation_timestamp': datetime.now().isoformat(),
                'quality_method': 'balanced',
                'extraction_dir': str(extraction_dir),
                'input_dir': str(input_path)
            }
            
            # 品質レポート保存
            quality_report_file = quality_dir / "quality_report.json"
            with open(quality_report_file, 'w', encoding='utf-8') as f:
                json.dump(quality_data, f, ensure_ascii=False, indent=2)
            
            # チェックポイント保存
            self.state_manager.save_checkpoint('phase5', quality_data)
            
            duration = (datetime.now() - phase_start).total_seconds()
            self.logger.info(f"Phase 5: 品質評価完了 ({duration:.1f}秒, 成功率: {success_rate:.1f}%)")
            
            return PhaseResult(
                phase_name="phase5",
                success=True,
                duration_seconds=duration,
                output_data=quality_data,
                errors=[]
            )
            
        except Exception as e:
            errors.append(f"Phase 5 実行エラー: {str(e)}")
            self.logger.error(f"Phase 5 失敗: {str(e)}")
            
            return PhaseResult(
                phase_name="phase5",
                success=False,
                duration_seconds=(datetime.now() - phase_start).total_seconds(),
                output_data={},
                errors=errors
            )
    
    def _execute_phase6(self) -> PhaseResult:
        """Phase 6: ダッシュボード生成"""
        phase_start = datetime.now()
        errors = []
        
        try:
            self.logger.info("Phase 6: ダッシュボード生成開始")
            
            # Phase 5の品質データ読み込み
            checkpoint = self.state_manager.load_checkpoint()
            if not checkpoint or 'phase_data' not in checkpoint:
                errors.append("Phase 5の品質データが見つかりません")
                return PhaseResult(
                    phase_name="phase6",
                    success=False,
                    duration_seconds=(datetime.now() - phase_start).total_seconds(),
                    output_data={},
                    errors=errors
                )
            
            quality_data = checkpoint['phase_data']
            
            # ダッシュボード生成
            dashboard_file = self.dashboard_generator.generate_dashboard(quality_data)
            
            # Pushover結果送信（オプション）
            pushover_success = False
            extraction_dir = self.workspace_dir / "extraction"
            if extraction_dir.exists() and PUSHOVER_AVAILABLE:
                self.logger.info("Pushover結果送信開始...")
                pushover_success = self.dashboard_generator.send_extraction_results_to_pushover(extraction_dir, max_images=10)
            
            # チェックポイント保存
            self.state_manager.save_checkpoint('phase6', {
                'dashboard_generated': True,
                'dashboard_file': dashboard_file,
                'pushover_sent': pushover_success
            })
            
            duration = (datetime.now() - phase_start).total_seconds()
            self.logger.info(f"Phase 6: ダッシュボード生成完了 ({duration:.1f}秒)")
            
            return PhaseResult(
                phase_name="phase6",
                success=True,
                duration_seconds=duration,
                output_data={
                    'dashboard_file': dashboard_file,
                    'dashboard_url': f"file://{dashboard_file}"
                },
                errors=[]
            )
            
        except Exception as e:
            errors.append(f"Phase 6 実行エラー: {str(e)}")
            self.logger.error(f"Phase 6 失敗: {str(e)}")
            
            return PhaseResult(
                phase_name="phase6",
                success=False,
                duration_seconds=(datetime.now() - phase_start).total_seconds(),
                output_data={},
                errors=errors
            )
    
    def _create_pipeline_result(self, start_time: datetime, success: bool, 
                                phase_results: List[PhaseResult]) -> PipelineResult:
        """パイプライン結果作成"""
        total_duration = (datetime.now() - start_time).total_seconds()
        dashboard_url = None
        
        if success and phase_results:
            # Phase 6の結果からダッシュボードURL取得
            phase6_result = next((pr for pr in phase_results if pr.phase_name == 'phase6'), None)
            if phase6_result and phase6_result.success:
                dashboard_url = phase6_result.output_data.get('dashboard_url')
        
        return PipelineResult(
            tracker_id=self.tracker_id,
            success=success,
            total_duration_seconds=total_duration,
            phase_results=phase_results,
            dashboard_url=dashboard_url
        )


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="統合品質パイプライン (Phase 3-6)")
    parser.add_argument('--config', required=True, help='設定ファイルパス')
    parser.add_argument('--tracker-id', required=True, help='トラッカーID')
    parser.add_argument('--resume', action='store_true', help='レジューム実行')
    parser.add_argument('--verbose', action='store_true', help='詳細ログ')
    
    args = parser.parse_args()
    
    try:
        # パイプライン実行
        pipeline = IntegratedQualityPipeline(args.config, args.tracker_id)
        result = pipeline.execute_pipeline(resume=args.resume)
        
        # 結果出力
        print(f"\n=== 統合パイプライン実行結果 ===")
        print(f"トラッカーID: {result.tracker_id}")
        print(f"成功: {'✅ Yes' if result.success else '❌ No'}")
        print(f"実行時間: {result.total_duration_seconds:.1f}秒")
        
        if result.dashboard_url:
            print(f"ダッシュボード: {result.dashboard_url}")
        
        print(f"\n=== フェーズ別結果 ===")
        for phase_result in result.phase_results:
            status = "✅ 成功" if phase_result.success else "❌ 失敗"
            print(f"{phase_result.phase_name}: {status} ({phase_result.duration_seconds:.1f}秒)")
            
            if phase_result.errors:
                for error in phase_result.errors:
                    print(f"  エラー: {error}")
        
        # 終了コード設定
        sys.exit(0 if result.success else 1)
        
    except Exception as e:
        print(f"❌ パイプライン実行失敗: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()