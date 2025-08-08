#!/usr/bin/env python3
"""
Google Sheets ポーリング監視システム

バックグラウンド実行中のバッチ処理の進捗を監視
- 動的間隔調整によるトークン節約
- 変更検知とリアルタイム通知
- 詳細な進捗解析と表示
"""

import asyncio
import json
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.progress_tracker.progress_manager import ProgressManager
from tools.progress_tracker.data_models import TaskStatus, ProgressTrackerConfig

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SheetsPollingMonitor:
    """Google Sheets ポーリング監視システム"""
    
    def __init__(self, tracker_id: str):
        self.tracker_id = tracker_id
        
        # ProgressManager初期化（デフォルト設定を作成）
        try:
            config = ProgressTrackerConfig(
                spreadsheet_id="1o83oR-oKWFwYBYBQT19KAeZO86oMRmzJJKVpEg5YhNg",
                sheet_name="Progress Tracker",
                auth_file_path="config/google_sheets_auth.json"
            )
            self.progress_manager = ProgressManager(config)
        except Exception as e:
            logger.warning(f"ProgressManager初期化に失敗: {e}")
            self.progress_manager = None
        
        # 監視設定
        self.initial_interval = 60    # 1分間隔
        self.medium_interval = 180    # 3分間隔  
        self.long_interval = 300      # 5分間隔
        self.max_monitoring_hours = 8 # 最大監視時間
        
        # 状態管理
        self.last_updated_time = None
        self.last_details = None
        self.monitoring_start_time = datetime.now()
        self.change_detected = False
        
        # 統計
        self.poll_count = 0
        self.token_usage_estimate = 0
        
        logger.info(f"ポーリング監視システム初期化: {tracker_id}")
    
    def calculate_current_interval(self) -> int:
        """現在の監視間隔を動的計算"""
        elapsed = datetime.now() - self.monitoring_start_time
        elapsed_hours = elapsed.total_seconds() / 3600
        
        if elapsed_hours < 1:
            return self.initial_interval      # 最初の1時間: 1分間隔
        elif elapsed_hours < 3:
            return self.medium_interval       # 1-3時間: 3分間隔
        else:
            return self.long_interval         # 3時間以降: 5分間隔
    
    def estimate_token_usage(self, api_call_tokens: int = 400) -> int:
        """トークン使用量推定"""
        self.poll_count += 1
        self.token_usage_estimate += api_call_tokens
        return self.token_usage_estimate
    
    async def fetch_current_status(self) -> Tuple[Optional[Dict[str, Any]], bool]:
        """現在のステータスを取得"""
        try:
            # ProgressManagerが無効な場合はダミーデータを返す
            if not self.progress_manager:
                return None, False
                
            # Google Sheets から情報取得
            task_info = self.progress_manager.get_task_status(self.tracker_id)
            
            if not task_info:
                logger.warning(f"トラッカー {self.tracker_id} が見つかりません")
                return None, False
            
            # 変更検知
            current_updated_time = task_info.get('updated_date')
            current_details = task_info.get('details', '{}')
            
            change_detected = (
                self.last_updated_time != current_updated_time or
                self.last_details != current_details
            )
            
            if change_detected:
                self.last_updated_time = current_updated_time
                self.last_details = current_details
                self.change_detected = True
            
            # トークン使用量推定
            self.estimate_token_usage()
            
            return task_info, change_detected
            
        except Exception as e:
            logger.error(f"ステータス取得エラー: {e}")
            return None, False
    
    def parse_progress_details(self, details_json: str) -> Dict[str, Any]:
        """進捗詳細をパース"""
        try:
            if not details_json:
                return {}
            
            details = json.loads(details_json)
            
            # 進捗情報の抽出・正規化
            parsed = {
                'progress_rate': details.get('progress_rate', '0%'),
                'current_success_rate': details.get('current_success_rate', '0%'),
                'processed': details.get('processed', 0),
                'failed': details.get('failed', 0),
                'remaining': details.get('remaining', 0),
                'total_images': details.get('total_images', 0),
                'execution_time': details.get('execution_time', 'N/A'),
                'avg_time_per_image': details.get('avg_time_per_image', 'N/A'),
                'timestamp': details.get('timestamp', ''),
                'error': details.get('error', None)
            }
            
            return parsed
            
        except json.JSONDecodeError:
            logger.warning(f"詳細情報のJSON解析失敗: {details_json[:100]}...")
            return {}
        except Exception as e:
            logger.warning(f"詳細情報解析エラー: {e}")
            return {}
    
    def display_status_update(self, task_info: Dict[str, Any], details: Dict[str, Any]):
        """ステータス更新の表示"""
        status = task_info.get('status', 'unknown')
        extraction_status = task_info.get('extraction_pipeline', 'unknown')
        
        print(f"\n{'='*60}")
        print(f"📊 {self.tracker_id} ステータス更新 - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        
        # 基本情報
        print(f"🔄 ステータス: {status}")
        print(f"⚙️  抽出パイプライン: {extraction_status}")
        print(f"📅 最終更新: {task_info.get('updated_date', 'N/A')}")
        
        # 進捗詳細
        if details:
            print(f"\n📈 進捗詳細:")
            print(f"   全体進捗: {details.get('progress_rate', 'N/A')}")
            
            if details.get('total_images', 0) > 0:
                print(f"   処理済み: {details.get('processed', 0)}枚")
                print(f"   失敗: {details.get('failed', 0)}枚")
                print(f"   残り: {details.get('remaining', 0)}枚")
                print(f"   現在成功率: {details.get('current_success_rate', 'N/A')}")
            
            if details.get('execution_time') != 'N/A':
                print(f"   実行時間: {details.get('execution_time', 'N/A')}")
                print(f"   平均処理時間: {details.get('avg_time_per_image', 'N/A')}")
        
        # エラー情報
        if details.get('error'):
            print(f"\n❌ エラー: {details['error']}")
        
        # 最終結果（完了時）
        if status == TaskStatus.RELEASE.value:
            success_rate = task_info.get('ab_evaluation_rate', 'N/A')
            fps = task_info.get('fps', 'N/A')
            print(f"\n🎉 完了結果:")
            print(f"   成功率: {success_rate}")
            print(f"   処理速度: {fps}枚/時間")
        
        print(f"{'='*60}\n")
    
    def display_monitoring_summary(self):
        """監視サマリー表示"""
        elapsed = datetime.now() - self.monitoring_start_time
        elapsed_str = str(elapsed).split('.')[0]  # ミリ秒削除
        
        print(f"\n📋 監視サマリー")
        print(f"   監視時間: {elapsed_str}")
        print(f"   ポーリング回数: {self.poll_count}")
        print(f"   推定トークン使用量: {self.token_usage_estimate:,}")
        print(f"   平均間隔: {elapsed.total_seconds() / self.poll_count:.1f}秒" if self.poll_count > 0 else "   平均間隔: N/A")
    
    async def monitor_extraction_progress(self) -> bool:
        """抽出進捗の監視メインループ"""
        print(f"🚀 {self.tracker_id} の監視を開始します")
        print(f"⏰ 監視間隔: 1分 → 3分 → 5分（時間経過で調整）")
        print(f"🔄 最大監視時間: {self.max_monitoring_hours}時間\n")
        
        monitoring_end_time = self.monitoring_start_time + timedelta(hours=self.max_monitoring_hours)
        
        while datetime.now() < monitoring_end_time:
            try:
                # 現在のステータス取得
                task_info, changed = await self.fetch_current_status()
                
                if not task_info:
                    print(f"⚠️  {datetime.now().strftime('%H:%M:%S')} - ステータス取得失敗")
                    await asyncio.sleep(self.initial_interval)
                    continue
                
                # 変更があった場合のみ詳細表示
                if changed:
                    details = self.parse_progress_details(task_info.get('details', '{}'))
                    self.display_status_update(task_info, details)
                    
                    # 完了チェック
                    status = task_info.get('status')
                    if status == TaskStatus.RELEASE.value:
                        print(f"✅ {self.tracker_id} が正常完了しました！")
                        return True
                    elif status in [TaskStatus.COMPLETED.value] and task_info.get('extraction_pipeline') == 'completed':
                        print(f"✅ {self.tracker_id} の抽出処理が完了しました！")
                        return True
                    elif task_info.get('extraction_pipeline') == 'failed':
                        print(f"❌ {self.tracker_id} の抽出処理が失敗しました")
                        return False
                else:
                    # 変更なしの場合は簡易表示
                    current_time = datetime.now().strftime('%H:%M:%S')
                    print(f"📡 {current_time} - 監視中... (変更なし, ポーリング#{self.poll_count})")
                
                # 次のポーリングまで待機
                interval = self.calculate_current_interval()
                await asyncio.sleep(interval)
                
            except KeyboardInterrupt:
                print(f"\n🛑 ユーザーによって監視が中断されました")
                return False
            except Exception as e:
                logger.error(f"監視エラー: {e}")
                await asyncio.sleep(self.initial_interval)
        
        print(f"⏰ 最大監視時間 ({self.max_monitoring_hours}時間) に達しました")
        return False
    
    async def start_monitoring(self) -> bool:
        """監視開始"""
        try:
            success = await self.monitor_extraction_progress()
            self.display_monitoring_summary()
            return success
        except Exception as e:
            logger.error(f"監視システムエラー: {e}")
            return False


async def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Google Sheetsポーリング監視システム")
    parser.add_argument("tracker_id", help="監視対象トラッカーID")
    parser.add_argument("--max-hours", type=int, default=8, help="最大監視時間（時間）")
    parser.add_argument("--initial-interval", type=int, default=60, help="初期監視間隔（秒）")
    
    args = parser.parse_args()
    
    monitor = SheetsPollingMonitor(args.tracker_id)
    if args.max_hours:
        monitor.max_monitoring_hours = args.max_hours
    if args.initial_interval:
        monitor.initial_interval = args.initial_interval
    
    try:
        success = await monitor.start_monitoring()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n監視が中断されました")
        sys.exit(1)
    except Exception as e:
        logger.error(f"監視システムエラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())