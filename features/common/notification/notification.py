"""
通知システム（Pushover）
"""

import json
import os
import requests
from pathlib import Path
from typing import Any, Dict, Optional, Union


class PushoverNotifier:
    """Pushover通知クライアント"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: 設定ファイルパス。Noneの場合はデフォルト位置を使用
        """
        if config_path is None:
            # プロジェクトルートのconfigディレクトリを参照
            project_root = Path(__file__).parent.parent.parent.parent
            config_path = project_root / "config" / "pushover.json"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.api_url = "https://api.pushover.net/1/messages.json"
    
    def _load_config(self) -> Optional[Dict[str, Any]]:
        """設定ファイルを読み込み"""
        try:
            if not self.config_path.exists():
                print(f"⚠️ Pushover設定ファイルが見つかりません: {self.config_path}")
                print(f"   {self.config_path}.example をコピーして設定してください")
                return None
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 必須項目チェック
            required_fields = ['token', 'user']
            for field in required_fields:
                if not config.get(field) or config[field] == f"your_{field}_here" or config[field] == f"your_{field}_key_here":
                    print(f"⚠️ Pushover設定の{field}が設定されていません")
                    return None
            
            return config
            
        except Exception as e:
            print(f"❌ Pushover設定読み込みエラー: {e}")
            return None
    
    def send_notification(self, message: str, title: Optional[str] = None, priority: int = 0) -> bool:
        """
        Pushover通知を送信
        
        Args:
            message: 通知メッセージ
            title: 通知タイトル（Noneの場合は設定ファイルのデフォルトを使用）
            priority: 優先度 (-2: 最低, -1: 低, 0: 通常, 1: 高, 2: 緊急)
        
        Returns:
            bool: 送信成功かどうか
        """
        if not self.config:
            print("⚠️ Pushover設定が無効なため通知をスキップします")
            return False
        
        try:
            # 通知データ作成
            data = {
                "token": self.config["token"],
                "user": self.config["user"],
                "message": message,
                "title": title or self.config.get("title", "Character Extraction"),
                "priority": priority
            }
            
            # デバイス指定がある場合
            if self.config.get("device"):
                data["device"] = self.config["device"]
            
            # API送信
            response = requests.post(self.api_url, data=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == 1:
                    print("📱 Pushover通知送信成功")
                    return True
                else:
                    print(f"❌ Pushover API エラー: {result.get('errors', 'Unknown error')}")
                    return False
            else:
                print(f"❌ Pushover HTTP エラー: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Pushover通信エラー: {e}")
            return False
        except Exception as e:
            print(f"❌ Pushover送信エラー: {e}")
            return False
    
    def send_notification_with_image(self, message: str, image_path: Union[str, Path], 
                                   title: Optional[str] = None, priority: int = 0) -> bool:
        """
        画像添付でPushover通知を送信
        
        Args:
            message: 通知メッセージ
            image_path: 添付する画像ファイルのパス
            title: 通知タイトル（Noneの場合は設定ファイルのデフォルトを使用）
            priority: 優先度 (-2: 最低, -1: 低, 0: 通常, 1: 高, 2: 緊急)
        
        Returns:
            bool: 送信成功かどうか
        """
        if not self.config:
            print("⚠️ Pushover設定が無効なため通知をスキップします")
            return False
        
        image_path = Path(image_path)
        if not image_path.exists():
            print(f"⚠️ 画像ファイルが見つかりません: {image_path}")
            return self.send_notification(message, title, priority)  # 画像なしで送信
        
        # ファイルサイズチェック（2.5MB制限）
        file_size = image_path.stat().st_size
        if file_size > 2.5 * 1024 * 1024:
            print(f"⚠️ 画像ファイルが大きすぎます: {file_size / 1024 / 1024:.1f}MB (制限: 2.5MB)")
            return self.send_notification(message, title, priority)  # 画像なしで送信
        
        # サポートされる画像形式チェック
        supported_formats = {'.jpg', '.jpeg', '.png', '.gif'}
        if image_path.suffix.lower() not in supported_formats:
            print(f"⚠️ サポートされていない画像形式: {image_path.suffix}")
            return self.send_notification(message, title, priority)  # 画像なしで送信
        
        try:
            # 通知データ作成
            data = {
                "token": self.config["token"],
                "user": self.config["user"],
                "message": message,
                "title": title or self.config.get("title", "Character Extraction"),
                "priority": priority
            }
            
            # デバイス指定がある場合
            if self.config.get("device"):
                data["device"] = self.config["device"]
            
            # 画像ファイルを添付
            with open(image_path, 'rb') as f:
                files = {"attachment": (image_path.name, f, "image/jpeg")}
                
                # multipart/form-dataでAPI送信
                response = requests.post(self.api_url, data=data, files=files, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == 1:
                    print(f"📱 Pushover画像付き通知送信成功: {image_path.name}")
                    return True
                else:
                    print(f"❌ Pushover API エラー: {result.get('errors', 'Unknown error')}")
                    return False
            else:
                print(f"❌ Pushover HTTP エラー: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Pushover通信エラー: {e}")
            return False
        except Exception as e:
            print(f"❌ Pushover送信エラー: {e}")
            return False
    
    def send_batch_complete(self, successful: int, total: int, failed: int, total_time: float) -> bool:
        """
        バッチ処理完了通知を送信
        
        Args:
            successful: 成功数
            total: 総数
            failed: 失敗数
            total_time: 処理時間（秒）
        
        Returns:
            bool: 送信成功かどうか
        """
        success_rate = (successful / total * 100) if total > 0 else 0
        
        message = f"""🎯 バッチ処理完了

📊 結果:
   成功: {successful}/{total} ({success_rate:.1f}%)
   失敗: {failed}
   処理時間: {total_time:.1f}秒

⚡ 1画像あたり平均: {total_time/total:.1f}秒"""
        
        # 成功率に応じて優先度設定
        if success_rate >= 90:
            priority = 0  # 通常
            title = "✅ キャラクター抽出完了"
        elif success_rate >= 70:
            priority = 0  # 通常
            title = "⚠️ キャラクター抽出完了（一部失敗）"
        else:
            priority = 1  # 高
            title = "❌ キャラクター抽出完了（多数失敗）"
        
        return self.send_notification(message, title, priority)
    
    def send_batch_complete_with_images(self, successful: int, total: int, failed: int, 
                                      total_time: float, image_dir: Union[str, Path], 
                                      max_images: int = 3) -> bool:
        """
        画像添付でバッチ処理完了通知を送信
        
        Args:
            successful: 成功数
            total: 総数
            failed: 失敗数
            total_time: 処理時間（秒）
            image_dir: 抽出された画像があるディレクトリ
            max_images: 添付する最大画像数
        
        Returns:
            bool: 送信成功かどうか
        """
        success_rate = (successful / total * 100) if total > 0 else 0
        
        message = f"""🎯 改善版バッチ処理完了

📊 結果:
   成功: {successful}/{total} ({success_rate:.1f}%)
   失敗: {failed}
   処理時間: {total_time:.1f}秒

⚡ 1画像あたり平均: {total_time/total:.1f}秒

🔧 改善点:
   ✅ マスク境界強化
   ✅ 多段階フォールバック
   ✅ F評価問題解消"""
        
        # 成功率に応じて優先度設定
        if success_rate >= 90:
            priority = 0  # 通常
            title = "✅ 改善版キャラクター抽出完了"
        elif success_rate >= 70:
            priority = 0  # 通常
            title = "⚠️ 改善版抽出完了（一部失敗）"
        else:
            priority = 1  # 高
            title = "❌ 改善版抽出完了（多数失敗）"
        
        # 画像ディレクトリから画像を取得
        image_dir = Path(image_dir)
        if image_dir.exists():
            image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
            if image_files:
                # 最初の画像を添付して送信
                first_image = image_files[0]
                
                # 送信する画像リストをメッセージに追加
                image_list = [f.name for f in image_files[:max_images]]
                if len(image_files) > max_images:
                    message += f"\n\n📸 抽出画像例（{max_images}/{len(image_files)}枚）:\n" + "\n".join(f"   • {name}" for name in image_list)
                else:
                    message += f"\n\n📸 抽出画像（{len(image_files)}枚）:\n" + "\n".join(f"   • {name}" for name in image_list)
                
                return self.send_notification_with_image(message, first_image, title, priority)
        
        # 画像がない場合は通常の通知
        return self.send_notification(message, title, priority)


def send_batch_notification(successful: int, total: int, failed: int, total_time: float) -> bool:
    """
    バッチ処理完了通知の簡易送信関数
    
    Args:
        successful: 成功数
        total: 総数
        failed: 失敗数
        total_time: 処理時間（秒）
    
    Returns:
        bool: 送信成功かどうか
    """
    # 既存のPushoverNotifierを優先使用
    notifier = PushoverNotifier()
    result = notifier.send_batch_complete(successful, total, failed, total_time)
    
    # 既存の方法で失敗した場合はglobal_pushoverを試行
    if not result:
        try:
            from .global_pushover import notify_process_complete
            result = notify_process_complete(
                title="キャラクター抽出完了",
                successful=successful,
                total=total,
                failed=failed,
                duration=total_time
            )
        except ImportError:
            pass
    
    return result

# 便利な関数群を追加（global_pushoverとの互換性）
def notify_success(title: str = "処理完了", message: str = "処理が正常に完了しました") -> bool:
    """成功通知"""
    notifier = PushoverNotifier()
    return notifier.send_notification(message, title, priority=0)

def notify_error(title: str = "エラー発生", message: str = "処理中にエラーが発生しました") -> bool:
    """エラー通知"""
    notifier = PushoverNotifier()
    return notifier.send_notification(message, title, priority=1)

def notify_warning(title: str = "警告", message: str = "注意が必要な状況が発生しました") -> bool:
    """警告通知"""
    notifier = PushoverNotifier()
    return notifier.send_notification(message, title, priority=0)


# 使用例
if __name__ == "__main__":
    # テスト通知
    notifier = PushoverNotifier()
    notifier.send_notification("テスト通知", "Character Extraction Test")