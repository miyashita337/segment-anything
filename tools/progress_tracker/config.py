#!/usr/bin/env python3
"""
進捗管理システム設定
Google Sheetsの設定とデフォルト値管理
"""

import os
from pathlib import Path
from .data_models import ProgressTrackerConfig


def get_default_config() -> ProgressTrackerConfig:
    """デフォルト設定取得"""
    
    # ユーザー指定のスプレッドシートID
    # 詳細: docs/google_sheets_reference.md を参照
    default_spreadsheet_id = "10B7JIXPR7AoVHBrLbIG6bvn4wfKha_SradJODwzUHFA"
    
    # 環境変数から設定を取得（オーバーライド可能）
    spreadsheet_id = os.environ.get('PROGRESS_TRACKER_SHEET_ID', default_spreadsheet_id)
    sheet_name = os.environ.get('PROGRESS_TRACKER_SHEET_NAME', 'Progress Tracker')
    auth_file = os.environ.get('PROGRESS_TRACKER_AUTH_FILE', 'config/google_sheets_auth.json')
    
    return ProgressTrackerConfig(
        spreadsheet_id=spreadsheet_id,
        sheet_name=sheet_name,
        auth_file_path=auth_file
    )


def setup_auth_file_template() -> str:
    """認証ファイルテンプレート作成"""
    
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    auth_template_path = config_dir / "google_sheets_auth.json.template"
    
    template_content = """{
  "type": "service_account",
  "project_id": "your-project-id",
  "private_key_id": "your-private-key-id",
  "private_key": "-----BEGIN PRIVATE KEY-----\\nYOUR_PRIVATE_KEY\\n-----END PRIVATE KEY-----\\n",
  "client_email": "your-service-account@your-project-id.iam.gserviceaccount.com",
  "client_id": "your-client-id",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://token.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/your-service-account%40your-project-id.iam.gserviceaccount.com"
}"""
    
    with open(auth_template_path, 'w') as f:
        f.write(template_content)
    
    return str(auth_template_path)


def check_configuration() -> dict:
    """設定状況チェック"""
    
    config = get_default_config()
    
    status = {
        'config_valid': True,
        'auth_file_exists': False,
        'spreadsheet_id_set': bool(config.spreadsheet_id),
        'issues': []
    }
    
    # 認証ファイル確認
    auth_path = Path(config.auth_file_path)
    if auth_path.exists():
        status['auth_file_exists'] = True
    else:
        status['config_valid'] = False
        status['issues'].append(f"認証ファイルが見つかりません: {auth_path}")
    
    # スプレッドシートID確認
    if not config.spreadsheet_id or config.spreadsheet_id == "your-spreadsheet-id":
        status['config_valid'] = False
        status['issues'].append("スプレッドシートIDが設定されていません")
    
    return status


def print_setup_instructions():
    """セットアップ手順表示"""
    print("""
🔧 Google Sheets 進捗管理システム セットアップ手順

1. Google Cloud Console設定:
   a) https://console.cloud.google.com/ にアクセス
   b) 新しいプロジェクトを作成 (例: "progress-tracker")
   c) Google Sheets API を有効化
      - ライブラリ → "Google Sheets API" を検索 → 有効化
   
2. サービスアカウント作成:
   a) IAM & Admin → サービスアカウント
   b) "サービスアカウントを作成" をクリック
   c) 名前: "progress-tracker-service" (任意)
   d) 役割: なし (スプレッドシート個別権限で管理)
   e) キーを作成 → JSON形式でダウンロード

3. スプレッドシート権限設定:
   a) https://docs.google.com/spreadsheets/d/10B7JIXPR7AoVHBrLbIG6bvn4wfKha_SradJODwzUHFA/edit
   b) 共有ボタンをクリック
   c) サービスアカウントのメールアドレスを追加
   d) 権限: "編集者" に設定

4. 認証ファイル配置:
   a) ダウンロードしたJSONファイルを次の場所に配置:
      config/google_sheets_auth.json
   b) または環境変数 PROGRESS_TRACKER_AUTH_FILE で指定

5. 動作確認:
   python tools/progress_tracker/cli.py --check-config

❗ 重要: 認証ファイルは機密情報です。Gitにコミットしないでください。
""")


if __name__ == "__main__":
    print_setup_instructions()