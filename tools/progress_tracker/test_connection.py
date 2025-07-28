#!/usr/bin/env python3
"""
Google Sheets接続テストスクリプト
"""

import sys
from pathlib import Path

# プロジェクトルート追加
sys.path.append(str(Path(__file__).parent.parent.parent))

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

def test_connection():
    """接続テスト"""
    try:
        # 認証
        SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
        credentials = Credentials.from_service_account_file(
            'config/google_sheets_auth.json', scopes=SCOPES
        )
        
        service = build('sheets', 'v4', credentials=credentials)
        
        # スプレッドシート情報取得
        # 詳細: docs/google_sheets_reference.md を参照
        spreadsheet_id = '10B7JIXPR7AoVHBrLbIG6bvn4wfKha_SradJODwzUHFA'
        
        print("📋 スプレッドシート情報取得中...")
        spreadsheet = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
        
        print(f"✅ スプレッドシートタイトル: {spreadsheet.get('properties', {}).get('title')}")
        print(f"📊 シート一覧:")
        
        for sheet in spreadsheet.get('sheets', []):
            sheet_props = sheet.get('properties', {})
            print(f"  - シート名: '{sheet_props.get('title')}'")
            print(f"    ID: {sheet_props.get('sheetId')}")
            print(f"    グリッド: {sheet_props.get('gridProperties', {})}")
        
        # 最初のシートで読み取りテスト
        if spreadsheet.get('sheets'):
            first_sheet = spreadsheet['sheets'][0]['properties']['title']
            print(f"\n🔍 '{first_sheet}'シートでテスト読み取り...")
            
            # A1セルのみ読み取り
            range_name = f"{first_sheet}!A1"
            result = service.spreadsheets().values().get(
                spreadsheetId=spreadsheet_id,
                range=range_name
            ).execute()
            
            values = result.get('values', [])
            if values:
                print(f"✅ A1セルの値: {values[0][0] if values[0] else '(空)'}")
            else:
                print("ℹ️ A1セルは空です")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_connection()