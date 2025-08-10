#!/usr/bin/env python3
"""
ハードコーディング除去・設定ファイルベース移行スクリプト
"""

import os
from pathlib import Path

def migrate_imports():
    """インポート文を設定ベース版に移行"""
    
    files_to_update = [
        'features/extraction/commands/extract_character.py',
        'tools/core/sam_yolo_character_segment.py',
        'tools/scripts/merge_qca001_dashboards.py',
    ]
    
    for file_path in files_to_update:
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"⚠️ ファイル未発見: {file_path}")
            continue
            
        try:
            # ファイル読み込み
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # インポート文の置換
            old_import = "from features.adaptation.author_parameter_adapter import AuthorParameterAdapter"
            new_import = "from features.adaptation.author_parameter_adapter_v2 import AuthorParameterAdapterV2 as AuthorParameterAdapter"
            
            if old_import in content:
                content = content.replace(old_import, new_import)
                
                # ファイル更新
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"✅ 更新完了: {file_path}")
            else:
                print(f"📋 更新不要: {file_path}")
                
        except Exception as e:
            print(f"❌ 更新エラー {file_path}: {e}")

def create_wrapper_script():
    """旧システム互換性ラッパー作成"""
    
    wrapper_content = '''#!/usr/bin/env python3
"""
旧AuthorParameterAdapterの互換性ラッパー
既存コードとの互換性を保ちつつ、設定ファイルベースシステムを使用
"""

# 新しい設定ベースシステムを使用
from features.adaptation.author_parameter_adapter_v2 import (
    AuthorParameterAdapterV2,
    AuthorProfile,
    AuthorCharacteristics,
    detect_author_from_path
)

# 後方互換性のためのエイリアス
AuthorParameterAdapter = AuthorParameterAdapterV2

# 静的メソッドの互換性サポート（古い使い方）
class AuthorParameterAdapterLegacy:
    """レガシー互換性クラス"""
    
    # クラス変数として設定ベースアダプターを保持
    _adapter = None
    
    @classmethod
    def _get_adapter(cls):
        if cls._adapter is None:
            cls._adapter = AuthorParameterAdapterV2()
        return cls._adapter
    
    @classmethod
    @property
    def AUTHOR_PROFILES(cls):
        return cls._get_adapter().AUTHOR_PROFILES
    
    @staticmethod
    def detect_author_from_path(image_path: str):
        return detect_author_from_path(image_path)

# 互換性のための上書き（必要に応じて）
if False:  # レガシーサポートが必要な場合のみ True に変更
    AuthorParameterAdapter = AuthorParameterAdapterLegacy
'''
    
    wrapper_path = Path('features/adaptation/author_parameter_adapter_legacy.py')
    with open(wrapper_path, 'w', encoding='utf-8') as f:
        f.write(wrapper_content)
    
    print(f"✅ 互換性ラッパー作成: {wrapper_path}")

def main():
    """メイン実行関数"""
    print("🔄 ハードコーディング除去・設定ファイルベース移行開始...")
    
    # インポート文移行
    migrate_imports()
    
    # 互換性ラッパー作成
    create_wrapper_script()
    
    print("✅ 移行完了")
    print("📋 次の手順:")
    print("1. 必要に応じて config/author_config.yaml を編集")
    print("2. 必要に応じて config/dashboard_merger_config.yaml を編集") 
    print("3. 新しいAuthorParameterAdapterV2を使用したテスト実行")
    print("4. 問題なければ古いファイルを削除")

if __name__ == "__main__":
    main()