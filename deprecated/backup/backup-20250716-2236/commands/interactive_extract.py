#!/usr/bin/env python3
"""
Phase 3: インタラクティブ抽出コマンド
コマンドラインからGUIアプリケーションを起動
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    """メイン関数"""
    try:
        # モデル初期化
        print("🔄 モデル初期化中...")
        from hooks.start import start
        start()
        print("✅ モデル初期化完了")
        
        # GUIアプリケーション起動
        print("🚀 インタラクティブGUI起動中...")
        from utils.interactive_assistant import InteractiveGUI
        
        app = InteractiveGUI()
        app.run()
        
    except ImportError as e:
        print(f"❌ インポートエラー: {e}")
        print("必要なライブラリがインストールされていない可能性があります:")
        print("pip install tkinter pillow opencv-python")
    except Exception as e:
        print(f"❌ エラー: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())