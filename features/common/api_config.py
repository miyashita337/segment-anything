#!/usr/bin/env python3
"""
API設定管理モジュール
"""
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class APIConfig:
    """API設定の統一管理クラス"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初期化
        Args:
            config_path: 設定ファイルパス（省略時は自動検出）
        """
        if config_path is None:
            # プロジェクトルートからの相対パスで設定ファイルを探す
            project_root = self._find_project_root()
            config_path = project_root / "config" / "api_keys.json"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _find_project_root(self) -> Path:
        """プロジェクトルートディレクトリを探す"""
        current = Path(__file__).parent
        
        # 最大5階層まで遡って探す
        for _ in range(5):
            if (current / "config").exists() or (current / "spec.md").exists():
                return current
            current = current.parent
        
        # 見つからない場合は現在のディレクトリから相対パス
        return Path(__file__).parent.parent.parent
    
    def _load_config(self) -> Dict[str, Any]:
        """設定ファイルを読み込む"""
        try:
            if not self.config_path.exists():
                logger.warning(f"設定ファイルが見つかりません: {self.config_path}")
                return self._get_default_config()
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            logger.info(f"設定ファイル読み込み完了: {self.config_path}")
            return config
            
        except Exception as e:
            logger.error(f"設定ファイル読み込み失敗: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定を返す"""
        return {
            "gemini": {
                "primary_key": "",
                "fallback_key": "",
                "model": "gemini-pro",
                "timeout": 30,
                "max_retries": 3
            },
            "openai": {
                "api_key": "",
                "model": "gpt-4o",
                "timeout": 30,
                "max_retries": 3
            },
            "claude": {
                "api_key": "",
                "model": "claude-3-sonnet-20240229",
                "timeout": 30,
                "max_retries": 3
            }
        }
    
    def get_gemini_api_key(self, use_fallback: bool = False) -> Optional[str]:
        """
        Gemini APIキーを取得
        Args:
            use_fallback: Trueの場合、フォールバックキーを使用
        Returns:
            APIキー文字列、または環境変数から取得
        """
        gemini_config = self.config.get("gemini", {})
        
        if use_fallback:
            key = gemini_config.get("fallback_key")
            if key:
                return key
        
        # プライマリキーを試す
        key = gemini_config.get("primary_key")
        if key:
            return key
        
        # 環境変数から取得を試す
        env_key = os.getenv('GEMINI_API_KEY')
        if env_key:
            logger.info("環境変数からGemini APIキーを取得")
            return env_key
        
        logger.warning("Gemini APIキーが見つかりません")
        return None
    
    def get_openai_api_key(self) -> Optional[str]:
        """OpenAI APIキーを取得"""
        openai_config = self.config.get("openai", {})
        
        # 設定ファイルから取得
        key = openai_config.get("api_key")
        if key:
            return key
        
        # 環境変数から取得
        env_key = os.getenv('OPENAI_API_KEY')
        if env_key:
            logger.info("環境変数からOpenAI APIキーを取得")
            return env_key
        
        logger.warning("OpenAI APIキーが見つかりません")
        return None
    
    def get_claude_api_key(self) -> Optional[str]:
        """Claude APIキーを取得"""
        claude_config = self.config.get("claude", {})
        
        # 設定ファイルから取得
        key = claude_config.get("api_key")
        if key:
            return key
        
        # 環境変数から取得
        env_key = os.getenv('CLAUDE_API_KEY')
        if env_key:
            logger.info("環境変数からClaude APIキーを取得")
            return env_key
        
        logger.warning("Claude APIキーが見つかりません")
        return None
    
    def get_api_config(self, service: str) -> Dict[str, Any]:
        """
        指定されたサービスの設定を取得
        Args:
            service: サービス名 (gemini, openai, claude)
        Returns:
            サービス設定辞書
        """
        return self.config.get(service, {})
    
    def update_api_key(self, service: str, key_type: str, api_key: str) -> bool:
        """
        APIキーを更新して設定ファイルに保存
        Args:
            service: サービス名 (gemini, openai, claude)
            key_type: キータイプ (primary_key, fallback_key, api_key)
            api_key: 新しいAPIキー
        Returns:
            更新成功可否
        """
        try:
            if service not in self.config:
                self.config[service] = {}
            
            self.config[service][key_type] = api_key
            
            # 設定ファイルに保存
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"{service}の{key_type}を更新しました")
            return True
            
        except Exception as e:
            logger.error(f"API設定更新失敗: {e}")
            return False
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """
        全APIキーの有効性をチェック
        Returns:
            サービス別の有効性辞書
        """
        validation_result = {}
        
        # Gemini
        gemini_key = self.get_gemini_api_key()
        validation_result['gemini'] = bool(gemini_key and len(gemini_key) > 30)
        
        # OpenAI
        openai_key = self.get_openai_api_key()
        validation_result['openai'] = bool(openai_key and openai_key.startswith('sk-'))
        
        # Claude
        claude_key = self.get_claude_api_key()
        validation_result['claude'] = bool(claude_key and len(claude_key) > 30)
        
        return validation_result


# グローバルインスタンス（シングルトン的に使用）
_api_config_instance = None

def get_api_config() -> APIConfig:
    """APIConfig のグローバルインスタンスを取得"""
    global _api_config_instance
    if _api_config_instance is None:
        _api_config_instance = APIConfig()
    return _api_config_instance

# 便利関数
def get_gemini_api_key(use_fallback: bool = False) -> Optional[str]:
    """Gemini APIキーを取得する便利関数"""
    return get_api_config().get_gemini_api_key(use_fallback)

def get_openai_api_key() -> Optional[str]:
    """OpenAI APIキーを取得する便利関数"""
    return get_api_config().get_openai_api_key()

def get_claude_api_key() -> Optional[str]:
    """Claude APIキーを取得する便利関数"""
    return get_api_config().get_claude_api_key()


if __name__ == "__main__":
    # テスト実行
    config = APIConfig()
    
    print("=== API設定テスト ===")
    print(f"Gemini APIキー: {config.get_gemini_api_key()[:20] if config.get_gemini_api_key() else 'なし'}...")
    print(f"OpenAI APIキー: {config.get_openai_api_key()[:20] if config.get_openai_api_key() else 'なし'}...")
    print(f"Claude APIキー: {config.get_claude_api_key()[:20] if config.get_claude_api_key() else 'なし'}...")
    
    print("\n=== バリデーション結果 ===")
    validation = config.validate_api_keys()
    for service, is_valid in validation.items():
        status = "✅ 有効" if is_valid else "❌ 無効"
        print(f"{service}: {status}")