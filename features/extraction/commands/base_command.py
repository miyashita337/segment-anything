"""Base command pattern for extraction operations.

Provides abstract base class for all extraction commands.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ExtractionConfig:
    """Configuration for extraction operations."""
    
    input_path: str
    output_path: str
    batch: bool = False
    verbose: bool = False
    no_notify: bool = False
    no_images: bool = False
    max_files: Optional[int] = None
    resume: bool = False
    sam_optimization_profile: str = 'p1_020_optimized'
    enable_dashboard: bool = False
    dashboard_port: int = 8080
    enable_backup: bool = False
    backup_retention_days: int = 7
    enable_quality_monitoring: bool = True
    quality_threshold: float = 0.7


class BaseExtractionCommand(ABC):
    """Abstract base class for extraction commands."""
    
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        import logging
        level = logging.DEBUG if self.config.verbose else logging.INFO
        logging.basicConfig(level=level)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        """Execute the extraction command.
        
        Returns:
            Dict containing execution results
        """
        pass
    
    def validate_config(self) -> bool:
        """Validate configuration parameters.
        
        Returns:
            True if configuration is valid
        """
        # 🚨 入力ディレクトリ存在チェック必須要件
        input_path = Path(self.config.input_path)
        
        # 入力パス未指定チェック
        if not self.config.input_path or self.config.input_path.strip() == "":
            error_msg = (
                "❌ エラー: 入力パスが指定されていません\n"
                "\n"
                "🔧 対処方法:\n"
                "   1. 入力パスを明示的に指定してください\n"
                "   2. 例: python extract_character.py /path/to/input -o /path/to/output\n"
                "\n"
                "⚠️ 注意: 入力パス未指定での実行は厳禁です"
            )
            self.logger.error(error_msg)
            return False
        
        # 入力パス存在チェック
        if not input_path.exists():
            error_msg = (
                f"❌ エラー: 入力ディレクトリが存在しません\n"
                f"   パス: {input_path}\n"
                f"\n"
                f"🔧 対処方法:\n"
                f"   1. パスの確認: ls {input_path.parent}\n"
                f"   2. 正しいパスの指定\n"
                f"   3. 必要に応じてディレクトリ作成\n"
                f"\n"
                f"⚠️ 注意: 存在しないパスでの強制実行は品質保証違反です"
            )
            self.logger.error(error_msg)
            return False
        
        # バッチモード固有チェック
        if self.config.batch and not input_path.is_dir():
            error_msg = (
                f"❌ エラー: バッチモードにはディレクトリが必要です\n"
                f"   指定されたパス: {input_path}\n"
                f"   パスタイプ: {'ファイル' if input_path.is_file() else '不明'}\n"
                f"\n"
                f"🔧 対処方法:\n"
                f"   1. ディレクトリパスを指定してください\n"
                f"   2. または --batch フラグを削除してください\n"
                f"\n"
                f"⚠️ 注意: バッチモードは画像ディレクトリでのみ動作します"
            )
            self.logger.error(error_msg)
            return False
        
        # シングルモード固有チェック  
        if not self.config.batch and not input_path.is_file():
            error_msg = (
                f"❌ エラー: シングルモードには画像ファイルが必要です\n"
                f"   指定されたパス: {input_path}\n"
                f"   パスタイプ: {'ディレクトリ' if input_path.is_dir() else '不明'}\n"
                f"\n"
                f"🔧 対処方法:\n"
                f"   1. 画像ファイルパスを指定してください\n"
                f"   2. またはディレクトリ処理には --batch フラグを追加してください\n"
                f"\n"
                f"⚠️ 注意: シングルモードは単一画像ファイルでのみ動作します"
            )
            self.logger.error(error_msg)
            return False
        
        # バッチモード時の空ディレクトリチェック
        if self.config.batch and input_path.is_dir():
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            image_files = []
            for ext in image_extensions:
                image_files.extend(input_path.glob(ext))
            
            if not image_files:
                error_msg = (
                    f"❌ エラー: 指定されたディレクトリに画像ファイルがありません\n"
                    f"   ディレクトリ: {input_path}\n"
                    f"   対応形式: {', '.join(image_extensions)}\n"
                    f"\n"
                    f"🔧 対処方法:\n"
                    f"   1. ディレクトリ内容確認: ls {input_path}\n"
                    f"   2. 対応形式の画像ファイルを配置\n"
                    f"   3. 正しいディレクトリパスを指定\n"
                    f"\n"
                    f"⚠️ 注意: 空ディレクトリでの処理実行は禁止されています"
                )
                self.logger.error(error_msg)
                return False
        
        # 出力パスの親ディレクトリ作成可能性チェック
        output_path = Path(self.config.output_path)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            error_msg = (
                f"❌ エラー: 出力ディレクトリの作成に失敗しました\n"
                f"   出力パス: {output_path}\n"
                f"   エラー: {str(e)}\n"
                f"\n"
                f"🔧 対処方法:\n"
                f"   1. 書き込み権限の確認\n"
                f"   2. ディスク容量の確認\n"
                f"   3. パス文字列の確認\n"
                f"\n"
                f"⚠️ 注意: 出力先確保は処理実行の前提条件です"
            )
            self.logger.error(error_msg)
            return False
        
        self.logger.info(f"✅ 設定検証完了: 入力={input_path}, 出力={output_path}")
        return True