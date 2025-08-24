"""
入力検証共通モジュール
統一された入力ディレクトリ存在チェック機能
"""

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)


class InputValidationError(Exception):
    """入力検証エラー"""
    pass


def validate_input_directory(input_path: Union[str, Path], 
                           description: str = "入力ディレクトリ",
                           check_images: bool = False,
                           supported_extensions: Optional[List[str]] = None) -> Path:
    """
    入力ディレクトリの存在チェック（統一仕様）
    
    Args:
        input_path: チェック対象のパス
        description: エラーメッセージ用の説明
        check_images: 画像ファイル存在チェックも行うか
        supported_extensions: サポートする拡張子リスト
    
    Returns:
        Path: 検証済みのPathオブジェクト
        
    Raises:
        InputValidationError: 検証失敗時
    """
    if supported_extensions is None:
        supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    path = Path(input_path)
    
    # 基本存在チェック
    if not path.exists():
        parent_path = path.parent
        parent_exists_msg = f"親ディレクトリ存在: {parent_path.exists()}" if parent_path != path else "N/A"
        
        error_msg = f"""❌ エラー: {description}が存在しません
   パス: {path.absolute()}
   {parent_exists_msg}
   
🔧 対処方法:
   1. パスの確認: ls {parent_path if parent_path != path else 'ディレクトリを確認してください'}
   2. 正しいパスの指定
   3. 必要に応じてディレクトリ作成
   
⚠️ 注意: 存在しないパスでの強制実行は品質保証違反です"""
        
        logger.error(error_msg)
        raise InputValidationError(error_msg)
    
    # ディレクトリかどうかのチェック
    if not path.is_dir():
        error_msg = f"""❌ エラー: {description}がディレクトリではありません
   パス: {path.absolute()}
   種類: {'ファイル' if path.is_file() else '不明'}
   
🔧 対処方法:
   1. 正しいディレクトリパスを指定してください
   2. ファイルパスではなく、ディレクトリパスが必要です"""
        
        logger.error(error_msg)
        raise InputValidationError(error_msg)
    
    # 画像ファイル存在チェック（オプション）
    if check_images:
        image_files = []
        for ext in supported_extensions:
            image_files.extend(path.glob(f"*{ext}"))
            image_files.extend(path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            error_msg = f"""❌ エラー: {description}に画像ファイルが見つかりません
   パス: {path.absolute()}
   サポート形式: {', '.join(supported_extensions)}
   
🔧 対処方法:
   1. ディレクトリ内容確認: ls {path}
   2. サポートされている画像形式で画像を配置
   3. ファイル名・拡張子の確認"""
            
            logger.error(error_msg)
            raise InputValidationError(error_msg)
        
        logger.info(f"✅ 画像ファイル確認完了: {len(image_files)}枚 in {path}")
    
    logger.info(f"✅ {description}検証完了: {path}")
    return path


def validate_input_file(input_path: Union[str, Path], 
                      description: str = "入力ファイル") -> Path:
    """
    入力ファイルの存在チェック（統一仕様）
    
    Args:
        input_path: チェック対象のパス
        description: エラーメッセージ用の説明
    
    Returns:
        Path: 検証済みのPathオブジェクト
        
    Raises:
        InputValidationError: 検証失敗時
    """
    path = Path(input_path)
    
    if not path.exists():
        parent_path = path.parent
        error_msg = f"""❌ エラー: {description}が存在しません
   パス: {path.absolute()}
   親ディレクトリ存在: {parent_path.exists()}
   
🔧 対処方法:
   1. パスの確認: ls {parent_path}
   2. 正しいファイルパスの指定
   3. ファイル名・拡張子の確認"""
        
        logger.error(error_msg)
        raise InputValidationError(error_msg)
    
    if not path.is_file():
        error_msg = f"""❌ エラー: {description}がファイルではありません
   パス: {path.absolute()}
   種類: {'ディレクトリ' if path.is_dir() else '不明'}
   
🔧 対処方法:
   1. 正しいファイルパスを指定してください
   2. ディレクトリパスではなく、ファイルパスが必要です"""
        
        logger.error(error_msg)
        raise InputValidationError(error_msg)
    
    logger.info(f"✅ {description}検証完了: {path}")
    return path


def validate_output_directory(output_path: Union[str, Path], 
                            description: str = "出力ディレクトリ",
                            create_if_missing: bool = True) -> Path:
    """
    出力ディレクトリの検証・作成
    
    Args:
        output_path: 出力先パス
        description: エラーメッセージ用の説明
        create_if_missing: 存在しない場合に作成するか
    
    Returns:
        Path: 検証済みのPathオブジェクト
        
    Raises:
        InputValidationError: 検証失敗時
    """
    path = Path(output_path)
    
    if path.exists() and not path.is_dir():
        error_msg = f"""❌ エラー: {description}がディレクトリではありません
   パス: {path.absolute()}
   種類: ファイル
   
🔧 対処方法:
   1. 既存ファイルを削除または移動
   2. 別の出力パスを指定"""
        
        logger.error(error_msg)
        raise InputValidationError(error_msg)
    
    if not path.exists() and create_if_missing:
        try:
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 {description}作成完了: {path}")
        except Exception as e:
            error_msg = f"""❌ エラー: {description}の作成に失敗しました
   パス: {path.absolute()}
   エラー: {e}
   
🔧 対処方法:
   1. 親ディレクトリの権限確認
   2. ディスク容量の確認
   3. パスの形式確認"""
            
            logger.error(error_msg)
            raise InputValidationError(error_msg)
    
    logger.info(f"✅ {description}検証完了: {path}")
    return path


def log_validation_summary(input_paths: List[Path], 
                         output_paths: List[Path], 
                         script_name: str = "処理スクリプト"):
    """
    検証結果のサマリーログ出力
    
    Args:
        input_paths: 検証済み入力パスリスト
        output_paths: 検証済み出力パスリスト
        script_name: スクリプト名
    """
    logger.info(f"🔍 {script_name} - 入出力検証サマリー")
    logger.info(f"📥 入力パス ({len(input_paths)}件):")
    for i, path in enumerate(input_paths, 1):
        logger.info(f"   {i}. {path}")
    
    logger.info(f"📤 出力パス ({len(output_paths)}件):")
    for i, path in enumerate(output_paths, 1):
        logger.info(f"   {i}. {path}")
    
    logger.info("✅ 全ての入出力パス検証完了")


# 使いやすいラッパー関数
def check_input_dir_with_images(input_dir: Union[str, Path], 
                               min_images: int = 1) -> Path:
    """
    画像ディレクトリの簡易チェック
    
    Args:
        input_dir: 入力ディレクトリ
        min_images: 最小画像数
    
    Returns:
        Path: 検証済みパス
    """
    path = validate_input_directory(input_dir, "画像入力ディレクトリ", check_images=True)
    
    # 最小画像数チェック
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
        image_files.extend(path.glob(f"*{ext}"))
        image_files.extend(path.glob(f"*{ext.upper()}"))
    
    if len(image_files) < min_images:
        error_msg = f"""❌ エラー: 画像数が不足しています
   パス: {path}
   検出画像数: {len(image_files)}枚
   必要画像数: {min_images}枚以上
   
🔧 対処方法:
   1. 追加の画像ファイルを配置
   2. ファイル形式の確認（jpg, png等）"""
        
        logger.error(error_msg)
        raise InputValidationError(error_msg)
    
    return path


if __name__ == "__main__":
    # テスト実行例
    import tempfile

    # テスト用一時ディレクトリ作成
    with tempfile.TemporaryDirectory() as temp_dir:
        test_path = Path(temp_dir)
        
        # 正常ケース
        try:
            result = validate_input_directory(test_path, "テストディレクトリ")
            print(f"✅ テスト成功: {result}")
        except InputValidationError as e:
            print(f"❌ テスト失敗: {e}")
        
        # 異常ケース（存在しないディレクトリ）
        try:
            fake_path = test_path / "nonexistent"
            result = validate_input_directory(fake_path, "存在しないディレクトリ")
            print(f"❌ テスト失敗: エラーが発生すべきでした")
        except InputValidationError as e:
            print(f"✅ テスト成功: 期待通りエラー発生")