#!/usr/bin/env python3
"""
インタラクティブパス入力システム (QUAL-033)

コマンドライン環境でのユーザーフレンドリーなパス入力支援システム。
厳密検証と組み合わせて、確実なパス指定を実現。

Created for: QUAL-033 - 厳密パス検証システム実装・全ワークフロー適用・意図しない挙動防止
Author: Claude Code Integration System
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class InputType(Enum):
    """入力タイプ"""
    INPUT_DIRECTORY = "input_directory"
    OUTPUT_DIRECTORY = "output_directory"
    TRACKER_ID = "tracker_id"
    AUTHOR_NAME = "author_name"
    WORK_NAME = "work_name"


@dataclass
class InputPrompt:
    """入力プロンプト設定"""
    prompt_text: str
    input_type: InputType
    validation_func: Optional[Callable] = None
    suggestions: List[str] = None
    required: bool = True
    default_value: Optional[str] = None
    help_text: Optional[str] = None


@dataclass
class InteractiveResult:
    """対話結果"""
    success: bool
    value: Optional[str]
    path: Optional[Path] = None
    cancelled: bool = False
    error_message: Optional[str] = None


class InteractivePathInput:
    """
    インタラクティブパス入力システム
    
    - ユーザーフレンドリーなプロンプト表示
    - 入力候補の自動提案
    - リアルタイム検証フィードバック
    - ヘルプ・キャンセル機能
    """
    
    def __init__(self, 
                 use_colors: bool = True,
                 max_attempts: int = 3,
                 auto_suggest: bool = True):
        """
        初期化
        
        Args:
            use_colors: カラー出力使用
            max_attempts: 最大試行回数
            auto_suggest: 自動候補提案
        """
        self.use_colors = use_colors
        self.max_attempts = max_attempts
        self.auto_suggest = auto_suggest
        
        # カラーコード設定
        self.colors = {
            'reset': '\033[0m',
            'bold': '\033[1m',
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'purple': '\033[95m',
            'cyan': '\033[96m'
        } if use_colors else {key: '' for key in ['reset', 'bold', 'red', 'green', 'yellow', 'blue', 'purple', 'cyan']}
    
    def _colorize(self, text: str, color: str) -> str:
        """テキストの色付け"""
        return f"{self.colors.get(color, '')}{text}{self.colors['reset']}"
    
    def _print_header(self, title: str):
        """ヘッダー表示"""
        border = "=" * len(title)
        print()
        print(self._colorize(border, 'blue'))
        print(self._colorize(title, 'bold'))
        print(self._colorize(border, 'blue'))
    
    def _print_help(self, input_prompt: InputPrompt):
        """ヘルプ表示"""
        print()
        print(self._colorize("📖 ヘルプ情報", 'cyan'))
        print(f"   入力タイプ: {input_prompt.input_type.value}")
        if input_prompt.help_text:
            print(f"   説明: {input_prompt.help_text}")
        
        # 入力タイプ別のヘルプ
        if input_prompt.input_type == InputType.INPUT_DIRECTORY:
            print("   💡 画像ファイル（jpg, png等）が含まれるディレクトリを指定してください")
            print("   例: /mnt/c/AItools/lora/train/yado/org/kana05/")
        elif input_prompt.input_type == InputType.OUTPUT_DIRECTORY:
            print("   💡 抽出結果を保存するディレクトリを指定してください")
            print("   例: /mnt/c/AItools/lora/train/yado/tracker-workspace/QUAL-033/extraction/")
        elif input_prompt.input_type == InputType.TRACKER_ID:
            print("   💡 トラッカーIDを入力してください（例: QUAL-033, EXTR-001）")
        elif input_prompt.input_type == InputType.AUTHOR_NAME:
            print("   💡 作者名を入力してください（例: yado, kiri, zundamon）")
        elif input_prompt.input_type == InputType.WORK_NAME:
            print("   💡 作品名を入力してください（例: kana05, kana08）")
        
        print("   📝 特殊コマンド:")
        print("      help または ? : このヘルプを表示")
        print("      cancel または quit : 処理をキャンセル")
        print("      suggest または tab : 候補を表示")
        print()
    
    def _get_path_suggestions(self, input_type: InputType, partial_input: str = "") -> List[str]:
        """パス候補の自動生成"""
        suggestions = []
        
        if input_type == InputType.INPUT_DIRECTORY:
            # 既知の入力ディレクトリ候補
            base_paths = [
                "/mnt/c/AItools/lora/train/yado/org/",
                "/mnt/c/AItools/lora/train/kiri/org/",
                "/mnt/c/AItools/lora/train/zundamon/org/"
            ]
            
            for base_path in base_paths:
                if Path(base_path).exists():
                    try:
                        for item in Path(base_path).iterdir():
                            if item.is_dir() and (not partial_input or partial_input.lower() in item.name.lower()):
                                suggestions.append(str(item))
                    except PermissionError:
                        continue
        
        elif input_type == InputType.OUTPUT_DIRECTORY:
            # ワークスペースベースの候補
            workspace_base = Path("/mnt/c/AItools/lora/train/yado/tracker-workspace/")
            if workspace_base.exists():
                try:
                    for item in workspace_base.iterdir():
                        if item.is_dir() and (not partial_input or partial_input.lower() in item.name.lower()):
                            suggestions.append(str(item / "extraction"))
                except PermissionError:
                    pass
        
        elif input_type == InputType.AUTHOR_NAME:
            suggestions = ["yado", "kiri", "zundamon"]
            if partial_input:
                suggestions = [s for s in suggestions if partial_input.lower() in s.lower()]
        
        elif input_type == InputType.WORK_NAME:
            # 作品名の候補（動的に生成）
            common_works = ["kana03", "kana04", "kana05", "kana06", "kana07", "kana08", "kana09"]
            if partial_input:
                suggestions = [w for w in common_works if partial_input.lower() in w.lower()]
            else:
                suggestions = common_works
        
        return suggestions[:10]  # 最大10候補
    
    def _show_suggestions(self, input_prompt: InputPrompt, partial_input: str = ""):
        """候補の表示"""
        suggestions = []
        
        # プロンプトで指定された候補
        if input_prompt.suggestions:
            suggestions.extend(input_prompt.suggestions)
        
        # 自動生成候補
        if self.auto_suggest:
            auto_suggestions = self._get_path_suggestions(input_prompt.input_type, partial_input)
            suggestions.extend(auto_suggestions)
        
        if suggestions:
            print()
            print(self._colorize("💡 候補:", 'yellow'))
            for i, suggestion in enumerate(suggestions[:10], 1):
                # パスの存在チェック
                status = ""
                if input_prompt.input_type in [InputType.INPUT_DIRECTORY, InputType.OUTPUT_DIRECTORY]:
                    path = Path(suggestion)
                    if path.exists():
                        status = self._colorize(" ✓", 'green')
                    else:
                        status = self._colorize(" ✗", 'red')
                
                print(f"   {i:2}. {suggestion}{status}")
            print()
    
    def _validate_input(self, value: str, input_prompt: InputPrompt) -> Optional[str]:
        """入力値の検証"""
        if not value.strip():
            if input_prompt.required:
                return "入力が必要です"
            return None
        
        # カスタム検証関数
        if input_prompt.validation_func:
            try:
                error = input_prompt.validation_func(value)
                if error:
                    return error
            except Exception as e:
                return f"検証エラー: {e}"
        
        # 入力タイプ別の基本検証
        if input_prompt.input_type in [InputType.INPUT_DIRECTORY, InputType.OUTPUT_DIRECTORY]:
            path = Path(value)
            
            # パス形式チェック
            try:
                path.resolve()
            except Exception:
                return "無効なパス形式です"
            
            # 入力ディレクトリの存在チェック
            if input_prompt.input_type == InputType.INPUT_DIRECTORY:
                if not path.exists():
                    return f"ディレクトリが存在しません: {path}"
                if not path.is_dir():
                    return f"ディレクトリではありません: {path}"
        
        elif input_prompt.input_type == InputType.TRACKER_ID:
            # トラッカーID形式チェック
            import re
            if not re.match(r'^[A-Z]+-\d+$', value.upper()):
                return "トラッカーIDは 'PREFIX-NUMBER' 形式で入力してください（例: QUAL-033）"
        
        return None
    
    def prompt_for_input(self, input_prompt: InputPrompt) -> InteractiveResult:
        """
        単一入力の対話的プロンプト
        
        Args:
            input_prompt: 入力プロンプト設定
            
        Returns:
            InteractiveResult: 対話結果
        """
        attempt = 0
        
        while attempt < self.max_attempts:
            print()
            print(self._colorize(f"🔍 {input_prompt.prompt_text}", 'bold'))
            
            # デフォルト値の表示
            if input_prompt.default_value and not input_prompt.required:
                print(f"   デフォルト: {input_prompt.default_value}")
                print("   Enterキーでデフォルト使用、または新しい値を入力:")
            else:
                print("   値を入力してください:")
            
            # 候補の表示（最初の試行時のみ）
            if attempt == 0:
                self._show_suggestions(input_prompt)
            
            try:
                user_input = input(self._colorize("> ", 'cyan')).strip()
                
                # 特殊コマンドの処理
                if user_input.lower() in ['help', '?']:
                    self._print_help(input_prompt)
                    continue
                
                if user_input.lower() in ['cancel', 'quit']:
                    return InteractiveResult(
                        success=False,
                        value=None,
                        cancelled=True
                    )
                
                if user_input.lower() in ['suggest', 'tab']:
                    self._show_suggestions(input_prompt, user_input)
                    continue
                
                # 空入力の処理
                if not user_input:
                    if input_prompt.default_value:
                        user_input = input_prompt.default_value
                    elif not input_prompt.required:
                        return InteractiveResult(success=True, value=None)
                    else:
                        print(self._colorize("❌ 入力が必要です", 'red'))
                        attempt += 1
                        continue
                
                # 入力検証
                validation_error = self._validate_input(user_input, input_prompt)
                if validation_error:
                    print(self._colorize(f"❌ {validation_error}", 'red'))
                    attempt += 1
                    continue
                
                # 成功
                result_path = None
                if input_prompt.input_type in [InputType.INPUT_DIRECTORY, InputType.OUTPUT_DIRECTORY]:
                    result_path = Path(user_input)
                
                print(self._colorize(f"✅ 入力確認: {user_input}", 'green'))
                
                return InteractiveResult(
                    success=True,
                    value=user_input,
                    path=result_path
                )
                
            except KeyboardInterrupt:
                print()
                print(self._colorize("❌ ユーザーによってキャンセルされました", 'red'))
                return InteractiveResult(
                    success=False,
                    value=None,
                    cancelled=True
                )
            except Exception as e:
                print(self._colorize(f"❌ 入力エラー: {e}", 'red'))
                attempt += 1
        
        return InteractiveResult(
            success=False,
            value=None,
            error_message=f"{self.max_attempts}回の試行に失敗しました"
        )
    
    def prompt_for_paths(self, 
                        title: str = "パス設定",
                        require_input: bool = True,
                        require_output: bool = True,
                        input_context: str = "画像入力ディレクトリ",
                        output_context: str = "抽出結果出力ディレクトリ") -> Dict[str, Any]:
        """
        入力・出力パスの対話的プロンプト
        
        Args:
            title: セクションタイトル
            require_input: 入力パス必須
            require_output: 出力パス必須
            input_context: 入力パスの説明
            output_context: 出力パスの説明
            
        Returns:
            Dict[str, Any]: プロンプト結果
        """
        self._print_header(title)
        
        results = {
            'success': False,
            'input_path': None,
            'output_path': None,
            'cancelled': False,
            'errors': []
        }
        
        # 入力パスのプロンプト
        if require_input:
            input_prompt = InputPrompt(
                prompt_text=f"{input_context}を指定してください",
                input_type=InputType.INPUT_DIRECTORY,
                required=True,
                help_text="画像ファイル（jpg, png等）が含まれるディレクトリ"
            )
            
            input_result = self.prompt_for_input(input_prompt)
            if input_result.cancelled:
                results['cancelled'] = True
                return results
            
            if not input_result.success:
                results['errors'].append(f"入力パス設定失敗: {input_result.error_message}")
                return results
            
            results['input_path'] = input_result.path
        
        # 出力パスのプロンプト
        if require_output:
            output_prompt = InputPrompt(
                prompt_text=f"{output_context}を指定してください",
                input_type=InputType.OUTPUT_DIRECTORY,
                required=True,
                help_text="抽出結果を保存するディレクトリ（存在しない場合は自動作成）"
            )
            
            output_result = self.prompt_for_input(output_prompt)
            if output_result.cancelled:
                results['cancelled'] = True
                return results
            
            if not output_result.success:
                results['errors'].append(f"出力パス設定失敗: {output_result.error_message}")
                return results
            
            results['output_path'] = output_result.path
        
        results['success'] = True
        
        # 確認表示
        print()
        print(self._colorize("📋 設定確認:", 'green'))
        if results['input_path']:
            print(f"   📥 入力: {results['input_path']}")
        if results['output_path']:
            print(f"   📤 出力: {results['output_path']}")
        
        return results
    
    def prompt_for_tracker_info(self, title: str = "トラッカー情報設定") -> Dict[str, Any]:
        """
        トラッカー情報の対話的プロンプト
        
        Args:
            title: セクションタイトル
            
        Returns:
            Dict[str, Any]: トラッカー情報
        """
        self._print_header(title)
        
        results = {
            'success': False,
            'tracker_id': None,
            'author': None,
            'work': None,
            'cancelled': False,
            'errors': []
        }
        
        # トラッカーIDプロンプト
        tracker_prompt = InputPrompt(
            prompt_text="トラッカーIDを入力してください",
            input_type=InputType.TRACKER_ID,
            required=True,
            help_text="例: QUAL-033, EXTR-001, INTG-002"
        )
        
        tracker_result = self.prompt_for_input(tracker_prompt)
        if tracker_result.cancelled:
            results['cancelled'] = True
            return results
        
        if not tracker_result.success:
            results['errors'].append(f"トラッカーID設定失敗: {tracker_result.error_message}")
            return results
        
        results['tracker_id'] = tracker_result.value
        
        # 作者名プロンプト（オプション）
        author_prompt = InputPrompt(
            prompt_text="作者名を入力してください（オプション）",
            input_type=InputType.AUTHOR_NAME,
            required=False,
            help_text="既知作者: yado, kiri, zundamon"
        )
        
        author_result = self.prompt_for_input(author_prompt)
        if author_result.cancelled:
            results['cancelled'] = True
            return results
        
        if author_result.success and author_result.value:
            results['author'] = author_result.value
        
        # 作品名プロンプト（オプション）
        work_prompt = InputPrompt(
            prompt_text="作品名を入力してください（オプション）",
            input_type=InputType.WORK_NAME,
            required=False,
            help_text="例: kana05, kana08"
        )
        
        work_result = self.prompt_for_input(work_prompt)
        if work_result.cancelled:
            results['cancelled'] = True
            return results
        
        if work_result.success and work_result.value:
            results['work'] = work_result.value
        
        results['success'] = True
        
        # 確認表示
        print()
        print(self._colorize("📋 トラッカー情報確認:", 'green'))
        print(f"   🎯 ID: {results['tracker_id']}")
        if results['author']:
            print(f"   👤 作者: {results['author']}")
        if results['work']:
            print(f"   📚 作品: {results['work']}")
        
        return results


# 便利な関数
def interactive_setup(title: str = "QUAL-033 厳密パス検証システム") -> Dict[str, Any]:
    """
    完全な対話的セットアップ
    
    Args:
        title: セットアップタイトル
        
    Returns:
        Dict[str, Any]: セットアップ結果
    """
    interactive = InteractivePathInput(use_colors=True, auto_suggest=True)
    
    print()
    print("🚀 " + "=" * 60)
    print(f"🚀 {title}")
    print("🚀 " + "=" * 60)
    print()
    print("💡 このシステムは厳密なパス検証により、意図しない動作を防ぎます")
    print("💡 help または ? でヘルプ、cancel または quit でキャンセル")
    
    results = {
        'success': False,
        'paths': {},
        'tracker_info': {},
        'cancelled': False,
        'errors': []
    }
    
    # トラッカー情報設定
    tracker_info = interactive.prompt_for_tracker_info()
    if tracker_info['cancelled']:
        results['cancelled'] = True
        return results
    
    if not tracker_info['success']:
        results['errors'].extend(tracker_info['errors'])
        return results
    
    results['tracker_info'] = tracker_info
    
    # パス設定
    path_info = interactive.prompt_for_paths()
    if path_info['cancelled']:
        results['cancelled'] = True
        return results
    
    if not path_info['success']:
        results['errors'].extend(path_info['errors'])
        return results
    
    results['paths'] = path_info
    results['success'] = True
    
    # 最終確認
    print()
    print(interactive._colorize("🎉 セットアップ完了!", 'green'))
    print(f"   トラッカー: {tracker_info['tracker_id']}")
    print(f"   入力パス: {path_info['input_path']}")
    print(f"   出力パス: {path_info['output_path']}")
    
    return results


if __name__ == "__main__":
    # テスト実行
    print("🧪 インタラクティブパス入力システム テスト")
    
    # 基本テスト
    interactive = InteractivePathInput(use_colors=True)
    
    test_prompt = InputPrompt(
        prompt_text="テスト用ディレクトリを指定してください",
        input_type=InputType.INPUT_DIRECTORY,
        help_text="テスト用のヘルプメッセージ"
    )
    
    result = interactive.prompt_for_input(test_prompt)
    print(f"結果: {result}")