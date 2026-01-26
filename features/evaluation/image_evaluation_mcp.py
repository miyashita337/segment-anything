#!/usr/bin/env python3
"""
GPT-4O + Gemini フォールバック画像評価システム
MCPサーバー実装

画像抽出結果の品質を自動評価する統合システム
- 第一選択: GPT-4O
- フォールバック: Gemini Pro Vision
- 制限到達時: エラー終了
"""

import base64
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# 必要に応じてインストール
try:
    import openai
    from openai import OpenAI
except ImportError:
    print("OpenAI library not found. Please install with: pip install openai")
    sys.exit(1)

try:
    import google.generativeai as genai
except ImportError:
    print(
        "Google Generative AI library not found. Please install with: pip install google-generativeai"
    )
    sys.exit(1)

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class ImageEvaluationMCP:
    """GPT-4O + Gemini フォールバック画像評価システム"""

    def __init__(self, openai_api_key: Optional[str] = None, gemini_api_key: Optional[str] = None):
        """
        初期化

        Args:
            openai_api_key: OpenAI API キー
            gemini_api_key: Gemini API キー
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")

        # OpenAI クライアント初期化
        if self.openai_api_key:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
        else:
            self.openai_client = None
            print("⚠️ OpenAI API key not found. GPT-4O will be unavailable.")

        # Gemini クライアント初期化
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel("gemini-pro-vision")
        else:
            self.gemini_model = None
            print("⚠️ Gemini API key not found. Gemini Pro Vision will be unavailable.")

        # 評価プロンプトテンプレート（人間評価フォーマットに統一）
        self.evaluation_prompt = """
あなたは画像抽出品質の専門家です。
この画像はアニメキャラクターの自動抽出結果です。元画像から対象キャラクターが適切に抽出されているかを評価してください。

**評価基準:**
- **A**: 優秀な抽出（ファインチューニング使用可能）
- **B**: 良好な抽出（軽微な問題あり）
- **C**: 普通の抽出（中程度の問題あり）
- **D**: 問題のある抽出（重大な問題あり）
- **E**: 悪い抽出（複数の重大な問題）
- **F**: 失敗した抽出（使用不可能）

**基本A以外はどこかしら抽出に失敗しています。**

**問題分類（該当する場合のみ）:**
- `抽出範囲不適切`: 背景やマスク、吹き出しなどキャラクター以外を抽出
- `境界不正確`: 境界線がぼやけている、ギザギザしている
- `顔部分欠損`: キャラクターの顔が切れている、欠けている
- `手足切断`: 手足が適切に抽出されていない、切断されている
- `他キャラクター混入`: 対象外の他キャラクターが含まれている
- `両足が抽出できてない`: 足部分の抽出が失敗している

**判定のポイント:**
1. キャラクターが主体として適切に抽出されているか
2. 手足、顔などの重要部位が欠損していないか
3. 背景や無関係な要素が混入していないか
4. 境界線が自然で正確か

以下のJSON形式で回答してください：
{
    "grade": "<A-Fの評価>",
    "issues": [<該当する問題分類のリスト>],
    "comments": "<具体的な評価理由と改善点>"
}
"""

    def encode_image_to_base64(self, image_path: str) -> str:
        """
        画像をBase64エンコード

        Args:
            image_path: 画像ファイルパス

        Returns:
            Base64エンコードされた画像データ
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            raise Exception(f"画像エンコードに失敗: {str(e)}")

    def evaluate_with_gpt4o(self, image_path: str) -> Dict[str, Any]:
        """
        GPT-4Oで画像評価

        Args:
            image_path: 画像ファイルパス

        Returns:
            評価結果辞書
        """
        if not self.openai_client:
            raise Exception("OpenAI client not initialized")

        try:
            # 画像をBase64エンコード
            base64_image = self.encode_image_to_base64(image_path)

            # GPT-4O API呼び出し
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.evaluation_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },
                        ],
                    }
                ],
                max_tokens=500,
                temperature=0.1,
            )

            # レスポンス解析
            result_text = response.choices[0].message.content

            # JSON解析を試行
            try:
                # コードブロック内のJSONを抽出
                import re

                json_match = re.search(r"```json\n(.*?)\n```", result_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    result_json = json.loads(json_str)
                    result_json["api_used"] = "gpt-4o"
                    return result_json
                else:
                    # コードブロックがない場合は直接パース
                    result_json = json.loads(result_text)
                    result_json["api_used"] = "gpt-4o"
                    return result_json
            except json.JSONDecodeError:
                # JSON解析に失敗した場合はテキストのまま返す
                return {"raw_response": result_text, "api_used": "gpt-4o", "parse_error": True}

        except openai.RateLimitError as e:
            print(f"🚨 GPT-4O Rate limit reached: {str(e)}")
            raise
        except Exception as e:
            print(f"❌ GPT-4O評価エラー: {str(e)}")
            raise

    def evaluate_with_gemini(self, image_path: str) -> Dict[str, Any]:
        """
        Gemini Pro Visionで画像評価

        Args:
            image_path: 画像ファイルパス

        Returns:
            評価結果辞書
        """
        if not self.gemini_model:
            raise Exception("Gemini model not initialized")

        try:
            # 画像を読み込み
            import PIL.Image

            image = PIL.Image.open(image_path)

            # Gemini API呼び出し
            response = self.gemini_model.generate_content([self.evaluation_prompt, image])

            result_text = response.text

            # JSON解析を試行
            try:
                result_json = json.loads(result_text)
                result_json["api_used"] = "gemini-pro-vision"
                return result_json
            except json.JSONDecodeError:
                # JSON解析に失敗した場合はテキストのまま返す
                return {
                    "raw_response": result_text,
                    "api_used": "gemini-pro-vision",
                    "parse_error": True,
                }

        except Exception as e:
            print(f"❌ Gemini評価エラー: {str(e)}")
            raise

    def evaluate_image(self, image_path: str) -> Dict[str, Any]:
        """
        画像評価（フォールバック機能付き）

        Args:
            image_path: 画像ファイルパス

        Returns:
            評価結果辞書
        """
        print(f"🔍 画像評価開始: {Path(image_path).name}")

        # 画像ファイルの存在確認
        if not Path(image_path).exists():
            raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")

        # 第一選択: GPT-4O
        if self.openai_client:
            try:
                print("📊 GPT-4Oで評価中...")
                result = self.evaluate_with_gpt4o(image_path)
                print(f"✅ GPT-4O評価完了")
                return result
            except openai.RateLimitError:
                print("🚨 GPT-4O制限到達 → Geminiにフォールバック")
            except Exception as e:
                print(f"❌ GPT-4O評価失敗: {str(e)}")

        # フォールバック: Gemini Pro Vision
        if self.gemini_model:
            try:
                print("📊 Gemini Pro Visionで評価中...")
                result = self.evaluate_with_gemini(image_path)
                print(f"✅ Gemini評価完了")
                return result
            except Exception as e:
                print(f"❌ Gemini評価失敗: {str(e)}")
                raise Exception(f"全てのAPI評価が失敗しました: {str(e)}")

        # 両方のAPIが利用できない場合
        raise Exception("利用可能なAPIがありません。APIキーを確認してください。")

    def batch_evaluate(self, image_list: List[str], output_path: str) -> Dict[str, Any]:
        """
        バッチ評価

        Args:
            image_list: 評価対象画像のリスト
            output_path: 結果出力ファイルパス

        Returns:
            バッチ評価結果サマリー
        """
        print(f"🚀 バッチ評価開始: {len(image_list)}枚の画像")

        results = []
        success_count = 0
        error_count = 0
        start_time = time.time()

        for i, image_path in enumerate(image_list, 1):
            print(f"\n📊 [{i}/{len(image_list)}] 評価中: {Path(image_path).name}")

            try:
                evaluation_result = self.evaluate_image(image_path)
                evaluation_result.update(
                    {
                        "image_path": image_path,
                        "timestamp": datetime.now().isoformat(),
                        "success": True,
                    }
                )
                results.append(evaluation_result)
                success_count += 1

                # 成功時の表示
                if not evaluation_result.get("parse_error", False):
                    grade = evaluation_result.get("grade", "N/A")
                    api_used = evaluation_result.get("api_used", "Unknown")
                    print(f"✅ 成功: 評価={grade}, API={api_used}")
                else:
                    print(f"⚠️ 成功（JSON解析エラー）: API={evaluation_result.get('api_used', 'Unknown')}")

            except Exception as e:
                error_result = {
                    "image_path": image_path,
                    "timestamp": datetime.now().isoformat(),
                    "success": False,
                    "error": str(e),
                }
                results.append(error_result)
                error_count += 1
                print(f"❌ 失敗: {str(e)}")

        # 結果サマリー
        total_time = time.time() - start_time
        success_rate = (success_count / len(image_list)) * 100

        summary = {
            "batch_info": {
                "total_images": len(image_list),
                "mcp_success_count": success_count,
                "mcp_error_count": error_count,
                "mcp_success_rate": success_rate,
                "total_time": total_time,
                "timestamp": datetime.now().isoformat(),
            },
            "results": results,
        }

        # 結果をファイルに保存
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # サマリー表示
        print(f"\n{'='*60}")
        print(f"📊 バッチ評価完了")
        print(f"{'='*60}")
        print(f"処理画像数: {len(image_list)}枚")
        print(f"MCP評価成功: {success_count}枚")
        print(f"MCP評価失敗: {error_count}枚")
        print(f"MCP評価成功率: {success_rate:.1f}%")
        print(f"総処理時間: {total_time:.1f}秒")
        print(f"結果保存先: {output_path}")

        return summary


def main():
    """メイン実行関数"""
    import argparse

    parser = argparse.ArgumentParser(description="画像評価システム")
    parser.add_argument("--image", type=str, help="単一画像の評価")
    parser.add_argument("--batch", type=str, help="バッチ評価（画像ディレクトリ）")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="結果出力ファイル")
    parser.add_argument("--openai-key", type=str, help="OpenAI API キー")
    parser.add_argument("--gemini-key", type=str, help="Gemini API キー")

    args = parser.parse_args()

    # 評価システム初期化
    evaluator = ImageEvaluationMCP(openai_api_key=args.openai_key, gemini_api_key=args.gemini_key)

    try:
        if args.image:
            # 単一画像評価
            result = evaluator.evaluate_image(args.image)
            print(f"\n📊 評価結果:")
            print(json.dumps(result, ensure_ascii=False, indent=2))

        elif args.batch:
            # バッチ評価
            batch_dir = Path(args.batch)
            if not batch_dir.exists():
                print(f"❌ ディレクトリが存在しません: {batch_dir}")
                return

            # 画像ファイルを取得（最終出力画像のみ）
            image_files = []
            for pattern in ["*.jpg", "*.jpeg", "*.png"]:
                for img_path in batch_dir.glob(pattern):
                    # マスクファイルや透明背景版は除外
                    if "_mask" not in img_path.name and "_transparent" not in img_path.name:
                        image_files.append(str(img_path))

            if not image_files:
                print(f"❌ 評価対象画像が見つかりません: {batch_dir}")
                return

            # バッチ評価実行
            summary = evaluator.batch_evaluate(image_files, args.output)

        else:
            print("❌ --image または --batch オプションを指定してください")

    except Exception as e:
        print(f"❌ 実行エラー: {str(e)}")
        print(f"スタックトレース: {traceback.format_exc()}")


if __name__ == "__main__":
    main()
