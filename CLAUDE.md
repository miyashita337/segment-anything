# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Claude Codeを日本語で応答してください。

## プロジェクト概要
MetaのSegment Anything Model (SAM)とYOLOを組み合わせたキャラクター抽出パイプライン。
画像からアニメキャラクターを自動検出・抽出する高精度システム。

## 主要機能
- **自動キャラクター検出**: YOLO + SAM の2段階処理
- **品質評価システム**: バランス、信頼度、サイズ、全身、中心優先の複数評価手法
- **バッチ処理**: 大量画像の自動処理とプログレス管理
- **通知システム**: Pushover統合による長時間処理の進捗通知

## 主要コマンド

### 環境セットアップ
```bash
# 仮想環境作成・有効化
python -m venv sam-env
source sam-env/bin/activate  # Linux
sam-env\Scripts\activate     # Windows

# 依存関係インストール
pip install -e .
pip install opencv-python pycocotools matplotlib onnxruntime onnx ultralytics easyocr
```

### キャラクター抽出実行
```bash
# インタラクティブバッチ処理
python sam_batch_interactive.py

# メインパイプライン実行（kaname03系列）
python extract_kaname03.py

# コマンドライン個別実行
python scripts/amg.py --checkpoint sam_vit_h_4b8939.pth --model-type vit_h --input <image_path> --output <output_path>
```

### テスト・開発
```bash
# Phase3 CLI テスト
python test_phase3_cli.py

# 困難姿勢テスト
python test_difficult_pose.py

# 開発用小規模テスト
python test_phase2_simple.py

# レジューム機能テスト
python test_resume_functionality.py

# コード品質チェック（flake8, black, mypy, isort）
./linter.sh

# 個別テスト実行
python -m pytest tests/test_extract.py -v
```

## プロジェクト構造

### 主要ディレクトリ
- `segment_anything/` - SAMコアモジュール
- `commands/` - CLI コマンドモジュール
- `models/` - SAM/YOLOラッパークラス
- `utils/` - 前処理・後処理・通知ユーティリティ
- `scripts/` - バッチ処理・ONNX変換スクリプト
- `test_small/` - テスト用小規模画像セット
- `results_batch/` - バッチ処理結果出力

### 設定ファイル
- `config/pushover.json` - 通知設定（pushover.json.exampleからコピー）
- `requirements.txt` - Python依存関係

### ログファイル
- `kaname03_*.log` - パイプライン実行ログ
- `v042_sequential_full.log` - フル実行ログ

## アーキテクチャ

### 処理フロー
1. **入力画像準備**: test_small/またはカスタムディレクトリから画像読み込み
2. **YOLO検出**: キャラクター候補の境界ボックス検出
3. **SAM精密分割**: YOLOの結果を元にSAMで高精度セグメンテーション
4. **品質評価**: 5つの評価手法から最適な抽出結果を選択
5. **後処理**: マスク適用、背景除去、リサイズ等
6. **結果保存**: results_batch/に処理結果を保存

### 品質評価システム

#### 評価手法
1. **balanced** - バランス重視（推奨）
2. **confidence_priority** - 信頼度優先
3. **size_priority** - サイズ優先  
4. **fullbody_priority** - 全身検出優先
5. **central_priority** - 中心位置優先

#### 実行例
```bash
# バランス手法で実行
python extract_kaname03.py --quality_method balanced

# 信頼度優先で実行  
python extract_kaname03.py --quality_method confidence_priority
```

### コアモジュール

#### models/
- `sam_wrapper.py` - SAMモデルのラッパークラス、CUDA最適化
- `yolo_wrapper.py` - YOLOモデルのラッパークラス、バッチ処理対応

#### commands/
- `extract_character.py` - メインのキャラクター抽出ロジック
- `interactive_extract.py` - インタラクティブ抽出インターフェース
- `quick_interactive.py` - 簡易インタラクティブモード

#### utils/
- `preprocessing.py` - 画像前処理（コントラスト調整、ノイズ除去）
- `postprocessing.py` - 後処理（マスク適用、背景除去）
- `notification.py` - Pushover通知システム
- `performance.py` - パフォーマンス監視
- `difficult_pose.py` - 困難な姿勢の検出・処理

## 開発ガイドライン

### 必須依存関係
- Python 3.8+
- PyTorch 1.7+ (CUDA推奨)
- SAMモデルファイル: `sam_vit_h_4b8939.pth` (2.6GB)
- YOLOモデル: `yolov8n.pt`, `yolov8x.pt`

### コード品質
- flake8, black, mypy, isortによる品質チェック必須
- `./linter.sh`で統合チェック実行
- 100文字行制限
- setup.pyのdev依存関係: `pip install -e .[dev]`

### テスト戦略
- 新機能追加時は対応するテストを作成
- `tests/`ディレクトリにpytest形式でテスト配置
- 段階的テスト: `test_phase2_simple.py` → `test_phase3_cli.py`
- 特殊ケーステスト: `test_difficult_pose.py`, `test_resume_functionality.py`

### 通知設定
```bash
# Pushover設定（オプション）
cp config/pushover.json.example config/pushover.json
# user_key, api_tokenを設定
```

### バッチ処理のベストプラクティス
- 大量処理前に小規模テスト(`test_small/`)で動作確認
- レジューム機能を活用して処理中断時の復旧
- ログファイルで処理状況を監視
- GPU利用可能性とメモリ使用量を事前確認

## トラブルシューティング

### よくある問題
1. **CUDA利用不可**: PyTorchのCUDA版を再インストール
2. **メモリ不足**: バッチサイズを削減、yolov8nモデル使用
3. **モデルファイル不在**: SAMモデル(sam_vit_h_4b8939.pth)をダウンロード
4. **権限エラー**: スクリプトに実行権限付与(`chmod +x`)

### デバッグ方法
```bash
# ログレベル変更
export LOG_LEVEL=DEBUG
python extract_kaname03.py

# テスト用小規模実行
python test_phase2_simple.py

# CUDA確認
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# モデルファイル確認
ls -la sam_vit_h_4b8939.pth yolov8*.pt
```

## 重要な注意事項
- SAMモデルファイル(2.6GB)が必要 - [Meta公式からダウンロード](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
- GPU推奨（CPU処理は非常に遅い）
- バッチ処理は長時間実行される可能性（大規模データセットでは数時間）
- 処理中断時はレジューム機能を利用
- メモリ使用量が大きい（8GB VRAM推奨）

## プロジェクト固有の設計判断

### YOLO + SAMアプローチ
- YOLO: 高速なキャラクター候補検出
- SAM: YOLOボックスをプロンプトとした精密セグメンテーション
- この組み合わせにより、速度と精度を両立

### 5段階品質評価
- 複数の評価軸（信頼度、サイズ、位置、全身等）で最適な結果を選択
- アニメキャラクター特有の課題（複雑な姿勢、部分隠蔽等）に対応

### レジューム機能
- 大規模バッチ処理の中断・再開をサポート
- 処理済みファイルをスキップして効率的に継続