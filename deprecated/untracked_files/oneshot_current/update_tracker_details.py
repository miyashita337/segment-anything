#!/usr/bin/env python3
"""
新規作成トラッカーの詳細情報と登録日時を一括更新
P1-B004, P1-B005, P1-B006, P1-B007を対象
"""

import sys
from datetime import datetime
from pathlib import Path

# プロジェクトルート追加
sys.path.append(str(Path(__file__).parent))

from tools.progress_tracker.config import get_default_config
from tools.progress_tracker.data_models import PriorityLevel, TaskRecord, TaskStatus
from tools.progress_tracker.progress_manager import ProgressManager


def update_tracker_details():
    """トラッカー詳細情報を更新"""

    # 設定と管理クラス初期化
    config = get_default_config()
    manager = ProgressManager(config)

    # 現在日時設定
    current_date = datetime.now()

    # 更新対象トラッカーの詳細情報
    tracker_updates = {
        "P1-B004": {
            "details": """【実装仕様】
• features/processing/adaptive_cropping.py新規作成
• MediaPipe顔検出統合による主要キャラ特定アルゴリズム
• 重複領域IoU計算ベース境界ボックス最適化機能
• マルチスケール候補生成(0.8x, 1.0x, 1.2x)と品質評価ベース最適選択
• YOLO検出結果を顔検出で検証・分割する二段階システム
• 混入キャラ自動除外のための信頼度閾値調整機能
• extract_character.pyへのオプション追加: --adaptive-cropping

【期待効果】他キャラ混入30% → 5-10%削減(67-83%改善)""",
            "description": "複数キャラ混入を防ぐ適応的クロッピング実装",
        },
        "P1-B005": {
            "details": """【実装仕様】
• features/processing/text_removal.py新規作成
• EasyOCR(日本語・英語対応)バッチ処理パイプライン構築
• 吹き出し検出: 楕円・矩形形状認識アルゴリズム実装
• 効果音検出: 文字密度・フォントサイズ解析による自動分類
• テキスト領域自動マスキング: OpenCV morphological operations使用
• 処理結果キャッシュ機能(SHA256ハッシュベース)
• extract_character.pyへのオプション追加: --remove-text --text-confidence-threshold

【期待効果】テキスト混入40% → 10-15%削減(62-75%改善)""",
            "description": "EasyOCR活用によるリアルタイムテキスト検出・除去",
        },
        "P1-B006": {
            "details": """【実装仕様】
• features/processing/smart_cropping.py新規作成
• テキスト領域回避型クロッピング戦略: 重要度マップベース領域選択
• 重要度計算: 顔領域(50%)、胴体(30%)、手足(20%)重み付け
• メモリ効率的バッチ処理: 画像分割＋並列処理アーキテクチャ
• 最小品質保証: アスペクト比1.2-2.5維持、最小面積5%確保
• フォールバック機能: スマートクロップ失敗時の従来処理維持
• extract_character.pyへのオプション追加: --smart-crop --min-character-area

【技術的根拠】インペイント代替案: GPU負荷7-10GB→3GB、処理時間25-50秒→5秒維持""",
            "description": "GPU負荷大なインペイント代替の軽量ソリューション",
        },
        "P1-B007": {
            "details": """【実装仕様】
• features/evaluation/pose_integrity_checker.py新規作成
• マルチマスク検証: 複数SAMマスク候補の姿勢一貫性相互検証
• 輪郭完整性評価: OpenCV Canny edge detection＋連結成分解析
• 部分欠損検出: 期待領域(頭・胴・四肢) vs 実検出領域の差分解析
• 補正機能: 境界ボックス10-20%拡張による欠損部位回復試行
• 品質スコア算出: 完整性度合い数値化(0.0-1.0)
• extract_character.pyへのオプション追加: --pose-validation --integrity-threshold

【期待効果】手足切断防止、キャラクター完整性保証による品質向上""",
            "description": "手足切断防止とキャラクター完整性保証",
        },
    }

    print("トラッカー詳細情報更新開始...")

    for tracker_id, update_data in tracker_updates.items():
        try:
            # 既存タスクを取得
            existing_task = manager.get_task(tracker_id)

            if existing_task:
                # 詳細情報を更新
                existing_task.details = update_data["details"]
                existing_task.description = update_data["description"]
                existing_task.priority = PriorityLevel.HIGH  # 優先度最高に設定

                # 作成日時が未設定の場合は現在日時を設定
                if not existing_task.created_date:
                    existing_task.created_date = current_date

                # 更新日時を現在日時に設定
                existing_task.updated_date = current_date

                # Google Sheetsに更新反映
                manager.client.update_task(existing_task)

                print(f"SUCCESS {tracker_id}: 詳細情報更新完了")
                print(f"   作成日時: {existing_task.created_date}")
                print(f"   更新日時: {existing_task.updated_date}")
                print(f"   詳細長: {len(update_data['details'])}文字")

            else:
                print(f"ERROR {tracker_id}: タスクが見つかりません")

        except Exception as e:
            print(f"ERROR {tracker_id}: 更新エラー - {e}")

    print("\n詳細情報更新処理完了")


if __name__ == "__main__":
    update_tracker_details()
