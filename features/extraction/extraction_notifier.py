"""
⚠️ DEPRECATED: このモジュールは廃止されました
代わりに features.common.notification.global_pushover を使用してください

from features.common.notification.global_pushover import (
    notify_success,
    notify_error,
    notify_process_complete
)
"""

"""
抽出パイプライン用Pushover通知システム

抽出完了時に成功画像のグリッドと統計情報を自動送信
"""

import numpy as np

import json
import logging
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import Any, Dict, List, Optional

try:
    from features.common.notification.global_pushover import (
        notify_error,
        notify_process_complete,
        notify_success,
    )
    PUSHOVER_AVAILABLE = True
except ImportError:
    PUSHOVER_AVAILABLE = False

logger = logging.getLogger(__name__)


class ExtractionNotifier:
    """抽出パイプライン専用通知システム"""
    
    def __init__(self):
        """初期化"""
        self.pushover = None
        if PUSHOVER_AVAILABLE:
            try:
                from features.common.notification.notification import PushoverNotifier
                self.pushover = PushoverNotifier()
                logger.info("Pushover通知システム初期化完了")
            except Exception as e:
                logger.warning(f"Pushover初期化失敗: {e}")
                self.pushover = None
        else:
            logger.warning("Pushover通知システム利用不可")
    
    def send_extraction_completion_notification(self, 
                                               extraction_results: Dict[str, Any],
                                               include_images: bool = True) -> bool:
        """
        抽出完了通知を送信
        
        Args:
            extraction_results: 抽出結果の辞書
            include_images: 成功画像を含めるか
            
        Returns:
            送信成功フラグ
        """
        if not self.pushover:
            logger.warning("Pushover通知システムが利用できません")
            return False
        
        try:
            # 統計情報の作成
            total_images = extraction_results.get('total_images', 0)
            successful_extractions = extraction_results.get('successful_extractions', [])
            success_count = len(successful_extractions)
            success_rate = (success_count / total_images * 100) if total_images > 0 else 0
            
            # 基本通知メッセージ
            message = f"""🎯 キャラクター抽出完了

📊 実行結果:
• 総画像数: {total_images}枚
• 成功数: {success_count}枚 ({success_rate:.1f}%)
• 失敗数: {total_images - success_count}枚

📝 品質分布:
{self._format_quality_distribution(extraction_results)}

⏱️ 処理時間: {extraction_results.get('total_processing_time', 'N/A')}秒

🔧 使用モデル: {extraction_results.get('quality_method', 'balanced')}
"""
            
            # 改善提案があれば追加
            if extraction_results.get('improvement_suggestions'):
                suggestions = extraction_results['improvement_suggestions'][:3]  # 最大3つ
                message += f"\n💡 改善提案:\n"
                for i, suggestion in enumerate(suggestions, 1):
                    message += f"  {i}. {suggestion}\n"
            
            # 画像付き通知
            if include_images and success_count > 0:
                try:
                    # 成功画像のパスを取得
                    success_image_paths = self._get_success_image_paths(successful_extractions)
                    
                    if success_image_paths:
                        # グリッド画像を作成
                        grid_path = self._create_success_images_grid(
                            success_image_paths, 
                            extraction_results.get('output_dir', '/tmp')
                        )
                        
                        if grid_path and Path(grid_path).exists():
                            # 画像付きで送信
                            success = self.pushover.send_notification_with_image(
                                message=message,
                                image_path=grid_path,
                                title="🎯 キャラクター抽出完了",
                                priority=1 if success_rate >= 80 else 0
                            )
                            
                            # グリッド画像を削除
                            try:
                                Path(grid_path).unlink()
                            except Exception as e:
                                logger.warning(f"グリッド画像削除失敗: {e}")
                            
                            if success:
                                logger.info(f"抽出完了通知送信成功（画像付き）: {success_count}枚")
                                return True
                except Exception as e:
                    logger.warning(f"画像付き通知失敗、テキストのみで再試行: {e}")
            
            # テキストのみ通知（フォールバック）
            success = self.pushover.send_notification(
                message=message,
                title="🎯 キャラクター抽出完了",
                priority=1 if success_rate >= 80 else 0
            )
            
            if success:
                logger.info(f"抽出完了通知送信成功（テキストのみ）")
                return True
            else:
                logger.error("抽出完了通知送信失敗")
                return False
                
        except Exception as e:
            logger.error(f"抽出完了通知送信エラー: {e}")
            return False
    
    def _format_quality_distribution(self, extraction_results: Dict[str, Any]) -> str:
        """品質分布のフォーマット"""
        try:
            quality_dist = extraction_results.get('quality_distribution', {})
            if not quality_dist:
                return "• 品質評価データなし"
            
            lines = []
            for grade in ['A', 'B', 'C', 'D', 'E', 'F']:
                count = quality_dist.get(grade, 0)
                if count > 0:
                    lines.append(f"• {grade}評価: {count}枚")
            
            return "\n".join(lines) if lines else "• 品質評価データなし"
            
        except Exception as e:
            logger.warning(f"品質分布フォーマットエラー: {e}")
            return "• 品質評価データ取得エラー"
    
    def _get_success_image_paths(self, successful_extractions: List[Dict]) -> List[str]:
        """成功した抽出画像のパスを取得"""
        image_paths = []
        
        try:
            for result in successful_extractions[:16]:  # 最大16枚
                # 出力ファイルパスを取得
                output_path = result.get('output_path')
                if output_path and Path(output_path).exists():
                    image_paths.append(str(output_path))
                else:
                    # フォールバック: extracted_filesから取得
                    extracted_files = result.get('extracted_files', [])
                    for file_path in extracted_files:
                        if Path(file_path).exists():
                            image_paths.append(str(file_path))
                            break
            
            logger.info(f"成功画像パス取得: {len(image_paths)}枚")
            return image_paths
            
        except Exception as e:
            logger.error(f"成功画像パス取得エラー: {e}")
            return []
    
    def _create_success_images_grid(self, image_paths: List[str], output_dir: str) -> Optional[str]:
        """成功画像のグリッドを作成"""
        try:
            if not image_paths:
                return None
            
            # グリッドサイズを決定
            num_images = len(image_paths)
            if num_images <= 4:
                grid_size = (2, 2)
            elif num_images <= 9:
                grid_size = (3, 3)
            else:
                grid_size = (4, 4)
            
            # 画像を読み込み、リサイズ
            thumbnail_size = (200, 200)
            grid_images = []
            
            for i in range(grid_size[0] * grid_size[1]):
                if i < len(image_paths):
                    try:
                        img = Image.open(image_paths[i])
                        # アスペクト比を保ってリサイズ
                        img.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
                        
                        # 正方形にパディング
                        new_img = Image.new('RGB', thumbnail_size, (255, 255, 255))
                        paste_x = (thumbnail_size[0] - img.width) // 2
                        paste_y = (thumbnail_size[1] - img.height) // 2
                        new_img.paste(img, (paste_x, paste_y))
                        
                        grid_images.append(new_img)
                    except Exception as e:
                        logger.warning(f"画像読み込みエラー {image_paths[i]}: {e}")
                        # エラー時は空白画像
                        blank = Image.new('RGB', thumbnail_size, (240, 240, 240))
                        grid_images.append(blank)
                else:
                    # 空白画像
                    blank = Image.new('RGB', thumbnail_size, (250, 250, 250))
                    grid_images.append(blank)
            
            # グリッド画像を作成
            grid_width = grid_size[0] * thumbnail_size[0]
            grid_height = grid_size[1] * thumbnail_size[1]
            grid_image = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
            
            for i, img in enumerate(grid_images):
                row = i // grid_size[0]
                col = i % grid_size[0]
                x = col * thumbnail_size[0]
                y = row * thumbnail_size[1]
                grid_image.paste(img, (x, y))
            
            # ファイル名とヘッダーを追加
            header_height = 60
            final_image = Image.new('RGB', (grid_width, grid_height + header_height), (255, 255, 255))
            
            # ヘッダーテキスト
            draw = ImageDraw.Draw(final_image)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            header_text = f"成功抽出結果 ({num_images}枚)"
            text_bbox = draw.textbbox((0, 0), header_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_x = (grid_width - text_width) // 2
            draw.text((text_x, 20), header_text, fill=(0, 0, 0), font=font)
            
            # グリッドを貼り付け
            final_image.paste(grid_image, (0, header_height))
            
            # 画像サイズを確認・調整（2.5MB制限）
            output_path = Path(output_dir) / f"extraction_success_grid_{Path().cwd().name}_{Path(image_paths[0]).stem}.jpg"
            
            # JPEG品質を調整して保存
            for quality in [95, 85, 75, 65]:
                final_image.save(output_path, 'JPEG', quality=quality, optimize=True)
                
                if output_path.stat().st_size <= 2.5 * 1024 * 1024:  # 2.5MB以下
                    logger.info(f"成功画像グリッド作成完了: {output_path} (品質: {quality})")
                    return str(output_path)
            
            # 最終的にサイズオーバーの場合
            logger.warning(f"グリッド画像サイズが制限を超過: {output_path.stat().st_size / 1024 / 1024:.1f}MB")
            return str(output_path)  # それでも送信を試行
            
        except Exception as e:
            logger.error(f"成功画像グリッド作成エラー: {e}")
            return None


def create_extraction_results_dict(total_images: int,
                                   successful_extractions: List[Dict],
                                   processing_time: float,
                                   quality_method: str = "balanced",
                                   output_dir: str = "",
                                   quality_distribution: Optional[Dict] = None) -> Dict[str, Any]:
    """
    抽出結果辞書を作成するヘルパー関数
    
    Args:
        total_images: 総画像数
        successful_extractions: 成功した抽出結果のリスト
        processing_time: 処理時間（秒）
        quality_method: 使用した品質評価方法
        output_dir: 出力ディレクトリ
        quality_distribution: 品質分布
        
    Returns:
        抽出結果辞書
    """
    return {
        'total_images': total_images,
        'successful_extractions': successful_extractions,
        'total_processing_time': processing_time,
        'quality_method': quality_method,
        'output_dir': output_dir,
        'quality_distribution': quality_distribution or {},
        'improvement_suggestions': [
            "YOLO閾値調整による検出精度向上",
            "SAM後処理パイプライン最適化",
            "困難姿勢への対応強化"
        ]
    }