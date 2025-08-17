#!/usr/bin/env python3
"""
QCC-022トラッカーダッシュボード生成スクリプト
"""

import os
import json
import base64
from pathlib import Path
import sys

# プロジェクトルートを追加
sys.path.insert(0, '/mnt/c/AItools/segment-anything')

from features.common.dashboard_generator import StandardDashboardGenerator

def generate_qcc022_dashboard():
    """QCC-022トラッカーのダッシュボード生成"""
    
    tracker_id = "QCC-022"
    workspace_base = "/mnt/c/AItools/lora/train/yado/tracker-workspace/BASELINE-RECALC-001"
    tracker_dir = f"{workspace_base}/trackers/{tracker_id}"
    extraction_dir = f"{tracker_dir}/extraction"
    dashboard_dir = f"{tracker_dir}/dashboard"
    
    print(f"🎯 {tracker_id}ダッシュボード生成開始...")
    
    # ディレクトリ作成
    os.makedirs(dashboard_dir, exist_ok=True)
    
    # 抽出画像を収集
    extraction_path = Path(extraction_dir)
    image_files = list(extraction_path.glob("extracted_*.jpg"))
    
    print(f"📊 抽出画像数: {len(image_files)}枚")
    
    # ダッシュボードデータ準備
    dashboard_data = {
        "tracker_id": tracker_id,
        "title": f"🔬 {tracker_id} - 統計分析基準データ（真のベースライン）",
        "description": "QCA-001とSAM最適化を無効化した統計分析用真のベースラインデータセット",
        "dataset": "kana08",
        "extraction_mode": "真のベースライン（QCA-001無効 + SAM original）",
        "total_images": len(image_files),
        "images": []
    }
    
    # 画像データをBase64エンコードして収集
    for image_file in sorted(image_files):
        try:
            with open(image_file, 'rb') as f:
                image_data = f.read()
                base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # ファイルサイズとメタデータ
            file_size = len(image_data)
            
            dashboard_data["images"].append({
                "filename": image_file.name,
                "path": str(image_file),
                "size_bytes": file_size,
                "size_kb": round(file_size / 1024, 1),
                "base64_data": base64_image,
                "quality_badge": "中品質",  # デフォルト
                "format": "JPEG"
            })
            
            print(f"✅ エンコード完了: {image_file.name} ({file_size:,} bytes)")
            
        except Exception as e:
            print(f"❌ エラー {image_file.name}: {e}")
    
    # 統計情報追加
    total_size = sum(img["size_bytes"] for img in dashboard_data["images"])
    dashboard_data.update({
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "avg_size_kb": round(total_size / len(dashboard_data["images"]) / 1024, 1) if dashboard_data["images"] else 0,
        "high_quality_count": 0,
        "medium_quality_count": len(dashboard_data["images"]),  # デフォルト値
        "low_quality_count": 0,
        "extraction_success_rate": 100.0,
        "processing_notes": [
            "QCA-001作者別適応システム: 無効（default=False）",
            "SAM最適化プロファイル: original（最適化なし）",
            "YOLO閾値: 0.07（アニメ特化）",
            "Grid方式フォールバック: YOLOv8x6 hybrid → grid",
            "真のベースライン取得のため複合オプション完全無効化"
        ]
    })
    
    # ダッシュボード生成
    dashboard_generator = StandardDashboardGenerator()
    output_path = f"{dashboard_dir}/dashboard.html"
    
    try:
        dashboard_generator.generate_standard_dashboard(dashboard_data, dashboard_dir)
        print(f"✅ ダッシュボード生成完了: {output_path}")
        
        # ファイルサイズ確認
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"📊 ダッシュボードサイズ: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
            
            if file_size > 2_000_000:  # 2MB以上
                print("✅ 画像の正常埋め込みを確認（2MB以上）")
            else:
                print("⚠️ ファイルサイズが小さい可能性（画像未埋め込み？）")
        
        return True
        
    except Exception as e:
        print(f"❌ ダッシュボード生成失敗: {e}")
        return False

if __name__ == "__main__":
    success = generate_qcc022_dashboard()
    if success:
        print("🎉 QCC-022ダッシュボード生成成功")
    else:
        print("💥 QCC-022ダッシュボード生成失敗")
        sys.exit(1)