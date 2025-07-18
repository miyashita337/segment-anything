#!/usr/bin/env python3
"""
kaname09 v0.3.3最高品質キャラクター抽出バッチ実行
Phase 1品質評価システム強化版 - 5つの新システム統合実行
"""

import sys
import os
import time
import json
from pathlib import Path
import traceback
from datetime import datetime
from typing import Dict, List, Any

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

def run_kaname09_v033_batch():
    """kaname09 v0.3.3最高品質バッチ抽出実行（Phase 1全機能統合）"""
    
    # パス設定（ユーザー指定）
    input_path = "/mnt/c/AItools/lora/train/yadokugaeru/org/kaname09"
    output_path = "/mnt/c/AItools/lora/train/yadokugaeru/clipped_boundingbox/kaname09_0_3_3"
    
    print("🚀 kaname09 v0.3.3最高品質キャラクター抽出バッチ実行開始")
    print(f"入力パス: {input_path}")
    print(f"出力パス: {output_path}")
    print(f"バージョン: v0.3.3 (Phase 1品質評価システム強化版)")
    
    # 入力パス検証
    if not Path(input_path).exists():
        print(f"❌ エラー: 入力パスが存在しません: {input_path}")
        sys.exit(1)
    
    # 画像ファイル取得
    image_files = list(Path(input_path).glob("*.jpg")) + list(Path(input_path).glob("*.png"))
    
    if not image_files:
        print(f"❌ エラー: 入力パスに画像ファイルがありません: {input_path}")
        sys.exit(1)
    
    print(f"📊 処理対象: {len(image_files)}個の画像ファイル")
    
    # 出力ディレクトリ作成
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # v0.3.3品質評価システム初期化
    quality_analyzers = initialize_v033_quality_systems()
    
    # バッチ処理実行
    try:
        from features.extraction.commands.extract_character import extract_character_from_path
        
        success_count = 0
        error_count = 0
        start_time = time.time()
        
        # v0.3.3品質メトリクス
        quality_scores = []
        processing_times = []
        v033_quality_results = []
        
        # 品質分析結果格納
        batch_quality_analysis = {
            'batch_info': {
                'version': 'v0.3.3',
                'input_path': input_path,
                'output_path': output_path,
                'total_images': len(image_files),
                'start_time': datetime.now().isoformat()
            },
            'individual_results': [],
            'summary_stats': {}
        }
        
        for i, image_file in enumerate(image_files, 1):
            print(f"\n🔄 [{i}/{len(image_files)}] 処理中: {image_file.name}")
            
            try:
                image_start = time.time()
                
                # 出力ファイル名設定（番号付きで整理）
                output_filename = f"{i:05d}_{image_file.stem}.jpg"
                output_file_path = Path(output_path) / output_filename
                
                # v0.3.3最高品質設定での抽出実行
                result = extract_character_from_path(
                    str(image_file),
                    output_path=str(output_file_path),
                    multi_character_criteria='fullbody_priority_enhanced',  # P1-003改良版全身検出
                    enhance_contrast=True,   # コントラスト強化
                    filter_text=True,        # テキストフィルタリング
                    save_mask=True,          # マスク保存
                    save_transparent=True,   # 透明背景保存
                    verbose=False,           # バッチ処理なので詳細ログは抑制
                    high_quality=True,       # 高品質処理
                    difficult_pose=True,     # 困難姿勢対応
                    adaptive_learning=True,  # 適応学習
                    manga_mode=True,         # 漫画モード
                    effect_removal=True,     # エフェクト除去
                    min_yolo_score=0.05,     # YOLO閾値を緩めに設定
                    # Phase 1追加オプション
                    use_enhanced_screentone=True,   # P1-004強化スクリーントーン検出
                    use_mosaic_boundary=True,       # P1-005モザイク境界処理
                    use_solid_fill_enhancement=True, # P1-006ベタ塗り領域改善
                    partial_extraction_check=True,  # P1-002部分抽出検出
                )
                
                image_time = time.time() - image_start
                processing_times.append(image_time)
                
                # v0.3.3品質分析の実行
                v033_analysis = perform_v033_quality_analysis(
                    image_file, output_file_path, result, quality_analyzers
                )
                
                if result.get('success', False):
                    success_count += 1
                    print(f"✅ 成功: {output_filename}")
                    
                    # v0.3.3品質情報表示
                    if 'quality_score' in result:
                        quality_score = result['quality_score']
                        quality_scores.append(quality_score)
                        print(f"   品質スコア: {quality_score:.3f}")
                    
                    # v0.3.3新機能品質情報
                    if v033_analysis:
                        print(f"   境界品質: {v033_analysis.get('boundary_grade', 'N/A')}")
                        print(f"   構造認識: {v033_analysis.get('structure_grade', 'N/A')}")
                        print(f"   分離品質: {v033_analysis.get('separation_grade', 'N/A')}")
                    
                    # Phase 1改善情報表示
                    if 'enhancement_applied' in result:
                        enhancements = result['enhancement_applied']
                        if enhancements:
                            print(f"   適用改善: {', '.join(enhancements)}")
                    
                    print(f"   処理時間: {image_time:.2f}秒")
                    
                else:
                    error_count += 1
                    error_msg = result.get('error', '不明なエラー')
                    print(f"❌ 失敗: {output_filename} - {error_msg}")
                
                # 個別結果を記録
                individual_result = {
                    'image_file': image_file.name,
                    'output_file': output_filename,
                    'success': result.get('success', False),
                    'processing_time': image_time,
                    'quality_score': result.get('quality_score'),
                    'v033_analysis': v033_analysis,
                    'enhancement_applied': result.get('enhancement_applied', []),
                    'error': result.get('error') if not result.get('success', False) else None
                }
                batch_quality_analysis['individual_results'].append(individual_result)
                v033_quality_results.append(v033_analysis)
                
            except Exception as e:
                error_count += 1
                print(f"❌ 処理エラー: {image_file.name} - {str(e)}")
                print(f"   スタックトレース: {traceback.format_exc()}")
                
                # エラー結果を記録
                individual_result = {
                    'image_file': image_file.name,
                    'output_file': output_filename,
                    'success': False,
                    'processing_time': 0.0,
                    'error': str(e),
                    'v033_analysis': None
                }
                batch_quality_analysis['individual_results'].append(individual_result)
        
        # バッチ処理完了統計
        total_time = time.time() - start_time
        success_rate = success_count / len(image_files) * 100
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # v0.3.3品質統計計算
        v033_stats = calculate_v033_quality_statistics(v033_quality_results)
        
        # サマリー統計をバッチ結果に追加
        batch_quality_analysis['summary_stats'] = {
            'total_images': len(image_files),
            'success_count': success_count,
            'error_count': error_count,
            'success_rate': success_rate,
            'total_time': total_time,
            'avg_processing_time': avg_processing_time,
            'avg_quality_score': avg_quality_score,
            'v033_quality_stats': v033_stats,
            'end_time': datetime.now().isoformat()
        }
        
        # 結果レポート出力
        print("\n" + "="*80)
        print("📊 kaname09 v0.3.3バッチ処理完了レポート")
        print("="*80)
        
        print(f"\n📈 処理結果:")
        print(f"  総処理数: {len(image_files)}枚")
        print(f"  成功: {success_count}枚")
        print(f"  失敗: {error_count}枚")
        print(f"  成功率: {success_rate:.1f}%")
        print(f"  総処理時間: {total_time:.1f}秒")
        print(f"  平均処理時間: {avg_processing_time:.2f}秒/枚")
        
        if quality_scores:
            print(f"\n🎯 品質統計:")
            print(f"  平均品質スコア: {avg_quality_score:.3f}")
            print(f"  最高品質スコア: {max(quality_scores):.3f}")
            print(f"  最低品質スコア: {min(quality_scores):.3f}")
        
        # v0.3.3新機能統計表示
        if v033_stats:
            print(f"\n🆕 v0.3.3品質評価統計:")
            for metric, stats in v033_stats.items():
                if stats and 'avg_score' in stats:
                    print(f"  {metric}: 平均{stats['avg_score']:.3f} (グレード分布: {stats.get('grade_distribution', {})})")
        
        # 詳細レポート保存
        report_path = Path(output_path) / f"kaname09_v033_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(batch_quality_analysis, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 詳細レポート保存: {report_path}")
        print(f"\n✅ kaname09 v0.3.3バッチ処理完了!")
        
        return success_rate >= 70.0  # 70%以上の成功率で成功判定
        
    except Exception as e:
        print(f"❌ バッチ処理で重大エラー: {str(e)}")
        print(f"スタックトレース: {traceback.format_exc()}")
        return False


def initialize_v033_quality_systems():
    """v0.3.3品質評価システムの初期化"""
    try:
        from features.evaluation.utils.boundary_analysis import BoundaryAnalyzer
        from features.evaluation.utils.human_structure_recognition import HumanStructureRecognizer
        from features.evaluation.utils.foreground_background_analyzer import ForegroundBackgroundAnalyzer
        from features.evaluation.utils.evaluation_difference_analyzer import EvaluationDifferenceAnalyzer
        from features.evaluation.utils.learning_data_collection import LearningDataCollectionPlanner
        
        return {
            'boundary_analyzer': BoundaryAnalyzer(),
            'structure_recognizer': HumanStructureRecognizer(),
            'separation_analyzer': ForegroundBackgroundAnalyzer(),
            'difference_analyzer': EvaluationDifferenceAnalyzer(),
            'collection_planner': LearningDataCollectionPlanner()
        }
    except ImportError as e:
        print(f"⚠️ v0.3.3品質システム初期化エラー: {e}")
        return {}


def perform_v033_quality_analysis(image_file, output_file, result, analyzers):
    """v0.3.3品質分析の実行"""
    if not analyzers or not result.get('success', False):
        return None
    
    try:
        import cv2
        import numpy as np
        
        # 抽出された画像とマスクを読み込み
        if not output_file.exists():
            return None
        
        extracted_image = cv2.imread(str(output_file))
        if extracted_image is None:
            return None
        
        # マスクファイルがあれば読み込み
        mask_file = output_file.parent / f"{output_file.stem}_mask.png"
        if mask_file.exists():
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        else:
            # RGB画像から簡易マスク作成
            gray = cv2.cvtColor(extracted_image, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        analysis_result = {}
        
        # P1-017: 境界線解析
        if 'boundary_analyzer' in analyzers:
            try:
                boundary_analysis = analyzers['boundary_analyzer'].calculate_boundary_quality_score(mask)
                analysis_result['boundary_analysis'] = boundary_analysis
                analysis_result['boundary_grade'] = boundary_analysis.get('quality_grade', 'F')
                analysis_result['boundary_score'] = boundary_analysis.get('overall_score', 0.0)
            except Exception as e:
                analysis_result['boundary_error'] = str(e)
        
        # P1-019: 人体構造認識
        if 'structure_recognizer' in analyzers:
            try:
                structure_analysis = analyzers['structure_recognizer'].analyze_mask_structure(mask)
                analysis_result['structure_analysis'] = structure_analysis
                overall_assessment = structure_analysis.get('overall_assessment', {})
                analysis_result['structure_grade'] = overall_assessment.get('overall_grade', 'unknown')
                analysis_result['structure_score'] = overall_assessment.get('overall_score', 0.0)
            except Exception as e:
                analysis_result['structure_error'] = str(e)
        
        # P1-021: 背景・前景分離
        if 'separation_analyzer' in analyzers:
            try:
                original_image = cv2.imread(str(image_file))
                if original_image is not None:
                    if original_image.shape[:2] != mask.shape[:2]:
                        original_image = cv2.resize(original_image, (mask.shape[1], mask.shape[0]))
                    
                    separation_analysis = analyzers['separation_analyzer'].analyze_separation_quality(original_image, mask)
                    analysis_result['separation_analysis'] = separation_analysis
                    separation_score = separation_analysis.get('separation_score', {})
                    analysis_result['separation_grade'] = separation_score.get('quality_grade', 'F')
                    analysis_result['separation_score'] = separation_score.get('overall_score', 0.0)
            except Exception as e:
                analysis_result['separation_error'] = str(e)
        
        return analysis_result
        
    except Exception as e:
        return {'error': str(e)}


def calculate_v033_quality_statistics(v033_results):
    """v0.3.3品質統計計算"""
    if not v033_results:
        return {}
    
    valid_results = [r for r in v033_results if r and 'error' not in r]
    if not valid_results:
        return {}
    
    stats = {}
    
    # 境界線品質統計
    boundary_scores = [r.get('boundary_score', 0) for r in valid_results if 'boundary_score' in r]
    boundary_grades = [r.get('boundary_grade', 'F') for r in valid_results if 'boundary_grade' in r]
    
    if boundary_scores:
        stats['boundary_quality'] = {
            'avg_score': sum(boundary_scores) / len(boundary_scores),
            'max_score': max(boundary_scores),
            'min_score': min(boundary_scores),
            'grade_distribution': {grade: boundary_grades.count(grade) for grade in set(boundary_grades)}
        }
    
    # 構造認識統計
    structure_scores = [r.get('structure_score', 0) for r in valid_results if 'structure_score' in r]
    structure_grades = [r.get('structure_grade', 'unknown') for r in valid_results if 'structure_grade' in r]
    
    if structure_scores:
        stats['structure_recognition'] = {
            'avg_score': sum(structure_scores) / len(structure_scores),
            'max_score': max(structure_scores),
            'min_score': min(structure_scores),
            'grade_distribution': {grade: structure_grades.count(grade) for grade in set(structure_grades)}
        }
    
    # 分離品質統計
    separation_scores = [r.get('separation_score', 0) for r in valid_results if 'separation_score' in r]
    separation_grades = [r.get('separation_grade', 'F') for r in valid_results if 'separation_grade' in r]
    
    if separation_scores:
        stats['separation_quality'] = {
            'avg_score': sum(separation_scores) / len(separation_scores),
            'max_score': max(separation_scores),
            'min_score': min(separation_scores),
            'grade_distribution': {grade: separation_grades.count(grade) for grade in set(separation_grades)}
        }
    
    return stats


def main():
    """メイン実行関数"""
    print("🚀 kaname09 v0.3.3最高品質バッチ処理開始")
    
    success = run_kaname09_v033_batch()
    
    if success:
        print("\n✅ バッチ処理成功!")
        sys.exit(0)
    else:
        print("\n❌ バッチ処理失敗")
        sys.exit(1)


if __name__ == "__main__":
    main()