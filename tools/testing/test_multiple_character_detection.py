from features.common.environment_manager import get_path, get_test_image_path, setup_test_env, is_ci_environment
#!/usr/bin/env python3
"""
Multiple Character Detection System - Test Script
複数キャラクター検出システムのテストとバリデーション
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2
import json
import time
from typing import List, Dict, Any, Optional

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from features.extraction.models.yolo_wrapper import YOLOModelWrapper
from features.evaluation.utils.multiple_character_detector import (
    MultipleCharacterDetector, 
    detect_multiple_characters_from_image,
    MultipleCharacterType
)
from features.evaluation.boundary_case_detector import BoundaryCaseDetector


class MultipleCharacterTestSuite:
    """複数キャラクター検出システムのテストスイート"""
    
    def __init__(self, test_image_dir: Path):
        """
        初期化
        
        Args:
            test_image_dir: テスト画像ディレクトリ
        """
        self.test_image_dir = Path(test_image_dir)
        self.yolo_wrapper = YOLOModelWrapper()
        self.detector = MultipleCharacterDetector()
        self.boundary_detector = BoundaryCaseDetector("TEST-MULTI-CHAR")
        
        # テスト結果保存用
        self.results = []
        self.performance_stats = {}
        
    def setup(self) -> bool:
        """
        テスト環境セットアップ
        
        Returns:
            成功フラグ
        """
        print("🔧 テスト環境セットアップ中...")
        
        # テスト画像ディレクトリ存在確認
        if not self.test_image_dir.exists():
            print(f"❌ テスト画像ディレクトリが存在しません: {self.test_image_dir}")
            return False
        
        # YOLO Wrapper初期化
        if not self.yolo_wrapper.load_model():
            print("❌ YOLO model loading failed")
            return False
        
        print("✅ テスト環境セットアップ完了")
        return True
    
    def collect_test_images(self) -> List[Path]:
        """
        テスト画像収集
        
        Returns:
            画像パスリスト
        """
        print(f"📁 テスト画像収集: {self.test_image_dir}")
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.test_image_dir.glob(f"*{ext}"))
            image_files.extend(self.test_image_dir.glob(f"*{ext.upper()}"))
        
        print(f"🖼️ {len(image_files)}枚の画像を発見")
        return sorted(image_files)
    
    def test_basic_detection(self, image_paths: List[Path]) -> Dict[str, Any]:
        """
        基本検出機能テスト
        
        Args:
            image_paths: テスト画像パス
            
        Returns:
            テスト結果統計
        """
        print("\n🔍 === 基本検出機能テスト ===")
        
        results = {
            'total_images': len(image_paths),
            'successful_detections': 0,
            'detection_errors': 0,
            'detection_types': {},
            'character_counts': {},
            'penalty_distribution': [],
            'processing_times': [],
        }
        
        start_time = time.time()
        
        for i, image_path in enumerate(image_paths):
            try:
                print(f"\n📸 テスト {i+1}/{len(image_paths)}: {image_path.name}")
                
                # 処理時間計測開始
                img_start_time = time.time()
                
                # 複数キャラクター検出実行
                multi_char_result = detect_multiple_characters_from_image(
                    image_path, self.yolo_wrapper, save_visualization=True
                )
                
                # 処理時間計測終了
                processing_time = time.time() - img_start_time
                results['processing_times'].append(processing_time)
                
                # 結果記録
                results['successful_detections'] += 1
                
                # 検出タイプ統計
                detection_type = multi_char_result.detection_type.value
                results['detection_types'][detection_type] = results['detection_types'].get(detection_type, 0) + 1
                
                # キャラクター数統計
                char_count = multi_char_result.character_count
                results['character_counts'][char_count] = results['character_counts'].get(char_count, 0) + 1
                
                # ペナルティスコア収集
                results['penalty_distribution'].append(multi_char_result.penalty_score)
                
                # 詳細結果出力
                print(f"   ✅ 検出完了: {char_count}体 (タイプ: {detection_type})")
                print(f"   📊 ペナルティ: {multi_char_result.penalty_score:.3f}")
                print(f"   ⏱️ 処理時間: {processing_time:.2f}s")
                
                if multi_char_result.is_multiple:
                    print(f"   🎯 メインキャラ: #{multi_char_result.primary_character_index + 1}")
                    print(f"   💡 改善提案: {len(multi_char_result.improvement_suggestions)}件")
                
                # 結果保存
                self.results.append({
                    'image_path': str(image_path),
                    'image_name': image_path.name,
                    'character_count': char_count,
                    'detection_type': detection_type,
                    'penalty_score': multi_char_result.penalty_score,
                    'is_multiple': multi_char_result.is_multiple,
                    'processing_time': processing_time,
                    'improvement_suggestions': multi_char_result.improvement_suggestions,
                })
                
            except Exception as e:
                results['detection_errors'] += 1
                print(f"   ❌ エラー: {str(e)}")
                
                self.results.append({
                    'image_path': str(image_path),
                    'image_name': image_path.name,
                    'error': str(e),
                    'processing_time': 0,
                })
        
        # 統計計算
        total_time = time.time() - start_time
        
        if results['penalty_distribution']:
            results['average_penalty'] = np.mean(results['penalty_distribution'])
            results['max_penalty'] = max(results['penalty_distribution'])
            results['min_penalty'] = min(results['penalty_distribution'])
        
        if results['processing_times']:
            results['average_processing_time'] = np.mean(results['processing_times'])
            results['total_processing_time'] = total_time
        
        results['success_rate'] = results['successful_detections'] / results['total_images'] * 100
        
        print(f"\n📈 基本検出テスト完了:")
        print(f"   成功率: {results['success_rate']:.1f}% ({results['successful_detections']}/{results['total_images']})")
        print(f"   平均処理時間: {results.get('average_processing_time', 0):.2f}s/枚")
        print(f"   平均ペナルティ: {results.get('average_penalty', 0):.3f}")
        
        return results
    
    def test_filtering_functionality(self, image_paths: List[Path]) -> Dict[str, Any]:
        """
        フィルタリング機能テスト
        
        Args:
            image_paths: テスト画像パス
            
        Returns:
            フィルタリングテスト結果
        """
        print("\n🚰 === フィルタリング機能テスト ===")
        
        # 異なる閾値でのフィルタリングテスト
        thresholds = [0.3, 0.5, 0.7]
        filtering_results = {}
        
        for threshold in thresholds:
            print(f"\n🎚️ 閾値 {threshold} でのフィルタリングテスト")
            
            single_char_images, multi_char_images, stats = self.yolo_wrapper.filter_single_character_images(
                image_paths, penalty_threshold=threshold, save_reports=False
            )
            
            filtering_results[threshold] = {
                'threshold': threshold,
                'single_character_count': len(single_char_images),
                'multiple_character_count': len(multi_char_images),
                'filtering_rate': stats.get('filtering_rate', 0),
                'statistics': stats,
            }
            
            print(f"   ✅ 単一キャラクター: {len(single_char_images)}枚")
            print(f"   🚫 除外: {len(multi_char_images)}枚 ({stats.get('filtering_rate', 0):.1f}%)")
        
        return filtering_results
    
    def test_boundary_case_integration(self, image_paths: List[Path]) -> Dict[str, Any]:
        """
        境界ケース検出統合テスト
        
        Args:
            image_paths: テスト画像パス
            
        Returns:
            統合テスト結果
        """
        print("\n🔗 === 境界ケース検出統合テスト ===")
        
        boundary_results = {
            'total_images': len(image_paths),
            'boundary_cases_found': 0,
            'multi_char_boundaries': 0,
            'boundary_types': {},
            'integration_errors': 0,
        }
        
        for i, image_path in enumerate(image_paths[:10]):  # サンプル10枚でテスト
            try:
                print(f"\n🔍 境界検出テスト {i+1}/10: {image_path.name}")
                
                # 境界ケース検出実行
                boundary_case = self.boundary_detector.process_image(image_path)
                
                if boundary_case:
                    boundary_results['boundary_cases_found'] += 1
                    
                    case_type = boundary_case.case_type.value
                    boundary_results['boundary_types'][case_type] = boundary_results['boundary_types'].get(case_type, 0) + 1
                    
                    if case_type == 'multiple_character_boundary':
                        boundary_results['multi_char_boundaries'] += 1
                    
                    print(f"   🚨 境界ケース: {case_type}")
                    print(f"   📊 信頼度: {boundary_case.confidence_score:.3f}")
                    print(f"   💭 理由: {boundary_case.detection_reason}")
                else:
                    print(f"   ✅ 境界ケースなし")
                    
            except Exception as e:
                boundary_results['integration_errors'] += 1
                print(f"   ❌ 統合エラー: {str(e)}")
        
        boundary_results['boundary_detection_rate'] = boundary_results['boundary_cases_found'] / min(len(image_paths), 10) * 100
        
        print(f"\n📈 境界ケース統合テスト完了:")
        print(f"   境界ケース検出率: {boundary_results['boundary_detection_rate']:.1f}%")
        print(f"   複数キャラ境界: {boundary_results['multi_char_boundaries']}件")
        
        return boundary_results
    
    def generate_test_report(self, 
                           basic_results: Dict[str, Any],
                           filtering_results: Dict[str, Any],
                           boundary_results: Dict[str, Any]) -> Path:
        """
        テストレポート生成
        
        Args:
            basic_results: 基本テスト結果
            filtering_results: フィルタリングテスト結果
            boundary_results: 境界ケース統合テスト結果
            
        Returns:
            レポートファイルパス
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = project_root / "test_results" / f"multi_char_detection_test_{timestamp}.json"
        
        # ディレクトリ作成
        report_file.parent.mkdir(exist_ok=True)
        
        # 統合レポート作成
        full_report = {
            'test_metadata': {
                'timestamp': timestamp,
                'test_image_directory': str(self.test_image_dir),
                'total_test_images': basic_results['total_images'],
                'yolo_model': self.yolo_wrapper.model_path,
            },
            'basic_detection_results': basic_results,
            'filtering_results': filtering_results,
            'boundary_integration_results': boundary_results,
            'individual_results': self.results,
            'performance_summary': {
                'overall_success_rate': basic_results.get('success_rate', 0),
                'average_processing_time': basic_results.get('average_processing_time', 0),
                'detection_accuracy': boundary_results.get('boundary_detection_rate', 0),
            }
        }
        
        # JSON保存
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(full_report, f, ensure_ascii=False, indent=2)
        
        print(f"📊 テストレポート保存: {report_file}")
        
        # マークダウンレポートも生成
        markdown_file = report_file.with_suffix('.md')
        self._generate_markdown_report(full_report, markdown_file)
        
        return report_file
    
    def _generate_markdown_report(self, report_data: Dict[str, Any], output_file: Path):
        """マークダウンレポート生成"""
        lines = [
            "# 複数キャラクター検出システム - テストレポート",
            "",
            f"**テスト実行日時**: {report_data['test_metadata']['timestamp']}",
            f"**テスト画像数**: {report_data['test_metadata']['total_test_images']}枚",
            f"**使用モデル**: {report_data['test_metadata']['yolo_model']}",
            "",
            "## 🎯 テスト結果サマリー",
            "",
            f"- **総合成功率**: {report_data['performance_summary']['overall_success_rate']:.1f}%",
            f"- **平均処理時間**: {report_data['performance_summary']['average_processing_time']:.2f}s/枚",
            f"- **境界ケース検出精度**: {report_data['performance_summary']['detection_accuracy']:.1f}%",
            "",
            "## 📊 基本検出結果",
            "",
        ]
        
        # 基本結果詳細
        basic = report_data['basic_detection_results']
        lines.extend([
            f"- **成功検出**: {basic['successful_detections']}/{basic['total_images']}枚",
            f"- **エラー**: {basic['detection_errors']}枚",
            f"- **平均ペナルティスコア**: {basic.get('average_penalty', 0):.3f}",
            "",
            "### 検出タイプ分布",
            "",
        ])
        
        for det_type, count in basic.get('detection_types', {}).items():
            lines.append(f"- **{det_type}**: {count}枚")
        
        lines.extend([
            "",
            "### キャラクター数分布",
            "",
        ])
        
        for char_count, count in basic.get('character_counts', {}).items():
            lines.append(f"- **{char_count}体**: {count}枚")
        
        # フィルタリング結果
        lines.extend([
            "",
            "## 🚰 フィルタリング結果",
            "",
        ])
        
        for threshold, result in report_data['filtering_results'].items():
            lines.extend([
                f"### 閾値 {threshold}",
                f"- 単一キャラクター: {result['single_character_count']}枚",
                f"- 除外: {result['multiple_character_count']}枚 ({result['filtering_rate']:.1f}%)",
                "",
            ])
        
        # 境界ケース統合結果
        boundary = report_data['boundary_integration_results']
        lines.extend([
            "## 🔗 境界ケース統合結果",
            "",
            f"- **境界ケース発見**: {boundary['boundary_cases_found']}件",
            f"- **複数キャラ境界**: {boundary['multi_char_boundaries']}件",
            f"- **統合エラー**: {boundary['integration_errors']}件",
            "",
        ])
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"📄 マークダウンレポート保存: {output_file}")
    
    def cleanup(self):
        """リソース解放"""
        try:
            self.yolo_wrapper.unload_model()
        except:
            pass
        print("🧹 テストリソース解放完了")


def main():
    """メイン実行関数"""
    print("🚀 Multiple Character Detection System - Test Suite")
    print("=" * 60)
    
    # テスト画像ディレクトリ設定
    test_dirs = [
        project_root / "test_small",  # プロジェクトのテスト画像
        Path(get_path("data", "org", "kana08")),  # 実データ（存在する場合）
    ]
    
    # 利用可能なテストディレクトリを探す
    test_dir = None
    for candidate in test_dirs:
        if candidate.exists() and any(candidate.glob("*.jpg")) or any(candidate.glob("*.png")):
            test_dir = candidate
            break
    
    if not test_dir:
        print("❌ テスト用画像ディレクトリが見つかりません")
        print("   以下のディレクトリのいずれかに画像を配置してください:")
        for d in test_dirs:
            print(f"   - {d}")
        return False
    
    # テストスイート実行
    test_suite = MultipleCharacterTestSuite(test_dir)
    
    try:
        # セットアップ
        if not test_suite.setup():
            return False
        
        # テスト画像収集
        image_paths = test_suite.collect_test_images()
        if not image_paths:
            print("❌ テスト画像が見つかりません")
            return False
        
        # 画像数制限（デモでは最大20枚）
        if len(image_paths) > 20:
            print(f"🎚️ 画像数制限: {len(image_paths)} → 20枚（デモモード）")
            image_paths = image_paths[:20]
        
        # テスト実行
        print(f"\n📋 テスト実行開始: {len(image_paths)}枚の画像で検証")
        
        # 1. 基本検出機能テスト
        basic_results = test_suite.test_basic_detection(image_paths)
        
        # 2. フィルタリング機能テスト
        filtering_results = test_suite.test_filtering_functionality(image_paths)
        
        # 3. 境界ケース統合テスト
        boundary_results = test_suite.test_boundary_case_integration(image_paths)
        
        # レポート生成
        report_file = test_suite.generate_test_report(basic_results, filtering_results, boundary_results)
        
        print("\n🎉 全テスト完了!")
        print(f"📊 詳細結果: {report_file}")
        print(f"📄 マークダウン: {report_file.with_suffix('.md')}")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️ テスト中断")
        return False
    except Exception as e:
        print(f"\n❌ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        test_suite.cleanup()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)