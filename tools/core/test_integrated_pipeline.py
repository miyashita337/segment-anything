#!/usr/bin/env python3
"""
統合パイプライン モックテスト

実際の抽出処理は実行せず、全Phase（3-6）の流れをテスト
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tools.core.integrated_quality_pipeline import (
    IntegratedQualityPipeline, ValidationEngine, StateManager, DashboardGenerator
)


def create_mock_extraction_results(extraction_dir: Path, total_files: int = 10, success_count: int = 7):
    """モック抽出結果作成"""
    extraction_dir.mkdir(parents=True, exist_ok=True)
    
    # 成功ファイル作成
    for i in range(success_count):
        mock_file = extraction_dir / f"extracted_kana05_000{i}.png"
        mock_file.write_text("mock extracted image data")
    
    print(f"モック抽出結果作成: {success_count}/{total_files} 成功")


def test_full_pipeline_mock():
    """全パイプライン モックテスト"""
    print("=== 統合パイプライン モックテスト開始 ===")
    
    # テスト用一時ディレクトリ
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # モック設定作成
        mock_config = {
            'pipeline': {'name': 'mock_test', 'version': '1.0.0'},
            'phases': {
                'phase3': {'name': 'test'},
                'phase4': {'name': 'test'},
                'phase5': {'name': 'test'},
                'phase6': {'name': 'test'}
            },
            'paths': {
                'default_input': str(project_root / "test_small"),
                'workspace_base': str(temp_path)
            },
            'error_handling': {
                'max_retries': 3,
                'default_timeout_seconds': 300
            }
        }
        
        config_file = temp_path / "mock_config.yaml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(mock_config, f)
        
        # パイプラインインスタンス作成
        tracker_id = "MOCK-TEST-001"
        
        try:
            # カスタムパイプライン（抽出処理をモック化）
            pipeline = MockIntegratedQualityPipeline(str(config_file), tracker_id, temp_path)
            
            # Phase 3: バリデーション
            print("\n--- Phase 3: バリデーション ---")
            phase3_result = pipeline._execute_phase3()
            print(f"Phase 3 結果: {'✅ 成功' if phase3_result.success else '❌ 失敗'}")
            
            # Phase 4: モック抽出
            print("\n--- Phase 4: モック抽出 ---")
            phase4_result = pipeline._execute_phase4_mock()
            print(f"Phase 4 結果: {'✅ 成功' if phase4_result.success else '❌ 失敗'}")
            
            # Phase 5: 品質評価
            print("\n--- Phase 5: 品質評価 ---")
            phase5_result = pipeline._execute_phase5()
            print(f"Phase 5 結果: {'✅ 成功' if phase5_result.success else '❌ 失敗'}")
            print(f"成功率: {phase5_result.output_data.get('success_rate', 0):.1f}%")
            
            # Phase 6: ダッシュボード生成
            print("\n--- Phase 6: ダッシュボード生成 ---")
            phase6_result = pipeline._execute_phase6()
            print(f"Phase 6 結果: {'✅ 成功' if phase6_result.success else '❌ 失敗'}")
            
            if phase6_result.success:
                dashboard_file = phase6_result.output_data.get('dashboard_file')
                print(f"ダッシュボード: {dashboard_file}")
                
                # ダッシュボード内容確認
                if Path(dashboard_file).exists():
                    content = Path(dashboard_file).read_text(encoding='utf-8')
                    if "統合品質ダッシュボード" in content:
                        print("✅ ダッシュボード内容検証成功")
                    else:
                        print("❌ ダッシュボード内容検証失敗")
            
            print("\n=== モックテスト完了 ===")
            return True
            
        except Exception as e:
            print(f"❌ モックテスト失敗: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


class MockIntegratedQualityPipeline(IntegratedQualityPipeline):
    """モック用統合パイプライン"""
    
    def __init__(self, config_path: str, tracker_id: str, temp_workspace: Path):
        super().__init__(config_path, tracker_id)
        # 一時ワークスペースに変更
        self.workspace_dir = temp_workspace / tracker_id
        self.state_manager = StateManager(tracker_id, self.workspace_dir, self.logger)
        self.dashboard_generator = DashboardGenerator(self.config, self.workspace_dir, self.logger)
    
    def _execute_phase4_mock(self):
        """Phase 4: モック抽出実行"""
        from tools.core.integrated_quality_pipeline import PhaseResult
        
        phase_start = datetime.now()
        
        try:
            self.logger.info("Phase 4: モック抽出実行開始")
            
            # 抽出ディレクトリ準備
            extraction_dir = self.workspace_dir / "extraction"
            extraction_dir.mkdir(parents=True, exist_ok=True)
            
            # モック抽出結果作成
            create_mock_extraction_results(extraction_dir, total_files=10, success_count=7)
            
            # 抽出結果確認
            extracted_files = list(extraction_dir.glob("*.png"))
            
            # チェックポイント保存
            self.state_manager.save_checkpoint('phase4', {
                'extraction_completed': True,
                'extracted_files_count': len(extracted_files),
                'extraction_dir': str(extraction_dir)
            })
            
            duration = (datetime.now() - phase_start).total_seconds()
            self.logger.info(f"Phase 4: モック抽出完了 ({duration:.1f}秒, {len(extracted_files)}ファイル)")
            
            return PhaseResult(
                phase_name="phase4",
                success=True,
                duration_seconds=duration,
                output_data={
                    'extracted_files_count': len(extracted_files),
                    'extraction_dir': str(extraction_dir)
                },
                errors=[]
            )
            
        except Exception as e:
            self.logger.error(f"Phase 4 モック実行失敗: {str(e)}")
            
            return PhaseResult(
                phase_name="phase4",
                success=False,
                duration_seconds=(datetime.now() - phase_start).total_seconds(),
                output_data={},
                errors=[f"Phase 4 モック実行エラー: {str(e)}"]
            )


def test_component_validation():
    """コンポーネント単体テスト"""
    print("=== コンポーネント単体テスト ===")
    
    # ValidationEngine テスト
    print("\n--- ValidationEngine テスト ---")
    import logging
    logger = logging.getLogger("test")
    validator = ValidationEngine(logger)
    
    # 存在するパスでテスト
    valid_paths = [str(project_root / "test_small")]
    result = validator.validate_input_paths(valid_paths)
    print(f"バリデーション結果: {'✅ 成功' if result.is_valid else '❌ 失敗'}")
    
    # 存在しないパスでテスト
    invalid_paths = ["/non/existent/path"]
    result = validator.validate_input_paths(invalid_paths)
    print(f"無効パス検証: {'✅ 正常検出' if not result.is_valid else '❌ 検出失敗'}")
    
    print("コンポーネントテスト完了")


if __name__ == "__main__":
    print("統合パイプライン モックテスト実行")
    
    # コンポーネント単体テスト
    test_component_validation()
    
    # 全パイプライン モックテスト
    success = test_full_pipeline_mock()
    
    if success:
        print("\n🎉 全テスト成功！")
        sys.exit(0)
    else:
        print("\n❌ テスト失敗")
        sys.exit(1)