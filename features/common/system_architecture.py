#!/usr/bin/env python3
"""
システムアーキテクチャ分析レポート
PH2-002のためのボトルネック特定と最適化ポイント分析
"""

import sys
import os
import json
import psutil
import torch
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import subprocess
import importlib.util

class SystemArchitectureAnalyzer:
    """システムアーキテクチャ分析クラス"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'system_resources': {},
            'code_structure': {},
            'performance_metrics': {},
            'bottlenecks': [],
            'optimization_points': []
        }
    
    def analyze_system_resources(self) -> Dict[str, Any]:
        """システムリソース分析"""
        print("🔍 システムリソース分析開始...")
        
        # CPU情報
        cpu_info = {
            'count': psutil.cpu_count(),
            'physical_cores': psutil.cpu_count(logical=False),
            'usage_percent': psutil.cpu_percent(interval=1),
            'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else 0
        }
        
        # メモリ情報
        memory = psutil.virtual_memory()
        memory_info = {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent': memory.percent
        }
        
        # GPU情報
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                'available': True,
                'count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'device_name': torch.cuda.get_device_name(0),
                'memory_total_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3),
                'memory_allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                'memory_reserved_gb': torch.cuda.memory_reserved() / (1024**3)
            }
        else:
            gpu_info = {'available': False}
        
        # ディスクI/O
        disk_io = psutil.disk_io_counters()
        disk_info = {
            'read_mb': disk_io.read_bytes / (1024**2),
            'write_mb': disk_io.write_bytes / (1024**2),
            'read_count': disk_io.read_count,
            'write_count': disk_io.write_count
        }
        
        self.analysis_results['system_resources'] = {
            'cpu': cpu_info,
            'memory': memory_info,
            'gpu': gpu_info,
            'disk': disk_info
        }
        
        return self.analysis_results['system_resources']
    
    def analyze_code_structure(self) -> Dict[str, Any]:
        """コード構造分析"""
        print("📂 コード構造分析開始...")
        
        structure = {
            'total_files': 0,
            'total_lines': 0,
            'modules': {},
            'error_handling': {
                'files_with_try_except': 0,
                'generic_exceptions': 0,
                'specific_exceptions': 0
            },
            'memory_management': {
                'files_with_gc': 0,
                'files_with_cuda_cache': 0
            }
        }
        
        # Pythonファイルをスキャン
        for py_file in self.project_root.rglob("*.py"):
            if any(skip in str(py_file) for skip in ['deprecated', 'sam-env', '__pycache__']):
                continue
            
            structure['total_files'] += 1
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    structure['total_lines'] += len(lines)
                    
                    # エラー処理分析
                    if 'try:' in content:
                        structure['error_handling']['files_with_try_except'] += 1
                    
                    generic_count = content.count('except Exception') + content.count('except:')
                    specific_count = content.count('except ') - generic_count
                    
                    structure['error_handling']['generic_exceptions'] += generic_count
                    structure['error_handling']['specific_exceptions'] += specific_count
                    
                    # メモリ管理分析
                    if 'gc.collect()' in content:
                        structure['memory_management']['files_with_gc'] += 1
                    if 'torch.cuda.empty_cache()' in content:
                        structure['memory_management']['files_with_cuda_cache'] += 1
                    
                    # モジュール別統計
                    module_path = py_file.relative_to(self.project_root).parts[0]
                    if module_path not in structure['modules']:
                        structure['modules'][module_path] = {
                            'files': 0,
                            'lines': 0
                        }
                    structure['modules'][module_path]['files'] += 1
                    structure['modules'][module_path]['lines'] += len(lines)
                    
            except Exception as e:
                print(f"  ⚠️ ファイル読み取りエラー: {py_file} - {e}")
        
        self.analysis_results['code_structure'] = structure
        return structure
    
    def analyze_performance_bottlenecks(self) -> List[Dict[str, Any]]:
        """パフォーマンスボトルネック分析"""
        print("🔍 パフォーマンスボトルネック分析開始...")
        
        bottlenecks = []
        
        # 1. メモリリーク可能性の検出
        if self.analysis_results['code_structure']['memory_management']['files_with_gc'] < 10:
            bottlenecks.append({
                'type': 'memory_management',
                'severity': 'medium',
                'description': 'ガベージコレクション呼び出しが少ない',
                'impact': 'メモリリークの可能性',
                'recommendation': '定期的なgc.collect()の追加'
            })
        
        # 2. エラー処理の質
        error_ratio = (
            self.analysis_results['code_structure']['error_handling']['generic_exceptions'] /
            max(1, self.analysis_results['code_structure']['error_handling']['generic_exceptions'] + 
                self.analysis_results['code_structure']['error_handling']['specific_exceptions'])
        )
        if error_ratio > 0.7:
            bottlenecks.append({
                'type': 'error_handling',
                'severity': 'high',
                'description': '汎用的なException catchが多すぎる',
                'impact': 'エラーの特定と対処が困難',
                'recommendation': '具体的な例外クラスの使用'
            })
        
        # 3. GPU利用効率
        if self.analysis_results['system_resources']['gpu']['available']:
            gpu_usage = self.analysis_results['system_resources']['gpu']['memory_allocated_gb']
            gpu_total = self.analysis_results['system_resources']['gpu']['memory_total_gb']
            if gpu_usage / gpu_total < 0.1:
                bottlenecks.append({
                    'type': 'gpu_utilization',
                    'severity': 'low',
                    'description': 'GPU利用率が低い',
                    'impact': '処理速度の非効率',
                    'recommendation': 'バッチサイズの増加やGPU並列処理の実装'
                })
        
        # 4. ファイルI/O
        if self.analysis_results['system_resources']['disk']['read_count'] > 10000:
            bottlenecks.append({
                'type': 'disk_io',
                'severity': 'medium',
                'description': 'ディスクI/Oが多い',
                'impact': '処理速度の低下',
                'recommendation': 'キャッシュの実装やバッチ読み込み'
            })
        
        self.analysis_results['bottlenecks'] = bottlenecks
        return bottlenecks
    
    def identify_optimization_points(self) -> List[Dict[str, Any]]:
        """最適化ポイントの特定"""
        print("💡 最適化ポイント特定開始...")
        
        optimizations = []
        
        # 1. 並列処理の機会
        cpu_cores = self.analysis_results['system_resources']['cpu']['physical_cores']
        if cpu_cores > 4:
            optimizations.append({
                'area': 'parallel_processing',
                'priority': 'high',
                'description': f'{cpu_cores}コアCPUを活用した並列処理',
                'expected_improvement': '処理速度2-4倍',
                'implementation': 'multiprocessing.Pool or concurrent.futures'
            })
        
        # 2. GPU最適化
        if self.analysis_results['system_resources']['gpu']['available']:
            optimizations.append({
                'area': 'gpu_optimization',
                'priority': 'high',
                'description': 'GPU演算の最適化とバッチ処理',
                'expected_improvement': '処理速度5-10倍',
                'implementation': 'torch.cuda.amp, バッチサイズ最適化'
            })
        
        # 3. キャッシュ戦略
        optimizations.append({
            'area': 'caching',
            'priority': 'medium',
            'description': '計算結果のキャッシュ実装',
            'expected_improvement': '重複計算の削減',
            'implementation': 'functools.lru_cache, Redis'
        })
        
        # 4. エラーハンドリング改善
        optimizations.append({
            'area': 'error_handling',
            'priority': 'high',
            'description': '階層的エラーハンドリングシステム',
            'expected_improvement': 'システム安定性向上',
            'implementation': 'カスタム例外クラスとリトライ機構'
        })
        
        # 5. リソース管理
        optimizations.append({
            'area': 'resource_management',
            'priority': 'medium',
            'description': 'コンテキストマネージャーによるリソース管理',
            'expected_improvement': 'メモリリーク防止',
            'implementation': 'with文, contextlib'
        })
        
        self.analysis_results['optimization_points'] = optimizations
        return optimizations
    
    def generate_report(self) -> Dict[str, Any]:
        """分析レポート生成"""
        print("\n📊 システムアーキテクチャ分析レポート生成中...")
        
        # 各種分析実行
        self.analyze_system_resources()
        self.analyze_code_structure()
        self.analyze_performance_bottlenecks()
        self.identify_optimization_points()
        
        # サマリー生成
        self.analysis_results['summary'] = {
            'total_python_files': self.analysis_results['code_structure']['total_files'],
            'total_lines_of_code': self.analysis_results['code_structure']['total_lines'],
            'identified_bottlenecks': len(self.analysis_results['bottlenecks']),
            'optimization_opportunities': len(self.analysis_results['optimization_points']),
            'high_priority_items': sum(1 for opt in self.analysis_results['optimization_points'] 
                                     if opt['priority'] == 'high')
        }
        
        return self.analysis_results
    
    def save_report(self, output_path: str = None):
        """レポートを保存"""
        if output_path is None:
            output_path = self.project_root / 'architecture_analysis_report.json'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ レポート保存完了: {output_path}")
        
        # サマリー表示
        print("\n📋 分析サマリー:")
        summary = self.analysis_results['summary']
        print(f"  - Pythonファイル数: {summary['total_python_files']}")
        print(f"  - 総コード行数: {summary['total_lines_of_code']:,}")
        print(f"  - 特定されたボトルネック: {summary['identified_bottlenecks']}")
        print(f"  - 最適化機会: {summary['optimization_opportunities']}")
        print(f"  - 高優先度項目: {summary['high_priority_items']}")
        
        print("\n🚨 主要なボトルネック:")
        for bottleneck in self.analysis_results['bottlenecks'][:3]:
            print(f"  - [{bottleneck['severity']}] {bottleneck['description']}")
            print(f"    影響: {bottleneck['impact']}")
            print(f"    推奨: {bottleneck['recommendation']}")
        
        print("\n💡 最優先最適化ポイント:")
        for opt in [o for o in self.analysis_results['optimization_points'] if o['priority'] == 'high']:
            print(f"  - {opt['description']}")
            print(f"    期待効果: {opt['expected_improvement']}")


def main():
    """メイン処理"""
    analyzer = SystemArchitectureAnalyzer()
    analyzer.generate_report()
    analyzer.save_report()


if __name__ == "__main__":
    main()