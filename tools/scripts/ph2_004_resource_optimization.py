#!/usr/bin/env python3
"""
PH2-004-RESOURCE: リソース管理最適化システム - 実行スクリプト

【概要】
GPU・CPU・メモリ・ディスクリソースの統合監視・最適化システム

【実行モード】
1. monitor: リアルタイム監視モード
2. check: 現在状況確認モード  
3. optimize: 手動最適化実行モード
4. report: レポート生成モード

【使用方法】
python tools/scripts/ph2_004_resource_optimization.py monitor --duration 300
python tools/scripts/ph2_004_resource_optimization.py check
python tools/scripts/ph2_004_resource_optimization.py optimize
python tools/scripts/ph2_004_resource_optimization.py report
"""

import sys
import argparse
import time
import json
from pathlib import Path

# プロジェクトルート追加
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from features.common.resource_monitor import ResourceMonitor

def monitor_mode(args):
    """監視モード実行"""
    print("🔧 PH2-004-RESOURCE: リソース監視開始")
    
    monitor = ResourceMonitor(
        monitoring_interval=args.interval,
        history_size=args.history_size
    )
    
    # アラートコールバック設定
    def alert_callback(alert):
        severity_emoji = "⚠️" if alert['severity'] == 'warning' else "🚨"
        print(f"{severity_emoji} {alert['message']}")
        
        # 重要アラートの場合、追加情報表示
        if alert['severity'] == 'critical':
            print(f"   現在値: {alert['value']}, 閾値: {alert['threshold']}")
    
    monitor.add_alert_callback(alert_callback)
    
    try:
        monitor.start_monitoring()
        print(f"📊 {args.duration}秒間の監視を開始します...")
        print("   Ctrl+Cで停止")
        
        # 定期的な統計表示
        start_time = time.time()
        while time.time() - start_time < args.duration:
            time.sleep(30)  # 30秒ごとに統計表示
            stats = monitor.get_statistics()
            if stats:
                print(f"📈 監視統計: 最適化{stats.get('optimization_count', 0)}回, "
                      f"アラート{stats.get('alert_count', 0)}回")
    
    except KeyboardInterrupt:
        print("\n⏹️ 監視を停止しています...")
    
    finally:
        monitor.stop_monitoring()
        
        # 最終統計表示
        stats = monitor.get_statistics()
        print("\n📊 監視完了レポート:")
        print(f"  監視時間: {stats.get('monitoring_duration_minutes', 0):.1f}分")
        print(f"  データポイント: {stats.get('data_points', 0)}")
        print(f"  最適化実行: {stats.get('optimization_count', 0)}回")
        print(f"  アラート発生: {stats.get('alert_count', 0)}回")
        
        # 平均値表示
        averages = stats.get('averages', {})
        if averages:
            print(f"  平均CPU使用率: {averages.get('cpu_percent', 0):.1f}%")
            print(f"  平均メモリ使用率: {averages.get('memory_percent', 0):.1f}%")

def check_mode(args):
    """現在状況確認モード"""
    print("🔍 PH2-004-RESOURCE: 現在のリソース状況確認")
    
    monitor = ResourceMonitor()
    status = monitor.get_current_status()
    
    print("\n📊 現在のリソース状況:")
    print(f"  🖥️ CPU使用率: {status.cpu_percent:.1f}%")
    print(f"  🧠 メモリ使用率: {status.memory_percent:.1f}% (利用可能: {status.memory_available_gb:.1f}GB)")
    print(f"  💾 ディスク使用率: {status.disk_percent:.1f}% (空き: {status.disk_free_gb:.1f}GB)")
    print(f"  📊 システム負荷: {status.load_average}")
    print(f"  🔄 アクティブプロセス: {status.active_processes}")
    print(f"  🐍 Pythonプロセス: {status.python_processes}")
    
    # GPU情報
    if status.gpu_count > 0:
        print(f"\n🎮 GPU情報 ({status.gpu_count}台):")
        for i in range(status.gpu_count):
            if i < len(status.gpu_memory_used):
                memory_used = status.gpu_memory_used[i]
                memory_total = status.gpu_memory_total[i]
                memory_percent = (memory_used / memory_total * 100) if memory_total > 0 else 0
                temp = status.gpu_temperature[i] if i < len(status.gpu_temperature) else 0
                util = status.gpu_utilization[i] if i < len(status.gpu_utilization) else 0
                
                print(f"  GPU{i}: {memory_percent:.1f}% ({memory_used:.1f}/{memory_total:.1f}GB), "
                      f"{temp}°C, {util}%使用率")
    else:
        print("\n🎮 GPU: 検出されませんでした")
    
    # アラートチェック
    alerts = monitor.check_alerts(status)
    if alerts:
        print(f"\n⚠️ アラート ({len(alerts)}件):")
        for alert in alerts:
            severity_emoji = "⚠️" if alert['severity'] == 'warning' else "🚨"
            print(f"  {severity_emoji} {alert['message']}")
    else:
        print("\n✅ 現在、アラートはありません")

def optimize_mode(args):
    """手動最適化モード"""
    print("🔧 PH2-004-RESOURCE: 手動最適化実行")
    
    monitor = ResourceMonitor()
    status = monitor.get_current_status()
    alerts = monitor.check_alerts(status)
    
    print(f"📊 最適化前状況:")
    print(f"  CPU: {status.cpu_percent:.1f}%, メモリ: {status.memory_percent:.1f}%")
    
    if alerts:
        print(f"  検出アラート: {len(alerts)}件")
        for alert in alerts:
            print(f"    - {alert['message']}")
    
    # 最適化実行
    actions = monitor.auto_optimize(status, alerts)
    
    if actions:
        print(f"\n✅ 最適化実行完了 ({len(actions)}件):")
        for action in actions:
            print(f"  🔧 {action}")
        
        # 最適化後の状況確認
        print("\n📊 最適化後状況確認...")
        time.sleep(2)  # 少し待ってから確認
        status_after = monitor.get_current_status()
        print(f"  CPU: {status_after.cpu_percent:.1f}%, メモリ: {status_after.memory_percent:.1f}%")
        
        # 改善度計算
        cpu_improvement = status.cpu_percent - status_after.cpu_percent
        memory_improvement = status.memory_percent - status_after.memory_percent
        
        if cpu_improvement > 0 or memory_improvement > 0:
            print(f"📈 改善度:")
            if cpu_improvement > 0:
                print(f"  CPU: {cpu_improvement:.1f}%改善")
            if memory_improvement > 0:
                print(f"  メモリ: {memory_improvement:.1f}%改善")
        else:
            print("📊 大きな改善は検出されませんでした（正常範囲内）")
    else:
        print("✅ 最適化の必要はありません（リソース使用状況は正常範囲内）")

def report_mode(args):
    """レポート生成モード"""
    print("📄 PH2-004-RESOURCE: レポート生成")
    
    monitor = ResourceMonitor()
    
    # 短時間監視してサンプルデータ取得
    print("📊 サンプルデータ収集中...")
    monitor.start_monitoring()
    time.sleep(30)  # 30秒間監視
    monitor.stop_monitoring()
    
    # レポート出力
    output_path = None
    if args.output:
        output_path = Path(args.output)
    
    report_path = monitor.export_report(output_path)
    
    # 統計サマリー表示
    stats = monitor.get_statistics()
    print(f"\n📊 レポートサマリー:")
    print(f"  データポイント: {stats.get('data_points', 0)}")
    print(f"  最適化実行: {stats.get('optimization_count', 0)}回")
    print(f"  アラート発生: {stats.get('alert_count', 0)}回")
    
    if stats.get('current_status'):
        current = stats['current_status']
        print(f"  現在のCPU使用率: {current['cpu_percent']:.1f}%")
        print(f"  現在のメモリ使用率: {current['memory_percent']:.1f}%")
    
    print(f"\n📄 レポートファイル: {report_path}")
    print(f"🌐 ダッシュボード: http://100.123.241.106:8088/resource-monitor/")

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(
        description="PH2-004-RESOURCE: リソース管理最適化システム"
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='実行モード')
    
    # 監視モード
    monitor_parser = subparsers.add_parser('monitor', help='リアルタイム監視')
    monitor_parser.add_argument('--duration', type=int, default=300, 
                               help='監視時間（秒、デフォルト: 300）')
    monitor_parser.add_argument('--interval', type=float, default=5.0,
                               help='監視間隔（秒、デフォルト: 5.0）')
    monitor_parser.add_argument('--history-size', type=int, default=100,
                               help='履歴保持数（デフォルト: 100）')
    
    # 確認モード
    check_parser = subparsers.add_parser('check', help='現在状況確認')
    
    # 最適化モード
    optimize_parser = subparsers.add_parser('optimize', help='手動最適化実行')
    
    # レポートモード
    report_parser = subparsers.add_parser('report', help='レポート生成')
    report_parser.add_argument('--output', type=str, help='出力ファイルパス')
    
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        return
    
    try:
        if args.mode == 'monitor':
            monitor_mode(args)
        elif args.mode == 'check':
            check_mode(args)
        elif args.mode == 'optimize':
            optimize_mode(args)
        elif args.mode == 'report':
            report_mode(args)
        else:
            print(f"❌ 不明なモード: {args.mode}")
            parser.print_help()
    
    except Exception as e:
        print(f"❌ 実行エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()