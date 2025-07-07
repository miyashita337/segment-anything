#!/usr/bin/env python3
"""
画像評価結果統計分析スクリプト
評価ツールで生成されたJSONファイルを詳細分析
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any

class EvaluationAnalyzer:
    """評価結果分析クラス"""
    
    def __init__(self, json_path: str):
        """
        初期化
        
        Args:
            json_path: 評価結果JSONファイルパス
        """
        self.json_path = json_path
        self.data = None
        self.load_data()
    
    def load_data(self):
        """JSONデータを読み込み"""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"✅ データ読み込み完了: {len(self.data['evaluationData'])}件")
        except Exception as e:
            print(f"❌ データ読み込みエラー: {e}")
            sys.exit(1)
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """評価統計を取得"""
        folder1_ratings = Counter()
        folder2_ratings = Counter()
        evaluated_count = 0
        
        for item in self.data['evaluationData']:
            f1_rating = item.get('folder1_rating', '')
            f2_rating = item.get('folder2_rating', '')
            
            if f1_rating:
                folder1_ratings[f1_rating] += 1
            if f2_rating:
                folder2_ratings[f2_rating] += 1
                
            if f1_rating or f2_rating:
                evaluated_count += 1
        
        total_count = len(self.data['evaluationData'])
        
        return {
            'total_images': total_count,
            'evaluated_images': evaluated_count,
            'unevaluated_images': total_count - evaluated_count,
            'folder1_ratings': dict(folder1_ratings),
            'folder2_ratings': dict(folder2_ratings)
        }
    
    def analyze_rating_changes(self) -> List[Dict[str, Any]]:
        """評価変化を分析"""
        rating_values = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 1, 'F': 0}
        changes = []
        
        for item in self.data['evaluationData']:
            f1_rating = item.get('folder1_rating', '')
            f2_rating = item.get('folder2_rating', '')
            
            if f1_rating and f2_rating:
                f1_value = rating_values.get(f1_rating, -1)
                f2_value = rating_values.get(f2_rating, -1)
                
                if f1_value >= 0 and f2_value >= 0:
                    change = f2_value - f1_value
                    changes.append({
                        'filename': item['filename'],
                        'folder1_rating': f1_rating,
                        'folder2_rating': f2_rating,
                        'change_value': change,
                        'change_type': 'improvement' if change > 0 else 'degradation' if change < 0 else 'same',
                        'folder1_issues': item.get('folder1_issues_list', []),
                        'folder2_issues': item.get('folder2_issues_list', []),
                        'notes': item.get('notes', '')
                    })
        
        return sorted(changes, key=lambda x: x['change_value'], reverse=True)
    
    def analyze_issue_patterns(self) -> Dict[str, Any]:
        """問題パターンを分析"""
        folder1_issues = Counter()
        folder2_issues = Counter()
        
        for item in self.data['evaluationData']:
            for issue in item.get('folder1_issues_list', []):
                folder1_issues[issue] += 1
            for issue in item.get('folder2_issues_list', []):
                folder2_issues[issue] += 1
        
        # 問題減少/増加分析
        all_issues = set(folder1_issues.keys()) | set(folder2_issues.keys())
        issue_changes = {}
        
        for issue in all_issues:
            f1_count = folder1_issues.get(issue, 0)
            f2_count = folder2_issues.get(issue, 0)
            issue_changes[issue] = {
                'folder1_count': f1_count,
                'folder2_count': f2_count,
                'change': f2_count - f1_count,
                'improvement_rate': ((f1_count - f2_count) / f1_count * 100) if f1_count > 0 else 0
            }
        
        return {
            'folder1_issues': dict(folder1_issues),
            'folder2_issues': dict(folder2_issues),
            'issue_changes': issue_changes
        }
    
    def calculate_success_rates(self) -> Dict[str, Any]:
        """成功率を計算（A-C を成功とみなす）"""
        success_ratings = {'A', 'B', 'C'}
        
        folder1_success = 0
        folder2_success = 0
        folder1_total = 0
        folder2_total = 0
        
        for item in self.data['evaluationData']:
            f1_rating = item.get('folder1_rating', '')
            f2_rating = item.get('folder2_rating', '')
            
            if f1_rating:
                folder1_total += 1
                if f1_rating in success_ratings:
                    folder1_success += 1
            
            if f2_rating:
                folder2_total += 1
                if f2_rating in success_ratings:
                    folder2_success += 1
        
        return {
            'folder1_success_rate': (folder1_success / folder1_total * 100) if folder1_total > 0 else 0,
            'folder2_success_rate': (folder2_success / folder2_total * 100) if folder2_total > 0 else 0,
            'folder1_success_count': folder1_success,
            'folder2_success_count': folder2_success,
            'folder1_total': folder1_total,
            'folder2_total': folder2_total
        }
    
    def generate_detailed_report(self) -> str:
        """詳細レポートを生成"""
        stats = self.get_evaluation_stats()
        changes = self.analyze_rating_changes()
        issues = self.analyze_issue_patterns()
        success_rates = self.calculate_success_rates()
        
        report = []
        report.append("# 📊 画像評価結果詳細分析レポート")
        report.append("")
        report.append(f"**分析日時**: {self.data.get('timestamp', 'N/A')}")
        report.append(f"**フォルダー1**: {self.data.get('folder1', 'N/A')}")
        report.append(f"**フォルダー2**: {self.data.get('folder2', 'N/A')}")
        report.append("")
        
        # 全体統計
        report.append("## 📈 全体統計")
        report.append("")
        report.append(f"- **総画像数**: {stats['total_images']}枚")
        report.append(f"- **評価済み**: {stats['evaluated_images']}枚")
        report.append(f"- **未評価**: {stats['unevaluated_images']}枚")
        report.append("")
        
        # 成功率比較
        report.append("## 🎯 成功率比較 (A-C評価)")
        report.append("")
        report.append(f"| バージョン | 成功数 | 総数 | 成功率 |")
        report.append(f"|-----------|--------|------|--------|")
        report.append(f"| フォルダー1 | {success_rates['folder1_success_count']} | {success_rates['folder1_total']} | {success_rates['folder1_success_rate']:.1f}% |")
        report.append(f"| フォルダー2 | {success_rates['folder2_success_count']} | {success_rates['folder2_total']} | {success_rates['folder2_success_rate']:.1f}% |")
        report.append("")
        
        improvement = success_rates['folder2_success_rate'] - success_rates['folder1_success_rate']
        if improvement > 0:
            report.append(f"**✅ フォルダー2が{improvement:.1f}ポイント改善**")
        elif improvement < 0:
            report.append(f"**⚠️ フォルダー2が{abs(improvement):.1f}ポイント悪化**")
        else:
            report.append("**➡️ 両フォルダー同等の性能**")
        report.append("")
        
        # 評価分布
        report.append("## 📊 評価分布")
        report.append("")
        report.append("| 評価 | フォルダー1 | フォルダー2 | 差分 |")
        report.append("|------|-------------|-------------|------|")
        
        for rating in ['A', 'B', 'C', 'D', 'E', 'F']:
            f1_count = stats['folder1_ratings'].get(rating, 0)
            f2_count = stats['folder2_ratings'].get(rating, 0)
            diff = f2_count - f1_count
            diff_str = f"+{diff}" if diff > 0 else str(diff) if diff < 0 else "0"
            report.append(f"| {rating} | {f1_count} | {f2_count} | {diff_str} |")
        report.append("")
        
        # 大幅改善ケース
        improvements = [c for c in changes if c['change_value'] >= 2]
        if improvements:
            report.append("## 🏆 大幅改善ケース (2段階以上)")
            report.append("")
            for item in improvements:
                report.append(f"### {item['filename']}")
                report.append(f"- **改善**: {item['folder1_rating']} → {item['folder2_rating']} (+{item['change_value']}段階)")
                if item['folder1_issues']:
                    report.append(f"- **フォルダー1問題**: {', '.join(item['folder1_issues'])}")
                if item['folder2_issues']:
                    report.append(f"- **フォルダー2問題**: {', '.join(item['folder2_issues'])}")
                if item['notes'].strip():
                    report.append(f"- **メモ**: {item['notes'].strip()}")
                report.append("")
        
        # 悪化ケース
        degradations = [c for c in changes if c['change_value'] <= -2]
        if degradations:
            report.append("## ⚠️ 大幅悪化ケース (2段階以上)")
            report.append("")
            for item in degradations:
                report.append(f"### {item['filename']}")
                report.append(f"- **悪化**: {item['folder1_rating']} → {item['folder2_rating']} ({item['change_value']}段階)")
                if item['folder1_issues']:
                    report.append(f"- **フォルダー1問題**: {', '.join(item['folder1_issues'])}")
                if item['folder2_issues']:
                    report.append(f"- **フォルダー2問題**: {', '.join(item['folder2_issues'])}")
                if item['notes'].strip():
                    report.append(f"- **メモ**: {item['notes'].strip()}")
                report.append("")
        
        # 問題パターン分析
        report.append("## 🔍 問題パターン分析")
        report.append("")
        report.append("| 問題 | フォルダー1 | フォルダー2 | 変化 | 改善率 |")
        report.append("|------|-------------|-------------|------|--------|")
        
        for issue, data in sorted(issues['issue_changes'].items(), 
                                key=lambda x: x[1]['improvement_rate'], reverse=True):
            f1_count = data['folder1_count']
            f2_count = data['folder2_count']
            change = data['change']
            improvement_rate = data['improvement_rate']
            
            change_str = f"+{change}" if change > 0 else str(change) if change < 0 else "0"
            improvement_str = f"{improvement_rate:.1f}%" if f1_count > 0 else "N/A"
            
            report.append(f"| {issue} | {f1_count} | {f2_count} | {change_str} | {improvement_str} |")
        report.append("")
        
        # ユーザーメモサマリー
        notes_with_content = [item for item in self.data['evaluationData'] 
                            if item.get('notes', '').strip()]
        if notes_with_content:
            report.append("## 📝 ユーザーメモサマリー")
            report.append("")
            for item in notes_with_content:
                report.append(f"### {item['filename']}")
                report.append(f"- **評価**: フォルダー1={item.get('folder1_rating', 'N/A')}, フォルダー2={item.get('folder2_rating', 'N/A')}")
                report.append(f"- **メモ**: {item['notes'].strip()}")
                report.append("")
        
        return "\n".join(report)
    
    def export_csv_summary(self, output_path: str):
        """CSV形式で統計サマリーを出力"""
        import csv
        
        changes = self.analyze_rating_changes()
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # ヘッダー
            writer.writerow([
                'ファイル名', 'フォルダー1評価', 'フォルダー2評価', '変化値', '変化タイプ',
                'フォルダー1問題数', 'フォルダー2問題数', 'フォルダー1問題', 'フォルダー2問題', 'メモ'
            ])
            
            # データ行
            for item in changes:
                writer.writerow([
                    item['filename'],
                    item['folder1_rating'],
                    item['folder2_rating'],
                    item['change_value'],
                    item['change_type'],
                    len(item['folder1_issues']),
                    len(item['folder2_issues']),
                    '; '.join(item['folder1_issues']),
                    '; '.join(item['folder2_issues']),
                    item['notes'].replace('\n', ' ').strip()
                ])
        
        print(f"✅ CSV出力完了: {output_path}")


def main():
    """メイン関数"""
    if len(sys.argv) < 2:
        print("使用方法: python analyze_results.py <evaluation_progress.json>")
        sys.exit(1)
    
    json_path = sys.argv[1]
    
    if not Path(json_path).exists():
        print(f"❌ ファイルが見つかりません: {json_path}")
        sys.exit(1)
    
    print(f"🔍 評価結果分析開始: {json_path}")
    
    # 分析実行
    analyzer = EvaluationAnalyzer(json_path)
    
    # 詳細レポート生成
    report = analyzer.generate_detailed_report()
    
    # レポート出力
    output_dir = Path(json_path).parent
    report_path = output_dir / "detailed_analysis_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ 詳細レポート出力: {report_path}")
    
    # CSV出力
    csv_path = output_dir / "evaluation_summary.csv"
    analyzer.export_csv_summary(str(csv_path))
    
    # 統計サマリー表示
    stats = analyzer.get_evaluation_stats()
    success_rates = analyzer.calculate_success_rates()
    
    print("\n📊 分析結果サマリー:")
    print(f"  総画像数: {stats['total_images']}")
    print(f"  評価済み: {stats['evaluated_images']}")
    print(f"  フォルダー1成功率: {success_rates['folder1_success_rate']:.1f}%")
    print(f"  フォルダー2成功率: {success_rates['folder2_success_rate']:.1f}%")
    
    improvement = success_rates['folder2_success_rate'] - success_rates['folder1_success_rate']
    print(f"  性能差: {improvement:+.1f}ポイント")


if __name__ == "__main__":
    main()