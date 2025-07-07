#!/usr/bin/env python3
"""
Phase 4 results analysis script
"""
import json
from collections import Counter

# 評価データを読み込み
with open('/mnt/c/Users/shakufuku/Downloads/evaluation_progress.json', 'r', encoding='utf-8') as f:
    eval_data = json.load(f)

# 0.0.3バージョンの成績分析
ratings_003 = []
issues_003 = []

for item in eval_data['evaluationData']:
    if item['folder2_rating'] and item['folder2_rating'] != '':
        ratings_003.append(item['folder2_rating'])
        issues_003.extend(item['folder2_issues_list'])

# 成績集計
grade_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0}
for rating in ratings_003:
    if rating in grade_counts:
        grade_counts[rating] += 1

total_rated = sum(grade_counts.values())
success_rate_003 = (grade_counts['A'] + grade_counts['B'] + grade_counts['C']) / total_rated if total_rated > 0 else 0

print('Phase 0.0.3 (従来版) 成績:')
print('  A評価: {} ({:.1f}%)'.format(grade_counts['A'], grade_counts['A']/total_rated*100))
print('  B評価: {} ({:.1f}%)'.format(grade_counts['B'], grade_counts['B']/total_rated*100))
print('  C評価: {} ({:.1f}%)'.format(grade_counts['C'], grade_counts['C']/total_rated*100))
print('  D評価: {} ({:.1f}%)'.format(grade_counts['D'], grade_counts['D']/total_rated*100))
print('  E評価: {} ({:.1f}%)'.format(grade_counts['E'], grade_counts['E']/total_rated*100))
print('  F評価: {} ({:.1f}%)'.format(grade_counts['F'], grade_counts['F']/total_rated*100))
print('  成功率(A-C): {:.1f}%'.format(success_rate_003*100))

# 主要問題集計
issue_counts = Counter(issues_003)
print('\n主要問題:')
for issue, count in issue_counts.most_common():
    print('  {}: {}件'.format(issue, count))

print('\nPhase 4で生成されたファイル数: 24個')
print('評価システムでPhase 4結果を比較する準備が完了しました。')
print('次のURLでPhase 0.0.3 vs 0.0.4 の比較評価を実行してください:')
print('http://127.0.0.1:3000/')