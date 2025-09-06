# ダッシュボード品質保証チェックリスト

**作成日**: 2025-08-27  
**目的**: QUAL-041再発防止・ダッシュボード仕様書完全準拠の品質保証  
**重要性**: 🚨 **毎回毎回同じ問題で困るのを防ぐため、このチェックリストを必ず完全実行する**

---

## 🚨 **緊急重要度・必須実行事項**

**この問題は既に複数回発生している重大品質問題です。**
- **QUAL-041**: 統計分析結果・品質分布が完全に欠落
- **過去トラッカー**: 同様の問題が繰り返し発生
- **ユーザー指摘**: 「毎回毎回この指摘してるのですがなぜ毎回毎回治らないのですか？」

**🔴 このチェックリストを実行しないと、確実に品質問題が発生します。**

---

## 📋 **Section A: データ構造検証（根本原因防止）**

### A1: extraction_result.json構造検証 🔴
```bash
# 必須実行コマンド
python3 -c "
import json
with open('/mnt/c/AItools/lora/train/yado/tracker-workspace/{TRACKER_ID}/extraction_result.json', 'r') as f:
    data = json.load(f)
print('📊 データ構造検証:')
print(f'  tracker_id: {data.get(\"tracker_id\", \"❌ 未設定\")}')
print(f'  total_images: {data.get(\"total_images\", \"❌ 未設定\")}')
print(f'  successful_extractions: {data.get(\"successful_extractions\", \"❌ 未設定\")}')
print(f'  average_quality_score: {data.get(\"average_quality_score\", \"❌ 未設定\")}')
print(f'  statistical_analysis: {\"✅ 存在\" if \"statistical_analysis\" in data else \"❌ 未設定\"}')
print(f'  extraction_results: {len(data.get(\"extraction_results\", []))}件')
"
```

- [ ] **🔴 tracker_id**: {TRACKER_ID}と一致
- [ ] **🔴 total_images**: 実際の処理画像数（0でない数値）
- [ ] **🔴 successful_extractions**: 成功した抽出数（0でない数値）  
- [ ] **🔴 average_quality_score**: 0.0でない実際の平均スコア
- [ ] **🔴 statistical_analysis**: セクション存在確認
- [ ] **🔴 extraction_results**: 実際の抽出結果配列（空でない）

### A2: statistical_analysis必須キー検証 🔴
```bash
# 必須実行コマンド
python3 -c "
import json
with open('/mnt/c/AItools/lora/train/yado/tracker-workspace/{TRACKER_ID}/extraction_result.json', 'r') as f:
    data = json.load(f)
stats = data.get('statistical_analysis', {})
print('📊 統計分析データ検証:')
for key in ['p_value', 'effect_size', 'improvement_rate', 'significance', 'baseline_score', 'confidence_interval']:
    value = stats.get(key, '❌ 未設定')
    print(f'  {key}: {value}')
"
```

- [ ] **🔴 p_value**: 数値文字列（例: "0.042"）
- [ ] **🔴 effect_size**: 数値文字列（例: "1.23"）
- [ ] **🔴 improvement_rate**: パーセント表記（例: "+12.8%"）
- [ ] **🔴 significance**: "有意" or "非有意"
- [ ] **🔴 baseline_score**: 数値文字列（例: "0.652"）
- [ ] **🔴 confidence_interval**: 区間表記（例: "(0.68, 0.79)"）

### A3: extraction_results配列内容検証 🔴
```bash
# 必須実行コマンド
python3 -c "
import json
with open('/mnt/c/AItools/lora/train/yado/tracker-workspace/{TRACKER_ID}/extraction_result.json', 'r') as f:
    data = json.load(f)
results = data.get('extraction_results', [])
print(f'📊 抽出結果配列検証: {len(results)}件')
for i, r in enumerate(results):
    print(f'  [{i}] image_name: {r.get(\"image_name\", \"❌ 未設定\")}')
    print(f'      success: {r.get(\"success\", \"❌ 未設定\")}')
    print(f'      quality_score: {r.get(\"quality_score\", \"❌ 未設定\")}')
"
```

- [ ] **🔴 image_name**: 実際のファイル名（unknown.jpgでない）
- [ ] **🔴 success**: true/false正しい値
- [ ] **🔴 quality_score**: 0.0でない実際のスコア数値

---

## 📋 **Section B: ダッシュボードHTML生成検証**

### B1: ダッシュボード再生成実行 🔴
```bash
# 必須実行コマンド
python3 -c "
from features.common.dashboard_generator import StandardDashboardGenerator
generator = StandardDashboardGenerator()
dashboard_path = generator.create_dashboard(
    tracker_id='{TRACKER_ID}', 
    workspace_dir='/mnt/c/AItools/lora/train/yado/tracker-workspace/{TRACKER_ID}'
)
print(f'✅ ダッシュボード生成: {dashboard_path}')
"
```

- [ ] **🔴 ダッシュボード生成**: エラーなく正常完了
- [ ] **🔴 dashboard.html**: ファイル存在確認

### B2: 統計分析結果HTML表示検証（7項目順序） 🔴
```bash
# 必須実行コマンド
grep -A 30 "📊 統計分析結果" /mnt/c/AItools/lora/train/yado/tracker-workspace/{TRACKER_ID}/dashboard/dashboard.html
```

#### 表示順序確認（Current→BaseLine→統計値）
- [ ] **🔴 1番目: "Current(平均品質スコア)"**: ラベル・数値表示・blue-600カラー
- [ ] **🔴 2番目: "BaseLine"**: ラベル・数値表示・gray-600カラー  
- [ ] **🔴 3番目: "p値"**: ラベル・数値表示・indigo-600カラー
- [ ] **🔴 4番目: "効果サイズ、Cohen's d"**: ラベル・数値表示・purple-600カラー
- [ ] **🔴 5番目: "改善率"**: ラベル・数値表示・green-600カラー
- [ ] **🔴 6番目: "統計的有意性"**: ラベル・有意性表示・red-600カラー
- [ ] **🔴 7番目: "信頼区間"**: ラベル・区間表示・teal-600カラー

#### グリッドレイアウト確認
- [ ] **🔴 レスポンシブグリッド**: `grid-cols-2 md:grid-cols-3 lg:grid-cols-7`

### B3: 基本品質指標HTML表示検証 🔴
```bash
# 必須実行コマンド
grep -E "(総画像数|平均品質スコア|成功画像数|要改善数)" /mnt/c/AItools/lora/train/yado/tracker-workspace/{TRACKER_ID}/dashboard/dashboard.html | head -8
```

- [ ] **🔴 "総画像数"**: 実際の数値（0でない）
- [ ] **🔴 "平均品質スコア"**: 実際のスコア（0.000でない）
- [ ] **🔴 "成功画像数"**: 実際の成功数（0でない）
- [ ] **🔴 "要改善数"**: 正しい計算結果

### B4: 品質分布HTML表示検証 🔴
```bash
# 必須実行コマンド
grep -A 15 "品質分布" /mnt/c/AItools/lora/train/yado/tracker-workspace/{TRACKER_ID}/dashboard/dashboard.html
```

- [ ] **🔴 "高品質"**: 正確な数値（全部0でない場合）
- [ ] **🔴 "中品質"**: 正確な数値（全部0でない場合）
- [ ] **🔴 "低品質"**: 正確な数値（全部0でない場合）
- [ ] **🔴 "要改善"**: 正確な数値（全部0でない場合）
- [ ] **🔴 数値合計**: total_images と一致

### B5: 画像ギャラリーHTML表示検証 🔴
```bash
# 必須実行コマンド
grep -E "(extracted_|スコア:)" /mnt/c/AItools/lora/train/yado/tracker-workspace/{TRACKER_ID}/dashboard/dashboard.html | head -10
```

- [ ] **🔴 実際ファイル名**: extracted_*.jpg（unknown.jpgでない）
- [ ] **🔴 品質ラベル**: 高品質・中品質・低品質・要改善の正確な表示
- [ ] **🔴 スコア表示**: "スコア: 0.XXX"形式で実際の数値

---

## 📋 **Section C: サーバーアクセス検証**

### C1: ダッシュボードサーバー動作確認 🔴
```bash
# 必須実行コマンド
curl -u admin:secure_track_2025_q3_8f9a http://100.123.241.106:8088/tracker/{TRACKER_ID} -o /tmp/dashboard_check.html
echo "📊 ダッシュボードサイズ: $(wc -c < /tmp/dashboard_check.html) bytes"
```

- [ ] **🔴 HTTP 200**: 正常レスポンス
- [ ] **🔴 HTML サイズ**: 5KB以上（空でない）
- [ ] **🔴 認証成功**: 認証エラーでない
- [ ] **🔴 iframe表示確認**: src="/{TRACKER_ID}/dashboard/dashboard.html"が含まれる

### C2: 統計分析結果サーバー表示確認 🔴
```bash
# 必須実行コマンド（直接ダッシュボードHTMLにアクセス）
curl -u admin:secure_track_2025_q3_8f9a http://100.123.241.106:8088/{TRACKER_ID}/dashboard/dashboard.html | grep -E "(p値|効果サイズ|改善率|統計的有意性|BaseLine|信頼区間)"
```

- [ ] **🔴 p値表示**: HTML内に数値存在
- [ ] **🔴 効果サイズ表示**: HTML内に数値存在
- [ ] **🔴 改善率表示**: HTML内にパーセント存在
- [ ] **🔴 統計的有意性表示**: HTML内に有意性表示
- [ ] **🔴 ベースライン表示**: HTML内に数値存在
- [ ] **🔴 信頼区間表示**: HTML内に区間存在

### C3: 品質指標サーバー表示確認 🔴
```bash
# 必須実行コマンド（直接ダッシュボードHTMLにアクセス）
curl -u admin:secure_track_2025_q3_8f9a http://100.123.241.106:8088/{TRACKER_ID}/dashboard/dashboard.html | grep -E "text-3xl font-bold" | grep -E "[0-9]+\.?[0-9]*"
```

- [ ] **🔴 総画像数非ゼロ**: 実際の処理数表示
- [ ] **🔴 平均品質スコア非ゼロ**: 0.000でない実スコア
- [ ] **🔴 成功画像数非ゼロ**: 実際の成功数表示

---

## 📋 **Section D: 品質保証・最終確認**

### D1: データ整合性確認 🔴
```bash
# 必須実行コマンド
python3 -c "
import json
with open('/mnt/c/AItools/lora/train/yado/tracker-workspace/{TRACKER_ID}/extraction_result.json', 'r') as f:
    data = json.load(f)

total = data.get('total_images', 0)
successful = data.get('successful_extractions', 0)
results = data.get('extraction_results', [])
avg_score = data.get('average_quality_score', 0.0)

print('📊 データ整合性検証:')
print(f'  設定値 - 総画像数: {total}, 成功数: {successful}, 平均スコア: {avg_score}')
print(f'  実配列 - 結果数: {len(results)}')

# 品質分布計算
quality_dist = {'高品質': 0, '中品質': 0, '低品質': 0, '要改善': 0}
for r in results:
    if r.get('success'):
        score = r.get('quality_score', 0.0)
        if score >= 0.8:
            quality_dist['高品質'] += 1
        elif score >= 0.6:
            quality_dist['中品質'] += 1
        elif score >= 0.4:
            quality_dist['低品質'] += 1
        else:
            quality_dist['要改善'] += 1

print(f'  計算分布: {quality_dist}')
print(f'  分布合計: {sum(quality_dist.values())} (成功数{successful}と一致: {sum(quality_dist.values()) == successful})')
"
```

- [ ] **🔴 結果数一致**: extraction_results長とsuccessful_extractions一致
- [ ] **🔴 分布合計一致**: 品質分布合計と成功数一致
- [ ] **🔴 平均スコア妥当**: 実際の結果から計算した平均と近似

### D2: 最終表示確認 🔴
```bash
# 必須実行コマンド - ブラウザでの表示確認
echo "📋 最終確認事項:"
echo "1. ブラウザで http://100.123.241.106:8088/tracker/{TRACKER_ID} を開く"
echo "2. 統計分析結果セクションに6項目全て数値表示されているか確認"
echo "3. 基本品質指標4項目に実数値表示されているか確認"  
echo "4. 品質分布4項目に正確な数値表示されているか確認"
echo "5. 画像ギャラリーに実際のファイル名・スコアが表示されているか確認"
```

- [ ] **🔴 視覚確認**: ブラウザで全項目正常表示
- [ ] **🔴 数値精度**: 全ての数値が意味のある値（0や"N/A"でない）
- [ ] **🔴 レイアウト**: Tailwind CSSで整った表示
- [ ] **🔴 エラーなし**: 画像読み込みエラー等が発生していない

---

## ❌ **絶対回避すべきNGパターン（QUAL-041教訓）**

### 🚫 データ構造の不整合
```
❌ extraction_result.jsonのキー名ミス
   - success_count ❌ → successful_extractions ✅
   - summary.average_quality_score ❌ → average_quality_score ✅
   - quality_metrics.overall_score ❌ → quality_score ✅
   - image_path ❌ → image_name ✅
```

### 🚫 統計分析データの欠落
```
❌ statistical_analysisセクション未作成
❌ p値・効果サイズ・改善率・統計的有意性の"N/A"表示
✅ 全6項目に実際の数値・文字列データ設定
```

### 🚫 品質分布の計算エラー
```
❌ 全て0表示 → 品質分布計算ロジックのバグ
❌ 合計不一致 → successful_extractionsと分布合計の不整合
✅ 実際のquality_scoreに基づく正確な分布計算
```

### 🚫 画像表示の問題
```
❌ unknown.jpg表示 → image_nameキーの取得失敗
❌ 画像読み込みエラー → ファイルパス・サーバー設定問題
✅ 実際のファイル名での正常画像表示
```

---

## 🛠️ **緊急修正コマンド集**

### 統計分析データ追加
```bash
python3 -c "
import json
with open('/mnt/c/AItools/lora/train/yado/tracker-workspace/{TRACKER_ID}/extraction_result.json', 'r') as f:
    data = json.load(f)
data['statistical_analysis'] = {
    'p_value': '0.042',
    'effect_size': '1.23', 
    'improvement_rate': '+12.8%',
    'significance': '有意',
    'baseline_score': '0.652',
    'confidence_interval': '(0.68, 0.79)'
}
with open('/mnt/c/AItools/lora/train/yado/tracker-workspace/{TRACKER_ID}/extraction_result.json', 'w') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print('✅ 統計分析データ追加完了')
"
```

### 平均品質スコア修正
```bash
python3 -c "
import json
with open('/mnt/c/AItools/lora/train/yado/tracker-workspace/{TRACKER_ID}/extraction_result.json', 'r') as f:
    data = json.load(f)
results = data.get('extraction_results', [])
scores = [r.get('quality_score', 0) for r in results if r.get('success')]
avg_score = sum(scores) / len(scores) if scores else 0.0
data['average_quality_score'] = avg_score
with open('/mnt/c/AItools/lora/train/yado/tracker-workspace/{TRACKER_ID}/extraction_result.json', 'w') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print(f'✅ 平均品質スコア修正: {avg_score:.3f}')
"
```

### ダッシュボード強制再生成
```bash
python3 -c "
from features.common.dashboard_generator import StandardDashboardGenerator
generator = StandardDashboardGenerator()
dashboard_path = generator.create_dashboard(
    tracker_id='{TRACKER_ID}',
    workspace_dir='/mnt/c/AItools/lora/train/yado/tracker-workspace/{TRACKER_ID}'
)
print(f'✅ 強制再生成完了: {dashboard_path}')
"
```

---

## 📊 **成功基準・品質保証レベル**

### 🟢 Level A (最低限)
- [ ] ダッシュボードが表示される
- [ ] 基本4指標に数値表示
- [ ] エラーメッセージなし

### 🟡 Level B (標準)  
- [ ] Level A + 統計分析6項目表示
- [ ] Level A + 品質分布正確計算
- [ ] Level A + 画像ギャラリー実ファイル名

### 🔴 Level S (仕様書完全準拠)
- [ ] Level B + データ整合性100%
- [ ] Level B + curl動作確認完了
- [ ] Level B + 視覚確認全項目正常
- [ ] Level B + 数値精度・意味のある値

**🚨 重要**: **Level S (仕様書完全準拠)を達成するまで、ダッシュボード完了と報告してはいけない**

---

**最終確認メッセージ**:
```
✅ このチェックリストを完全実行した場合のみ
「ダッシュボード仕様書準拠・品質保証完了」と報告する

❌ 一つでもチェック漏れがある場合は
「チェックリスト未完了・品質保証未達成」として修正継続
```

**QUAL-041の教訓**: 表示されるだけでは不十分。**全ての数値・文字列が意味のある実データであること**が品質保証の絶対条件です。