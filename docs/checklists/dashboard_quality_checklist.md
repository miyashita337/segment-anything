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

### C2: 統計分析結果サーバー表示確認 🔴
```bash
# 必須実行コマンド
curl -u admin:secure_track_2025_q3_8f9a http://100.123.241.106:8088/tracker/{TRACKER_ID} | grep -E "(p値|効果サイズ|改善率|統計的有意性|ベースライン|信頼区間)"
```

- [ ] **🔴 p値表示**: HTML内に数値存在
- [ ] **🔴 効果サイズ表示**: HTML内に数値存在
- [ ] **🔴 改善率表示**: HTML内にパーセント存在
- [ ] **🔴 統計的有意性表示**: HTML内に有意性表示
- [ ] **🔴 ベースライン表示**: HTML内に数値存在
- [ ] **🔴 信頼区間表示**: HTML内に区間存在

### C3: 品質指標サーバー表示確認 🔴
```bash
# 必須実行コマンド
curl -u admin:secure_track_2025_q3_8f9a http://100.123.241.106:8088/tracker/{TRACKER_ID} | grep -E "text-3xl font-bold" | grep -E "[0-9]+\.?[0-9]*"
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

---

## 📋 **Section F: メインダッシュボード安定性検証（INTG-086完全対応）**

### F1: メインダッシュボード表示構造検証 🔴

#### F1.1: 入れ子表示防止検証（5回目発生防止）
```bash
# 必須実行コマンド - メインダッシュボード構造確認
curl -u admin:secure_track_2025_q3_8f9a -s "http://100.123.241.106:8088/" | grep -E "iframe.*src"
```

- [ ] **🔴 iframe src確認**: `src="/"` や `src=""` の入れ子参照が存在しない
- [ ] **🔴 メインダッシュボード表示**: 「メインダッシュボードの中にメインダッシュボード」が発生していない
- [ ] **🔴 正常iframe**: 個別トラッカーへの正しいiframe参照のみ存在

#### F1.2: 左ペイン・右ペイン構造確認
```bash
# 必須実行コマンド - ページ構造検証
curl -u admin:secure_track_2025_q3_8f9a -s "http://100.123.241.106:8088/" | grep -E "(sidebar|main-content)"
```

- [ ] **🔴 左ペイン存在**: `sidebar` クラス・要素が正常に存在
- [ ] **🔴 右ペイン存在**: `main-content` クラス・要素が正常に存在
- [ ] **🔴 レイアウト構造**: 2ペイン構造が正しく表示される

#### F1.3: ダッシュボード一覧表示確認
```bash
# 必須実行コマンド - ダッシュボード一覧確認
curl -u admin:secure_track_2025_q3_8f9a -s "http://100.123.241.106:8088/dashboard-list" | grep -E "tracker.*list"
```

- [ ] **🔴 ダッシュボード一覧**: 79個のダッシュボードが一覧表示される
- [ ] **🔴 トラッカーリンク**: 各トラッカーへの適切なリンクが生成されている
- [ ] **🔴 ページ一覧復元**: 「ようこそ画面」ではなく実際のページ一覧が表示

### F2: 個別トラッカー画面遷移安定性検証 🔴

#### F2.1: 左ペイン消失防止検証
```bash
# 必須実行コマンド - 個別トラッカーページでの左ペイン確認
curl -u admin:secure_track_2025_q3_8f9a -s "http://100.123.241.106:8088/tracker/{TRACKER_ID}" | grep -E "sidebar"
```

- [ ] **🔴 左ペイン継続表示**: 個別トラッカー画面でも左ペインが表示される
- [ ] **🔴 ナビゲーション安定**: トラッカー間の遷移で左ペインが消失しない
- [ ] **🔴 wrapper構造**: `_generate_navigation_wrapper` が正しく動作している

#### F2.2: 右ペイン表示制御検証
```bash
# 必須実行コマンド - 右ペイン表示状態確認
curl -u admin:secure_track_2025_q3_8f9a -s "http://100.123.241.106:8088/tracker/{TRACKER_ID}" | grep -E "(dashboard\.html|main-content)"
```

- [ ] **🔴 トラッカーダッシュボード表示**: 個別のdashboard.htmlが正しく表示される
- [ ] **🔴 iframe構造**: 適切なiframe内でダッシュボードが表示される
- [ ] **🔴 右ペイン表示制御**: 空白表示/一覧表示が意図通りに制御される

### F3: サーバー統合性・プロセス管理検証 🔴

#### F3.1: 適切なサーバー動作確認
```bash
# 必須実行コマンド - サーバープロセス確認
ps aux | grep integrated_dashboard_server
ps aux | grep "python.*http.server"
ss -tulpn | grep 8088
```

- [ ] **🔴 integrated_dashboard_server動作**: `integrated_dashboard_server.py` が正常動作
- [ ] **🔴 単純HTTPサーバー排除**: `python3 -m http.server` 等の不適切サーバーが動作していない
- [ ] **🔴 ポート8088専有**: 適切なサーバーがポート8088を使用している

#### F3.2: 統合ダッシュボードアーキテクチャ検証
```bash
# 必須実行コマンド - システム構造整合性確認
python3 -c "
import subprocess
result = subprocess.run(['curl', '-u', 'admin:secure_track_2025_q3_8f9a', '-s', 'http://100.123.241.106:8088/'], capture_output=True, text=True)
print('✅ 統合ダッシュボードサーバー応答:', len(result.stdout), 'bytes')
print('🔍 入れ子iframe:', 'src=\"/\"' in result.stdout or 'src=\"\"' in result.stdout)
"
```

- [ ] **🔴 統合システム動作**: 統合ダッシュボードシステムが正常応答
- [ ] **🔴 個別ファイル変更影響なし**: 個別トラッカーファイル変更が全体構造に影響していない
- [ ] **🔴 既存システム活用**: 独自実装でなく既存 `integrated_dashboard_server.py` を活用

### F4: 自動品質保証システム実行 🔴

#### F4.1: 構造検証ツール実行
```bash
# 必須実行コマンド - 自動構造検証
python3 html/validation/dashboard_structure_validator.py --validate-all --output /tmp/dashboard_validation.json
```

- [ ] **🔴 自動検証合格**: 構造検証ツールが全テスト合格
- [ ] **🔴 重大エラー0件**: CRITICAL severity エラーが0件
- [ ] **🔴 入れ子検証合格**: 入れ子表示検証が合格
- [ ] **🔴 左ペイン検証合格**: 左ペイン安定性検証が合格

#### F4.2: 回帰テスト実行
```bash
# 必須実行コマンド - 主要ダッシュボード回帰テスト
for tracker_id in QUAL-001 QUAL-044 INTG-086; do
    echo "🧪 テスト: $tracker_id"
    curl -u admin:secure_track_2025_q3_8f9a -s "http://100.123.241.106:8088/tracker/$tracker_id" > /tmp/test_$tracker_id.html
    echo "   サイズ: $(wc -c < /tmp/test_$tracker_id.html) bytes"
    grep -q "sidebar" /tmp/test_$tracker_id.html && echo "   左ペイン: ✅" || echo "   左ペイン: ❌"
done
```

- [ ] **🔴 全トラッカー表示**: 主要トラッカーが全て正常表示
- [ ] **🔴 左ペイン一貫表示**: 全トラッカーで左ペインが表示
- [ ] **🔴 HTML サイズ適正**: 各ダッシュボードが5KB以上の適正サイズ

### F5: 将来防止プロトコル確立 🔴

#### F5.1: HTMLバージョン管理確認
```bash
# 必須実行コマンド - HTMLテンプレート管理確認
ls -la html/templates/
ls -la html/validation/
git status html/
```

- [ ] **🔴 テンプレート管理**: `html/templates/` でHTMLテンプレートが管理されている
- [ ] **🔴 検証ツール配置**: `html/validation/` で自動検証ツールが配置されている
- [ ] **🔴 Git管理**: HTMLテンプレート・検証ツールがGit管理されている

#### F5.2: 統合テンプレート更新確認
```bash
# 必須実行コマンド - 統合テンプレート内容確認
grep -A 10 -B 5 "ダッシュボード安定性" docs/workflows/templates/unified_tracker_template.md
```

- [ ] **🔴 テンプレート更新**: 統合テンプレートにダッシュボード安定性確認が追加されている
- [ ] **🔴 手順明記**: メイン・個別ダッシュボード確認手順が明記されている
- [ ] **🔴 品質基準統合**: チェックリストとテンプレートが整合している

---

## ❌ **絶対回避すべきメインダッシュボードNGパターン（INTG-086教訓）**

### 🚫 メインダッシュボード入れ子表示（5回目発生）
```
❌ iframe src="/" による自己参照
❌ 「メインダッシュボードの中にメインダッシュボード」表示
❌ 無限ループ的な画面構造
✅ 適切なiframe src設定（dashboard-list等）
```

### 🚫 左ペイン消失問題
```
❌ 個別トラッカー画面遷移で左ペイン非表示
❌ _generate_navigation_wrapper の不適切な実装
❌ sidebar要素の欠如
✅ 全画面で一貫した左ペイン表示
```

### 🚫 不適切なサーバー管理
```
❌ python3 -m http.server による単純サーバー
❌ integrated_dashboard_server.py を無視した独自実装
❌ ポート競合による不安定動作
✅ 統合ダッシュボードサーバーの適切活用
```

### 🚫 システム理解不足による破壊
```
❌ 既存システム理解なしでの個別ファイル変更
❌ 影響範囲確認なしでの修正実施
❌ 「とりあえず動かす」応急処置
✅ システム全体理解に基づく適切な修正
```

---

## 🛠️ **メインダッシュボード緊急修復コマンド集**

### 入れ子表示修復
```bash
# integrated_dashboard_server.pyのiframe src修正
python3 -c "
import re
with open('integrated_dashboard_server.py', 'r') as f:
    content = f.read()
# iframe src=\"/\" を dashboard-list に修正
content = re.sub(r'iframe.*src=\"/\"', 'iframe src=\"/dashboard-list\"', content)
with open('integrated_dashboard_server.py', 'w') as f:
    f.write(content)
print('✅ iframe src修正完了')
"
```

### 左ペイン表示強制修復
```bash
# navigation wrapper生成の修復
python3 -c "
import re
with open('integrated_dashboard_server.py', 'r') as f:
    content = f.read()
# handle_tracker内のreturn修正確認
if 'nav_html = self._generate_navigation_wrapper' not in content:
    print('⚠️ navigation wrapper生成が不完全')
else:
    print('✅ navigation wrapper生成は正常')
"
```

### サーバー正常化
```bash
# 不適切サーバー終了→適切サーバー起動
pkill -f "python.*http.server"
nohup python3 integrated_dashboard_server.py --port 8088 > /tmp/dashboard_server_fixed.log 2>&1 &
echo "✅ 統合ダッシュボードサーバー起動完了"
```

---

## 📊 **メインダッシュボード品質保証レベル強化**

### 🟢 Level A (基本動作)
- [ ] メインダッシュボードが表示される
- [ ] 左ペイン・右ペインが表示される  
- [ ] 入れ子表示が発生していない

### 🟡 Level B (安定性保証)
- [ ] Level A + 個別トラッカー遷移で左ペイン継続表示
- [ ] Level A + 適切なサーバープロセス動作
- [ ] Level A + ダッシュボード一覧正常表示

### 🔴 Level S (完全安定性保証)
- [ ] Level B + 自動構造検証ツール全合格
- [ ] Level B + 回帰テスト全合格
- [ ] Level B + HTMLバージョン管理体制確立
- [ ] Level B + 将来防止プロトコル完全実装

**🚨 重要**: **Level S (完全安定性保証)を達成するまで、メインダッシュボード完了と報告してはいけない**

**🔒 6回目発生絶対防止**: このSection F全項目実行により、メインダッシュボード問題の6回目発生を完全に防止する

---

## 📋 **Section E: システム統合性検証（QUAL-044事例対応）**

### E1: 既存システム理解確認 🔴
```bash
# 必須実行コマンド - integrated_dashboard_server.py 動作確認
ps aux | grep integrated_dashboard_server
ss -tulpn | grep 8088
```

- [ ] **🔴 統合ダッシュボードサーバー動作確認**: `integrated_dashboard_server.py` プロセス存在
- [ ] **🔴 外部サーバーポート確認**: ポート8088で正しいサーバーが動作
- [ ] **🔴 単純HTTPサーバー排除**: `python3 -m http.server` 等の不適切プロセス不存在

### E2: サーバー構造整合性確認 🔴
```bash
# 必須実行コマンド - メインダッシュボード構造確認
curl -u admin:secure_track_2025_q3_8f9a -s "http://100.123.241.106:8088/" | grep -E "(sidebar|main-content|tracker.*list)"
```

- [ ] **🔴 左ペイン構造**: `sidebar` 要素存在確認
- [ ] **🔴 右画面構造**: `main-content` 要素存在確認  
- [ ] **🔴 トラッカー一覧**: 複数のトラッカーIDがリスト表示
- [ ] **🔴 統合ダッシュボード**: QUAL-044単体ではなく統合画面が表示

### E3: 個別ファイル変更影響範囲確認 🔴
```bash
# 必須実行コマンド - 個別トラッカーファイル確認
ls -la /mnt/c/AItools/lora/train/yado/tracker-workspace/{TRACKER_ID}/dashboard/index.html
diff -u /mnt/c/AItools/lora/train/yado/tracker-workspace/{TRACKER_ID}/dashboard/dashboard.html /mnt/c/AItools/lora/train/yado/tracker-workspace/{TRACKER_ID}/dashboard/index.html
```

- [ ] **🔴 index.html = dashboard.html**: 個別トラッカーのindex.htmlがdashboard.htmlと同一内容
- [ ] **🔴 統合ダッシュボード形式でない**: 個別ファイルが統合ダッシュボード構造でない
- [ ] **🔴 外部サーバー影響なし**: 個別ファイル変更が外部サーバー全体構造に影響を与えていない

### E4: 新規実装vs既存システム活用確認 🔴

- [ ] **🔴 既存システム優先確認**: `integrated_dashboard_server.py` 等の既存実装を最大限活用
- [ ] **🔴 独自実装最小化**: 既存システムで解決できない部分のみ新規実装
- [ ] **🔴 重複実装排除**: 同一機能の重複実装が存在しない

### E5: サーバー管理標準遵守確認 🔴

- [ ] **🔴 適切なサーバー選択**: 統合管理システム使用（単純http.server不使用）
- [ ] **🔴 プロセス管理適切**: 不要プロセス終了後に適切なサーバー起動
- [ ] **🔴 ポート競合回避**: ポート使用状況確認後の適切なサーバー切り替え

**🚨 重要**: **Section E 全項目クリア必須。一項目でも不適合の場合、システム統合性問題として修正継続**

### 📋 QUAL-044事例学習ポイント

```
❌ 避けるべき行為:
  - integrated_dashboard_server.py を無視した独自HTML実装
  - 個別トラッカーindex.htmlの無断変更による全体構造破壊
  - 単純HTTPサーバーでの応急処置

✅ 正しいアプローチ:
  - 既存統合システムの理解・活用最優先
  - ファイル変更前の影響範囲完全把握
  - 適切なサーバー管理による統合ダッシュボード提供
```