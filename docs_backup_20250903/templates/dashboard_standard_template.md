# ダッシュボード標準化テンプレート

**作成日**: 2025-08-27  
**目的**: 全トラッカーで統一されたダッシュボードフォーマットの提供

## 標準HTMLテンプレート

今後の全トラッカーで以下の統一フォーマットを使用:

```html
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{TRACKER_ID} - 品質評価ダッシュボード</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .quality-badge-high { @apply bg-green-500 text-white px-2 py-1 rounded text-xs font-semibold; }
        .quality-badge-medium { @apply bg-yellow-500 text-white px-2 py-1 rounded text-xs font-semibold; }
        .quality-badge-low { @apply bg-orange-500 text-white px-2 py-1 rounded text-xs font-semibold; }
        .quality-badge-poor { @apply bg-red-500 text-white px-2 py-1 rounded text-xs font-semibold; }
        
        .image-container { 
            max-width: 100%; 
            height: auto; 
            border-radius: 8px; 
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
    </style>
</head>
<body class="bg-gray-100 min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <!-- 標準化されたセクション構造 -->
        
        <!-- 1. ヘッダー -->
        <header class="bg-white rounded-lg shadow-md p-6 mb-8">
            <h1 class="text-3xl font-bold text-gray-800 mb-2">{TRACKER_ID} 品質評価ダッシュボード</h1>
            <p class="text-gray-600">生成日時: {GENERATION_DATE}</p>
        </header>
        
        <!-- 2. 統計サマリー -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <!-- 総画像数、平均品質スコア、成功画像数、要改善数 -->
        </div>
        
        <!-- 3. 品質分布 -->
        <div class="bg-white rounded-lg shadow-md p-6 mb-8">
            <h2 class="text-xl font-semibold text-gray-800 mb-4">品質分布</h2>
            <!-- 高品質、中品質、低品質、要改善の分布 -->
        </div>
        
        <!-- 4. 統計分析結果 -->
        <div class="bg-white rounded-lg shadow-md p-6 mb-8">
            <h2 class="text-xl font-semibold text-gray-800 mb-4">📊 統計分析結果</h2>
            <!-- p値、効果サイズ、改善率、統計的有意性 -->
        </div>
        
        <!-- 5. 画像ギャラリー -->
        <div class="bg-white rounded-lg shadow-md p-6">
            <h2 class="text-xl font-semibold text-gray-800 mb-6">画像品質評価結果</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                <!-- 画像パス形式: /{TRACKER_ID}/extraction/{filename} -->
            </div>
        </div>
    </div>
</body>
</html>
```

## 必須要件

### 1. フレームワーク
- **Tailwind CSS** (CDN版) を使用
- カスタムCSSは最小限に留める
- グラデーション背景等の装飾的要素は禁止

### 2. データソース
- `statistical_report.json` から統計データを取得
- `extraction_result.json` から画像リストを取得
- ハードコードされた値は使用しない

### 3. 画像表示
- パス形式: `/{TRACKER_ID}/extraction/{filename}`
- Base64埋め込みは禁止
- 画像読み込みエラー時のフォールバック処理必須

### 4. 品質バッジ
- 高品質 (>= 0.8): `quality-badge-high` (緑)
- 中品質 (>= 0.7): `quality-badge-medium` (黄)
- 低品質 (>= 0.5): `quality-badge-low` (橙)
- 要改善 (< 0.5): `quality-badge-poor` (赤)

### 5. レスポンシブデザイン
- モバイル: `grid-cols-1`
- タブレット: `grid-cols-2`
- デスクトップ: `grid-cols-3` または `grid-cols-4`

## 統合方法

### unified_dashboard_wrapper.py での使用

```python
from features.common.dashboard_generator import StandardDashboardGenerator

# 標準ダッシュボード生成器の強制使用
generator = StandardDashboardGenerator(
    tracker_id=tracker_id,
    template_path="docs/templates/dashboard_standard_template.md"
)
dashboard_html = generator.generate()
```

### run_quality_workflow.sh での呼び出し

```bash
# Step 9B: ダッシュボード生成（標準化テンプレート使用）
python tools/scripts/unified_dashboard_wrapper.py \
    --tracker_id ${TRACKER_ID} \
    --use_standard_template
```

## 参照実装

- **正しい実装例**: INCI-003ダッシュボード
- **仕様書**: `docs/workflows/templates/unified_tracker_template.md`
- **ワークフロー**: `docs/workflows/checklists/tracker_workflow_checklist.md`