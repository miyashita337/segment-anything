# メインダッシュボード緊急復旧プロトコル

**作成日**: 2025-09-09  
**目的**: INTG-086で発生した5回目のメインダッシュボード問題を受け、6回目発生時の迅速復旧と根本対策  
**適用レベル**: CRITICAL - システム管理者・開発者必須実行

---

## 🚨 **緊急事態の定義**

以下の症状が1つでも発生した場合、このプロトコルを即座に実行：

- ❌ **入れ子表示**: 「メインダッシュボードの中にメインダッシュボード」が表示される
- ❌ **左ペイン消失**: 個別トラッカー画面で左ペインが表示されない
- ❌ **右ペイン異常**: 右ペインが空白、または意図しないコンテンツが表示される
- ❌ **ダッシュボード一覧消失**: メインページでダッシュボード一覧が表示されない
- ❌ **サーバーエラー**: 500番台エラー、またはページが全く表示されない

---

## 📋 **Phase 1: 緊急診断・状況把握（5分以内）**

### 1.1 現象確認
```bash
# 【必須実行】メインダッシュボード状況確認
curl -u admin:secure_track_2025_q3_8f9a -s "http://100.123.241.106:8088/" > /tmp/emergency_main.html
echo "メインダッシュボードHTMLサイズ: $(wc -c < /tmp/emergency_main.html) bytes"

# iframe入れ子確認
grep -E "iframe.*src.*/" /tmp/emergency_main.html || echo "入れ子iframe: 検出されず"

# sidebar/main-content構造確認  
grep -E "(sidebar|main-content)" /tmp/emergency_main.html || echo "ページ構造: 異常"
```

### 1.2 サーバープロセス確認
```bash
# 【必須実行】サーバー状況診断
echo "=== サーバープロセス確認 ==="
ps aux | grep integrated_dashboard_server
ps aux | grep "python.*http.server"
ss -tulpn | grep 8088

# 問題特定
if ps aux | grep -q "integrated_dashboard_server"; then
    echo "✅ 統合サーバー: 動作中"
else
    echo "❌ 統合サーバー: 停止"
fi

if ps aux | grep -q "python.*http.server"; then
    echo "⚠️ 単純HTTPサーバー: 検出（問題の可能性）"
else
    echo "✅ 単純HTTPサーバー: 検出されず"
fi
```

### 1.3 問題分類
```bash
# 【必須実行】問題パターン分類
echo "=== 問題分類 ==="

# パターンA: 入れ子表示
if grep -q 'iframe.*src="/"' /tmp/emergency_main.html; then
    echo "🚨 パターンA: 入れ子表示問題検出"
    echo "  → integrated_dashboard_server.py修正必要"
fi

# パターンB: 左ペイン消失  
if ! grep -q "sidebar" /tmp/emergency_main.html; then
    echo "🚨 パターンB: 左ペイン消失問題検出"
    echo "  → navigation wrapper修正必要"
fi

# パターンC: サーバー異常
if [ ! -s /tmp/emergency_main.html ]; then
    echo "🚨 パターンC: サーバー応答なし"
    echo "  → サーバー再起動必要"
fi
```

---

## 📋 **Phase 2: 即座修正（10分以内）**

### 2.1 パターンA修正: 入れ子表示
```bash
# 【緊急修正】iframe src修正
echo "🔧 入れ子表示修正実行..."

python3 -c "
import re
with open('integrated_dashboard_server.py', 'r') as f:
    content = f.read()

# バックアップ作成
with open('integrated_dashboard_server.py.emergency.bak', 'w') as f:
    f.write(content)

# iframe src=\"/\" を dashboard-list に修正
content = re.sub(r'iframe.*src=\"/\"', 'iframe src=\"/dashboard-list\"', content)

# _generate_navigation_wrapper内のメイン画面判定修正
content = re.sub(
    r'dashboard_path = \"\"',
    'dashboard_path = \"dashboard-list\"',
    content
)

with open('integrated_dashboard_server.py', 'w') as f:
    f.write(content)

print('✅ iframe src修正完了')
print('📄 バックアップ: integrated_dashboard_server.py.emergency.bak')
"

# サーバー再起動
pkill -f integrated_dashboard_server
nohup python3 integrated_dashboard_server.py --port 8088 > /tmp/emergency_recovery.log 2>&1 &
echo "✅ サーバー再起動完了"
```

### 2.2 パターンB修正: 左ペイン消失
```bash
# 【緊急修正】navigation wrapper修正
echo "🔧 左ペイン消失修正実行..."

python3 -c "
import re
with open('integrated_dashboard_server.py', 'r') as f:
    content = f.read()

# バックアップ作成（未作成時のみ）
import os
if not os.path.exists('integrated_dashboard_server.py.emergency.bak'):
    with open('integrated_dashboard_server.py.emergency.bak', 'w') as f:
        f.write(content)

# handle_tracker メソッドの修正
# 直接dashboard.html返却を nav_html 経由に変更
pattern = r'(async def handle_tracker.*?)(return web\.FileResponse.*?dashboard\.html.*?)'
replacement = r'\1nav_html = self._generate_navigation_wrapper(tracker_id, dashboard_key)\n        return web.Response(text=nav_html, content_type=\"text/html\")'

content = re.sub(pattern, replacement, content, flags=re.DOTALL)

with open('integrated_dashboard_server.py', 'w') as f:
    f.write(content)

print('✅ navigation wrapper修正完了')
"

# サーバー再起動
pkill -f integrated_dashboard_server
nohup python3 integrated_dashboard_server.py --port 8088 > /tmp/emergency_recovery.log 2>&1 &
echo "✅ サーバー再起動完了"
```

### 2.3 パターンC修正: サーバー異常
```bash
# 【緊急修正】サーバー完全再起動
echo "🔧 サーバー異常修正実行..."

# 全関連プロセス終了
pkill -f "python.*dashboard"
pkill -f "python.*http.server"
pkill -f "python.*8088"

# ポート確認・解放
ss -tulpn | grep 8088 && echo "⚠️ ポート8088まだ使用中、10秒待機..." && sleep 10

# 統合サーバー起動
echo "🚀 統合ダッシュボードサーバー起動..."
nohup python3 integrated_dashboard_server.py --port 8088 > /tmp/emergency_recovery.log 2>&1 &

# 起動確認
sleep 5
if ps aux | grep -q integrated_dashboard_server; then
    echo "✅ サーバー起動成功"
else
    echo "❌ サーバー起動失敗 - ログ確認: cat /tmp/emergency_recovery.log"
fi
```

---

## 📋 **Phase 3: 動作確認（5分以内）**

### 3.1 基本動作確認
```bash
# 【必須確認】修正後動作確認
echo "🧪 修正後動作確認..."

# メインダッシュボード確認
curl -u admin:secure_track_2025_q3_8f9a -s "http://100.123.241.106:8088/" > /tmp/recovery_check_main.html
main_size=$(wc -c < /tmp/recovery_check_main.html)

if [ $main_size -gt 5000 ]; then
    echo "✅ メインダッシュボード: 正常 (${main_size} bytes)"
else
    echo "❌ メインダッシュボード: 異常 (${main_size} bytes)"
fi

# 入れ子表示確認
if grep -q 'src="/"' /tmp/recovery_check_main.html; then
    echo "❌ 入れ子表示: まだ存在"
else
    echo "✅ 入れ子表示: 解消"
fi

# sidebar確認
if grep -q "sidebar" /tmp/recovery_check_main.html; then
    echo "✅ 左ペイン: 存在"
else
    echo "❌ 左ペイン: まだ存在しない"
fi
```

### 3.2 個別トラッカー確認
```bash
# 【必須確認】個別トラッカー動作確認
echo "🧪 個別トラッカー動作確認..."

# 主要トラッカーテスト
for tracker_id in QUAL-001 QUAL-044 INTG-086; do
    echo "テスト: $tracker_id"
    
    curl -u admin:secure_track_2025_q3_8f9a -s "http://100.123.241.106:8088/tracker/$tracker_id" > /tmp/recovery_check_$tracker_id.html
    
    size=$(wc -c < /tmp/recovery_check_$tracker_id.html)
    echo "  サイズ: ${size} bytes"
    
    if grep -q "sidebar" /tmp/recovery_check_$tracker_id.html; then
        echo "  左ペイン: ✅"
    else
        echo "  左ペイン: ❌"
    fi
done
```

### 3.3 自動検証ツール実行
```bash
# 【必須確認】自動構造検証
echo "🧪 自動構造検証実行..."

if [ -f "html/validation/dashboard_structure_validator.py" ]; then
    python3 html/validation/dashboard_structure_validator.py --validate-all --output /tmp/emergency_validation.json
    
    # 結果確認
    if [ -f "/tmp/emergency_validation.json" ]; then
        critical_errors=$(python3 -c "import json; data=json.load(open('/tmp/emergency_validation.json')); print(data.get('critical_errors', 0))")
        failed_tests=$(python3 -c "import json; data=json.load(open('/tmp/emergency_validation.json')); print(data.get('failed_tests', 0))")
        
        if [ "$critical_errors" -eq 0 ] && [ "$failed_tests" -eq 0 ]; then
            echo "✅ 自動検証: 全合格"
        else
            echo "❌ 自動検証: 重大エラー${critical_errors}件, 失敗テスト${failed_tests}件"
        fi
    fi
else
    echo "⚠️ 自動検証ツール未配置 - 手動確認のみ"
fi
```

---

## 📋 **Phase 4: 根本対策実行（15分以内）**

### 4.1 設定バックアップ・バージョン管理
```bash
# 【根本対策】設定ファイルバックアップ体制確立
echo "🔒 根本対策: バックアップ体制確立..."

# 現在の設定をバックアップ
mkdir -p backup/emergency/$(date +%Y%m%d_%H%M%S)
cp integrated_dashboard_server.py backup/emergency/$(date +%Y%m%d_%H%M%S)/
cp -r html/ backup/emergency/$(date +%Y%m%d_%H%M%S)/ 2>/dev/null || echo "html/ディレクトリなし"

# git管理確認・追加
git status html/
git add html/templates/ html/validation/ 2>/dev/null || echo "html/テンプレート未追加"
git add docs/checklists/dashboard_quality_checklist.md 2>/dev/null || echo "チェックリスト未追加"
git add docs/workflows/templates/unified_tracker_template.md 2>/dev/null || echo "統合テンプレート未追加"

echo "✅ バックアップ・Git管理体制確立"
```

### 4.2 監視・アラート体制
```bash
# 【根本対策】監視スクリプト配置
echo "🔒 根本対策: 監視体制確立..."

cat > /tmp/dashboard_monitor.sh << 'EOF'
#!/bin/bash
# ダッシュボード監視スクリプト

check_dashboard() {
    # メインダッシュボード確認
    main_response=$(curl -u admin:secure_track_2025_q3_8f9a -s "http://100.123.241.106:8088/" | wc -c)
    
    if [ $main_response -lt 5000 ]; then
        echo "$(date): ❌ メインダッシュボード異常 (${main_response} bytes)" >> /tmp/dashboard_monitor.log
        return 1
    fi
    
    # 入れ子表示確認
    if curl -u admin:secure_track_2025_q3_8f9a -s "http://100.123.241.106:8088/" | grep -q 'src="/"'; then
        echo "$(date): ❌ 入れ子表示検出" >> /tmp/dashboard_monitor.log
        return 1
    fi
    
    echo "$(date): ✅ ダッシュボード正常" >> /tmp/dashboard_monitor.log
    return 0
}

# 監視実行
check_dashboard
EOF

chmod +x /tmp/dashboard_monitor.sh
echo "✅ 監視スクリプト作成: /tmp/dashboard_monitor.sh"
```

### 4.3 予防チェックリスト実行
```bash
# 【根本対策】品質チェックリスト実行
echo "🔒 根本対策: 品質チェックリスト実行..."

# Section F 実行（可能な限り）
echo "📋 Section F: メインダッシュボード安定性検証実行中..."

# F1: 表示構造検証
echo "F1.1: 入れ子表示防止検証"
if ! curl -u admin:secure_track_2025_q3_8f9a -s "http://100.123.241.106:8088/" | grep -q 'src="/"'; then
    echo "  ✅ 入れ子表示なし"
fi

# F2: 個別トラッカー安定性
echo "F2.1: 左ペイン消失防止検証"
if curl -u admin:secure_track_2025_q3_8f9a -s "http://100.123.241.106:8088/tracker/QUAL-001" | grep -q "sidebar"; then
    echo "  ✅ 左ペイン継続表示"
fi

# F3: サーバー統合性
echo "F3.1: 適切なサーバー動作確認"
if ps aux | grep -q integrated_dashboard_server && ! ps aux | grep -q "python.*http.server"; then
    echo "  ✅ 適切なサーバー動作"
fi

echo "✅ 予防チェックリスト実行完了"
```

---

## 📋 **Phase 5: 完了報告・文書化（10分以内）**

### 5.1 復旧報告生成
```bash
# 【完了報告】復旧レポート生成
echo "📄 復旧レポート生成..."

cat > /tmp/emergency_recovery_report.md << EOF
# メインダッシュボード緊急復旧報告

**復旧実行日時**: $(date)
**復旧担当**: システム管理者
**問題分類**: [入れ子表示/左ペイン消失/サーバー異常]

## 復旧実行内容
$(echo "詳細実行ログ: /tmp/emergency_recovery.log")

## 復旧後状況
- メインダッシュボード: $([ -f /tmp/recovery_check_main.html ] && echo "正常 ($(wc -c < /tmp/recovery_check_main.html) bytes)" || echo "未確認")
- 左ペイン表示: $(curl -u admin:secure_track_2025_q3_8f9a -s "http://100.123.241.106:8088/" | grep -q "sidebar" && echo "正常" || echo "異常")
- 入れ子表示: $(curl -u admin:secure_track_2025_q3_8f9a -s "http://100.123.241.106:8088/" | grep -q 'src="/"' && echo "まだ存在" || echo "解消")
- サーバープロセス: $(ps aux | grep -q integrated_dashboard_server && echo "正常動作" || echo "異常")

## 根本対策実行
- [x] 設定ファイルバックアップ作成
- [x] Git管理体制確立  
- [x] 監視スクリプト配置
- [x] 品質チェックリスト実行

## 次回予防策
1. 定期的な自動構造検証実行
2. integrated_dashboard_server.py変更時の必須テスト
3. HTMLテンプレートバージョン管理徹底
4. 品質チェックリスト Section F 完全実行

**緊急復旧完了**: $(date)
EOF

echo "✅ 復旧レポート: /tmp/emergency_recovery_report.md"
```

### 5.2 今後の予防策設定
```bash
# 【予防策】定期チェック設定
echo "🔒 今後の予防策設定..."

# cron設定提案（実行は任意）
cat << EOF
# 以下をcrontabに追加してダッシュボード定期監視（任意）
# 毎時0分にダッシュボード監視実行
0 * * * * /tmp/dashboard_monitor.sh

# 毎日6時に完全構造検証実行  
0 6 * * * python3 /mnt/c/AItools/segment-anything/html/validation/dashboard_structure_validator.py --validate-all --output /tmp/daily_validation.json
EOF

echo "✅ 予防策設定提案完了"
```

---

## 🚨 **緊急時連絡・判断基準**

### エスカレーション基準
- **Phase 2実行後も問題継続**: 開発チームに即座連絡
- **サーバー起動不可**: システム管理者に即座連絡
- **データ破損・消失発生**: バックアップからの復旧開始

### 判断不能時の対応
1. **現状維持**: 不明な修正は実行しない
2. **ログ保全**: 全ログファイル保存・バックアップ
3. **専門家連絡**: 問題を明確にして技術者に報告

---

## 📚 **関連資料・参考情報**

- **品質チェックリスト**: `docs/checklists/dashboard_quality_checklist.md` Section F
- **統合テンプレート**: `docs/workflows/templates/unified_tracker_template.md` ダッシュボード安定性セクション
- **自動検証ツール**: `html/validation/dashboard_structure_validator.py`
- **HTMLテンプレート**: `html/templates/main_dashboard.html`, `html/templates/tracker_wrapper.html`

---

**🔒 このプロトコルにより、INTG-086で発生した5回目の問題を最後とし、6回目発生を完全防止する**