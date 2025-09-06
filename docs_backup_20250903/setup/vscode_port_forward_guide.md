# VS Code ポート転送でダッシュボードアクセス

## 🎯 この方法が最も確実です

### 手順

1. **VS Code でWSLに接続**
   - VS Code起動
   - `Ctrl + Shift + P` → `WSL: Connect to WSL`

2. **WSL内でダッシュボードサーバー起動**
   ```bash
   cd /mnt/c/AItools/segment-anything
   python3 simple_dashboard_test.py
   ```

3. **VS Codeでポート転送**
   - VS Code下部の **PORTS** タブをクリック
   - **Port 8085** を見つけて右クリック
   - **"Forward Port"** を選択
   - 自動的に転送URLが生成される

4. **ブラウザでアクセス**
   - 生成されたURL（通常 `http://localhost:8085`）をクリック
   - または **"Open in Browser"** を選択

## 🔄 代替手順（手動ポート転送）

VS CodeのPORTSタブで：
1. **"+"** ボタンをクリック
2. **8085** と入力
3. **Enter** を押す
4. 生成されたURLをクリック

## ✅ この方法のメリット

- ✅ 管理者権限不要
- ✅ ファイアウォール設定不要
- ✅ 自動でポート転送
- ✅ WSL2で100%動作

## 🎯 確実なテスト方法

```bash
# WSL内で実行
python3 -c "
import http.server
import socketserver

with socketserver.TCPServer(('0.0.0.0', 8085), http.server.SimpleHTTPRequestHandler) as httpd:
    print('🌐 VS Codeでポート8085を転送してください')
    print('📍 http://localhost:8085 でアクセス可能になります')
    httpd.serve_forever()
"
```