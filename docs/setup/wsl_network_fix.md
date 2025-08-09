# WSL2 ネットワーク接続問題 解決ガイド

## 🚨 問題: ERR_CONNECTION_REFUSED

WSL2環境で動作するサーバーにWindowsからアクセスできない問題

## 🔧 解決手順

### 1. Windows PowerShell (管理者権限) で実行

```powershell
# WSL IP確認
wsl hostname -I

# ファイアウォール規則追加
New-NetFirewallRule -DisplayName "WSL Dashboard" -Direction Inbound -Protocol TCP -LocalPort 8085 -Action Allow

# ポートプロキシ設定
netsh interface portproxy delete v4tov4 listenport=8085
netsh interface portproxy add v4tov4 listenport=8085 listenaddress=0.0.0.0 connectport=8085 connectaddress=172.29.132.130

# 設定確認
netsh interface portproxy show all
```

### 2. WSL側での確認

```bash
# ポート8085が使用中か確認
netstat -tlnp | grep 8085

# ファイアウォール状態確認 (Ubuntu)
sudo ufw status

# 必要に応じてファイアウォール無効化
sudo ufw disable
```

### 3. アクセステスト用URL

- **Windows ローカル**: http://localhost:8085
- **WSL直接**: http://172.29.132.130:8085
- **テスト用**: http://127.0.0.1:8085

## 🔄 代替アクセス方法

### VS Code Port Forwarding
1. VS Code で WSL に接続
2. ターミナルでサーバー起動
3. PORTS タブで 8085 を転送
4. 生成されたローカルURLでアクセス

### Windows Terminal での確認
```cmd
# ポート使用状況確認
netstat -an | findstr 8085

# プロセス確認  
tasklist | findstr python
```

## 🎯 最終的なアクセス URL

設定完了後、以下のURLでアクセス可能:
- http://localhost:8085 (推奨)
- http://127.0.0.1:8085