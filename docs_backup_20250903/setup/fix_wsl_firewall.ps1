# P1-B002: WSL ファイアウォール設定スクリプト
# Windows PowerShell (管理者権限) で実行

Write-Host "🔥 WSL ダッシュボードアクセス用ファイアウォール設定" -ForegroundColor Green

# WSL ネットワークプロファイルをプライベートに設定
Write-Host "📡 WSL ネットワークをプライベートに設定中..." -ForegroundColor Yellow
Get-NetConnectionProfile | Where-Object {$_.InterfaceAlias -like "*WSL*"} | Set-NetConnectionProfile -NetworkCategory Private

# ファイアウォール規則を追加 (ポート 8084)
Write-Host "🛡️ ファイアウォール規則追加中..." -ForegroundColor Yellow
New-NetFirewallRule -DisplayName "WSL Dashboard Port 8084" -Direction Inbound -Protocol TCP -LocalPort 8084 -Action Allow
New-NetFirewallRule -DisplayName "WSL Dashboard Port 8082" -Direction Inbound -Protocol TCP -LocalPort 8082 -Action Allow
New-NetFirewallRule -DisplayName "WSL Dashboard Port 8080" -Direction Inbound -Protocol TCP -LocalPort 8080 -Action Allow

# ポートプロキシ設定
Write-Host "🔄 ポートプロキシ設定中..." -ForegroundColor Yellow
netsh interface portproxy delete v4tov4 listenport=8084 listenaddress=0.0.0.0
netsh interface portproxy add v4tov4 listenport=8084 listenaddress=0.0.0.0 connectport=8084 connectaddress=172.29.132.130

# 設定確認
Write-Host "✅ 現在のポートプロキシ設定:" -ForegroundColor Green
netsh interface portproxy show all

Write-Host "🎯 設定完了! 以下のURLでアクセス可能:" -ForegroundColor Green
Write-Host "   • http://localhost:8084" -ForegroundColor Cyan
Write-Host "   • http://127.0.0.1:8084" -ForegroundColor Cyan