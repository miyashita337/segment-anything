# P1-B002: Windows ポート転送設定スクリプト
# PowerShell (管理者権限) で実行してください

Write-Host "🚀 P1-B002 ダッシュボード用ポート転送設定" -ForegroundColor Green
Write-Host "=" * 50 -ForegroundColor Yellow

# WSL IP取得
$wslIP = (wsl hostname -I).Trim()
Write-Host "📡 WSL IPアドレス: $wslIP" -ForegroundColor Cyan

# 既存のポート転送を削除
Write-Host "🧹 既存ポート転送クリア中..." -ForegroundColor Yellow
netsh interface portproxy delete v4tov4 listenport=8080 2>$null
netsh interface portproxy delete v4tov4 listenport=8082 2>$null  
netsh interface portproxy delete v4tov4 listenport=8085 2>$null

# 新しいポート転送を追加
Write-Host "🔄 新しいポート転送設定中..." -ForegroundColor Yellow
netsh interface portproxy add v4tov4 listenport=8080 listenaddress=0.0.0.0 connectport=8080 connectaddress=$wslIP
netsh interface portproxy add v4tov4 listenport=8082 listenaddress=0.0.0.0 connectport=8082 connectaddress=$wslIP
netsh interface portproxy add v4tov4 listenport=8085 listenaddress=0.0.0.0 connectport=8085 connectaddress=$wslIP

# ファイアウォール規則追加
Write-Host "🛡️ ファイアウォール規則追加中..." -ForegroundColor Yellow
New-NetFirewallRule -DisplayName "WSL Dashboard 8080" -Direction Inbound -Protocol TCP -LocalPort 8080 -Action Allow -Force 2>$null
New-NetFirewallRule -DisplayName "WSL Dashboard 8082" -Direction Inbound -Protocol TCP -LocalPort 8082 -Action Allow -Force 2>$null
New-NetFirewallRule -DisplayName "WSL Dashboard 8085" -Direction Inbound -Protocol TCP -LocalPort 8085 -Action Allow -Force 2>$null

# 設定確認
Write-Host "`n✅ 設定完了! 現在のポート転送:" -ForegroundColor Green
netsh interface portproxy show all

Write-Host "`n🎯 テスト用URL:" -ForegroundColor Green
Write-Host "   • http://localhost:8080" -ForegroundColor Cyan
Write-Host "   • http://localhost:8082" -ForegroundColor Cyan  
Write-Host "   • http://localhost:8085" -ForegroundColor Cyan
Write-Host "   • http://127.0.0.1:8080" -ForegroundColor Cyan

Write-Host "`n📝 次の手順:" -ForegroundColor Yellow
Write-Host "1. WSLでダッシュボードサーバーを起動"
Write-Host "2. 上記URLでWindowsブラウザからアクセス" 
Write-Host "3. うまくいかない場合は解決策2を試してください"

Write-Host "`n⚠️  注意: 管理者権限で実行する必要があります" -ForegroundColor Red