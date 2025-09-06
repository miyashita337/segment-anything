# ポート443（HTTPS）を使用する緊急用設定
# 管理者権限で実行してください

Write-Host "🔧 ポート443（HTTPS）への切り替え設定" -ForegroundColor Green

# WSL2 IP取得
$wslIP = wsl hostname -I | ForEach-Object { $_.Trim() } | Select-Object -First 1
Write-Host "WSL2 IP: $wslIP"

# ポート443のプロキシ設定
Write-Host "ポート443の設定中..."
netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=443 2>$null
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=443 connectaddress=$wslIP connectport=8088

# ファイアウォール設定
Write-Host "ファイアウォール設定中..."
New-NetFirewallRule -DisplayName "Allow port 443 for Dashboard" -Direction Inbound -LocalPort 443 -Protocol TCP -Action Allow -ErrorAction SilentlyContinue

Write-Host "✅ 設定完了！"
Write-Host "🌐 アクセスURL: https://100.123.241.106/"
Write-Host "⚠️ HTTPSの警告が出ますが、続行してください"