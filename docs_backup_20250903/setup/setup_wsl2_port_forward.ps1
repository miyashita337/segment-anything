# WSL2ポートフォワーディング設定スクリプト
# PowerShellを管理者権限で実行してください

Write-Host "WSL2ポートフォワーディング設定開始..." -ForegroundColor Green

# WSL2のIPアドレス取得
$wslIp = (wsl hostname -I).Trim().Split()[0]
Write-Host "WSL2 IP: $wslIp" -ForegroundColor Yellow

# 既存のポートフォワーディングルール削除
Write-Host "既存のルール削除中..." -ForegroundColor Yellow
netsh interface portproxy delete v4tov4 listenport=8080 listenaddress=0.0.0.0 2>$null

# 新しいポートフォワーディングルール追加
Write-Host "ポートフォワーディング設定中..." -ForegroundColor Yellow
netsh interface portproxy add v4tov4 listenport=8080 listenaddress=0.0.0.0 connectport=8080 connectaddress=$wslIp

# ファイアウォールルール確認・追加
$firewallRule = Get-NetFirewallRule -DisplayName "Allow port 8080" -ErrorAction SilentlyContinue
if (-not $firewallRule) {
    Write-Host "ファイアウォールルール追加中..." -ForegroundColor Yellow
    New-NetFirewallRule -DisplayName "Allow port 8080" -Direction Inbound -Protocol TCP -LocalPort 8080 -Action Allow
} else {
    Write-Host "ファイアウォールルールは既に存在します" -ForegroundColor Green
}

# WindowsホストのIPアドレス取得
$hostIps = Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.InterfaceAlias -notlike "*Loopback*" -and $_.InterfaceAlias -notlike "*WSL*"}

Write-Host "`n設定完了！" -ForegroundColor Green
Write-Host "以下のURLでアクセス可能です:" -ForegroundColor Cyan

foreach ($ip in $hostIps) {
    Write-Host "  http://$($ip.IPAddress):8080" -ForegroundColor White
}

Write-Host "`nTailscale経由の場合:" -ForegroundColor Cyan
Write-Host "  http://100.123.241.106:8080" -ForegroundColor White

Write-Host "`nBasic認証情報:" -ForegroundColor Cyan
Write-Host "  ユーザー名: admin" -ForegroundColor White
Write-Host "  パスワード: integrate36" -ForegroundColor White

# 現在の設定確認
Write-Host "`n現在のポートフォワーディング設定:" -ForegroundColor Yellow
netsh interface portproxy show v4tov4