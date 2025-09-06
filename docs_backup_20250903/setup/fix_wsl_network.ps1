# WSL2ネットワーク設定修復スクリプト
# 管理者権限で実行してください

Write-Host "🔧 WSL2外部アクセス問題修復スクリプト" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green

# 現在のWSL2 IPアドレス取得
Write-Host "📡 WSL2 IPアドレス取得中..."
$wslIP = wsl hostname -I | ForEach-Object { $_.Trim() } | Select-Object -First 1
Write-Host "WSL2 IP: $wslIP" -ForegroundColor Yellow

if (-not $wslIP) {
    Write-Host "❌ WSL2 IPアドレスが取得できませんでした" -ForegroundColor Red
    exit 1
}

# 既存のポートプロキシ削除
Write-Host "🗑️ 既存のポートプロキシ設定削除中..."
netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=8088

# 新しいポートプロキシ設定
Write-Host "🔗 新しいポートプロキシ設定追加中..."
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=8088 connectaddress=$wslIP connectport=8088

# ファイアウォール設定確認・追加
Write-Host "🛡️ ファイアウォール設定確認中..."
$firewallRule = Get-NetFirewallRule -DisplayName "Allow port 8088" -ErrorAction SilentlyContinue
if (-not $firewallRule) {
    Write-Host "🔥 ファイアウォールルール追加中..."
    New-NetFirewallRule -DisplayName "Allow port 8088" -Direction Inbound -LocalPort 8088 -Protocol TCP -Action Allow
} else {
    Write-Host "✅ ファイアウォールルールは既に存在します"
}

# Hyper-V仮想スイッチリセット（高度な修復）
Write-Host "🔄 Hyper-V仮想スイッチ確認中..."
try {
    $vmSwitch = Get-VMSwitch -Name "WSL" -ErrorAction SilentlyContinue
    if ($vmSwitch) {
        Write-Host "✅ WSL仮想スイッチが見つかりました"
    }
} catch {
    Write-Host "⚠️ Hyper-V仮想スイッチ情報を取得できませんでした（権限不足の可能性）"
}

# 設定確認
Write-Host "📋 現在の設定確認:"
netsh interface portproxy show v4tov4

Write-Host ""
Write-Host "✅ 設定完了！以下のURLでテストしてください:" -ForegroundColor Green
Write-Host "🌐 https://100.123.241.106/tracker" -ForegroundColor Cyan
Write-Host "🔐 認証: admin / dashboard2025!" -ForegroundColor Cyan

Write-Host ""
Write-Host "⚠️ まだ接続できない場合の追加手順:" -ForegroundColor Yellow
Write-Host "1. WSLを再起動: wsl --shutdown" -ForegroundColor White
Write-Host "2. PCを再起動" -ForegroundColor White
Write-Host "3. ルーターのポートフォワーディング確認（100.123.241.106 → PCのローカルIP:8088）" -ForegroundColor White