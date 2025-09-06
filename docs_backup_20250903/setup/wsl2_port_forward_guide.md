# WSL2から外部アクセス設定ガイド

## 問題
- WSL2は独自の仮想ネットワーク（172.29.132.130）を使用
- 外部デバイス（iPhone等）から直接アクセス不可
- IP 100.123.241.106はTailscale VPNのIPアドレスの可能性

## 解決方法

### 方法1: Windows側でポートフォワーディング設定（推奨）

PowerShellを**管理者権限**で実行：

```powershell
# WSL2のIPアドレス取得
$wslIp = (wsl hostname -I).Trim()

# ポートフォワーディング設定
netsh interface portproxy add v4tov4 listenport=8080 listenaddress=0.0.0.0 connectport=8080 connectaddress=$wslIp

# Windowsファイアウォール設定
New-NetFirewallRule -DisplayName "Allow port 8080" -Direction Inbound -Protocol TCP -LocalPort 8080 -Action Allow
```

### 方法2: Tailscale経由でアクセス（既に設定済みの場合）

1. iPhoneにTailscaleアプリをインストール
2. 同じTailscaleネットワークに接続
3. http://100.123.241.106:8080 でアクセス

### 方法3: ngrokトンネル使用（最も簡単）

```bash
# ngrokインストール
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
sudo apt update && sudo apt install ngrok

# ngrok認証（無料アカウント作成後）
ngrok authtoken YOUR_AUTH_TOKEN

# トンネル開始
ngrok http 8080 --basic-auth="admin:integrate36"
```

## 現在の状況確認

- WSL2内部IP: 172.29.132.130
- サーバー稼働: ポート8080でリッスン中
- 外部アクセス: ポートフォワーディング未設定

## 推奨アクション

1. **Windowsホスト側でポートフォワーディング設定**
2. **WindowsホストのIPアドレスでアクセス**
   - 例: http://192.168.1.xxx:8080 (LAN内IPアドレス)
3. **またはngrokでインターネット経由アクセス**