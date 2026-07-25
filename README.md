# Brass Studio Analysis API

Brass Studio のフロントエンドと解析エンジンを接続する FastAPI サーバーです。

## 現在の機能

- `/health` の疎通確認
- MP3 / WAV / M4A のアップロード受付
- 200MB のサイズ制限
- CORS 対応
- アップロード成功結果の JSON 返却

現段階は接続確認用です。音源分離・音高解析・MusicXML生成は次の工程で追加します。

## Render

このリポジトリを Render の Web Service に接続してください。

- Runtime: Docker
- Health Check Path: `/health`

デプロイ後:

- `https://あなたのURL.onrender.com/health`
- `https://あなたのURL.onrender.com/docs`

## CORS

開発中は `ALLOWED_ORIGINS=*` です。

公開時は Render の Environment で以下のように設定してください。

```text
ALLOWED_ORIGINS=https://あなたのStackBlitz公開URL
```
