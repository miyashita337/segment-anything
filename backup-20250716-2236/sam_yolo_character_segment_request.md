# 🧠 Claude Code向け 実装依頼書：SAM + YOLOv8 による漫画キャラ切り出しパイプライン

## 🎯 目的
漫画画像から **キャラクターのみを高精度に切り出す**ことを目的とした、セグメンテーションパイプラインのスクリプトを作成したい。

## 📌 実現したい構成（概要）

1. 入力画像：白黒グレースケールの漫画画像（JPEG/PNG）
2. セグメント手法：Metaの Segment Anything Model（SAM）を利用
3. フィルタ手法：YOLOv8（学習済モデル）を使ってマスク候補の中から「人物らしい領域」を選択
4. 出力形式：キャラクター部分をマスクで切り出し、JPEGまたはPNGとして保存

---

## 🧱 構成要素

### Segment Anything Model（SAM）
- 使用モデル：`vit_h`
- モデル重み：ローカルに配置済（例: `sam_vit_h_4b8939.pth`）
- 出力：複数のマスク（候補領域）

### YOLOv8
- モデル用途：SAM出力マスクのバウンディングボックス内に「人物らしい」スコアが高いものを検出
- モデル種類：YOLOv8n/y/m いずれかの人物検出モデルでOK（学習済のもので軽量ならなお良い）

---

## 🖥️ 実装要件

### スクリプト名案：`sam_yolo_character_segment.py`

### 実行形式：
```bash
# 対話形式：1枚の画像を確認しながら最適化
python sam_yolo_character_segment.py --mode interactive --input image.jpg

# バッチ形式：自動で複数枚を一気に処理
python sam_yolo_character_segment.py --mode batch --input_dir ./manga_images/ --output_dir ./results/

# 選択形式：マスク候補をクリックで手動選択（ML訓練データ収集用）
python sam_yolo_character_segment.py --mode choice --input_dir ./manga_images/ --output_dir ./results/
```

---

## 🤖 処理フロー詳細

### 基本フロー
1. 入力画像を読み込む（白黒 or グレースケール前提、カラー画像は自動スキップ）
2. SAMで複数マスク候補を生成
3. 各マスク領域に対応するBBoxを抽出
4. 各BBoxをYOLOv8で人物判定し、スコアでランキング
5. 該当マスクで元画像を切り出し・保存

### モード別詳細
- **Interactive**: ユーザーに選択肢を表示し、手動選択も可能
- **Batch**: 最高スコアのマスクを自動選択して一括処理
- **Choice**: 10色でマスクを表示し、クリックで複数選択・結合可能

### 選択モード（Choice）の特徴
- 上位10個のマスクを10色で表示
- 右クリックで複数選択（追加/削除）
- 左クリックで確定（単一選択または複数結合）
- パフォーマンス監視機能でボトルネック分析
- プレビュー画像自動生成・7日間保持

---

## ✅ 想定オプション・設定

| オプション名 | 内容 |
|--------------|------|
| `--mode` | 処理モード（`interactive`/`batch`/`choice`） |
| `--model-type` | SAMモデルの種類（デフォルト：`vit_h`） |
| `--sam-checkpoint` | SAMのチェックポイントファイルのパス |
| `--yolo-model` | YOLOv8モデルのパス or モデル名（例：`yolov8n.pt`） |
| `--input` | 単一画像のパス（対話形式） |
| `--input_dir` | バッチ処理時の画像ディレクトリ |
| `--output_dir` | 切り出した画像の保存先（デフォルト：`./results`） |
| `--mask_choice` | 手動でマスク番号を指定（0-4） |
| `--score-threshold` | YOLOv8の人物スコア閾値（デフォルト：0.15） |
| `--anime-mode` | アニメ・マンガ専用YOLOモデルを使用 |

---

## 💬 備考・注意点

- 入力画像は白黒・グレースケールの漫画コマ画像（背景はほぼ白、線画中心）
  - キャラが検出できなかった場合は切り出しをしない
- 人物のマスク抽出が精度において最優先であり、背景や小物の混入はなるべく避けたい
- 初期は対話モードで挙動を試し、満足したらバッチ実行で大量処理へ移行したい
  - バッチモードはディレクトリ以下を走査して、出力を同じ階層にさせる
  - 例)
   [input]  "C:\AItools\lora\train\yadokugaer\org"
   [output]  "C:\AItools\lora\train\yadokugaer\clipped_boundingbox"
   ※input 以下のディレクトリ構造や画像名をoutput以下と同じにさせる
---

## 🙏 最終目的
「自動で漫画のキャラクター領域を抽出し、背景透過 or 切り抜き画像を大量に生成できるパイプライン」を完成させたい。
