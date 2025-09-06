# LoRA学習用画像評価依頼書

## 評価対象
QC品質調査で抽出した107枚のキャラクター画像をLoRA学習に使用する際の適性評価

## サンプル画像パス
- KANA08: `/mnt/c/AItools/lora/train/yado/tracker-workspace/workspace/QC-KANA08/`
- KANA05: `/mnt/c/AItools/lora/train/yado/tracker-workspace/workspace/QC-KANA05/`
- KANA07: `/mnt/c/AItools/lora/train/yado/tracker-workspace/workspace/QC-KANA07/`

## 観察された懸念点

### 1. 他キャラクターの混入
- 背景に他キャラの手足が部分的に含まれる
- メインキャラ以外の要素が混在

### 2. テキスト要素の混入
- 吹き出し（セリフ）
- 効果音・オノマトペ
- 背景文字

### 3. アスペクト比の不統一
- 縦長・横長が混在
- サイズのばらつき

### 4. その他の懸念
- 背景の一部残存
- エフェクトの混入
- 画質の不均一性

## 評価依頼項目

1. **LoRA学習への影響度評価**
   - 各懸念点がモデル学習に与える影響
   - 学習品質への影響度（高/中/低）

2. **改善提案**
   - 前処理で対処可能な項目
   - 手動選別が必要な項目

3. **使用可否判定**
   - 現状での使用可否
   - 推奨される追加処理

## 技術仕様
- 抽出方式: SAM + YOLO
- 背景処理: 白背景統一
- 形式: PNG（透明度対応）