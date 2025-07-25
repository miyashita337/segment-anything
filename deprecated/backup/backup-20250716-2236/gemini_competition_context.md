# Gemini競技用コンテキスト情報

## 競技概要
Claude vs Gemini でYOLOモデルを使った人物検出の性能・結果を比較する

## プロジェクト前提
- **ディレクトリ**: `/mnt/c/AItools/segment-anything`
- **使用モデル**: YOLO v8 (yolov8n.pt)、SAM ViT-H
- **環境**: WSL2 Ubuntu、RTX 4070 Ti SUPER、CUDA 12.6
- **テストデータ**: kana04データセット（漫画キャラクター画像）

## Claudeの実行結果
### 環境構築
- Python 3.10.12
- PyTorch 2.7.1+cu126
- ultralytics 8.3.161
- CUDA利用可能

### テスト結果
1. **単体テスト**: ✅ YOLO yolov8n.pt loaded on cuda
2. **統合テスト**: ✅ SAM+YOLO integrated system 
3. **対話式テスト**: 5枚の画像で8人検出
   - 0003.jpg: 1人 (信頼度0.283)
   - 0005.jpg: 3人 (信頼度0.450)
   - 0010.jpg: 4人 (信頼度0.424)

## Geminiへの競技タスク
以下のコードを実行して結果を比較：

```python
import sys
sys.path.append('.')
from models.yolo_wrapper import YOLOModelWrapper
import cv2
import os

# YOLOモデル初期化・テスト
yolo = YOLOModelWrapper(model_path='yolov8n.pt', confidence_threshold=0.1)
yolo.load_model()

# kana04の同一画像でテスト
test_files = ['0003.jpg', '0005.jpg', '0010.jpg']
test_dir = '/mnt/c/AItools/lora/train/yado/org/kana04'

for filename in test_files:
    filepath = os.path.join(test_dir, filename)
    image = cv2.imread(filepath)
    persons = yolo.detect_persons(image)
    print(f'{filename}: {len(persons)}人検出')
    if persons:
        best = persons[0]
        print(f'  最高信頼度: {best["confidence"]:.3f}')

yolo.unload_model()
```

## 評価基準
1. **検出人数の一致性**
2. **信頼度スコアの精度**
3. **実行時間の効率性**
4. **エラーハンドリング**
5. **コードの簡潔性**

Claude の実装では YOLOModelWrapper クラスを使用し、統合されたSAM+YOLOシステムでの動作を確認済み。