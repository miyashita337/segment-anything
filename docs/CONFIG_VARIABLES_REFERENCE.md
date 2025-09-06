# 設定変数リファレンス

このドキュメントでは以下の変数が使用可能です：

- `${DEFAULT_INPUT_DIR}` → `/mnt/c/AItools/lora/train/yado/org/kana05`
- `${PROJECT_ROOT}` → `/mnt/c/AItools/segment-anything`
- `${PUSHOVER_CONFIG}` → `/mnt/c/AItools/segment-anything/config/pushover.json`
- `${SAM_MODEL_PATH}` → `sam_vit_h_4b8939.pth`
- `${TRACKER_WORKSPACE_BASE}` → `/mnt/c/AItools/lora/train/yado/tracker-workspace`
- `${TRACKER_WORKSPACE_ROOT}` → `/mnt/c/AItools/lora/train/yado/tracker-workspace`
- `${YOLO_MODEL_PATH}` → `yolov8x.pt`

## 使用例

```bash
# 抽出コマンド例
python features/extraction/commands/extract_character.py \
  --input_dir ${DEFAULT_INPUT_DIR} \
  --output_dir ${TRACKER_WORKSPACE_BASE}/${TRACKER_ID}/extraction/ \
  --batch
```
