# Deprecated Tools

## 🚫 **警告: このディレクトリ内のファイルは非推奨です**

### 📋 **非推奨ファイル一覧**

#### `sam_yolo_character_segment.py`
- **移動日**: 2025-08-11
- **理由**: ユーザー指定により非推奨化
- **代替手段**: `features/extraction/commands/extract_character.py` を使用
- **警告**: このファイルをtools/coreに戻すことは禁止されています

### 🛡️ **保護機能**

このディレクトリからファイルを移動して再使用しようとした場合、自動的に警告が発生し、実行が停止されます。

### 📝 **使用方法**

```bash
# ❌ 非推奨（使用禁止）
python tools/core/sam_yolo_character_segment.py

# ✅ 正しい方法
python features/extraction/commands/extract_character.py
```

---
**注意**: これらのファイルは技術的な理由または設計方針の変更により非推奨となっています。使用しないでください。