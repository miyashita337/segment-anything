
# Claude Request

Hi Claude,

I forked the [Segment Anything project](https://github.com/facebookresearch/segment-anything) and created a custom version for manga character extraction on this branch:  
https://github.com/miyashita337/segment-anything/tree/manga-character-extraction

This version combines SAM with YOLO to extract characters from manga images more effectively.

Now, I'd like to restructure this into a Claude-compatible project using the conventions described in CLAUDE.md:  
https://github.com/modelcontextprotocol/python-sdk/blob/main/CLAUDE.md

---

## 📌 Goals

Please refactor the codebase into the Claude Code directory structure with:

- 🪝 `hooks/start.py`: load both the SAM and YOLO models
- 📦 `commands/extract_character.py`: a command to run character extraction given an image path
- 🧠 `models/sam_wrapper.py` and `models/yolo_wrapper.py`: for model loading and inference
- 🧰 `utils/preprocessing.py`, `utils/postprocessing.py`: image handling, mask processing, etc.
- 💬 `prompts/extract.prompt.md`: a brief explanation of the extraction command
- 🧪 `tests/test_extract.py`: a simple test with a sample manga image
- 🗂 `.claude`: metadata for the Claude project

The goal is to make `/extract_character` a fully functional Claude Code command that outputs the character mask.

---

## 🧾 Notes

- You can reuse and modularize the logic from `sam_yolo_character_segment.py`
- You may ignore the `demo/` folder (React UI) for this task
- Please use `assets/masks1.png` etc. as test examples
- The evaluation results and pipeline notes are in `evaluation_report_20250705.md` and `reproduce_pipeline_20250705.md`

I’d like the final structure to be clean, testable, and extensible for further improvements.

Thanks!
