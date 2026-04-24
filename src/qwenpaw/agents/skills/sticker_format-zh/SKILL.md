---
name: sticker_format
description: "當你需要將任何圖片（PNG、JPG、WEBP、GIF、BMP）轉成符合 messenger sticker 格式嘅 WebP 檔時用呢個 skill。觸發情況：用戶叫你『整個 sticker』、『發呢張做 sticker』、『將呢張圖整成 sticker』；你啱啱用 codex image gen / dalle / 其他 model 生咗圖，要餵入 Signal/WhatsApp sticker pipeline（signal_create_sticker_pack、signal_add_stickers_to_pack、WhatsApp 嘅 `.sticker.webp`）；之前 call 過 sticker tool 被 reject 話『not a WebP』、『512x512』、『too large』；想喺 upload 之前 pre-check 張圖合唔合格。唔好用喺一般 thumbnail / 非 sticker resize 任務。"
license: Proprietary. See repo LICENSE.
metadata:
  builtin_skill_version: "1.0"
---

# Sticker 格式轉換

將任何圖片轉成合格 messenger sticker WebP：

* **512×512** 正正方方（source 唔係 square 就透明 padding）
* **RIFF/WEBP** magic（signal_create_sticker_pack preflight 會 check）
* **≤300 KB**（Signal 上限；我哋先嘗試 ≤100 KB，逐級降 quality 先加）
* 有 alpha 就保留

## 幾時用

image gen → sticker send 之間幾乎一定要轉一轉。典型 pipeline：

1. Agent 用 codex image gen 生圖 → `/tmp/out.png`（多數 1024×1024 PNG）
2. **run 呢個 skill** → `/tmp/out.sticker.webp`
3. 餵入 `signal_create_sticker_pack` / `signal_add_stickers_to_pack`（Signal）
   或 `send_file_to_user`（WhatsApp — `.sticker.webp` suffix
   會自動行 sticker send path）

## 點樣行個 conversion

喺 skill directory 入面：

```bash
python scripts/prepare_sticker_webp.py --input /path/to/source.png
# → 寫 /path/to/source.sticker.webp
```

指定 output path：

```bash
python scripts/prepare_sticker_webp.py \
    --input /path/to/source.png \
    --output /path/to/out.sticker.webp
```

Exit code：

* `0`：success，寫咗個 sticker 檔。
* `1`：input 唔存在 / 唔係合法圖 / quality=35 仍然超過 300 KB；stderr 會寫原因。

## Input 處理

* **PNG/JPG**：直接用。
* **Animated GIF / WebP**：淨係取第一 frame — animated sticker 需要另一條 libwebp 路徑（未 wire）。
* **冇 alpha 嘅 JPG**：強制轉 RGBA 令透明 pad work，視覺上冇改動。

## Failure mode

* **輸出太大**：複雜 gradient / 照片 noise 抗壓。Quality step：95 → 85 → 75 → 65 → 55 → 45 → 35。全部 fail 就要簡化 source（拉平背景、減細節）先 retry。
* **Input 唔啱**：`unidentified image` 即係 Pillow 解唔到，check 返個檔真係唔係你諗嗰款。

## Python 接口

如果 agent 喺 CoPaw env 入面跑，可以直接 import 唔使 shell：

```python
from qwenpaw.agents.tools.sticker_convert import prepare_sticker_webp
prepare_sticker_webp("/tmp/out.png", "/tmp/out.sticker.webp")
```

個 CLI script 係畀 sandbox / 非 CoPaw venv 用。
