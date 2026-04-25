---
name: sticker_format
description: "當你需要將任何圖片（PNG、JPG、WEBP、GIF、BMP）轉成符合 messenger sticker 格式嘅檔時用呢個 skill。預設 output 係 PNG（Signal-friendly）；只有要餵 WhatsApp 嗰陣先用 --format webp。觸發情況：用戶叫你『整個 sticker』、『發呢張做 sticker』、『將呢張圖整成 sticker』；你啱啱用 codex image gen / dalle / 其他 model 生咗圖，要餵入 Signal/WhatsApp sticker pipeline（signal_create_sticker_pack、signal_add_stickers_to_pack、WhatsApp 嘅 `.sticker.webp`）；之前 call 過 sticker tool 被 reject 話『not PNG or WebP』、『512x512』、『too large』；想喺 upload 之前 pre-check 張圖合唔合格。唔好用喺一般 thumbnail / 非 sticker resize 任務。"
license: Proprietary. See repo LICENSE.
metadata:
  builtin_skill_version: "1.1"
---

# Sticker 格式轉換

將任何圖片轉成合格 messenger sticker 檔：

* **512×512** 正正方方（source 唔係 square 就透明 padding）
* **PNG**（預設）或 **WebP**（WhatsApp 先 opt-in）；兩者
  `signal_create_sticker_pack` preflight 都 check
* **≤300 KB**（Signal 上限；PNG `optimize=True` 通常一定夠位，
  WebP 仲有 quality ladder，先嘗試 ≤100 KB 再逐級降）
* 有 alpha 就保留

## 點解預設 PNG

Signal 規格 PNG / WebP 兩者都收，但實測 Signal **Android** 收到
**非 Signal Desktop creator 整出嚟嘅 WebP sticker** 會 render
做 voice-message blob —— VP8L 同 VP8X 都 reproduce。第三方
work 到嘅 pack（例如 LIHKG Dog）一律用 PNG，PNG 喺所有 Signal
client 都 render 得乾淨。WhatsApp 因為要靠 `.sticker.webp`
filename 行 send-as-sticker path，所以仍然要 `--format webp`。

## 幾時用

image gen → sticker send 之間幾乎一定要轉一轉。典型 pipeline：

1. Agent 用 codex image gen 生圖 → `/tmp/out.png`（多數 1024×1024 PNG）
2. **run 呢個 skill** → `/tmp/out.sticker.png`
3. 餵入 `signal_create_sticker_pack` / `signal_add_stickers_to_pack`
   （Signal — PNG 同 WebP 都收），
   或者用 `--format webp` 轉完傳俾 `send_file_to_user`（WhatsApp —
   `.sticker.webp` suffix 會自動行 sticker send path）

## 點樣行個 conversion

喺 skill directory 入面：

```bash
python scripts/prepare_sticker_webp.py --input /path/to/source.png
# → 寫 /path/to/source.sticker.png（預設 Signal-friendly PNG）
```

指定 format / output path：

```bash
# 預設 PNG，自訂 output path
python scripts/prepare_sticker_webp.py \
    --input /path/to/source.png \
    --output /path/to/out.sticker.png

# WebP — 畀 WhatsApp send-as-sticker filename convention 用
python scripts/prepare_sticker_webp.py \
    --input /path/to/source.png \
    --format webp
# → 寫 /path/to/source.sticker.webp
```

Exit code：

* `0`：success，寫咗個 sticker 檔。
* `1`：input 唔存在 / 唔係合法圖 / 仍然超過 300 KB；
  stderr 會寫原因。

## Input 處理

* **PNG/JPG**：直接用。
* **Animated GIF / WebP**：淨係取第一 frame — animated sticker 需要
  另一條 libwebp / APNG 路徑（未 wire）。
* **冇 alpha 嘅 JPG**：強制轉 RGBA 令透明 pad work，視覺上冇改動。

## Failure mode

* **PNG 太大**：好罕見；optimize PNG encoder 對 512×512 基本上
  都頂得到 300 KB 以下。如果 fail 即係 source 顏色多到無論點
  encode 都壓唔細 —— 簡化（拉平背景、減 gradient）再 retry。
* **WebP 太大**：複雜 gradient / 照片 noise 抗壓。Quality step：
  95 → 85 → 75 → 65 → 55 → 45 → 35。全部 fail 就要簡化 source
  先 retry。
* **Input 唔啱**：`unidentified image` 即係 Pillow 解唔到，check
  返個檔真係唔係你諗嗰款。

## Python 接口

如果 agent 喺 CoPaw env 入面跑，可以直接 import 唔使 shell：

```python
from qwenpaw.agents.tools.sticker_convert import prepare_sticker_webp
# 預設 PNG
prepare_sticker_webp("/tmp/out.png", "/tmp/out.sticker.png")
# 要 WebP 餵 WhatsApp 嗰陣
prepare_sticker_webp("/tmp/out.png", "/tmp/out.sticker.webp",
                     output_format="webp")
```

個 CLI script 係畀 sandbox / 非 CoPaw venv 用。

## 下一步：喺 Signal 發送

呢個 skill 只到生成合規 sticker 檔為止。Signal 側完整流程 ——
發現 pack、preview、send、由呢度嘅檔整自己 pack —— 睇
**`signal_stickers`** skill。WhatsApp 更簡單：直接將
`.sticker.webp` 傳俾 `send_file_to_user`，filename 後綴會自動行
sticker 發送路徑。
