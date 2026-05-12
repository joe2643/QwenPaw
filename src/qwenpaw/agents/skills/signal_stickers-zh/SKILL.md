---
name: signal_stickers
description: "當你要透過 Signal channel 處理 sticker 時用呢個 skill：send 特定 sticker 去當前對話或指定對象、list bot 帳號有嘅 pack、install 人哋用 signal.art link 分享嘅 pack、用本地 webp 整新 pack、擴充現有 pack。觸發情況：用戶話『send 個 sticker』、『用 sticker 答返我』、『用呢啲圖整 sticker pack』、『install 呢個 sticker pack (signal.art link)』、『你有咩 sticker pack』；剛生成 / 收到嘅圖要上到 Signal 做 sticker。Signal pack ID 係 immutable — 永遠先 discover 再 send 唔好 guess。將原圖轉 sticker 格式用 `sticker_format` skill。唔好用喺 WhatsApp sticker（WhatsApp 用 `.sticker.webp` filename convention 經 `send_file_to_user`）。"
license: Proprietary. See repo LICENSE.
metadata:
  builtin_skill_version: "1.1"
---

# Signal stickers — 端到端流程

Signal sticker 棧由七個 `signal_*` tool 加一個持久 registry 嘅
`{media_dir}/sticker_packs.json` 組成。呢個 skill 將佢哋串埋：
**discover → preview → send**，加兩條 build path
（**install 別人分享嘅 pack** 同 **create / grow 自己 pack**）。

## 流水線概覽

```
┌─────────────────────┐   ┌─────────────────────┐   ┌────────────────────┐
│ signal_list_sticker │ → │ signal_preview_     │ → │ signal_send_       │
│     _packs          │   │   sticker           │   │   sticker          │
└─────────────────────┘   └─────────────────────┘   └────────────────────┘
       發掘                    確認圖                     發送

Build path:
  人哋分享:   signal_install_sticker_pack(pack_id, pack_key)
  新 pack:    sticker_format → signal_create_sticker_pack(title, author, stickers)
  擴充 pack:  signal_add_stickers_to_pack(base_pack_id, new_stickers)
```

> **預設 sticker 格式係 PNG。**  Signal Android 收到非 Signal
> Desktop creator 整出嚟嘅 WebP sticker 會 render 做 voice-message
> blob（VP8L、VP8X 都 reproduce）。所有 sticker tool 都收 PNG **或**
> WebP，而 `signal_prepare_sticker_webp` 預設輸出 PNG。冇特別原因
> 唔好改用 WebP。

`signal_send_sticker(to=None)` **自動解析成當前對話 context**
（看 runner 嘅 `channel_meta`：群組時取 `group_id`，否則取 DM
source）。只有轉發去另一對話先需要傳 `to`。

---

## 常見工作流

### 發送 bot 已有嘅 sticker

1. `signal_list_sticker_packs()` — 返 JSON array，含
   `{pack_id, title, author, installed, source, label, sticker_count, stickers: [{id, emoji, ...}]}`。
2. 掃 emoji 搵啱答嘅 pack/sticker。例如用戶要 🦀 反應，搵第一
   個 sticker emoji 包含 🦀 嘅 entry。
3. *（第一次見嗰個 pack 強烈建議）* `signal_preview_sticker(pack_id, sticker_id)` —
   返 webp 做 `ImageBlock`，你可以目視確認合適。
4. `signal_send_sticker(pack_id, sticker_id)` — `to` 省略 → 發
   入當前對話。

### 群組答 sticker

Signal 群組本身需要 @mention 先喚醒 bot，所以收到嘅 request
已經帶 `channel_meta.group_id`。流程同 DM 一模一樣：

```
signal_send_sticker(pack_id, sticker_id)
# 不傳 to / is_group — runner 自動取 group_id
```

### 安裝別人分享嘅 pack（signal.art link）

```
# URL 形如 https://signal.art/addstickers/#pack_id=<hex>&pack_key=<hex>
signal_install_sticker_pack(pack_id=<hex>, pack_key=<hex>, label="friends-memes")
```

裝完後 `signal_list_sticker_packs` 會見到 `installed=true`。
`label` 可選但強烈建議設：佢 registry 入面做人類可讀 handle
（`"friends-memes"` 好過下次要 send 時搵個 32 字 hex）。

### 由本地圖整新 pack（譬如 AI 生成）

每張 sticker 必須已係合規 sticker-format **PNG 或 WebP**。任何
唔係 512×512 / PNG-或-WebP / ≤ 300 KB 嘅圖，先行 `sticker_format`
skill。預設 PNG —— Signal Android 會把 user-uploaded WebP 渲染做
voice message，所以冇特別原因都用 PNG：

```
# 1. 每張圖轉 sticker 格式（預設 PNG）
signal_prepare_sticker_webp(input_path="/tmp/smug.png")
# → "/tmp/smug.sticker.png"
signal_prepare_sticker_webp(input_path="/tmp/pout.png")
# → "/tmp/pout.sticker.png"

# 2. upload 做新 pack
signal_create_sticker_pack(
    title="Agent reactions",
    author="CoPaw",
    label="agent-reactions-v1",        # 可選但建議
    stickers=[
        {"path": "/tmp/smug.sticker.png", "emoji": "😏"},
        {"path": "/tmp/pout.sticker.png", "emoji": "😤"},
    ],
)
# → JSON { pack_id, pack_key, install_url, stickers: [{id, emoji, source_path, staged_path}] }
```

`signal_create_sticker_pack` 會憑 magic byte 自動分辨每個 input
係 PNG 定 WebP，並喺 manifest 寫返正確 `contentType`
（`image/png` 或 `image/webp`），所以同一個 pack 入面可以混
PNG 同 WebP —— 但實踐上全 PNG 最簡單。

回傳 payload 裏 `stickers[i].id` 先係 `signal_send_sticker` 要
用嘅 id — 唔好按入參順序去估。

### 擴充現有 pack

Signal pack 係**協議級不可變**。`signal_add_stickers_to_pack`
底層會重新 build 一個取代舊嘅 pack：

```
signal_add_stickers_to_pack(
    base_pack_id="<現有 hex>",
    new_stickers=[
        {"path": "/tmp/facepalm.sticker.png", "emoji": "🤦"},
    ],
)
# → JSON { pack_id (新), pack_key (新), previous_pack_id: <舊>,
#          stickers: [...舊 + 新，重新編號...] }
```

舊 pack 對已發 message 繼續有效；之後嘅 send 應該用新
`pack_id`。Registry 記住 `previous_pack_id` / `superseded_by`，
多次擴充後你仍然可以追溯血緣。

---

## Do / Don't

**Do**

- 從冇用過嘅 pack 第一次發 sticker 前先 preview。一次 50-300 KB
  fetch 成本細，發錯 sticker 社交成本高。
- 每個 `install` / `create` / `add_stickers` call 都設 `label`。
  `signal_list_sticker_packs` 會 surface label，唔使記 hex id。
- 未轉格式嘅圖一定先走 `sticker_format` skill。
  `signal_create_sticker_pack` 嘅 preflight 會 reject 任何唔係
  512×512 / PNG-或-WebP / ≤ 300 KB 嘅輸入。
- `sticker_format` / `signal_prepare_sticker_webp` 預設 PNG，
  畀 Signal 用就咁好。除非你同時要餵 WhatsApp（要
  `.sticker.webp` filename convention）先轉做 WebP。

**Don't**

- 答當前對話唔好傳 `to` 去 `signal_send_sticker`。自動 resolve
  比你自己由 context 提取 sender id 可靠。
- 唔好為同一主題每張新 sticker call 一次
  `signal_create_sticker_pack`。用 `signal_add_stickers_to_pack`
  等 registry 按 lineage 管理，唔好喺 Signal CDN 堆一堆細 pack。
- 絕對唔好用 `execute_shell_command` 自己起 `signal-cli`。會
  spawn 第二個 process 搶 account lock，卡死跑緊嘅 daemon。

---

## 錯誤對照表

| 症狀                                                               | 常見原因                                                                              |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| `Error: sticker dimensions must be 512x512`                        | 冇走 `sticker_format` / `signal_prepare_sticker_webp`。                               |
| `Error: sticker is not a PNG or WebP`                              | Magic byte 唔係 `\x89PNG\r\n\x1a\n` 或 `RIFF...WEBP`，重走 `sticker_format`。           |
| Sticker 喺 Signal Android 變咗 **voice message**                    | Pack 用咗 user-uploaded **WebP**，重新整 pack 用 PNG（`--format png`，預設）。           |
| `Upload error (maybe image size too large): Unable to parse entity`| signal-cli 係 GraalVM native binary 唔係 JAR，運維層修（睇 channel docs）。           |
| `Error: base_pack_id not found in this account's sticker packs`    | 先 `signal_install_sticker_pack(pack_id, pack_key)`，pack 本地未知。                  |
| `Error: ``to`` omitted but no current Signal request context`      | 喺 Signal 對話以外 call `signal_send_sticker`。明傳 `to`。                            |
| log 見 `signal: require_mention drop ... mentions=[]`               | 收件側問題，唔關 send 事 — signal-cli 版本冇 emit 結構化 mentions。                   |
