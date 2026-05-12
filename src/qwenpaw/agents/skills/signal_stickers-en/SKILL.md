---
name: signal_stickers
description: "Use this skill whenever you need to work with Signal stickers via the Signal channel: send a specific sticker to the current chat or a named recipient; discover what sticker packs are already available on the bot's account; install a pack someone shared via a signal.art link; create a new pack from local webp files; grow an existing pack by adding stickers. Triggers: user says 'send me a sticker', 'reply with a sticker', 'make a sticker pack from these images', 'install this sticker pack (signal.art link)', 'what sticker packs do you have'; you just generated or received images you want to ship as a Signal sticker pack. Signal pack IDs are immutable — always prefer discover-then-send over guessing. For conversion of a raw image into sticker-format WebP, use the `sticker_format` skill first. Do NOT use for WhatsApp stickers (WhatsApp uses filename convention `.sticker.webp` via `send_file_to_user`)."
license: Proprietary. See repo LICENSE.
metadata:
  builtin_skill_version: "1.1"
---

# Signal stickers — end-to-end workflow

The Signal sticker stack is exposed as seven `signal_*` tools plus
a persistent registry at `{media_dir}/sticker_packs.json` that
tracks every pack CoPaw uploaded or installed. This skill ties
them together: **discover → preview → send**, and the two build
paths (**install a shared pack** and **create/grow your own**).

## Pipeline at a glance

```
┌─────────────────────┐   ┌─────────────────────┐   ┌────────────────────┐
│ signal_list_sticker │ → │ signal_preview_     │ → │ signal_send_       │
│     _packs          │   │   sticker           │   │   sticker          │
└─────────────────────┘   └─────────────────────┘   └────────────────────┘
       discover                 confirm image               ship

Build paths:
  shared pack:     signal_install_sticker_pack(pack_id, pack_key)
  new pack:        sticker_format → signal_create_sticker_pack(title, author, stickers)
  grow a pack:     signal_add_stickers_to_pack(base_pack_id, new_stickers)
```

> **Default sticker format is PNG.**  Signal Android renders
> user-uploaded WebP stickers as voice-message blobs (independent
> of VP8L vs VP8X encoding) unless the WebP came from Signal
> Desktop's own creator.  All sticker tools accept PNG **or** WebP,
> and `signal_prepare_sticker_webp` defaults to PNG output.  Stick
> with PNG unless you have a specific reason not to.

`signal_send_sticker(to=None)` **auto-resolves to the current chat
context** via the runner's `channel_meta` (picks `group_id` when
the inbound was a group message, else the DM source). Pass `to`
explicitly only when forwarding to a different conversation.

---

## Common workflows

### Send a sticker the bot already has

1. `signal_list_sticker_packs()` — returns a JSON array of
   `{pack_id, title, author, installed, source, label, sticker_count, stickers: [{id, emoji, ...}]}`.
2. Scan for the pack/emoji that fits the reply. Example: user
   asked for a 🦀 reaction; find the first entry where any
   sticker's emoji contains 🦀.
3. *(Optional but recommended for the first time you see a pack)*
   `signal_preview_sticker(pack_id, sticker_id)` — returns the
   webp as an `ImageBlock` so you can visually confirm it fits
   the moment.
4. `signal_send_sticker(pack_id, sticker_id)` — `to` omitted →
   sent to the current chat.

### Reply with a sticker to a Signal group

Signal groups require a **mention** for the bot to be woken up at
all, so the incoming request already came with `channel_meta.group_id`
populated. Exactly the same flow as DM:

```
signal_send_sticker(pack_id, sticker_id)
# no to / is_group — runner resolves to group_id automatically
```

### Install a pack someone shared (signal.art link)

```
# URL looks like https://signal.art/addstickers/#pack_id=<hex>&pack_key=<hex>
signal_install_sticker_pack(pack_id=<hex>, pack_key=<hex>, label="friends-memes")
```

After install, the pack is visible in `signal_list_sticker_packs`
with `installed=true`. `label` is optional but highly recommended:
it becomes a human-readable handle in the registry (`"friends-memes"`
is easier than a 32-char hex id the next time you need to send).

### Create a new pack from local images (e.g. AI-generated)

Every sticker MUST already be a valid sticker-format **PNG or
WebP**. Run the `sticker_format` skill first on anything that
isn't already 512×512 / PNG-or-WebP / ≤ 300 KB.  PNG is the
default — Signal Android mis-renders user-uploaded WebP as voice
messages, so prefer PNG unless you have a specific reason not to:

```
# 1. Convert each image to sticker format (default = PNG)
signal_prepare_sticker_webp(input_path="/tmp/smug.png")
# → "/tmp/smug.sticker.png"
signal_prepare_sticker_webp(input_path="/tmp/pout.png")
# → "/tmp/pout.sticker.png"

# 2. Upload them as a new pack
signal_create_sticker_pack(
    title="Agent reactions",
    author="CoPaw",
    label="agent-reactions-v1",        # optional but recommended
    stickers=[
        {"path": "/tmp/smug.sticker.png",  "emoji": "😏"},
        {"path": "/tmp/pout.sticker.png",  "emoji": "😤"},
    ],
)
# → JSON { pack_id, pack_key, install_url, stickers: [{id, emoji, source_path, staged_path}] }
```

`signal_create_sticker_pack` detects each input file's format
(PNG vs WebP) from its magic bytes and writes the correct
`contentType` (`image/png` or `image/webp`) per sticker in the
manifest, so you can mix formats inside the same pack — though in
practice all-PNG is the simplest path.

The returned `stickers[i].id` is what you pass to `signal_send_sticker`
— never guess from the order of your input list.

### Grow an existing pack

Signal packs are **protocol-immutable**. `signal_add_stickers_to_pack`
rebuilds a superseding pack under the hood:

```
signal_add_stickers_to_pack(
    base_pack_id="<existing hex>",
    new_stickers=[
        {"path": "/tmp/facepalm.sticker.png", "emoji": "🤦"},
    ],
)
# → JSON { pack_id (NEW), pack_key (NEW), previous_pack_id: <existing>,
#          stickers: [...existing + new, renumbered...] }
```

The old pack keeps working for messages already sent; future sends
should reference the new `pack_id`. The registry records
`previous_pack_id` / `superseded_by` so you can trace lineage
across multiple grows.

---

## Do / Don't

**Do**

- Preview before sending a sticker from a pack you haven't sent
  from before. One 50-300 KB fetch is cheap; sending the wrong
  sticker is socially expensive.
- Set a `label` on every `install` / `create` / `add_stickers`
  call. `signal_list_sticker_packs` surfaces labels so you can
  find a pack without memorising hex ids.
- Route unconverted images through the `sticker_format` skill
  first. `signal_create_sticker_pack`'s preflight rejects
  anything not exactly 512×512 / PNG-or-WebP / ≤ 300 KB.
- Default to PNG output from `sticker_format` /
  `signal_prepare_sticker_webp` for Signal.  Only switch to WebP
  if you also need to send the same file through WhatsApp's
  `.sticker.webp` filename convention.

**Don't**

- Don't pass `to` to `signal_send_sticker` when replying to the
  current chat. Auto-resolve is more reliable than you parsing
  sender IDs out of the context hint line.
- Don't re-run `signal_create_sticker_pack` for each new sticker
  of the same theme — use `signal_add_stickers_to_pack` so the
  registry tracks it as a lineage instead of creating 10 tiny
  packs on Signal's CDN.
- Don't shell out to `signal-cli` yourself via `execute_shell_command`.
  That spawns a second process contesting the account lock and
  deadlocks the running daemon.

---

## Error cheat sheet

| Symptom                                                            | Usually means                                                                              |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| `Error: sticker dimensions must be 512x512`                        | You skipped `sticker_format` skill / `signal_prepare_sticker_webp`.                        |
| `Error: sticker is not a PNG or WebP`                              | Magic bytes don't match `\x89PNG\r\n\x1a\n` or `RIFF...WEBP`. Re-run `sticker_format`.       |
| Sticker arrives as a **voice message** on Signal Android           | The pack contains user-uploaded **WebP**. Re-create the pack with PNG (`--format png`, the default).|
| `Upload error (maybe image size too large): Unable to parse entity`| The signal-cli binary is the GraalVM native build, not the JAR. Ops-level fix (see channel docs). |
| `Error: base_pack_id not found in this account's sticker packs`    | Run `signal_install_sticker_pack` with the pack_key first (pack isn't known locally).      |
| `Error: ``to`` omitted but no current Signal request context`      | You called `signal_send_sticker` outside an active Signal conversation. Pass `to` explicitly. |
| `signal: require_mention drop ... mentions=[]` in the log          | Inbound side, not send side — signal-cli version doesn't emit structured mentions.          |
