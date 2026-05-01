# -*- coding: utf-8 -*-
# pylint: disable=too-many-branches
"""Custom memory implementation with bugfixes and extensions."""

import json
import logging
from datetime import datetime
from pathlib import Path

import aiofiles
import aiofiles.os
from agentscope.agent._react_agent import _MemoryMark  # noqa
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg

from .as_msg_handler import AsMsgHandler
from .compactor_prompts import SUMMARY_PROMPT_EN
from ..utils.estimate_token_counter import EstimatedTokenCounter

logger = logging.getLogger(__name__)

# FIFO cap on the compacted-msg-id tombstone set.
#
# **Structural tradeoff.** Sibling runs hold a pre-compaction baseline
# for at most their own lifetime (seconds to a few minutes for chat
# replies); the tombstones only need to outlive concurrent saves, not
# forever. But there is a residual eviction-then-resurrection scenario:
# if a tombstone is FIFO-evicted while a *very* long-lived sibling still
# holds the original message in its baseline, a later concurrent merge
# can re-add the compacted msg through the sibling's content list.
#
# The cap is sized so this is practically unreachable for chat workloads
# — 50000 evictions during a single sibling's overlap window would
# require ~50000 compactions (~7.5 billion input tokens) in seconds-to-
# minutes, which is far beyond any realistic conversation. We also
# WARN on the first eviction in any session so any real-world
# occurrence is visible in logs. 50000 ids ≈ 1.2 MB on disk per session
# — large but bounded.
_TOMBSTONE_CAP = 50000


class AgentContext(InMemoryMemory):
    """Extended InMemoryMemory with bugfixes and summary support."""

    def __init__(
        self,
        token_counter: EstimatedTokenCounter,
        dialog_path: str | Path | None = None,
    ):
        """Initialize the AgentContext.

        Args:
            token_counter: Token counter for measuring content length.
            dialog_path: Path to the dialog storage directory.
                If provided, messages will be persisted to jsonl
                files when cleared or compressed.
        """
        super().__init__()
        self._token_counter: EstimatedTokenCounter = token_counter
        self._msg_handler: AsMsgHandler = AsMsgHandler(token_counter)
        self._dialog_path: Path | None = (
            Path(dialog_path) if dialog_path else None
        )
        # Tombstones for compacted/cleared message ids. Persisted in
        # state_dict so concurrent same-session saves can drop these
        # ids from the on-disk content list and avoid resurrecting
        # messages that auto-compaction already removed.
        # Stored as a dict[str, None] (not a set) so insertion order is
        # preserved — that lets us FIFO-evict the oldest tombstones once
        # the count exceeds ``_TOMBSTONE_CAP``.
        self._compressed_msg_ids: dict[str, None] = {}
        # Cumulative count of tombstones FIFO-evicted from this session.
        # Persisted so it survives reload. We use this to (a) emit a
        # one-time WARNING the first time an eviction happens (so the
        # eviction-then-resurrection scenario is visible in production
        # logs) and (b) signal stale-baseline risk to the merge code in
        # ``session.py`` when a sibling save is reconciled.
        self._compressed_msg_evicted_count: int = 0

    async def _append_messages_to_dialog(self, messages: list[Msg]) -> int:
        """Append messages to dialog storage file.

        Saves messages to jsonl files named by message date (YYYY-mm-dd.jsonl).
        Each line is a JSON representation of a message.
        Messages are grouped by their timestamp date.

        Args:
            messages: List of messages to append to the dialog file.

        Returns:
            Number of messages successfully appended.
        """
        if not messages:
            return 0

        if self._dialog_path is None:
            logger.warning(
                "dialog_path is not set, skipping dialog persistence",
            )
            return 0

        # Ensure dialog directory exists
        try:
            await aiofiles.os.makedirs(self._dialog_path, exist_ok=True)
        except Exception as e:
            logger.exception(
                f"Failed to create dialog directory {self._dialog_path}: {e}",
            )
            return 0

        # Group messages by date (extracted from timestamp)
        # timestamp format: "YYYY-mm-dd HH:MM:SS.fff"
        messages_by_date: dict[str, list[Msg]] = {}
        for msg in messages:
            try:
                if msg.timestamp:
                    # Extract date part from timestamp
                    date_str = msg.timestamp.split()[0]  # "YYYY-mm-dd"
                else:
                    date_str = datetime.now().strftime("%Y-%m-%d")

                if date_str not in messages_by_date:
                    messages_by_date[date_str] = []
                messages_by_date[date_str].append(msg)
            except Exception as e:
                logger.warning(
                    f"Failed to process message timestamp: {e}, "
                    f"using today's date",
                )
                date_str = datetime.now().strftime("%Y-%m-%d")
                if date_str not in messages_by_date:
                    messages_by_date[date_str] = []
                messages_by_date[date_str].append(msg)

        # Append messages to corresponding date files
        # (sorted by timestamp within each date)
        total_count = 0
        for date_str, msgs in messages_by_date.items():
            # Sort messages by timestamp within the same date
            try:
                msgs_sorted = sorted(msgs, key=lambda m: m.timestamp or "")
            except Exception as e:
                logger.warning(f"Failed to sort messages by timestamp: {e}")
                msgs_sorted = msgs

            filename = f"{date_str}.jsonl"
            filepath = self._dialog_path / filename

            try:
                async with aiofiles.open(
                    filepath,
                    mode="a",
                    encoding="utf-8",
                ) as f:
                    for msg in msgs_sorted:
                        msg_dict = msg.to_dict()
                        await f.write(
                            json.dumps(msg_dict, ensure_ascii=False) + "\n",
                        )
                        total_count += 1
                logger.info(
                    f"Appended {len(msgs_sorted)} messages to {filepath}",
                )
            except Exception as e:
                logger.exception(
                    f"Failed to append messages to dialog "
                    f"file {filepath}: {e}",
                )

        return total_count

    async def get_memory(
        self,
        prepend_summary: bool = True,
        **_kwargs,
    ) -> list[Msg]:
        """Get the messages from the memory by mark (if provided).

        Args:
            prepend_summary: Whether to prepend compressed summary
            **_kwargs: Additional keyword arguments (ignored)

        Returns:
            List of filtered messages
        """
        filtered_content = [
            (msg, marks)
            for msg, marks in self.content
            if _MemoryMark.COMPRESSED not in marks
        ]

        if prepend_summary and self._compressed_summary:
            return [
                Msg(
                    "user",
                    SUMMARY_PROMPT_EN.format(summary=self._compressed_summary),
                    "user",
                ),
                *[msg for msg, _ in filtered_content],
            ]

        return [msg for msg, _ in filtered_content]

    def get_compressed_summary(self) -> str:
        """Get the compressed summary of the memory."""
        return self._compressed_summary

    def state_dict(self) -> dict:
        """Get the state dictionary for serialization."""
        return {
            "content": [[msg.to_dict(), marks] for msg, marks in self.content],
            "_compressed_summary": self._compressed_summary,
            # Preserve insertion order on disk so FIFO eviction stays
            # meaningful across save/load round-trips.
            "_compressed_msg_ids": list(self._compressed_msg_ids.keys()),
            "_compressed_msg_evicted_count": self._compressed_msg_evicted_count,
        }

    # pylint: disable=attribute-defined-outside-init
    def load_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        """Load the state dictionary for deserialization."""
        if strict and "content" not in state_dict:
            raise KeyError(
                "The state_dict does not contain 'content' key "
                "required for InMemoryMemory.",
            )

        self.content = []  # pylint: disable=attribute-defined-outside-init
        for item in state_dict.get("content", []):
            if isinstance(item, (tuple, list)) and len(item) == 2:
                msg_dict, marks = item
                msg = Msg.from_dict(msg_dict)
                self.content.append((msg, marks))

            elif isinstance(item, dict):
                # For compatibility with older versions
                msg = Msg.from_dict(item)
                self.content.append((msg, []))

            else:
                raise ValueError(
                    "Invalid item format in state_dict for InMemoryMemory.",
                )

        self._compressed_summary = state_dict.get("_compressed_summary", "")
        loaded_ids = state_dict.get("_compressed_msg_ids", []) or []
        # Rebuild as insertion-ordered dict so a load preserves the
        # FIFO ordering that the on-disk list was written with.
        self._compressed_msg_ids = {}
        for raw in loaded_ids:
            if not raw:
                continue
            self._compressed_msg_ids[str(raw)] = None
        self._compressed_msg_evicted_count = int(
            state_dict.get("_compressed_msg_evicted_count", 0) or 0,
        )
        self._trim_compressed_msg_ids()

    async def mark_messages_compressed(
        self,
        messages: list[Msg],
    ) -> int:
        """Mark messages as compressed, persist them to dialog,
        and remove from memory.

        This method:
        1. Persists the given messages to the dialog storage
        2. Removes them from memory
        3. Records msg ids as tombstones so a concurrent same-session
           save cannot resurrect them from the stale on-disk state.

        Args:
            messages: List of messages to mark as compressed.

        Returns:
            Number of messages marked as compressed.
        """
        if not messages:
            return 0

        # Persist messages to dialog storage instead of compressed
        await self._append_messages_to_dialog(messages)

        # Remove messages from memory. Preserve the input batch order
        # when inserting into the tombstone dict so FIFO eviction reflects
        # the order in which messages were actually compacted (a ``set``
        # would scramble it and make the eldest-evicted ID arbitrary).
        ordered_msg_ids: list[str] = []
        seen: set[str] = set()
        for msg in messages:
            if not msg.id:
                continue
            if msg.id in seen:
                continue
            ordered_msg_ids.append(msg.id)
            seen.add(msg.id)
        initial_size = len(self.content)
        self.content = [
            (msg, marks)
            for msg, marks in self.content
            if msg.id not in seen
        ]
        removed_count = initial_size - len(self.content)
        for msg_id in ordered_msg_ids:
            self._compressed_msg_ids[msg_id] = None
        self._trim_compressed_msg_ids()

        logger.info(
            f"Marked {removed_count} messages as compressed "
            f"and removed from memory",
        )
        return removed_count

    def _trim_compressed_msg_ids(self) -> None:
        """FIFO-evict the oldest tombstones once over ``_TOMBSTONE_CAP``.

        Called from any path that mutates ``_compressed_msg_ids``. The
        cap protects long-lived sessions from unbounded session-state
        growth while keeping more than enough headroom for sibling runs
        to finish saving (sibling runs are seconds-long; the cap covers
        many compaction cycles' worth of overlapping sibling baselines).

        Increments :attr:`_compressed_msg_evicted_count` and emits a
        WARNING the first time an eviction happens for this session, so
        the residual eviction-then-resurrection scenario (a tombstone
        evicted while a long-lived sibling still holds the original
        message) is visible in production logs.
        """
        excess = len(self._compressed_msg_ids) - _TOMBSTONE_CAP
        if excess <= 0:
            return
        was_zero = self._compressed_msg_evicted_count == 0
        # ``dict`` preserves insertion order; popping the first ``excess``
        # keys removes the oldest tombstones.
        for key in list(self._compressed_msg_ids)[:excess]:
            self._compressed_msg_ids.pop(key, None)
        self._compressed_msg_evicted_count += excess
        if was_zero:
            logger.warning(
                "Tombstone cap reached: evicted %d compressed-msg-id "
                "tombstone(s); cap=%d. Long-lived sibling runs holding "
                "a pre-eviction baseline may resurrect compacted msgs "
                "on save (see agent_context._TOMBSTONE_CAP).",
                excess,
                _TOMBSTONE_CAP,
            )

    def clear_compressed_summary(self):
        """Clear the compressed summary."""
        self._compressed_summary = (
            ""  # pylint: disable=attribute-defined-outside-init
        )

    async def clear_content(self):
        """Persist all messages to dialog storage and clear the content.

        This method:
        1. Persists all messages in memory to the dialog storage
        2. Records msg ids as tombstones so a concurrent same-session
           save cannot resurrect them
        3. Clears the in-memory content
        """
        # Persist all messages to dialog storage
        if self.content:
            messages = [msg for msg, _ in self.content]
            await self._append_messages_to_dialog(messages)
            for msg, _ in self.content:
                if msg.id:
                    self._compressed_msg_ids[msg.id] = None
            self._trim_compressed_msg_ids()

        # Clear in-memory content
        self.content.clear()
        logger.info("Cleared all messages from memory")

    async def estimate_tokens(self, max_input_length: int) -> dict:
        """Estimate token usage for current memory.

        Args:
            max_input_length: Max input length for context usage calculation.

        Returns:
            Dict containing detailed token statistics:
            - total_messages: Number of messages
            - compressed_summary_tokens: Tokens in compressed summary
            - messages_tokens: Tokens in messages
            - estimated_tokens: Total estimated tokens
            - max_input_length: Max input length from config
            - context_usage_ratio: Usage percentage
            - messages_detail: List of per-message AsMsgStat objects
        """
        messages = await self.get_memory(prepend_summary=False)

        compressed_summary = self.get_compressed_summary()
        compressed_summary_tokens = await self._msg_handler.count_str_token(
            compressed_summary,
        )

        # Build per-message token details using AsMsgHandler
        messages_detail = [
            await self._msg_handler.stat_message(msg) for msg in messages
        ]

        # Calculate total message tokens from stats
        messages_tokens = sum(stat.total_tokens for stat in messages_detail)
        estimated_tokens = messages_tokens + compressed_summary_tokens

        # Calculate context usage ratio
        context_usage_ratio = (
            (estimated_tokens / max_input_length * 100)
            if max_input_length > 0
            else 0
        )

        return {
            "total_messages": len(messages),
            "compressed_summary_tokens": compressed_summary_tokens,
            "messages_tokens": messages_tokens,
            "estimated_tokens": estimated_tokens,
            "max_input_length": max_input_length,
            "context_usage_ratio": context_usage_ratio,
            "messages_detail": messages_detail,
        }

    async def get_history_str(self, max_input_length: int) -> str:
        """Get formatted history string similar to /history command output.

        Args:
            max_input_length: Max input length for context usage calculation.

        Returns:
            Formatted string containing conversation history details
        """
        stats = await self.estimate_tokens(max_input_length)

        lines = []
        for i, msg_stat in enumerate(stats["messages_detail"], 1):
            blocks_info = ""
            if msg_stat.content:
                block_strs = [
                    f"{b.block_type}(tokens={b.token_count})"
                    for b in msg_stat.content
                ]
                blocks_info = f"\n    content: [{', '.join(block_strs)}]"

            lines.append(
                f"[{i}] **{msg_stat.role}** "
                f"(total_tokens={msg_stat.total_tokens})"
                f"{blocks_info}\n    preview: {msg_stat.preview}",
            )

        return (
            f"**Conversation History**\n\n"
            f"- Total messages: {stats['total_messages']}\n"
            f"- Estimated tokens: {stats['estimated_tokens']}\n"
            f"- Max input length: {stats['max_input_length']}\n"
            f"- Context usage: "
            f"{stats['context_usage_ratio']:.1f}%\n"
            f"- Compressed summary tokens: "
            f"{stats['compressed_summary_tokens']}\n\n" + "\n\n".join(lines)
        )
