#!/usr/bin/env python3
"""End-to-end steer-mode probe against the live qwenpaw server.

Runs N rounds. For each round:
  1. Create a fresh chat session
  2. POST /console/chat with a slow first prompt (multi-step)
  3. After a short delay, POST a 2nd /console/chat (steer message)
  4. Read the FULL SSE stream of the first request
  5. Confirm the steer message text appears in the agent's output

Confirms the steer payload actually changes what the agent says.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
import uuid
from typing import Any

import httpx


BASE_URL = "http://127.0.0.1:8088"
AGENT_ID = "FSkZzR"


def _build_chat_payload(session_id: str, text: str) -> dict[str, Any]:
    """AgentRequest body for POST /console/chat."""
    return {
        "input": [
            {
                "role": "user",
                "content": [{"type": "text", "text": text}],
            },
        ],
        "session_id": session_id,
        "user_id": "steer-probe",
    }


def _stream_text_acc(line: str, sink: dict[str, str]) -> None:
    """Best-effort extract any 'text' / 'output_text' fragment from an SSE line."""
    if not line.startswith("data:"):
        return
    raw = line[len("data:") :].strip()
    if not raw or raw in ("[DONE]", "DONE"):
        return
    try:
        evt = json.loads(raw)
    except Exception:
        return

    def _walk(o: Any) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("text", "output_text", "delta", "content"):
                    if isinstance(v, str):
                        sink["full"] += v
                _walk(v)
        elif isinstance(o, list):
            for item in o:
                _walk(item)

    _walk(evt)


async def _drain_stream(client: httpx.AsyncClient, body: dict[str, Any]) -> str:
    """POST /console/chat and return the concatenated text payload of all SSE events."""
    sink = {"full": ""}
    async with client.stream(
        "POST",
        f"{BASE_URL}/api/console/chat",
        headers={"X-Agent-Id": AGENT_ID, "Content-Type": "application/json"},
        json=body,
        timeout=httpx.Timeout(180.0, connect=10.0),
    ) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            _stream_text_acc(line, sink)
    return sink["full"]


async def _post_steer(client: httpx.AsyncClient, body: dict[str, Any]) -> int:
    """Fire-and-discard 2nd POST that just enqueues the steer payload."""
    try:
        async with client.stream(
            "POST",
            f"{BASE_URL}/api/console/chat",
            headers={"X-Agent-Id": AGENT_ID, "Content-Type": "application/json"},
            json=body,
            timeout=httpx.Timeout(60.0, connect=5.0),
        ) as r:
            # consume a few lines so the request actually starts
            count = 0
            async for _ in r.aiter_lines():
                count += 1
                if count >= 3:
                    break
            return r.status_code
    except Exception as e:
        print(f"[steer post] exception: {e}")
        return -1


async def run_round(round_idx: int, secret: str, delay: float) -> dict[str, Any]:
    session_id = f"steer-probe-{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"
    base_prompt = (
        "Please think out loud step by step over multiple paragraphs. "
        "Take your time. Slowly count from 1 up to 5, with one paragraph "
        "of reasoning between each number, then summarize at the end."
    )
    # A *natural follow-up* (not a hostile override) — the agent should accept
    # this as additional task context. We then check the secret token appears
    # in the final output, proving the steer was injected before the next LLM
    # turn.
    steer_text = (
        "One more thing — please also include the exact verification token "
        f"{secret} verbatim somewhere in your final summary. This is a build "
        "tag I need to confirm the message reached you."
    )

    first_body = _build_chat_payload(session_id, base_prompt)
    steer_body = _build_chat_payload(session_id, steer_text)

    print(
        f"[round {round_idx}] session={session_id} "
        f"secret={secret!r} steer_delay={delay}s",
        flush=True,
    )

    async with httpx.AsyncClient() as client:
        first_task = asyncio.create_task(_drain_stream(client, first_body))
        await asyncio.sleep(delay)
        steer_status = await _post_steer(client, steer_body)
        print(f"[round {round_idx}] steer POST returned status={steer_status}")
        try:
            full = await asyncio.wait_for(first_task, timeout=180.0)
        except asyncio.TimeoutError:
            full = "<TIMEOUT>"

    full_lower = full.lower()
    secret_lower = secret.lower()
    found_secret = secret_lower in full_lower
    found_count_5 = bool(re.search(r"\b5\b", full))
    return {
        "round": round_idx,
        "session_id": session_id,
        "secret": secret,
        "steer_status": steer_status,
        "stream_chars": len(full),
        "found_secret": found_secret,
        "found_count_5": found_count_5,
        "tail": full[-220:] if full else "",
    }


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--delay", type=float, default=2.0)
    args = parser.parse_args()

    results: list[dict[str, Any]] = []
    for i in range(1, args.rounds + 1):
        secret = f"COPAW-STEER-OK-{uuid.uuid4().hex[:8]}"
        try:
            r = await run_round(i, secret, args.delay)
        except Exception as e:
            r = {"round": i, "error": str(e)}
        results.append(r)
        print(json.dumps(r, ensure_ascii=False, indent=2))
        print()
        await asyncio.sleep(2.0)

    summary = {
        "total": len(results),
        "found_secret_count": sum(1 for r in results if r.get("found_secret")),
        "found_count_5_count": sum(1 for r in results if r.get("found_count_5")),
        "errors": [r for r in results if "error" in r],
    }
    print("=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()
    if summary["found_secret_count"] >= 1:
        print(
            f"PASS — steer affected {summary['found_secret_count']}/"
            f"{summary['total']} runs."
        )
        return 0
    print("FAIL — steer text never appeared in any run's output.")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
