/**
 * Unit tests for the Codex reasoning-config serde.  Mirrors the
 * Claude ``reasoningConfig.test.ts`` structure — kept separate
 * because the two serdes write into different generate_kwargs keys
 * and should never leak into each other.
 *
 *   $ cd console && bun test src/pages/Settings/Models/components/modals/__tests__/codexReasoningConfig.test.ts
 */

import { describe, expect, test } from "bun:test";
import {
  CODEX_EFFORT_LEVELS,
  DEFAULT_CODEX_REASONING_FORM,
  fromGenerateKwargs,
  supportsReasoningEffort,
  toGenerateKwargs,
} from "../codexReasoningConfig";

describe("fromGenerateKwargs (codex)", () => {
  test("empty / null / undefined → defaults", () => {
    expect(fromGenerateKwargs(undefined)).toEqual(
      DEFAULT_CODEX_REASONING_FORM,
    );
    expect(fromGenerateKwargs(null)).toEqual(DEFAULT_CODEX_REASONING_FORM);
    expect(fromGenerateKwargs({})).toEqual(DEFAULT_CODEX_REASONING_FORM);
  });

  test("extracts each valid effort level round-trippably", () => {
    for (const eff of CODEX_EFFORT_LEVELS) {
      const result = fromGenerateKwargs({ reasoning_effort: eff });
      expect(result.effort).toBe(eff);
    }
  });

  test("rejects invalid effort values silently", () => {
    // Values the backend accepts *in the error message* but no
    // current model actually supports (``minimal``), or Claude-only
    // values (``max``), or garbage — all must fall back to default
    // rather than silently forwarding an unsupported level.
    expect(fromGenerateKwargs({ reasoning_effort: "minimal" }).effort)
      .toBeUndefined();
    expect(fromGenerateKwargs({ reasoning_effort: "max" }).effort)
      .toBeUndefined();
    expect(fromGenerateKwargs({ reasoning_effort: 42 }).effort)
      .toBeUndefined();
    expect(fromGenerateKwargs({ reasoning_effort: null }).effort)
      .toBeUndefined();
  });

  test("ignores unrelated keys", () => {
    // Presence of other generate_kwargs shouldn't affect effort
    // extraction — the UI must still see "no effort set".
    const result = fromGenerateKwargs({
      temperature: 0.7,
      extra_body: { foo: "bar" },
    });
    expect(result).toEqual(DEFAULT_CODEX_REASONING_FORM);
  });
});

describe("toGenerateKwargs (codex)", () => {
  test("effort=undefined removes the key without touching others", () => {
    const base = { temperature: 0.7, extra_body: { foo: "bar" } };
    const out = toGenerateKwargs(base, { effort: undefined });
    expect(out).toEqual({ temperature: 0.7, extra_body: { foo: "bar" } });
    // Shouldn't mutate base.
    expect(base).toEqual({ temperature: 0.7, extra_body: { foo: "bar" } });
  });

  test("effort=<level> writes the key, preserves others", () => {
    const base = { temperature: 0.2 };
    const out = toGenerateKwargs(base, { effort: "high" });
    expect(out).toEqual({ temperature: 0.2, reasoning_effort: "high" });
  });

  test("overwrites previous effort value", () => {
    const base = { reasoning_effort: "low", temperature: 0.2 };
    const out = toGenerateKwargs(base, { effort: "xhigh" });
    expect(out.reasoning_effort).toBe("xhigh");
    expect(out.temperature).toBe(0.2);
  });

  test("clearing effort drops the key on a previously-set kwargs", () => {
    const base = { reasoning_effort: "high" };
    const out = toGenerateKwargs(base, { effort: undefined });
    expect(out).toEqual({});
  });

  test("null base behaves like empty", () => {
    expect(toGenerateKwargs(null, { effort: "medium" })).toEqual({
      reasoning_effort: "medium",
    });
    expect(toGenerateKwargs(undefined, { effort: undefined })).toEqual({});
  });
});

describe("round-trip", () => {
  test("from → to is identity for each level", () => {
    for (const eff of CODEX_EFFORT_LEVELS) {
      const form = fromGenerateKwargs({ reasoning_effort: eff });
      const kwargs = toGenerateKwargs({}, form);
      expect(kwargs).toEqual({ reasoning_effort: eff });
    }
  });

  test("from → to preserves unrelated keys", () => {
    const input = { reasoning_effort: "high", temperature: 0.5, foo: "bar" };
    const form = fromGenerateKwargs(input);
    const out = toGenerateKwargs(input, form);
    expect(out).toEqual(input);
  });
});

describe("supportsReasoningEffort", () => {
  test("gpt-5 family → true", () => {
    expect(supportsReasoningEffort("gpt-5.4")).toBe(true);
    expect(supportsReasoningEffort("gpt-5.4-mini")).toBe(true);
    expect(supportsReasoningEffort("gpt-5.2")).toBe(true);
    expect(supportsReasoningEffort("GPT-5.4")).toBe(true);
  });

  test("non-gpt-5 → false", () => {
    expect(supportsReasoningEffort("gpt-4")).toBe(false);
    expect(supportsReasoningEffort("gpt-4-turbo")).toBe(false);
    expect(supportsReasoningEffort("")).toBe(false);
    expect(supportsReasoningEffort("claude-opus-4-7")).toBe(false);
  });
});
