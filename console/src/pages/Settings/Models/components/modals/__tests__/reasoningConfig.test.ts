/**
 * Unit tests for the reasoning-config serde.  Runs with ``bun test``
 * (Jest-compatible API, no config needed).
 *
 *   $ cd console && bun test src/pages/Settings/Models/components/modals/__tests__/
 */

import { describe, expect, test } from "bun:test";
import {
  DEFAULT_REASONING_FORM,
  EFFORT_LEVELS,
  fromGenerateKwargs,
  modelRejectsReasoningKwargs,
  supportsAdaptiveThinking,
  toGenerateKwargs,
} from "../reasoningConfig";

describe("fromGenerateKwargs", () => {
  test("empty / null / undefined → defaults", () => {
    expect(fromGenerateKwargs(undefined)).toEqual(DEFAULT_REASONING_FORM);
    expect(fromGenerateKwargs(null)).toEqual(DEFAULT_REASONING_FORM);
    expect(fromGenerateKwargs({})).toEqual(DEFAULT_REASONING_FORM);
  });

  test("extracts adaptive thinking with display", () => {
    const result = fromGenerateKwargs({
      thinking: { type: "adaptive", display: "summarized" },
    });
    expect(result.thinking_mode).toBe("adaptive");
    expect(result.thinking_display).toBe("summarized");
  });

  test("extracts adaptive thinking without display", () => {
    const result = fromGenerateKwargs({
      thinking: { type: "adaptive" },
    });
    expect(result.thinking_mode).toBe("adaptive");
    expect(result.thinking_display).toBeUndefined();
  });

  test("ignores manual enabled-mode thinking (not adaptive)", () => {
    // We don't expose budget_tokens mode in the form — legacy config,
    // leave the JSON alone and treat form as 'off'.
    const result = fromGenerateKwargs({
      thinking: { type: "enabled", budget_tokens: 2000 },
    });
    expect(result.thinking_mode).toBe("off");
  });

  test("extracts effort from output_config", () => {
    for (const eff of EFFORT_LEVELS) {
      const result = fromGenerateKwargs({ output_config: { effort: eff } });
      expect(result.effort).toBe(eff);
    }
  });

  test("ignores unknown effort values", () => {
    const result = fromGenerateKwargs({
      output_config: { effort: "ultra" as never },
    });
    expect(result.effort).toBeUndefined();
  });

  test("extracts numeric max_tokens only", () => {
    expect(fromGenerateKwargs({ max_tokens: 32000 }).max_tokens).toBe(32000);
    expect(fromGenerateKwargs({ max_tokens: 0 }).max_tokens).toBeUndefined();
    expect(fromGenerateKwargs({ max_tokens: -1 }).max_tokens).toBeUndefined();
    expect(
      fromGenerateKwargs({ max_tokens: "32000" as never }).max_tokens,
    ).toBeUndefined();
  });

  test("tolerates malformed shapes", () => {
    const r = fromGenerateKwargs({
      thinking: "not an object" as never,
      output_config: ["not an object"] as never,
      max_tokens: NaN as never,
    });
    expect(r).toEqual(DEFAULT_REASONING_FORM);
  });
});

describe("toGenerateKwargs", () => {
  test("off mode strips thinking key", () => {
    const result = toGenerateKwargs(
      { thinking: { type: "adaptive" } },
      { ...DEFAULT_REASONING_FORM, thinking_mode: "off" },
    );
    expect(result.thinking).toBeUndefined();
  });

  test("adaptive with display", () => {
    const result = toGenerateKwargs(
      {},
      {
        ...DEFAULT_REASONING_FORM,
        thinking_mode: "adaptive",
        thinking_display: "omitted",
      },
    );
    expect(result.thinking).toEqual({ type: "adaptive", display: "omitted" });
  });

  test("adaptive without display (model default wins)", () => {
    const result = toGenerateKwargs(
      {},
      { ...DEFAULT_REASONING_FORM, thinking_mode: "adaptive" },
    );
    expect(result.thinking).toEqual({ type: "adaptive" });
  });

  test("effort set inserts output_config", () => {
    const result = toGenerateKwargs(
      {},
      { ...DEFAULT_REASONING_FORM, effort: "max" },
    );
    expect(result.output_config).toEqual({ effort: "max" });
  });

  test("effort cleared drops output_config", () => {
    const result = toGenerateKwargs(
      { output_config: { effort: "high" } },
      { ...DEFAULT_REASONING_FORM, effort: undefined },
    );
    expect(result.output_config).toBeUndefined();
  });

  test("effort cleared preserves other output_config keys", () => {
    const result = toGenerateKwargs(
      { output_config: { effort: "high", custom_knob: 42 } },
      { ...DEFAULT_REASONING_FORM, effort: undefined },
    );
    expect(result.output_config).toEqual({ custom_knob: 42 });
  });

  test("max_tokens round-trips", () => {
    const result = toGenerateKwargs(
      {},
      { ...DEFAULT_REASONING_FORM, max_tokens: 32000 },
    );
    expect(result.max_tokens).toBe(32000);
  });

  test("max_tokens cleared strips the key", () => {
    const result = toGenerateKwargs(
      { max_tokens: 8192 },
      { ...DEFAULT_REASONING_FORM, max_tokens: undefined },
    );
    expect(result.max_tokens).toBeUndefined();
  });

  test("unrelated keys are preserved", () => {
    const result = toGenerateKwargs(
      {
        temperature: 0.7,
        extra_body: { something: "else" },
        metadata: { user_id: "u1" },
      },
      {
        thinking_mode: "adaptive",
        thinking_display: "summarized",
        effort: "high",
        max_tokens: 16000,
      },
    );
    expect(result).toMatchObject({
      temperature: 0.7,
      extra_body: { something: "else" },
      metadata: { user_id: "u1" },
      thinking: { type: "adaptive", display: "summarized" },
      output_config: { effort: "high" },
      max_tokens: 16000,
    });
  });

  test("round-trip through from/to is stable", () => {
    const orig = {
      thinking: { type: "adaptive", display: "summarized" },
      output_config: { effort: "max" },
      max_tokens: 32000,
      temperature: 0.3,
    };
    const form = fromGenerateKwargs(orig);
    const back = toGenerateKwargs(orig, form);
    expect(back).toEqual(orig);
  });

  test("does not mutate base input", () => {
    const base = { thinking: { type: "adaptive" }, existing: 1 };
    const snapshot = JSON.parse(JSON.stringify(base));
    toGenerateKwargs(base, {
      ...DEFAULT_REASONING_FORM,
      thinking_mode: "off",
    });
    expect(base).toEqual(snapshot);
  });
});

describe("supportsAdaptiveThinking", () => {
  test.each([
    ["claude-opus-4-7", true],
    ["claude-opus-4-6", true],
    ["claude-sonnet-4-6", true],
    ["claude-mythos-preview", true],
    ["claude-haiku-4-5", false],
    ["claude-opus-4-5", false],
    ["claude-sonnet-4-5", false],
    ["claude-opus-4-1", false],
    ["gpt-4", false],
    ["", false],
  ])("%s → %s", (modelId, expected) => {
    expect(supportsAdaptiveThinking(modelId)).toBe(expected);
  });
});

describe("modelRejectsReasoningKwargs", () => {
  test("flags haiku variants", () => {
    expect(modelRejectsReasoningKwargs("claude-haiku-4-5")).toBe(true);
    expect(modelRejectsReasoningKwargs("claude-haiku-4-5-20251001")).toBe(true);
    expect(modelRejectsReasoningKwargs("CLAUDE-HAIKU-foo")).toBe(true);
  });

  test("clears non-haiku models", () => {
    expect(modelRejectsReasoningKwargs("claude-opus-4-7")).toBe(false);
    expect(modelRejectsReasoningKwargs("claude-sonnet-4-6")).toBe(false);
    expect(modelRejectsReasoningKwargs("")).toBe(false);
  });
});
