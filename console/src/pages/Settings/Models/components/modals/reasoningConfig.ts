/**
 * Reasoning-config serde between a form value and an Anthropic
 * ``generate_kwargs`` dict.  Pure functions — UI code calls these,
 * unit tests also call these.
 *
 * Shape we target (per Anthropic docs for Claude 4.6 / 4.7):
 *   generate_kwargs = {
 *     thinking: {type: "adaptive", display: "summarized" | "omitted"}
 *         | undefined,
 *     output_config: {effort: "low"|"medium"|"high"|"xhigh"|"max"} | undefined,
 *     max_tokens: number | undefined,
 *     ...any other keys the user typed into the JSON editor
 *   }
 *
 * ``xhigh`` is an Opus 4.7-only API value; ``max`` is Opus 4.6+; the
 * lower three are universal across adaptive-capable models.  Haiku
 * rejects adaptive thinking and effort entirely — the backend
 * ``ClaudeOAuthChatModel`` auto-strips those for haiku calls so the
 * user can safely set a provider-level default here.
 */

export type ThinkingMode = "off" | "adaptive";
export type ThinkingDisplay = "summarized" | "omitted";
export type EffortLevel = "low" | "medium" | "high" | "xhigh" | "max";

export const EFFORT_LEVELS: readonly EffortLevel[] = [
  "low",
  "medium",
  "high",
  "xhigh",
  "max",
];

export const THINKING_DISPLAYS: readonly ThinkingDisplay[] = [
  "summarized",
  "omitted",
];

export interface ReasoningFormValue {
  thinking_mode: ThinkingMode;
  thinking_display: ThinkingDisplay | undefined;
  effort: EffortLevel | undefined;
  max_tokens: number | undefined;
}

export const DEFAULT_REASONING_FORM: ReasoningFormValue = {
  thinking_mode: "off",
  thinking_display: undefined,
  effort: undefined,
  max_tokens: undefined,
};

/** Read reasoning fields OUT of a stored generate_kwargs dict. */
export function fromGenerateKwargs(
  gk: Record<string, unknown> | undefined | null,
): ReasoningFormValue {
  if (!gk || typeof gk !== "object") {
    return { ...DEFAULT_REASONING_FORM };
  }

  const thinking = gk["thinking"];
  let thinking_mode: ThinkingMode = "off";
  let thinking_display: ThinkingDisplay | undefined = undefined;
  if (
    thinking &&
    typeof thinking === "object" &&
    !Array.isArray(thinking) &&
    (thinking as Record<string, unknown>)["type"] === "adaptive"
  ) {
    thinking_mode = "adaptive";
    const disp = (thinking as Record<string, unknown>)["display"];
    if (disp === "summarized" || disp === "omitted") {
      thinking_display = disp;
    }
  }

  const oc = gk["output_config"];
  let effort: EffortLevel | undefined = undefined;
  if (oc && typeof oc === "object" && !Array.isArray(oc)) {
    const e = (oc as Record<string, unknown>)["effort"];
    if (
      e === "low" ||
      e === "medium" ||
      e === "high" ||
      e === "xhigh" ||
      e === "max"
    ) {
      effort = e;
    }
  }

  const maxTok = gk["max_tokens"];
  const max_tokens =
    typeof maxTok === "number" && Number.isFinite(maxTok) && maxTok > 0
      ? maxTok
      : undefined;

  return { thinking_mode, thinking_display, effort, max_tokens };
}

/**
 * Merge form-provided reasoning fields INTO an existing generate_kwargs
 * dict, preserving any unrelated keys the user typed into the JSON
 * editor (``extra_body``, ``temperature``, etc.).  Returns a new
 * object; does not mutate ``base``.
 *
 * Semantics:
 *   - thinking_mode=off   → ``thinking`` key removed
 *   - thinking_mode=adaptive with no display → omits ``display``
 *     (Opus 4.7 then defaults to "omitted", Opus 4.6 to "summarized")
 *   - effort=undefined → ``output_config`` key removed
 *     (or reduced to ``{}`` stripped to undefined — we drop it)
 *   - max_tokens=undefined → ``max_tokens`` key removed
 */
export function toGenerateKwargs(
  base: Record<string, unknown> | undefined | null,
  form: ReasoningFormValue,
): Record<string, unknown> {
  const out: Record<string, unknown> = { ...(base ?? {}) };

  // thinking
  if (form.thinking_mode === "adaptive") {
    const thinking: Record<string, unknown> = { type: "adaptive" };
    if (form.thinking_display) {
      thinking.display = form.thinking_display;
    }
    out.thinking = thinking;
  } else {
    delete out.thinking;
  }

  // output_config.effort
  if (form.effort) {
    const prev = out.output_config;
    const base_oc =
      prev && typeof prev === "object" && !Array.isArray(prev)
        ? { ...(prev as Record<string, unknown>) }
        : {};
    base_oc.effort = form.effort;
    out.output_config = base_oc;
  } else {
    const prev = out.output_config;
    if (prev && typeof prev === "object" && !Array.isArray(prev)) {
      const copy = { ...(prev as Record<string, unknown>) };
      delete copy.effort;
      if (Object.keys(copy).length === 0) {
        delete out.output_config;
      } else {
        out.output_config = copy;
      }
    } else {
      delete out.output_config;
    }
  }

  // max_tokens
  if (typeof form.max_tokens === "number" && form.max_tokens > 0) {
    out.max_tokens = form.max_tokens;
  } else {
    delete out.max_tokens;
  }

  return out;
}

/**
 * Test whether a model id is in the Anthropic adaptive-thinking
 * family.  Used to gate the thinking controls in the UI so Haiku
 * and legacy models don't show options that would 400 server-side.
 * Pattern matches ``claude-opus-4-6``/``4-7``, ``claude-sonnet-4-6``,
 * ``claude-mythos-*`` — everything else returns false.
 */
export function supportsAdaptiveThinking(modelId: string): boolean {
  const id = modelId.trim().toLowerCase();
  if (!id) return false;
  if (id.includes("haiku")) return false;
  if (/claude-(opus|sonnet)-4-(6|7)\b/.test(id)) return true;
  if (id.startsWith("claude-mythos")) return true;
  return false;
}

/** Haiku is the single model that rejects both adaptive thinking and
 *  ``output_config.effort`` outright; its presence is easy to detect
 *  by the substring ``haiku`` in the model id.  Used symmetrically in
 *  the UI (to warn) and in the Python ``ClaudeOAuthChatModel`` wrapper
 *  (to silently strip). */
export function modelRejectsReasoningKwargs(modelId: string): boolean {
  return modelId.trim().toLowerCase().includes("haiku");
}
