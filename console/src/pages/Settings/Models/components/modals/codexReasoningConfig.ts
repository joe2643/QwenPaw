/**
 * Reasoning-config serde between a form value and a Codex OAuth
 * ``generate_kwargs`` dict.  Parallel to ``reasoningConfig.ts`` (for
 * Anthropic adaptive thinking) but much simpler — the ChatGPT Codex
 * backend surface exposes only one knob, ``reasoning.effort``, which
 * we forward as the top-level ``reasoning_effort`` kwarg (OpenAI SDK
 * convention).  The Python translator in ``codex_translate.py`` then
 * repacks it into ``reasoning: {effort: ...}`` for the upstream
 * Responses API body.
 *
 * Accepted effort values (probed live against
 * ``chatgpt.com/backend-api/codex/responses`` on 2026-04-24):
 *
 * | Effort    | gpt-5.4 | gpt-5.4-mini | gpt-5.2 |
 * |-----------|---------|--------------|---------|
 * | none      | ✓       | ✓            | ✓       |
 * | low       | ✓       | ✓            | ✓       |
 * | medium    | ✓       | ✓            | ✓       |
 * | high      | ✓       | ✓            | ✓       |
 * | xhigh     | ✓       | ✓            | ✓       |
 * | minimal   | ✗       | ✗            | ✗       |
 *
 * ``minimal`` is listed in backend error messages as a globally
 * "supported" value but no current model actually accepts it — we
 * omit it from the picker to keep the UI honest.  ``none`` means
 * "skip reasoning entirely" (fastest, no think tokens).
 */

export type CodexEffortLevel = "none" | "low" | "medium" | "high" | "xhigh";

export const CODEX_EFFORT_LEVELS: readonly CodexEffortLevel[] = [
  "none",
  "low",
  "medium",
  "high",
  "xhigh",
];

/** Codex speed tier — "fast" and "standard" are the only values the
 *  ChatGPT app actually surfaces; "flex" is probe-accepted but not
 *  advertised to consumer plans, so we keep it out of the picker.
 *
 *  Wire mapping (done server-side in ``codex_translate.py``):
 *  - ``fast``     → ``service_tier: "priority"`` (+ ~15-25% throughput,
 *                                                 2x/2.5x credit cost)
 *  - ``standard`` → field omitted (default routing)
 */
export type CodexSpeedLevel = "fast" | "standard";

export const CODEX_SPEED_LEVELS: readonly CodexSpeedLevel[] = [
  "standard",
  "fast",
];

export interface CodexReasoningFormValue {
  effort: CodexEffortLevel | undefined;
  speed: CodexSpeedLevel | undefined;
}

export const DEFAULT_CODEX_REASONING_FORM: CodexReasoningFormValue = {
  effort: undefined,
  speed: undefined,
};

/** Read reasoning fields OUT of a stored generate_kwargs dict. */
export function fromGenerateKwargs(
  gk: Record<string, unknown> | undefined | null,
): CodexReasoningFormValue {
  if (!gk || typeof gk !== "object") {
    return { ...DEFAULT_CODEX_REASONING_FORM };
  }
  const rawEffort = gk["reasoning_effort"];
  const effort: CodexEffortLevel | undefined =
    rawEffort === "none" ||
    rawEffort === "low" ||
    rawEffort === "medium" ||
    rawEffort === "high" ||
    rawEffort === "xhigh"
      ? rawEffort
      : undefined;
  const rawSpeed = gk["service_tier"];
  const speed: CodexSpeedLevel | undefined =
    rawSpeed === "fast" || rawSpeed === "standard" ? rawSpeed : undefined;
  return { effort, speed };
}

/**
 * Merge form-provided effort INTO an existing generate_kwargs dict,
 * preserving any unrelated keys the user typed into the JSON editor.
 *
 * Semantics:
 *   - effort=undefined → ``reasoning_effort`` key removed (falls
 *     through to the backend default set in ``codex_translate.py``)
 *   - effort=<level>   → ``reasoning_effort`` set to that string
 */
export function toGenerateKwargs(
  base: Record<string, unknown> | undefined | null,
  form: CodexReasoningFormValue,
): Record<string, unknown> {
  const out: Record<string, unknown> = { ...(base ?? {}) };
  if (form.effort) {
    out.reasoning_effort = form.effort;
  } else {
    delete out.reasoning_effort;
  }
  // Speed tier: only persist "fast" (explicit opt-in for priority
  // routing + extra credit cost).  "standard" resolves to no field
  // since that matches the Codex CLI's default for consumer plans,
  // and storing it would be noise in generate_kwargs.
  if (form.speed === "fast") {
    out.service_tier = "fast";
  } else {
    delete out.service_tier;
  }
  return out;
}

/**
 * Whether a model id accepts ``reasoning_effort``.  All three
 * currently-exposed Codex models do; non-reasoning GPT-4 models
 * rejected it historically, but none of those are in
 * CODEX_OAUTH_MODELS — gate is a pure allow-list based on the
 * ``gpt-5`` prefix.
 */
export function supportsReasoningEffort(modelId: string): boolean {
  return modelId.trim().toLowerCase().startsWith("gpt-5");
}
