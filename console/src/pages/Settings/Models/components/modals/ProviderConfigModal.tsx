import { useState, useEffect, useMemo, useRef } from "react";
import type { KeyboardEvent, ReactNode, UIEvent } from "react";
import {
  Form,
  Input,
  InputNumber,
  Modal,
  Button,
  Select,
  Radio,
} from "@agentscope-ai/design";
import { theme } from "antd";
import {
  EFFORT_LEVELS,
  THINKING_DISPLAYS,
  fromGenerateKwargs,
  toGenerateKwargs,
  type ReasoningFormValue,
} from "./reasoningConfig";
import {
  CODEX_EFFORT_LEVELS,
  CODEX_SPEED_LEVELS,
  fromGenerateKwargs as codexFromGenerateKwargs,
  toGenerateKwargs as codexToGenerateKwargs,
  type CodexEffortLevel,
  type CodexSpeedLevel,
  type CodexReasoningFormValue,
} from "./codexReasoningConfig";
import { useAppMessage } from "../../../../../hooks/useAppMessage";
import {
  ApiOutlined,
  CloseOutlined,
  DownOutlined,
  RightOutlined,
  CheckCircleFilled,
  CloseCircleFilled,
  CopyOutlined,
  ReloadOutlined,
} from "@ant-design/icons";
import type {
  BaseUrlOption,
  ClaudeOAuthStatus,
  CodexOAuthStatus,
  ProviderConfigRequest,
} from "../../../../../api/types";
import api from "../../../../../api";
import { useTranslation } from "react-i18next";
import { getLocalizedTestConnectionMessage } from "./testConnectionMessage";
import styles from "../../index.module.less";

interface ProviderConfigFormValues
  extends Omit<
    ProviderConfigRequest,
    "generate_kwargs" | "custom_headers" | "auth_mode"
  > {
  generate_kwargs_text?: string;
}

interface HeaderEntry {
  key: string;
  value: string;
}

interface JsonCodeEditorProps {
  value?: string;
  onChange?: (value: string) => void;
  placeholder?: string;
  rows?: number;
}

function highlightJson(text: string): ReactNode[] {
  const tokens: ReactNode[] = [];
  const pattern =
    /("(?:\\.|[^"\\])*")(\s*:)?|\btrue\b|\bfalse\b|\bnull\b|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|[{}\[\],:]/g;

  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = pattern.exec(text)) !== null) {
    const [token, stringToken, keySuffix] = match;

    if (match.index > lastIndex) {
      tokens.push(text.slice(lastIndex, match.index));
    }

    if (stringToken) {
      tokens.push(
        <span
          key={`${match.index}-${token}`}
          className={
            keySuffix ? styles.jsonEditorTokenKey : styles.jsonEditorTokenString
          }
        >
          {token}
        </span>,
      );
    } else if (token === "true" || token === "false") {
      tokens.push(
        <span
          key={`${match.index}-${token}`}
          className={styles.jsonEditorTokenBoolean}
        >
          {token}
        </span>,
      );
    } else if (token === "null") {
      tokens.push(
        <span
          key={`${match.index}-${token}`}
          className={styles.jsonEditorTokenNull}
        >
          {token}
        </span>,
      );
    } else if (/^-?\d/.test(token)) {
      tokens.push(
        <span
          key={`${match.index}-${token}`}
          className={styles.jsonEditorTokenNumber}
        >
          {token}
        </span>,
      );
    } else {
      tokens.push(
        <span
          key={`${match.index}-${token}`}
          className={styles.jsonEditorTokenPunctuation}
        >
          {token}
        </span>,
      );
    }

    lastIndex = match.index + token.length;
  }

  if (lastIndex < text.length) {
    tokens.push(text.slice(lastIndex));
  }

  return tokens;
}

function JsonCodeEditor({
  value = "",
  onChange,
  placeholder,
  rows = 8,
}: JsonCodeEditorProps) {
  const indentUnit = "  ";
  const highlightRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleScroll = (event: UIEvent<HTMLTextAreaElement>) => {
    if (!highlightRef.current) {
      return;
    }

    highlightRef.current.scrollTop = event.currentTarget.scrollTop;
    highlightRef.current.scrollLeft = event.currentTarget.scrollLeft;
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== "Tab") {
      return;
    }

    event.preventDefault();

    const textarea = event.currentTarget;
    const selectionStart = textarea.selectionStart;
    const selectionEnd = textarea.selectionEnd;
    const hasSelection = selectionStart !== selectionEnd;
    const selectedText = value.slice(selectionStart, selectionEnd);

    if (!hasSelection || !selectedText.includes("\n")) {
      if (event.shiftKey) {
        const lineStart = value.lastIndexOf("\n", selectionStart - 1) + 1;
        const linePrefix = value.slice(lineStart, selectionStart);

        if (!linePrefix.endsWith(indentUnit)) {
          return;
        }

        const nextValue =
          value.slice(0, selectionStart - indentUnit.length) +
          value.slice(selectionStart);

        onChange?.(nextValue);

        requestAnimationFrame(() => {
          textareaRef.current?.setSelectionRange(
            selectionStart - indentUnit.length,
            selectionStart - indentUnit.length,
          );
        });
        return;
      }

      const nextValue =
        value.slice(0, selectionStart) + indentUnit + value.slice(selectionEnd);

      onChange?.(nextValue);

      requestAnimationFrame(() => {
        const nextCursor = selectionStart + indentUnit.length;
        textareaRef.current?.setSelectionRange(nextCursor, nextCursor);
      });
      return;
    }

    const lineStart = value.lastIndexOf("\n", selectionStart - 1) + 1;
    const block = value.slice(lineStart, selectionEnd);
    const lines = block.split("\n");

    if (event.shiftKey) {
      const updatedLines = lines.map((line) =>
        line.startsWith(indentUnit) ? line.slice(indentUnit.length) : line,
      );
      const removedFromFirstLine = lines[0].startsWith(indentUnit)
        ? indentUnit.length
        : 0;
      const removedTotal = lines.reduce(
        (total, line) =>
          total + (line.startsWith(indentUnit) ? indentUnit.length : 0),
        0,
      );
      const nextValue =
        value.slice(0, lineStart) +
        updatedLines.join("\n") +
        value.slice(selectionEnd);

      onChange?.(nextValue);

      requestAnimationFrame(() => {
        textareaRef.current?.setSelectionRange(
          selectionStart - removedFromFirstLine,
          selectionEnd - removedTotal,
        );
      });
      return;
    }

    const updatedLines = lines.map((line) => `${indentUnit}${line}`);
    const nextValue =
      value.slice(0, lineStart) +
      updatedLines.join("\n") +
      value.slice(selectionEnd);

    onChange?.(nextValue);

    requestAnimationFrame(() => {
      textareaRef.current?.setSelectionRange(
        selectionStart + indentUnit.length,
        selectionEnd + indentUnit.length * lines.length,
      );
    });
  };

  return (
    <div className={styles.jsonEditorContainer}>
      <div
        ref={highlightRef}
        aria-hidden="true"
        className={styles.jsonEditorHighlight}
      >
        {value ? highlightJson(value) : placeholder}
        {!value && <span>{"\n"}</span>}
      </div>
      <textarea
        ref={textareaRef}
        rows={rows}
        value={value}
        onChange={(event) => onChange?.(event.target.value)}
        onKeyDown={handleKeyDown}
        onScroll={handleScroll}
        placeholder={placeholder}
        spellCheck={false}
        className={styles.jsonEditorTextarea}
      />
    </div>
  );
}

function formatExpiresIn(seconds: number | null | undefined): string {
  if (seconds == null) return "—";
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  const rem = m % 60;
  return rem ? `${h}h ${rem}m` : `${h}h`;
}

interface OAuthStatusLike {
  logged_in: boolean;
  credentials_path: string;
  expires_in_s: number | null;
  error: string | null;
}

interface OAuthLoginStatusPanelProps {
  status: OAuthStatusLike | null;
  loading: boolean;
  /** The CLI command users run to log in — e.g. ``claude login`` or
   *  ``codex login``. */
  loginCommand: string;
  /** Optional status badges shown next to "Logged in" — used to
   *  surface provider-specific fields (Claude: subscription plan,
   *  Codex: plan / email / mode). */
  extraBadges?: { label: string; value: string }[];
  onRefresh: () => void;
  /** Optional disk-reread action — shown as a second button beside
   *  Refresh.  Only Codex needs this (the long-lived in-agent
   *  ``CodexAuth`` caches creds across requests).  Claude Code's
   *  status is recomputed from disk on every render. */
  onReload?: () => void;
  onCopyLogin: () => void;
}

function OAuthLoginStatusPanel({
  status,
  loading,
  loginCommand,
  extraBadges,
  onRefresh,
  onReload,
  onCopyLogin,
}: OAuthLoginStatusPanelProps) {
  const loggedIn = status?.logged_in === true;
  // Resolve theme-aware colors at render time.  We can't rely on
  // ``var(--ant-color-fill-quaternary, ...)`` because this app runs AntD
  // without ``cssVar: true``, so those variables never get emitted and
  // the hardcoded light fallback wins in dark mode.
  const { token } = theme.useToken();
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 8,
        padding: "10px 12px",
        border: `1px solid ${token.colorBorder}`,
        borderRadius: 6,
        background: token.colorFillQuaternary,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          flexWrap: "wrap",
        }}
      >
        {loggedIn ? (
          <CheckCircleFilled style={{ color: "#52c41a" }} />
        ) : (
          <CloseCircleFilled style={{ color: "#ff4d4f" }} />
        )}
        <strong>{loggedIn ? "Logged in" : "Not logged in"}</strong>
        {loggedIn &&
          extraBadges?.map((b) => (
            <span key={b.label} style={{ opacity: 0.7 }}>
              · {b.label}: <code>{b.value}</code>
            </span>
          ))}
        {loggedIn && (
          <span style={{ opacity: 0.7 }}>
            · expires in {formatExpiresIn(status?.expires_in_s)}
          </span>
        )}
        <div style={{ marginLeft: "auto", display: "flex", gap: 6 }}>
          {onReload && (
            <Button
              size="small"
              loading={loading}
              onClick={onReload}
              title="Re-read auth.json from disk (pick up a fresh codex login)"
            >
              Reload
            </Button>
          )}
          <Button
            size="small"
            icon={<ReloadOutlined />}
            loading={loading}
            onClick={onRefresh}
          >
            Refresh
          </Button>
        </div>
      </div>
      {status?.error && (
        <div style={{ color: "#ff4d4f", fontSize: 12 }}>{status.error}</div>
      )}
      {!loggedIn && (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            fontSize: 12,
            opacity: 0.85,
          }}
        >
          <span>Run</span>
          <code
            style={{
              padding: "2px 6px",
              background: token.colorFillTertiary,
              borderRadius: 4,
            }}
          >
            {loginCommand}
          </code>
          <Button size="small" icon={<CopyOutlined />} onClick={onCopyLogin}>
            Copy
          </Button>
          <span>in a terminal, then click Refresh.</span>
        </div>
      )}
      {status?.credentials_path && (
        <div style={{ fontSize: 11, opacity: 0.55 }}>
          <code>{status.credentials_path}</code>
        </div>
      )}
    </div>
  );
}

interface ReasoningSectionProps {
  value: ReasoningFormValue;
  onChange: (patch: Partial<ReasoningFormValue>) => void;
  /** When true, the provider's model list includes a haiku entry —
   *  surface a one-liner warning that reasoning kwargs don't apply
   *  to haiku (the backend strips them silently). */
  haikuInList: boolean;
}

function ReasoningSection({
  value,
  onChange,
  haikuInList,
}: ReasoningSectionProps) {
  const { thinking_mode, thinking_display, effort, max_tokens } = value;
  const adaptiveOn = thinking_mode === "adaptive";
  const { token } = theme.useToken();
  return (
    <div
      style={{
        border: `1px solid ${token.colorBorder}`,
        borderRadius: 6,
        padding: "12px 14px",
        marginBottom: 14,
        background: token.colorFillQuaternary,
      }}
    >
      <div style={{ fontWeight: 600, marginBottom: 8 }}>
        Reasoning (Anthropic adaptive thinking + effort)
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
          gap: 12,
        }}
      >
        <Form.Item
          label="Thinking mode"
          style={{ marginBottom: 0 }}
          extra="Adaptive lets Claude decide when to think. Opus 4.7 accepts only adaptive."
        >
          <Select
            value={thinking_mode}
            onChange={(v) =>
              onChange({ thinking_mode: v as "off" | "adaptive" })
            }
            options={[
              { label: "Off (direct answer)", value: "off" },
              { label: "Adaptive (Opus/Sonnet 4.6+)", value: "adaptive" },
            ]}
          />
        </Form.Item>
        <Form.Item
          label="Thinking display"
          style={{ marginBottom: 0 }}
          extra="Opus 4.7 defaults to 'omitted' (no visible thinking text)."
        >
          <Select
            value={thinking_display}
            placeholder="(model default)"
            allowClear
            disabled={!adaptiveOn}
            onChange={(v) =>
              onChange({
                thinking_display: (v ?? undefined) as
                  | "summarized"
                  | "omitted"
                  | undefined,
              })
            }
            options={THINKING_DISPLAYS.map((d) => ({ label: d, value: d }))}
          />
        </Form.Item>
        <Form.Item
          label="Effort"
          style={{ marginBottom: 0 }}
          extra="Soft budget for thinking + response. xhigh = Opus 4.7 only; max = Opus 4.6+."
        >
          <Select
            value={effort}
            placeholder="(model default = high)"
            allowClear
            onChange={(v) =>
              onChange({
                effort: (v ?? undefined) as
                  | "low"
                  | "medium"
                  | "high"
                  | "xhigh"
                  | "max"
                  | undefined,
              })
            }
            options={EFFORT_LEVELS.map((e) => ({ label: e, value: e }))}
          />
        </Form.Item>
        <Form.Item
          label="Max output tokens"
          style={{ marginBottom: 0 }}
          extra="Hard cap on thinking + text. Raise if you see stop_reason=max_tokens."
        >
          <InputNumber
            value={max_tokens ?? undefined}
            min={256}
            max={128000}
            step={1024}
            placeholder="(provider default)"
            style={{ width: "100%" }}
            onChange={(v) =>
              onChange({
                max_tokens:
                  typeof v === "number" && Number.isFinite(v) && v > 0
                    ? v
                    : undefined,
              })
            }
          />
        </Form.Item>
      </div>
      {haikuInList && (
        <div
          style={{
            marginTop: 10,
            fontSize: 12,
            opacity: 0.7,
          }}
        >
          Note: Haiku models silently strip <code>thinking.adaptive</code> and{" "}
          <code>output_config.effort</code> server-side — these fields apply
          only to Opus / Sonnet 4.6+ in this provider.
        </div>
      )}
    </div>
  );
}

interface CodexReasoningSectionProps {
  value: CodexReasoningFormValue;
  onChange: (patch: Partial<CodexReasoningFormValue>) => void;
}

function CodexReasoningSection({
  value,
  onChange,
}: CodexReasoningSectionProps) {
  const { effort, speed } = value;
  const { token } = theme.useToken();
  return (
    <div
      style={{
        border: `1px solid ${token.colorBorder}`,
        borderRadius: 6,
        padding: "12px 14px",
        marginBottom: 14,
        background: token.colorFillQuaternary,
      }}
    >
      <div style={{ fontWeight: 600, marginBottom: 8 }}>Codex routing</div>
      <Form.Item
        label="Effort"
        style={{ marginBottom: 12 }}
        extra="ChatGPT backend budget: none=no thinking, xhigh=deepest. Default = low."
      >
        <Select
          value={effort}
          placeholder="(backend default = low)"
          allowClear
          onChange={(v) =>
            onChange({
              effort: (v ?? undefined) as CodexEffortLevel | undefined,
            })
          }
          options={CODEX_EFFORT_LEVELS.map((e) => ({ label: e, value: e }))}
        />
      </Form.Item>
      <Form.Item
        label="Speed"
        style={{ marginBottom: 0 }}
        extra="Fast uses priority routing (~15-25% faster throughput, ~2x credit cost on Pro plan). Standard = default routing."
      >
        <Select
          value={speed}
          placeholder="(standard)"
          allowClear
          onChange={(v) =>
            onChange({
              speed: (v ?? undefined) as CodexSpeedLevel | undefined,
            })
          }
          options={CODEX_SPEED_LEVELS.map((s) => ({ label: s, value: s }))}
        />
      </Form.Item>
    </div>
  );
}

interface ProviderConfigModalProps {
  provider: {
    id: string;
    name: string;
    api_key?: string;
    api_key_prefix?: string;
    base_url?: string;
    is_custom: boolean;
    freeze_url: boolean;
    chat_model: string;
    support_connection_check: boolean;
    generate_kwargs: Record<string, unknown>;
    require_api_key?: boolean;
    custom_headers?: Record<string, string>;
    auth_mode?: "api_key" | "auth_token";
    meta?: Record<string, unknown>;
  };
  activeModels: any;
  open: boolean;
  onClose: () => void;
  onSaved: () => void;
}

export function ProviderConfigModal({
  provider,
  activeModels,
  open,
  onClose,
  onSaved,
}: ProviderConfigModalProps) {
  const { t } = useTranslation();
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [formDirty, setFormDirty] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [form] = Form.useForm<ProviderConfigFormValues>();
  const { message } = useAppMessage();
  const [authMode, setAuthMode] = useState<"api_key" | "auth_token">(
    provider.auth_mode ?? "api_key",
  );
  const [customHeaders, setCustomHeaders] = useState<HeaderEntry[]>(
    Object.entries(provider.custom_headers ?? {}).map(([key, value]) => ({
      key,
      value,
    })),
  );
  const selectedChatModel = Form.useWatch("chat_model", form);
  const canEditBaseUrl = !provider.freeze_url;

  const baseUrlOptions = useMemo<BaseUrlOption[]>(() => {
    const raw = provider.meta?.base_url_options;
    if (!Array.isArray(raw)) return [];
    return raw.flatMap((item) => {
      if (
        item &&
        typeof item === "object" &&
        typeof (item as BaseUrlOption).label === "string" &&
        typeof (item as BaseUrlOption).value === "string"
      ) {
        return [item as BaseUrlOption];
      }
      return [];
    });
  }, [provider.meta]);

  const useBaseUrlSelect = canEditBaseUrl && baseUrlOptions.length > 0;

  // Both Claude Code OAuth and Codex (ChatGPT) OAuth carry their
  // credentials in external files managed by their respective CLIs
  // (``claude login`` / ``codex login``).  For those providers the
  // API-key input is replaced with a login-status panel; other
  // ``require_api_key=false`` providers (opencode, ollama, lmstudio)
  // keep their existing optional api_key input.
  const isClaudeOAuth = provider.id === "claude-oauth";
  const isCodexOAuth = provider.id === "codex-oauth";
  const isOAuthProvider = isClaudeOAuth || isCodexOAuth;
  const apiKeyHint =
    typeof provider.meta?.api_key_hint === "string"
      ? (provider.meta!.api_key_hint as string)
      : undefined;

  const [claudeOAuthStatus, setClaudeOAuthStatus] =
    useState<ClaudeOAuthStatus | null>(null);
  const [codexOAuthStatus, setCodexOAuthStatus] =
    useState<CodexOAuthStatus | null>(null);
  const [oauthLoading, setOauthLoading] = useState(false);

  const refreshOAuthStatus = async () => {
    if (!isOAuthProvider) return;
    setOauthLoading(true);
    try {
      if (isClaudeOAuth) {
        setClaudeOAuthStatus(await api.getClaudeOAuthStatus());
      } else if (isCodexOAuth) {
        setCodexOAuthStatus(await api.getCodexOAuthStatus());
      }
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : String(err);
      if (isClaudeOAuth) {
        setClaudeOAuthStatus({
          logged_in: false,
          credentials_path: "~/.claude/.credentials.json",
          expires_in_s: null,
          scopes: [],
          subscription: null,
          error: errMsg,
        });
      } else if (isCodexOAuth) {
        setCodexOAuthStatus({
          logged_in: false,
          credentials_path: "~/.codex/auth.json",
          expires_in_s: null,
          auth_mode: null,
          account_id: null,
          email: null,
          plan_type: null,
          org_title: null,
          subscription_active_until: null,
          error: errMsg,
        });
      }
    } finally {
      setOauthLoading(false);
    }
  };

  const reloadCodexOAuth = async () => {
    if (!isCodexOAuth) return;
    setOauthLoading(true);
    try {
      const status = await api.reloadCodexOAuth();
      setCodexOAuthStatus(status);
      message.success(
        status.logged_in
          ? `Auth reloaded · ${status.plan_type ?? "unknown plan"}`
          : "Reloaded — no valid credentials on disk",
      );
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : String(err);
      message.error(`Reload failed: ${errMsg}`);
    } finally {
      setOauthLoading(false);
    }
  };

  useEffect(() => {
    if (open && isOAuthProvider) {
      void refreshOAuthStatus();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, isOAuthProvider]);

  // Reasoning controls — only meaningful for Anthropic-family providers.
  // Two-way bind into the generate_kwargs JSON editor: reasoning fields
  // are derived live from the JSON; edits on them re-serialize back.
  // Single source of truth stays in ``generate_kwargs_text``.
  const isAnthropicFamily =
    (provider.is_custom ? selectedChatModel : provider.chat_model) ===
    "AnthropicChatModel";
  const watchedGenerateKwargsText = Form.useWatch("generate_kwargs_text", form);
  const parsedGenerateKwargs: Record<string, unknown> = useMemo(() => {
    const text = watchedGenerateKwargsText?.trim();
    if (!text) return {};
    try {
      const parsed = JSON.parse(text);
      return parsed && typeof parsed === "object" && !Array.isArray(parsed)
        ? (parsed as Record<string, unknown>)
        : {};
    } catch {
      return {};
    }
  }, [watchedGenerateKwargsText]);
  const reasoning: ReasoningFormValue = useMemo(
    () => fromGenerateKwargs(parsedGenerateKwargs),
    [parsedGenerateKwargs],
  );
  const applyReasoningPatch = (patch: Partial<ReasoningFormValue>) => {
    const next: ReasoningFormValue = { ...reasoning, ...patch };
    // Normalise: turning thinking off drops display; clearing display
    // stays at "off" implicit default.
    if (next.thinking_mode === "off") {
      next.thinking_display = undefined;
    }
    const merged = toGenerateKwargs(parsedGenerateKwargs, next);
    const text =
      Object.keys(merged).length > 0
        ? JSON.stringify(merged, null, 2)
        : undefined;
    form.setFieldValue("generate_kwargs_text", text);
    setFormDirty(true);
  };

  // Codex OAuth reasoning (parallel two-way bind for the
  // ``reasoning_effort`` kwarg that ``codex_translate`` reads).
  const codexReasoning: CodexReasoningFormValue = useMemo(
    () => codexFromGenerateKwargs(parsedGenerateKwargs),
    [parsedGenerateKwargs],
  );
  const applyCodexReasoningPatch = (
    patch: Partial<CodexReasoningFormValue>,
  ) => {
    const next: CodexReasoningFormValue = { ...codexReasoning, ...patch };
    const merged = codexToGenerateKwargs(parsedGenerateKwargs, next);
    const text =
      Object.keys(merged).length > 0
        ? JSON.stringify(merged, null, 2)
        : undefined;
    form.setFieldValue("generate_kwargs_text", text);
    setFormDirty(true);
  };

  const parseGenerateConfig = (value?: string) => {
    const trimmed = value?.trim();
    if (!trimmed) {
      return undefined;
    }

    let parsed: unknown;
    try {
      parsed = JSON.parse(trimmed);
    } catch {
      throw new Error(t("models.generateConfigInvalidJson"));
    }

    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      throw new Error(t("models.generateConfigMustBeObject"));
    }

    return parsed as Record<string, unknown>;
  };

  const effectiveChatModel = useMemo(() => {
    if (!provider.is_custom) {
      return provider.chat_model;
    }
    return selectedChatModel || provider.chat_model || "OpenAIChatModel";
  }, [provider.chat_model, provider.is_custom, selectedChatModel]);

  const isAnthropicProvider = useMemo(
    () =>
      provider.id === "anthropic" ||
      provider.chat_model === "AnthropicChatModel" ||
      effectiveChatModel === "AnthropicChatModel",
    [provider.id, provider.chat_model, effectiveChatModel],
  );

  const apiKeyPlaceholder = useMemo(() => {
    if (provider.api_key) {
      return t("models.leaveBlankKeep");
    }
    if (provider.api_key_prefix) {
      return t("models.enterApiKey", { prefix: provider.api_key_prefix });
    }
    return t("models.enterApiKeyOptional");
  }, [provider.api_key, provider.api_key_prefix, t]);

  const baseUrlExtra = useMemo(() => {
    if (!canEditBaseUrl) {
      return undefined;
    }
    if (useBaseUrlSelect) {
      return t("models.selectBaseURLHint");
    }
    if (provider.id === "azure-openai") {
      return t("models.azureEndpointHint");
    }
    if (provider.id === "anthropic") {
      return t("models.anthropicEndpointHint");
    }
    if (provider.id === "openai") {
      return t("models.openAIEndpoint");
    }
    if (provider.id === "opencode") {
      return t("models.openAICompatibleEndpoint");
    }
    if (provider.id === "ollama") {
      return t("models.ollamaEndpointHint");
    }
    if (provider.id === "lmstudio") {
      return t("models.lmstudioEndpointHint");
    }
    if (provider.is_custom) {
      return effectiveChatModel === "AnthropicChatModel"
        ? t("models.anthropicEndpointHint")
        : t("models.openAICompatibleEndpoint");
    }
    return t("models.apiEndpointHint");
  }, [
    canEditBaseUrl,
    useBaseUrlSelect,
    provider.id,
    provider.is_custom,
    effectiveChatModel,
    t,
  ]);

  const baseUrlPlaceholder = useMemo(() => {
    if (!canEditBaseUrl) {
      return "";
    }
    if (provider.id === "azure-openai") {
      return "https://<resource>.openai.azure.com/openai/v1";
    }
    if (provider.id === "anthropic") {
      return "https://api.anthropic.com";
    }
    if (provider.id === "openai") {
      return "https://api.openai.com/v1";
    }
    if (provider.id === "opencode") {
      return "https://opencode.ai/zen/v1";
    }
    if (provider.id === "ollama") {
      return "http://localhost:11434";
    }
    if (provider.id === "lmstudio") {
      return "http://localhost:1234/v1";
    }
    if (provider.is_custom && effectiveChatModel === "AnthropicChatModel") {
      return "https://api.anthropic.com";
    }
    return "https://api.example.com";
  }, [canEditBaseUrl, provider.id, provider.is_custom, effectiveChatModel]);

  // Sync form when modal opens or provider data changes
  useEffect(() => {
    if (open) {
      form.setFieldsValue({
        api_key: undefined,
        base_url: provider.base_url || undefined,
        chat_model: provider.chat_model || "OpenAIChatModel",
        generate_kwargs_text:
          provider.generate_kwargs &&
          Object.keys(provider.generate_kwargs).length > 0
            ? JSON.stringify(provider.generate_kwargs, null, 2)
            : undefined,
      });
      setAdvancedOpen(false);
      setFormDirty(false);
      setAuthMode(provider.auth_mode ?? "api_key");
      setCustomHeaders(
        Object.entries(provider.custom_headers ?? {}).map(([key, value]) => ({
          key,
          value,
        })),
      );
    }
  }, [provider, form, open]);

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields();
      setSaving(true);
      const generateConfig = parseGenerateConfig(values.generate_kwargs_text);
      const hasGenerateConfigInput = Boolean(
        values.generate_kwargs_text?.trim(),
      );

      // Validate connection before saving
      // For local providers, we might skip this or just check if models exist (which the backend does)
      if (provider.support_connection_check) {
        const testHeaders = customHeaders
          .filter((h) => h.key.trim())
          .reduce<Record<string, string>>((acc, h) => {
            acc[h.key.trim()] = h.value;
            return acc;
          }, {});
        const result = await api.testProviderConnection(provider.id, {
          api_key: values.api_key,
          base_url: values.base_url,
          chat_model: values.chat_model,
          custom_headers: testHeaders,
          auth_mode: isAnthropicProvider ? authMode : undefined,
        });

        if (!result.success) {
          message.error(getLocalizedTestConnectionMessage(result, t));
          // For built-in providers, we want to enforce valid config before saving
          return;
        }
      }

      const headersObj = customHeaders
        .filter((h) => h.key.trim())
        .reduce<Record<string, string>>((acc, h) => {
          acc[h.key.trim()] = h.value;
          return acc;
        }, {});

      await api.configureProvider(provider.id, {
        api_key: values.api_key,
        base_url: values.base_url,
        chat_model: values.chat_model,
        generate_kwargs: hasGenerateConfigInput ? generateConfig : {},
        custom_headers: headersObj,
        auth_mode: isAnthropicProvider ? authMode : undefined,
      });

      await onSaved();
      setFormDirty(false);
      onClose();
      message.success(t("models.configurationSaved", { name: provider.name }));
    } catch (error) {
      if (error && typeof error === "object" && "errorFields" in error) return;
      const errMsg =
        error instanceof Error ? error.message : t("models.failedToSaveConfig");
      message.error(errMsg);
    } finally {
      setSaving(false);
    }
  };

  const handleTest = async () => {
    setTesting(true);
    try {
      const values = await form.validateFields([
        "api_key",
        "base_url",
        "chat_model",
      ]);
      const testHeaders = customHeaders
        .filter((h) => h.key.trim())
        .reduce<Record<string, string>>((acc, h) => {
          acc[h.key.trim()] = h.value;
          return acc;
        }, {});
      const result = await api.testProviderConnection(provider.id, {
        api_key: values.api_key,
        base_url: values.base_url,
        chat_model: values.chat_model,
        custom_headers: testHeaders,
        auth_mode: isAnthropicProvider ? authMode : undefined,
      });
      if (result.success) {
        message.success(getLocalizedTestConnectionMessage(result, t));
      } else {
        message.warning(getLocalizedTestConnectionMessage(result, t));
      }
    } catch (error) {
      if (error && typeof error === "object" && "errorFields" in error) return;
      const errMsg =
        error instanceof Error
          ? error.message
          : t("models.testConnectionError");
      message.error(errMsg);
    } finally {
      setTesting(false);
    }
  };

  const isActiveLlmProvider =
    activeModels?.active_llm?.provider_id === provider.id;

  const handleRevoke = () => {
    const confirmContent = isActiveLlmProvider
      ? t("models.revokeConfirmContent", { name: provider.name })
      : t("models.revokeConfirmSimple", { name: provider.name });

    Modal.confirm({
      title: t("models.revokeAuthorization"),
      content: confirmContent,
      okText: t("models.revokeAuthorization"),
      okButtonProps: { danger: true },
      cancelText: t("models.cancel"),
      onOk: async () => {
        try {
          await api.configureProvider(provider.id, { api_key: "" });
          await onSaved();
          onClose();
          if (isActiveLlmProvider) {
            message.success(
              t("models.authorizationRevoked", { name: provider.name }),
            );
          } else {
            message.success(
              t("models.authorizationRevokedSimple", { name: provider.name }),
            );
          }
        } catch (error) {
          const errMsg =
            error instanceof Error ? error.message : t("models.failedToRevoke");
          message.error(errMsg);
        }
      },
    });
  };

  return (
    <Modal
      width={800}
      title={t("models.configureProvider", { name: provider.name })}
      open={open}
      onCancel={onClose}
      footer={
        <div className={styles.modalFooter}>
          <div className={styles.modalFooterLeft}>
            {provider.api_key && !isOAuthProvider && (
              <Button danger size="small" onClick={handleRevoke}>
                {t("models.revokeAuthorization")}
              </Button>
            )}
            {provider.support_connection_check && (
              <Button
                size="small"
                icon={<ApiOutlined />}
                onClick={handleTest}
                loading={testing}
              >
                {t("models.testConnection")}
              </Button>
            )}
          </div>
          <div className={styles.modalFooterRight}>
            <Button onClick={onClose}>{t("models.cancel")}</Button>
            <Button
              type="primary"
              loading={saving}
              disabled={!formDirty}
              onClick={handleSubmit}
            >
              {t("models.save")}
            </Button>
          </div>
        </div>
      }
      destroyOnHidden
    >
      <Form
        form={form}
        layout="vertical"
        initialValues={{
          base_url: provider.base_url || undefined,
          chat_model: provider.chat_model || "OpenAIChatModel",
          generate_kwargs_text:
            provider.generate_kwargs &&
            Object.keys(provider.generate_kwargs).length > 0
              ? JSON.stringify(provider.generate_kwargs, null, 2)
              : undefined,
        }}
        onValuesChange={() => setFormDirty(true)}
      >
        {provider.is_custom && (
          <Form.Item
            name="chat_model"
            label={t("models.protocol")}
            rules={[
              {
                required: true,
                message: t("models.selectProtocol"),
              },
            ]}
            extra={t("models.protocolHint")}
          >
            <Select
              disabled
              options={[
                {
                  value: "OpenAIChatModel",
                  label: t("models.protocolOpenAI"),
                },
                {
                  value: "AnthropicChatModel",
                  label: t("models.protocolAnthropic"),
                },
              ]}
            />
          </Form.Item>
        )}

        {/* Base URL */}
        <Form.Item
          name="base_url"
          label={t("models.baseURL")}
          rules={
            canEditBaseUrl
              ? [
                  ...(!provider.freeze_url
                    ? [
                        {
                          required: true,
                          message: t("models.pleaseEnterBaseURL"),
                        },
                      ]
                    : []),
                  {
                    validator: (_: unknown, value: string) => {
                      if (!value || !value.trim()) return Promise.resolve();
                      try {
                        const url = new URL(value.trim());
                        if (!["http:", "https:"].includes(url.protocol)) {
                          return Promise.reject(
                            new Error(t("models.pleaseEnterValidURL")),
                          );
                        }
                        return Promise.resolve();
                      } catch {
                        return Promise.reject(
                          new Error(t("models.pleaseEnterValidURL")),
                        );
                      }
                    },
                  },
                ]
              : []
          }
          extra={baseUrlExtra}
        >
          {useBaseUrlSelect ? (
            <Select
              options={baseUrlOptions.map((option) => ({
                label: `${option.label} — ${option.value}`,
                value: option.value,
              }))}
              placeholder={t("models.selectBaseURL")}
            />
          ) : (
            <Input
              placeholder={baseUrlPlaceholder}
              disabled={!canEditBaseUrl}
            />
          )}
        </Form.Item>

        {/* OAuth-provider login-status panel replaces the API key
            input for claude-oauth / codex-oauth. */}
        {isOAuthProvider && (
          <Form.Item label={t("models.apiKey")} extra={apiKeyHint}>
            {isClaudeOAuth && (
              <OAuthLoginStatusPanel
                status={claudeOAuthStatus}
                loading={oauthLoading}
                loginCommand="claude login"
                extraBadges={
                  claudeOAuthStatus?.subscription
                    ? [
                        {
                          label: "plan",
                          value: claudeOAuthStatus.subscription,
                        },
                      ]
                    : []
                }
                onRefresh={refreshOAuthStatus}
                onCopyLogin={() => {
                  void navigator.clipboard.writeText("claude login");
                  message.success("Copied");
                }}
              />
            )}
            {isCodexOAuth && (
              <OAuthLoginStatusPanel
                status={codexOAuthStatus}
                loading={oauthLoading}
                loginCommand="codex login"
                extraBadges={(() => {
                  const badges: { label: string; value: string }[] = [];
                  if (codexOAuthStatus?.plan_type) {
                    badges.push({
                      label: "plan",
                      value: codexOAuthStatus.plan_type,
                    });
                  }
                  if (codexOAuthStatus?.email) {
                    badges.push({
                      label: "email",
                      value: codexOAuthStatus.email,
                    });
                  }
                  if (
                    codexOAuthStatus?.auth_mode &&
                    codexOAuthStatus.auth_mode !== "chatgpt"
                  ) {
                    // Only surface mode when it's noteworthy (apikey);
                    // the common "chatgpt" case is implied by plan/email.
                    badges.push({
                      label: "mode",
                      value: codexOAuthStatus.auth_mode,
                    });
                  }
                  return badges;
                })()}
                onRefresh={refreshOAuthStatus}
                onReload={reloadCodexOAuth}
                onCopyLogin={() => {
                  void navigator.clipboard.writeText("codex login");
                  message.success("Copied");
                }}
              />
            )}
          </Form.Item>
        )}

        {/* API Key — replaced by the status panel above for OAuth
            providers.  Other providers keep the input (optional for
            opencode/ollama/lmstudio, required for the rest). */}
        {!isOAuthProvider && (
          <Form.Item
            name="api_key"
            label={t("models.apiKey")}
            extra={apiKeyHint}
            rules={[
              {
                validator: (_, value) => {
                  if (
                    value &&
                    provider.api_key_prefix &&
                    !value.startsWith(provider.api_key_prefix)
                  ) {
                    return Promise.reject(
                      new Error(
                        t("models.apiKeyShouldStart", {
                          prefix: provider.api_key_prefix,
                        }),
                      ),
                    );
                  }
                  return Promise.resolve();
                },
              },
            ]}
          >
            <Input.Password placeholder={apiKeyPlaceholder} />
          </Form.Item>
        )}

        <div className={styles.advancedConfigSection}>
          <button
            type="button"
            className={styles.advancedConfigToggle}
            onClick={() => setAdvancedOpen((prev) => !prev)}
          >
            <span className={styles.advancedConfigToggleLabel}>
              {advancedOpen ? <DownOutlined /> : <RightOutlined />}
              {t("models.advancedConfig")}
            </span>
          </button>

          {isAnthropicFamily && advancedOpen && (
            <ReasoningSection
              value={reasoning}
              onChange={applyReasoningPatch}
              // These bound-model hints are advisory — the real strip
              // happens server-side.  We surface them so users know
              // the provider-level default won't take effect on Haiku.
              haikuInList={provider.id === "claude-oauth"}
            />
          )}

          {isCodexOAuth && advancedOpen && (
            <CodexReasoningSection
              value={codexReasoning}
              onChange={applyCodexReasoningPatch}
            />
          )}

          {/* Anthropic auth mode selector */}
          {isAnthropicProvider && advancedOpen && (
            <Form.Item label={t("models.authMode")}>
              <Radio.Group
                value={authMode}
                onChange={(e) => {
                  setAuthMode(e.target.value);
                  setFormDirty(true);
                }}
              >
                <Radio value="api_key">{t("models.authModeApiKey")}</Radio>
                <Radio value="auth_token">
                  {t("models.authModeAuthToken")}
                </Radio>
              </Radio.Group>
            </Form.Item>
          )}

          {/* Custom Headers editor */}
          {advancedOpen && (
            <Form.Item
              label={t("models.customHeaders")}
              extra={t("models.customHeadersHint")}
            >
              <div className={styles.customHeadersSection}>
                {customHeaders.map((header, index) => (
                  <div key={index} className={styles.customHeaderRow}>
                    <Input
                      className={styles.customHeaderKey}
                      placeholder={t("models.customHeaderKey")}
                      value={header.key}
                      onChange={(e) => {
                        const next = [...customHeaders];
                        next[index] = { ...next[index], key: e.target.value };
                        setCustomHeaders(next);
                        setFormDirty(true);
                      }}
                    />
                    <Input
                      className={styles.customHeaderValue}
                      placeholder={t("models.customHeaderValue")}
                      value={header.value}
                      onChange={(e) => {
                        const next = [...customHeaders];
                        next[index] = {
                          ...next[index],
                          value: e.target.value,
                        };
                        setCustomHeaders(next);
                        setFormDirty(true);
                      }}
                    />
                    <CloseOutlined
                      className={styles.customHeaderDelete}
                      onClick={() => {
                        setCustomHeaders(
                          customHeaders.filter((_, i) => i !== index),
                        );
                        setFormDirty(true);
                      }}
                    />
                  </div>
                ))}
                <button
                  type="button"
                  className={styles.addHeaderBtn}
                  onClick={() => {
                    setCustomHeaders([
                      ...customHeaders,
                      { key: "", value: "" },
                    ]);
                    setFormDirty(true);
                  }}
                >
                  {t("models.addHeader")}
                </button>
              </div>
            </Form.Item>
          )}

          <Form.Item
            hidden={!advancedOpen}
            name="generate_kwargs_text"
            label={t("models.generateConfig")}
            extra={t("models.generateConfigHint")}
            rules={[
              {
                validator: (_: unknown, value?: string) => {
                  try {
                    parseGenerateConfig(value);
                    return Promise.resolve();
                  } catch (error) {
                    return Promise.reject(
                      error instanceof Error
                        ? error
                        : new Error(t("models.generateConfigInvalidJson")),
                    );
                  }
                },
              },
            ]}
          >
            <JsonCodeEditor
              rows={8}
              placeholder={`Example:\n{\n  "extra_body": {\n    "enable_thinking": false\n  },\n  "max_tokens": 2048\n}`}
            />
          </Form.Item>
        </div>
      </Form>
    </Modal>
  );
}
