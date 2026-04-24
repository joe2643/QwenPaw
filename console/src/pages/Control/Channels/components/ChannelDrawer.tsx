import {
  Drawer,
  Form,
  Input,
  InputNumber,
  Switch,
  Button,
  Select,
} from "@agentscope-ai/design";
import { useAppMessage } from "../../../../hooks/useAppMessage";
import { Alert, ConfigProvider, Spin } from "antd";
import { LinkOutlined } from "@ant-design/icons";
import { useTranslation } from "react-i18next";
import type { FormInstance } from "antd";
import { useCallback, useEffect, useRef, useState } from "react";
import { getChannelLabel, type ChannelKey } from "./constants";
import { useChannelQrcode } from "./useChannelQrcode";
import { api } from "../../../../api";
import styles from "../index.module.less";
import { useTheme } from "../../../../contexts/ThemeContext";
import { useAgentStore } from "../../../../stores/agentStore";

const CHANNELS_WITH_ACCESS_CONTROL: ChannelKey[] = [
  "telegram",
  "dingtalk",
  "discord",
  "feishu",
  "wecom",
  "mattermost",
  "matrix",
  "weixin",
  "whatsapp",
  "signal",
  "imessage",
  "onebot",
];

// Doc EN URLs per channel (anchors on https://qwenpaw.agentscope.io/docs/channels)
const CHANNEL_DOC_EN_URLS: Partial<Record<ChannelKey, string>> = {
  dingtalk:
    "https://qwenpaw.agentscope.io/docs/channels/?lang=en#DingTalk-recommended",
  feishu: "https://qwenpaw.agentscope.io/docs/channels/?lang=en#Feishu-Lark",
  imessage:
    "https://qwenpaw.agentscope.io/docs/channels/?lang=en#iMessage-macOS-only",
  discord: "https://qwenpaw.agentscope.io/docs/channels/?lang=en#Discord",
  qq: "https://qwenpaw.agentscope.io/docs/channels/?lang=en#QQ",
  telegram: "https://qwenpaw.agentscope.io/docs/channels/?lang=en#Telegram",
  mqtt: "https://qwenpaw.agentscope.io/docs/channels/?lang=en#MQTT",
  mattermost: "https://qwenpaw.agentscope.io/docs/channels/?lang=en#Mattermost",
  matrix: "https://qwenpaw.agentscope.io/docs/channels/?lang=en#Matrix",
  sip: "https://qwenpaw.agentscope.io/docs/channels/?lang=en#SIP",
  wecom:
    "https://qwenpaw.agentscope.io/docs/channels/?lang=en#WeCom-WeChat-Work",
  weixin:
    "https://qwenpaw.agentscope.io/docs/channels/?lang=en#WeChat-Personal-iLink",
  whatsapp:
    "https://qwenpaw.agentscope.io/docs/channels/?lang=en#WhatsApp",
  signal: "https://qwenpaw.agentscope.io/docs/channels/?lang=en#Signal",
  xiaoyi:
    "https://developer.huawei.com/consumer/cn/doc/service/openclaw-0000002518410344",
  onebot:
    "https://qwenpaw.agentscope.io/docs/channels/?lang=en#OneBot-v11-NapCat--QQ-full-protocol",
};

// Doc ZH URLs per channel (anchors on https://qwenpaw.agentscope.io/docs/channels)
const CHANNEL_DOC_ZH_URLS: Partial<Record<ChannelKey, string>> = {
  dingtalk: "https://qwenpaw.agentscope.io/docs/channels/?lang=zh#钉钉推荐",
  feishu: "https://qwenpaw.agentscope.io/docs/channels/?lang=zh#飞书",
  imessage:
    "https://qwenpaw.agentscope.io/docs/channels/?lang=zh#iMessage仅-macOS",
  discord: "https://qwenpaw.agentscope.io/docs/channels/?lang=zh#Discord",
  qq: "https://qwenpaw.agentscope.io/docs/channels/?lang=zh#QQ",
  telegram: "https://qwenpaw.agentscope.io/docs/channels/?lang=zh#Telegram",
  mqtt: "https://qwenpaw.agentscope.io/docs/channels/?lang=zh#MQTT",
  mattermost: "https://qwenpaw.agentscope.io/docs/channels/?lang=zh#Mattermost",
  matrix: "https://qwenpaw.agentscope.io/docs/channels/?lang=zh#Matrix",
  sip: "https://qwenpaw.agentscope.io/docs/channels/?lang=zh#SIP",
  wecom: "https://qwenpaw.agentscope.io/docs/channels/?lang=zh#企业微信",
  weixin: "https://qwenpaw.agentscope.io/docs/channels/?lang=zh#微信个人iLink",
  whatsapp: "https://qwenpaw.agentscope.io/docs/channels/?lang=zh#WhatsApp",
  signal: "https://qwenpaw.agentscope.io/docs/channels/?lang=zh#Signal",
  xiaoyi:
    "https://developer.huawei.com/consumer/cn/doc/service/openclaw-0000002518410344",
  onebot:
    "https://qwenpaw.agentscope.io/docs/channels/?lang=zh#OneBot-v11NapCat--QQ-完整协议",
};

const TWILIO_CONSOLE_URL = "https://console.twilio.com";

const BASE_FIELDS = [
  "enabled",
  "bot_prefix",
  "filter_tool_messages",
  "filter_thinking",
  "isBuiltin",
];

interface ChannelDrawerProps {
  open: boolean;
  activeKey: ChannelKey | null;
  activeLabel: string;
  form: FormInstance<Record<string, unknown>>;
  saving: boolean;
  initialValues: Record<string, unknown> | undefined;
  isBuiltin: boolean;
  onClose: () => void;
  onSubmit: (values: Record<string, unknown>) => void;
}

export function ChannelDrawer({
  open,
  activeKey,
  activeLabel,
  form,
  saving,
  initialValues,
  isBuiltin,
  onClose,
  onSubmit,
}: ChannelDrawerProps) {
  const { t, i18n } = useTranslation();
  const { isDark } = useTheme();
  const { selectedAgent, agents } = useAgentStore();
  const currentAgent = agents.find((a) => a.id === selectedAgent);
  const defaultMediaDir = currentAgent?.workspace_dir
    ? `${currentAgent.workspace_dir}/media`
    : "~/.qwenpaw/media";
  const currentLang = i18n.language?.startsWith("zh") ? "zh" : "en";
  const label = activeKey ? getChannelLabel(activeKey, t) : activeLabel;
  const { message } = useAppMessage();

  // WeChat QR code hook
  const weixinQrcode = useChannelQrcode({
    channel: "weixin",
    successStatus: "confirmed",
    successCredentialKey: "bot_token",
    pollInterval: 2000,
    onSuccess: useCallback(
      (credentials: Record<string, string>) => {
        form.setFieldsValue({ bot_token: credentials.bot_token });
        message.success(t("channels.weixinLoginSuccess"));
      },
      [form, message, t],
    ),
    onError: useCallback(
      (type: "fetch" | "expired") => {
        if (type === "expired") {
          message.warning(t("channels.weixinQrcodeExpired"));
        } else {
          message.error(t("channels.weixinQrcodeFailed"));
        }
      },
      [message, t],
    ),
  });

  // DingTalk QR code hook
  const dingtalkQrcode = useChannelQrcode({
    channel: "dingtalk",
    successStatus: "success",
    successCredentialKey: "client_id",
    pollInterval: 5000,
    onSuccess: useCallback(
      (credentials: Record<string, string>) => {
        form.setFieldsValue({
          client_id: credentials.client_id,
          client_secret: credentials.client_secret,
        });
        message.success(t("channels.dingtalkAuthSuccess"));
      },
      [form, message, t],
    ),
    onExpired: useCallback(() => {
      message.warning(t("channels.dingtalkQrcodeExpired"));
    }, [message, t]),
    onError: useCallback(
      (type: "fetch" | "expired") => {
        if (type === "expired") {
          message.warning(t("channels.dingtalkQrcodeExpired"));
        } else {
          message.error(t("channels.dingtalkQrcodeFailed"));
        }
      },
      [message, t],
    ),
  });

  // WeCom QR code hook
  const wecomQrcode = useChannelQrcode({
    channel: "wecom",
    successStatus: "confirmed",
    successCredentialKey: "bot_id",
    pollInterval: 2000,
    onSuccess: useCallback(
      (credentials: Record<string, string>) => {
        form.setFieldsValue({ bot_id: credentials.bot_id, secret: credentials.secret });
        message.success(t("channels.wecomAuthSuccess"));
      },
      [form, message, t],
    ),
    onError: useCallback(
      (type: "fetch" | "expired") => {
        if (type === "expired") {
          message.warning(t("channels.weixinQrcodeExpired"));
        } else {
          message.error(t("channels.wecomAuthFailedGeneric"));
        }
      },
      [message, t],
    ),
  });

  // WhatsApp pair code state
  const [waPhone, setWaPhone] = useState<string>("");
  const [waPairCode, setWaPairCode] = useState<string>("");
  const [waQrImage, setWaQrImage] = useState<string>("");
  const [waPairLoading, setWaPairLoading] = useState(false);
  const [waPairStatus, setWaPairStatus] = useState<string>("idle");
  const waPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // WhatsApp linked state
  const [waLinked, setWaLinked] = useState(false);
  const stopWaPoll = useCallback(() => {
    if (waPollRef.current) {
      clearInterval(waPollRef.current);
      waPollRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (activeKey === "whatsapp") {
      // Bootstrap the drawer state from /status so the "connected" UI
      // shows up immediately on open without requiring the user to first
      // click Pair. setWaPairStatus mirrors setWaLinked so the rest of
      // the form (which reads waPairStatus) stays consistent.
      api
        .getWhatsappStatus()
        .then((s) => {
          setWaLinked(s.linked);
          if (s.linked) setWaPairStatus("connected");
        })
        .catch(() => {});
    }
    return () => {
      stopWaPoll();
    };
  }, [activeKey, stopWaPoll]);


  const handleWhatsappPair = useCallback(async () => {
    stopWaPoll();
    setWaPairLoading(true);
    setWaPairCode("");
    setWaQrImage("");
    setWaPairStatus("pairing");
    try {
      const data = await api.startWhatsappPair(waPhone);
      if (data.pair_code) {
        setWaPairCode(data.pair_code);
        setWaPairStatus("waiting_pair");
      }
      if (data.qr_image) {
        setWaQrImage(data.qr_image);
        setWaPairStatus("waiting_qr");
      }
      // Poll for connection
      waPollRef.current = setInterval(async () => {
        try {
          const s = await api.checkWhatsappPairStatus();
          if (s.status === "connected") {
            stopWaPoll();
            setWaPairCode("");
            setWaQrImage("");
            setWaPairStatus("connected");
            setWaPairLoading(false);
            t("channels.whatsappLinkedSuccess") && message.success(t("channels.whatsappLinkedSuccess"));
          }
        } catch { /* ignore */ }
      }, 3000);
    } catch (err) {
      message.error(t("channels.whatsappPairFailed"));
      setWaPairStatus("idle");
    } finally {
      setWaPairLoading(false);
    }
  }, [stopWaPoll, message, t, waPhone]);

  const handleWhatsappUnbind = useCallback(async () => {
    try {
      await api.unbindWhatsapp();
      setWaPairCode("");
      setWaQrImage("");
      setWaPairStatus("idle");
      setWaLinked(false);  // keep /status-derived flag in sync so UI flips back immediately
      message.success(t("channels.whatsappUnlinked"));
    } catch (err) {
      message.error(t("channels.whatsappUnbindFailed"));
    }
  }, [message, t]);


  // ── Signal link flow state ──────────────────────────────────────────
  // Parallels WhatsApp's waPhone / waQrImage / waPairStatus pattern but
  // with signal-cli subprocess semantics: no phone-number param (the
  // subprocess returns one), QR is the only path (no pair-code fallback).
  const [sigLinked, setSigLinked] = useState(false);
  const [sigPhone, setSigPhone] = useState<string>("");
  // UUID is not shown in the drawer but is stored on the form; the
  // setter is still useful for that purpose (hence the _ prefix — it
  // signals "intentionally unread state" to Copilot / strict-TS).
  const [, setSigUuid] = useState<string>("");
  const [sigQrImage, setSigQrImage] = useState<string>("");
  const [sigPairStatus, setSigPairStatus] = useState<string>("idle");
  const [sigPairLoading, setSigPairLoading] = useState(false);
  const [sigDeviceName, setSigDeviceName] = useState<string>("QwenPaw");
  const sigPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // Directory options for the Signal drawer dropdowns. Populated from
  // backend /channels/signal/{contacts,groups} when linked.
  const [sigContacts, setSigContacts] = useState<
    Array<{ number: string; uuid: string; name: string }>
  >([]);
  const [sigGroups, setSigGroups] = useState<
    Array<{ id: string; blocked: boolean }>
  >([]);

  const stopSigPoll = useCallback(() => {
    if (sigPollRef.current) {
      clearInterval(sigPollRef.current);
      sigPollRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (activeKey === "signal") {
      api
        .getSignalStatus()
        .then((s) => {
          setSigLinked(Boolean(s.linked));
          if (s.linked) {
            setSigPairStatus("linked");
            if (s.phone) setSigPhone(s.phone);
            if (s.uuid) setSigUuid(s.uuid);
            // Auto-populate account / account_uuid from the authoritative
            // signal-cli account store. Users shouldn't have to type in the
            // phone number after linking — the backend already knows it.
            // Only fill when the form field is empty to avoid clobbering a
            // user-entered override.
            const currentAccount = form.getFieldValue("account");
            const currentUuid = form.getFieldValue("account_uuid");
            const patch: Record<string, string> = {};
            if (!currentAccount && s.phone) patch.account = s.phone;
            if (!currentUuid && s.uuid) patch.account_uuid = s.uuid;
            if (Object.keys(patch).length) form.setFieldsValue(patch);
            // Fetch contacts + groups to populate the allow_from /
            // group_allow_from / groups dropdowns so users don't have
            // to type raw phone numbers or base64 group ids.
            api
              .listSignalContacts()
              .then((r) => setSigContacts(r.contacts || []))
              .catch(() => setSigContacts([]));
            api
              .listSignalGroups()
              .then((r) => setSigGroups(r.groups || []))
              .catch(() => setSigGroups([]));
          } else {
            setSigContacts([]);
            setSigGroups([]);
          }
        })
        .catch(() => {
          /* ignore — endpoint may be 404 on older backends */
        });
    }
    return () => {
      stopSigPoll();
    };
  }, [activeKey, stopSigPoll, form]);

  const handleSignalLink = useCallback(async () => {
    stopSigPoll();
    setSigPairLoading(true);
    setSigQrImage("");
    setSigPairStatus("starting");
    try {
      const data = await api.startSignalLink(sigDeviceName || "QwenPaw");
      if (data.qr_image) {
        setSigQrImage(data.qr_image);
        setSigPairStatus(data.status || "waiting_qr");
      }
      // Poll for completion. Slightly longer interval than WhatsApp
      // (3s) because signal-cli link rarely confirms in under 10s.
      sigPollRef.current = setInterval(async () => {
        try {
          const s = await api.checkSignalLinkStatus();
          if (s.status === "linked") {
            stopSigPoll();
            setSigQrImage("");
            setSigPairStatus("linked");
            setSigLinked(true);
            if (s.phone) setSigPhone(s.phone);
            if (s.uuid) setSigUuid(s.uuid);
            setSigPairLoading(false);
            form.setFieldsValue({
              account: s.phone || "",
              account_uuid: s.uuid || "",
            });
            // Auto-persist account + account_uuid to agent config so the
            // Signal channel can start against the linked account without
            // the user having to click Save. Other form fields (enabled,
            // policies, etc.) still require explicit Save — we only push
            // the two values signal-cli itself just authoritative-told us.
            if (s.phone || s.uuid) {
              const allFields = form.getFieldsValue();
              const persisted: Record<string, unknown> = {
                ...allFields,
                account: s.phone || allFields.account || "",
                account_uuid: s.uuid || allFields.account_uuid || "",
                filter_tool_messages: !allFields.filter_tool_messages,
                filter_thinking: !allFields.filter_thinking,
              };
              api
                .updateChannelConfig(
                  "signal",
                  persisted as unknown as Parameters<
                    typeof api.updateChannelConfig
                  >[1],
                )
                .catch((err) => {
                  // Non-fatal: user can still click Save manually.
                  console.warn("signal: auto-persist after link failed:", err);
                });
            }
            message.success(t("channels.signalLinkSuccess"));
          } else if (s.status === "error") {
            stopSigPoll();
            setSigPairStatus("error");
            setSigPairLoading(false);
            message.error(s.error || t("channels.signalLinkFailed"));
          }
        } catch {
          /* transient — keep polling */
        }
      }, 3000);
    } catch (err) {
      const msg = (err as Error)?.message || t("channels.signalLinkFailed");
      message.error(msg);
      setSigPairStatus("error");
      // Error-path terminal state: clear loading immediately. The previous
      // unconditional `finally` fired right after the polling interval was
      // scheduled, which flipped the spinner off while the link flow was
      // still in progress. Loading now only clears on terminal states —
      // inline clears inside the poll loop (linked / error branches) handle
      // the happy path and the server-reported error.
      setSigPairLoading(false);
    }
  }, [sigDeviceName, stopSigPoll, form, message, t]);

  const handleSignalUnbind = useCallback(async () => {
    stopSigPoll();
    try {
      await api.unbindSignal();
      setSigLinked(false);
      setSigPhone("");
      setSigUuid("");
      setSigQrImage("");
      setSigPairStatus("idle");
      form.setFieldsValue({ account: "", account_uuid: "" });
      message.success(t("channels.signalUnlinkSuccess"));
    } catch (err) {
      const msg = (err as Error)?.message || t("channels.signalUnlinkFailed");
      message.error(msg);
    }
  }, [stopSigPoll, form, message, t]);

  // ── Access control fields (shared across multiple channels) ──────────────

  // ── Access control fields (shared across multiple channels) ──────────────

  const renderAccessControlFields = () => (
    <>
      <Form.Item
        name="dm_policy"
        label={t("channels.dmPolicy")}
        tooltip={t("channels.dmPolicyTooltip")}
        initialValue="open"
      >
        <Select
          options={[
            { value: "open", label: t("channels.policyOpen") },
            { value: "allowlist", label: t("channels.policyAllowlist") },
          ]}
        />
      </Form.Item>
      <Form.Item
        name="group_policy"
        label={t("channels.groupPolicy")}
        tooltip={t("channels.groupPolicyTooltip")}
        initialValue="open"
      >
        <Select
          options={[
            { value: "open", label: t("channels.policyOpen") },
            { value: "allowlist", label: t("channels.policyAllowlist") },
          ]}
        />
      </Form.Item>
      <Form.Item
        name="require_mention"
        label={t("channels.requireMention")}
        valuePropName="checked"
        tooltip={t("channels.requireMentionTooltip")}
      >
        <Switch />
      </Form.Item>
      <Form.Item
        name="allow_from"
        label={t("channels.allowFrom")}
        tooltip={t("channels.allowFromTooltip")}
      >
        <Select
          mode="tags"
          placeholder={t("channels.allowFromPlaceholder")}
          tokenSeparators={[","]}
          // For Signal, populate with known contacts from
          // signal-cli's account.db — value is the phone (preferred,
          // since allowlist often matches on phone), with UUID shown in
          // the option label. Users can still type free-form to add
          // values not in the directory (e.g. uuid: prefix or unknown
          // phones).
          options={
            activeKey === "signal" && sigContacts.length
              ? sigContacts.map((c) => {
                  const value = c.number || (c.uuid ? `uuid:${c.uuid}` : "");
                  const label = [c.name, c.number, c.uuid && `uuid:${c.uuid.slice(0, 8)}…`]
                    .filter(Boolean)
                    .join(" · ");
                  return { value, label: label || value };
                })
              : undefined
          }
        />
      </Form.Item>
    </>
  );

  // ── Builtin channel-specific fields ─────────────────────────────────────

  const renderBuiltinExtraFields = (key: ChannelKey) => {
    switch (key) {
      case "matrix":
        return (
          <>
            <Form.Item
              name="homeserver"
              label="Homeserver URL"
              rules={[{ required: true }]}
            >
              <Input placeholder="https://matrix.org" />
            </Form.Item>
            <Form.Item
              name="user_id"
              label="User ID"
              rules={[{ required: true }]}
            >
              <Input placeholder="@bot:matrix.org" />
            </Form.Item>
            <Form.Item
              name="access_token"
              label="Access Token"
              rules={[{ required: true }]}
            >
              <Input.Password placeholder="syt_..." />
            </Form.Item>
          </>
        );

      case "imessage":
        return (
          <>
            <Form.Item
              name="db_path"
              label="DB Path"
              rules={[{ required: true, message: "Please input DB path" }]}
            >
              <Input placeholder="~/Library/Messages/chat.db" />
            </Form.Item>
            <Form.Item
              name="poll_sec"
              label="Poll Interval (sec)"
              rules={[
                { required: true, message: "Please input poll interval" },
              ]}
            >
              <InputNumber min={0.1} step={0.1} style={{ width: "100%" }} />
            </Form.Item>
          </>
        );

      case "discord":
        return (
          <>
            <Form.Item
              name="bot_token"
              label="Bot Token"
              rules={[{ required: true }]}
            >
              <Input.Password placeholder="Discord bot token" />
            </Form.Item>
            <Form.Item name="http_proxy" label="HTTP Proxy">
              <Input placeholder="http://127.0.0.1:18118" />
            </Form.Item>
            <Form.Item name="http_proxy_auth" label="HTTP Proxy Auth">
              <Input placeholder="user:password" />
            </Form.Item>
            <Form.Item
              name="accept_bot_messages"
              label={t("channels.acceptBotMessages")}
              valuePropName="checked"
              tooltip={t("channels.acceptBotMessagesTooltip")}
            >
              <Switch />
            </Form.Item>
          </>
        );

      case "dingtalk":
        return (
          <>
            <ConfigProvider prefixCls="ant">
              <Alert
                type="info"
                showIcon
                message={t("channels.dingtalkSetupGuide")}
                style={{ marginBottom: 16 }}
              />
            </ConfigProvider>
            <Form.Item label={t("channels.dingtalkScanAuth")}>
              <Button
                type="primary"
                block
                loading={dingtalkQrcode.loading}
                onClick={dingtalkQrcode.fetchQrcode}
              >
                {t("channels.dingtalkGetQrcode")}
              </Button>
              {dingtalkQrcode.loading && (
                <div style={{ textAlign: "center", marginTop: 12 }}>
                  <Spin />
                </div>
              )}
              {dingtalkQrcode.qrcodeImg && !dingtalkQrcode.loading && (
                <div style={{ textAlign: "center", marginTop: 12 }}>
                  <img
                    src={`data:image/png;base64,${dingtalkQrcode.qrcodeImg}`}
                    alt="DingTalk QR Code"
                    style={{ width: 200, height: 200 }}
                  />
                  <div
                    style={{
                      marginTop: 8,
                      fontSize: 12,
                      color: isDark
                        ? "rgba(255,255,255,0.45)"
                        : "rgba(0,0,0,0.45)",
                    }}
                  >
                    {t("channels.dingtalkScanHint")}
                  </div>
                </div>
              )}
            </Form.Item>
            <Form.Item
              name="client_id"
              label="Client ID"
              rules={[{ required: true }]}
            >
              <Input placeholder="dingxxxxx" />
            </Form.Item>
            <Form.Item
              name="client_secret"
              label="Client Secret"
              rules={[{ required: true }]}
            >
              <Input.Password />
            </Form.Item>
            <Form.Item
              name="message_type"
              label="Message Type"
              tooltip="markdown: regular messages; card: AI interactive card"
            >
              <Select
                options={[
                  { label: "markdown", value: "markdown" },
                  { label: "card", value: "card" },
                ]}
              />
            </Form.Item>
            <Form.Item
              name="cron_message_type"
              label="Cron Message Type"
              tooltip="Message type for cron/scheduled task sends. Independent from the chat message type above."
            >
              <Select
                options={[
                  { label: "markdown", value: "markdown" },
                  { label: "card", value: "card" },
                ]}
              />
            </Form.Item>
            <Form.Item
              noStyle
              shouldUpdate={(prev, cur) =>
                prev.message_type !== cur.message_type ||
                prev.cron_message_type !== cur.cron_message_type
              }
            >
              {({ getFieldValue }) => {
                const needsCard =
                  getFieldValue("message_type") === "card" ||
                  getFieldValue("cron_message_type") === "card";
                if (!needsCard) return null;
                return (
                  <>
                    <Form.Item
                      name="card_template_id"
                      label="Card Template ID"
                      rules={[
                        {
                          required: true,
                          message:
                            "Please input card template id when message_type=card",
                        },
                      ]}
                    >
                      <Input placeholder="dt_card_template_xxx" />
                    </Form.Item>
                    <Form.Item
                      name="card_template_key"
                      label="Card Template Key"
                      tooltip="Must exactly match the template variable name"
                    >
                      <Input placeholder="content" />
                    </Form.Item>
                    <Form.Item
                      name="robot_code"
                      label="Robot Code"
                      tooltip="Recommended to configure explicitly for group chats"
                    >
                      <Input placeholder="robot code (default client_id)" />
                    </Form.Item>
                  </>
                );
              }}
            </Form.Item>
            <Form.Item
              name="at_sender_on_reply"
              label={t("channels.atSenderOnReply")}
              tooltip={t("channels.atSenderOnReplyTooltip")}
              valuePropName="checked"
            >
              <Switch />
            </Form.Item>
          </>
        );

      case "feishu":
        return (
          <>
            <Form.Item
              name="domain"
              label={t("channels.feishuRegion")}
              initialValue="feishu"
              tooltip={t("channels.feishuRegionTooltip")}
            >
              <Select>
                <Select.Option value="feishu">
                  {t("channels.feishuChina")}
                </Select.Option>
                <Select.Option value="lark">
                  {t("channels.feishuInternational")}
                </Select.Option>
              </Select>
            </Form.Item>
            <Form.Item
              name="app_id"
              label="App ID"
              rules={[{ required: true }]}
            >
              <Input placeholder="cli_xxx" />
            </Form.Item>
            <Form.Item
              name="app_secret"
              label="App Secret"
              rules={[{ required: true }]}
            >
              <Input.Password placeholder="App Secret" />
            </Form.Item>
            <Form.Item name="encrypt_key" label="Encrypt Key">
              <Input placeholder="Optional, for event encryption" />
            </Form.Item>
            <Form.Item name="verification_token" label="Verification Token">
              <Input placeholder="Optional" />
            </Form.Item>
            <Form.Item name="media_dir" label={t("channels.weixinMediaDir")}>
              <Input placeholder={defaultMediaDir} />
            </Form.Item>
          </>
        );

      case "qq":
        return (
          <>
            <Form.Item
              name="app_id"
              label="App ID"
              rules={[{ required: true }]}
            >
              <Input />
            </Form.Item>
            <Form.Item
              name="client_secret"
              label="Client Secret"
              rules={[{ required: true }]}
            >
              <Input.Password />
            </Form.Item>
            <Form.Item
              name="ack_message"
              label={t("channels.ackMessage")}
              tooltip={t("channels.ackMessageTooltip")}
            >
              <Input placeholder={t("channels.ackMessagePlaceholder")} />
            </Form.Item>
          </>
        );

      case "telegram":
        return (
          <>
            <Form.Item
              name="bot_token"
              label="Bot Token"
              rules={[{ required: true }]}
            >
              <Input.Password placeholder="Telegram bot token from BotFather" />
            </Form.Item>
            <Form.Item name="http_proxy" label="HTTP Proxy">
              <Input placeholder="http://127.0.0.1:18118" />
            </Form.Item>
            <Form.Item name="http_proxy_auth" label="HTTP Proxy Auth">
              <Input placeholder="user:password" />
            </Form.Item>
            <Form.Item
              name="show_typing"
              label="Show Typing"
              valuePropName="checked"
            >
              <Switch />
            </Form.Item>
          </>
        );

      case "mqtt":
        return (
          <>
            <Form.Item
              name="host"
              label="MQTT Host"
              rules={[{ required: true }]}
            >
              <Input placeholder="127.0.0.1" />
            </Form.Item>
            <Form.Item
              name="port"
              label="MQTT Port"
              rules={[
                { required: true },
                {
                  type: "number",
                  min: 1,
                  max: 65535,
                  message: "Port must be between 1 and 65535",
                },
              ]}
            >
              <InputNumber
                min={1}
                max={65535}
                style={{ width: "100%" }}
                placeholder="1883"
              />
            </Form.Item>
            <Form.Item
              name="transport"
              label="Transport"
              initialValue="tcp"
              rules={[{ required: true }]}
            >
              <Select>
                <Select.Option value="tcp">MQTT (tcp)</Select.Option>
                <Select.Option value="websockets">
                  WS (websockets)
                </Select.Option>
              </Select>
            </Form.Item>
            <Form.Item
              name="clean_session"
              label="Clean Session"
              valuePropName="checked"
            >
              <Switch defaultChecked />
            </Form.Item>
            <Form.Item
              name="qos"
              label="QoS"
              initialValue="2"
              rules={[{ required: true }]}
            >
              <Select>
                <Select.Option value="0">At Most Once (0)</Select.Option>
                <Select.Option value="1">At Least Once (1)</Select.Option>
                <Select.Option value="2">Exactly Once (2)</Select.Option>
              </Select>
            </Form.Item>
            <Form.Item name="username" label="MQTT Username">
              <Input placeholder="Leave blank to disable / not use" />
            </Form.Item>
            <Form.Item name="password" label="MQTT Password">
              <Input.Password placeholder="Leave blank to disable / not use" />
            </Form.Item>
            <Form.Item
              name="subscribe_topic"
              label="Subscribe Topic"
              rules={[{ required: true }]}
            >
              <Input placeholder="server/+/up" />
            </Form.Item>
            <Form.Item
              name="publish_topic"
              label="Publish Topic"
              rules={[{ required: true }]}
            >
              <Input placeholder="client/{client_id}/down" />
            </Form.Item>
            <Form.Item
              name="tls_enabled"
              label="TLS Enabled"
              valuePropName="checked"
            >
              <Switch />
            </Form.Item>
            <Form.Item name="tls_ca_certs" label="TLS CA Certs">
              <Input placeholder="Path to CA certificates file" />
            </Form.Item>
            <Form.Item name="tls_certfile" label="TLS Certfile">
              <Input placeholder="Path to client certificate file" />
            </Form.Item>
            <Form.Item name="tls_keyfile" label="TLS Keyfile">
              <Input placeholder="Path to client private key file" />
            </Form.Item>
          </>
        );

      case "mattermost":
        return (
          <>
            <Form.Item
              name="url"
              label="Mattermost URL"
              rules={[{ required: true }]}
            >
              <Input placeholder="https://mattermost.example.com" />
            </Form.Item>
            <Form.Item
              name="bot_token"
              label="Bot Token"
              rules={[{ required: true }]}
            >
              <Input.Password placeholder="Mattermost bot token" />
            </Form.Item>
            <Form.Item name="media_dir" label={t("channels.weixinMediaDir")}>
              <Input placeholder={defaultMediaDir} />
            </Form.Item>
            <Form.Item
              name="show_typing"
              label="Show Typing"
              valuePropName="checked"
            >
              <Switch />
            </Form.Item>
            <Form.Item
              name="thread_follow_without_mention"
              label="Thread Follow Without Mention"
              valuePropName="checked"
            >
              <Switch />
            </Form.Item>
          </>
        );

      case "voice":
        return (
          <>
            <ConfigProvider prefixCls="ant">
              <Alert
                type="info"
                showIcon
                message={t("channels.voiceSetupGuide")}
                style={{ marginBottom: 16 }}
              />
            </ConfigProvider>
            <Form.Item
              name="twilio_account_sid"
              label={t("channels.twilioAccountSid")}
              rules={[{ required: true }]}
            >
              <Input placeholder="ACxxxxxxxx" />
            </Form.Item>
            <Form.Item
              name="twilio_auth_token"
              label={t("channels.twilioAuthToken")}
              rules={[{ required: true }]}
            >
              <Input.Password />
            </Form.Item>
            <Form.Item name="phone_number" label={t("channels.phoneNumber")}>
              <Input placeholder="+15551234567" />
            </Form.Item>
            <Form.Item
              name="phone_number_sid"
              label={t("channels.phoneNumberSid")}
              tooltip={t("channels.phoneNumberSidHelp")}
            >
              <Input placeholder="PNxxxxxxxx" />
            </Form.Item>
            <Form.Item name="tts_provider" label={t("channels.ttsProvider")}>
              <Input placeholder="google" />
            </Form.Item>
            <Form.Item name="tts_voice" label={t("channels.ttsVoice")}>
              <Input placeholder="en-US-Journey-D" />
            </Form.Item>
            <Form.Item name="stt_provider" label={t("channels.sttProvider")}>
              <Input placeholder="deepgram" />
            </Form.Item>
            <Form.Item name="language" label={t("channels.language")}>
              <Input placeholder="en-US" />
            </Form.Item>
            <Form.Item
              name="welcome_greeting"
              label={t("channels.welcomeGreeting")}
            >
              <Input.TextArea rows={2} />
            </Form.Item>
          </>
        );

      case "sip":
        return (
          <>
            <ConfigProvider prefixCls="ant">
              <Alert
                type="info"
                showIcon
                message={t("channels.sipSetupGuide")}
                style={{ marginBottom: 16 }}
              />
            </ConfigProvider>
            <Form.Item
              name="sip_mode"
              label={t("channels.sipMode")}
              tooltip={t("channels.sipModeTooltip")}
              initialValue="dev"
            >
              <Select
                options={[
                  { value: "dev", label: "Dev (pyVoIP)" },
                  { value: "livekit", label: "Production (LiveKit)" },
                ]}
              />
            </Form.Item>
            <Form.Item
              shouldUpdate={(
                prev: Record<string, unknown>,
                cur: Record<string, unknown>,
              ) => prev.sip_mode !== cur.sip_mode}
              noStyle
            >
              {({
                getFieldValue,
              }: {
                getFieldValue: (name: string) => unknown;
              }) => (
                <Form.Item name="sip_server" label={t("channels.sipServer")}>
                  <Input
                    placeholder={
                      getFieldValue("sip_mode") === "livekit"
                        ? t("channels.sipServerPlaceholderLivekit")
                        : t("channels.sipServerPlaceholder")
                    }
                  />
                </Form.Item>
              )}
            </Form.Item>
            <Form.Item name="sip_username" label={t("channels.sipUsername")}>
              <Input placeholder="1001" />
            </Form.Item>
            <Form.Item name="sip_password" label={t("channels.sipPassword")}>
              <Input.Password />
            </Form.Item>
            <Form.Item
              name="sip_port"
              label={t("channels.sipPort")}
              rules={[
                {
                  type: "number",
                  min: 1,
                  max: 65535,
                },
              ]}
            >
              <InputNumber
                min={1}
                max={65535}
                style={{ width: "100%" }}
                placeholder="5061"
              />
            </Form.Item>
            <Form.Item
              name="sip_transport"
              label={t("channels.sipTransport")}
              initialValue="UDP"
            >
              <Select
                options={[
                  { value: "UDP", label: "UDP" },
                  { value: "TCP", label: "TCP" },
                  { value: "TLS", label: "TLS" },
                ]}
              />
            </Form.Item>
            <Form.Item
              name="dashscope_api_key"
              label={t("channels.sipDashscopeApiKey")}
              tooltip={t("channels.sipDashscopeApiKeyTooltip")}
            >
              <Input.Password placeholder="sk-..." />
            </Form.Item>
            <Form.Item name="tts_provider" label={t("channels.ttsProvider")}>
              <Input placeholder="aliyun" />
            </Form.Item>
            <Form.Item name="tts_voice" label={t("channels.ttsVoice")}>
              <Input placeholder="longxiaochun" />
            </Form.Item>
            <Form.Item name="stt_provider" label={t("channels.sttProvider")}>
              <Input placeholder="aliyun" />
            </Form.Item>
            <Form.Item name="language" label={t("channels.language")}>
              <Input placeholder="zh-CN" />
            </Form.Item>
            <Form.Item
              name="welcome_greeting"
              label={t("channels.welcomeGreeting")}
            >
              <Input.TextArea rows={2} />
            </Form.Item>
            <Form.Item
              noStyle
              shouldUpdate={(prev, cur) => prev.sip_mode !== cur.sip_mode}
            >
              {({ getFieldValue }) => {
                if (getFieldValue("sip_mode") !== "livekit") return null;
                return (
                  <>
                    <Form.Item
                      name="livekit_url"
                      label={t("channels.livekitUrl")}
                      rules={[{ required: true }]}
                    >
                      <Input placeholder="ws://localhost:7880" />
                    </Form.Item>
                    <Form.Item
                      name="livekit_api_key"
                      label={t("channels.livekitApiKey")}
                      rules={[{ required: true }]}
                    >
                      <Input />
                    </Form.Item>
                    <Form.Item
                      name="livekit_api_secret"
                      label={t("channels.livekitApiSecret")}
                      rules={[{ required: true }]}
                    >
                      <Input.Password />
                    </Form.Item>
                    <Form.Item
                      name="livekit_sip_trunk_id"
                      label={t("channels.livekitSipTrunkId")}
                    >
                      <Input placeholder="ST_xxxx" />
                    </Form.Item>
                    <Form.Item
                      name="livekit_room_name"
                      label={t("channels.livekitRoomName")}
                      tooltip={t("channels.livekitRoomNameTooltip")}
                    >
                      <Input placeholder="sip-inbound" />
                    </Form.Item>
                  </>
                );
              }}
            </Form.Item>
          </>
        );

      case "wecom":
        return (
          <>
            <ConfigProvider prefixCls="ant">
              <Alert
                type="warning"
                showIcon
                message={t("channels.wecomSetupGuide")}
                style={{ marginBottom: 16 }}
              />
            </ConfigProvider>
            <Form.Item label={t("channels.wecomScanAuth")}>
              <Button
                type="primary"
                block
                loading={wecomQrcode.loading}
                onClick={wecomQrcode.fetchQrcode}
              >
                {t("channels.loginWeCom")}
              </Button>
              {wecomQrcode.loading && (
                <div style={{ textAlign: "center", marginTop: 12 }}>
                  <Spin />
                </div>
              )}
              {wecomQrcode.qrcodeImg && !wecomQrcode.loading && (
                <div style={{ textAlign: "center", marginTop: 12 }}>
                  <img
                    src={`data:image/png;base64,${wecomQrcode.qrcodeImg}`}
                    alt="WeCom QR Code"
                    style={{ width: 200, height: 200 }}
                  />
                  <div
                    style={{
                      marginTop: 8,
                      fontSize: 12,
                      color: isDark
                        ? "rgba(255,255,255,0.45)"
                        : "rgba(0,0,0,0.45)",
                    }}
                  >
                    {t("channels.wecomAuthHint")}
                  </div>
                </div>
              )}
            </Form.Item>
            <Form.Item
              name="bot_id"
              label="Bot ID"
              rules={[{ required: true, message: "Please input Bot ID" }]}
            >
              <Input placeholder="Bot ID from WeCom backend" />
            </Form.Item>
            <Form.Item
              name="secret"
              label="Secret"
              rules={[{ required: true, message: "Please input Secret" }]}
            >
              <Input.Password placeholder="Secret from WeCom backend" />
            </Form.Item>
            <Form.Item name="media_dir" label={t("channels.weixinMediaDir")}>
              <Input placeholder={defaultMediaDir} />
            </Form.Item>
            <Form.Item
              name="welcome_text"
              label={t("channels.welcomeText")}
              tooltip={t("channels.welcomeTextTooltip")}
            >
              <Input placeholder={t("channels.welcomeTextPlaceholder")} />
            </Form.Item>
          </>
        );

      case "xiaoyi":
        return (
          <>
            <ConfigProvider prefixCls="ant">
              <Alert
                type="info"
                showIcon
                message={t("channels.xiaoyiSetupGuide")}
                style={{ marginBottom: 16 }}
              />
            </ConfigProvider>
            <Form.Item
              name="ak"
              label="Access Key (AK)"
              rules={[{ required: true, message: "Please input Access Key" }]}
            >
              <Input placeholder="Access Key from Huawei Developer Platform" />
            </Form.Item>
            <Form.Item
              name="sk"
              label="Secret Key (SK)"
              rules={[{ required: true, message: "Please input Secret Key" }]}
            >
              <Input.Password placeholder="Secret Key from Huawei Developer Platform" />
            </Form.Item>
            <Form.Item
              name="agent_id"
              label="Agent ID"
              rules={[{ required: true, message: "Please input Agent ID" }]}
            >
              <Input placeholder="Agent ID from XiaoYi platform" />
            </Form.Item>
            <Form.Item name="ws_url" label="WebSocket URL">
              <Input placeholder="wss://hag.cloud.huawei.com/openclaw/v1/ws/link" />
            </Form.Item>
          </>
        );

      case "weixin":
        return (
          <>
            <ConfigProvider prefixCls="ant">
              <Alert
                type="info"
                showIcon
                message={t("channels.weixinSetupGuide")}
                style={{ marginBottom: 16 }}
              />
              <Alert
                type="warning"
                showIcon
                message={t("channels.weixinContextTokenLimit")}
                style={{ marginBottom: 16 }}
              />
            </ConfigProvider>
            <Form.Item label={t("channels.weixinScanLogin")}>
              <Button
                type="primary"
                block
                loading={weixinQrcode.loading}
                onClick={weixinQrcode.fetchQrcode}
              >
                {t("channels.weixinGetQrcode")}
              </Button>
              {weixinQrcode.loading && (
                <div style={{ textAlign: "center", marginTop: 12 }}>
                  <Spin />
                </div>
              )}
              {weixinQrcode.qrcodeImg && !weixinQrcode.loading && (
                <div style={{ textAlign: "center", marginTop: 12 }}>
                  <img
                    src={`data:image/png;base64,${weixinQrcode.qrcodeImg}`}
                    alt="WeChat QR Code"
                    style={{ width: 200, height: 200 }}
                  />
                  <div
                    style={{
                      marginTop: 8,
                      fontSize: 12,
                      color: isDark
                        ? "rgba(255,255,255,0.45)"
                        : "rgba(0,0,0,0.45)",
                    }}
                  >
                    {t("channels.weixinScanHint")}
                  </div>
                </div>
              )}
            </Form.Item>
            <Form.Item
              name="bot_token"
              label={t("channels.weixinBotToken")}
              tooltip={t("channels.weixinBotTokenTooltip")}
            >
              <Input.Password
                placeholder={t("channels.weixinBotTokenPlaceholder")}
              />
            </Form.Item>
            <Form.Item
              name="bot_token_file"
              label={t("channels.weixinBotTokenFile")}
              tooltip={t("channels.weixinBotTokenFileTooltip")}
            >
              <Input placeholder="~/.qwenpaw/weixin_bot_token" />
            </Form.Item>
            <Form.Item name="media_dir" label={t("channels.weixinMediaDir")}>
              <Input placeholder={defaultMediaDir} />
            </Form.Item>
          </>
        );

      case "whatsapp":
        return (
          <>
            <Form.Item label={t("channels.whatsappConnection")}>
              {(waPairStatus === "connected" || waLinked) ? (
                <>
                  <Alert
                    type="success"
                    showIcon
                    message={t("channels.whatsappConnected")}
                    description={t("channels.whatsappSessionActive")}
                    style={{ marginBottom: 12 }}
                  />
                  <Button
                    danger
                    block
                    loading={waPairLoading}
                    onClick={handleWhatsappUnbind}
                  >
                    {t("channels.whatsappUnbind")}
                  </Button>
                </>
              ) : (
                <>
                  <Input
                    placeholder={t("channels.whatsappPhonePlaceholder")}
                    value={waPhone}
                    onChange={(e) => setWaPhone(e.target.value)}
                    style={{ marginBottom: 8 }}
                  />
                  <Button
                    type="primary"
                    block
                    loading={waPairLoading}
                    onClick={handleWhatsappPair}
                    disabled={!waPhone}
                  >
                    {t("channels.whatsappGetPairCode")}
                  </Button>
                  <Button
                    style={{ marginTop: 8 }}
                    block
                    onClick={async () => {
                      setWaPairLoading(true);
                      setWaQrImage("");
                      try {
                        const data = await api.getWhatsappQrcode();
                        if (data.qr_image) {
                          setWaQrImage(data.qr_image);
                          setWaPairStatus("waiting_qr");
                        }
                      } catch { /* ignore */ }
                      setWaPairLoading(false);
                    }}
                  >
                    {t("channels.whatsappShowQR")}
                  </Button>
                  {waPairCode && (
                    <div style={{ textAlign: "center", marginTop: 12, padding: "16px", background: "rgba(0,0,0,0.05)", borderRadius: 8 }}>
                      <div style={{ fontSize: 24, fontWeight: "bold", letterSpacing: 4 }}>{waPairCode}</div>
                      <div style={{ marginTop: 8, fontSize: 12, opacity: 0.6 }}>
                        {t("channels.whatsappPairInstructions")}
                      </div>
                    </div>
                  )}
                  {waQrImage && (
                    <div style={{ textAlign: "center", marginTop: 12 }}>
                      <img
                        src={`data:image/png;base64,${waQrImage}`}
                        alt="WhatsApp QR Code"
                        style={{ width: 200, height: 200 }}
                      />
                      <div style={{ marginTop: 8, fontSize: 12, opacity: 0.6 }}>
                        {t("channels.whatsappScanQR")}
                      </div>
                    </div>
                  )}
                </>
              )}
              {waLinked && (
                <Button
                  danger
                  block
                  style={{ marginTop: 8 }}
                  onClick={async () => {
                    await api.unbindWhatsapp();
                    setWaLinked(false);
                    setWaPairCode("");
                    setWaQrImage("");
                    setWaPairStatus("idle");
                    message.success(t("channels.whatsappUnlinked"));
                  }}
                >
                  {t("channels.whatsappUnlinkDevice")}
                </Button>
              )}
            </Form.Item>
            <ConfigProvider prefixCls="ant">
              <Alert
                type="info"
                showIcon
                message={t("channels.whatsappAuthInfo")}
                style={{ marginBottom: 16 }}
              />
            </ConfigProvider>
            <Form.Item
              name="auth_dir"
              label={t("channels.whatsappAuthDir")}
              tooltip={t("channels.whatsappAuthDirTooltip")}
            >
              {/* Intentionally no initialValue — leave the field empty so the
                  backend resolves the default via `_resolve_wa_auth_dir`
                  (explicit > workspace_dir > WORKING_DIR). Setting a fixed
                  initialValue would persist a hard path into agent.json and
                  defeat per-agent/workspace scoping. */}
              <Input placeholder="$WORKING_DIR/credentials/whatsapp/default (auto)" />
            </Form.Item>
            <Form.Item
              name="send_read_receipts"
              label={t("channels.whatsappReadReceipts")}
              valuePropName="checked"
              initialValue={true}
            >
              <Switch defaultChecked />
            </Form.Item>
            <Form.Item
              name="text_chunk_limit"
              label={t("channels.whatsappTextChunkLimit")}
              tooltip={t("channels.whatsappTextChunkLimitTooltip")}
              initialValue={4096}
            >
              <InputNumber min={256} max={8192} step={256} style={{ width: "100%" }} />
            </Form.Item>
            <Form.Item name="self_chat_mode" label={t("channels.whatsappSelfChatMode")} valuePropName="checked" tooltip={t("channels.whatsappSelfChatModeTooltip")}>
              <Switch />
            </Form.Item>
            <Form.Item name="groups" label={t("channels.whatsappGroupAllowlist")} tooltip={t("channels.whatsappGroupAllowlistTooltip")}>
              <Select mode="tags" placeholder="120363421135228220@g.us" tokenSeparators={[","," ","\n"]} />
            </Form.Item>
            <Form.Item name="group_allow_from" label={t("channels.whatsappGroupAllowFrom")} tooltip={t("channels.whatsappGroupAllowFromTooltip")}>
              <Select mode="tags" placeholder="* (everyone)" tokenSeparators={[","," "]} />
            </Form.Item>
            <Form.Item name="reply_to_trigger" label={t("channels.replyToTrigger")} valuePropName="checked" tooltip={t("channels.replyToTriggerTooltip")} initialValue={true}>
              <Switch defaultChecked />
            </Form.Item>
            {/* filter_thinking is rendered by the shared global section below — don't duplicate it here */}
          </>
        );


      case "signal":
        return (
          <>
            <Form.Item label={t("channels.signalConnection")}>
              {sigLinked || sigPairStatus === "linked" ? (
                <>
                  <Alert
                    type="success"
                    showIcon
                    message={t("channels.signalLinked")}
                    description={
                      sigPhone
                        ? `${t("channels.signalLinkedAs")}: ${sigPhone}`
                        : t("channels.signalSessionActive")
                    }
                    style={{ marginBottom: 12 }}
                  />
                  <Button
                    danger
                    block
                    loading={sigPairLoading}
                    onClick={handleSignalUnbind}
                  >
                    {t("channels.signalUnlinkDevice")}
                  </Button>
                </>
              ) : (
                <>
                  <Input
                    placeholder={t("channels.signalDeviceNamePlaceholder")}
                    value={sigDeviceName}
                    onChange={(e) => setSigDeviceName(e.target.value)}
                    style={{ marginBottom: 8 }}
                  />
                  <Button
                    type="primary"
                    block
                    loading={sigPairLoading}
                    onClick={handleSignalLink}
                  >
                    {t("channels.signalLinkDevice")}
                  </Button>
                  {sigQrImage && (
                    <div style={{ textAlign: "center", marginTop: 12 }}>
                      <img
                        src={`data:image/png;base64,${sigQrImage}`}
                        alt="Signal link QR"
                        style={{ width: 220, height: 220 }}
                      />
                      <div
                        style={{
                          marginTop: 8,
                          fontSize: 12,
                          opacity: 0.7,
                        }}
                      >
                        {t("channels.signalScanInstructions")}
                      </div>
                    </div>
                  )}
                  {sigPairStatus === "error" && (
                    <div style={{ marginTop: 8 }}>
                      <Alert
                        type="error"
                        showIcon
                        message={t("channels.signalLinkFailed")}
                      />
                    </div>
                  )}
                </>
              )}
            </Form.Item>
            {/*
              account + account_uuid are read-only: they're the identity
              signal-cli itself authoritative-told us (populated from the
              linked account store on drawer open, or from the link-device
              flow's success handler). Letting users type a different
              phone here would silently detach the channel config from
              the actual session data_dir — the channel would either fail
              to start ("User +XXX is not registered") or connect to a
              different account than the UI shows. Better to force users
              through the Link Device flow to change these.
            */}
            <Form.Item
              name="account"
              label={t("channels.signalAccount")}
              tooltip={t("channels.signalAccountTooltip")}
              rules={[{ required: true }]}
            >
              <Input placeholder="+85212345678" readOnly />
            </Form.Item>
            <Form.Item
              name="account_uuid"
              label={t("channels.signalAccountUuid")}
              tooltip={t("channels.signalAccountUuidTooltip")}
            >
              <Input
                placeholder="447e962a-0000-0000-0000-000000000000"
                readOnly
              />
            </Form.Item>
            <Form.Item
              name="signal_cli_path"
              label={t("channels.signalCliPath")}
              tooltip={t("channels.signalCliPathTooltip")}
            >
              <Input placeholder={t("channels.signalCliPathPlaceholder")} />
            </Form.Item>
            <Form.Item
              name="data_dir"
              label={t("channels.signalDataDir")}
              tooltip={t("channels.signalDataDirTooltip")}
            >
              <Input placeholder={t("channels.signalDataDirPlaceholder")} />
            </Form.Item>
            <Form.Item
              name="extra_args"
              label={t("channels.signalExtraArgs")}
              tooltip={t("channels.signalExtraArgsTooltip")}
              initialValue={[]}
            >
              <Select
                mode="tags"
                placeholder={t("channels.signalExtraArgsPlaceholder")}
                tokenSeparators={[","]}
              />
            </Form.Item>
            <Form.Item
              name="show_typing"
              label={t("channels.signalShowTyping")}
              valuePropName="checked"
            >
              <Switch />
            </Form.Item>
            <Form.Item
              name="send_read_receipts"
              label={t("channels.signalSendReadReceipts")}
              valuePropName="checked"
            >
              <Switch />
            </Form.Item>
            <Form.Item
              name="text_chunk_limit"
              label={t("channels.signalTextChunkLimit")}
              tooltip={t("channels.signalTextChunkLimitTooltip")}
            >
              <InputNumber
                min={100}
                max={10000}
                style={{ width: "100%" }}
                placeholder="4000"
              />
            </Form.Item>
            <Form.Item
              name="groups"
              label={t("channels.signalGroups")}
              tooltip={t("channels.signalGroupsTooltip")}
              initialValue={[]}
            >
              <Select
                mode="tags"
                placeholder={t("channels.signalGroupsPlaceholder")}
                tokenSeparators={[","]}
                // Populated from signal-cli's group_v2 table. Group names
                // are in a protobuf BLOB we don't decode yet, so the
                // label is just a truncated base64 id — still lets users
                // pick without typing 40-char strings. Free-form tag
                // input still works for custom entries.
                options={
                  sigGroups.length
                    ? sigGroups.map((g) => ({
                        value: g.id,
                        label: `${g.id.slice(0, 20)}…${g.blocked ? " (blocked)" : ""}`,
                      }))
                    : undefined
                }
              />
            </Form.Item>
            <Form.Item
              name="group_allow_from"
              label={t("channels.signalGroupAllowFrom")}
              tooltip={t("channels.signalGroupAllowFromTooltip")}
              initialValue={[]}
            >
              <Select
                mode="tags"
                placeholder={t("channels.signalGroupAllowFromPlaceholder")}
                tokenSeparators={[","]}
                options={
                  sigContacts.length
                    ? [
                        { value: "*", label: "* (everyone)" },
                        ...sigContacts.map((c) => {
                          const value = c.number || (c.uuid ? `uuid:${c.uuid}` : "");
                          const label = [c.name, c.number, c.uuid && `uuid:${c.uuid.slice(0, 8)}…`]
                            .filter(Boolean)
                            .join(" · ");
                          return { value, label: label || value };
                        }),
                      ]
                    : undefined
                }
              />
            </Form.Item>
            <Form.Item
              name="reply_to_trigger"
              label={t("channels.signalReplyToTrigger")}
              tooltip={t("channels.signalReplyToTriggerTooltip")}
              valuePropName="checked"
            >
              <Switch />
            </Form.Item>
          </>
        );


      case "onebot":
        return (
          <>
            <Form.Item
              name="ws_host"
              label="WebSocket Host"
              rules={[{ required: true }]}
            >
              <Input placeholder="0.0.0.0" />
            </Form.Item>
            <Form.Item
              name="ws_port"
              label="WebSocket Port"
              rules={[
                { required: true },
                {
                  type: "number",
                  min: 1,
                  max: 65535,
                  message: "Port must be between 1 and 65535",
                },
              ]}
            >
              <InputNumber
                min={1}
                max={65535}
                style={{ width: "100%" }}
                placeholder="6199"
              />
            </Form.Item>
            <Form.Item name="access_token" label="Access Token">
              <Input.Password placeholder="Access token for authentication" />
            </Form.Item>
            <Form.Item
              name="share_session_in_group"
              label={t("channels.onebotShareSessionInGroup")}
              valuePropName="checked"
              tooltip={t("channels.onebotShareSessionInGroupTooltip")}
            >
              <Switch />
            </Form.Item>
          </>
        );

      default:
        return null;
    }
  };

  // ── Custom channel fields (key-value editor) ─────────────────────────────

  const renderCustomExtraFields = (
    values: Record<string, unknown> | undefined,
  ) => {
    if (!values) return null;
    const extraKeys = Object.keys(values).filter(
      (k) => !BASE_FIELDS.includes(k),
    );
    if (extraKeys.length === 0) return null;

    return (
      <>
        <div style={{ marginBottom: 8, fontWeight: 500 }}>Custom Fields</div>
        {extraKeys.map((fieldKey) => {
          const value = values[fieldKey];
          return (
            <Form.Item key={fieldKey} name={fieldKey} label={fieldKey}>
              {typeof value === "boolean" ? (
                <Switch />
              ) : typeof value === "number" ? (
                <InputNumber style={{ width: "100%" }} />
              ) : (
                <Input />
              )}
            </Form.Item>
          );
        })}
      </>
    );
  };

  // ── Drawer title ─────────────────────────────────────────────────────────

  const drawerTitle = (
    <div className={styles.drawerTitle}>
      <span>
        {label
          ? `${label} ${t("channels.settings")}`
          : t("channels.channelSettings")}
      </span>
      {activeKey &&
        CHANNEL_DOC_EN_URLS[activeKey] &&
        CHANNEL_DOC_ZH_URLS[activeKey] && (
          <Button
            type="text"
            size="small"
            icon={<LinkOutlined />}
            onClick={() => {
              const url =
                CHANNEL_DOC_EN_URLS[activeKey]! ||
                CHANNEL_DOC_ZH_URLS[activeKey]!;
              const isQwenPawDoc = url.includes(
                "qwenpaw.agentscope.io/docs/channels/",
              );
              const finalUrl =
                isQwenPawDoc && currentLang === "zh"
                  ? CHANNEL_DOC_ZH_URLS[activeKey]!
                  : CHANNEL_DOC_EN_URLS[activeKey]!;
              window.open(finalUrl, "_blank");
            }}
            className={styles.dingtalkDocBtn}
            style={{ color: "#FF7F16" }}
          >
            {label} Doc
          </Button>
        )}
      {activeKey === "voice" && (
        <Button
          type="text"
          size="small"
          icon={<LinkOutlined />}
          onClick={() =>
            window.open(TWILIO_CONSOLE_URL, "_blank", "noopener,noreferrer")
          }
          className={styles.dingtalkDocBtn}
          style={{ color: "#FF7F16" }}
        >
          {t("channels.voiceSetupLink")}
        </Button>
      )}
    </div>
  );

  // ── Render ───────────────────────────────────────────────────────────────

  const drawerFooter = (
    <div className={styles.formActions}>
      <Button onClick={onClose}>{t("common.cancel")}</Button>
      <Button type="primary" loading={saving} onClick={() => form.submit()}>
        {t("common.save")}
      </Button>
    </div>
  );

  return (
    <Drawer
      width={420}
      placement="right"
      title={drawerTitle}
      open={open}
      onClose={onClose}
      destroyOnClose
      footer={drawerFooter}
    >
      {activeKey && (
        <Form
          form={form}
          layout="vertical"
          initialValues={initialValues}
          onFinish={onSubmit}
        >
          <Form.Item
            name="enabled"
            label={t("common.enabled")}
            valuePropName="checked"
          >
            <Switch />
          </Form.Item>

          {activeKey !== "voice" && (
            <Form.Item name="bot_prefix" label="Bot Prefix">
              <Input placeholder="@bot" />
            </Form.Item>
          )}

          {activeKey !== "console" && (
            <>
              <Form.Item
                name="filter_tool_messages"
                label={t("channels.filterToolMessages")}
                valuePropName="checked"
                tooltip={t("channels.filterToolMessagesTooltip")}
              >
                <Switch />
              </Form.Item>
              <Form.Item
                name="filter_thinking"
                label={t("channels.filterThinking")}
                valuePropName="checked"
                tooltip={t("channels.filterThinkingTooltip")}
              >
                <Switch />
              </Form.Item>
            </>
          )}

          {isBuiltin
            ? renderBuiltinExtraFields(activeKey)
            : renderCustomExtraFields(initialValues)}

          {CHANNELS_WITH_ACCESS_CONTROL.includes(activeKey) &&
            renderAccessControlFields()}
        </Form>
      )}
    </Drawer>
  );
}
