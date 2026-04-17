import { useEffect, useState, useCallback } from "react";
import {
  Button,
  Card,
  Form,
  Input,
  InputNumber,
  Select,
  Switch,
  Alert,
} from "@agentscope-ai/design";
import { useTranslation } from "react-i18next";
import { useAppMessage } from "../../../hooks/useAppMessage";
import { mediaServerApi } from "../../../api/modules/mediaServer";
import type {
  MediaServerConfig,
  MediaServerStatus,
  TunnelMode,
} from "../../../api/modules/mediaServer";
import { PageHeader } from "@/components/PageHeader";
import styles from "./index.module.less";

export default function MediaServerPage() {
  const { t } = useTranslation();
  const { message } = useAppMessage();
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState<MediaServerStatus | null>(null);

  const fetchConfig = useCallback(async () => {
    setLoading(true);
    try {
      const cfg = await mediaServerApi.getConfig();
      form.setFieldsValue(cfg);
    } catch {
      message.error(t("mediaServer.loadFailed"));
    } finally {
      setLoading(false);
    }
  }, [form, message, t]);

  const fetchStatus = useCallback(async () => {
    try {
      const s = await mediaServerApi.getStatus();
      setStatus(s);
    } catch {
      setStatus(null);
    }
  }, []);

  useEffect(() => {
    fetchConfig();
    fetchStatus();
  }, [fetchConfig, fetchStatus]);

  const handleSave = async () => {
    try {
      const values = await form.validateFields();
      setSaving(true);
      await mediaServerApi.updateConfig(values as MediaServerConfig);
      message.success(t("mediaServer.saveSuccess"));
      fetchStatus();
    } catch {
      message.error(t("mediaServer.saveFailed"));
    } finally {
      setSaving(false);
    }
  };

  const mediaEnabled = Form.useWatch("enabled", form) ?? false;
  const tunnelMode: TunnelMode =
    (Form.useWatch("tunnel_mode", form) as TunnelMode) ?? "manual";
  const managedTunnel = tunnelMode === "quick" || tunnelMode === "named";

  // Poll status while a cloudflared tunnel is coming up so the URL appears
  // without requiring the user to refresh manually. Stops as soon as the
  // tunnel_url is known (cloudflared typically emits it within 2-5s).
  const tunnelPending = managedTunnel && mediaEnabled && !status?.tunnel_url;
  useEffect(() => {
    if (!tunnelPending) return;
    const id = setInterval(fetchStatus, 3000);
    return () => clearInterval(id);
  }, [tunnelPending, fetchStatus]);

  if (loading) {
    return (
      <div className={styles.mediaServerPage}>
        <PageHeader
          parent={t("nav.control")}
          current={t("nav.mediaServer")}
        />
        <div className={styles.pageContent}>
          <span>{t("common.loading")}</span>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.mediaServerPage}>
      <PageHeader
        parent={t("nav.control")}
        current={t("nav.mediaServer")}
      />
      <div className={styles.pageContent}>
        <Card>
          <Alert
            type="info"
            showIcon
            message={t("mediaServer.description")}
            style={{ marginBottom: 16 }}
          />

          {status && (
            <div style={{ marginBottom: 16 }}>
              <span
                className={`${styles.statusBadge} ${
                  status.running ? styles.statusRunning : styles.statusStopped
                }`}
              >
                {status.running
                  ? t("mediaServer.statusRunning")
                  : t("mediaServer.statusStopped")}
              </span>
            </div>
          )}

          <Form form={form} layout="vertical">
            <Form.Item
              label={t("mediaServer.enabled")}
              name="enabled"
              valuePropName="checked"
            >
              <Switch />
            </Form.Item>

            <Form.Item
              label={t("mediaServer.serverUrl")}
              name="server_url"
              tooltip={t("mediaServer.serverUrlTooltip")}
            >
              <Input
                placeholder="http://localhost:8089"
                disabled={!mediaEnabled}
              />
            </Form.Item>

            <Form.Item
              label={t("mediaServer.tunnelMode")}
              name="tunnel_mode"
              tooltip={t("mediaServer.tunnelModeTooltip")}
            >
              <Select
                disabled={!mediaEnabled}
                options={[
                  {
                    value: "manual",
                    label: t("mediaServer.tunnelModeManual"),
                  },
                  {
                    value: "quick",
                    label: t("mediaServer.tunnelModeQuick"),
                  },
                  {
                    value: "named",
                    label: t("mediaServer.tunnelModeNamed"),
                  },
                ]}
              />
            </Form.Item>

            {managedTunnel && status?.tunnel_url && (
              <Alert
                type="success"
                showIcon
                message={t("mediaServer.tunnelLive")}
                description={
                  <a
                    href={status.tunnel_url}
                    target="_blank"
                    rel="noreferrer noopener"
                  >
                    {status.tunnel_url}
                  </a>
                }
                style={{ marginBottom: 16 }}
              />
            )}
            {managedTunnel && mediaEnabled && !status?.tunnel_url && (
              <Alert
                type="warning"
                showIcon
                message={t("mediaServer.tunnelStarting")}
                style={{ marginBottom: 16 }}
              />
            )}

            <Form.Item
              label={t("mediaServer.tunnelDomain")}
              name="tunnel_domain"
              tooltip={t("mediaServer.tunnelDomainTooltip")}
            >
              <Input
                placeholder="https://media.example.com"
                disabled={!mediaEnabled || tunnelMode !== "manual"}
              />
            </Form.Item>

            {tunnelMode === "named" && (
              <>
                <Form.Item
                  label={t("mediaServer.namedTunnelName")}
                  name="named_tunnel_name"
                  tooltip={t("mediaServer.namedTunnelNameTooltip")}
                  rules={[
                    {
                      required: true,
                      message: t("mediaServer.namedTunnelNameRequired"),
                    },
                  ]}
                >
                  <Input
                    placeholder="media"
                    disabled={!mediaEnabled}
                  />
                </Form.Item>

                <Form.Item
                  label={t("mediaServer.namedTunnelHostname")}
                  name="named_tunnel_hostname"
                  tooltip={t("mediaServer.namedTunnelHostnameTooltip")}
                  rules={[
                    {
                      required: true,
                      message: t(
                        "mediaServer.namedTunnelHostnameRequired",
                      ),
                    },
                  ]}
                >
                  <Input
                    placeholder="media.example.com"
                    disabled={!mediaEnabled}
                  />
                </Form.Item>

                <Form.Item
                  label={t("mediaServer.namedTunnelConfigFile")}
                  name="named_tunnel_config_file"
                  tooltip={t("mediaServer.namedTunnelConfigFileTooltip")}
                >
                  <Input
                    placeholder="~/.cloudflared/config.yml"
                    disabled={!mediaEnabled}
                  />
                </Form.Item>
              </>
            )}

            <Form.Item
              label={t("mediaServer.secret")}
              name="media_secret"
            >
              <Input.Password
                placeholder="qwenpaw-media-2026"
                disabled={!mediaEnabled}
              />
            </Form.Item>

            <Form.Item
              label={t("mediaServer.allowedDirs")}
              name="allowed_dirs"
              tooltip={t("mediaServer.allowedDirsTooltip")}
              getValueFromEvent={(e: React.ChangeEvent<HTMLInputElement>) =>
                e.target.value
                  .split(",")
                  .map((s: string) => s.trim())
                  .filter(Boolean)
              }
              getValueProps={(value: string[]) => ({
                value: Array.isArray(value) ? value.join(", ") : value,
              })}
            >
              <Input
                placeholder="/tmp, /home/user/media"
                disabled={!mediaEnabled}
              />
            </Form.Item>

            <Form.Item
              label={t("mediaServer.maxSize")}
              name="max_size_mb"
              rules={[
                {
                  type: "number",
                  min: 1,
                  message: t("mediaServer.maxSizeMin"),
                },
              ]}
            >
              <InputNumber
                style={{ width: "100%" }}
                min={1}
                step={10}
                addonAfter="MB"
                disabled={!mediaEnabled}
              />
            </Form.Item>
          </Form>
        </Card>
      </div>

      <div className={styles.footerActions}>
        <Button
          onClick={fetchConfig}
          disabled={saving}
          style={{ marginRight: 8 }}
        >
          {t("common.reset")}
        </Button>
        <Button type="primary" onClick={handleSave} loading={saving}>
          {t("common.save")}
        </Button>
      </div>
    </div>
  );
}
