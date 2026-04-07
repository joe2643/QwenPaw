import { useEffect, useState, useCallback } from "react";
import {
  Button,
  Card,
  Form,
  Input,
  InputNumber,
  Switch,
  Alert,
} from "@agentscope-ai/design";
import { useTranslation } from "react-i18next";
import { useAppMessage } from "../../../hooks/useAppMessage";
import { mediaServerApi } from "../../../api/modules/mediaServer";
import type {
  MediaServerConfig,
  MediaServerStatus,
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
              label={t("mediaServer.tunnelDomain")}
              name="tunnel_domain"
              tooltip={t("mediaServer.tunnelDomainTooltip")}
            >
              <Input
                placeholder="https://media.example.com"
                disabled={!mediaEnabled}
              />
            </Form.Item>

            <Form.Item
              label={t("mediaServer.secret")}
              name="media_secret"
            >
              <Input.Password
                placeholder="copaw-media-2026"
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
