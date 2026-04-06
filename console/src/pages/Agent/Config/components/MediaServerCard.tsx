import { Form, Card, Switch, Input, InputNumber, Alert } from "@agentscope-ai/design";
import { useTranslation } from "react-i18next";
import styles from "../index.module.less";

export function MediaServerCard() {
  const { t } = useTranslation();
  const mediaEnabled = Form.useWatch(["media_server", "enabled"]) ?? false;

  return (
    <Card
      className={styles.formCard}
      title={t("agentConfig.mediaServerTitle")}
      style={{ marginTop: 16 }}
    >
      <Alert
        type="info"
        showIcon
        message={t("agentConfig.mediaServerTooltip")}
        style={{ marginBottom: 16 }}
      />

      <Form.Item
        label={t("agentConfig.mediaServerEnabled")}
        name={["media_server", "enabled"]}
        valuePropName="checked"
      >
        <Switch />
      </Form.Item>

      <Form.Item
        label={t("agentConfig.mediaServerUrl")}
        name={["media_server", "server_url"]}
        tooltip={t("agentConfig.mediaServerUrlTooltip")}
      >
        <Input
          placeholder="http://localhost:8089"
          disabled={!mediaEnabled}
        />
      </Form.Item>

      <Form.Item
        label={t("agentConfig.mediaServerTunnelDomain")}
        name={["media_server", "tunnel_domain"]}
        tooltip={t("agentConfig.mediaServerTunnelDomainTooltip")}
      >
        <Input
          placeholder="https://media.example.com"
          disabled={!mediaEnabled}
        />
      </Form.Item>

      <Form.Item
        label={t("agentConfig.mediaServerSecret")}
        name={["media_server", "media_secret"]}
      >
        <Input.Password
          placeholder="copaw-media-2026"
          disabled={!mediaEnabled}
        />
      </Form.Item>

      <Form.Item
        label={t("agentConfig.mediaServerAllowedDirs")}
        name={["media_server", "allowed_dirs"]}
        tooltip={t("agentConfig.mediaServerAllowedDirsTooltip")}
        getValueFromEvent={(e: React.ChangeEvent<HTMLInputElement>) =>
          e.target.value.split(",").map((s: string) => s.trim()).filter(Boolean)
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
        label={t("agentConfig.mediaServerMaxSize")}
        name={["media_server", "max_size_mb"]}
        rules={[
          {
            type: "number",
            min: 1,
            message: t("agentConfig.mediaServerMaxSizeMin"),
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
    </Card>
  );
}
