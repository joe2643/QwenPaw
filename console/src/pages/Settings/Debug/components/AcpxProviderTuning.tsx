import { useEffect, useState } from "react";
import {
  App,
  Button,
  Card,
  Col,
  Form,
  InputNumber,
  Row,
  Space,
  Typography,
} from "antd";
import { useTranslation } from "react-i18next";
import { acpxProviderApi } from "@/api/modules/acpxProvider";

const { Paragraph, Text } = Typography;

export function AcpxProviderTuning() {
  const { t } = useTranslation();
  const { message } = App.useApp();
  const [form] = Form.useForm<{
    turn_timeout_seconds: number;
    terminal_wait_seconds: number;
  }>();
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const cfg = await acpxProviderApi.getAcpxProviderConfig();
        if (cancelled) return;
        form.setFieldsValue(cfg);
      } catch (err) {
        if (cancelled) return;
        message.error(
          t(
            "debug.acpx.loadError",
            `Failed to load acpx config: ${err instanceof Error ? err.message : String(err)}`,
          ),
        );
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [form, message, t]);

  const onSave = async () => {
    try {
      setSaving(true);
      const values = await form.validateFields();
      const next = await acpxProviderApi.updateAcpxProviderConfig(values);
      form.setFieldsValue(next);
      message.success(t("debug.acpx.saved", "ACPX timeouts updated"));
    } catch (err) {
      if ((err as { errorFields?: unknown }).errorFields) return;
      message.error(
        t(
          "debug.acpx.saveError",
          `Failed to save: ${err instanceof Error ? err.message : String(err)}`,
        ),
      );
    } finally {
      setSaving(false);
    }
  };

  return (
    <Card
      title={t("debug.acpx.title", "Claude ACPX timeouts")}
      loading={loading}
    >
      <Paragraph type="secondary" style={{ marginBottom: 16 }}>
        {t(
          "debug.acpx.desc",
          "Live-applied tuning for the claude-acpx provider. Changes take effect on the next turn — no service restart needed.",
        )}
      </Paragraph>
      <Form form={form} layout="vertical">
        <Row gutter={16}>
          <Col xs={24} md={12}>
            <Form.Item
              name="turn_timeout_seconds"
              label={t(
                "debug.acpx.turnTimeout",
                "Per-turn stdout-stall timeout (seconds)",
              )}
              rules={[
                { required: true, message: "Required" },
                { type: "number", min: 10, max: 3600 },
              ]}
              extra={
                <Text type="secondary">
                  {t(
                    "debug.acpx.turnTimeoutHelp",
                    "Hard cap on a single acpx prompt subprocess. Bump for slow tool chains; lower for snappy chats. Default 300.",
                  )}
                </Text>
              }
            >
              <InputNumber
                style={{ width: "100%" }}
                min={10}
                max={3600}
                step={30}
              />
            </Form.Item>
          </Col>
          <Col xs={24} md={12}>
            <Form.Item
              name="terminal_wait_seconds"
              label={t(
                "debug.acpx.terminalWait",
                "Terminal wait_for_exit timeout (seconds)",
              )}
              rules={[
                { required: true, message: "Required" },
                { type: "number", min: 0, max: 7200 },
              ]}
              extra={
                <Text type="secondary">
                  {t(
                    "debug.acpx.terminalWaitHelp",
                    "Cap on a single Bash/command spawned by Claude Code in bypassPermissions mode. 0 = wait forever (legacy). Default 600.",
                  )}
                </Text>
              }
            >
              <InputNumber
                style={{ width: "100%" }}
                min={0}
                max={7200}
                step={30}
              />
            </Form.Item>
          </Col>
        </Row>
        <Space>
          <Button type="primary" loading={saving} onClick={onSave}>
            {t("common.save", "Save")}
          </Button>
        </Space>
      </Form>
    </Card>
  );
}
