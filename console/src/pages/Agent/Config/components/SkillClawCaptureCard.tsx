import { useState, useEffect, useCallback } from "react";
import { Card, Switch, Input, Select } from "@agentscope-ai/design";
import { Typography, Descriptions, Spin } from "antd";
import { ApiOutlined } from "@ant-design/icons";
import {
  skillclawCaptureApi,
  type SkillClawCaptureConfig,
  type SkillClawCaptureMode,
} from "@/api/modules/skillclaw_capture";
import styles from "../index.module.less";

const { Text } = Typography;

const DEFAULT_CONFIG: SkillClawCaptureConfig = {
  enabled: false,
  mode: "file",
  records_dir: "",
  ingest_url: "",
  ingest_api_key: "",
  session_id_prefix: "",
};

export function SkillClawCaptureCard() {
  const [config, setConfig] = useState<SkillClawCaptureConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  const loadConfig = useCallback(async () => {
    try {
      setLoading(true);
      const data = await skillclawCaptureApi.getConfig();
      setConfig(data);
    } catch {
      setConfig({ ...DEFAULT_CONFIG });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadConfig();
  }, [loadConfig]);

  // Optimistic update: state flips immediately, server validates and
  // returns the canonical config; we revert on failure to keep UI
  // honest about what's actually persisted.
  const update = async (patch: Partial<SkillClawCaptureConfig>) => {
    if (!config) return;
    const next = { ...config, ...patch };
    setConfig(next);
    setSaving(true);
    try {
      const result = await skillclawCaptureApi.updateConfig(next);
      setConfig(result);
    } catch {
      loadConfig();
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <Card
        className={styles.formCard}
        title="SkillClaw Capture"
        style={{ marginTop: 16 }}
      >
        <Spin />
      </Card>
    );
  }

  return (
    <Card
      className={styles.formCard}
      title={
        <>
          <ApiOutlined style={{ marginRight: 8 }} />
          SkillClaw Capture
        </>
      }
      style={{ marginTop: 16 }}
      extra={saving ? <Text type="secondary">saving...</Text> : null}
    >
      <Descriptions column={1} size="small" style={{ marginBottom: 0 }}>
        <Descriptions.Item label="Enabled">
          <Switch
            size="small"
            checked={config?.enabled ?? false}
            onChange={(v) => update({ enabled: v })}
          />
          <Text type="secondary" style={{ marginLeft: 8, fontSize: 12 }}>
            Append every turn to <code>conversations.jsonl</code> for SkillClaw
            evolve_server
          </Text>
        </Descriptions.Item>
      </Descriptions>

      {config?.enabled && (
        <Descriptions column={1} size="small" style={{ marginTop: 12 }}>
          <Descriptions.Item label="Transport">
            <Select
              size="small"
              value={config?.mode ?? "file"}
              onChange={(v: SkillClawCaptureMode) => update({ mode: v })}
              style={{ width: 120 }}
              options={[
                { value: "file", label: "file" },
                { value: "http", label: "http" },
              ]}
            />
            <Text type="secondary" style={{ marginLeft: 8, fontSize: 12 }}>
              {config?.mode === "http"
                ? "POST to SkillClaw ingest API (decouples from internal storage)"
                : "Append to local jsonl (no SkillClaw server needed)"}
            </Text>
          </Descriptions.Item>

          {config?.mode === "http" && (
            <>
              <Descriptions.Item label="Ingest URL">
                <Input
                  size="small"
                  placeholder="http://localhost:8787/v1/sessions/ingest"
                  value={config?.ingest_url ?? ""}
                  onChange={(e) => update({ ingest_url: e.target.value })}
                  style={{ maxWidth: 480 }}
                />
              </Descriptions.Item>
              <Descriptions.Item label="API key">
                <Input.Password
                  size="small"
                  placeholder="(optional bearer token)"
                  value={config?.ingest_api_key ?? ""}
                  onChange={(e) => update({ ingest_api_key: e.target.value })}
                  style={{ maxWidth: 360 }}
                />
              </Descriptions.Item>
            </>
          )}

          <Descriptions.Item
            label={config?.mode === "http" ? "Fallback dir" : "Records dir"}
          >
            <Input
              size="small"
              placeholder="(default: ~/.skillclaw/records)"
              value={config?.records_dir ?? ""}
              onChange={(e) => update({ records_dir: e.target.value })}
              style={{ maxWidth: 360 }}
            />
            {config?.mode === "http" && (
              <Text type="secondary" style={{ marginLeft: 8, fontSize: 12 }}>
                used if HTTP POST fails
              </Text>
            )}
          </Descriptions.Item>

          <Descriptions.Item label="Session ID prefix">
            <Input
              size="small"
              placeholder="(optional, e.g. copaw-default--)"
              value={config?.session_id_prefix ?? ""}
              onChange={(e) => update({ session_id_prefix: e.target.value })}
              style={{ maxWidth: 360 }}
            />
          </Descriptions.Item>
        </Descriptions>
      )}
    </Card>
  );
}
