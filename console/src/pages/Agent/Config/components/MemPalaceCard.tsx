import { useState, useEffect, useCallback } from "react";
import { Card, Switch, InputNumber } from "@agentscope-ai/design";
import { Space, Typography, Descriptions, Spin } from "antd";
import { DatabaseOutlined } from "@ant-design/icons";
import { mempalaceApi } from "@/api/modules/mempalace";
import styles from "../index.module.less";

const { Text } = Typography;

export function MemPalaceCard() {
  const [config, setConfig] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  const loadConfig = useCallback(async () => {
    try {
      setLoading(true);
      const data = await mempalaceApi.getConfig();
      setConfig(data);
    } catch {
      setConfig({ enabled: false });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadConfig(); }, [loadConfig]);

  const update = async (patch: any) => {
    const newConfig = { ...config, ...patch };
    setConfig(newConfig);
    setSaving(true);
    try {
      const result = await mempalaceApi.updateConfig(newConfig);
      setConfig(result);
    } catch {
      loadConfig(); // revert on error
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <Card className={styles.formCard} title="MemPalace Hooks" style={{ marginTop: 16 }}>
        <Spin />
      </Card>
    );
  }

  return (
    <Card
      className={styles.formCard}
      title={<><DatabaseOutlined style={{ marginRight: 8 }} />MemPalace Hooks</>}
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
        </Descriptions.Item>
      </Descriptions>

      {config?.enabled && (
        <Descriptions column={1} size="small" style={{ marginTop: 12 }}>
          <Descriptions.Item label="Interval Save">
            <Space>
              <Switch
                size="small"
                checked={config?.interval_save?.enabled ?? false}
                onChange={(v) => update({
                  interval_save: { ...config.interval_save, enabled: v },
                })}
              />
              {config?.interval_save?.enabled && (
                <>
                  <Text type="secondary">every</Text>
                  <InputNumber
                    size="small"
                    min={5}
                    max={100}
                    value={config?.interval_save?.write_interval ?? 15}
                    onChange={(v) => update({
                      interval_save: { ...config.interval_save, write_interval: v },
                    })}
                    style={{ width: 60 }}
                  />
                  <Text type="secondary">msgs</Text>
                </>
              )}
            </Space>
          </Descriptions.Item>

          <Descriptions.Item label="PreCompact Save">
            <Space>
              <Switch
                size="small"
                checked={config?.precompact_save?.enabled ?? false}
                onChange={(v) => update({
                  precompact_save: { ...config.precompact_save, enabled: v },
                })}
              />
              {config?.precompact_save?.enabled && (
                <>
                  <Text type="secondary">at</Text>
                  <InputNumber
                    size="small"
                    min={0.5}
                    max={0.95}
                    step={0.05}
                    value={config?.precompact_save?.threshold ?? 0.75}
                    onChange={(v) => update({
                      precompact_save: { ...config.precompact_save, threshold: v },
                    })}
                    style={{ width: 65 }}
                  />
                  <Text type="secondary">context</Text>
                </>
              )}
            </Space>
          </Descriptions.Item>

          <Descriptions.Item label="Safety Save on /new">
            <Switch
              size="small"
              checked={config?.pre_reply_save ?? false}
              onChange={(v) => update({ pre_reply_save: v })}
            />
          </Descriptions.Item>

          <Descriptions.Item label="BgSave on /new">
            <Switch
              size="small"
              checked={config?.bg_save_on_new ?? false}
              onChange={(v) => update({ bg_save_on_new: v })}
            />
          </Descriptions.Item>

          <Descriptions.Item label="Session WAL">
            <Switch
              size="small"
              checked={config?.session_wal ?? false}
              onChange={(v) => update({ session_wal: v })}
            />
          </Descriptions.Item>
        </Descriptions>
      )}
    </Card>
  );
}
