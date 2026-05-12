import { useEffect, useMemo, useState } from "react";
import { Alert, Button, Card, Select } from "@agentscope-ai/design";
import { useTranslation } from "react-i18next";
import api from "../../../../api";
import type { FallbackImageModel } from "../../../../api/modules/agent";
import type { ModelInfo, ProviderInfo } from "../../../../api/types";
import { useAppMessage } from "../../../../hooks/useAppMessage";
import styles from "../index.module.less";

interface ImageModelOption {
  providerId: string;
  providerName: string;
  modelId: string;
  modelName: string;
}

/** Flatten every provider's image-capable models into a single list.
 *  Filter rule: only models whose ``supports_image`` (or the legacy
 *  combined ``supports_multimodal``) is explicitly ``true`` — ``null``
 *  means "not yet probed", which is too weak a signal to stake the
 *  fallback path on. */
function imageCapableOptions(providers: ProviderInfo[]): ImageModelOption[] {
  const out: ImageModelOption[] = [];
  for (const p of providers) {
    const all: ModelInfo[] = [...(p.models || []), ...(p.extra_models || [])];
    for (const m of all) {
      if (m.supports_image === true || m.supports_multimodal === true) {
        out.push({
          providerId: p.id,
          providerName: p.name,
          modelId: m.id,
          modelName: m.name || m.id,
        });
      }
    }
  }
  return out;
}

function optionValue(v: ImageModelOption): string {
  return `${v.providerId}::${v.modelId}`;
}

function parseOptionValue(
  value: string,
): { providerId: string; modelId: string } | null {
  const idx = value.indexOf("::");
  if (idx < 0) return null;
  return {
    providerId: value.slice(0, idx),
    modelId: value.slice(idx + 2),
  };
}

export function FallbackImageModelCard() {
  const { t } = useTranslation();
  const { message } = useAppMessage();
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [options, setOptions] = useState<ImageModelOption[]>([]);
  const [current, setCurrent] = useState<FallbackImageModel>({
    provider_id: null,
    model: null,
  });

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
      try {
        const [providers, slot] = await Promise.all([
          api.listProviders(),
          api.getFallbackImageModel(),
        ]);
        if (cancelled) return;
        setOptions(imageCapableOptions(providers));
        setCurrent(slot);
      } catch (err) {
        if (cancelled) return;
        const msg =
          err instanceof Error
            ? err.message
            : "Failed to load fallback image model";
        message.error(msg);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [message]);

  const selectValue = useMemo(() => {
    if (!current.provider_id || !current.model) return undefined;
    return `${current.provider_id}::${current.model}`;
  }, [current]);

  const selectOptions = useMemo(() => {
    const byProvider = new Map<string, ImageModelOption[]>();
    for (const opt of options) {
      const list = byProvider.get(opt.providerName) || [];
      list.push(opt);
      byProvider.set(opt.providerName, list);
    }
    return Array.from(byProvider.entries()).map(([label, items]) => ({
      label,
      options: items.map((o) => ({
        label: o.modelName,
        value: optionValue(o),
      })),
    }));
  }, [options]);

  const handleChange = async (value: string | undefined) => {
    const next: FallbackImageModel = value
      ? (() => {
          const parsed = parseOptionValue(value);
          return parsed
            ? { provider_id: parsed.providerId, model: parsed.modelId }
            : { provider_id: null, model: null };
        })()
      : { provider_id: null, model: null };

    setSaving(true);
    try {
      const saved = await api.updateFallbackImageModel(next);
      setCurrent(saved);
      message.success(
        saved.provider_id && saved.model
          ? `Fallback image model → ${saved.provider_id}/${saved.model}`
          : "Fallback image model cleared",
      );
    } catch (err) {
      const msg =
        err instanceof Error
          ? err.message
          : "Failed to save fallback image model";
      message.error(msg);
    } finally {
      setSaving(false);
    }
  };

  return (
    <Card
      className={styles.formCard}
      title={t("agentConfig.fallbackImageModelTitle", "Fallback image model")}
    >
      <Alert
        type="info"
        showIcon
        message={t(
          "agentConfig.fallbackImageModelHint",
          "When the active model can't see image, view_image delegates " +
            "to this model with a prompt + the ImageBlock and returns " +
            "the text description back to the primary agent.  Only " +
            "models with supports_image = true (or the combined " +
            "supports_multimodal = true) are shown.",
        )}
        style={{ marginBottom: 16 }}
      />

      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <Select
          allowClear
          loading={loading || saving}
          disabled={loading}
          style={{ flex: 1, maxWidth: 420 }}
          placeholder={
            options.length === 0
              ? "(no image-capable models configured)"
              : "(none — keep default placeholder hint)"
          }
          value={selectValue}
          onChange={(v) => handleChange(typeof v === "string" ? v : undefined)}
          options={selectOptions}
        />
        {current.provider_id && current.model && (
          <Button
            size="small"
            danger
            disabled={saving}
            onClick={() => handleChange(undefined)}
          >
            Clear
          </Button>
        )}
      </div>
      {options.length === 0 && !loading && (
        <div style={{ marginTop: 10, fontSize: 12, opacity: 0.65 }}>
          Add a provider whose model has <code>supports_image: true</code>{" "}
          (Gemini, GPT-4o, Claude 3+, Qwen-VL, etc.) under Settings → Models.
        </div>
      )}
    </Card>
  );
}
