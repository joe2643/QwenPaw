import { useEffect, useMemo, useState } from "react";
import { Alert, Button, Card, Select } from "@agentscope-ai/design";
import { useTranslation } from "react-i18next";
import api from "../../../../api";
import type { FallbackVideoModel } from "../../../../api/modules/agent";
import type { ModelInfo, ProviderInfo } from "../../../../api/types";
import { useAppMessage } from "../../../../hooks/useAppMessage";
import styles from "../index.module.less";

interface VideoModelOption {
  providerId: string;
  providerName: string;
  modelId: string;
  modelName: string;
}

/** Flatten every provider's video-capable models into a single list
 *  the Select can render as grouped options.  Filter rule: only models
 *  whose ``supports_video`` is explicitly ``true`` — ``null`` means
 *  "not yet probed", which is too weak a signal to stake the fallback
 *  path on. */
function videoCapableOptions(providers: ProviderInfo[]): VideoModelOption[] {
  const out: VideoModelOption[] = [];
  for (const p of providers) {
    const all: ModelInfo[] = [
      ...(p.models || []),
      ...(p.extra_models || []),
    ];
    for (const m of all) {
      if (m.supports_video === true) {
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

function optionValue(v: VideoModelOption): string {
  // The Select's option value uniquely identifies provider+model so
  // we can recover both halves on change without maintaining a
  // parallel map.  Separator ``::`` is safe because provider/model
  // ids are otherwise kebab/snake.
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

export function FallbackVideoModelCard() {
  const { t } = useTranslation();
  const { message } = useAppMessage();
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [options, setOptions] = useState<VideoModelOption[]>([]);
  const [current, setCurrent] = useState<FallbackVideoModel>({
    provider_id: null,
    model: null,
  });

  // Load providers + current slot in parallel.  Providers come from
  // the global registry; the slot lives on the active agent's config.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
      try {
        const [providers, slot] = await Promise.all([
          api.listProviders(),
          api.getFallbackVideoModel(),
        ]);
        if (cancelled) return;
        setOptions(videoCapableOptions(providers));
        setCurrent(slot);
      } catch (err) {
        if (cancelled) return;
        const msg =
          err instanceof Error
            ? err.message
            : "Failed to load fallback video model";
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
    // Group options by provider to keep the dropdown scannable when
    // several providers each contribute a handful of models.
    const byProvider = new Map<string, VideoModelOption[]>();
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
    const next: FallbackVideoModel = value
      ? (() => {
          const parsed = parseOptionValue(value);
          return parsed
            ? { provider_id: parsed.providerId, model: parsed.modelId }
            : { provider_id: null, model: null };
        })()
      : { provider_id: null, model: null };

    setSaving(true);
    try {
      const saved = await api.updateFallbackVideoModel(next);
      setCurrent(saved);
      message.success(
        saved.provider_id && saved.model
          ? `Fallback video model → ${saved.provider_id}/${saved.model}`
          : "Fallback video model cleared",
      );
    } catch (err) {
      const msg =
        err instanceof Error
          ? err.message
          : "Failed to save fallback video model";
      message.error(msg);
    } finally {
      setSaving(false);
    }
  };

  return (
    <Card
      className={styles.formCard}
      title={t(
        "agentConfig.fallbackVideoModelTitle",
        "Fallback video model",
      )}
    >
      <Alert
        type="info"
        showIcon
        message={t(
          "agentConfig.fallbackVideoModelHint",
          "When the active model can't see video, view_video delegates " +
            "to this model with a prompt + the VideoBlock and returns " +
            "the text description back to the primary agent.  Only " +
            "models with supports_video = true are shown.",
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
              ? "(no video-capable models configured)"
              : "(none — keep default placeholder hint)"
          }
          value={selectValue}
          onChange={(v) =>
            handleChange(typeof v === "string" ? v : undefined)
          }
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
          Add a provider whose model has <code>supports_video: true</code>{" "}
          (Gemini 2.x, GPT-4o with video, etc.) under Settings → Models.
        </div>
      )}
    </Card>
  );
}
