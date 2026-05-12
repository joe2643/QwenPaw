import { useCallback, useEffect, useMemo, useState } from "react";
import { DatePicker } from "antd";
import { useTranslation } from "react-i18next";
import dayjs, { type Dayjs } from "dayjs";
import { useTheme } from "../../../contexts/ThemeContext";
import api from "../../../api";
import type { TokenUsageRecord } from "../../../api/types/tokenUsage";
import { useAppMessage } from "../../../hooks/useAppMessage";
import { PageHeader } from "@/components/PageHeader";
import {
  LoadingState,
  SummaryCards,
  ModelTrendChart,
  TokenTypeChart,
  DataTables,
  EmptyState,
} from "./components";
import { useDataAggregation } from "./hooks/useDataAggregation";
import { useModelTrendConfig } from "./hooks/useModelTrendConfig";
import { useTokenTypeConfig } from "./hooks/useTokenTypeConfig";
import styles from "./index.module.less";

function TokenUsagePage() {
  const { t } = useTranslation();
  const { message } = useAppMessage();
  const { isDark } = useTheme();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [records, setRecords] = useState<TokenUsageRecord[]>([]);
  const [startDate, setStartDate] = useState<Dayjs>(
    dayjs().subtract(30, "day"),
  );
  const [endDate, setEndDate] = useState<Dayjs>(dayjs());

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(false);
    try {
      const detailsData = await api.getTokenUsageDetails({
        start_date: startDate.format("YYYY-MM-DD"),
        end_date: endDate.format("YYYY-MM-DD"),
      });
      setRecords(detailsData);
    } catch (err) {
      console.error("Failed to load token usage:", err);
      message.error(t("tokenUsage.loadFailed"));
      setRecords([]);
      setError(true);
    } finally {
      setLoading(false);
    }
  }, [startDate, endDate, message, t]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleDateChange = (dates: [Dayjs | null, Dayjs | null] | null) => {
    if (!dates || !dates[0] || !dates[1]) return;
    setStartDate(dates[0]);
    setEndDate(dates[1]);
  };

  const aggregatedData = useDataAggregation(records);

  const modelTrendConfig = useModelTrendConfig({
    byDateModel: aggregatedData?.by_date_model ?? null,
    startDate,
    endDate,
    isDark,
  });

  const tokenTypeConfig = useTokenTypeConfig({
    byDate: aggregatedData?.by_date ?? null,
    startDate,
    endDate,
    isDark,
  });

  const byModelData = useMemo(() => {
    if (!aggregatedData?.by_model) return [];
    return Object.entries(aggregatedData.by_model).map(([key, stats]) => ({
      key,
      model: key,
      prompt_tokens: stats.prompt_tokens,
      completion_tokens: stats.completion_tokens,
      call_count: stats.call_count,
    }));
  }, [aggregatedData?.by_model]);

  const byDateData = useMemo(() => {
    if (!aggregatedData?.by_date) return [];
    return Object.entries(aggregatedData.by_date)
      .map(([date, stats]) => ({
        key: date,
        date,
        prompt_tokens: stats.prompt_tokens,
        completion_tokens: stats.completion_tokens,
        call_count: stats.call_count,
      }))
      .sort((a, b) => b.date.localeCompare(a.date));
  }, [aggregatedData?.by_date]);

  const byModelColumns: ColumnsType<ByModelRow> = useMemo(
    () => [
      {
        title: t("tokenUsage.provider"),
        dataIndex: "provider_id",
        key: "provider_id",
        render: (v: string) => v ?? "",
      },
      {
        title: t("tokenUsage.model"),
        dataIndex: "model",
        key: "model",
        render: (v: string, r) => v ?? r.key,
      },
      {
        title: t("tokenUsage.promptTokens"),
        dataIndex: "prompt_tokens",
        key: "prompt_tokens",
        render: (n: number) => formatCompact(n),
      },
      {
        title: t("tokenUsage.completionTokens"),
        dataIndex: "completion_tokens",
        key: "completion_tokens",
        render: (n: number) => formatCompact(n),
      },
      {
        title: t("tokenUsage.cacheReadTokens"),
        dataIndex: "cache_read_tokens",
        key: "cache_read_tokens",
        render: (n: number | undefined) => formatCompact(n ?? 0),
      },
      {
        title: t("tokenUsage.cacheCreationTokens"),
        dataIndex: "cache_creation_tokens",
        key: "cache_creation_tokens",
        render: (n: number | undefined) => formatCompact(n ?? 0),
      },
      {
        title: t("tokenUsage.totalCalls"),
        dataIndex: "call_count",
        key: "call_count",
        render: (n: number) => formatCompact(n),
      },
    ],
    [t],
  );

  const byDateColumns: ColumnsType<ByDateRow> = useMemo(
    () => [
      { title: t("tokenUsage.date"), dataIndex: "date", key: "date" },
      {
        title: t("tokenUsage.promptTokens"),
        dataIndex: "prompt_tokens",
        key: "prompt_tokens",
        render: (n: number) => formatCompact(n),
      },
      {
        title: t("tokenUsage.completionTokens"),
        dataIndex: "completion_tokens",
        key: "completion_tokens",
        render: (n: number) => formatCompact(n),
      },
      {
        title: t("tokenUsage.cacheReadTokens"),
        dataIndex: "cache_read_tokens",
        key: "cache_read_tokens",
        render: (n: number | undefined) => formatCompact(n ?? 0),
      },
      {
        title: t("tokenUsage.cacheCreationTokens"),
        dataIndex: "cache_creation_tokens",
        key: "cache_creation_tokens",
        render: (n: number | undefined) => formatCompact(n ?? 0),
      },
      {
        title: t("tokenUsage.totalCalls"),
        dataIndex: "call_count",
        key: "call_count",
        render: (n: number) => formatCompact(n),
      },
    ],
    [t],
  );

  const pageHeader = (
    <PageHeader parent={t("nav.settings")} current={t("tokenUsage.title")} />
  );

  if (loading) {
    return (
      <div className={styles.container}>
        {pageHeader}
        <LoadingState message={t("common.loading", "Loading...")} />
      </div>
    );
  }

  if (error && records.length === 0) {
    return (
      <div className={styles.container}>
        {pageHeader}
        <LoadingState
          message={t("tokenUsage.loadFailed")}
          error
          onRetry={fetchData}
        />
      </div>
    );
  }

  return (
    <div className={styles.container}>
      {pageHeader}

      <div className={styles.content}>
        <div className={styles.toolbar}>
          <DatePicker.RangePicker
            value={[startDate, endDate]}
            onChange={handleDateChange}
            disabledDate={(current) =>
              current && current.isAfter(dayjs(), "day")
            }
          />
        </div>

        {aggregatedData && (
          <SummaryCards
            totalCalls={aggregatedData.total_calls}
            totalPromptTokens={aggregatedData.total_prompt_tokens}
            totalCompletionTokens={aggregatedData.total_completion_tokens}
            totalTokens={
              aggregatedData.total_prompt_tokens +
              aggregatedData.total_completion_tokens
            }
          />
        )}

        <div className={styles.trendRow}>
          <ModelTrendChart chartConfig={modelTrendConfig} />
          <TokenTypeChart chartConfig={tokenTypeConfig} />
        </div>

        {byModelData.length === 0 && byDateData.length === 0 ? (
          <EmptyState message={t("tokenUsage.noData")} />
        ) : (
          <>
            <div className={styles.filters}>
              <DatePicker.RangePicker
                value={[startDate, endDate]}
                onChange={handleDateChange}
                className={styles.datePicker}
              />
              <Button type="primary" onClick={fetchData} loading={loading}>
                {t("tokenUsage.refresh")}
              </Button>
            </div>

            {data && data.total_calls > 0 ? (
              <>
                <div className={styles.summaryCards}>
                  <Card className={styles.card}>
                    <div className={styles.cardValue}>
                      {formatCompact(data.total_prompt_tokens)}
                    </div>
                    <div className={styles.cardLabel}>
                      {t("tokenUsage.promptTokens")}
                    </div>
                  </Card>
                  <Card className={styles.card}>
                    <div className={styles.cardValue}>
                      {formatCompact(data.total_completion_tokens)}
                    </div>
                    <div className={styles.cardLabel}>
                      {t("tokenUsage.completionTokens")}
                    </div>
                  </Card>
                  <Card className={styles.card}>
                    <div className={styles.cardValue}>
                      {formatCompact(data.total_cache_read_tokens ?? 0)}
                    </div>
                    <div className={styles.cardLabel}>
                      {t("tokenUsage.cacheReadTokens")}
                    </div>
                  </Card>
                  <Card className={styles.card}>
                    <div className={styles.cardValue}>
                      {formatCompact(data.total_cache_creation_tokens ?? 0)}
                    </div>
                    <div className={styles.cardLabel}>
                      {t("tokenUsage.cacheCreationTokens")}
                    </div>
                  </Card>
                </div>

                {byModelDataSource.length > 0 && (
                  <Card
                    className={styles.tableCard}
                    title={t("tokenUsage.byModel")}
                    bodyStyle={{ padding: 0 }}
                  >
                    <Table<ByModelRow>
                      columns={byModelColumns}
                      dataSource={byModelDataSource}
                      rowKey="key"
                      pagination={false}
                    />
                  </Card>
                )}

                {byDateDataSource.length > 0 && (
                  <Card
                    className={styles.tableCard}
                    title={t("tokenUsage.byDate")}
                    bodyStyle={{ padding: 0 }}
                  >
                    <Table<ByDateRow>
                      columns={byDateColumns}
                      dataSource={byDateDataSource}
                      rowKey="key"
                      pagination={false}
                    />
                  </Card>
                )}
              </>
            ) : (
              <EmptyState message={t("tokenUsage.noData")} />
            )}
          </>
        )}
      </div>
    </div>
  );
}

export default TokenUsagePage;
