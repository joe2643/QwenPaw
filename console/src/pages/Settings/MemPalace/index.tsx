import { useState, useEffect, useCallback, useRef } from "react";
import {
  Card,
  Table,
  Switch,
  Tag,
  Button,
  Empty,
  Input,
} from "@agentscope-ai/design";
import {
  Tabs,
  Tree,
  Space,
  Statistic,
  Popconfirm,
  Typography,
} from "antd";
import {
  ReloadOutlined,
  DeleteOutlined,
  DatabaseOutlined,
  ApartmentOutlined,
  NodeIndexOutlined,
  FileTextOutlined,
} from "@ant-design/icons";
import type { DataNode } from "antd/es/tree";
import { PageHeader } from "@/components/PageHeader";
import { useTheme } from "@/contexts/ThemeContext";
import { useMemPalace } from "./useMemPalace";
import styles from "./index.module.less";

const { Text, Title } = Typography;
const { TabPane } = Tabs;

// ── Helpers ──────────────────────────────────────────────────────────────

function hallColor(hall: string): string {
  const map: Record<string, string> = {
    hall_facts: "blue",
    hall_events: "purple",
    hall_discoveries: "gold",
    hall_preferences: "green",
    hall_advice: "orange",
    hall_diary: "cyan",
  };
  return map[hall] || "default";
}

// ── Overview Tab ─────────────────────────────────────────────────────────

function OverviewTab({
  status,
  kgStats,
  onRefresh,
}: {
  status: any;
  kgStats: any;
  onRefresh: () => void;
}) {

  return (
    <Space direction="vertical" size="middle" style={{ width: "100%" }}>
      {/* Stats row */}
      <div className={styles.statsRow}>
        <Card size="small" className={styles.statCard}>
          <Statistic title="Total Drawers" value={status?.total_drawers ?? "-"} prefix={<DatabaseOutlined />} />
        </Card>
        <Card size="small" className={styles.statCard}>
          <Statistic title="Wings" value={status?.wings ? Object.keys(status.wings).length : "-"} prefix={<ApartmentOutlined />} />
        </Card>
        <Card size="small" className={styles.statCard}>
          <Statistic title="KG Entities" value={kgStats?.entity_count ?? status?.kg?.entity_count ?? "-"} prefix={<NodeIndexOutlined />} />
        </Card>
        <Card size="small" className={styles.statCard}>
          <Statistic title="KG Triples" value={kgStats?.triple_count ?? status?.kg?.triple_count ?? "-"} prefix={<FileTextOutlined />} />
        </Card>
      </div>


    </Space>
  );
}

// ── Structure Tab ────────────────────────────────────────────────────────

function StructureTab({
  wings,
  drawers,
  drawerTotal,
  loading,
  onSelectRoom,
  onDeleteDrawer,
  onRefreshWings,
}: {
  wings: any[];
  drawers: any[];
  drawerTotal: number;
  loading: boolean;
  onSelectRoom: (wing: string, room: string, offset?: number, limit?: number) => void;
  onDeleteDrawer: (id: string) => void;
  onRefreshWings: () => void;
}) {
  const { isDark } = useTheme();
  const [selectedRoom, setSelectedRoom] = useState<{ wing: string; room: string } | null>(null);
  const [page, setPage] = useState(1);

  const treeData: DataNode[] = wings.map((wing) => ({
    key: `wing:${wing.name}`,
    title: (
      <span>
        <ApartmentOutlined style={{ marginRight: 6 }} />
        <Text strong>{wing.name}</Text>
        <Text type="secondary" style={{ marginLeft: 4 }}>
          ({(wing.rooms || []).reduce((s: number, r: any) => s + (r.drawer_count ?? r.count ?? 0), 0)})
        </Text>
      </span>
    ),
    children: (wing.rooms || []).map((room: any) => ({
      key: `room:${wing.name}/${room.name}`,
      title: (
        <span>
          {room.name} <Tag>{room.drawer_count ?? room.count ?? "?"}</Tag>
        </span>
      ),
      isLeaf: true,
    })),
  }));

  const handleSelect = (_: any, info: any) => {
    const key = info.node.key as string;
    if (key.startsWith("room:")) {
      const path = key.replace("room:", "");
      const [wing, room] = path.split("/");
      setSelectedRoom({ wing, room });
      setPage(1);
      onSelectRoom(wing, room, 0, 50);
    }
  };

  const drawerColumns = [
    {
      title: "ID",
      dataIndex: "id",
      key: "id",
      width: 220,
      ellipsis: true,
      render: (id: string) => <Text copyable={{ text: id }} style={{ fontSize: 11 }}>{id}</Text>,
    },
    {
      title: "Hall",
      key: "hall",
      width: 160,
      render: (_: any, record: any) => {
        const hall = record.hall || record.metadata?.hall;
        return hall ? <Tag color={hallColor(hall)}>{hall}</Tag> : <Text type="secondary">—</Text>;
      },
    },
    {
      title: "Content",
      dataIndex: "content_preview",
      key: "content",
      ellipsis: true,
      render: (text: string, record: any) => (
        <Text style={{ fontSize: 12 }}>{(text || record.content || "").substring(0, 120)}</Text>
      ),
    },
    {
      title: "Date",
      key: "filed_at",
      width: 100,
      render: (_: any, record: any) => {
        const ts = record.filed_at || record.metadata?.filed_at || record.metadata?.date;
        return ts ? ts.substring(0, 10) : "—";
      },
    },
    {
      title: "",
      key: "actions",
      width: 50,
      render: (_: any, record: any) => (
        <Popconfirm
          title="Delete?"
          onConfirm={() => {
            onDeleteDrawer(record.id);
            if (selectedRoom) setTimeout(() => onSelectRoom(selectedRoom.wing, selectedRoom.room), 300);
          }}
        >
          <Button type="text" danger size="small" icon={<DeleteOutlined />} />
        </Popconfirm>
      ),
    },
  ];

  return (
    <div className={styles.structureLayout}>
      <Card
        size="small"
        title="Wings & Rooms"
        extra={<Button size="small" icon={<ReloadOutlined />} onClick={onRefreshWings} />}
        className={styles.treePanel}
      >
        {treeData.length === 0 ? (
          <Empty description="No wings found" />
        ) : (
          <Tree treeData={treeData} defaultExpandAll onSelect={handleSelect}
            selectedKeys={selectedRoom ? [`room:${selectedRoom.wing}/${selectedRoom.room}`] : []}
          />
        )}
      </Card>

      <div className={styles.drawerPanel}>
        {selectedRoom ? (
          <Card
            size="small"
            title={<><Tag color="blue">{selectedRoom.wing}</Tag><Tag color="green">{selectedRoom.room}</Tag>
              <Text type="secondary" style={{ marginLeft: 8 }}>{drawerTotal} drawer{drawerTotal !== 1 ? "s" : ""}</Text></>}
          >
            <Table
              dataSource={drawers}
              columns={drawerColumns}
              rowKey="id"
              loading={loading}
              size="small"
              pagination={{
                current: page, pageSize: 50, total: drawerTotal,
                onChange: (p) => { setPage(p); onSelectRoom(selectedRoom.wing, selectedRoom.room, (p - 1) * 50, 50); },
                showTotal: (total) => `Total ${total}`,
              }}
            />
          </Card>
        ) : (
          <Card size="small"><Empty description="Select a room to view drawers" /></Card>
        )}
      </div>
    </div>
  );
}

// ── Knowledge Graph Tab ──────────────────────────────────────────────────

function KnowledgeGraphTab({
  kgEntities, kgTriples, kgEntityTotal, kgTripleTotal, kgStats,
  onLoadEntities, onLoadTriples, onLoadKgStats,
}: {
  kgEntities: any[]; kgTriples: any[]; kgEntityTotal: number; kgTripleTotal: number; kgStats: any;
  onLoadEntities: (offset?: number, limit?: number) => void;
  onLoadTriples: (offset?: number, limit?: number) => void;
  onLoadKgStats: () => void;
}) {
  const [entityPage, setEntityPage] = useState(1);
  const [triplePage, setTriplePage] = useState(1);

  useEffect(() => { onLoadEntities(0, 50); onLoadTriples(0, 50); }, []);

  return (
    <Space direction="vertical" size="middle" style={{ width: "100%" }}>
      {kgStats && (
        <div className={styles.statsRow}>
          <Card size="small" className={styles.statCard}>
            <Statistic title="Entities" value={kgStats.entity_count ?? "-"} />
          </Card>
          <Card size="small" className={styles.statCard}>
            <Statistic title="Triples" value={kgStats.triple_count ?? "-"} />
          </Card>
        </div>
      )}

      <Card title="Entities" size="small"
        extra={<Button size="small" icon={<ReloadOutlined />} onClick={() => { onLoadEntities(0, 50); onLoadKgStats(); }}>Refresh</Button>}
      >
        <Table
          dataSource={kgEntities}
          columns={[
            { title: "Name", dataIndex: "name", key: "name", ellipsis: true },
            { title: "Type", dataIndex: "type", key: "type", width: 120, render: (t: string) => <Tag>{t || "auto"}</Tag> },
            { title: "Properties", dataIndex: "properties", key: "properties", ellipsis: true, render: (p: any) => p ? JSON.stringify(p).substring(0, 80) : "-" },
          ]}
          rowKey={(r) => r.id ?? r.name ?? JSON.stringify(r)}
          size="small"
          pagination={{
            current: entityPage, pageSize: 50, total: kgEntityTotal,
            onChange: (p) => { setEntityPage(p); onLoadEntities((p - 1) * 50, 50); },
            showTotal: (total) => `Total ${total}`,
          }}
        />
      </Card>

      <Card title="Triples" size="small"
        extra={<Button size="small" icon={<ReloadOutlined />} onClick={() => onLoadTriples(0, 50)}>Refresh</Button>}
      >
        <Table
          dataSource={kgTriples}
          columns={[
            { title: "Subject", dataIndex: "subject", key: "subject", ellipsis: true },
            { title: "Predicate", dataIndex: "predicate", key: "predicate", width: 160, render: (p: string) => <Tag color="purple">{p}</Tag> },
            { title: "Object", dataIndex: "object", key: "object", ellipsis: true },
            { title: "Valid From", dataIndex: "valid_from", key: "valid_from", width: 100 },
            { title: "Source", dataIndex: "source_closet", key: "source", width: 120, ellipsis: true },
          ]}
          rowKey={(r) => r.id ?? `${r.subject}-${r.predicate}-${r.object}`}
          size="small"
          pagination={{
            current: triplePage, pageSize: 50, total: kgTripleTotal,
            onChange: (p) => { setTriplePage(p); onLoadTriples((p - 1) * 50, 50); },
            showTotal: (total) => `Total ${total}`,
          }}
        />
      </Card>
    </Space>
  );
}

// ── Hooks Tab ────────────────────────────────────────────────────────────

function HooksTab({
  hookLog,
  onLoadLog,
}: {
  hookLog: string;
  onLoadLog: (lines?: number) => void;
}) {
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(false);

  useEffect(() => { onLoadLog(200); }, []);

  useEffect(() => {
    if (autoRefresh) intervalRef.current = setInterval(() => onLoadLog(200), 5000);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [autoRefresh, onLoadLog]);

  // Parse log lines for colored display
  const logLines = (hookLog || "").split("\n").filter(Boolean);

  return (
    <Card
      title="Hook Execution Log"
      size="small"
      extra={
        <Space>
          <Text type="secondary">Auto-refresh</Text>
          <Switch size="small" checked={autoRefresh} onChange={setAutoRefresh} />
          <Button size="small" icon={<ReloadOutlined />} onClick={() => onLoadLog(200)}>Refresh</Button>
        </Space>
      }
    >
      <div className={styles.hookLogContainer}>
        {logLines.length === 0 ? (
          <Text type="secondary">(no logs)</Text>
        ) : (
          logLines.map((line, i) => {
            let cls = styles.logLineDefault;
            if (line.includes("ERROR") || line.includes("FAILED")) cls = styles.logLineError;
            else if (line.includes("BgSave") || line.includes("Diary")) cls = styles.logLineSave;
            else if (line.includes("Interval") || line.includes("PreCompact")) cls = styles.logLineInterval;
            else if (line.includes("PreReply")) cls = styles.logLinePreReply;
            return <div key={i} className={cls} style={{ whiteSpace: "pre-wrap" }}>{line}</div>;
          })
        )}
      </div>
    </Card>
  );
}

// ── Main Page ────────────────────────────────────────────────────────────

function MemPalacePage() {
  const {
    status, wings, drawers, drawerTotal, config, hookLog,
    kgStats, kgEntities, kgTriples, kgEntityTotal, kgTripleTotal,
    loading,
    loadStatus, loadWings, loadDrawers, loadConfig, updateConfig,
    loadHookLog, loadKgStats, loadKgEntities, loadKgTriples, deleteDrawer,
  } = useMemPalace();

  const handleRefreshAll = () => {
    loadStatus(); loadWings(); loadConfig(); loadKgStats();
  };

  return (
    <div className={styles.mempalacePage}>
      <PageHeader
        items={[{ title: "Settings" }, { title: "MemPalace" }]}
        extra={<Button icon={<ReloadOutlined />} onClick={handleRefreshAll}>Refresh</Button>}
      />
      <div className={styles.content}>
        <Tabs defaultActiveKey="overview">
          <TabPane tab="Overview" key="overview">
            <OverviewTab status={status} config={config} kgStats={kgStats} onRefresh={handleRefreshAll} onConfigChange={updateConfig} />
          </TabPane>
          <TabPane tab="Structure" key="structure">
            <StructureTab wings={wings} drawers={drawers} drawerTotal={drawerTotal} loading={loading}
              onSelectRoom={loadDrawers} onDeleteDrawer={deleteDrawer} onRefreshWings={loadWings} />
          </TabPane>
          <TabPane tab="Knowledge Graph" key="kg">
            <KnowledgeGraphTab kgEntities={kgEntities} kgTriples={kgTriples}
              kgEntityTotal={kgEntityTotal} kgTripleTotal={kgTripleTotal} kgStats={kgStats}
              onLoadEntities={loadKgEntities} onLoadTriples={loadKgTriples} onLoadKgStats={loadKgStats} />
          </TabPane>
          <TabPane tab="Hooks" key="hooks">
            <HooksTab hookLog={hookLog} onLoadLog={loadHookLog} />
          </TabPane>
        </Tabs>
      </div>
    </div>
  );
}

export default MemPalacePage;
