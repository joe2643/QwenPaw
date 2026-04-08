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
import { useMemPalace } from "./useMemPalace";

const { Text } = Typography;
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
  config,
  kgStats,
  onRefresh,
  onConfigChange,
}: {
  status: any;
  config: any;
  kgStats: any;
  onRefresh: () => void;
  onConfigChange: (data: any) => void;
}) {
  return (
    <div>
      <Space direction="vertical" size="middle" style={{ width: "100%" }}>
        {/* Stats row */}
        <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
          <Card size="small" style={{ flex: 1, minWidth: 180 }}>
            <Statistic
              title="Total Drawers"
              value={status?.total_drawers ?? "-"}
              prefix={<DatabaseOutlined />}
            />
          </Card>
          <Card size="small" style={{ flex: 1, minWidth: 180 }}>
            <Statistic
              title="Wings"
              value={status?.wing_count ?? "-"}
              prefix={<ApartmentOutlined />}
            />
          </Card>
          <Card size="small" style={{ flex: 1, minWidth: 180 }}>
            <Statistic
              title="KG Entities"
              value={kgStats?.entity_count ?? status?.kg_entities ?? "-"}
              prefix={<NodeIndexOutlined />}
            />
          </Card>
          <Card size="small" style={{ flex: 1, minWidth: 180 }}>
            <Statistic
              title="KG Triples"
              value={kgStats?.triple_count ?? status?.kg_triples ?? "-"}
              prefix={<FileTextOutlined />}
            />
          </Card>
        </div>

        {/* Config */}
        {config && (
          <Card title="Configuration" size="small" extra={<Button size="small" icon={<ReloadOutlined />} onClick={onRefresh}>Refresh</Button>}>
            <Space direction="vertical" size="small" style={{ width: "100%" }}>
              {typeof config.enabled !== "undefined" && (
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <Text>MemPalace Enabled</Text>
                  <Switch
                    checked={config.enabled}
                    onChange={(checked) => onConfigChange({ ...config, enabled: checked })}
                  />
                </div>
              )}
              {typeof config.kg_enabled !== "undefined" && (
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <Text>Knowledge Graph</Text>
                  <Switch
                    checked={config.kg_enabled}
                    onChange={(checked) => onConfigChange({ ...config, kg_enabled: checked })}
                  />
                </div>
              )}
              {typeof config.hooks_enabled !== "undefined" && (
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <Text>Hooks</Text>
                  <Switch
                    checked={config.hooks_enabled}
                    onChange={(checked) => onConfigChange({ ...config, hooks_enabled: checked })}
                  />
                </div>
              )}
              {config.palace_path && (
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <Text>Palace Path</Text>
                  <Text type="secondary" copyable>{config.palace_path}</Text>
                </div>
              )}
            </Space>
          </Card>
        )}
      </Space>
    </div>
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
  onSelectRoom: (wing: string, room: string) => void;
  onDeleteDrawer: (id: string) => void;
  onRefreshWings: () => void;
}) {
  const [selectedRoom, setSelectedRoom] = useState<{ wing: string; room: string } | null>(null);
  const [page, setPage] = useState(1);

  const treeData: DataNode[] = wings.map((wing) => ({
    key: `wing:${wing.name}`,
    title: (
      <span>
        <ApartmentOutlined style={{ marginRight: 6 }} />
        {wing.name}
      </span>
    ),
    children: (wing.rooms || []).map((room: any) => ({
      key: `room:${wing.name}/${room.name}`,
      title: (
        <span>
          {room.name} <Tag size="small">{room.drawer_count ?? "?"}</Tag>
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
      onSelectRoom(wing, room);
    }
  };

  const drawerColumns = [
    {
      title: "ID",
      dataIndex: "id",
      key: "id",
      width: 220,
      ellipsis: true,
    },
    {
      title: "Hall",
      dataIndex: "hall",
      key: "hall",
      width: 160,
      render: (hall: string) => <Tag color={hallColor(hall)}>{hall}</Tag>,
    },
    {
      title: "Summary",
      dataIndex: "summary",
      key: "summary",
      ellipsis: true,
      render: (text: string, record: any) => text || record.content?.substring(0, 100) || "-",
    },
    {
      title: "Created",
      dataIndex: "created_at",
      key: "created_at",
      width: 180,
      render: (ts: string) => (ts ? new Date(ts).toLocaleString() : "-"),
    },
    {
      title: "Actions",
      key: "actions",
      width: 80,
      render: (_: any, record: any) => (
        <Popconfirm
          title="Delete this drawer?"
          onConfirm={() => {
            onDeleteDrawer(record.id);
            if (selectedRoom) {
              setTimeout(() => onSelectRoom(selectedRoom.wing, selectedRoom.room), 300);
            }
          }}
        >
          <Button type="text" danger size="small" icon={<DeleteOutlined />} />
        </Popconfirm>
      ),
    },
  ];

  return (
    <div style={{ display: "flex", gap: 16, minHeight: 400 }}>
      {/* Left: tree */}
      <Card
        size="small"
        title="Wings & Rooms"
        extra={<Button size="small" icon={<ReloadOutlined />} onClick={onRefreshWings} />}
        style={{ width: 280, flexShrink: 0 }}
      >
        {treeData.length === 0 ? (
          <Empty description="No wings found" />
        ) : (
          <Tree
            treeData={treeData}
            defaultExpandAll
            onSelect={handleSelect}
            selectedKeys={selectedRoom ? [`room:${selectedRoom.wing}/${selectedRoom.room}`] : []}
          />
        )}
      </Card>

      {/* Right: drawer table */}
      <div style={{ flex: 1 }}>
        {selectedRoom ? (
          <Card
            size="small"
            title={
              <span>
                <Tag color="blue">{selectedRoom.wing}</Tag>
                <Tag color="green">{selectedRoom.room}</Tag>
                <Text type="secondary" style={{ marginLeft: 8 }}>
                  {drawerTotal} drawer{drawerTotal !== 1 ? "s" : ""}
                </Text>
              </span>
            }
          >
            <Table
              dataSource={drawers}
              columns={drawerColumns}
              rowKey="id"
              loading={loading}
              size="small"
              pagination={{
                current: page,
                pageSize: 50,
                total: drawerTotal,
                onChange: (p) => {
                  setPage(p);
                  onSelectRoom(selectedRoom.wing, selectedRoom.room);
                },
                showTotal: (total) => `Total ${total}`,
              }}
            />
          </Card>
        ) : (
          <Card size="small">
            <Empty description="Select a room to view drawers" />
          </Card>
        )}
      </div>
    </div>
  );
}

// ── Knowledge Graph Tab ──────────────────────────────────────────────────

function KnowledgeGraphTab({
  kgEntities,
  kgTriples,
  kgEntityTotal,
  kgTripleTotal,
  kgStats,
  onLoadEntities,
  onLoadTriples,
  onLoadKgStats,
}: {
  kgEntities: any[];
  kgTriples: any[];
  kgEntityTotal: number;
  kgTripleTotal: number;
  kgStats: any;
  onLoadEntities: (offset?: number, limit?: number) => void;
  onLoadTriples: (offset?: number, limit?: number) => void;
  onLoadKgStats: () => void;
}) {
  const [entityPage, setEntityPage] = useState(1);
  const [triplePage, setTriplePage] = useState(1);

  useEffect(() => {
    onLoadEntities(0, 50);
    onLoadTriples(0, 50);
  }, []);

  const entityColumns = [
    { title: "Name", dataIndex: "name", key: "name", ellipsis: true },
    { title: "Type", dataIndex: "type", key: "type", width: 140, render: (t: string) => <Tag>{t || "unknown"}</Tag> },
    { title: "Properties", dataIndex: "properties", key: "properties", ellipsis: true, render: (p: any) => (p ? JSON.stringify(p).substring(0, 100) : "-") },
  ];

  const tripleColumns = [
    { title: "Subject", dataIndex: "subject", key: "subject", ellipsis: true },
    { title: "Predicate", dataIndex: "predicate", key: "predicate", width: 180, render: (p: string) => <Tag color="purple">{p}</Tag> },
    { title: "Object", dataIndex: "object", key: "object", ellipsis: true },
    { title: "Source", dataIndex: "source", key: "source", width: 140, ellipsis: true },
  ];

  return (
    <Space direction="vertical" size="middle" style={{ width: "100%" }}>
      {kgStats && (
        <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
          <Card size="small" style={{ flex: 1, minWidth: 160 }}>
            <Statistic title="Entities" value={kgStats.entity_count ?? "-"} />
          </Card>
          <Card size="small" style={{ flex: 1, minWidth: 160 }}>
            <Statistic title="Triples" value={kgStats.triple_count ?? "-"} />
          </Card>
          <Card size="small" style={{ flex: 1, minWidth: 160 }}>
            <Statistic title="Communities" value={kgStats.community_count ?? "-"} />
          </Card>
        </div>
      )}

      <Card
        title="Entities"
        size="small"
        extra={
          <Button size="small" icon={<ReloadOutlined />} onClick={() => { onLoadEntities(0, 50); onLoadKgStats(); }}>
            Refresh
          </Button>
        }
      >
        <Table
          dataSource={kgEntities}
          columns={entityColumns}
          rowKey={(r) => r.id ?? r.name ?? JSON.stringify(r)}
          size="small"
          pagination={{
            current: entityPage,
            pageSize: 50,
            total: kgEntityTotal,
            onChange: (p) => {
              setEntityPage(p);
              onLoadEntities((p - 1) * 50, 50);
            },
            showTotal: (total) => `Total ${total}`,
          }}
        />
      </Card>

      <Card
        title="Triples"
        size="small"
        extra={
          <Button size="small" icon={<ReloadOutlined />} onClick={() => onLoadTriples(0, 50)}>
            Refresh
          </Button>
        }
      >
        <Table
          dataSource={kgTriples}
          columns={tripleColumns}
          rowKey={(r) => r.id ?? `${r.subject}-${r.predicate}-${r.object}`}
          size="small"
          pagination={{
            current: triplePage,
            pageSize: 50,
            total: kgTripleTotal,
            onChange: (p) => {
              setTriplePage(p);
              onLoadTriples((p - 1) * 50, 50);
            },
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

  useEffect(() => {
    onLoadLog(200);
  }, []);

  useEffect(() => {
    if (autoRefresh) {
      intervalRef.current = setInterval(() => onLoadLog(200), 5000);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [autoRefresh, onLoadLog]);

  return (
    <Card
      title="Hook Execution Log"
      size="small"
      extra={
        <Space>
          <Text type="secondary">Auto-refresh</Text>
          <Switch size="small" checked={autoRefresh} onChange={setAutoRefresh} />
          <Button size="small" icon={<ReloadOutlined />} onClick={() => onLoadLog(200)}>
            Refresh
          </Button>
        </Space>
      }
    >
      <Input.TextArea
        value={hookLog || "(no logs)"}
        readOnly
        autoSize={{ minRows: 16, maxRows: 30 }}
        style={{ fontFamily: "monospace", fontSize: 12 }}
      />
    </Card>
  );
}

// ── Main Page ────────────────────────────────────────────────────────────

function MemPalacePage() {
  const {
    status,
    wings,
    drawers,
    drawerTotal,
    config,
    hookLog,
    kgStats,
    kgEntities,
    kgTriples,
    kgEntityTotal,
    kgTripleTotal,
    loading,
    loadStatus,
    loadWings,
    loadDrawers,
    loadConfig,
    updateConfig,
    loadHookLog,
    loadKgStats,
    loadKgEntities,
    loadKgTriples,
    deleteDrawer,
  } = useMemPalace();

  const handleRefreshAll = () => {
    loadStatus();
    loadWings();
    loadConfig();
    loadKgStats();
  };

  return (
    <div>
      <PageHeader
        items={[{ title: "Agent" }, { title: "MemPalace" }]}
        extra={
          <Button icon={<ReloadOutlined />} onClick={handleRefreshAll}>
            Refresh
          </Button>
        }
      />

      <div style={{ padding: 20 }}>
        <Tabs defaultActiveKey="overview">
          <TabPane tab="Overview" key="overview">
            <OverviewTab
              status={status}
              config={config}
              kgStats={kgStats}
              onRefresh={handleRefreshAll}
              onConfigChange={updateConfig}
            />
          </TabPane>
          <TabPane tab="Structure" key="structure">
            <StructureTab
              wings={wings}
              drawers={drawers}
              drawerTotal={drawerTotal}
              loading={loading}
              onSelectRoom={loadDrawers}
              onDeleteDrawer={deleteDrawer}
              onRefreshWings={loadWings}
            />
          </TabPane>
          <TabPane tab="Knowledge Graph" key="kg">
            <KnowledgeGraphTab
              kgEntities={kgEntities}
              kgTriples={kgTriples}
              kgEntityTotal={kgEntityTotal}
              kgTripleTotal={kgTripleTotal}
              kgStats={kgStats}
              onLoadEntities={loadKgEntities}
              onLoadTriples={loadKgTriples}
              onLoadKgStats={loadKgStats}
            />
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
