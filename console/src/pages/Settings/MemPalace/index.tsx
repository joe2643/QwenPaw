import { useState, useEffect, useCallback, useRef } from "react";
import {
  Card,
  Table,
  Switch,
  Tag,
  Button,
  Empty,
} from "@agentscope-ai/design";
import {
  Tabs,
  Tree,
  Space,
  Statistic,
  Popconfirm,
  Typography,
  Descriptions,
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

const { Text } = Typography;
const { TabPane } = Tabs;

function hallColor(hall: string): string {
  const map: Record<string, string> = {
    hall_facts: "blue", hall_events: "purple", hall_discoveries: "gold",
    hall_preferences: "green", hall_advice: "orange", hall_diary: "cyan",
  };
  return map[hall] || "default";
}

/** dark-mode-aware text style */
function useDarkStyles() {
  const { isDark } = useTheme();
  return {
    text: { color: isDark ? "rgba(255,255,255,0.85)" : undefined },
    textSecondary: { color: isDark ? "rgba(255,255,255,0.45)" : undefined },
    card: { borderColor: isDark ? "#303030" : undefined },
    cardDark: isDark ? { background: "#1f1f1f", borderColor: "#303030" } : {},
    mono: {
      fontFamily: "monospace", fontSize: 12, lineHeight: 1.6, padding: 12,
      borderRadius: 6, maxHeight: 500, overflowY: "auto" as const,
      background: isDark ? "#141414" : "#fafafa",
      color: isDark ? "rgba(255,255,255,0.85)" : "rgba(0,0,0,0.88)",
      border: isDark ? "1px solid #303030" : "1px solid #e8e8e8",
    },
    isDark,
  };
}

// ── Overview Tab ─────────────────────────────────────────────────────────

function OverviewTab({ status, kgStats, onRefresh }: { status: any; kgStats: any; onRefresh: () => void }) {
  const ds = useDarkStyles();
  return (
    <Space direction="vertical" size="middle" style={{ width: "100%" }}>
      <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
        {[
          { title: "Total Drawers", value: status?.total_drawers, icon: <DatabaseOutlined /> },
          { title: "Wings", value: status?.wings ? Object.keys(status.wings).length : "-", icon: <ApartmentOutlined /> },
          { title: "KG Entities", value: kgStats?.entity_count ?? status?.kg?.entity_count ?? "-", icon: <NodeIndexOutlined /> },
          { title: "KG Triples", value: kgStats?.triple_count ?? status?.kg?.triple_count ?? "-", icon: <FileTextOutlined /> },
        ].map((s) => (
          <Card key={s.title} size="small" style={{ flex: 1, minWidth: 180, ...ds.card }}>
            <Statistic title={<span style={ds.text}>{s.title}</span>} value={s.value ?? "-"} prefix={s.icon}
              valueStyle={ds.text} />
          </Card>
        ))}
      </div>
    </Space>
  );
}

// ── Structure Tab ────────────────────────────────────────────────────────

function StructureTab({
  wings, drawers, drawerTotal, loading, onSelectRoom, onDeleteDrawer, onRefreshWings,
}: {
  wings: any[]; drawers: any[]; drawerTotal: number; loading: boolean;
  onSelectRoom: (wing: string, room: string, offset?: number, limit?: number) => void;
  onDeleteDrawer: (id: string) => void; onRefreshWings: () => void;
}) {
  const ds = useDarkStyles();
  const [selectedRoom, setSelectedRoom] = useState<{ wing: string; room: string } | null>(null);
  const [page, setPage] = useState(1);

  const treeData: DataNode[] = wings.map((wing) => ({
    key: `wing:${wing.name}`,
    title: <span style={ds.text}><ApartmentOutlined style={{ marginRight: 6 }} /><b>{wing.name}</b>
      <span style={ds.textSecondary}> ({(wing.rooms || []).reduce((s: number, r: any) => s + (r.drawer_count ?? r.count ?? 0), 0)})</span></span>,
    children: (wing.rooms || []).map((room: any) => ({
      key: `room:${wing.name}/${room.name}`,
      title: <span style={ds.text}>{room.name} <Tag>{room.drawer_count ?? room.count ?? "?"}</Tag></span>,
      isLeaf: true,
    })),
  }));

  const handleSelect = (_: any, info: any) => {
    const key = info.node.key as string;
    if (key.startsWith("room:")) {
      const [wing, room] = key.replace("room:", "").split("/");
      setSelectedRoom({ wing, room });
      setPage(1);
      onSelectRoom(wing, room, 0, 50);
    }
  };

  const columns = [
    { title: "ID", dataIndex: "id", key: "id", width: 200, ellipsis: true,
      render: (id: string) => <Text copyable={{ text: id }} style={{ fontSize: 11, ...ds.text }}>{id}</Text> },
    { title: "Hall", key: "hall", width: 140,
      render: (_: any, r: any) => { const h = r.hall || r.metadata?.hall; return h ? <Tag color={hallColor(h)}>{h}</Tag> : <Text style={ds.textSecondary}>—</Text>; } },
    { title: "Content", key: "content", ellipsis: true,
      render: (_: any, r: any) => <Text style={{ fontSize: 12, ...ds.text }}>{(r.content_preview || r.content || "").substring(0, 120)}</Text> },
    { title: "Date", key: "date", width: 100,
      render: (_: any, r: any) => <Text style={ds.textSecondary}>{(r.filed_at || r.metadata?.filed_at || "").substring(0, 10) || "—"}</Text> },
    { title: "", key: "actions", width: 50,
      render: (_: any, r: any) => (
        <Popconfirm title="Delete?" onConfirm={() => { onDeleteDrawer(r.id); if (selectedRoom) setTimeout(() => onSelectRoom(selectedRoom.wing, selectedRoom.room, (page-1)*50, 50), 300); }}>
          <Button type="text" danger size="small" icon={<DeleteOutlined />} />
        </Popconfirm>
      ) },
  ];

  return (
    <div style={{ display: "flex", gap: 16, minHeight: 400 }}>
      <Card size="small" title={<span style={ds.text}>Wings & Rooms</span>}
        extra={<Button size="small" icon={<ReloadOutlined />} onClick={onRefreshWings} />}
        style={{ width: 280, flexShrink: 0, ...ds.cardDark }}>
        {treeData.length === 0 ? <Empty description="No wings" /> :
          <Tree treeData={treeData} defaultExpandAll onSelect={handleSelect}
            selectedKeys={selectedRoom ? [`room:${selectedRoom.wing}/${selectedRoom.room}`] : []} />}
      </Card>
      <div style={{ flex: 1 }}>
        {selectedRoom ? (
          <Card size="small" style={ds.card}
            title={<span style={ds.text}><Tag color="blue">{selectedRoom.wing}</Tag><Tag color="green">{selectedRoom.room}</Tag>
              <span style={ds.textSecondary}>{drawerTotal} drawer{drawerTotal !== 1 ? "s" : ""}</span></span>}>
            <Table dataSource={drawers} columns={columns} rowKey="id" loading={loading} size="small"
              pagination={{ current: page, pageSize: 50, total: drawerTotal, showTotal: (t) => `Total ${t}`,
                onChange: (p) => { setPage(p); onSelectRoom(selectedRoom.wing, selectedRoom.room, (p-1)*50, 50); } }} />
          </Card>
        ) : <Card size="small" style={ds.card}><Empty description="Select a room" /></Card>}
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
  onLoadEntities: (o?: number, l?: number) => void; onLoadTriples: (o?: number, l?: number) => void; onLoadKgStats: () => void;
}) {
  const ds = useDarkStyles();
  const [ePage, setEPage] = useState(1);
  const [tPage, setTPage] = useState(1);
  useEffect(() => { onLoadEntities(0, 50); onLoadTriples(0, 50); }, []);

  return (
    <Space direction="vertical" size="middle" style={{ width: "100%" }}>
      {kgStats && (
        <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
          <Card size="small" style={{ flex: 1, minWidth: 160, ...ds.card }}>
            <Statistic title={<span style={ds.text}>Entities</span>} value={kgStats.entity_count ?? "-"} valueStyle={ds.text} />
          </Card>
          <Card size="small" style={{ flex: 1, minWidth: 160, ...ds.card }}>
            <Statistic title={<span style={ds.text}>Triples</span>} value={kgStats.triple_count ?? "-"} valueStyle={ds.text} />
          </Card>
        </div>
      )}
      <Card title={<span style={ds.text}>Entities</span>} size="small" style={ds.card}
        extra={<Button size="small" icon={<ReloadOutlined />} onClick={() => { onLoadEntities(0,50); onLoadKgStats(); }}>Refresh</Button>}>
        <Table dataSource={kgEntities} columns={[
          { title: "Name", dataIndex: "name", key: "name", ellipsis: true },
          { title: "Type", dataIndex: "type", key: "type", width: 120, render: (t: string) => <Tag>{t || "auto"}</Tag> },
          { title: "Properties", dataIndex: "properties", key: "props", ellipsis: true, render: (p: any) => <Text style={ds.textSecondary}>{p ? JSON.stringify(p).substring(0,80) : "—"}</Text> },
        ]} rowKey={(r) => r.id ?? r.name} size="small"
          pagination={{ current: ePage, pageSize: 50, total: kgEntityTotal, showTotal: (t) => `Total ${t}`,
            onChange: (p) => { setEPage(p); onLoadEntities((p-1)*50, 50); } }} />
      </Card>
      <Card title={<span style={ds.text}>Triples</span>} size="small" style={ds.card}
        extra={<Button size="small" icon={<ReloadOutlined />} onClick={() => onLoadTriples(0,50)}>Refresh</Button>}>
        <Table dataSource={kgTriples} columns={[
          { title: "Subject", dataIndex: "subject", key: "s", ellipsis: true },
          { title: "Predicate", dataIndex: "predicate", key: "p", width: 160, render: (p: string) => <Tag color="purple">{p}</Tag> },
          { title: "Object", dataIndex: "object", key: "o", ellipsis: true },
          { title: "From", dataIndex: "valid_from", key: "vf", width: 100 },
          { title: "Source", dataIndex: "source_closet", key: "src", width: 120, ellipsis: true },
        ]} rowKey={(r) => r.id ?? `${r.subject}-${r.predicate}-${r.object}`} size="small"
          pagination={{ current: tPage, pageSize: 50, total: kgTripleTotal, showTotal: (t) => `Total ${t}`,
            onChange: (p) => { setTPage(p); onLoadTriples((p-1)*50, 50); } }} />
      </Card>
    </Space>
  );
}

// ── Hooks Tab ────────────────────────────────────────────────────────────

function HooksTab({ hookLog, onLoadLog }: { hookLog: string; onLoadLog: (n?: number) => void }) {
  const ds = useDarkStyles();
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(false);
  useEffect(() => { onLoadLog(200); }, []);
  useEffect(() => {
    if (autoRefresh) intervalRef.current = setInterval(() => onLoadLog(200), 5000);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [autoRefresh, onLoadLog]);

  const lines = (hookLog || "").split("\n").filter(Boolean);
  const lineColor = (line: string) => {
    if (line.includes("ERROR") || line.includes("FAILED")) return "#ff4d4f";
    if (line.includes("BgSave") || line.includes("Diary")) return ds.isDark ? "#52c41a" : "#389e0d";
    if (line.includes("Interval") || line.includes("PreCompact")) return ds.isDark ? "#1890ff" : "#096dd9";
    if (line.includes("PreReply")) return ds.isDark ? "#faad14" : "#d48806";
    return ds.isDark ? "rgba(255,255,255,0.65)" : "rgba(0,0,0,0.65)";
  };

  return (
    <Card title={<span style={ds.text}>Hook Log</span>} size="small" style={ds.card}
      extra={<Space><Text style={ds.textSecondary}>Auto</Text><Switch size="small" checked={autoRefresh} onChange={setAutoRefresh} />
        <Button size="small" icon={<ReloadOutlined />} onClick={() => onLoadLog(200)}>Refresh</Button></Space>}>
      <div style={ds.mono}>
        {lines.length === 0 ? <Text style={ds.textSecondary}>(no logs)</Text> :
          lines.map((l, i) => <div key={i} style={{ color: lineColor(l), whiteSpace: "pre-wrap" }}>{l}</div>)}
      </div>
    </Card>
  );
}

// ── Main ─────────────────────────────────────────────────────────────────

export default function MemPalacePage() {
  const ds = useDarkStyles();
  const {
    status, wings, drawers, drawerTotal, hookLog,
    kgStats, kgEntities, kgTriples, kgEntityTotal, kgTripleTotal, loading,
    loadStatus, loadWings, loadDrawers, loadConfig,
    loadHookLog, loadKgStats, loadKgEntities, loadKgTriples, deleteDrawer,
  } = useMemPalace();

  const refresh = () => { loadStatus(); loadWings(); loadConfig(); loadKgStats(); };

  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column" }}>
      <PageHeader items={[{ title: "Settings" }, { title: "MemPalace" }]}
        extra={<Button icon={<ReloadOutlined />} onClick={refresh}>Refresh</Button>} />
      <div style={{ flex: 1, overflowY: "auto", padding: 20 }}>
        <Tabs defaultActiveKey="overview">
          <TabPane tab={<span style={ds.text}>Overview</span>} key="overview">
            <OverviewTab status={status} kgStats={kgStats} onRefresh={refresh} />
          </TabPane>
          <TabPane tab={<span style={ds.text}>Structure</span>} key="structure">
            <StructureTab wings={wings} drawers={drawers} drawerTotal={drawerTotal} loading={loading}
              onSelectRoom={loadDrawers} onDeleteDrawer={deleteDrawer} onRefreshWings={loadWings} />
          </TabPane>
          <TabPane tab={<span style={ds.text}>Knowledge Graph</span>} key="kg">
            <KnowledgeGraphTab kgEntities={kgEntities} kgTriples={kgTriples}
              kgEntityTotal={kgEntityTotal} kgTripleTotal={kgTripleTotal} kgStats={kgStats}
              onLoadEntities={loadKgEntities} onLoadTriples={loadKgTriples} onLoadKgStats={loadKgStats} />
          </TabPane>
          <TabPane tab={<span style={ds.text}>Hooks</span>} key="hooks">
            <HooksTab hookLog={hookLog} onLoadLog={loadHookLog} />
          </TabPane>
        </Tabs>
      </div>
    </div>
  );
}
