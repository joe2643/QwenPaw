import { useCallback, useEffect, useRef, useState } from "react";
import { useAppMessage } from "../../../hooks/useAppMessage";
import { mempalaceApi } from "../../../api/modules/mempalace";
import { useAgentStore } from "../../../stores/agentStore";

export interface WingInfo {
  name: string;
  rooms: { name: string; drawer_count: number }[];
}

export function useMemPalace() {
  const { selectedAgent } = useAgentStore();
  const { message } = useAppMessage();

  const [status, setStatus] = useState<any>(null);
  const [wings, setWings] = useState<WingInfo[]>([]);
  const [drawers, setDrawers] = useState<any[]>([]);
  const [drawerTotal, setDrawerTotal] = useState(0);
  const [config, setConfig] = useState<any>(null);
  const [hookLog, setHookLog] = useState("");
  const [kgStats, setKgStats] = useState<any>(null);
  const [kgEntities, setKgEntities] = useState<any[]>([]);
  const [kgTriples, setKgTriples] = useState<any[]>([]);
  const [kgEntityTotal, setKgEntityTotal] = useState(0);
  const [kgTripleTotal, setKgTripleTotal] = useState(0);
  const [loading, setLoading] = useState(false);

  // -- Status --
  const loadStatus = useCallback(async () => {
    try {
      const data = await mempalaceApi.getStatus();
      setStatus(data);
    } catch (err) {
      console.error("Failed to load MemPalace status:", err);
    }
  }, []);

  // -- Wings --
  const loadWings = useCallback(async () => {
    try {
      const data = await mempalaceApi.getWings();
      setWings(Array.isArray(data?.wings) ? data.wings : Array.isArray(data) ? data : []);
    } catch (err) {
      console.error("Failed to load wings:", err);
      message.error("Failed to load MemPalace wings");
    }
  }, [message]);

  // -- Drawers --
  const loadDrawers = useCallback(
    async (wing: string, room: string, offset = 0, limit = 50) => {
      setLoading(true);
      try {
        const data = await mempalaceApi.getDrawers(wing, room, offset, limit);
        const items = data?.items ?? data?.drawers ?? (Array.isArray(data) ? data : []);
        setDrawers(items);
        setDrawerTotal(data?.total ?? items.length);
      } catch (err) {
        console.error("Failed to load drawers:", err);
        message.error("Failed to load drawers");
      } finally {
        setLoading(false);
      }
    },
    [message],
  );

  // -- Config --
  const loadConfig = useCallback(async () => {
    try {
      const data = await mempalaceApi.getConfig();
      setConfig(data);
    } catch (err) {
      console.error("Failed to load MemPalace config:", err);
    }
  }, []);

  const updateConfig = useCallback(
    async (data: any) => {
      try {
        await mempalaceApi.updateConfig(data);
        message.success("Configuration updated");
        await loadConfig();
      } catch (err: any) {
        message.error(err?.message || "Failed to update config");
      }
    },
    [message, loadConfig],
  );

  // -- Hook log --
  const loadHookLog = useCallback(async (lines = 200) => {
    try {
      const data = await mempalaceApi.getHookLog(lines);
      setHookLog(typeof data === "string" ? data : data?.log ?? data?.lines?.join?.("\n") ?? JSON.stringify(data, null, 2));
    } catch (err) {
      console.error("Failed to load hook log:", err);
    }
  }, []);

  // -- Knowledge Graph --
  const loadKgStats = useCallback(async () => {
    try {
      const data = await mempalaceApi.getKgStats();
      setKgStats(data);
    } catch (err) {
      console.error("Failed to load KG stats:", err);
    }
  }, []);

  const loadKgEntities = useCallback(
    async (offset = 0, limit = 50) => {
      try {
        const data = await mempalaceApi.getKgEntities(offset, limit);
        const items = data?.items ?? data?.entities ?? (Array.isArray(data) ? data : []);
        setKgEntities(items);
        setKgEntityTotal(data?.total ?? items.length);
      } catch (err) {
        console.error("Failed to load KG entities:", err);
        message.error("Failed to load KG entities");
      }
    },
    [message],
  );

  const loadKgTriples = useCallback(
    async (offset = 0, limit = 50) => {
      try {
        const data = await mempalaceApi.getKgTriples(offset, limit);
        const items = data?.items ?? data?.triples ?? (Array.isArray(data) ? data : []);
        setKgTriples(items);
        setKgTripleTotal(data?.total ?? items.length);
      } catch (err) {
        console.error("Failed to load KG triples:", err);
        message.error("Failed to load KG triples");
      }
    },
    [message],
  );

  // -- Delete drawer --
  const deleteDrawer = useCallback(
    async (id: string) => {
      try {
        await mempalaceApi.deleteDrawer(id);
        message.success("Drawer deleted");
      } catch (err: any) {
        message.error(err?.message || "Failed to delete drawer");
      }
    },
    [message],
  );

  // -- Initial load --
  useEffect(() => {
    loadStatus();
    loadWings();
    loadConfig();
    loadKgStats();
  }, [selectedAgent, loadStatus, loadWings, loadConfig, loadKgStats]);

  return {
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
  };
}
