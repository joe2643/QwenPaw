import { request } from "../request";

export const mempalaceApi = {
  /** Palace overview: wing count, drawer totals, KG stats */
  getStatus: () => request<any>("/mempalace/status"),

  /** List all wings with their rooms */
  getWings: () => request<any[]>("/mempalace/wings"),

  /** List drawers in a specific wing/room */
  getDrawers: (wing: string, room: string, offset?: number, limit?: number) => {
    const params = new URLSearchParams();
    if (offset != null) params.set("offset", String(offset));
    if (limit != null) params.set("limit", String(limit));
    const qs = params.toString();
    return request<any>(`/mempalace/wings/${encodeURIComponent(wing)}/rooms/${encodeURIComponent(room)}${qs ? `?${qs}` : ""}`);
  },

  /** Get a single drawer by ID */
  getDrawer: (id: string) => request<any>(`/mempalace/drawer/${encodeURIComponent(id)}`),

  /** Update a drawer */
  updateDrawer: (id: string, data: any) =>
    request<any>(`/mempalace/drawer/${encodeURIComponent(id)}`, {
      method: "PUT",
      body: JSON.stringify(data),
    }),

  /** Delete a drawer */
  deleteDrawer: (id: string) =>
    request<any>(`/mempalace/drawer/${encodeURIComponent(id)}`, {
      method: "DELETE",
    }),

  /** Knowledge graph statistics */
  getKgStats: () => request<any>("/mempalace/kg/stats"),

  /** Knowledge graph entities */
  getKgEntities: (offset?: number, limit?: number) => {
    const params = new URLSearchParams();
    if (offset != null) params.set("offset", String(offset));
    if (limit != null) params.set("limit", String(limit));
    const qs = params.toString();
    return request<any>(`/mempalace/kg/entities${qs ? `?${qs}` : ""}`);
  },

  /** Knowledge graph triples */
  getKgTriples: (offset?: number, limit?: number) => {
    const params = new URLSearchParams();
    if (offset != null) params.set("offset", String(offset));
    if (limit != null) params.set("limit", String(limit));
    const qs = params.toString();
    return request<any>(`/mempalace/kg/triples${qs ? `?${qs}` : ""}`);
  },

  /** Hook execution log */
  getHookLog: (lines?: number) => {
    const params = new URLSearchParams();
    if (lines != null) params.set("lines", String(lines));
    const qs = params.toString();
    return request<any>(`/mempalace/hooks/log${qs ? `?${qs}` : ""}`);
  },

  /** MemPalace configuration */
  getConfig: () => request<any>("/config/mempalace"),

  /** Update MemPalace configuration */
  updateConfig: (data: any) =>
    request<any>("/config/mempalace", {
      method: "PUT",
      body: JSON.stringify(data),
    }),
};
