import { request } from "../request";

export interface MediaServerConfig {
  enabled: boolean;
  server_url: string;
  tunnel_domain: string;
  media_secret: string;
  allowed_dirs: string[];
  max_size_mb: number;
}

export interface MediaServerStatus {
  running: boolean;
  reason?: string;
  health?: { status: string; service: string };
}

export const mediaServerApi = {
  getConfig: () => request<MediaServerConfig>("/config/media-server"),

  updateConfig: (body: MediaServerConfig) =>
    request<MediaServerConfig>("/config/media-server", {
      method: "PUT",
      body: JSON.stringify(body),
    }),

  getStatus: () => request<MediaServerStatus>("/config/media-server/status"),
};
