import { request } from "../request";

export type TunnelMode = "manual" | "quick" | "named";

export interface MediaServerConfig {
  enabled: boolean;
  server_url: string;
  tunnel_domain: string;
  tunnel_mode: TunnelMode;
  named_tunnel_name: string;
  named_tunnel_hostname: string;
  named_tunnel_config_file: string;
  /** @deprecated use tunnel_mode='quick' */
  use_cloudflare_tunnel?: boolean;
  media_secret: string;
  allowed_dirs: string[];
  max_size_mb: number;
}

export interface MediaServerStatus {
  running: boolean;
  port?: number | null;
  tunnel_mode?: TunnelMode;
  tunnel_url?: string;
  tunnel_running?: boolean;
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
