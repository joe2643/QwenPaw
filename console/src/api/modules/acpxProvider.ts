import { request } from "../request";

export interface AcpxProviderConfig {
  turn_timeout_seconds: number;
  terminal_wait_seconds: number;
}

export const acpxProviderApi = {
  getAcpxProviderConfig: () =>
    request<AcpxProviderConfig>("/config/acpx-provider"),

  updateAcpxProviderConfig: (patch: Partial<AcpxProviderConfig>) =>
    request<AcpxProviderConfig>("/config/acpx-provider", {
      method: "PUT",
      body: JSON.stringify(patch),
    }),
};
