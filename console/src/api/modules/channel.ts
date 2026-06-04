import { request } from "../request";
import type { ChannelConfig, SingleChannelConfig } from "../types";

export const channelApi = {
  listChannelTypes: () => request<string[]>("/config/channels/types"),

  listChannels: () => request<ChannelConfig>("/config/channels"),

  updateChannels: (body: ChannelConfig) =>
    request<ChannelConfig>("/config/channels", {
      method: "PUT",
      body: JSON.stringify(body),
    }),

  getChannelConfig: (channelName: string) =>
    request<SingleChannelConfig>(
      `/config/channels/${encodeURIComponent(channelName)}`,
    ),

  updateChannelConfig: (channelName: string, body: SingleChannelConfig) =>
    request<SingleChannelConfig>(
      `/config/channels/${encodeURIComponent(channelName)}`,
      {
        method: "PUT",
        body: JSON.stringify(body),
      },
    ),

  getChannelQrcode: (channel: string, params?: Record<string, string>) => {
    const qs = params ? "?" + new URLSearchParams(params).toString() : "";
    return request<{ qrcode_img: string; poll_token: string }>(
      `/config/channels/${encodeURIComponent(channel)}/qrcode${qs}`,
    );
  },

  getChannelQrcodeStatus: (
    channel: string,
    token: string,
    params?: Record<string, string>,
  ) => {
    const extra = params ? "&" + new URLSearchParams(params).toString() : "";
    return request<{
      status: string;
      credentials: Record<string, string>;
    }>(
      `/config/channels/${encodeURIComponent(
        channel,
      )}/qrcode/status?token=${encodeURIComponent(token)}${extra}`,
    );
  },

  startWhatsappPair: (phone?: string) =>
    request<{
      status: string;
      pair_code?: string;
      qr_image?: string;
      phone?: string;
    }>(
      `/config/channels/whatsapp/pair${
        phone ? `?phone=${encodeURIComponent(phone)}` : ""
      }`,
      { method: "POST" },
    ),
  checkWhatsappPairStatus: () =>
    request<{ status: string; pair_code?: string; qr_image?: string }>(
      "/config/channels/whatsapp/pair/status",
    ),
  stopWhatsappPair: () =>
    request<{ status: string }>("/config/channels/whatsapp/pair/stop", {
      method: "POST",
    }),
  getWhatsappQrcode: () =>
    request<{ status: string; qr_image?: string }>(
      "/config/channels/whatsapp/qrcode",
      { method: "POST" },
    ),
  unbindWhatsapp: () =>
    request<{ status: string; detail?: string }>(
      "/config/channels/whatsapp/unbind",
      { method: "POST" },
    ),
  getWhatsappStatus: () =>
    request<{ linked: boolean; phone?: string }>(
      "/config/channels/whatsapp/status",
    ),

  // ── Signal link flow (signal-cli subprocess pairing) ─────────────────────
  startSignalLink: (device_name?: string) =>
    request<{
      status: string;
      qr_image?: string;
      link_url?: string;
      device_name?: string;
    }>("/config/channels/signal/link", {
      method: "POST",
      body: JSON.stringify({ device_name: device_name || "QwenPaw" }),
      headers: { "Content-Type": "application/json" },
    }),
  checkSignalLinkStatus: () =>
    request<{
      status: string;
      phone?: string;
      uuid?: string;
      link_url?: string;
      error?: string;
    }>("/config/channels/signal/link/status"),
  stopSignalLink: () =>
    request<{ status: string }>("/config/channels/signal/link/stop", {
      method: "POST",
    }),
  unbindSignal: () =>
    request<{ status: string; detail?: string }>(
      "/config/channels/signal/unbind",
      { method: "POST" },
    ),
  getSignalStatus: () =>
    request<{ linked: boolean; phone?: string | null; uuid?: string | null }>(
      "/config/channels/signal/status",
    ),
  listSignalContacts: () =>
    request<{
      contacts: Array<{ number: string; uuid: string; name: string }>;
    }>("/config/channels/signal/contacts"),
  listSignalGroups: () =>
    request<{ groups: Array<{ id: string; blocked: boolean }> }>(
      "/config/channels/signal/groups",
    ),
};
