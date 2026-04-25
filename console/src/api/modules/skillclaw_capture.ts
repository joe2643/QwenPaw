import { request } from "../request";

export type SkillClawCaptureMode = "file" | "http";

export interface SkillClawCaptureConfig {
  enabled: boolean;
  mode: SkillClawCaptureMode;
  records_dir: string;
  ingest_url: string;
  ingest_api_key: string;
  session_id_prefix: string;
}

export const skillclawCaptureApi = {
  /** Read the active agent's SkillClaw capture config. */
  getConfig: () => request<SkillClawCaptureConfig>("/config/skillclaw-capture"),

  /** Save + hot-reload the active agent's SkillClaw capture config. */
  updateConfig: (data: Partial<SkillClawCaptureConfig>) =>
    request<SkillClawCaptureConfig>("/config/skillclaw-capture", {
      method: "PUT",
      body: JSON.stringify(data),
    }),
};
