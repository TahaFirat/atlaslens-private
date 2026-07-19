import {
  mapillaryDemoQuerySchema,
  mapillaryDemoStatusSchema,
  type MapillaryDemoQuery,
  type MapillaryDemoStatus,
} from "./models";
import type { paths } from "../api/generated";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");
const STATUS_PATH = "/api/v1/mapillary-demo/status" satisfies keyof paths;
const QUERY_PATH = "/api/v1/mapillary-demo/query" satisfies keyof paths;

export interface MapillaryDemoClient {
  getStatus: () => Promise<MapillaryDemoStatus>;
  query: (file: File, authorizationAcknowledged: boolean) => Promise<MapillaryDemoQuery>;
}

async function checkedJson(response: Response): Promise<unknown> {
  if (!response.ok) throw new Error("mapillary_demo_request_failed");
  try {
    return await response.json();
  } catch {
    throw new Error("mapillary_demo_invalid_response");
  }
}

export const mapillaryDemoClient: MapillaryDemoClient = {
  async getStatus() {
    const payload = await checkedJson(await fetch(`${API_BASE_URL}${STATUS_PATH}`, {
      headers: { Accept: "application/json" },
      cache: "no-store",
    }));
    const parsed = mapillaryDemoStatusSchema.safeParse(payload);
    if (!parsed.success) throw new Error("mapillary_demo_invalid_response");
    return parsed.data;
  },

  async query(file, authorizationAcknowledged) {
    const body = new FormData();
    body.set("image", file);
    body.set("authorization_acknowledged", String(authorizationAcknowledged));
    body.set("analysis_scope", "ankara_reference_pilot");
    const payload = await checkedJson(await fetch(`${API_BASE_URL}${QUERY_PATH}`, {
      method: "POST",
      headers: { Accept: "application/json" },
      body,
      cache: "no-store",
    }));
    const parsed = mapillaryDemoQuerySchema.safeParse(payload);
    if (!parsed.success) throw new Error("mapillary_demo_invalid_response");
    return parsed.data;
  },
};
