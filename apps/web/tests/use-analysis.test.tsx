import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";
import type { AtlasLensApiClient } from "../src/api/client";
import type { Analysis } from "../src/api/schemas";
import { useAnalysis } from "../src/use-analysis";
import { analysisId, completedAnalysis, createFakeClient } from "./fixtures";

describe("useAnalysis stream freshness", () => {
  it("drops stale SSE progress after disconnect and accepts fresher polling data", async () => {
    const initial: Analysis = { ...completedAnalysis, status: "processing", candidates: [], progress: { stage: "quality", percent: 40, message_key: "progress.quality" } };
    const fresh: Analysis = { ...initial, progress: { stage: "retrieval", percent: 82, message_key: "progress.retrieval" } };
    const getAnalysis = vi.fn().mockResolvedValueOnce(initial).mockResolvedValue(fresh);
    let subscriptions = 0;
    const subscribeImpl: AtlasLensApiClient["subscribeAnalysis"] = (_id, handlers) => {
      subscriptions += 1;
      if (subscriptions === 1) {
        window.setTimeout(() => handlers.onOpen?.(), 5);
        window.setTimeout(() => handlers.onEvent({
          event_id: "progress-70",
          event_type: "progress",
          analysis_id: analysisId,
          occurred_at: "2026-07-12T10:00:00Z",
          status: "processing",
          progress: { stage: "global_prediction", percent: 70, message_key: "progress.global" },
        }), 10);
        window.setTimeout(() => handlers.onDisconnect(), 20);
      } else {
        window.setTimeout(() => handlers.onOpen?.(), 5);
      }
      return { close: vi.fn() };
    };
    const subscribeAnalysis = vi.fn(subscribeImpl);
    const { client } = createFakeClient({ clientOverrides: { getAnalysis, subscribeAnalysis } });
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
    const { result, unmount } = renderHook(() => useAnalysis(client, analysisId), { wrapper });

    await waitFor(() => expect(result.current.progress?.percent).toBe(70));
    await waitFor(() => expect(result.current.streamState).toBe("polling"));
    expect(result.current.progress?.percent).toBe(70);
    await waitFor(() => expect(result.current.progress?.percent).toBe(82), { timeout: 3000 });
    expect(getAnalysis).toHaveBeenCalledTimes(2);
    unmount();
  });
});
