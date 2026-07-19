import { useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import type { AtlasLensApiClient } from "./api/client";
import type { Analysis, AnalysisEvent, Progress } from "./api/schemas";

export type AnalysisStreamState = "idle" | "connecting" | "live" | "polling" | "reconnecting";

export function useAnalysis(client: AtlasLensApiClient, analysisId: string | null) {
  const queryClient = useQueryClient();
  const [streamState, setStreamState] = useState<AnalysisStreamState>("idle");
  const [eventProgress, setEventProgress] = useState<Progress | null>(null);

  const query = useQuery({
    queryKey: ["analysis", analysisId],
    queryFn: () => client.getAnalysis(analysisId as string),
    enabled: analysisId !== null,
    retry: 2,
    retryDelay: (attempt) => Math.min(500 * 2 ** attempt, 2500),
    refetchInterval: (state) => {
      const status = state.state.data?.status;
      return status === "queued" || status === "processing" ? 1500 : false;
    },
  });

  useEffect(() => {
    setEventProgress(null);
    if (!analysisId) {
      setStreamState("idle");
      return undefined;
    }
    let disposed = false;
    let terminal = false;
    let reconnectAttempt = 0;
    let reconnectTimer: ReturnType<typeof setTimeout> | undefined;
    let subscription: ReturnType<AtlasLensApiClient["subscribeAnalysis"]> | undefined;

    const connect = () => {
      if (disposed || terminal) return;
      setStreamState(reconnectAttempt === 0 ? "connecting" : "reconnecting");
      subscription = client.subscribeAnalysis(analysisId, {
        onOpen() {
          if (disposed) return;
          reconnectAttempt = 0;
          setStreamState("live");
        },
        onEvent(event: AnalysisEvent) {
          if (disposed) return;
          if (event.progress) {
            const polled = queryClient.getQueryData<Analysis>(["analysis", analysisId])?.progress;
            setEventProgress((current) => {
              const floor = Math.max(current?.percent ?? 0, polled?.percent ?? 0);
              const fallback = polled && (!current || polled.percent >= current.percent) ? polled : current;
              return event.progress && event.progress.percent >= floor ? event.progress : fallback ?? null;
            });
            queryClient.setQueryData(["analysis", analysisId], (current: Analysis | undefined) => {
              if (!current || !event.progress || event.progress.percent < current.progress.percent) return current;
              return { ...current, status: event.status, progress: event.progress };
            });
          }
          if (["completed", "failed", "deleted"].includes(event.event_type)) {
            terminal = true;
            setStreamState("idle");
            void queryClient.invalidateQueries({ queryKey: ["analysis", analysisId] });
          }
        },
        onDisconnect() {
          if (disposed || terminal) return;
          setEventProgress(null);
          setStreamState("polling");
          reconnectAttempt += 1;
          const delay = Math.min(1000 * 2 ** (reconnectAttempt - 1), 15_000);
          reconnectTimer = setTimeout(connect, delay);
        },
      });
    };

    connect();
    return () => {
      disposed = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      subscription?.close();
    };
  }, [analysisId, client, queryClient]);

  return {
    ...query,
    streamConnected: streamState === "live",
    streamState,
    progress: streamState === "live" ? eventProgress ?? query.data?.progress ?? null : query.data?.progress ?? eventProgress ?? null,
  };
}
