import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import type { AnalysisListFilters, AtlasLensApiClient } from "../api/client";
import { useI18n } from "../i18n";

const pageSize = 12;

export function HistoryPanel({
  client,
  onOpenAnalysis,
  onRerunAccepted,
}: {
  client: AtlasLensApiClient;
  onOpenAnalysis: (id: string) => void;
  onRerunAccepted: (id: string) => void;
}) {
  const { locale, t, serverMessage } = useI18n();
  const queryClient = useQueryClient();
  const [offset, setOffset] = useState(0);
  const [search, setSearch] = useState("");
  const [provider, setProvider] = useState("");
  const [status, setStatus] = useState<AnalysisListFilters["status"]>();
  const [classification, setClassification] = useState<AnalysisListFilters["classification"]>();
  const [createdFrom, setCreatedFrom] = useState("");
  const [createdTo, setCreatedTo] = useState("");
  const filters: AnalysisListFilters = {
    limit: pageSize,
    offset,
    search: search.trim() || undefined,
    provider: provider.trim() || undefined,
    status,
    classification,
    createdFrom: createdFrom ? `${createdFrom}T00:00:00Z` : undefined,
    createdTo: createdTo ? `${createdTo}T23:59:59Z` : undefined,
  };
  const history = useQuery({
    queryKey: ["history", filters],
    queryFn: () => client.listAnalyses(filters),
  });
  const remove = useMutation({
    mutationFn: (id: string) => client.deleteAnalysis(id),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: ["history"] }),
  });
  const rerun = useMutation({
    mutationFn: (id: string) => client.rerunAnalysis(id),
    onSuccess: (accepted) => onRerunAccepted(accepted.id),
  });
  const formatter = new Intl.DateTimeFormat(locale === "tr" ? "tr-TR" : "en-US", { dateStyle: "medium", timeStyle: "short" });

  return (
    <main id="main-content" className="workspace-shell">
      <header className="workspace-heading">
        <div><p className="eyebrow">{t("history.eyebrow")}</p><h1>{t("history.title")}</h1><p>{t("history.description")}</p></div>
      </header>
      <section className="filter-bar" aria-label={t("history.filters")}>
        <label><span>{t("history.search")}</span><input value={search} onChange={(event) => { setSearch(event.target.value); setOffset(0); }} /></label>
        <label><span>{t("history.provider")}</span><input value={provider} onChange={(event) => { setProvider(event.target.value); setOffset(0); }} /></label>
        <label><span>{t("history.status")}</span><select value={status ?? ""} onChange={(event) => { setStatus((event.target.value || undefined) as AnalysisListFilters["status"]); setOffset(0); }}><option value="">{t("common.all")}</option><option value="completed">{t("result.completed")}</option><option value="processing">{t("history.processing")}</option><option value="failed">{t("history.failed")}</option></select></label>
        <label><span>{t("history.classification")}</span><select value={classification ?? ""} onChange={(event) => { setClassification((event.target.value || undefined) as AnalysisListFilters["classification"]); setOffset(0); }}><option value="">{t("common.all")}</option><option value="real">{t("history.real")}</option><option value="simulated">{t("history.simulated")}</option></select></label>
        <label><span>{t("history.from")}</span><input type="date" value={createdFrom} onChange={(event) => { setCreatedFrom(event.target.value); setOffset(0); }} /></label>
        <label><span>{t("history.to")}</span><input type="date" value={createdTo} onChange={(event) => { setCreatedTo(event.target.value); setOffset(0); }} /></label>
      </section>
      {history.isPending ? <p className="workspace-state" role="status">{t("history.loading")}</p> : null}
      {history.isError ? <div className="workspace-state notice notice--error" role="alert">{serverMessage(history.error.message)}</div> : null}
      {history.data?.items.length === 0 ? <div className="workspace-state"><h2>{t("history.emptyTitle")}</h2><p>{t("history.empty")}</p></div> : null}
      {history.data?.items.length ? (
        <ul className="history-list">
          {history.data.items.map((item) => (
            <li key={item.id} className="history-item">
              <div className="history-item__main">
                <span className={`status-badge ${item.result_classification === "simulated" ? "status-badge--partial" : ""}`}>{t(item.result_classification === "simulated" ? "history.simulated" : "history.real")}</span>
                <h2>{item.primary_label ?? t("history.noPrimary")}</h2>
                <p>{formatter.format(new Date(item.created_at))}</p>
                <div className="history-facts"><span>{item.candidate_count} {t("history.candidates")}</span><span>{item.evidence_count} {t("history.evidence")}</span><span>{item.provider_ids.length ? item.provider_ids.join(", ") : t("history.noProvider")}</span></div>
              </div>
              <div className="history-actions">
                <button type="button" className="secondary-button" onClick={() => onOpenAnalysis(item.id)}>{t("history.open")}</button>
                <button type="button" className="secondary-button" disabled={!item.source_retained || rerun.isPending} title={!item.source_retained ? t("history.rerunUnavailable") : undefined} onClick={() => rerun.mutate(item.id)}>{t("history.rerun")}</button>
                <button type="button" className="secondary-button danger-button" disabled={remove.isPending} onClick={() => remove.mutate(item.id)}>{t("result.delete")}</button>
              </div>
            </li>
          ))}
        </ul>
      ) : null}
      {rerun.isError ? <div className="notice notice--warning" role="alert">{t("history.rerunUnavailable")}</div> : null}
      {remove.isError ? <div className="notice notice--error" role="alert">{t("errors.delete")}</div> : null}
      {history.data && history.data.total > pageSize ? <nav className="pagination" aria-label={t("history.pagination")}><button type="button" className="secondary-button" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - pageSize))}>{t("common.previous")}</button><span>{offset + 1}–{Math.min(offset + pageSize, history.data.total)} / {history.data.total}</span><button type="button" className="secondary-button" disabled={offset + pageSize >= history.data.total} onClick={() => setOffset(offset + pageSize)}>{t("common.next")}</button></nav> : null}
    </main>
  );
}
