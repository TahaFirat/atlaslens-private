import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import type { AtlasLensApiClient } from "../api/client";
import { humanizeToken } from "../display";
import { useI18n } from "../i18n";

export function DatasetQaDashboard({ client }: { client: AtlasLensApiClient }) {
  const { locale, t, serverMessage } = useI18n();
  const [selected, setSelected] = useState<string | null>(null);
  const reports = useQuery({ queryKey: ["dataset-qa"], queryFn: client.listDatasetQaReports });
  const selectedId = selected ?? reports.data?.reports[0]?.report_id ?? null;
  const detail = useQuery({ queryKey: ["dataset-qa", selectedId], queryFn: () => client.getDatasetQaReport(selectedId as string), enabled: selectedId !== null });
  const formatter = new Intl.DateTimeFormat(locale === "tr" ? "tr-TR" : "en-US", { dateStyle: "medium", timeStyle: "short" });
  return (
    <main id="main-content" className="workspace-shell">
      <header className="workspace-heading"><div><p className="eyebrow">{t("qa.eyebrow")}</p><h1>{t("qa.title")}</h1><p>{t("qa.description")}</p></div></header>
      {reports.isPending ? <p className="workspace-state" role="status">{t("qa.loading")}</p> : null}
      {reports.isError ? <div className="workspace-state notice notice--warning" role="alert">{serverMessage(reports.error.message)}</div> : null}
      {reports.data?.reports.length === 0 ? <div className="workspace-state"><h2>{t("qa.emptyTitle")}</h2><p>{t("qa.empty")}</p></div> : null}
      {reports.data?.reports.length ? <div className="dashboard-layout"><nav className="dashboard-index" aria-label={t("qa.reports")}>{reports.data.reports.map((item) => <button type="button" aria-current={item.report_id === selectedId ? "page" : undefined} key={item.report_id} onClick={() => setSelected(item.report_id)}><strong>{humanizeToken(item.dataset_type)}</strong><span>{formatter.format(new Date(item.created_at))}</span><span>{item.error_count} {t("qa.errors")} · {item.warning_count} {t("qa.warnings")}</span></button>)}</nav>
        <section className="dashboard-detail" aria-live="polite">
          {detail.isPending ? <p role="status">{t("qa.loadingDetail")}</p> : null}
          {detail.isError ? <div className="notice notice--error" role="alert">{serverMessage(detail.error.message)}</div> : null}
          {detail.data ? <>
            <header><div><p className="eyebrow">{humanizeToken(detail.data.summary.dataset_type)}</p><h2>{t("qa.reportTitle")}</h2></div><span className={detail.data.summary.error_count ? "status-badge status-badge--partial" : "status-badge"}>{detail.data.summary.error_count ? t("qa.actionNeeded") : t("qa.noErrors")}</span></header>
            <dl className="metric-grid"><div><dt>{t("qa.images")}</dt><dd>{detail.data.summary.scanned_images}</dd></div><div><dt>{t("qa.masks")}</dt><dd>{detail.data.summary.scanned_masks}</dd></div><div><dt>{t("qa.errors")}</dt><dd>{detail.data.summary.error_count}</dd></div><div><dt>{t("qa.warnings")}</dt><dd>{detail.data.summary.warning_count}</dd></div></dl>
            <dl className="report-provenance"><div><dt>{t("qa.fingerprint")}</dt><dd>{detail.data.summary.dataset_fingerprint}</dd></div><div><dt>{t("qa.created")}</dt><dd>{formatter.format(new Date(detail.data.summary.created_at))}</dd></div></dl>
            <section className="dashboard-section"><h3>{t("qa.checks")}</h3><ul className="check-list">{Object.entries(detail.data.checks).map(([name, status]) => <li key={name}><span>{humanizeToken(name)}</span><strong data-status={status}>{humanizeToken(status)}</strong></li>)}</ul></section>
            <section className="dashboard-section"><h3>{t("qa.issues")}</h3>{detail.data.issues.length ? <ul className="issue-list">{detail.data.issues.map((issue, index) => <li key={`${issue.asset_key}-${issue.code}-${index}`} data-severity={issue.severity}><strong>{humanizeToken(issue.message_key)}</strong><span>{issue.asset_key}</span>{issue.field ? <span>{humanizeToken(issue.field)}</span> : null}</li>)}</ul> : <p>{t("qa.noIssues")}</p>}</section>
            {detail.data.limitations.length ? <section className="dashboard-section"><h3>{t("evaluation.limitations")}</h3><ul>{detail.data.limitations.map((item, index) => <li key={`${item}-${index}`}>{humanizeToken(item)}</li>)}</ul></section> : null}
          </> : null}
        </section>
      </div> : null}
    </main>
  );
}
