import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import type { AtlasLensApiClient } from "../api/client";
import type { EvaluationReportSummary } from "../api/schemas";
import { humanizeToken } from "../display";
import { useI18n } from "../i18n";

function metric(value: EvaluationReportSummary["country_top1"], unavailable: string, locale: "en" | "tr"): string {
  if (value.value === null || value.denominator === 0) return unavailable;
  return `${new Intl.NumberFormat(locale === "tr" ? "tr-TR" : "en-US", { style: "percent", maximumFractionDigits: 1 }).format(value.value)} (${value.numerator}/${value.denominator})`;
}

export function EvaluationDashboard({ client }: { client: AtlasLensApiClient }) {
  const { locale, t, serverMessage } = useI18n();
  const [selected, setSelected] = useState<string | null>(null);
  const reports = useQuery({ queryKey: ["evaluations"], queryFn: client.listEvaluations });
  const selectedId = selected ?? reports.data?.reports[0]?.report_id ?? null;
  const detail = useQuery({ queryKey: ["evaluation", selectedId], queryFn: () => client.getEvaluation(selectedId as string), enabled: selectedId !== null });
  const report = detail.data;
  return (
    <main id="main-content" className="workspace-shell">
      <header className="workspace-heading"><div><p className="eyebrow">{t("evaluation.eyebrow")}</p><h1>{t("evaluation.title")}</h1><p>{t("evaluation.description")}</p></div></header>
      {reports.isPending ? <p className="workspace-state" role="status">{t("evaluation.loading")}</p> : null}
      {reports.isError ? <div className="workspace-state notice notice--warning" role="alert">{serverMessage(reports.error.message)}</div> : null}
      {reports.data?.reports.length === 0 ? <div className="workspace-state"><h2>{t("evaluation.emptyTitle")}</h2><p>{t("evaluation.empty")}</p></div> : null}
      {reports.data?.reports.length ? <div className="dashboard-layout"><nav className="dashboard-index" aria-label={t("evaluation.reports")}>{reports.data.reports.map((item) => <button type="button" aria-current={item.report_id === selectedId ? "page" : undefined} key={item.report_id} onClick={() => setSelected(item.report_id)}><strong>{item.provider_id}</strong><span>{item.image_count} {t("evaluation.images")}</span><span>{item.calibration_state}</span></button>)}</nav>
        <section className="dashboard-detail" aria-live="polite">
          {detail.isPending ? <p role="status">{t("evaluation.loadingDetail")}</p> : null}
          {detail.isError ? <div className="notice notice--error" role="alert">{serverMessage(detail.error.message)}</div> : null}
          {report ? <>
            <header><div><p className="eyebrow">{report.provider_id}</p><h2>{t("evaluation.reportTitle")}</h2></div><span className="status-badge">{humanizeToken(report.calibration_state)}</span></header>
            <dl className="metric-grid"><div><dt>{t("evaluation.countryTop1")}</dt><dd>{metric(report.country_top1, t("common.unavailable"), locale)}</dd></div><div><dt>{t("evaluation.countryTop5")}</dt><dd>{metric(report.country_top5, t("common.unavailable"), locale)}</dd></div><div><dt>{t("evaluation.regionTop1")}</dt><dd>{metric(report.region_top1, t("common.unavailable"), locale)}</dd></div><div><dt>{t("evaluation.cityTop1")}</dt><dd>{metric(report.city_top1, t("common.unavailable"), locale)}</dd></div><div><dt>{t("evaluation.medianError")}</dt><dd>{report.median_error_km === null || report.median_error_km === undefined ? t("common.unavailable") : `${report.median_error_km.toFixed(1)} km`}</dd></div><div><dt>{t("evaluation.latency")}</dt><dd>{report.latency_median_ms === null || report.latency_median_ms === undefined ? t("common.unavailable") : `${Math.round(report.latency_median_ms)} ms`}</dd></div><div><dt>{t("evaluation.abstention")}</dt><dd>{metric(report.abstention, t("common.unavailable"), locale)}</dd></div><div><dt>{t("evaluation.failures")}</dt><dd>{metric(report.provider_failure, t("common.unavailable"), locale)}</dd></div></dl>
            <dl className="report-provenance"><div><dt>{t("evaluation.fingerprint")}</dt><dd>{report.evaluation_fingerprint}</dd></div><div><dt>{t("evaluation.revision")}</dt><dd>{report.model_revision}</dd></div><div><dt>{t("evaluation.sampleCount")}</dt><dd>{report.image_count}</dd></div></dl>
            <section className="dashboard-section"><h3>{t("evaluation.recall")}</h3><ul>{Object.entries(report.recall_top1).map(([distance, ratio]) => <li key={distance}><span>{humanizeToken(distance)}</span><strong>{metric(ratio, t("common.unavailable"), locale)}</strong></li>)}</ul></section>
            <section className="dashboard-section"><h3>{t("evaluation.limitations")}</h3>{report.limitations.length ? <ul>{report.limitations.map((item, index) => <li key={`${item}-${index}`}>{humanizeToken(item)}</li>)}</ul> : <p>{t("common.none")}</p>}</section>
          </> : null}
        </section>
      </div> : null}
    </main>
  );
}
