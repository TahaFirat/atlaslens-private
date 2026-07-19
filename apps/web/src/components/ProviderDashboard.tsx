import { useQuery } from "@tanstack/react-query";
import type {
  SystemIntelligenceModelCard,
  SystemIntelligenceReferenceIndexCard,
} from "../api/schemas";
import type { AtlasLensApiClient } from "../api/client";
import { humanizeToken } from "../display";
import { useI18n, type TranslationKey } from "../i18n";
import { safeOperatorText, safeRepositoryUrl } from "../safe-display";

function byteSize(value: number): string {
  if (value < 1_024) return `${value} B`;
  if (value < 1_048_576) return `${(value / 1_024).toFixed(1)} KiB`;
  if (value < 1_073_741_824) return `${(value / 1_048_576).toFixed(1)} MiB`;
  return `${(value / 1_073_741_824).toFixed(2)} GiB`;
}

function BooleanState({ value }: { value: boolean | null | undefined }) {
  const { t } = useI18n();
  return <>{value === null ? t("system.notReported") : t(value ? "ensemble.yes" : "ensemble.no")}</>;
}

function ModelCard({ model }: { model: SystemIntelligenceModelCard }) {
  const { locale, t } = useI18n();
  const ready = model.enabled && model.available && model.status === "ready";
  const name = safeOperatorText(model.display_name) ?? t("system.valueRedacted");
  const purpose = safeOperatorText(model.purpose) ?? t("system.valueRedacted");
  const repository = safeRepositoryUrl(model.repository_url);
  const sourceRevision = safeOperatorText(model.source_revision);
  const modelRevision = safeOperatorText(model.model_revision);
  const runtimeModel = safeOperatorText(model.runtime_model_id);
  const device = safeOperatorText(model.device);
  const license = safeOperatorText(model.license);
  const errorCode = safeOperatorText(model.error_code);
  const lastSuccess = model.last_success_at
    ? new Intl.DateTimeFormat(locale === "tr" ? "tr-TR" : "en-US", { dateStyle: "medium", timeStyle: "short" }).format(new Date(model.last_success_at))
    : null;

  return (
    <article className="system-model-card">
      <header>
        <div><h3>{name}</h3><span>{humanizeToken(model.current_participation)}</span></div>
        <span className={`status-badge ${ready ? "" : "status-badge--partial"}`} data-status={model.status}>
          {t(`capabilities.status.${model.status}` as TranslationKey)}
        </span>
      </header>
      <p>{purpose}</p>
      <dl className="system-fact-grid">
        <div><dt>{t("system.enabled")}</dt><dd>{t(model.enabled ? "ensemble.yes" : "ensemble.no")}</dd></div>
        <div><dt>{t("system.available")}</dt><dd>{t(model.available ? "ensemble.yes" : "ensemble.no")}</dd></div>
        <div><dt>{t("system.execution")}</dt><dd>{humanizeToken(model.execution_mode)}</dd></div>
        <div><dt>{t("system.device")}</dt><dd>{device ?? t("system.notReported")}</dd></div>
        <div><dt>{t("system.installed")}</dt><dd><BooleanState value={model.installed} /></dd></div>
        <div><dt>{t("system.weights")}</dt><dd><BooleanState value={model.weights_available} /></dd></div>
        <div><dt>{t("system.worker")}</dt><dd><BooleanState value={model.worker_reachable} /></dd></div>
        <div><dt>{t("system.loaded")}</dt><dd><BooleanState value={model.model_loaded} /></dd></div>
        <div><dt>{t("system.loadVerified")}</dt><dd><BooleanState value={model.load_verified} /></dd></div>
        <div><dt>{t("system.inferenceVerified")}</dt><dd><BooleanState value={model.real_inference_verified} /></dd></div>
        {model.last_latency_ms !== null ? <div><dt>{t("system.lastLatency")}</dt><dd>{model.last_latency_ms} ms</dd></div> : null}
        {lastSuccess ? <div><dt>{t("system.lastSuccess")}</dt><dd>{lastSuccess}</dd></div> : null}
      </dl>
      {(runtimeModel || modelRevision || sourceRevision || license || repository || errorCode) ? (
        <details>
          <summary>{t("system.provenance")}</summary>
          <dl className="system-provenance">
            {runtimeModel ? <div><dt>{t("system.runtimeModel")}</dt><dd>{runtimeModel}</dd></div> : null}
            {modelRevision ? <div><dt>{t("system.modelRevision")}</dt><dd>{modelRevision}</dd></div> : null}
            {sourceRevision ? <div><dt>{t("system.sourceRevision")}</dt><dd>{sourceRevision}</dd></div> : null}
            {license ? <div><dt>{t("system.license")}</dt><dd>{license}</dd></div> : null}
            {errorCode ? <div><dt>{t("system.reason")}</dt><dd>{humanizeToken(errorCode)}</dd></div> : null}
          </dl>
          {repository ? <a href={repository} target="_blank" rel="noreferrer">{t("system.repository")}</a> : null}
        </details>
      ) : null}
    </article>
  );
}

function Distribution({ title, values }: { title: string; values: Record<string, number> | null | undefined }) {
  const { t } = useI18n();
  if (!values || Object.keys(values).length === 0) return <p>{t("system.distributionUnavailable")}</p>;
  const entries = Object.entries(values)
    .filter(([key]) => safeOperatorText(key) !== null)
    .sort(([left], [right]) => left.localeCompare(right))
    .slice(0, 100);
  return (
    <details className="system-distribution">
      <summary>{title} ({entries.length})</summary>
      <dl>{entries.map(([label, count]) => <div key={label}><dt>{label}</dt><dd>{count}</dd></div>)}</dl>
    </details>
  );
}

function ReferenceIndexCard({ index }: { index: SystemIntelligenceReferenceIndexCard }) {
  const { t } = useI18n();
  const usable = index.enabled && index.status === "ready" && index.leakage_status === "passed";
  const reason = safeOperatorText(index.reason_code);
  const version = safeOperatorText(index.index_version);
  const descriptor = safeOperatorText(index.descriptor_version);
  const attributions = (index.attributions ?? []).map((value) => safeOperatorText(value)).filter((value): value is string => value !== null);
  return (
    <section className="detail-card system-index-card" aria-labelledby="system-index-heading">
      <header>
        <div><p className="eyebrow">{t("system.dataEyebrow")}</p><h2 id="system-index-heading">{t("system.indexTitle")}</h2></div>
        <span className={`status-badge ${usable ? "" : "status-badge--partial"}`} data-status={index.status}>{humanizeToken(index.status)}</span>
      </header>
      <p>{t("system.indexDescription")}</p>
      {index.leakage_status !== "passed" ? <div className="notice notice--warning" role="status">{t("system.leakageBlocked", { status: humanizeToken(index.leakage_status ?? "not_run") })}</div> : null}
      <dl className="system-index-metrics">
        <div><dt>{t("system.health")}</dt><dd>{humanizeToken(index.health)}</dd></div>
        <div><dt>{t("system.leakage")}</dt><dd>{humanizeToken(index.leakage_status ?? "not_run")}</dd></div>
        <div><dt>{t("system.references")}</dt><dd>{index.count}</dd></div>
        <div><dt>{t("system.sequences")}</dt><dd>{index.sequences}</dd></div>
        <div><dt>{t("system.countries")}</dt><dd>{index.countries ?? t("system.notReported")}</dd></div>
        <div><dt>{t("system.provinces")}</dt><dd>{index.provinces ?? t("system.notReported")}</dd></div>
        <div><dt>{t("system.diskUsage")}</dt><dd>{byteSize(index.disk_usage_bytes)}</dd></div>
        <div><dt>{t("system.excluded")}</dt><dd>{index.excluded ?? t("system.notReported")}</dd></div>
        <div><dt>{t("system.duplicates")}</dt><dd>{index.duplicates ?? t("system.notReported")}</dd></div>
        {index.leakage_audit ? <div><dt>{t("system.auditChecked")}</dt><dd>{index.leakage_audit.checked_reference_count}</dd></div> : null}
        {index.leakage_audit ? <div><dt>{t("system.descriptorChecked")}</dt><dd>{index.leakage_audit.descriptor_checked_count}</dd></div> : null}
      </dl>
      {reason ? <p className="system-reason"><strong>{t("system.reason")}:</strong> {humanizeToken(reason)}</p> : null}
      {(version || descriptor) ? <p className="system-versions">{version ? `${t("system.indexVersion")}: ${version}` : ""}{version && descriptor ? " · " : ""}{descriptor ? `${t("system.descriptorVersion")}: ${descriptor}` : ""}</p> : null}
      <div className="system-distributions">
        <Distribution title={t("system.sourceDistribution")} values={index.source_distribution} />
        <Distribution title={t("system.provinceDistribution")} values={index.images_per_province} />
      </div>
      <div className="system-index-attributions">
        <h3>{t("system.attributions")}</h3>
        {attributions.length ? <ul>{attributions.map((value) => <li key={value}>{value}</li>)}</ul> : <p>{t("system.attributionUnavailable")}</p>}
      </div>
    </section>
  );
}

export function ProviderDashboard({ client }: { client: AtlasLensApiClient }) {
  const { t, serverMessage } = useI18n();
  const intelligence = useQuery({
    queryKey: ["system-intelligence"],
    queryFn: client.getSystemIntelligence,
    staleTime: 15_000,
    retry: false,
  });
  return (
    <main id="main-content" className="workspace-shell">
      <header className="workspace-heading">
        <div><p className="eyebrow">{t("system.eyebrow")}</p><h1>{t("system.title")}</h1><p>{t("system.description")}</p></div>
        {intelligence.data ? <span className="status-badge">{intelligence.data.active_pipeline_version}</span> : null}
      </header>
      {intelligence.isPending ? <section className="detail-card"><p role="status">{t("system.loading")}</p></section> : null}
      {intelligence.isError ? <section className="detail-card"><div className="notice notice--warning" role="alert">{serverMessage(intelligence.error.message)}</div><p>{t("system.errorHelp")}</p></section> : null}
      {intelligence.data ? (
        <>
          <ReferenceIndexCard index={intelligence.data.reference_index} />
          <section className="system-model-section" aria-labelledby="system-models-heading">
            <div className="section-heading"><div><p className="eyebrow">{t("system.modelsEyebrow")}</p><h2 id="system-models-heading">{t("system.modelsTitle")}</h2></div></div>
            <div className="system-model-grid">{intelligence.data.models.map((model) => <ModelCard model={model} key={model.model_id} />)}</div>
          </section>
        </>
      ) : null}
      <p className="workspace-footnote">{t("system.operatorOnly")}</p>
    </main>
  );
}
