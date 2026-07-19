import type {
  Analysis,
  Phase6BModelPredictionSummary,
  Phase6BSourceFamily,
} from "../api/schemas";
import { humanizeToken } from "../display";
import { useI18n, type TranslationKey } from "../i18n";

const statusKeys: Record<Phase6BModelPredictionSummary["status"], TranslationKey> = {
  completed: "ensemble.status.completed",
  skipped: "ensemble.status.skipped",
  disabled: "ensemble.status.disabled",
  failed: "ensemble.status.failed",
  timeout: "ensemble.status.timeout",
};

const familyKeys: Record<Phase6BSourceFamily, TranslationKey> = {
  mp16_family: "ensemble.family.mp16",
  osv5m_family: "ensemble.family.osv5m",
  yfcc_family: "ensemble.family.yfcc",
  inat_family: "ensemble.family.inat",
  textual_evidence_family: "ensemble.family.textual",
  cloud_reasoning_family: "ensemble.family.cloud",
};

function providerName(key: string, prediction: Phase6BModelPredictionSummary): string {
  const value = `${key} ${prediction.provider} ${prediction.model_id}`.toLowerCase();
  if (value.includes("plonk")) return "PLONK";
  if (value.includes("osv5m") || value.includes("osv-5m")) return "OSV-5M";
  if (value.includes("geoclip")) return "GeoCLIP";
  return prediction.provider;
}

function specializationKey(prediction: Phase6BModelPredictionSummary): TranslationKey {
  const model = prediction.model_id.toLowerCase();
  if (model.includes("plonk") && model.includes("osv")) return "ensemble.specialization.street";
  if (model.includes("plonk") && model.includes("inat")) return "ensemble.specialization.nature";
  if (model.includes("plonk") && model.includes("yfcc")) return "ensemble.specialization.mixed";
  if (model.includes("osv")) return "ensemble.specialization.streetModel";
  if (model.includes("geoclip")) return "ensemble.specialization.global";
  return "ensemble.specialization.general";
}

function ModelRuns({ predictions }: { predictions: NonNullable<Analysis["model_predictions"]> }) {
  const { t } = useI18n();
  const entries = Object.entries(predictions).filter((entry): entry is [string, Phase6BModelPredictionSummary] => Boolean(entry[1]));
  if (!entries.length) return null;
  return (
    <section className="ensemble-section" aria-labelledby="ensemble-models-heading">
      <h3 id="ensemble-models-heading">{t("ensemble.models")}</h3>
      <ul className="ensemble-model-list">
        {entries.map(([key, prediction]) => {
          const first = prediction.candidates[0];
          const name = providerName(key, prediction);
          return (
            <li key={key}>
              <div className="ensemble-model-heading">
                <div><strong>{name}</strong><span>{t(specializationKey(prediction))}</span></div>
                <span className={`status-badge ${prediction.status === "completed" ? "" : "status-badge--partial"}`}>{t(statusKeys[prediction.status])}</span>
              </div>
              <dl className="ensemble-model-meta">
                <div><dt>{t("ensemble.model")}</dt><dd>{prediction.model_id}</dd></div>
                <div><dt>{t("ensemble.family")}</dt><dd>{t(familyKeys[prediction.source_family])}</dd></div>
                <div><dt>{t("ensemble.execution")}</dt><dd>{prediction.device} · {prediction.duration_ms} ms</dd></div>
              </dl>
              {prediction.status === "completed" && first ? (
                <p className="ensemble-point">
                  {prediction.score_semantics === "direct_regression"
                    ? t("ensemble.modelPoint", { latitude: first.latitude.toFixed(4), longitude: first.longitude.toFixed(4) })
                    : t("ensemble.candidateCount", { count: prediction.candidates.length })}
                </p>
              ) : null}
              {prediction.reason_code ? <p className="muted">{humanizeToken(prediction.reason_code)}</p> : null}
            </li>
          );
        })}
      </ul>
    </section>
  );
}

function Agreement({ fusion }: { fusion: NonNullable<Analysis["fusion"]> }) {
  const { t } = useI18n();
  const agreement = fusion.agreement_summary;
  const topCluster = fusion.candidate_clusters[0];
  return (
    <section className="ensemble-section" aria-labelledby="ensemble-agreement-heading">
      <h3 id="ensemble-agreement-heading">{t("ensemble.agreement")}</h3>
      <dl className="ensemble-agreement-grid">
        <div><dt>{t("ensemble.independentFamilies")}</dt><dd>{agreement.independent_family_count}</dd></div>
        <div><dt>{t("ensemble.sameFamilySupport")}</dt><dd>{agreement.same_family_duplicate_support}</dd></div>
        <div><dt>{t("ensemble.geographicAgreement")}</dt><dd>{t(agreement.geographic_disagreement ? "ensemble.disagreement" : "ensemble.noStrongDisagreement")}</dd></div>
        <div><dt>{t("ensemble.ocrRelation")}</dt><dd>{t(agreement.ocr_contradiction ? "ensemble.ocrContradiction" : agreement.ocr_agreement ? "ensemble.ocrSupport" : "ensemble.ocrNeutral")}</dd></div>
      </dl>
      {agreement.same_family_duplicate_support > 0 ? <p className="notice notice--warning">{t("ensemble.sameFamilyWarning")}</p> : null}
      <div className="ensemble-family-list" aria-label={t("ensemble.sourceFamilies")}>
        {fusion.source_families.map((family) => <span className="source-badge" key={family}>{t(familyKeys[family])}</span>)}
      </div>
      {topCluster?.contributions.length ? (
        <div className="ensemble-explanation">
          <strong>{t("ensemble.fusedExplanation")}</strong>
          <ul>{topCluster.contributions.map((item) => <li key={`${item.name}-${item.reason}`}>{item.reason}</li>)}</ul>
          <p>{t("ensemble.clusterSpread", { distance: Math.round(topCluster.spread_km), providers: topCluster.provider_count })}</p>
        </div>
      ) : <p className="muted">{t("ensemble.noClusters")}</p>}
    </section>
  );
}

function OCRSummary({ ocr }: { ocr: NonNullable<Analysis["ocr"]> }) {
  const { t } = useI18n();
  return (
    <section className="ensemble-section" aria-labelledby="ensemble-ocr-heading">
      <div className="ensemble-subheading"><h3 id="ensemble-ocr-heading">{t("ensemble.ocrTitle")}</h3><span className="source-badge">{ocr.provider}{ocr.fallback_used ? ` · ${t("ensemble.fallback")}` : ""}</span></div>
      <p className="muted">{t("ensemble.ocrNotTruth")}</p>
      {ocr.detections.length ? <ul className="ocr-detection-list">{ocr.detections.map((item, index) => <li key={`${item.profile}-${index}`}><strong>{item.redacted_text}</strong><span>{humanizeToken(item.script)} · {item.provider}</span></li>)}</ul> : <p>{t("ensemble.noOcrText")}</p>}
      {ocr.place_evidence.length ? <div className="ocr-place-evidence"><strong>{t("ensemble.ocrPlaceEvidence")}</strong><ul>{ocr.place_evidence.map((item, index) => <li key={`${item.normalized_name}-${index}`}>{item.normalized_name}{item.country_code ? ` · ${item.country_code}` : ""} <span>· {humanizeToken(item.match_type)} · {item.source}</span></li>)}</ul></div> : null}
    </section>
  );
}

function CloudAssist({ cloud }: { cloud: NonNullable<Analysis["cloud_assist"]> }) {
  const { locale, t } = useI18n();
  const formatUsd = (value: number, maximumFractionDigits = 6) => new Intl.NumberFormat(locale === "tr" ? "tr-TR" : "en-US", { style: "currency", currency: "USD", minimumFractionDigits: maximumFractionDigits === 2 ? 2 : 0, maximumFractionDigits }).format(value);
  const cost = cloud.estimated_cost_usd === null
    ? t("ensemble.costUnavailable")
    : formatUsd(cloud.estimated_cost_usd);
  const state = !cloud.allowed
    ? t("ensemble.cloudNotAllowed")
    : cloud.cache_hit
      ? t("ensemble.cloudCacheHit")
      : cloud.triggered
        ? t("ensemble.cloudTriggered")
        : t("ensemble.cloudNotTriggered");
  return (
    <section className="ensemble-section cloud-review-section" aria-labelledby="ensemble-cloud-heading">
      <div className="ensemble-subheading"><h3 id="ensemble-cloud-heading">{t("ensemble.cloudTitle")}</h3><span className="status-badge status-badge--partial">{state}</span></div>
      <p className="notice notice--warning">{t("ensemble.cloudNotFact")}</p>
      <dl className="ensemble-model-meta">
        <div><dt>{t("ensemble.model")}</dt><dd>{cloud.model}</dd></div>
        <div><dt>{t("ensemble.promptVersion")}</dt><dd>{cloud.prompt_version}</dd></div>
        <div><dt>{t("ensemble.estimatedCost")}</dt><dd>{cost}</dd></div>
        <div><dt>{t("ensemble.cache")}</dt><dd>{t(cloud.cache_hit ? "ensemble.yes" : "ensemble.no")}</dd></div>
      </dl>
      <p className="muted">{t("ensemble.budgetControlled")}</p>
      {cloud.budget ? <div className="cloud-budget-summary"><p>{t("ensemble.callsToday", { count: cloud.budget.calls_today })}</p><p>{t("ensemble.monthlySpend", { spend: formatUsd(cloud.budget.estimated_month_spend_usd, 2), budget: formatUsd(cloud.budget.configured_monthly_budget_usd, 2) })}</p><p>{t("ensemble.monthlyBudgetRemaining", { amount: formatUsd(cloud.budget.remaining_budget_usd, 2) })}</p></div> : null}
      {cloud.monthly_budget_remaining_usd !== null && cloud.monthly_budget_remaining_usd !== undefined ? <p>{t("ensemble.monthlyBudgetRemaining", { amount: formatUsd(cloud.monthly_budget_remaining_usd, 2) })}</p> : null}
      {cloud.daily_calls_remaining !== null && cloud.daily_calls_remaining !== undefined ? <p>{t("ensemble.dailyCallsRemaining", { count: cloud.daily_calls_remaining })}</p> : null}
      {cloud.reason ?? cloud.reason_code ? <p>{t("ensemble.triggerReason")}: {humanizeToken(cloud.reason ?? cloud.reason_code ?? "")}</p> : null}
      {cloud.review ? <div className="cloud-review-output"><strong>{t("ensemble.reviewDecision")}: {humanizeToken(cloud.review.decision)}</strong><p>{cloud.review.uncertainty_reason}</p>{cloud.review.observed_clues.length ? <ul>{cloud.review.observed_clues.map((clue, index) => <li key={`${clue.type}-${index}`}><span>{humanizeToken(clue.type)}</span>{clue.observation}</li>)}</ul> : null}</div> : null}
    </section>
  );
}

export function Phase6BAnalysisCard({ analysis }: { analysis: Analysis }) {
  const { t } = useI18n();
  if (!analysis.model_predictions && !analysis.fusion && !analysis.ocr && !analysis.cloud_assist) return null;
  return (
    <section className="detail-card phase6b-analysis-card" aria-labelledby="ensemble-heading">
      <header className="ensemble-card-heading"><div><p className="eyebrow">{t("ensemble.eyebrow")}</p><h2 id="ensemble-heading">{t("ensemble.title")}</h2></div><span className="support-category">{t("ensemble.uncalibrated")}</span></header>
      <p className="confidence-disclaimer">{t("ensemble.subtitle")}</p>
      <div className="ensemble-sections">
        {analysis.model_predictions ? <ModelRuns predictions={analysis.model_predictions} /> : null}
        {analysis.fusion ? <Agreement fusion={analysis.fusion} /> : null}
        {analysis.ocr ? <OCRSummary ocr={analysis.ocr} /> : null}
        {analysis.cloud_assist ? <CloudAssist cloud={analysis.cloud_assist} /> : null}
      </div>
    </section>
  );
}
