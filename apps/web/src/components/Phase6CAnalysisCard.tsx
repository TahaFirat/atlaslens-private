import type { Analysis, Phase6CProviderRunSummary } from "../api/schemas";
import { humanizeToken } from "../display";
import { useI18n } from "../i18n";
import { safeOperatorText } from "../safe-display";

function ProviderOutcome({ provider }: { provider: Phase6CProviderRunSummary }) {
  const { t } = useI18n();
  const reason = safeOperatorText(provider.reason_code);
  const successful = provider.status === "completed";
  return (
    <li>
      <div>
        <strong>{safeOperatorText(provider.provider_id) ?? t("system.valueRedacted")}</strong>
        <span>{t("phase6c.duration")}: {provider.duration_ms} ms · {t("phase6c.candidates")}: {provider.candidates_produced}</span>
        {reason ? <span>{t("phase6c.reason")}: {humanizeToken(reason)}</span> : null}
      </div>
      <span className={`status-badge ${successful ? "" : "status-badge--partial"}`} data-status={provider.status}>{humanizeToken(provider.status)}</span>
    </li>
  );
}

export function Phase6CAnalysisCard({ analysis }: { analysis: Analysis }) {
  const { t } = useI18n();
  const phase6c = analysis.phase6c;
  if (!phase6c) return null;
  const attributions = phase6c.reference_attributions
    .map((value) => safeOperatorText(value))
    .filter((value): value is string => value !== null);
  const indexVersion = safeOperatorText(phase6c.reference_index_version);
  return (
    <section className="detail-card phase6c-analysis-card" aria-labelledby="phase6c-analysis-heading">
      <header className="ensemble-card-heading">
        <div><p className="eyebrow">{t("phase6c.eyebrow")}</p><h2 id="phase6c-analysis-heading">{t("phase6c.title")}</h2><p>{t("phase6c.subtitle")}</p></div>
        <span className="status-badge">{phase6c.pipeline_version}</span>
      </header>
      <dl className="phase6c-metrics">
        <div><dt>{t("phase6c.pipeline")}</dt><dd>{phase6c.pipeline_version}</dd></div>
        <div><dt>{t("phase6c.fusion")}</dt><dd>{phase6c.fusion_version}</dd></div>
        <div><dt>{t("phase6c.publicationCandidates")}</dt><dd>{phase6c.publication_candidate_count}</dd></div>
        <div><dt>{t("phase6c.turkiyeSignals")}</dt><dd>{phase6c.turkiye_signal_count}</dd></div>
        <div><dt>{t("phase6c.hierarchyCount")}</dt><dd>{phase6c.hierarchical_candidates.length}</dd></div>
        <div><dt>{t("phase6c.megalocCount")}</dt><dd>{phase6c.megaloc_matches.length}</dd></div>
        <div><dt>{t("phase6c.g3Count")}</dt><dd>{phase6c.g3_scores.length}</dd></div>
        <div><dt>{t("phase6c.leakage")}</dt><dd>{humanizeToken(phase6c.leakage_audit.status)}</dd></div>
      </dl>
      {indexVersion ? <p className="phase6c-index-version"><strong>{t("phase6c.indexVersion")}:</strong> {indexVersion}</p> : null}
      <section aria-labelledby="phase6c-providers-heading">
        <h3 id="phase6c-providers-heading">{t("phase6c.providerRuns")}</h3>
        {phase6c.providers.length ? <ul className="operator-list phase6c-provider-list">{phase6c.providers.map((provider) => <ProviderOutcome provider={provider} key={provider.provider_id} />)}</ul> : <p>{t("phase6c.providersEmpty")}</p>}
      </section>
      <section aria-labelledby="phase6c-fused-heading">
        <h3 id="phase6c-fused-heading">{t("phase6c.fusedCandidates")}</h3>
        {phase6c.fusion_candidates.length ? (
          <ol className="phase6c-fused-list">
            {phase6c.fusion_candidates.slice(0, 8).map((candidate) => (
              <li key={candidate.cluster_id}>
                <span className="candidate-rank" aria-hidden="true">{candidate.final_rank}</span>
                <div>
                  <strong>{t("result.candidate", { rank: candidate.final_rank })}</strong>
                  <span>{t("phase6c.radius")}: {candidate.uncertainty_radius_km.toFixed(1)} km · {t("phase6c.spread")}: {candidate.spread_km.toFixed(1)} km</span>
                  <span>{t("phase6c.independentGroups")}: {candidate.independent_source_family_count} · {t("phase6c.correlatedGroups")}: {candidate.correlated_source_family_count} · {t("phase6c.providersCount")}: {candidate.provider_count}</span>
                  <small>{t("phase6c.notProbability")}</small>
                </div>
              </li>
            ))}
          </ol>
        ) : <p>{t("phase6c.noFused")}</p>}
      </section>
      <section className="phase6c-attributions" aria-labelledby="phase6c-attributions-heading">
        <h3 id="phase6c-attributions-heading">{t("phase6c.attributions")}</h3>
        {attributions.length ? <ul>{attributions.map((attribution) => <li key={attribution}>{attribution}</li>)}</ul> : <p>{t("phase6c.attributionUnavailable")}</p>}
      </section>
      {phase6c.ablations.length ? (
        <section aria-labelledby="phase6c-ablations-heading">
          <h3 id="phase6c-ablations-heading">{t("phase6c.ablations")}</h3>
          <ul className="phase6c-ablation-list">
            {phase6c.ablations.map((ablation) => (
              <li key={ablation.profile_id}>
                <strong>{humanizeToken(ablation.profile_id)}</strong>
                <span>{t("phase6c.candidates")}: {ablation.candidate_count} · {t("phase6c.publicationCandidates")}: {ablation.publication_candidate_count}</span>
                <span>{ablation.abstained ? t("phase6c.abstained") : t("phase6c.returnedCandidates")}</span>
              </li>
            ))}
          </ul>
        </section>
      ) : null}
    </section>
  );
}
