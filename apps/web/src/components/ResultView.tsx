import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import type { Analysis, Candidate, Evidence } from "../api/schemas";
import { candidateLabel, copyableCoordinates, displayedCoordinates, displayedRadius, humanizeToken, openStreetMapUrl } from "../display";
import { useI18n, type TranslationKey } from "../i18n";
import { providerWarningKey } from "../provider-warnings";
import { Phase4AssessmentPanel } from "./Phase4AssessmentPanel";
import { Phase5BAssessmentPanel, Phase5BDiagnosticsPanel } from "./Phase5BAssessmentPanel";
import { Phase6BAnalysisCard } from "./Phase6BAnalysisCard";
import { Phase6CAnalysisCard } from "./Phase6CAnalysisCard";
import { SceneAnalysisCard } from "./SceneAnalysisCard";
import { SimulationWatermark } from "./SimulationWatermark";

const ResultMap = lazy(() => import("./ResultMap"));

function percent(value: number | null): number | null {
  return value === null ? null : Math.round(value * 100);
}

const evidenceLabelKeys: Record<string, TranslationKey> = {
  "evidence.embedded_gps": "evidence.label.embeddedGps",
  "evidence.image_quality": "evidence.label.imageQuality",
  "evidence.ocr_redacted_text": "evidence.label.ocrText",
  "evidence.visible_geographic_clues": "evidence.label.visualClues",
  "evidence.global_model_prediction": "evidence.label.globalModel",
};

const evidenceDisplayKeys: Record<string, TranslationKey> = {
  "evidence.embedded_gps_present": "evidence.display.embeddedGps",
  "evidence.image_quality_measured": "evidence.display.imageQuality",
  "evidence.global_model_prediction_present": "evidence.display.globalModel",
};

const candidateTextKeys: Record<string, TranslationKey> = {
  "evidence.global_model_prediction_unverified": "result.modelEvidenceSummary",
  "phase5.topk_geodesic_dispersion_with_750km_floor": "result.modelUncertaintyBasis",
  "phase5.no_confidence_before_calibration": "result.modelConfidenceBasis",
  "phase5.uncalibrated_model_score_not_confidence": "result.modelConfidenceBasis",
};

const modelLimitationKeys: Record<string, TranslationKey> = {
  model_prediction_is_unverified: "result.modelLimitation.unverified",
  "phase5.model_prediction_is_unverified": "result.modelLimitation.unverified",
  gallery_softmax_is_not_calibrated_probability: "result.modelLimitation.notProbability",
  fixed_gallery_limits_geographic_resolution: "result.modelLimitation.fixedGallery",
  broad_global_model_not_street_level_proof: "result.modelLimitation.notStreetLevel",
  "phase5.gallery_softmax_is_not_geographic_probability": "result.modelLimitation.notGeographicProbability",
  "phase5.minimum_model_only_radius_750km": "result.modelLimitation.radiusFloor",
};

function granularityKey(candidate: Candidate): TranslationKey {
  const valid = ["exact_metadata", "city", "region", "country", "broad_area"].includes(candidate.granularity);
  return valid ? (`result.granularity.${candidate.granularity}` as TranslationKey) : "result.granularity.unknown";
}

function verificationKey(candidate: Candidate): TranslationKey {
  const valid = ["metadata_only", "unverified_model", "corroborated", "geometrically_verified"].includes(candidate.verification_status);
  return valid ? (`result.verification.${candidate.verification_status}` as TranslationKey) : "result.verification.unknown";
}

function supportKey(candidate: Candidate): TranslationKey {
  if (candidate.confidence_kind === "calibrated_probability") return "result.support.calibrated";
  if (candidate.confidence_kind === "source_reliability") return "result.support.source";
  return "result.support.uncalibrated";
}

function SourceBadges({ candidate }: { candidate: Candidate }) {
  const sources = [...new Set(candidate.provenance.map((item) => item.provider_id))];
  return <span className="source-badge-list">{sources.map((source) => <span className="source-badge" key={source}>{source}</span>)}</span>;
}

function EvidenceCard({ evidence }: { evidence: Evidence }) {
  const { t } = useI18n();
  const knownTypes = ["exif", "ocr", "visual_clue", "quality", "system", "global_model_prediction"];
  const typeKey = knownTypes.includes(evidence.type) ? (`evidence.${evidence.type}` as TranslationKey) : "evidence.unknown";
  const labelKey = evidenceLabelKeys[evidence.label] ?? "evidence.unknown";
  const displayKey = evidence.display_value ? evidenceDisplayKeys[evidence.display_value] : undefined;
  const basisKey = candidateTextKeys[evidence.confidence_basis];
  return (
    <li className="evidence-item">
      <div className="evidence-topline"><span className="evidence-type">{t(typeKey)}</span><span>{t(evidence.confidence === null ? "evidence.supportUnavailable" : "evidence.supportAvailable")}</span></div>
      <strong>{t(labelKey)}</strong>
      <p>{evidence.sensitive ? t("evidence.sensitive") : displayKey ? t(displayKey) : (evidence.display_value ?? "—")}</p>
      <dl className="inline-definition">
        <div><dt>{t("result.source")}</dt><dd>{evidence.source}</dd></div>
        <div><dt>{t("result.confidenceBasis")}</dt><dd>{basisKey ? t(basisKey) : t("result.support.explained")}</dd></div>
        <div><dt>{t("result.provenance")}</dt><dd>{evidence.provenance.provider_id} · {evidence.provenance.execution_boundary}</dd></div>
      </dl>
    </li>
  );
}

function CandidateSummary({ candidate, selected, onSelect }: { candidate: Candidate; selected: boolean; onSelect: () => void }) {
  const { t } = useI18n();
  const label = candidateLabel(candidate, t("result.candidate", { rank: candidate.rank }));
  const diversity = candidate.phase5b_assessment?.provider_diversity ?? new Set(candidate.provenance.map((item) => item.provider_id)).size;
  return (
    <li>
      <button className="candidate-summary" type="button" aria-pressed={selected} onClick={onSelect}>
        <span className="candidate-rank" aria-hidden="true">{candidate.rank}</span>
        <span><strong>{label}</strong><small>{t(granularityKey(candidate))} · {t(supportKey(candidate))}</small><small>{t("result.radius", { radius: displayedRadius(candidate.radius_km) })} · {diversity} {t("result.providersShort")}</small>{candidate.geoclip_cluster ? <small>{t("result.clusterMembers", { count: candidate.geoclip_cluster.member_count })}</small> : null}<SourceBadges candidate={candidate} /></span>
      </button>
    </li>
  );
}

function CandidateInspector({ candidate, comparisons }: { candidate: Candidate; comparisons: Analysis["provider_comparisons"] }) {
  const { locale, t } = useI18n();
  const [copied, setCopied] = useState(false);
  const model = candidate.model_prediction;
  const cluster = candidate.geoclip_cluster;
  const confidenceAssessment = candidate.confidence_assessment;
  const label = candidateLabel(candidate, t("result.candidate", { rank: candidate.rank }));
  const evidenceSummaryKey = candidateTextKeys[candidate.evidence_summary];
  const uncertaintyBasisKey = candidateTextKeys[candidate.uncertainty_basis];
  const confidenceBasisKey = candidateTextKeys[candidate.confidence_basis];
  const supports = candidate.phase5b_assessment?.supports ?? [];
  const contradictions = candidate.phase5b_assessment?.contradictions ?? candidate.phase4_assessment?.contradictions ?? [];
  const diversity = candidate.phase5b_assessment?.provider_diversity ?? new Set(candidate.provenance.map((item) => item.provider_id)).size;
  const copyCoordinates = async () => {
    try {
      await navigator.clipboard?.writeText(copyableCoordinates(candidate));
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      setCopied(false);
    }
  };
  return (
    <article className="candidate-inspector" aria-labelledby={`candidate-${candidate.id}`}>
      <header className="candidate-heading">
        <div><span className="candidate-label">{t("result.primaryHypothesis", { rank: candidate.rank })}</span><h2 id={`candidate-${candidate.id}`}>{label}</h2></div>
        <span className="support-category">{t(supportKey(candidate))}</span>
      </header>
      <SourceBadges candidate={candidate} />
      <p className="confidence-disclaimer">{t("result.notCertainty")}</p>
      <dl className="primary-assessment">
        <div><dt>{t("result.hypothesisCenter")}</dt><dd>{displayedCoordinates(candidate)}</dd></div>
        <div><dt>{t("result.radiusLabel")}</dt><dd>{t("result.radius", { radius: displayedRadius(candidate.radius_km) })}</dd></div>
        <div><dt>{t("result.verification")}</dt><dd>{t(verificationKey(candidate))}</dd></div>
        <div><dt>{t("result.providerDiversity")}</dt><dd>{diversity}</dd></div>
        {cluster ? <div><dt>{t("result.clusterSize")}</dt><dd>{t("result.clusterMembers", { count: cluster.member_count })}</dd></div> : null}
        {confidenceAssessment ? <div><dt>{t("result.confidenceLevel")}</dt><dd>{t(`result.confidenceLabel.${confidenceAssessment.label}` as TranslationKey)}</dd></div> : null}
        <div><dt>{t("result.calibrationState")}</dt><dd>{confidenceAssessment ? t("result.notCalibrated") : model ? t(`result.calibration.${model.calibration_state}` as TranslationKey) : t(supportKey(candidate))}</dd></div>
      </dl>
      {confidenceAssessment ? <div className="notice notice--warning confidence-assessment"><strong>{t("result.confidenceLevel")}: {t(`result.confidenceLabel.${confidenceAssessment.label}` as TranslationKey)}</strong><span>{t("result.confidenceNotProbability")}</span><ul>{confidenceAssessment.basis.map((basis) => <li key={basis}>{humanizeToken(basis)}</li>)}</ul></div> : null}
      <div className="candidate-actions">
        <button className="secondary-button" type="button" onClick={() => void copyCoordinates()}>{t("result.copyCoordinates")}</button>
        <a className="secondary-button" href={openStreetMapUrl(candidate)} target="_blank" rel="noreferrer">{t("result.openOsm")}</a>
        <span role="status" aria-live="polite">{copied ? t("result.coordinatesCopied") : ""}</span>
      </div>
      {candidate.reverse_geocode ? <section className="reverse-geocode-summary" aria-label={t("result.coordinateName")}><h3>{t("result.coordinateName")}</h3><strong>{candidate.reverse_geocode.display_name}</strong><span>{candidate.reverse_geocode.provider} · {candidate.reverse_geocode.dataset_version} · {candidate.reverse_geocode.license}</span><p>{t("result.reverseGeocodeNeutral")}</p></section> : null}
      <div className="assessment-columns">
        <section><h3>{t("result.supportingEvidence")}</h3>{supports.length ? <ul>{supports.map((item, index) => <li key={`${item}-${index}`}>{humanizeToken(item)}</li>)}</ul> : <p>{t("result.noIndependentSupport")}</p>}</section>
        <section><h3>{t("result.contradictions")}</h3>{contradictions.length ? <ul>{contradictions.map((item, index) => <li key={`${item}-${index}`}>{humanizeToken(item)}</li>)}</ul> : <p>{t("common.none")}</p>}</section>
      </div>
      {comparisons?.length ? <section className="comparison-summary"><h3>{t("result.modelDisagreement")}</h3><ul>{comparisons.map((item) => <li key={`${item.provider_id}-${item.mode}`}><strong>{item.provider_id}</strong><span>{humanizeToken(item.mode)} · {humanizeToken(item.status)}{item.distance_to_primary_km !== null && item.distance_to_primary_km !== undefined ? ` · ${Math.round(item.distance_to_primary_km)} km` : ""}</span></li>)}</ul></section> : null}
      {candidate.granularity === "exact_metadata" ? <div className="notice notice--warning exif-notice">{t("result.exifWarning")}</div> : null}

      <details className="advanced-details">
        <summary>{t("result.advancedDiagnostics")}</summary>
        {model ? <section className="model-diagnostics" aria-label={t("result.modelDiagnostics") }>
          <div className="notice notice--warning"><strong>{t("result.modelUnverified")}</strong><span>{t("result.modelScore", { score: model.raw_score.toFixed(6) })}</span><span>{t("result.modelScoreWarning")}</span></div>
          <dl className="candidate-details"><div><dt>{t("result.modelProvider")}</dt><dd>{model.provider_id}</dd></div><div><dt>{t("result.modelName")}</dt><dd>{model.model_name} · {model.model_revision}</dd></div><div><dt>{t("result.modelDevice")}</dt><dd>{model.device} · {model.dtype}</dd></div><div><dt>{t("result.modelRuntime")}</dt><dd>{model.inference_ms} ms</dd></div><div><dt>{t("result.modelOriginalRank")}</dt><dd>{model.original_rank}</dd></div></dl>
          {model.place_label && model.place_label.source !== "coordinate_fallback" ? <div className="place-label-diagnostics"><strong>{t("result.placeLabel")}</strong><dl className="candidate-details"><div><dt>{t("result.placeCountry")}</dt><dd>{model.place_label.country ?? t("result.placeUnavailable")}</dd></div><div><dt>{t("result.placeRegion")}</dt><dd>{model.place_label.region ?? t("result.placeUnavailable")}</dd></div><div><dt>{t("result.placeCity")}</dt><dd>{model.place_label.city ?? t("result.placeUnavailable")}</dd></div><div><dt>{t("result.placeDistance")}</dt><dd>{model.place_label.distance_to_place_km.toFixed(1)} km</dd></div><div><dt>{t("result.placeDataset")}</dt><dd>{model.place_label.source} · {model.place_label.dataset_version}</dd></div></dl></div> : null}
          <div className="provenance-block"><strong>{t("result.modelLimitations")}</strong><ul>{model.limitations.map((limitation) => <li key={limitation}>{t(modelLimitationKeys[limitation] ?? "result.modelLimitation.unknown")}</li>)}</ul></div>
        </section> : null}
        {cluster ? <section className="cluster-diagnostics" aria-label={t("result.clusterDiagnostics")}><h3>{t("result.clusterDiagnostics")}</h3><p className="notice notice--warning">{t("result.clusterScoreWarning")}</p><dl className="candidate-details"><div><dt>{t("result.clusterSize")}</dt><dd>{cluster.member_count}</dd></div><div><dt>{t("result.clusterRanks")}</dt><dd>{cluster.member_ranks.join(", ")}</dd></div><div><dt>{t("result.clusterMaxSimilarity")}</dt><dd>{cluster.max_raw_similarity.toFixed(6)}</dd></div><div><dt>{t("result.clusterMeanSimilarity")}</dt><dd>{cluster.mean_raw_similarity.toFixed(6)}</dd></div><div><dt>{t("result.clusterSupport")}</dt><dd>{cluster.cluster_support.toFixed(6)}</dd></div></dl></section> : null}
        <dl className="candidate-details"><div><dt>{t("result.evidenceSummary")}</dt><dd>{evidenceSummaryKey ? t(evidenceSummaryKey) : t("result.support.explained")}</dd></div><div><dt>{t("result.uncertaintyBasis")}</dt><dd>{uncertaintyBasisKey ? t(uncertaintyBasisKey) : t("result.uncertaintyExplained")}</dd></div><div><dt>{t("result.confidenceBasis")}</dt><dd>{confidenceBasisKey ? t(confidenceBasisKey) : t("result.support.explained")}</dd></div><div><dt>{t("result.source")}</dt><dd>{candidate.source}</dd></div></dl>
        <div className="provenance-block"><strong>{t("result.provenance")}</strong><ul>{candidate.provenance.map((item, index) => <li key={`${item.provider_id}-${index}`}>{t("result.provider", { provider: item.provider_id, boundary: item.execution_boundary, version: item.provider_version })}</li>)}</ul></div>
        {candidate.phase5b_assessment ? <Phase5BAssessmentPanel assessment={candidate.phase5b_assessment} locale={locale} headingId={`phase5b-assessment-${candidate.id}`} /> : candidate.phase4_assessment ? <Phase4AssessmentPanel assessment={candidate.phase4_assessment} locale={locale} headingId={`phase4-assessment-${candidate.id}`} /> : null}
      </details>
    </article>
  );
}

export function ResultView({ analysis, onDelete, deleting, onNew }: { analysis: Analysis; onDelete: () => void; deleting: boolean; onNew?: () => void }) {
  const { locale, t } = useI18n();
  const orderedCandidates = useMemo(() => [...analysis.candidates].sort((a, b) => a.rank - b.rank), [analysis.candidates]);
  const [selectedCandidateId, setSelectedCandidateId] = useState<string | null>(orderedCandidates[0]?.id ?? null);
  useEffect(() => {
    if (!orderedCandidates.some((candidate) => candidate.id === selectedCandidateId)) setSelectedCandidateId(orderedCandidates[0]?.id ?? null);
  }, [orderedCandidates, selectedCandidateId]);
  const selectedCandidate = orderedCandidates.find((candidate) => candidate.id === selectedCandidateId) ?? orderedCandidates[0];
  const partial = analysis.warnings.length > 0;
  const dateFormatter = useMemo(() => new Intl.DateTimeFormat(locale === "tr" ? "tr-TR" : "en-US", { dateStyle: "medium", timeStyle: "short" }), [locale]);
  return (
    <main id="main-content" className="result-shell">
      {analysis.result_classification === "simulated" ? <SimulationWatermark /> : null}
      <header className="result-header">
        <div><p className="eyebrow">{t("result.eyebrow")}</p><h1>{t("result.title")}</h1><div className="result-meta"><span className={`status-badge ${partial ? "status-badge--partial" : ""}`}>{t(partial ? "result.partial" : "result.completed")}</span><span>{t("result.mode")}: {t(analysis.analysis_mode === "local_only" ? "mode.local.title" : "mode.cloud.title")}</span><span>{t("result.created")}: {dateFormatter.format(new Date(analysis.created_at))}</span>{analysis.expires_at ? <span>{t("result.expires")}: {dateFormatter.format(new Date(analysis.expires_at))}</span> : null}</div></div>
        <div className="result-header__actions">{onNew ? <button className="secondary-button" type="button" onClick={onNew}>{t("result.new")}</button> : null}<button className="secondary-button danger-button" type="button" disabled={deleting} onClick={onDelete}>{t(deleting ? "result.deleting" : "result.delete")}</button></div>
      </header>
      <div className="result-grid">
        <Suspense fallback={<section className="map-section" aria-label={t("result.mapTitle")}><div className="map-heading"><p className="eyebrow">{t("result.mapTitle")}</p></div><div className="map-frame-wrap"><div className="map-frame" data-testid="map-region" role="region" aria-label={t("result.mapTitle")} aria-busy="true" /><div className="map-overlay" role="status">{t("result.mapLoading")}</div></div></section>}>
          <ResultMap candidates={orderedCandidates} phase6c={analysis.phase6c} modelPredictions={analysis.model_predictions} selectedCandidateId={selectedCandidateId} onSelectCandidate={setSelectedCandidateId} />
        </Suspense>
        <section className="candidate-workspace" aria-label={t("result.title")}>
          <ol className="candidate-summary-list">{orderedCandidates.map((candidate) => <CandidateSummary candidate={candidate} selected={candidate.id === selectedCandidateId} onSelect={() => setSelectedCandidateId(candidate.id)} key={candidate.id} />)}</ol>
          {selectedCandidate ? <CandidateInspector candidate={selectedCandidate} comparisons={analysis.provider_comparisons} /> : null}
        </section>
      </div>
      <div className="detail-grid">
        <Phase6CAnalysisCard analysis={analysis} />
        <Phase6BAnalysisCard analysis={analysis} />
        {analysis.phase5b_diagnostics ? <Phase5BDiagnosticsPanel diagnostics={analysis.phase5b_diagnostics} locale={locale} /> : null}
        {analysis.scene_analysis ? <SceneAnalysisCard scene={analysis.scene_analysis} /> : null}
        <section className="detail-card" aria-labelledby="evidence-heading"><h2 id="evidence-heading">{t("evidence.title")}</h2>{analysis.evidence.length > 0 ? <ul className="evidence-list">{analysis.evidence.map((item) => <EvidenceCard evidence={item} key={item.id} />)}</ul> : <p className="muted">{t("evidence.none")}</p>}</section>
        <section className="detail-card" aria-labelledby="warnings-heading"><h2 id="warnings-heading">{t("warnings.title")}</h2>{analysis.warnings.length > 0 ? <ul className="warning-list">{analysis.warnings.map((warning, index) => <li key={`${warning}-${index}`}>{t(providerWarningKey(warning))}</li>)}</ul> : <p className="muted">{t("warnings.none")}</p>}<h2 className="subheading">{t("quality.title")}</h2>{analysis.quality ? <dl className="quality-grid">{([ ["quality.blur", analysis.quality.blur_score], ["quality.brightness", analysis.quality.brightness_score], ["quality.contrast", analysis.quality.contrast_score], ["quality.resolution", analysis.quality.resolution_score] ] as const).map(([label, score]) => <div key={label}><dt>{t(label)}</dt><dd>{percent(score)}%</dd><span><i style={{ width: `${percent(score)}%` }} /></span></div>)}</dl> : <p className="muted">{t("quality.notAvailable")}</p>}{analysis.image ? <div className="image-summary"><h2 className="subheading">{t("image.title")}</h2><p>{t("image.dimensions", { width: analysis.image.width, height: analysis.image.height, megapixels: analysis.image.megapixels.toFixed(1), format: analysis.image.format.toUpperCase() })}</p><span>{t(analysis.image.exif_present ? "image.exifPresent" : "image.exifAbsent")}</span></div> : null}</section>
      </div>
    </main>
  );
}
