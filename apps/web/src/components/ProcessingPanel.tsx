import type { Analysis, Progress } from "../api/schemas";
import { useI18n, type TranslationKey } from "../i18n";
import type { AnalysisStreamState } from "../use-analysis";
import { SimulationWatermark } from "./SimulationWatermark";

const stageAliases: Record<string, TranslationKey> = {
  request_validation: "processing.stage.validating",
  authorization: "processing.stage.validating",
  validating: "processing.stage.validating",
  validation: "processing.stage.validating",
  queued: "processing.stage.validating",
  file_signature: "processing.stage.decoding",
  safe_decode: "processing.stage.decoding",
  decoding: "processing.stage.decoding",
  decode: "processing.stage.decoding",
  orientation: "processing.stage.decoding",
  preprocessing: "processing.stage.preprocessing",
  image_metadata: "processing.stage.metadata",
  reading_metadata: "processing.stage.metadata",
  metadata: "processing.stage.metadata",
  hashing: "processing.stage.metadata",
  exif_gps: "processing.stage.exif",
  exif: "processing.stage.exif",
  quality: "processing.stage.quality",
  image_quality: "processing.stage.quality",
  ocr: "processing.stage.ocr",
  extracting_text: "processing.stage.ocr",
  visual_clues: "processing.stage.visualClues",
  extracting_visual_clues: "processing.stage.visualClues",
  segmentation: "processing.stage.segmentation",
  scene_segmentation: "processing.stage.segmentation",
  segmenting_scene: "processing.stage.segmentation",
  global_prediction: "processing.stage.global_prediction",
  global_geolocation: "processing.stage.global_prediction",
  custom_model_shadow: "processing.stage.customShadow",
  retrieval: "processing.stage.reference",
  retrieving_references: "processing.stage.reference",
  resolving_places: "processing.stage.places",
  map: "processing.stage.map",
  map_constraints: "processing.stage.map",
  checking_map_evidence: "processing.stage.map",
  normalization: "processing.stage.normalization",
  candidate_normalization: "processing.stage.normalization",
  clustering: "processing.stage.clustering",
  candidate_clustering: "processing.stage.clustering",
  clustering_candidates: "processing.stage.clustering",
  reverse_geocoding: "processing.stage.reverseGeocoding",
  reverse_geocode: "processing.stage.reverseGeocoding",
  naming_candidate_clusters: "processing.stage.reverseGeocoding",
  reranking: "processing.stage.reranking",
  fusion: "processing.stage.reranking",
  final: "processing.stage.final",
  finalizing: "processing.stage.final",
  confidence_estimation: "processing.stage.confidence",
  estimating_confidence: "processing.stage.confidence",
  candidate_synthesis: "processing.stage.final",
  abstention: "processing.stage.final",
  persistence: "processing.stage.final",
  completed: "processing.stage.completed",
  complete: "processing.stage.completed",
};

function stageKey(stage: string): TranslationKey {
  return stageAliases[stage.toLowerCase()] ?? "processing.stage.unknown";
}

const streamKeys: Record<AnalysisStreamState, TranslationKey> = {
  idle: "processing.polling",
  connecting: "processing.connecting",
  live: "processing.realtime",
  polling: "processing.polling",
  reconnecting: "processing.reconnecting",
};

export function ProcessingPanel({
  progress,
  streamState,
  analysis,
  previewUrl,
  onCancel,
  deleting,
}: {
  progress: Progress | null;
  streamState: AnalysisStreamState;
  analysis?: Analysis;
  previewUrl?: string | null;
  onCancel: () => void;
  deleting: boolean;
}) {
  const { t } = useI18n();
  const stage = stageKey(progress?.stage ?? "queued");
  const percent = progress?.percent ?? 0;
  const streamClass = streamState === "live" ? "live-state" : "live-state live-state--fallback";
  return (
    <main id="main-content" className="processing-shell processing-shell--workspace">
      {analysis?.result_classification === "simulated" ? <SimulationWatermark /> : null}
      <section className="processing-card" aria-labelledby="processing-heading" aria-busy="true">
        <div className="processing-head">
          <div>
            <p className="eyebrow">{t("processing.eyebrow")}</p>
            <h1 id="processing-heading">{t("processing.title")}</h1>
          </div>
          <span className={streamClass} role="status" aria-live="polite">{t(streamKeys[streamState])}</span>
        </div>

        <div className="processing-workspace">
          <section className="processing-source" aria-label={t("processing.sourcePreview")}>
            {previewUrl ? <img src={previewUrl} alt={t("processing.sourcePreviewAlt")} /> : <div className="processing-source__empty" />}
            <p>{t("processing.previewPrivate")}</p>
          </section>
          <section className="processing-status" aria-labelledby="processing-current-stage">
            <p className="eyebrow">{t("processing.currentTask")}</p>
            <h2 id="processing-current-stage" aria-live="polite">{t(stage)}</h2>
            <div
              className="analysis-progress"
              role="progressbar"
              aria-label={t(stage)}
              aria-valuenow={percent}
              aria-valuemin={0}
              aria-valuemax={100}
            >
              <span style={{ width: `${percent}%` }} />
            </div>
            <strong className="progress-percent">{percent}%</strong>
            <dl className="processing-snapshot">
              <div><dt>{t("processing.metadata")}</dt><dd>{analysis?.image ? t("processing.available") : t("processing.pending")}</dd></div>
              <div><dt>{t("processing.evidence")}</dt><dd>{analysis?.evidence.length ?? 0}</dd></div>
              <div><dt>{t("processing.candidates")}</dt><dd>{analysis?.candidates.length ?? 0}</dd></div>
            </dl>
          </section>
        </div>

        <p className="processing-honesty">{t("processing.honesty")}</p>
        <button className="secondary-button danger-button" type="button" disabled={deleting} onClick={onCancel}>
          {t("processing.cancel")}
        </button>
      </section>
    </main>
  );
}
