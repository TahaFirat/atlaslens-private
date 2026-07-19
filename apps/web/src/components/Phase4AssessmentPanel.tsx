import type { Phase4AssessmentView } from "../phase4-view";

type Locale = "en" | "tr";

const copy = {
  en: {
    title: "Candidate assessment",
    score: "Relative rank score",
    warning: "This uncalibrated relative score is not a probability or a location guarantee.",
    breakdown: "Score breakdown",
    diversity: "Independent sources",
    map: "Map observations",
    geometry: "Geometry diagnostics",
    contradictions: "Contradictions",
    references: "Reference attribution",
    denied: "Reference display denied by policy",
    allowed: "Reference display permitted; no asset is embedded in this panel",
    limitations: "Limitations",
    none: "None reported",
    matches: "{inliers} inliers from {matches} filtered matches",
  },
  tr: {
    title: "Aday değerlendirmesi",
    score: "Göreli sıralama puanı",
    warning: "Bu kalibre edilmemiş göreli puan, olasılık veya konum garantisi değildir.",
    breakdown: "Puan dökümü",
    diversity: "Bağımsız kaynaklar",
    map: "Harita gözlemleri",
    geometry: "Geometri tanıları",
    contradictions: "Çelişkiler",
    references: "Referans atfı",
    denied: "Referans gösterimi ilke gereği engellendi",
    allowed: "Referans gösterimine izin verildi; bu panel görüntü içermez",
    limitations: "Sınırlamalar",
    none: "Bildirilmedi",
    matches: "{matches} filtrelenmiş eşleşmeden {inliers} iç eşleşme",
  },
} as const;

const phase5DiagnosticCopy: Record<string, Record<Locale, string>> = {
  model_only: {
    en: "Model-only ranking",
    tr: "Yaln\u0131zca modele dayal\u0131 s\u0131ralama",
  },
  global_model_gallery_softmax: {
    en: "Global model gallery score",
    tr: "K\u00fcresel model galeri puan\u0131",
  },
  "phase5.raw_uncalibrated_model_rank_feature": {
    en: "Raw uncalibrated model-ranking feature",
    tr: "Ham, kalibre edilmemi\u015f model s\u0131ralama \u00f6zelli\u011fi",
  },
  model_prediction_is_unverified: {
    en: "The model hypothesis is unverified.",
    tr: "Model hipotezi do\u011frulanmam\u0131\u015ft\u0131r.",
  },
  "phase5.model_prediction_is_unverified": {
    en: "The model hypothesis is unverified.",
    tr: "Model hipotezi do\u011frulanmam\u0131\u015ft\u0131r.",
  },
  gallery_softmax_is_not_calibrated_probability: {
    en: "The gallery score is not a calibrated probability.",
    tr: "Galeri puan\u0131 kalibre edilmi\u015f bir olas\u0131l\u0131k de\u011fildir.",
  },
  fixed_gallery_limits_geographic_resolution: {
    en: "The fixed gallery limits geographic resolution.",
    tr: "Sabit galeri co\u011frafi \u00e7\u00f6z\u00fcn\u00fcrl\u00fc\u011f\u00fc s\u0131n\u0131rlar.",
  },
  broad_global_model_not_street_level_proof: {
    en: "The broad global model is not street-level proof.",
    tr: "Geni\u015f kapsaml\u0131 k\u00fcresel model sokak d\u00fczeyinde kan\u0131t de\u011fildir.",
  },
  "phase5.gallery_softmax_is_not_geographic_probability": {
    en: "The gallery score is not a geographic probability.",
    tr: "Galeri puan\u0131 co\u011frafi bir olas\u0131l\u0131k de\u011fildir.",
  },
  "phase5.minimum_model_only_radius_750km": {
    en: "Model-only uncertainty has a 750 km minimum radius.",
    tr: "Yaln\u0131zca modele dayal\u0131 belirsizli\u011fin yar\u0131\u00e7ap\u0131 en az 750 km'dir.",
  },
};

function diagnosticText(value: string, locale: Locale): string {
  return phase5DiagnosticCopy[value]?.[locale] ?? value.replaceAll(/[._]/g, " ");
}

function interpolate(value: string, values: Record<string, string | number>): string {
  return Object.entries(values).reduce(
    (current, [key, replacement]) => current.replaceAll(`{${key}}`, String(replacement)),
    value,
  );
}

function TextList({ values, none }: { values: string[]; none: string }) {
  return values.length ? <ul>{values.map((value, index) => <li key={`${value}-${index}`}>{value}</li>)}</ul> : <p className="muted small">{none}</p>;
}

export function Phase4AssessmentPanel({ assessment, locale = "en", headingId = "phase4-assessment-title" }: { assessment: Phase4AssessmentView; locale?: Locale; headingId?: string }) {
  const t = copy[locale];
  return (
    <section className="detail-card phase4-assessment" aria-labelledby={headingId}>
      <header className="phase4-assessment__header">
        <div>
          <p className="eyebrow">Phase 4</p>
          <h2 id={headingId}>{t.title}</h2>
          <span className="status-badge">{diagnosticText(assessment.classification, locale)}</span>
        </div>
        <div className="phase4-assessment__score" aria-label={t.score}>
          <small>{t.score}</small>
          <strong>{assessment.relative_rank_score.toFixed(3)}</strong>
        </div>
      </header>
      <p className="notice notice--warning">{t.warning}</p>

      <details>
        <summary>{t.breakdown}</summary>
        <div className="phase4-breakdown">
          {assessment.score_breakdown.map((item) => (
            <dl key={item.feature}>
              <div><dt>{diagnosticText(item.feature, locale)}</dt><dd>{diagnosticText(item.reason_code, locale)}</dd></div>
              <div><dt>raw</dt><dd>{item.raw_value}</dd></div>
              <div><dt>weight</dt><dd>{item.weight}</dd></div>
              <div><dt>contribution</dt><dd>{item.contribution}</dd></div>
            </dl>
          ))}
        </div>
      </details>

      <div className="phase4-assessment__grid">
        <section><h3>{t.diversity}</h3><p>{assessment.source_diversity}</p></section>
        <section><h3>{t.map}</h3>{assessment.map_observations.length ? <ul>{assessment.map_observations.map((item) => <li key={`${item.provider}-${item.clue}-${item.map_feature}`}><strong>{item.status}</strong> — {item.clue} / {item.map_feature} ({item.provider}){item.limitation ? ` — ${item.limitation}` : ""}</li>)}</ul> : <p className="muted small">{t.none}</p>}</section>
        <section><h3>{t.geometry}</h3>{assessment.geometry_results.length ? <ul>{assessment.geometry_results.map((item) => <li key={`${item.provider}-${item.reference_id}`}><strong>{item.status}</strong> — {interpolate(t.matches, { inliers: item.inliers, matches: item.filtered_matches })}; {item.robust_model_type ?? "no model"}</li>)}</ul> : <p className="muted small">{t.none}</p>}</section>
        <section><h3>{t.contradictions}</h3><TextList values={assessment.contradictions} none={t.none} /></section>
        <section><h3>{t.references}</h3>{assessment.reference_attributions.length ? <ul>{assessment.reference_attributions.map((item) => <li key={item.reference_id}><strong>{item.source}</strong> — {item.license}<br /><span>{item.attribution}</span><br /><span>{item.display_allowed ? t.allowed : t.denied}</span></li>)}</ul> : <p className="muted small">{t.none}</p>}</section>
        <section><h3>{t.limitations}</h3><TextList values={assessment.limitations.map((item) => diagnosticText(item, locale))} none={t.none} /></section>
      </div>
    </section>
  );
}
