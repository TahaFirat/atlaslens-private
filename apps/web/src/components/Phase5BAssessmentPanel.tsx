import type { Phase5BAssessment, Phase5BDiagnostics } from "../api/schemas";

type Locale = "en" | "tr";

const copy = {
  en: {
    title: "Combined evidence assessment",
    warning: "This raw relative rank is not a probability or a guarantee of location.",
    score: "Raw relative rank",
    providers: "Independent providers",
    sources: "Independent sources",
    breakdown: "Weighted evidence",
    places: "Place-name evidence",
    retrieval: "Licensed image retrieval",
    map: "Map evidence",
    support: "Supporting evidence",
    contradictions: "Contradictions",
    movement: "Ranking movement",
    limitations: "Limitations",
    none: "None reported",
    ambiguity: "Ambiguity count",
    rawStrength: "Raw strength",
    rawSimilarity: "Raw relative similarity",
    displayDenied: "Reference display denied by source policy",
    displayAllowed: "Reference display permitted; no image is embedded here",
    diagnostics: "Evidence provider status",
    offline: "offline",
    online: "network-capable",
    index: "Reference index",
    images: "images",
    unavailable: "Unavailable",
  },
  tr: {
    title: "Birleşik kanıt değerlendirmesi",
    warning: "Bu ham göreli sıra bir olasılık veya konum garantisi değildir.",
    score: "Ham göreli sıra",
    providers: "Bağımsız sağlayıcılar",
    sources: "Bağımsız kaynaklar",
    breakdown: "Ağırlıklı kanıt",
    places: "Yer adı kanıtı",
    retrieval: "Lisanslı görüntü retrieval",
    map: "Harita kanıtı",
    support: "Destekleyen kanıt",
    contradictions: "Çelişkiler",
    movement: "Sıralama hareketi",
    limitations: "Sınırlamalar",
    none: "Bildirilmedi",
    ambiguity: "Belirsiz eşleşme sayısı",
    rawStrength: "Ham destek",
    rawSimilarity: "Ham göreli benzerlik",
    displayDenied: "Referans gösterimi kaynak ilkesi gereği engelli",
    displayAllowed: "Referans gösterimine izin var; burada görüntü gömülmedi",
    diagnostics: "Kanıt sağlayıcı durumu",
    offline: "çevrimdışı",
    online: "ağ kullanabilir",
    index: "Referans indeksi",
    images: "görüntü",
    unavailable: "Kullanılamıyor",
  },
} as const;

const labels: Record<string, Record<Locale, string>> = {
  model_only: { en: "Model only", tr: "Yalnızca model" },
  place_supported: { en: "Place supported", tr: "Yer adı destekli" },
  retrieval_supported: { en: "Retrieval supported", tr: "Retrieval destekli" },
  map_supported: { en: "Map supported", tr: "Harita destekli" },
  multi_source_supported: { en: "Multiple sources support this candidate", tr: "Birden çok kaynak bu adayı destekliyor" },
  contradicted: { en: "Contradicted", tr: "Çelişkili" },
  model_prior: { en: "Global-model prior", tr: "Küresel model öncülü" },
  place_support: { en: "Place-name support", tr: "Yer adı desteği" },
  retrieval_similarity: { en: "Retrieval similarity", tr: "Retrieval benzerliği" },
  retrieval_compactness: { en: "Retrieval cluster compactness", tr: "Retrieval küme sıkılığı" },
  map_support: { en: "Map support", tr: "Harita desteği" },
  provider_diversity: { en: "Provider diversity", tr: "Sağlayıcı çeşitliliği" },
  source_diversity: { en: "Source diversity", tr: "Kaynak çeşitliliği" },
  contradiction_penalty: { en: "Contradiction penalty", tr: "Çelişki cezası" },
  "phase5b.support.place": { en: "Resolved public place supports this area", tr: "Çözümlenen kamusal yer bu bölgeyi destekliyor" },
  "phase5b.support.retrieval": { en: "Licensed visual references support this area", tr: "Lisanslı görsel referanslar bu bölgeyi destekliyor" },
  "phase5b.support.map": { en: "Structured map features support observed clues", tr: "Yapısal harita özellikleri gözlenen ipuçlarını destekliyor" },
  "phase5b.rank_increased.evidence_support": { en: "Moved up because independent evidence added support", tr: "Bağımsız kanıt desteğiyle yukarı taşındı" },
  "phase5b.rank_decreased.relative_support": { en: "Moved down because other candidates had stronger relative support", tr: "Diğer adayların göreli desteği daha güçlü olduğu için aşağı indi" },
  "phase5b.rank_preserved": { en: "Relative position was preserved", tr: "Göreli konum korundu" },
  "phase5b.relative_rank_is_not_probability": { en: "Relative rank is not a calibrated probability", tr: "Göreli sıra kalibre edilmiş olasılık değildir" },
  "phase5b.place_evidence_unavailable": { en: "No usable place-name evidence", tr: "Kullanılabilir yer adı kanıtı yok" },
  "phase5b.retrieval_evidence_unavailable": { en: "No usable licensed retrieval evidence", tr: "Kullanılabilir lisanslı retrieval kanıtı yok" },
  "phase5b.map_evidence_unavailable": { en: "No usable map evidence", tr: "Kullanılabilir harita kanıtı yok" },
  "phase5b.repeated_source_members_suppressed": { en: "Repeated source or capture-family members were suppressed", tr: "Yinelenen kaynak veya çekim ailesi üyeleri bastırıldı" },
};

function safeLabel(value: string, locale: Locale): string {
  return labels[value]?.[locale] ?? (locale === "tr" ? "Ek yapılandırılmış kanıt" : "Additional structured evidence");
}

function SafeList({ values, locale, none }: { values: string[]; locale: Locale; none: string }) {
  return values.length ? <ul>{values.map((value, index) => <li key={`${value}-${index}`}>{safeLabel(value, locale)}</li>)}</ul> : <p className="muted small">{none}</p>;
}

export function Phase5BAssessmentPanel({ assessment, locale, headingId }: { assessment: Phase5BAssessment; locale: Locale; headingId: string }) {
  const t = copy[locale];
  return (
    <section className="detail-card phase5b-assessment" aria-labelledby={headingId}>
      <header className="phase5b-assessment__header">
        <div><p className="eyebrow">{String(assessment.reranker_version) === "phase6a-v1" ? "Phase 6A" : "Phase 5B"}</p><h2 id={headingId}>{t.title}</h2><span className="status-badge">{safeLabel(assessment.classification, locale)}</span></div>
        <div className="phase5b-assessment__score"><small>{t.score}</small><strong>{assessment.relative_rank_score.toFixed(3)}</strong></div>
      </header>
      <p className="notice notice--warning">{t.warning}</p>
      <dl className="phase5b-diversity"><div><dt>{t.providers}</dt><dd>{assessment.provider_diversity}</dd></div><div><dt>{t.sources}</dt><dd>{assessment.source_diversity}</dd></div></dl>
      <details><summary>{t.breakdown}</summary><div className="phase5b-breakdown">{assessment.score_breakdown.map((item) => <dl key={item.feature}><div><dt>{safeLabel(item.feature, locale)}</dt><dd>{item.reason_code.startsWith("phase5b.feature.") ? safeLabel(item.feature, locale) : safeLabel(item.reason_code, locale)}</dd></div><div><dt>raw</dt><dd>{item.raw_value.toFixed(4)}</dd></div><div><dt>weight</dt><dd>{item.weight.toFixed(4)}</dd></div><div><dt>contribution</dt><dd>{item.contribution.toFixed(4)}</dd></div></dl>)}</div></details>
      <div className="phase5b-assessment__grid">
        <section><h3>{t.places}</h3>{assessment.place_matches.length ? <ul>{assessment.place_matches.map((item) => <li key={`${item.normalized_name}-${item.source}`}><strong>{item.normalized_name}</strong>{item.region ? ` · ${item.region}` : ""}{item.country_code ? ` · ${item.country_code}` : ""}<br /><span>{item.match_type} · {t.ambiguity}: {item.ambiguity_count} · {t.rawStrength}: {item.evidence_strength.toFixed(3)}</span><br /><span>{item.source} · {item.dataset_version} · {item.license}</span></li>)}</ul> : <p className="muted small">{t.none}</p>}</section>
        <section><h3>{t.retrieval}</h3>{assessment.retrieval_matches.length ? <ul>{assessment.retrieval_matches.map((item) => <li key={item.reference_id}><strong>{item.source}</strong> · {item.provider}<br /><span>{t.rawSimilarity}: {item.relative_similarity.toFixed(4)} · distance: {item.distance.toFixed(4)}</span><br /><span>{item.license} · {item.attribution}</span><br /><span>{item.display_allowed ? t.displayAllowed : t.displayDenied}</span></li>)}</ul> : <p className="muted small">{t.none}</p>}</section>
        <section><h3>{t.map}</h3>{assessment.map_observations.length ? <ul>{assessment.map_observations.map((item) => <li key={`${item.provider}-${item.clue}-${item.map_feature}`}><strong>{item.status}</strong> · {item.map_feature} · {item.provider} · raw {item.reliability.toFixed(3)}</li>)}</ul> : <p className="muted small">{t.none}</p>}</section>
        <section><h3>{t.support}</h3><SafeList values={assessment.supports} locale={locale} none={t.none} /></section>
        <section><h3>{t.contradictions}</h3><SafeList values={assessment.contradictions} locale={locale} none={t.none} /></section>
        <section><h3>{t.movement}</h3><SafeList values={assessment.movement_reasons} locale={locale} none={t.none} /></section>
        <section><h3>{t.limitations}</h3><SafeList values={assessment.limitations} locale={locale} none={t.none} /></section>
      </div>
    </section>
  );
}

export function Phase5BDiagnosticsPanel({ diagnostics, locale }: { diagnostics: Phase5BDiagnostics; locale: Locale }) {
  const t = copy[locale];
  return (
    <section className="detail-card phase5b-diagnostics" aria-labelledby="phase5b-diagnostics-heading">
      <h2 id="phase5b-diagnostics-heading">{t.diagnostics}</h2>
      {diagnostics.providers.length ? <ul>{diagnostics.providers.map((provider) => <li key={`${provider.provider_id}-${provider.provider_type}`}><strong>{provider.provider_type}</strong> · {provider.provider_id}<br /><span>{provider.status} · {provider.duration_ms} ms · {provider.offline ? t.offline : t.online}{provider.device ? ` · ${provider.device}` : ""}</span>{provider.reason_code ? <><br /><span>{safeLabel(provider.reason_code, locale)}</span></> : null}</li>)}</ul> : <p className="muted">{t.none}</p>}
      <h3>{t.index}</h3>
      {diagnostics.reference_index ? <p>{diagnostics.reference_index.status} · {diagnostics.reference_index.image_count} {t.images}{diagnostics.reference_index.embedding_provider ? ` · ${diagnostics.reference_index.embedding_provider} ${diagnostics.reference_index.embedding_version ?? ""}` : ""}</p> : <p className="muted">{t.unavailable}</p>}
      <SafeList values={diagnostics.partial_failures} locale={locale} none={t.none} />
    </section>
  );
}
