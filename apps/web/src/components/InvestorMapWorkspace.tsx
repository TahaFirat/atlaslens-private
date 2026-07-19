import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type {
  Analysis,
  CaseAuditEvent,
  CaseAuditIntegrity,
  CaseEvidenceRecord,
  CaseMedia,
  InvestigationCase,
  LocationHypothesis,
} from "../api/schemas";
import { useI18n } from "../i18n";
import { humanizeToken } from "../display";
import { safeOperatorText } from "../safe-display";
import { decisionLabel } from "../cases/copy";
import { InvestorWorkspaceMap } from "./InvestorWorkspaceMap";

type AnalysisState = "loading" | "result" | "abstained" | "provider_unavailable";

interface RetrievalContextView {
  analysis_scope?: string | null;
  coverage_status?: string | null;
  coverage_label?: string | null;
  retrieval_provider?: string | null;
  retrieval_scope?: string | null;
  result_semantics?: string | null;
  abstained?: boolean | null;
  abstention_reason?: string | null;
  similarity_semantics?: string | null;
  supported_region?: string | null;
  benchmark_version?: string | null;
  evidence_version?: string | null;
}

function retrievalContextFrom(value: unknown): RetrievalContextView | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const record = value as Record<string, unknown>;
  if (typeof record.coverage_status !== "string" || typeof record.analysis_scope !== "string") return null;
  return {
    analysis_scope: record.analysis_scope,
    coverage_status: record.coverage_status,
    coverage_label: typeof record.coverage_label === "string" ? record.coverage_label : null,
    retrieval_provider: typeof record.retrieval_provider === "string" ? record.retrieval_provider : null,
    retrieval_scope: typeof record.retrieval_scope === "string" ? record.retrieval_scope : null,
    result_semantics: typeof record.result_semantics === "string" ? record.result_semantics : null,
    abstained: typeof record.abstained === "boolean" ? record.abstained : null,
    abstention_reason: typeof record.abstention_reason === "string" ? record.abstention_reason : null,
    similarity_semantics: typeof record.similarity_semantics === "string" ? record.similarity_semantics : null,
    supported_region: typeof record.supported_region === "string" ? record.supported_region : null,
    benchmark_version: typeof record.benchmark_version === "string" ? record.benchmark_version : null,
    evidence_version: typeof record.evidence_version === "string" ? record.evidence_version : null,
  };
}

function retrievalContextFor(analysis: Analysis | undefined, evidence: CaseEvidenceRecord[]): RetrievalContextView | null {
  const analysisRecord = analysis as (Analysis & {
    retrieval_context?: unknown;
    evidence: Array<Analysis["evidence"][number] & { retrieval_context?: unknown }>;
  }) | undefined;
  const direct = retrievalContextFrom(analysisRecord?.retrieval_context);
  if (direct) return direct;
  for (const item of analysisRecord?.evidence ?? []) {
    const context = retrievalContextFrom(item.retrieval_context);
    if (context) return context;
  }
  for (const item of evidence) {
    const context = retrievalContextFrom(item.structured_payload);
    if (context) return context;
  }
  return null;
}

interface InvestorMapWorkspaceProps {
  caseRecord: InvestigationCase;
  media: CaseMedia[];
  selectedMedia: CaseMedia | null;
  hypotheses: LocationHypothesis[];
  evidence: CaseEvidenceRecord[];
  audit: CaseAuditEvent[];
  integrity: CaseAuditIntegrity;
  analysis?: Analysis;
  analysisLoading: boolean;
  analysisFailed: boolean;
  drawerExtra?: ReactNode;
}

function placeLabel(
  item: LocationHypothesis,
  locale: "en" | "tr",
  analysisScope?: string | null,
): string {
  const parts = [item.locality_name, item.region_name, item.country_code].filter(Boolean);
  if (parts.length) return parts.join(", ");
  if (analysisScope === "ankara_reference_pilot") {
    const rank = item.rank ?? "—";
    return locale === "tr" ? `Ankara pilot referansı #${rank}` : `Ankara pilot reference #${rank}`;
  }
  return locale === "tr" ? "Adsız konum hipotezi" : "Unnamed location hypothesis";
}

function sourceTypeLabel(value: CaseMedia["source_type"], locale: "en" | "tr"): string {
  if (locale === "en") return humanizeToken(value);
  return {
    upload: "Yükleme",
    source_url: "Kaynak URL'si",
    external_archive: "Harici arşiv",
    other: "Diğer",
  }[value];
}

function storageStateLabel(value: CaseMedia["storage_state"], locale: "en" | "tr"): string {
  if (locale === "en") return humanizeToken(value);
  return {
    ephemeral: "Geçici",
    deleted_after_analysis: "Analiz sonrası silindi",
    unavailable: "Kullanılamıyor",
    externally_managed: "Harici olarak yönetiliyor",
  }[value];
}

function safeSimilarity(evidence: CaseEvidenceRecord[], hypothesis: LocationHypothesis): number | null {
  for (const id of hypothesis.supporting_evidence_ids) {
    const payload = evidence.find((item) => item.id === id)?.structured_payload;
    if (!payload || typeof payload !== "object" || Array.isArray(payload)) continue;
    for (const key of ["cosine_similarity", "similarity", "relative_similarity"]) {
      const value = payload[key];
      if (typeof value === "number" && Number.isFinite(value)) return value;
    }
  }
  return null;
}

export function InvestorMapWorkspace({
  caseRecord,
  media,
  selectedMedia,
  hypotheses,
  evidence,
  audit,
  integrity,
  analysis,
  analysisLoading,
  analysisFailed,
  drawerExtra,
}: InvestorMapWorkspaceProps) {
  const { locale } = useI18n();
  const retrievalContext = retrievalContextFor(analysis, evidence);
  const allRanked = useMemo(
    () => [...hypotheses].sort((left, right) => (left.rank ?? Number.MAX_SAFE_INTEGER) - (right.rank ?? Number.MAX_SAFE_INTEGER) || left.created_at.localeCompare(right.created_at) || left.id.localeCompare(right.id)),
    [hypotheses],
  );
  const coverageStatus = retrievalContext?.coverage_status?.toLowerCase() ?? "";
  const providerUnavailable = coverageStatus === "provider_unavailable"
    || retrievalContext?.abstention_reason?.toLowerCase() === "provider_unavailable";
  const contextAbstained = retrievalContext?.abstained === true
    || ["out_of_coverage", "insufficient", "unsupported", "not_eligible", "limited_coverage"].includes(coverageStatus);
  const ordered = useMemo(
    () => contextAbstained || providerUnavailable ? [] : allRanked,
    [allRanked, contextAbstained, providerUnavailable],
  );
  const [selectedId, setSelectedId] = useState<string | null>(ordered[0]?.id ?? null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [queryOpen, setQueryOpen] = useState(false);
  const selectedCardRef = useRef<HTMLElement>(null);

  useEffect(() => {
    if (!ordered.some((item) => item.id === selectedId)) setSelectedId(ordered[0]?.id ?? null);
  }, [ordered, selectedId]);

  const state: AnalysisState = analysisLoading
    ? "loading"
    : analysisFailed || analysis?.status === "failed" || providerUnavailable
      ? "provider_unavailable"
      : ordered.length === 0
        ? "abstained"
        : "result";
  const selected = ordered.find((item) => item.id === selectedId) ?? ordered[0] ?? null;
  const deleted = selectedMedia ? ["deleted_after_analysis", "unavailable"].includes(selectedMedia.storage_state) : true;
  const providerNames = Array.from(new Set(evidence.map((item) => safeOperatorText(item.provider)).filter((item): item is string => Boolean(item))));

  const text = locale === "tr" ? {
    status: { loading: "Analiz yükleniyor", result: "İncelemeye hazır", abstained: "Çekimser sonuç", provider_unavailable: "Sağlayıcı kullanılamıyor" } satisfies Record<AnalysisState, string>,
    scope: "Ankara referans pilotu",
    pilotMeaning: "Ankara referans koleksiyonu içinde görsel benzerlik araması.",
    notGeneral: "Türkiye-geneli konum tespiti veya kalibre edilmiş doğruluk değildir.",
    query: "Sorgu görseli",
    deleted: "Analiz sonrası görsel kaldırıldı",
    deletedHelp: "Ham piksel saklama politikası gereği gösterilmiyor. Analiz kaydı ve güvenli kaynak geçmişi korunur.",
    previewUnavailable: "Önizleme bu çalışma alanında saklanmıyor",
    provenance: "Kaynak geçmişi",
    source: "Kaynak türü",
    metadata: "Meta veri temizliği",
    metadataValue: "Dosya ve vaka bağlamı çıkarıma gönderilmez",
    retention: "Saklama",
    localOnly: "Yalnızca yerel analiz",
    cloudAssist: "Açık onaylı bulut desteği",
    modeChecking: "Analiz modu denetleniyor",
    inspect: "Sonucu incele",
    result: "Sonuç",
    top: "Birincil benzerlik adayı",
    abstain: "Konum ileri sürülmedi",
    abstainBody: "Kapsam veya doğrulama yeterli değil. AtlasLens yanlış bir şehir üretmek yerine çekimser kaldı.",
    providerBody: "Hazır ve uygun bir sağlayıcı sonucu yok. AtlasLens yedek sonuç uydurmadı.",
    loadingBody: "Canlı vaka ve analiz durumu okunuyor.",
    alternatives: "Alternatif adaylar",
    sourceType: "Kaynak türü",
    pilotRetrieval: "Yerel pilot benzerlik getirimi",
    similarity: "Benzerlik semantiği",
    similarityMeaning: "Kalibre edilmemiş göreli benzerlik; confidence veya olasılık değildir",
    unpublished: "Değer yayımlanmadı",
    uncertainty: "Belirsizlik",
    verification: "Doğrulama",
    evidence: "Kanıt",
    adjudication: "Değerlendirme",
    audit: "Audit",
    technical: "Teknik ayrıntılar",
    drawer: "Kanıt ve teknik ayrıntılar",
    openDrawer: "Ayrıntıları aç",
    closeDrawer: "Ayrıntıları kapat",
    integrity: "Audit bütünlüğü",
    noEvidence: "Güvenli kanıt özeti yok; kanıt uydurulmadı.",
    noDecision: "Operatör kararı henüz yok.",
    privacy: "Özel ve geçici",
    queryDetails: "Sorgu ayrıntıları",
    showQuery: "Sorgu panelini aç",
    hideQuery: "Sorgu panelini kapat",
  } : {
    status: { loading: "Loading analysis", result: "Ready for review", abstained: "Abstained", provider_unavailable: "Provider unavailable" } satisfies Record<AnalysisState, string>,
    scope: "Ankara reference pilot",
    pilotMeaning: "Visual similarity search within the Ankara reference collection.",
    notGeneral: "This is not Turkey-wide geolocation or calibrated accuracy.",
    query: "Query image",
    deleted: "Image removed after analysis",
    deletedHelp: "Raw pixels are not displayed under the retention policy. The analysis record and safe provenance remain.",
    previewUnavailable: "Preview is not retained in this workspace",
    provenance: "Provenance",
    source: "Source type",
    metadata: "Metadata cleaning",
    metadataValue: "Filename and case context are excluded from inference",
    retention: "Retention",
    localOnly: "Local-only analysis",
    cloudAssist: "Explicitly consented cloud assist",
    modeChecking: "Checking analysis mode",
    inspect: "Inspect result",
    result: "Result",
    top: "Primary similarity candidate",
    abstain: "No location asserted",
    abstainBody: "Coverage or verification is insufficient. AtlasLens abstained instead of producing a false city.",
    providerBody: "No ready, eligible provider result is available. AtlasLens did not invent a fallback.",
    loadingBody: "Reading the live case and analysis state.",
    alternatives: "Alternative candidates",
    sourceType: "Source type",
    pilotRetrieval: "Local pilot similarity retrieval",
    similarity: "Similarity semantics",
    similarityMeaning: "Uncalibrated relative similarity; not confidence or probability",
    unpublished: "Value not published",
    uncertainty: "Uncertainty",
    verification: "Verification",
    evidence: "Evidence",
    adjudication: "Adjudication",
    audit: "Audit",
    technical: "Technical details",
    drawer: "Evidence and technical details",
    openDrawer: "Open details",
    closeDrawer: "Close details",
    integrity: "Audit integrity",
    noEvidence: "No safe evidence summary is available; none was invented.",
    noDecision: "No operator decision yet.",
    privacy: "Private and temporary",
    queryDetails: "Query details",
    showQuery: "Open query panel",
    hideQuery: "Close query panel",
  };

  const contractText = (value: string | null | undefined, fallback: string): string => {
    const safe = value ? safeOperatorText(value) : null;
    if (!safe) return fallback;
    const normalized = safe.toLowerCase();
    if (["ankara_reference_pilot_similarity", "bounded_reference_similarity", "visual_similarity_within_limited_reference_collection", "ankara_reference_collection_visual_similarity_not_general_geolocation"].includes(normalized)) return text.pilotMeaning;
    if (["uncalibrated_relative_similarity", "relative_similarity_not_probability", "uncalibrated_cosine_similarity", "cosine_similarity_not_confidence"].includes(normalized)) return text.similarityMeaning;
    return humanizeToken(safe);
  };
  const rawCoverageLabel = safeOperatorText(retrievalContext?.coverage_label ?? retrievalContext?.supported_region);
  const coverageLabel = rawCoverageLabel === "Ankara reference pilot" ? text.scope : rawCoverageLabel ?? text.scope;
  const resultSemantics = contractText(retrievalContext?.result_semantics, text.pilotMeaning);
  const similaritySemantics = contractText(retrievalContext?.similarity_semantics, text.similarityMeaning);
  const retrievalProvider = safeOperatorText(retrievalContext?.retrieval_provider) ?? text.pilotRetrieval;
  const abstentionReason = retrievalContext?.abstention_reason
    ? contractText(retrievalContext.abstention_reason, "")
    : null;
  const analysisMode = !analysis
    ? text.modeChecking
    : analysis.analysis_mode === "cloud_assisted"
      ? text.cloudAssist
      : text.localOnly;

  const selectCandidate = (id: string) => {
    setSelectedId(id);
    requestAnimationFrame(() => selectedCardRef.current?.focus());
  };

  return (
    <main id="main-content" className="investor-workspace" data-analysis-state={state}>
      <section className={`investor-query-panel${queryOpen ? " investor-query-panel--open" : ""}`} aria-labelledby="investor-query-title">
        <button type="button" className="investor-mobile-panel-toggle" aria-expanded={queryOpen} onClick={() => setQueryOpen((current) => !current)}>
          <strong>{text.queryDetails}</strong><span>{queryOpen ? text.hideQuery : text.showQuery}</span>
        </button>
        <header>
          <span className="investor-section-kicker">01</span>
          <div><h1 id="investor-query-title">{text.query}</h1><p>{text.privacy}</p></div>
        </header>
        <div className={`investor-query-preview${deleted ? " investor-query-preview--deleted" : ""}`} data-testid="investor-query-preview">
          <span className="investor-query-preview__mark" aria-hidden="true">⌁</span>
          <strong>{deleted ? text.deleted : text.previewUnavailable}</strong>
          <small>{text.deletedHelp}</small>
        </div>
        <dl className="investor-query-facts">
          <div><dt>{text.source}</dt><dd>{selectedMedia ? sourceTypeLabel(selectedMedia.source_type, locale) : "—"}</dd></div>
          <div><dt>{text.provenance}</dt><dd>{providerNames.join(" · ") || "—"}</dd></div>
          <div><dt>{text.metadata}</dt><dd>{text.metadataValue}</dd></div>
          <div><dt>{text.retention}</dt><dd>{selectedMedia ? storageStateLabel(selectedMedia.storage_state, locale) : caseRecord.retention_policy}</dd></div>
        </dl>
        <p className="investor-query-mode"><i aria-hidden="true" />{analysisMode}</p>
        <button className="investor-primary-action" type="button" onClick={() => {
          if (state === "result" && selected) selectCandidate(selected.id);
          else setDrawerOpen(true);
        }}>{text.inspect}</button>
      </section>

      <InvestorWorkspaceMap hypotheses={ordered} selectedId={selectedId} onSelect={selectCandidate} />

      <aside className="investor-result-panel" aria-labelledby="investor-result-title">
        <header>
          <span className="investor-section-kicker">02</span>
          <div><p>{text.result}</p><strong className={`investor-result-state investor-result-state--${state}`}>{text.status[state]}</strong></div>
        </header>
        <div className="investor-scope-note"><strong>{coverageLabel}</strong><span>{resultSemantics}</span><small>{text.notGeneral}</small></div>
        {state === "loading" ? <div className="investor-result-empty" role="status"><span className="spinner" aria-hidden="true" /><h2 id="investor-result-title">{text.status.loading}</h2><p>{text.loadingBody}</p></div> : null}
        {state === "provider_unavailable" ? <div className="investor-result-empty" role="alert"><span aria-hidden="true">!</span><h2 id="investor-result-title">{text.status.provider_unavailable}</h2><p>{text.providerBody}{abstentionReason ? ` · ${abstentionReason}` : ""}</p></div> : null}
        {state === "abstained" ? <div className="investor-result-empty investor-result-empty--abstain"><span aria-hidden="true">—</span><h2 id="investor-result-title">{text.abstain}</h2><p>{text.abstainBody}{abstentionReason ? ` · ${abstentionReason}` : ""}</p></div> : null}
        {state === "result" && selected ? (
          <>
            <article className="investor-primary-result" tabIndex={-1} ref={selectedCardRef}>
              <p>{text.top} · #{selected.rank ?? 1}</p>
              <h2 id="investor-result-title">{placeLabel(selected, locale, retrievalContext?.analysis_scope)}</h2>
              <dl>
                <div><dt>{text.sourceType}</dt><dd>{retrievalProvider}</dd></div>
                <div><dt>{text.similarity}</dt><dd>{safeSimilarity(evidence, selected)?.toFixed(4) ?? text.unpublished}<small>{similaritySemantics}</small></dd></div>
                <div><dt>{text.uncertainty}</dt><dd>{Math.round(selected.uncertainty_radius_m / 1000).toLocaleString(locale)} km radius</dd></div>
                <div><dt>{text.verification}</dt><dd>{decisionLabel(selected.latest_adjudication?.decision ?? null, locale)}</dd></div>
              </dl>
            </article>
            {ordered.length > 1 ? <section className="investor-alternatives" aria-labelledby="investor-alternatives-title"><h3 id="investor-alternatives-title">{text.alternatives}</h3><ol>{ordered.map((item, index) => (
              <li key={item.id}><button type="button" aria-pressed={item.id === selectedId} onMouseEnter={() => setSelectedId(item.id)} onFocus={() => setSelectedId(item.id)} onClick={() => selectCandidate(item.id)}><span>{item.rank ?? index + 1}</span><strong>{placeLabel(item, locale, retrievalContext?.analysis_scope)}</strong><small>{Math.round(item.uncertainty_radius_m / 1000).toLocaleString(locale)} km</small></button></li>
            ))}</ol></section> : null}
          </>
        ) : null}
      </aside>

      <section className={`investor-detail-drawer${drawerOpen ? " investor-detail-drawer--open" : ""}`} aria-labelledby="investor-drawer-title">
        <button type="button" className="investor-drawer-toggle" aria-expanded={drawerOpen} onClick={() => setDrawerOpen((current) => !current)}>
          <span><strong id="investor-drawer-title">{text.drawer}</strong><small>{text.evidence} · {text.adjudication} · {text.audit} · {text.technical}</small></span>
          <span>{drawerOpen ? text.closeDrawer : text.openDrawer}</span>
        </button>
        {drawerOpen ? <div className="investor-drawer-content">
          <section><h3>{text.evidence}</h3>{evidence.length ? <ul>{evidence.map((item) => <li key={item.id}><strong>{item.provider}</strong><span>{item.summary}</span></li>)}</ul> : <p>{text.noEvidence}</p>}</section>
          <section><h3>{text.adjudication}</h3><p>{selected?.latest_adjudication ? `${decisionLabel(selected.latest_adjudication.decision, locale)} · ${selected.latest_adjudication.rationale}` : text.noDecision}</p>{drawerExtra}</section>
          <section><h3>{text.audit}</h3><p><strong>{text.integrity}:</strong> {integrity.valid ? "✓" : "!"}</p><p>{audit.length} {locale === "tr" ? "uygulama olayı" : "application events"}</p></section>
          <section><h3>{text.technical}</h3><dl><div><dt>Case</dt><dd>{caseRecord.id.slice(0, 8)}…</dd></div><div><dt>Media</dt><dd>{media.length}</dd></div><div><dt>Scope</dt><dd>{coverageLabel}</dd></div><div><dt>Semantics</dt><dd>{similaritySemantics}</dd></div>{retrievalContext?.benchmark_version ? <div><dt>Benchmark</dt><dd>{safeOperatorText(retrievalContext.benchmark_version)}</dd></div> : null}</dl></section>
        </div> : null}
      </section>
    </main>
  );
}
