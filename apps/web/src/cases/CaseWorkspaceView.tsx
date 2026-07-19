import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ApiError, type AtlasLensApiClient } from "../api/client";
import type { CaseEvidenceRecord } from "../api/schemas";
import { validateImageFile } from "../file-validation";
import { useI18n } from "../i18n";
import { useAnalysis } from "../use-analysis";
import {
  InvestorDemoSteps,
  InvestorLimitations,
  InvestorProviderPanel,
  InvestorSourcePanel,
} from "../components/InvestorDemoView";
import { InvestorMapWorkspace } from "../components/InvestorMapWorkspace";
import { caseCopy, LOCAL_ANALYST_ID, purposeLabel, sensitivityLabel, statusLabel } from "./copy";
import { HypothesisReview } from "./HypothesisReview";

interface CaseWorkspaceViewProps {
  client: AtlasLensApiClient;
  caseId: string;
  onBack: () => void;
  guided?: boolean;
}

function sanitizeFilename(value: string): string {
  const leaf = value.replaceAll(String.fromCharCode(92), "/").split("/").pop() ?? "unnamed-image";
  const withoutControls = Array.from(leaf, (character) => {
    const code = character.charCodeAt(0);
    return code < 32 || code === 127 ? "_" : character;
  }).join("");
  const safe = withoutControls
    .replaceAll(/[^\p{L}\p{N} .()[\]_-]/gu, "_")
    .replaceAll(/\s+/g, " ")
    .replaceAll(/_+/g, "_")
    .trim()
    .slice(0, 160);
  return safe || "unnamed-image";
}

async function fileSha256(file: File): Promise<string> {
  if (!globalThis.crypto?.subtle) throw new Error("secure_digest_unavailable");
  const digest = await globalThis.crypto.subtle.digest("SHA-256", await file.arrayBuffer());
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function EvidencePanel({ items }: { items: CaseEvidenceRecord[] }) {
  const { locale } = useI18n();
  const text = caseCopy[locale];
  const grouped = useMemo(() => {
    const result = new Map<CaseEvidenceRecord["evidence_type"], Map<string, CaseEvidenceRecord[]>>();
    items.forEach((item) => {
      const byProvider = result.get(item.evidence_type) ?? new Map<string, CaseEvidenceRecord[]>();
      byProvider.set(item.provider, [...(byProvider.get(item.provider) ?? []), item]);
      result.set(item.evidence_type, byProvider);
    });
    return result;
  }, [items]);
  const evidenceTypeLabel = (value: CaseEvidenceRecord["evidence_type"]) => {
    const labels: Record<CaseEvidenceRecord["evidence_type"], { en: string; tr: string }> = {
      metadata: { en: "Metadata", tr: "Meta veri" },
      ocr: { en: "OCR", tr: "OCR" },
      visual_clue: { en: "Visual clue", tr: "Görsel ipucu" },
      model_hypothesis: { en: "Model hypothesis", tr: "Model hipotezi" },
      retrieval_match: { en: "Retrieval match", tr: "Getirim eşleşmesi" },
      map_evidence: { en: "Map evidence", tr: "Harita kanıtı" },
      analyst_note: { en: "Analyst note", tr: "Analist notu" },
    };
    return labels[value][locale];
  };

  return (
    <section className="case-evidence-panel" aria-labelledby="case-evidence-title">
      <header>
        <p className="eyebrow">{text.provenance}</p>
        <h2 id="case-evidence-title">{text.evidenceTitle}</h2>
      </header>
      {items.length === 0 ? <p className="workspace-empty">{text.noEvidence}</p> : (
        <div className="evidence-groups">
          {Array.from(grouped, ([type, providers]) => (
            <section key={type} className="evidence-group">
              <h3>{evidenceTypeLabel(type)}</h3>
              {Array.from(providers, ([provider, records]) => (
                <div key={provider} className="evidence-provider">
                  <header>
                    <strong>{provider}</strong>
                    <span>{text.providerFamily}: {records[0]?.provider_family}</span>
                  </header>
                  <ul>
                    {records.map((item) => (
                      <li key={item.id}>
                        <p>{item.summary}</p>
                        <small>
                          {text.provenance}: {item.provider_family} · {item.immutable_source_hash.slice(0, 10)}…
                        </small>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </section>
          ))}
        </div>
      )}
    </section>
  );
}

export function CaseWorkspaceView({ client, caseId, onBack, guided = false }: CaseWorkspaceViewProps) {
  const { locale } = useI18n();
  const text = caseCopy[locale];
  const queryClient = useQueryClient();
  const [selectedMediaId, setSelectedMediaId] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [fileInvalid, setFileInvalid] = useState(false);
  const [imageAuthorization, setImageAuthorization] = useState(false);
  const [pendingLink, setPendingLink] = useState<{ mediaId: string; analysisId: string } | null>(null);
  const materializationAttempts = useRef(new Set<string>());

  const capabilitiesQuery = useQuery({
    queryKey: ["capabilities"],
    queryFn: client.getCapabilities,
    staleTime: 60_000,
  });
  const workspaceQuery = useQuery({
    queryKey: ["case-workspace", caseId],
    queryFn: async () => {
      const [caseRecord, media, evidence, hypotheses, audit, integrity] = await Promise.all([
        client.getCase(caseId),
        client.listCaseMedia(caseId),
        client.listCaseEvidence(caseId),
        client.listCaseHypotheses(caseId),
        client.listCaseAuditEvents(caseId),
        client.getCaseAuditIntegrity(caseId),
      ]);
      return { caseRecord, media, evidence, hypotheses, audit, integrity };
    },
    retry: (failureCount, error) => (
      !(error instanceof ApiError && error.status === 404) && failureCount < 3
    ),
  });

  useEffect(() => {
    const items = workspaceQuery.data?.media.items;
    if (!items?.length) return;
    if (!selectedMediaId || !items.some((item) => item.id === selectedMediaId)) {
      setSelectedMediaId(items[0]?.id ?? null);
    }
  }, [selectedMediaId, workspaceQuery.data?.media.items]);

  useEffect(() => () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
  }, [previewUrl]);

  const selectedMedia = workspaceQuery.data?.media.items.find((item) => item.id === selectedMediaId) ?? null;
  const selectedAnalysisId = selectedMedia?.analysis_id
    ?? (pendingLink?.mediaId === selectedMediaId ? pendingLink.analysisId : null);
  const materializationMediaId = selectedMedia?.id
    ?? (pendingLink?.analysisId === selectedAnalysisId ? pendingLink.mediaId : null);
  const hasPendingMaterialization = pendingLink?.analysisId === selectedAnalysisId
    && pendingLink.mediaId === materializationMediaId;
  const needsMaterialization = hasPendingMaterialization || selectedMedia?.materialized_at === null;
  const analysisQuery = useAnalysis(client, selectedAnalysisId);

  const materializeMutation = useMutation({
    mutationFn: ({ mediaId, analysisId }: { mediaId: string; analysisId: string }) =>
      client.materializeCaseAnalysis(caseId, analysisId, mediaId, LOCAL_ANALYST_ID),
    onSuccess: () => {
      setPendingLink(null);
      void queryClient.invalidateQueries({ queryKey: ["case-workspace", caseId] });
      void queryClient.invalidateQueries({ queryKey: ["cases"] });
    },
  });

  useEffect(() => {
    if (
      !selectedAnalysisId
      || !materializationMediaId
      || !needsMaterialization
      || analysisQuery.data?.status !== "completed"
      || materializeMutation.isPending
    ) return;
    const attemptKey = selectedAnalysisId + ":" + materializationMediaId;
    if (materializationAttempts.current.has(attemptKey)) return;
    materializationAttempts.current.add(attemptKey);
    materializeMutation.mutate({ mediaId: materializationMediaId, analysisId: selectedAnalysisId });
  }, [
    analysisQuery.data?.status,
    materializationMediaId,
    materializeMutation,
    needsMaterialization,
    selectedAnalysisId,
  ]);

  const retryMaterialization = () => {
    if (!selectedAnalysisId || !materializationMediaId || !needsMaterialization) return;
    const attemptKey = selectedAnalysisId + ":" + materializationMediaId;
    materializationAttempts.current.delete(attemptKey);
    materializeMutation.reset();
    materializationAttempts.current.add(attemptKey);
    materializeMutation.mutate({ mediaId: materializationMediaId, analysisId: selectedAnalysisId });
  };

  const uploadMutation = useMutation({
    mutationFn: async ({ selected, caseRecord }: { selected: File; caseRecord: NonNullable<typeof workspaceQuery.data>["caseRecord"] }) => {
      const sha256 = await fileSha256(selected);
      const accepted = await client.createAnalysis({
        file: selected,
        mode: "local_only",
        cloudConsent: false,
        allowCloudAssist: false,
        authorizationAcknowledged: true,
      });
      const media = await client.createCaseMedia(caseId, {
        actor_id: LOCAL_ANALYST_ID,
        media_type: "image",
        source_type: "upload",
        original_filename_display: sanitizeFilename(selected.name),
        mime_type: selected.type as "image/jpeg" | "image/png" | "image/webp",
        byte_size: selected.size,
        sha256,
        captured_at: null,
        source_url: null,
        archive_url: null,
        source_description: caseRecord.source_context,
        authorization_attested: true,
        storage_state: "ephemeral",
      });
      await client.linkCaseAnalysis(caseId, accepted.id, media.id, LOCAL_ANALYST_ID);
      return { accepted, media };
    },
    onSuccess: ({ accepted, media }) => {
      setPendingLink({ mediaId: media.id, analysisId: accepted.id });
      setSelectedMediaId(media.id);
      setFile(null);
      setImageAuthorization(false);
      setPreviewUrl((current) => {
        if (current) URL.revokeObjectURL(current);
        return null;
      });
      void queryClient.invalidateQueries({ queryKey: ["case-workspace", caseId] });
      void queryClient.invalidateQueries({ queryKey: ["cases"] });
    },
  });

  const chooseFile = async (selected: File) => {
    const invalid = await validateImageFile(selected, capabilitiesQuery.data);
    setFile(selected);
    setFileInvalid(Boolean(invalid));
    uploadMutation.reset();
    setPreviewUrl((current) => {
      if (current) URL.revokeObjectURL(current);
      return invalid ? null : URL.createObjectURL(selected);
    });
  };

  const submitImage = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!file || fileInvalid || !imageAuthorization || !workspaceQuery.data) return;
    uploadMutation.mutate({ selected: file, caseRecord: workspaceQuery.data.caseRecord });
  };

  const refreshWorkspace = () => {
    void queryClient.invalidateQueries({ queryKey: ["case-workspace", caseId] });
    void queryClient.invalidateQueries({ queryKey: ["cases"] });
  };

  if (workspaceQuery.isPending) {
    return (
      <main id="main-content" className="workspace-shell">
        <section className="workspace-state" role="status"><span className="spinner" aria-hidden="true" /><p>{text.workspaceLoading}</p></section>
      </main>
    );
  }
  if (workspaceQuery.isError) {
    const caseNotFound = workspaceQuery.error instanceof ApiError && workspaceQuery.error.status === 404;
    return (
      <main id="main-content" className="workspace-shell">
        <section className="workspace-state" role="alert">
          <h1>{caseNotFound ? text.workspaceNotFound : text.workspaceError}</h1>
          <button type="button" className="secondary-button" onClick={() => void workspaceQuery.refetch()}>{text.retry}</button>
          <button type="button" className="text-button" onClick={onBack}>{text.backToCases}</button>
        </section>
      </main>
    );
  }

  const data = workspaceQuery.data;
  const mediaEvidence = data.evidence.items.filter((item) => !selectedMedia || item.media_id === selectedMedia.id);
  const mediaHypotheses = data.hypotheses.items.filter((item) => !selectedMedia || item.media_id === selectedMedia.id);
  const analysis = analysisQuery.data;

  if (guided) {
    return (
      <InvestorMapWorkspace
        caseRecord={data.caseRecord}
        media={data.media.items}
        selectedMedia={selectedMedia}
        hypotheses={mediaHypotheses}
        evidence={mediaEvidence}
        audit={data.audit.items}
        integrity={data.integrity}
        analysis={analysis}
        analysisLoading={Boolean(selectedAnalysisId) && (analysisQuery.isPending || analysis?.status === "queued" || analysis?.status === "processing")}
        analysisFailed={analysisQuery.isError || analysis?.status === "failed"}
      />
    );
  }

  return (
    <main id="main-content" className="case-workspace">
      {guided ? <InvestorDemoSteps /> : null}
      <header id={guided ? "demo-case" : undefined} className="case-workspace__header">
        <button type="button" className="text-button case-back" onClick={onBack}>← {text.backToCases}</button>
        <div className="case-workspace__identity">
          <div>
            <p className="eyebrow">{text.eyebrow}</p>
            <h1>{data.caseRecord.title}</h1>
            {data.caseRecord.description ? <p>{data.caseRecord.description}</p> : null}
          </div>
          <dl>
            <div><dt>{text.status}</dt><dd>{statusLabel(data.caseRecord.status, locale)}</dd></div>
            <div><dt>{text.casePurpose}</dt><dd>{purposeLabel(data.caseRecord.purpose, locale)}</dd></div>
            <div><dt>{text.caseSensitivity}</dt><dd>{sensitivityLabel(data.caseRecord.sensitivity, locale)}</dd></div>
          </dl>
        </div>
      </header>

      {data.caseRecord.sensitivity === "conflict_related" ? (
        <p className="case-sensitive-warning" role="alert">{text.conflictWarning}</p>
      ) : null}

      {guided ? <InvestorSourcePanel item={data.caseRecord} /> : null}

      <div className="case-workspace__layout">
        <aside className="case-media-rail" aria-labelledby="case-media-title">
          <h2 id="case-media-title">{text.evidenceMedia}</h2>
          {data.media.items.length === 0 ? <p className="workspace-empty">{text.noMedia}</p> : (
            <ol>
              {data.media.items.map((item, index) => (
                <li key={item.id}>
                  <button
                    type="button"
                    aria-current={selectedMediaId === item.id ? "true" : undefined}
                    onClick={() => setSelectedMediaId(item.id)}
                  >
                    <span>#{index + 1} · {item.original_filename_display}</span>
                    <small>{item.analysis_status_at_link ?? (item.analysis_id ? text.analysisComplete : text.analysisNotLinked)}</small>
                    {["deleted_after_analysis", "unavailable"].includes(item.storage_state) ? (
                      <strong>{text.mediaExpired}</strong>
                    ) : null}
                  </button>
                </li>
              ))}
            </ol>
          )}

          <form className="case-upload" onSubmit={submitImage}>
            <h3>{text.addImage}</h3>
            <label className="file-button" htmlFor="case-image-input">{text.chooseImage}</label>
            <input
              id="case-image-input"
              data-testid="case-file-input"
              className="visually-hidden"
              type="file"
              accept="image/jpeg,image/png,image/webp"
              onChange={(event) => {
                const selected = event.target.files?.[0];
                if (selected) void chooseFile(selected);
              }}
            />
            {previewUrl ? <img src={previewUrl} alt={locale === "tr" ? "Seçili görselin yerel önizlemesi" : "Local preview of selected image"} /> : null}
            {file ? <span className="case-upload__filename">{sanitizeFilename(file.name)}</span> : null}
            {fileInvalid ? <p className="field-error" role="alert">{text.invalidFile}</p> : null}
            <label className="case-upload__authorization">
              <input type="checkbox" checked={imageAuthorization} onChange={(event) => setImageAuthorization(event.target.checked)} />
              <span>{text.imageAuthorization}</span>
            </label>
            <p className="field-hint">{text.uploadPrivacy}</p>
            {uploadMutation.isError ? <p className="field-error" role="alert">{text.uploadFailed}</p> : null}
            <button className="primary-button" type="submit" disabled={!file || fileInvalid || !imageAuthorization || uploadMutation.isPending}>
              {uploadMutation.isPending ? text.analyzing : text.analyzeAndLink}
            </button>
          </form>
        </aside>

        <div className="case-review-canvas">
          <section id={guided ? "demo-analysis" : undefined} className="analysis-summary" aria-labelledby="selected-analysis-title">
            <header>
              <p className="eyebrow">{selectedMedia?.original_filename_display ?? text.noMedia}</p>
              <h2 id="selected-analysis-title">{text.selectedAnalysis}</h2>
            </header>
            {!selectedAnalysisId ? <p>{text.analysisNotLinked}</p>
              : analysisQuery.isPending || analysis?.status === "queued" || analysis?.status === "processing"
                ? <p role="status">{text.analysisPending}</p>
                : analysisQuery.isError || analysis?.status === "failed"
                  ? <p className="notice notice--error">{text.analysisFailed}</p>
                  : analysis?.status === "completed" ? (
                    <dl>
                      <div><dt>{text.status}</dt><dd>{text.analysisComplete}</dd></div>
                      <div><dt>{text.evidenceTitle}</dt><dd>{analysis.evidence.length}</dd></div>
                      <div><dt>{text.hypothesesTitle}</dt><dd>{analysis.candidates.length}</dd></div>
                      {guided ? <div><dt>{locale === "tr" ? "Analiz modu" : "Analysis mode"}</dt><dd>{analysis.analysis_mode === "local_only" ? (locale === "tr" ? "Yalnızca yerel" : "Local only") : (locale === "tr" ? "Açık onaylı bulut desteği" : "Explicitly consented cloud assist")}</dd></div> : null}
                      <div><dt>{locale === "tr" ? "Çekimserlik" : "Abstention"}</dt><dd>{analysis.abstention ? text.noHypotheses : "—"}</dd></div>
                    </dl>
                  ) : <p>{text.analysisNotLinked}</p>}
            {materializeMutation.isError ? (
              <div className="notice notice--error" role="alert">
                <p>{text.materializeFailed}</p>
                <button type="button" className="secondary-button" onClick={retryMaterialization}>
                  {text.retryMaterialize}
                </button>
              </div>
            ) : null}
          </section>

          {guided ? (
            <InvestorProviderPanel
              capabilities={capabilitiesQuery.data}
              loading={capabilitiesQuery.isPending}
              evidence={mediaEvidence}
            />
          ) : null}
          {!guided ? <EvidencePanel items={mediaEvidence} /> : null}
          <HypothesisReview
            client={client}
            caseId={caseId}
            media={selectedMedia}
            hypotheses={mediaHypotheses}
            evidence={mediaEvidence}
            onChanged={refreshWorkspace}
            guided={guided}
            betweenMapAndReview={guided ? <EvidencePanel items={mediaEvidence} /> : undefined}
          />

          <section id={guided ? "demo-audit" : undefined} className="audit-panel" aria-labelledby="case-audit-title">
            <header>
              <div>
                <p className="eyebrow">{text.auditLegal}</p>
                <h2 id="case-audit-title">{text.auditTitle}</h2>
              </div>
              <span className={"integrity-badge integrity-badge--" + (data.integrity.valid ? "valid" : "invalid")}>
                {data.integrity.valid ? text.auditValid : text.auditInvalid}
              </span>
            </header>
            {!data.integrity.valid ? <p className="notice notice--error" role="alert">{data.integrity.reason ?? text.auditInvalid}</p> : null}
            {data.audit.items.length === 0 ? <p className="workspace-empty">{text.noAudit}</p> : (
              <ol className="audit-timeline">
                {data.audit.items.map((item) => (
                  <li key={item.id}>
                    <span className="audit-sequence">{item.sequence_number}</span>
                    <div>
                      <strong>{item.event_type.replaceAll("_", " ")}</strong>
                      <span>{text.actor}: {item.actor_id} · {item.actor_type}</span>
                      <time dateTime={item.created_at}>{new Date(item.created_at).toLocaleString(locale)}</time>
                    </div>
                  </li>
                ))}
              </ol>
            )}
          </section>
          {guided ? <InvestorLimitations /> : null}
        </div>
      </div>
    </main>
  );
}
