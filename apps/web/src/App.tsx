import { useEffect, useState } from "react";
import { QueryClient, QueryClientProvider, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ApiError, apiClient, type AtlasLensApiClient } from "./api/client";
import type { AnalysisMode } from "./api/schemas";
import { CaseListView } from "./cases/CaseListView";
import { CaseWorkspaceView } from "./cases/CaseWorkspaceView";
import { CreateCaseView } from "./cases/CreateCaseView";
import { ErrorState } from "./components/ErrorState";
import { DatasetQaDashboard } from "./components/DatasetQaDashboard";
import { EvaluationDashboard } from "./components/EvaluationDashboard";
import { HistoryPanel } from "./components/HistoryPanel";
import { InvestorDemoView, investorDemoNavLabel } from "./components/InvestorDemoView";
import { LanguageSwitcher } from "./components/LanguageSwitcher";
import { MapillaryDemoView } from "./components/MapillaryDemoView";
import { NoSignal } from "./components/NoSignal";
import { ProcessingPanel } from "./components/ProcessingPanel";
import { ProviderDashboard } from "./components/ProviderDashboard";
import { ResultView } from "./components/ResultView";
import { UploadPanel } from "./components/UploadPanel";
import { validateImageFile, type FileValidationCode } from "./file-validation";
import { I18nProvider, useI18n, type Locale, type TranslationKey } from "./i18n";
import { useAnalysis } from "./use-analysis";

interface AtlasLensAppProps {
  client?: AtlasLensApiClient;
  initialLocale?: Locale;
}

type WorkspacePage =
  | "analysis"
  | "cases"
  | "create-case"
  | "case"
  | "investor-demo"
  | "investor-create-case"
  | "investor-case"
  | "history"
  | "evaluation"
  | "dataset-qa"
  | "providers"
  | "mapillary-demo";

type InvestorCaseIdIssue = "missing" | "invalid" | null;

const canonicalUuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

function initialWorkspaceRoute(): {
  page: WorkspacePage;
  caseId: string | null;
  caseIdIssue: InvestorCaseIdIssue;
} {
  if (typeof window === "undefined") return { page: "analysis", caseId: null, caseIdIssue: null };
  const query = new URLSearchParams(window.location.search);
  const investorDemo = query.get("demo") === "investor" || window.location.hash === "#investor-demo";
  if (!investorDemo) return { page: "analysis", caseId: null, caseIdIssue: null };
  const caseIds = query.getAll("caseId");
  if (caseIds.length === 0) return { page: "investor-demo", caseId: null, caseIdIssue: "missing" };
  if (caseIds.length !== 1 || !canonicalUuid.test(caseIds[0] ?? "")) {
    return { page: "investor-demo", caseId: null, caseIdIssue: "invalid" };
  }
  return { page: "investor-case", caseId: caseIds[0]!.toLowerCase(), caseIdIssue: null };
}

function requestedLocale(): Locale | undefined {
  if (typeof window === "undefined") return undefined;
  const value = new URLSearchParams(window.location.search).get("lang");
  return value === "tr" || value === "en" ? value : undefined;
}

function messageCode(error: unknown): string | undefined {
  if (error instanceof ApiError) return error.problem?.message_key ?? error.message;
  return undefined;
}

function safeDiagnostic(error: unknown): { code: string; requestId: string } | null {
  if (!(error instanceof ApiError) || !error.problem) return null;
  const clean = (value: string) => value.replaceAll(/[^a-zA-Z0-9_.:-]/g, "").slice(0, 128);
  const code = clean(error.problem.code);
  const requestId = clean(error.problem.request_id);
  return code && requestId ? { code, requestId } : null;
}

function AppContent({ client }: { client: AtlasLensApiClient }) {
  const { locale, t, serverMessage } = useI18n();
  const queryClient = useQueryClient();
  const [initialRoute] = useState(initialWorkspaceRoute);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [validationError, setValidationError] = useState<FileValidationCode | null>(null);
  const [mode, setMode] = useState<AnalysisMode>("local_only");
  const [authorization, setAuthorization] = useState(false);
  const [cloudConsent, setCloudConsent] = useState(false);
  const [allowCloudAssist, setAllowCloudAssist] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [analysisId, setAnalysisId] = useState<string | null>(null);
  const [deletedNotice, setDeletedNotice] = useState(false);
  const [deleteError, setDeleteError] = useState(false);
  const [page, setPage] = useState<WorkspacePage>(initialRoute.page);
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(initialRoute.caseId);
  const [investorCaseIdIssue, setInvestorCaseIdIssue] = useState<InvestorCaseIdIssue>(initialRoute.caseIdIssue);

  const capabilitiesQuery = useQuery({
    queryKey: ["capabilities"],
    queryFn: client.getCapabilities,
    staleTime: 60_000,
    retry: 1,
  });
  const analysisQuery = useAnalysis(client, analysisId);
  const createMutation = useMutation({
    mutationFn: client.createAnalysis,
    onSuccess(accepted) {
      setAnalysisId(accepted.id);
      setUploadProgress(100);
      setPage("analysis");
    },
  });
  const deleteMutation = useMutation({
    mutationFn: async (id: string) => client.deleteAnalysis(id),
    onSuccess: () => {
      if (analysisId) queryClient.removeQueries({ queryKey: ["analysis", analysisId] });
      setAnalysisId(null);
      setFile(null);
      setPreviewUrl((current) => {
        if (current) URL.revokeObjectURL(current);
        return null;
      });
      setAuthorization(false);
      setCloudConsent(false);
      setAllowCloudAssist(false);
      setMode("local_only");
      setDeletedNotice(true);
      setDeleteError(false);
    },
    onError: () => setDeleteError(true),
  });

  useEffect(() => {
    document.documentElement.lang = locale;
    if (typeof window === "undefined") return;
    const query = new URLSearchParams(window.location.search);
    if (query.get("demo") !== "investor") return;
    query.set("lang", locale);
    const search = query.toString();
    window.history.replaceState(null, "", `${window.location.pathname}${search ? `?${search}` : ""}${window.location.hash}`);
  }, [locale]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const inInvestorDemo = page === "investor-demo" || page === "investor-create-case" || page === "investor-case";
    const query = new URLSearchParams(window.location.search);
    if (inInvestorDemo) {
      query.set("demo", "investor");
      query.set("lang", locale);
      if (page === "investor-case" && selectedCaseId) query.set("caseId", selectedCaseId);
      else query.delete("caseId");
    } else {
      query.delete("demo");
      query.delete("caseId");
    }
    const search = query.toString();
    const hash = window.location.hash === "#investor-demo" ? "" : window.location.hash;
    const target = `${window.location.pathname}${search ? `?${search}` : ""}${hash}`;
    const current = `${window.location.pathname}${window.location.search}${window.location.hash}`;
    if (target !== current) window.history.replaceState(null, "", target);
  }, [locale, page, selectedCaseId]);

  useEffect(
    () => () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    },
    [previewUrl],
  );

  const cloudAvailable = Boolean(
    capabilitiesQuery.data?.enabled_analysis_modes.includes("cloud_assisted") &&
      capabilitiesQuery.data.providers.cloud_vision.enabled &&
      capabilitiesQuery.data.providers.cloud_vision.available,
  );

  const canSubmit = Boolean(
    file &&
      !validationError &&
      authorization &&
      (mode === "local_only" || (cloudConsent && cloudAvailable)),
  );

  const handleFile = async (selected: File) => {
    setDeletedNotice(false);
    createMutation.reset();
    const error = await validateImageFile(selected, capabilitiesQuery.data);
    setFile(selected);
    setValidationError(error);
    setPreviewUrl((current) => {
      if (current) URL.revokeObjectURL(current);
      return error ? null : URL.createObjectURL(selected);
    });
  };

  const submit = () => {
    if (!file || !canSubmit || createMutation.isPending) return;
    setUploadProgress(0);
    setDeletedNotice(false);
    createMutation.mutate({
      file,
      mode,
      cloudConsent: mode === "cloud_assisted" && cloudConsent,
      allowCloudAssist: mode === "cloud_assisted" && allowCloudAssist,
      authorizationAcknowledged: authorization,
      onUploadProgress: setUploadProgress,
    });
  };

  const reset = () => {
    setAnalysisId(null);
    setDeleteError(false);
    setUploadProgress(0);
    createMutation.reset();
  };

  const beginNewAnalysis = () => {
    setAnalysisId(null);
    setFile(null);
    setPreviewUrl((current) => {
      if (current) URL.revokeObjectURL(current);
      return null;
    });
    setValidationError(null);
    setAuthorization(false);
    setCloudConsent(false);
    setAllowCloudAssist(false);
    setMode("local_only");
    setUploadProgress(0);
    setDeleteError(false);
    setDeletedNotice(false);
    createMutation.reset();
    setPage("analysis");
  };

  const deleteCurrent = () => {
    if (analysisId) deleteMutation.mutate(analysisId);
  };

  const analysis = analysisQuery.data;
  const submitDiagnostic = safeDiagnostic(createMutation.error);
  const submitError = createMutation.isError
    ? `${serverMessage(messageCode(createMutation.error))}${
        submitDiagnostic
          ? ` ${t("errors.diagnostic", { code: submitDiagnostic.code, requestId: submitDiagnostic.requestId })}`
          : ""
      }`
    : null;
  const operatorUi = import.meta.env.VITE_ENABLE_OPERATOR_UI === "true";
  let content;
  if (page === "investor-demo") {
    content = (
      <InvestorDemoView
        client={client}
        capabilities={capabilitiesQuery.data}
        capabilitiesLoading={capabilitiesQuery.isPending}
        caseIdIssue={investorCaseIdIssue}
        onCreate={() => {
          setInvestorCaseIdIssue(null);
          setPage("investor-create-case");
        }}
        onOpen={(caseId) => {
          setInvestorCaseIdIssue(null);
          setSelectedCaseId(caseId);
          setPage("investor-case");
        }}
      />
    );
  } else if (page === "investor-create-case") {
    content = (
      <CreateCaseView
        client={client}
        onBack={() => setPage("investor-demo")}
        onCreated={(caseId) => {
          setInvestorCaseIdIssue(null);
          setSelectedCaseId(caseId);
          setPage("investor-case");
        }}
      />
    );
  } else if (page === "investor-case" && selectedCaseId) {
    content = <CaseWorkspaceView client={client} caseId={selectedCaseId} guided onBack={() => {
      setSelectedCaseId(null);
      setInvestorCaseIdIssue(null);
      setPage("investor-demo");
    }} />;
  } else if (page === "cases") {
    content = (
      <CaseListView
        client={client}
        onCreate={() => setPage("create-case")}
        onOpen={(caseId) => {
          setSelectedCaseId(caseId);
          setPage("case");
        }}
      />
    );
  } else if (page === "create-case") {
    content = (
      <CreateCaseView
        client={client}
        onBack={() => setPage("cases")}
        onCreated={(caseId) => {
          setSelectedCaseId(caseId);
          setPage("case");
        }}
      />
    );
  } else if (page === "case" && selectedCaseId) {
    content = <CaseWorkspaceView client={client} caseId={selectedCaseId} onBack={() => setPage("cases")} />;
  } else if (page === "history") {
    content = <HistoryPanel client={client} onOpenAnalysis={(id) => { setAnalysisId(id); setPage("analysis"); }} onRerunAccepted={(id) => { setAnalysisId(id); setPage("analysis"); }} />;
  } else if (page === "evaluation") {
    content = <EvaluationDashboard client={client} />;
  } else if (page === "mapillary-demo") {
    content = <MapillaryDemoView />;
  } else if (page === "dataset-qa" && operatorUi) {
    content = <DatasetQaDashboard client={client} />;
  } else if (page === "providers" && operatorUi) {
    content = <ProviderDashboard client={client} />;
  } else if (!analysisId) {
    content = (
      <UploadPanel
        file={file}
        previewUrl={previewUrl}
        validationError={validationError}
        onFile={(selected) => void handleFile(selected)}
        mode={mode}
        onMode={(nextMode) => {
          setMode(nextMode);
          if (nextMode === "local_only") {
            setCloudConsent(false);
            setAllowCloudAssist(false);
          }
        }}
        authorization={authorization}
        onAuthorization={setAuthorization}
        cloudConsent={cloudConsent}
        onCloudConsent={setCloudConsent}
        allowCloudAssist={allowCloudAssist}
        onAllowCloudAssist={setAllowCloudAssist}
        capabilities={capabilitiesQuery.data}
        capabilitiesLoading={capabilitiesQuery.isPending}
        capabilitiesFailureMessage={
          capabilitiesQuery.isError
            ? serverMessage(messageCode(capabilitiesQuery.error) ?? "errors.capabilities")
            : undefined
        }
        cloudAvailable={cloudAvailable}
        submitting={createMutation.isPending}
        uploadProgress={uploadProgress}
        canSubmit={canSubmit}
        onSubmit={submit}
        submitError={submitError}
        deletedNotice={deletedNotice}
      />
    );
  } else if (analysisQuery.isError) {
    content = <ErrorState message={serverMessage(messageCode(analysisQuery.error))} retryable onBack={reset} />;
  } else if (!analysis || analysis.status === "queued" || analysis.status === "processing") {
    content = (
      <ProcessingPanel
        progress={analysisQuery.progress}
        streamState={analysisQuery.streamState}
        analysis={analysis}
        previewUrl={previewUrl}
        onCancel={deleteCurrent}
        deleting={deleteMutation.isPending}
      />
    );
  } else if (analysis.status === "failed") {
    content = (
      <ErrorState
        message={serverMessage(analysis.failure?.message_key ?? analysis.failure?.code)}
        retryable={analysis.failure?.retryable ?? false}
        onBack={reset}
      />
    );
  } else if (analysis.status === "deleted") {
    content = <ErrorState message={serverMessage("analysis_not_found")} retryable={false} onBack={reset} />;
  } else if (analysis.candidates.length === 0) {
    content = <NoSignal analysis={analysis} onDelete={deleteCurrent} deleting={deleteMutation.isPending} onNew={beginNewAnalysis} />;
  } else {
    content = <ResultView analysis={analysis} onDelete={deleteCurrent} deleting={deleteMutation.isPending} onNew={beginNewAnalysis} />;
  }

  const providerValues = capabilitiesQuery.data
    ? Object.values(capabilitiesQuery.data.providers).filter((provider): provider is NonNullable<typeof provider> => provider !== undefined)
    : [];
  const capabilityDegraded = providerValues.some((provider) => provider.enabled && (!provider.available || (provider.operational_status && provider.operational_status !== "ready")));
  const serviceKey: TranslationKey = capabilitiesQuery.isPending ? "status.checking" : capabilitiesQuery.isError ? "status.unavailable" : capabilityDegraded ? "status.degraded" : "status.connected";
  const serviceOn = capabilitiesQuery.isSuccess;
  const investorWorkspace = page === "investor-case";

  return (
    <div className={`app-shell${investorWorkspace ? " app-shell--investor" : ""}`}>
      <a className="skip-link" href="#main-content">{t("app.skip")}</a>
      <header className={`site-header${investorWorkspace ? " site-header--investor" : ""}`}>
        <a className="brand" href="#analysis" aria-label={t("app.name")} onClick={(event) => { event.preventDefault(); setPage("analysis"); }}>
          <span className="brand-mark" aria-hidden="true"><i /><i /></span>
          <span><strong>{t("app.name")}</strong><small>{investorWorkspace ? (locale === "tr" ? "Konum inceleme alanı" : "Location review workspace") : t("app.phase")}</small></span>
        </a>
        {!investorWorkspace ? <nav className="workspace-nav" aria-label={t("nav.label")}>
          <button type="button" aria-current={page === "analysis" ? "page" : undefined} onClick={() => setPage("analysis")}>{t(analysisId ? "nav.analysis" : "nav.new")}</button>
          <button type="button" aria-current={["investor-demo", "investor-create-case", "investor-case"].includes(page) ? "page" : undefined} onClick={() => {
            setSelectedCaseId(null);
            setInvestorCaseIdIssue(null);
            setPage("investor-demo");
          }}>{investorDemoNavLabel(locale)}</button>
          <button type="button" aria-current={["cases", "create-case", "case"].includes(page) ? "page" : undefined} onClick={() => setPage("cases")}>{t("nav.cases")}</button>
          <button type="button" aria-current={page === "history" ? "page" : undefined} onClick={() => setPage("history")}>{t("nav.history")}</button>
          <button type="button" aria-current={page === "evaluation" ? "page" : undefined} onClick={() => setPage("evaluation")}>{t("nav.evaluation")}</button>
          <button type="button" aria-current={page === "mapillary-demo" ? "page" : undefined} onClick={() => setPage("mapillary-demo")}>Mapillary demo</button>
          {operatorUi ? <button type="button" aria-current={page === "dataset-qa" ? "page" : undefined} onClick={() => setPage("dataset-qa")}>{t("nav.datasetQa")}</button> : null}
          {operatorUi ? <button type="button" aria-current={page === "providers" ? "page" : undefined} onClick={() => setPage("providers")}>{t("nav.providers")}</button> : null}
        </nav> : <div className="investor-header-status"><span>{locale === "tr" ? "Özel oturum" : "Private session"}</span><strong>{locale === "tr" ? "Ankara referans pilotu" : "Ankara reference pilot"}</strong></div>}
        <div className="header-actions">
          <span className={`service-state ${serviceOn ? "service-state--on" : ""}`} role="status">
            <i aria-hidden="true" />
            {t(serviceKey)}
          </span>
          <LanguageSwitcher />
          {investorWorkspace ? <button type="button" className="investor-new-analysis" onClick={() => {
            setSelectedCaseId(null);
            setInvestorCaseIdIssue(null);
            setPage("investor-demo");
          }}>{locale === "tr" ? "Yeni analiz" : "New analysis"}</button> : null}
        </div>
      </header>
      {deleteError ? <div className="global-alert notice notice--error" role="alert">{t("errors.delete")}</div> : null}
      {content}
      {!investorWorkspace ? <footer className="site-footer"><span>{t("app.tagline")}</span><span>{t("app.phase")} · v{capabilitiesQuery.data?.version ?? "0.1"}</span></footer> : null}
    </div>
  );
}

export function AtlasLensApp({ client = apiClient, initialLocale }: AtlasLensAppProps) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: { refetchOnWindowFocus: false },
          mutations: { retry: false },
        },
      }),
  );
  return (
    <QueryClientProvider client={queryClient}>
      <I18nProvider initialLocale={initialLocale ?? requestedLocale()}>
        <AppContent client={client} />
      </I18nProvider>
    </QueryClientProvider>
  );
}
