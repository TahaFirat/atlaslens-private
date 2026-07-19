/* eslint-disable react-refresh/only-export-components -- guided components share one bounded copy and route contract */
import { useQuery } from "@tanstack/react-query";
import type { AtlasLensApiClient } from "../api/client";
import type {
  Capabilities,
  CaseEvidenceRecord,
  InvestigationCase,
  ProviderCapability,
} from "../api/schemas";
import { humanizeToken } from "../display";
import { useI18n, type Locale } from "../i18n";
import { safeOperatorText } from "../safe-display";
import { purposeLabel, sensitivityLabel } from "../cases/copy";

const demoCopy = {
  en: {
    nav: "Investor demo",
    eyebrow: "Private investor walkthrough",
    title: "From authorized source to reviewable location hypotheses",
    intro: "A 3–5 minute guided view of the existing local case, evidence, map, adjudication, and audit workflow.",
    ankaraPilot: "Ankara pilot only",
    localDefault: "Local analysis by default",
    duration: "3–5 minute flow",
    preparedTitle: "Prepared demo case",
    preparedIntro: "The launcher prepares this case through the standard case and evidence services. No result is embedded in this screen.",
    preparedMissing: "This URL does not identify a prepared case. Restart the launcher for an exact link, or select an authorized case below.",
    preparedInvalid: "The caseId in this URL is malformed and was not sent to the API. Use the launcher link or select an authorized case below.",
    newCase: "New investigation",
    otherCases: "Other authorized cases",
    openCase: "Open in guided mode",
    loadingCases: "Checking prepared case…",
    caseFailure: "Cases could not be read. The demo is not ready yet.",
    retry: "Try again",
    capabilityTitle: "Runtime readiness",
    capabilityIntro: "These states come from the current server. Disabled providers are not presented as successful work.",
    localProviders: "Local providers reported ready",
    localProvidersUnavailable: "No local provider is currently reported ready.",
    nvidiaTitle: "NVIDIA vision review",
    nvidiaReady: "Optional / experimental · available",
    nvidiaUnavailable: "Optional provider unavailable",
    nvidiaChecking: "Optional provider status is being checked",
    nvidiaHelp: "NVIDIA is off by default and never blocks the local case workflow. Cloud use requires explicit selection and consent.",
    stepsLabel: "Investor demo steps",
    steps: [
      "Case",
      "Authority and purpose",
      "Local analysis",
      "Providers and sources",
      "Map candidates",
      "Evidence groups",
      "Operator adjudication",
      "Audit integrity",
      "Limitations",
    ],
    sourceTitle: "Source, authority, and purpose",
    sourceIntro: "The investigation context is recorded before media analysis.",
    sourceContext: "Source context",
    authorization: "Authority attested",
    authorized: "Yes — recorded on the case",
    unauthorized: "No — do not continue analysis",
    purpose: "Legitimate purpose",
    retention: "Retention policy",
    scope: "Product scope",
    scopeValue: "Ankara private pilot; this is not a result or ground truth.",
    providerTitle: "Provider health and sources used",
    providerIntro: "Capability is separate from actual use. The source list below is derived only from materialized case evidence.",
    localRuntime: "Local runtime",
    available: "Available",
    unavailable: "Unavailable",
    disabled: "Disabled",
    checking: "Checking",
    usedSources: "Sources used in this case",
    noUsedSources: "No materialized evidence source was returned. AtlasLens does not invent one.",
    providerFamily: "family",
    localContinues: "The local analysis and review remain available; this is not an overall analysis failure.",
    limitationsTitle: "What does this not prove?",
    limitationsIntro: "Keep these boundaries visible during the investor conversation.",
    limitations: [
      "This is an Ankara pilot. There is no Turkey-wide accuracy claim.",
      "Candidate centers and positive uncertainty circles are hypotheses, not supplied ground truth or confirmed locations.",
      "NVIDIA is optional and experimental; it is disabled by default and may be unavailable.",
      "Confidence values or labels are not calibrated probabilities unless a result explicitly says otherwise.",
      "The audit hash chain is an application-integrity check, not legally certified evidence.",
    ],
  },
  tr: {
    nav: "Yatırımcı demosu",
    eyebrow: "Özel yatırımcı anlatımı",
    title: "Yetkili kaynaktan incelenebilir konum hipotezlerine",
    intro: "Mevcut yerel vaka, kanıt, harita, operatör değerlendirmesi ve audit akışının 3–5 dakikalık rehberli görünümü.",
    ankaraPilot: "Yalnızca Ankara pilotu",
    localDefault: "Varsayılan yalnızca yerel analiz",
    duration: "3–5 dakikalık akış",
    preparedTitle: "Hazır demo vakası",
    preparedIntro: "Başlatıcı bu vakayı standart vaka ve kanıt servisleri üzerinden hazırlar. Bu ekrana hiçbir sonuç gömülmez.",
    preparedMissing: "Bu URL hazırlanmış bir vakayı tanımlamıyor. Kesin bağlantı için başlatıcıyı yeniden çalıştırın veya aşağıdan yetkili bir vaka seçin.",
    preparedInvalid: "Bu URL'deki caseId geçersiz biçimde ve API'ye gönderilmedi. Başlatıcı bağlantısını kullanın veya aşağıdan yetkili bir vaka seçin.",
    newCase: "Yeni inceleme",
    otherCases: "Diğer yetkili vakalar",
    openCase: "Rehberli modda aç",
    loadingCases: "Hazır vaka denetleniyor…",
    caseFailure: "Vakalar okunamadı. Demo henüz hazır değil.",
    retry: "Yeniden dene",
    capabilityTitle: "Çalışma zamanı hazırlığı",
    capabilityIntro: "Bu durumlar mevcut sunucudan gelir. Devre dışı sağlayıcılar başarılı iş gibi gösterilmez.",
    localProviders: "Hazır bildirilen yerel sağlayıcılar",
    localProvidersUnavailable: "Şu anda hazır bildirilen yerel sağlayıcı yok.",
    nvidiaTitle: "NVIDIA görsel inceleme",
    nvidiaReady: "Opsiyonel / deneysel · kullanılabilir",
    nvidiaUnavailable: "Opsiyonel sağlayıcı kullanılamıyor",
    nvidiaChecking: "Opsiyonel sağlayıcı durumu denetleniyor",
    nvidiaHelp: "NVIDIA varsayılan olarak kapalıdır ve yerel vaka akışını engellemez. Bulut kullanımı açık seçim ve onay gerektirir.",
    stepsLabel: "Yatırımcı demo adımları",
    steps: [
      "Vaka",
      "Yetki ve amaç",
      "Yerel analiz",
      "Sağlayıcılar ve kaynaklar",
      "Harita adayları",
      "Kanıt grupları",
      "Operatör değerlendirmesi",
      "Audit bütünlüğü",
      "Sınırlamalar",
    ],
    sourceTitle: "Kaynak, yetki ve amaç",
    sourceIntro: "İnceleme bağlamı medya analizinden önce kaydedilir.",
    sourceContext: "Kaynak bağlamı",
    authorization: "Yetki beyanı",
    authorized: "Evet — vaka üzerinde kayıtlı",
    unauthorized: "Hayır — analize devam etmeyin",
    purpose: "Meşru amaç",
    retention: "Saklama politikası",
    scope: "Ürün kapsamı",
    scopeValue: "Ankara özel pilotu; bu bir sonuç veya ground truth değildir.",
    providerTitle: "Provider health ve kullanılan kaynaklar",
    providerIntro: "Kabiliyet ile fiilî kullanım ayrıdır. Aşağıdaki kaynak listesi yalnızca vakaya dönüştürülmüş kanıttan türetilir.",
    localRuntime: "Yerel çalışma zamanı",
    available: "Kullanılabilir",
    unavailable: "Kullanılamıyor",
    disabled: "Devre dışı",
    checking: "Denetleniyor",
    usedSources: "Bu vakada kullanılan kaynaklar",
    noUsedSources: "Vaka için dönüştürülmüş kanıt kaynağı döndürülmedi. AtlasLens kaynak uydurmaz.",
    providerFamily: "aile",
    localContinues: "Yerel analiz ve inceleme kullanılabilir kalır; bu, genel analiz hatası değildir.",
    limitationsTitle: "Bu neyi kanıtlamıyor?",
    limitationsIntro: "Yatırımcı anlatımı boyunca bu sınırları görünür tutun.",
    limitations: [
      "Bu bir Ankara pilotudur. Türkiye-geneli doğruluk iddiası bulunmamaktadır.",
      "Aday merkezleri ve pozitif belirsizlik çemberleri hipotezdir; verilmiş ground truth veya doğrulanmış konum değildir.",
      "NVIDIA opsiyonel ve deneyseldir; varsayılan olarak kapalıdır ve kullanılamayabilir.",
      "Confidence değerleri veya etiketleri, sonuç açıkça aksini söylemedikçe kalibre edilmiş olasılık değildir.",
      "Audit özet zinciri bir uygulama bütünlüğü kontrolüdür; hukuken sertifikalı delil değildir.",
    ],
  },
} as const;

function copy(locale: Locale) {
  return demoCopy[locale];
}

function providerReady(provider: ProviderCapability): boolean {
  return provider.enabled
    && provider.available
    && (!provider.operational_status || provider.operational_status === "ready");
}

function nvidiaCapability(capabilities: Capabilities | undefined): ProviderCapability | null {
  const cloud = capabilities?.providers.cloud_vision;
  return cloud?.provider_id.toLocaleLowerCase().includes("nvidia") ? cloud : null;
}

function localProviders(capabilities: Capabilities | undefined): Array<[string, ProviderCapability]> {
  if (!capabilities) return [];
  return Object.entries(capabilities.providers)
    .filter((entry): entry is [string, ProviderCapability] => Boolean(entry[1]?.execution_boundary === "local"))
    .filter(([, provider]) => provider.enabled || provider.available)
    .sort(([left], [right]) => left.localeCompare(right));
}

function providerState(provider: ProviderCapability, locale: Locale): string {
  const text = copy(locale);
  if (!provider.enabled) return text.disabled;
  if (!providerReady(provider)) {
    const detail = safeOperatorText(provider.operational_status ?? provider.reason_code);
    return detail ? `${text.unavailable} · ${humanizeToken(detail)}` : text.unavailable;
  }
  return text.available;
}

function NvidiaState({ capabilities, loading = false }: { capabilities?: Capabilities; loading?: boolean }) {
  const { locale } = useI18n();
  const text = copy(locale);
  const nvidia = nvidiaCapability(capabilities);
  const ready = Boolean(nvidia && providerReady(nvidia));
  const label = loading && !capabilities
    ? text.nvidiaChecking
    : ready
      ? text.nvidiaReady
      : text.nvidiaUnavailable;
  return (
    <article className={`investor-provider-card investor-provider-card--${ready ? "ready" : "optional"}`}>
      <header>
        <strong>{text.nvidiaTitle}</strong>
        <span data-testid="investor-nvidia-state">{label}</span>
      </header>
      <p>{text.nvidiaHelp}</p>
      {!ready && !loading ? <small>{text.localContinues}</small> : null}
    </article>
  );
}

export function InvestorDemoView({
  client,
  capabilities,
  capabilitiesLoading,
  caseIdIssue,
  onOpen,
  onCreate,
}: {
  client: AtlasLensApiClient;
  capabilities?: Capabilities;
  capabilitiesLoading: boolean;
  caseIdIssue: "missing" | "invalid" | null;
  onOpen: (caseId: string) => void;
  onCreate: () => void;
}) {
  const { locale } = useI18n();
  const text = copy(locale);
  const cases = useQuery({
    queryKey: ["cases"],
    queryFn: () => client.listCases({ limit: 100, offset: 0 }),
    retry: false,
  });
  const otherCases = cases.data?.items ?? [];
  const readyLocalCount = localProviders(capabilities).filter(([, provider]) => providerReady(provider)).length;

  return (
    <main id="main-content" className="workspace-shell investor-demo-landing">
      <header className="investor-demo-hero">
        <div>
          <p className="eyebrow">{text.eyebrow}</p>
          <h1>{text.title}</h1>
          <p>{text.intro}</p>
        </div>
        <div className="investor-demo-scope" aria-label={text.ankaraPilot}>
          <span>{text.ankaraPilot}</span>
          <span>{text.localDefault}</span>
          <span>{text.duration}</span>
        </div>
      </header>

      <InvestorDemoSteps landing />

      <div className="investor-demo-entry-grid">
        <section className="investor-demo-entry" aria-labelledby="prepared-demo-heading">
          <p className="eyebrow">01</p>
          <h2 id="prepared-demo-heading">{text.preparedTitle}</h2>
          <p>{text.preparedIntro}</p>
          {cases.isPending ? <p role="status">{text.loadingCases}</p> : null}
          {cases.isError ? (
            <div className="notice notice--error" role="alert">
              <p>{text.caseFailure}</p>
              <button type="button" className="secondary-button" onClick={() => void cases.refetch()}>{text.retry}</button>
            </div>
          ) : null}
          {caseIdIssue ? (
            <p className="notice notice--warning" role="status" data-testid="investor-route-notice">
              {caseIdIssue === "invalid" ? text.preparedInvalid : text.preparedMissing}
            </p>
          ) : null}
          <button type="button" className="secondary-button investor-new-case" onClick={onCreate}>{text.newCase}</button>
          {otherCases.length > 0 ? (
            <details className="investor-other-cases" data-testid="investor-authorized-cases">
              <summary>{text.otherCases} ({otherCases.length})</summary>
              <ul>
                {otherCases.map((item) => (
                  <li key={item.id}>
                    <span>{item.title}</span>
                    <button type="button" className="text-button" data-testid="investor-authorized-case-open" onClick={() => onOpen(item.id)}>{text.openCase}</button>
                  </li>
                ))}
              </ul>
            </details>
          ) : null}
        </section>

        <section className="investor-demo-readiness" aria-labelledby="demo-readiness-heading">
          <p className="eyebrow">{text.localDefault}</p>
          <h2 id="demo-readiness-heading">{text.capabilityTitle}</h2>
          <p>{text.capabilityIntro}</p>
          <article className="investor-provider-card investor-provider-card--local">
            <header>
              <strong>{text.localProviders}</strong>
              <span>{capabilitiesLoading && !capabilities ? text.checking : readyLocalCount}</span>
            </header>
            {!capabilitiesLoading && readyLocalCount === 0 ? <small>{text.localProvidersUnavailable}</small> : null}
          </article>
          <NvidiaState capabilities={capabilities} loading={capabilitiesLoading} />
        </section>
      </div>

      <InvestorLimitations />
    </main>
  );
}

export function InvestorDemoSteps({ landing = false }: { landing?: boolean }) {
  const { locale } = useI18n();
  const text = copy(locale);
  const anchors = landing
    ? ["prepared-demo-heading", "prepared-demo-heading", "demo-readiness-heading", "demo-readiness-heading", "prepared-demo-heading", "prepared-demo-heading", "prepared-demo-heading", "prepared-demo-heading", "demo-limitations"]
    : ["demo-case", "demo-source", "demo-analysis", "demo-providers", "demo-map", "case-evidence-title", "demo-adjudication", "demo-audit", "demo-limitations"];
  return (
    <nav className="investor-demo-steps" aria-label={text.stepsLabel}>
      <ol>
        {text.steps.map((step, index) => (
          <li key={step}>
            <button
              type="button"
              onClick={() => document.getElementById(anchors[index] ?? "")?.scrollIntoView({ behavior: "smooth", block: "start" })}
            >
              <span>{String(index + 1).padStart(2, "0")}</span>
              {step}
            </button>
          </li>
        ))}
      </ol>
    </nav>
  );
}

export function InvestorSourcePanel({ item }: { item: InvestigationCase }) {
  const { locale } = useI18n();
  const text = copy(locale);
  return (
    <section id="demo-source" className="investor-guide-panel" aria-labelledby="demo-source-heading">
      <header>
        <div><p className="eyebrow">02 · {text.sourceContext}</p><h2 id="demo-source-heading">{text.sourceTitle}</h2></div>
        <span className={`integrity-badge integrity-badge--${item.authorization_attested ? "valid" : "invalid"}`}>
          {item.authorization_attested ? text.authorized : text.unauthorized}
        </span>
      </header>
      <p>{text.sourceIntro}</p>
      <dl className="investor-guide-facts">
        <div><dt>{text.sourceContext}</dt><dd>{item.source_context}</dd></div>
        <div><dt>{text.purpose}</dt><dd>{purposeLabel(item.purpose, locale)}</dd></div>
        <div><dt>{text.authorization}</dt><dd>{item.authorization_attested ? text.authorized : text.unauthorized}</dd></div>
        <div><dt>{text.retention}</dt><dd>{humanizeToken(item.retention_policy)}</dd></div>
        <div><dt>{text.scope}</dt><dd>{text.scopeValue}</dd></div>
        <div><dt>{locale === "tr" ? "Hassasiyet" : "Sensitivity"}</dt><dd>{sensitivityLabel(item.sensitivity, locale)}</dd></div>
      </dl>
    </section>
  );
}

export function InvestorProviderPanel({
  capabilities,
  loading,
  evidence,
}: {
  capabilities?: Capabilities;
  loading: boolean;
  evidence: CaseEvidenceRecord[];
}) {
  const { locale } = useI18n();
  const text = copy(locale);
  const providers = localProviders(capabilities);
  const sources = Array.from(
    new Map(
      evidence.map((item) => [`${item.provider}\u0000${item.provider_family}`, { provider: item.provider, family: item.provider_family }]),
    ).values(),
  );
  return (
    <section id="demo-providers" className="investor-guide-panel" aria-labelledby="demo-providers-heading">
      <header><div><p className="eyebrow">04 · {text.localRuntime}</p><h2 id="demo-providers-heading">{text.providerTitle}</h2></div></header>
      <p>{text.providerIntro}</p>
      <div className="investor-provider-grid">
        {loading && !capabilities ? <p role="status">{text.checking}</p> : providers.map(([name, provider]) => (
          <article className={`investor-provider-card investor-provider-card--${providerReady(provider) ? "ready" : "unavailable"}`} key={name}>
            <header><strong>{safeOperatorText(provider.provider_id) ?? name}</strong><span>{providerState(provider, locale)}</span></header>
            <small>{text.localRuntime}</small>
          </article>
        ))}
        {!loading && providers.length === 0 ? <p className="workspace-empty">{text.localProvidersUnavailable}</p> : null}
        <NvidiaState capabilities={capabilities} loading={loading} />
      </div>
      <div className="investor-used-sources">
        <h3>{text.usedSources}</h3>
        {sources.length === 0 ? <p className="workspace-empty">{text.noUsedSources}</p> : (
          <ul>
            {sources.map((source) => (
              <li key={`${source.provider}:${source.family}`}>
                <strong>{source.provider}</strong><span>{text.providerFamily}: {source.family}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}

export function InvestorLimitations() {
  const { locale } = useI18n();
  const text = copy(locale);
  return (
    <section id="demo-limitations" className="investor-limitations" aria-labelledby="demo-limitations-heading">
      <header><div><p className="eyebrow">09 · {text.ankaraPilot}</p><h2 id="demo-limitations-heading">{text.limitationsTitle}</h2></div></header>
      <p>{text.limitationsIntro}</p>
      <ul>{text.limitations.map((item) => <li key={item}>{item}</li>)}</ul>
    </section>
  );
}

export function investorDemoNavLabel(locale: Locale): string {
  return copy(locale).nav;
}
