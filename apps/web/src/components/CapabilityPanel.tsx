import type { Capabilities, ProviderCapability } from "../api/schemas";
import { useI18n, type TranslationKey } from "../i18n";

interface CapabilityPanelProps {
  capabilities?: Capabilities;
  loading: boolean;
  failureMessage?: string;
}

export function CapabilityPanel({ capabilities, loading, failureMessage }: CapabilityPanelProps) {
  const { t } = useI18n();
  if (loading) return <p className="muted capability-loading">{t("capabilities.loading")}</p>;
  if (!capabilities) {
    return failureMessage ? <div className="notice notice--warning" role="alert">{failureMessage}</div> : null;
  }
  const providers: Array<[TranslationKey, ProviderCapability]> = [
    ["capabilities.exif", capabilities.providers.exif],
    ["capabilities.quality", capabilities.providers.quality],
    ["capabilities.ocr", capabilities.providers.ocr],
    ["capabilities.cloudVision", capabilities.providers.cloud_vision],
  ];
  if (capabilities.providers.global_geolocation) {
    providers.push(["capabilities.globalGeolocation", capabilities.providers.global_geolocation]);
  }
  if (capabilities.providers.embedding) providers.push(["capabilities.embedding", capabilities.providers.embedding]);
  if (capabilities.providers.retrieval) providers.push(["capabilities.retrieval", capabilities.providers.retrieval]);
  if (capabilities.providers.place_research) providers.push(["capabilities.placeResearch", capabilities.providers.place_research]);
  if (capabilities.providers.map_research) providers.push(["capabilities.mapResearch", capabilities.providers.map_research]);
  if (capabilities.providers.visual_clues) providers.push(["capabilities.visualClues", capabilities.providers.visual_clues]);
  return (
    <section className="capabilities-card" aria-labelledby="capabilities-heading">
      <div>
        <p className="eyebrow">{t("capabilities.title")}</p>
        <p id="capabilities-heading" className="muted small">{t("capabilities.description")}</p>
      </div>
      <ul className="provider-list">
        {providers.map(([label, provider]) => {
          const state = !provider.enabled
            ? t("capabilities.disabled")
            : provider.operational_status
              ? t(`capabilities.status.${provider.operational_status}` as TranslationKey)
              : provider.available
                ? t("capabilities.available")
                : t("capabilities.unavailable");
          const operational = provider.enabled
            && provider.available
            && (!provider.operational_status || provider.operational_status === "ready");
          return (
            <li key={provider.provider_id}>
              <span className={`provider-dot ${operational ? "provider-dot--on" : ""}`} aria-hidden="true" />
              <span>
                <strong>{t(label)}</strong>
                <small>
                  {state} · {t(provider.execution_boundary === "cloud" ? "capabilities.cloud" : "capabilities.local")}
                </small>
                {provider.model_name ? <small>{t("capabilities.modelDetails", {
                  model: provider.model_name,
                  revision: provider.model_revision ?? t("capabilities.unknownValue"),
                  device: provider.device ?? t("capabilities.unknownValue"),
                })}</small> : null}
              </span>
            </li>
          );
        })}
      </ul>
      <p className="retention-note">
        {t(capabilities.retention.originals_deleted_after_analysis ? "capabilities.retentionDeleted" : "capabilities.retentionKept")}
      </p>
    </section>
  );
}
