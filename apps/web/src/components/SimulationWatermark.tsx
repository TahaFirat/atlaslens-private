import { useI18n } from "../i18n";

export function SimulationWatermark() {
  const { t } = useI18n();
  return (
    <aside className="simulation-watermark" role="alert" aria-label={t("simulation.watermark")}>
      <strong>{t("simulation.watermark")}</strong>
      <span>{t("simulation.description")}</span>
    </aside>
  );
}
