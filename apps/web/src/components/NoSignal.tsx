import type { Analysis } from "../api/schemas";
import { useI18n } from "../i18n";
import { providerWarningKey } from "../provider-warnings";
import { Phase6BAnalysisCard } from "./Phase6BAnalysisCard";
import { Phase6CAnalysisCard } from "./Phase6CAnalysisCard";
import { SimulationWatermark } from "./SimulationWatermark";

export function NoSignal({ analysis, onDelete, deleting, onNew }: { analysis: Analysis; onDelete: () => void; deleting: boolean; onNew?: () => void }) {
  const { t, reasonMessage } = useI18n();
  const hasPhase6BDetails = Boolean(analysis.model_predictions || analysis.fusion || analysis.ocr || analysis.cloud_assist);
  const hasProviderDetails = hasPhase6BDetails || Boolean(analysis.phase6c);
  return (
    <main id="main-content" className={`state-shell ${hasProviderDetails ? "state-shell--evidence" : ""}`}>
      {analysis.result_classification === "simulated" ? <SimulationWatermark /> : null}
      <section className="state-card no-signal-card" aria-labelledby="no-signal-heading">
        <span className="state-symbol" aria-hidden="true">∅</span>
        <p className="eyebrow">{t("nosignal.eyebrow")}</p>
        <h1 id="no-signal-heading">{t("nosignal.title")}</h1>
        <p>{t("nosignal.description")}</p>
        <p className="reason-line">{t("nosignal.reason", { reason: reasonMessage(analysis.abstention?.reason_code) })}</p>
        {analysis.warnings.length > 0 ? (
          <div className="notice notice--warning">
            <strong>{t("warnings.title")}</strong>
            <ul>{analysis.warnings.map((warning, index) => <li key={`${warning}-${index}`}>{t(providerWarningKey(warning))}</li>)}</ul>
          </div>
        ) : null}
        <div className="state-actions">
          {onNew ? <button className="secondary-button" type="button" onClick={onNew}>{t("result.new")}</button> : null}
          <button className="secondary-button danger-button" type="button" disabled={deleting} onClick={onDelete}>{t("result.delete")}</button>
        </div>
      </section>
      {analysis.phase6c ? <Phase6CAnalysisCard analysis={analysis} /> : null}
      {hasPhase6BDetails ? <Phase6BAnalysisCard analysis={analysis} /> : null}
    </main>
  );
}
