import { useI18n } from "../i18n";

export function ErrorState({ message, retryable, onBack }: { message: string; retryable: boolean; onBack: () => void }) {
  const { t } = useI18n();
  return (
    <main id="main-content" className="state-shell">
      <section className="state-card error-card" aria-labelledby="error-heading">
        <span className="state-symbol" aria-hidden="true">!</span>
        <p className="eyebrow">{t("error.eyebrow")}</p>
        <h1 id="error-heading">{t("error.title")}</h1>
        <p role="alert">{message}</p>
        <p className="muted">{t(retryable ? "error.retryable" : "error.notRetryable")}</p>
        <button className="secondary-button" type="button" onClick={onBack}>{t("error.back")}</button>
      </section>
    </main>
  );
}
