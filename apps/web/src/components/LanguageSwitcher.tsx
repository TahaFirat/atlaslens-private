import { useI18n, type Locale } from "../i18n";

export function LanguageSwitcher() {
  const { locale, setLocale, t } = useI18n();
  const options: Locale[] = ["en", "tr"];
  return (
    <div className="language-switcher" role="group" aria-label={t("language.label")}>
      {options.map((option) => (
        <button
          className={locale === option ? "language-option language-option--active" : "language-option"}
          type="button"
          aria-pressed={locale === option}
          onClick={() => setLocale(option)}
          key={option}
        >
          {t(option === "en" ? "language.en" : "language.tr")}
        </button>
      ))}
    </div>
  );
}
