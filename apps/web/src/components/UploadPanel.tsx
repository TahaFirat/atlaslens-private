import { useRef, type ChangeEvent, type DragEvent, type KeyboardEvent } from "react";
import type { AnalysisMode, Capabilities } from "../api/schemas";
import type { FileValidationCode } from "../file-validation";
import { useI18n, type TranslationKey } from "../i18n";
import { CapabilityPanel } from "./CapabilityPanel";

interface UploadPanelProps {
  file: File | null;
  previewUrl: string | null;
  validationError: FileValidationCode | null;
  onFile: (file: File) => void;
  mode: AnalysisMode;
  onMode: (mode: AnalysisMode) => void;
  authorization: boolean;
  onAuthorization: (checked: boolean) => void;
  cloudConsent: boolean;
  onCloudConsent: (checked: boolean) => void;
  allowCloudAssist: boolean;
  onAllowCloudAssist: (checked: boolean) => void;
  capabilities?: Capabilities;
  capabilitiesLoading: boolean;
  capabilitiesFailureMessage?: string;
  cloudAvailable: boolean;
  submitting: boolean;
  uploadProgress: number;
  canSubmit: boolean;
  onSubmit: () => void;
  submitError: string | null;
  deletedNotice: boolean;
}

function formatBytes(bytes: number): string {
  return `${Math.max(1, Math.round(bytes / 1024 / 1024))} MB`;
}

export function UploadPanel(props: UploadPanelProps) {
  const { t } = useI18n();
  const inputRef = useRef<HTMLInputElement>(null);
  const choose = () => {
    if (!props.submitting) inputRef.current?.click();
  };
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (props.submitting) return;
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      choose();
    }
  };
  const handleInput = (event: ChangeEvent<HTMLInputElement>) => {
    const selected = event.target.files?.item(0);
    if (selected) props.onFile(selected);
  };
  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    if (props.submitting) return;
    const selected = event.dataTransfer.files.item(0);
    if (selected) props.onFile(selected);
  };
  const validationKey = props.validationError ? (`upload.validation.${props.validationError}` as TranslationKey) : null;
  const maxBytes = props.capabilities?.max_upload_bytes ?? 20 * 1024 * 1024;

  return (
    <main id="main-content" className="landing-layout">
      <section className="upload-intro" aria-labelledby="upload-heading">
        <p className="eyebrow">{t("upload.eyebrow")}</p>
        <h1 id="upload-heading">{t("upload.title")}</h1>
        <p className="lead">{t("upload.intro")}</p>
        <CapabilityPanel
          capabilities={props.capabilities}
          loading={props.capabilitiesLoading}
          failureMessage={props.capabilitiesFailureMessage}
        />
      </section>

      <section className="upload-card" aria-label={t("upload.eyebrow")}>
        {props.deletedNotice ? <div className="notice notice--success" role="status">{t("result.deleted")}</div> : null}
        <input
          className="visually-hidden"
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          onChange={handleInput}
          disabled={props.submitting}
          tabIndex={-1}
          data-testid="file-input"
        />
        <div
          className={`drop-zone ${props.file ? "drop-zone--selected" : ""}`}
          role="button"
          tabIndex={props.submitting ? -1 : 0}
          aria-disabled={props.submitting}
          aria-label={t("upload.zone")}
          aria-describedby="upload-formats upload-keyboard"
          onClick={choose}
          onKeyDown={onKeyDown}
          onDragOver={(event) => event.preventDefault()}
          onDrop={handleDrop}
        >
          {props.previewUrl ? (
            <div className="preview-wrap">
              <img src={props.previewUrl} alt={t("upload.previewAlt")} />
              <div>
                <span className="preview-label">{t("upload.selected")}</span>
                <strong>{props.file?.name}</strong>
                <small>{t("upload.change")}</small>
              </div>
            </div>
          ) : (
            <div className="drop-zone-copy">
              <span className="upload-glyph" aria-hidden="true">＋</span>
              <strong>{t("upload.zone")}</strong>
              <span id="upload-formats">{t("upload.zoneHint", { size: formatBytes(maxBytes) })}</span>
              <small id="upload-keyboard">{t("upload.keyboardHint")}</small>
            </div>
          )}
        </div>
        {validationKey ? <p className="field-error" role="alert">{t(validationKey)}</p> : null}

        <fieldset className="mode-fieldset">
          <legend>{t("mode.legend")}</legend>
          <label className={`mode-option ${props.mode === "local_only" ? "mode-option--active" : ""}`}>
            <input
              type="radio"
              name="mode"
              value="local_only"
              checked={props.mode === "local_only"}
              disabled={props.submitting}
              onChange={() => props.onMode("local_only")}
            />
            <span>
              <strong>{t("mode.local.title")} <small>{t("mode.local.badge")}</small></strong>
              <span>{t("mode.local.description")}</span>
            </span>
          </label>
          <label className={`mode-option ${props.mode === "cloud_assisted" ? "mode-option--active" : ""}`}>
            <input
              type="radio"
              name="mode"
              value="cloud_assisted"
              checked={props.mode === "cloud_assisted"}
              disabled={props.submitting || !props.cloudAvailable}
              onChange={() => props.onMode("cloud_assisted")}
            />
            <span>
              <strong>{t("mode.cloud.title")}</strong>
              <span>{t("mode.cloud.description")}</span>
            </span>
          </label>
        </fieldset>

        {!props.cloudAvailable ? (
          <div className="notice notice--warning" role="alert">{t("mode.unavailable")}</div>
        ) : null}

        <div className="consent-group">
          <label className="check-row">
            <input
              type="checkbox"
              checked={props.authorization}
              disabled={props.submitting}
              onChange={(event) => props.onAuthorization(event.target.checked)}
            />
            <span>{t("consent.authorization")}</span>
          </label>
          {props.mode === "cloud_assisted" ? (
            <>
              <label className="check-row">
                <input
                  type="checkbox"
                  checked={props.cloudConsent}
                  disabled={props.submitting}
                  onChange={(event) => props.onCloudConsent(event.target.checked)}
                />
                <span>{t("consent.cloud")}</span>
              </label>
              <label className="check-row check-row--explained">
                <input
                  type="checkbox"
                  checked={props.allowCloudAssist}
                  disabled={props.submitting}
                  onChange={(event) => props.onAllowCloudAssist(event.target.checked)}
                />
                <span className="check-row__copy">
                  <strong>{t("consent.cloudAssist")}</strong>
                  <small>{t("consent.cloudAssistPrivacy")}</small>
                </span>
              </label>
            </>
          ) : null}
        </div>

        {props.submitError ? <div className="notice notice--error" role="alert">{props.submitError}</div> : null}
        {props.submitting ? (
          <div className="upload-meter" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={props.uploadProgress}>
            <span style={{ width: `${props.uploadProgress}%` }} />
            <strong>{t("upload.uploading", { percent: props.uploadProgress })}</strong>
          </div>
        ) : null}
        <button className="primary-button" type="button" disabled={!props.canSubmit || props.submitting} onClick={props.onSubmit}>
          {t("upload.analyze")}
        </button>
        <p className="privacy-copy">{t("upload.privacy")}</p>
      </section>
    </main>
  );
}
