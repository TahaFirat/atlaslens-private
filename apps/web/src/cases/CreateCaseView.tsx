import { useState, type FormEvent } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import type { AtlasLensApiClient, CreateCaseInput } from "../api/client";
import type { CasePurpose, CaseSensitivity } from "../api/schemas";
import { useI18n } from "../i18n";
import {
  caseCopy,
  LOCAL_ANALYST_ID,
  purposeLabel,
  sensitivityLabel,
} from "./copy";

interface CreateCaseViewProps {
  client: AtlasLensApiClient;
  onBack: () => void;
  onCreated: (caseId: string) => void;
}

interface FormErrors {
  title?: string;
  source?: string;
  purposeDetail?: string;
  authorization?: string;
}

const purposes: CasePurpose[] = [
  "journalism",
  "humanitarian",
  "disaster_response",
  "insurance",
  "authorized_security_research",
  "other",
];
const sensitivities: CaseSensitivity[] = ["standard", "sensitive", "conflict_related"];

export function CreateCaseView({ client, onBack, onCreated }: CreateCaseViewProps) {
  const { locale } = useI18n();
  const text = caseCopy[locale];
  const queryClient = useQueryClient();
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [purpose, setPurpose] = useState<CasePurpose>("journalism");
  const [purposeDetail, setPurposeDetail] = useState("");
  const [sensitivity, setSensitivity] = useState<CaseSensitivity>("standard");
  const [sourceContext, setSourceContext] = useState("");
  const [retentionPolicy, setRetentionPolicy] = useState("analysis_metadata_only");
  const [authorization, setAuthorization] = useState(false);
  const [errors, setErrors] = useState<FormErrors>({});

  const mutation = useMutation({
    mutationFn: (input: CreateCaseInput) => client.createCase(input),
    onSuccess: (created) => {
      void queryClient.invalidateQueries({ queryKey: ["cases"] });
      onCreated(created.id);
    },
  });

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const nextErrors: FormErrors = {};
    if (!title.trim()) nextErrors.title = text.required;
    if (!sourceContext.trim()) nextErrors.source = text.required;
    if (purpose === "other" && !purposeDetail.trim()) {
      nextErrors.purposeDetail = text.purposeDetailRequired;
    }
    if (!authorization) nextErrors.authorization = text.authorizationRequired;
    setErrors(nextErrors);
    if (Object.keys(nextErrors).length > 0) return;

    mutation.mutate({
      title: title.trim(),
      description: description.trim() || null,
      purpose,
      purpose_detail: purpose === "other" ? purposeDetail.trim() : null,
      source_context: sourceContext.trim(),
      sensitivity,
      authorization_attested: true,
      retention_policy: retentionPolicy,
      created_by_actor_id: LOCAL_ANALYST_ID,
    });
  };

  return (
    <main id="main-content" className="workspace-shell create-case-view">
      <button type="button" className="text-button case-back" onClick={onBack}>
        ← {text.backToCases}
      </button>
      <header className="workspace-heading">
        <p className="eyebrow">{text.eyebrow}</p>
        <h1>{text.createTitle}</h1>
        <p>{text.createIntro}</p>
      </header>

      <form className="case-form" noValidate onSubmit={submit}>
        <div className="form-field case-form__wide">
          <label htmlFor="case-title">{text.title}</label>
          <input
            id="case-title"
            value={title}
            maxLength={160}
            aria-invalid={Boolean(errors.title)}
            aria-describedby={errors.title ? "case-title-error" : undefined}
            onChange={(event) => setTitle(event.target.value)}
          />
          {errors.title ? <span id="case-title-error" className="field-error">{errors.title}</span> : null}
        </div>

        <div className="form-field case-form__wide">
          <label htmlFor="case-description">{text.description}</label>
          <textarea
            id="case-description"
            value={description}
            maxLength={2000}
            rows={3}
            onChange={(event) => setDescription(event.target.value)}
          />
        </div>

        <div className="form-field">
          <label htmlFor="case-purpose">{text.purpose}</label>
          <select
            id="case-purpose"
            value={purpose}
            onChange={(event) => setPurpose(event.target.value as CasePurpose)}
          >
            {purposes.map((value) => <option key={value} value={value}>{purposeLabel(value, locale)}</option>)}
          </select>
        </div>

        <div className="form-field">
          <label htmlFor="case-sensitivity">{text.sensitivity}</label>
          <select
            id="case-sensitivity"
            value={sensitivity}
            onChange={(event) => setSensitivity(event.target.value as CaseSensitivity)}
          >
            {sensitivities.map((value) => <option key={value} value={value}>{sensitivityLabel(value, locale)}</option>)}
          </select>
        </div>

        {purpose === "other" ? (
          <div className="form-field case-form__wide">
            <label htmlFor="case-purpose-detail">{text.purposeDetail}</label>
            <textarea
              id="case-purpose-detail"
              value={purposeDetail}
              maxLength={500}
              rows={2}
              aria-invalid={Boolean(errors.purposeDetail)}
              aria-describedby={errors.purposeDetail ? "case-purpose-error" : undefined}
              onChange={(event) => setPurposeDetail(event.target.value)}
            />
            {errors.purposeDetail ? <span id="case-purpose-error" className="field-error">{errors.purposeDetail}</span> : null}
          </div>
        ) : null}

        <div className="form-field case-form__wide">
          <label htmlFor="case-source-context">{text.sourceContext}</label>
          <textarea
            id="case-source-context"
            value={sourceContext}
            maxLength={1000}
            rows={3}
            aria-invalid={Boolean(errors.source)}
            aria-describedby="case-source-hint case-source-error"
            onChange={(event) => setSourceContext(event.target.value)}
          />
          <span id="case-source-hint" className="field-hint">{text.sourceContextHint}</span>
          {errors.source ? <span id="case-source-error" className="field-error">{errors.source}</span> : null}
        </div>

        <div className="form-field">
          <label htmlFor="case-retention">{text.retention}</label>
          <select
            id="case-retention"
            value={retentionPolicy}
            onChange={(event) => setRetentionPolicy(event.target.value)}
          >
            <option value="analysis_metadata_only">{locale === "tr" ? "Yalnızca analiz meta verisi" : "Analysis metadata only"}</option>
            <option value="30_days">{locale === "tr" ? "30 gün" : "30 days"}</option>
            <option value="90_days">{locale === "tr" ? "90 gün" : "90 days"}</option>
            <option value="case_lifetime">{locale === "tr" ? "Vaka yaşam döngüsü" : "Case lifetime"}</option>
          </select>
        </div>

        {sensitivity === "conflict_related" ? (
          <p className="notice notice--warning case-form__wide" role="alert">{text.conflictCreateWarning}</p>
        ) : null}

        <label className="consent-row case-form__wide">
          <input
            type="checkbox"
            checked={authorization}
            aria-invalid={Boolean(errors.authorization)}
            onChange={(event) => setAuthorization(event.target.checked)}
          />
          <span><strong>{text.authorization}</strong><small>{text.localIdentity}</small></span>
        </label>
        {errors.authorization ? <p className="field-error case-form__wide">{errors.authorization}</p> : null}

        {mutation.isError ? <p className="notice notice--error case-form__wide" role="alert">{text.createFailed}</p> : null}
        <div className="case-form__actions case-form__wide">
          <button type="button" className="secondary-button" onClick={onBack}>{text.backToCases}</button>
          <button type="submit" className="primary-button" disabled={mutation.isPending}>
            {mutation.isPending ? text.saving : text.createCase}
          </button>
        </div>
      </form>
    </main>
  );
}
