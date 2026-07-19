import { useState, type FormEvent, type ReactNode } from "react";
import { useMutation } from "@tanstack/react-query";
import type { AtlasLensApiClient } from "../api/client";
import type {
  AdjudicationDecision,
  CaseEvidenceRecord,
  CaseMedia,
  LocationHypothesis,
} from "../api/schemas";
import { useI18n } from "../i18n";
import { CaseHypothesisMap } from "./CaseHypothesisMap";
import {
  caseCopy,
  decisionLabel,
  LOCAL_ANALYST_ID,
  originLabel,
} from "./copy";

interface HypothesisReviewProps {
  client: AtlasLensApiClient;
  caseId: string;
  media: CaseMedia | null;
  hypotheses: LocationHypothesis[];
  evidence: CaseEvidenceRecord[];
  onChanged: () => void;
  guided?: boolean;
  betweenMapAndReview?: ReactNode;
}

function HypothesisCard({
  client,
  caseId,
  hypothesis,
  selected,
  onSelect,
  onChanged,
}: {
  client: AtlasLensApiClient;
  caseId: string;
  hypothesis: LocationHypothesis;
  selected: boolean;
  onSelect: () => void;
  onChanged: () => void;
}) {
  const { locale } = useI18n();
  const text = caseCopy[locale];
  const [rationale, setRationale] = useState("");
  const [rationaleError, setRationaleError] = useState(false);
  const mutation = useMutation({
    mutationFn: (decision: AdjudicationDecision) =>
      client.adjudicateCaseHypothesis(caseId, hypothesis.id, {
        actor_id: LOCAL_ANALYST_ID,
        decision,
        rationale: rationale.trim(),
        supersedes_adjudication_id: hypothesis.latest_adjudication?.id ?? null,
      }),
    onSuccess: () => {
      setRationale("");
      setRationaleError(false);
      onChanged();
    },
  });

  const decide = (decision: AdjudicationDecision) => {
    if (!rationale.trim()) {
      setRationaleError(true);
      return;
    }
    mutation.mutate(decision);
  };
  const status = hypothesis.latest_adjudication?.decision ?? null;
  const place = [hypothesis.locality_name, hypothesis.region_name, hypothesis.country_code]
    .filter(Boolean)
    .join(", ");

  return (
    <article
      className={"hypothesis-card" + (selected ? " hypothesis-card--selected" : "")}
      aria-labelledby={"hypothesis-" + hypothesis.id}
    >
      <button type="button" className="hypothesis-card__select" onClick={onSelect}>
        <span className={"origin-badge origin-badge--" + hypothesis.origin}>
          {originLabel(hypothesis.origin, locale)}
        </span>
        <span className={"decision-badge decision-badge--" + (status ?? "unverified")}>
          {decisionLabel(status, locale)}
        </span>
      </button>
      <h3 id={"hypothesis-" + hypothesis.id}>
        #{hypothesis.rank ?? "—"} {place || (locale === "tr" ? "Adsız konum hipotezi" : "Unnamed location hypothesis")}
      </h3>
      <p className="hypothesis-coordinate">
        {hypothesis.latitude.toFixed(6)}, {hypothesis.longitude.toFixed(6)}
      </p>
      <dl className="hypothesis-facts">
        <div>
          <dt>{text.uncertainty}</dt>
          <dd>{Math.round(hypothesis.uncertainty_radius_m).toLocaleString(locale)} m</dd>
        </div>
        <div>
          <dt>{locale === "tr" ? "Kalibrasyon" : "Calibration"}</dt>
          <dd>
            {hypothesis.calibration_state === "uncalibrated"
              ? text.uncalibrated
              : hypothesis.calibration_state === "not_applicable"
                ? text.notApplicable
                : text.calibratedLabel + ": " + (hypothesis.confidence_label ?? "—")}
          </dd>
        </div>
        <div>
          <dt>{text.modelFamilies}</dt>
          <dd>{hypothesis.model_family_groups.join(", ") || "—"}</dd>
        </div>
        <div>
          <dt>{text.supportingEvidence}</dt>
          <dd>{hypothesis.supporting_evidence_ids.length || "—"}</dd>
        </div>
      </dl>
      {hypothesis.adjudications.length > 0 ? (
        <ol className="adjudication-history" aria-label={locale === "tr" ? "Değerlendirme geçmişi" : "Adjudication history"}>
          {hypothesis.adjudications.map((item) => (
            <li key={item.id}>
              <strong>{decisionLabel(item.decision, locale)}</strong>
              <span>{item.rationale}</span>
              <time dateTime={item.created_at}>{new Date(item.created_at).toLocaleString(locale)}</time>
            </li>
          ))}
        </ol>
      ) : null}
      <div className="hypothesis-decision">
        <label htmlFor={"rationale-" + hypothesis.id}>{text.rationale}</label>
        <textarea
          id={"rationale-" + hypothesis.id}
          value={rationale}
          rows={2}
          maxLength={2000}
          aria-invalid={rationaleError}
          onChange={(event) => {
            setRationale(event.target.value);
            if (event.target.value.trim()) setRationaleError(false);
          }}
        />
        {rationaleError ? <span className="field-error">{text.rationaleRequired}</span> : null}
        {mutation.isError ? <span className="field-error" role="alert">{text.decisionFailed}</span> : null}
        <div className="hypothesis-decision__actions">
          <button type="button" disabled={mutation.isPending} onClick={() => decide("accepted")}>{text.accept}</button>
          <button type="button" disabled={mutation.isPending} onClick={() => decide("rejected")}>{text.reject}</button>
          <button type="button" disabled={mutation.isPending} onClick={() => decide("needs_more_evidence")}>{text.requestEvidence}</button>
        </div>
      </div>
    </article>
  );
}

function OperatorCorrectionForm({
  client,
  caseId,
  media,
  evidence,
  onChanged,
}: {
  client: AtlasLensApiClient;
  caseId: string;
  media: CaseMedia | null;
  evidence: CaseEvidenceRecord[];
  onChanged: () => void;
}) {
  const { locale } = useI18n();
  const text = caseCopy[locale];
  const [latitude, setLatitude] = useState("");
  const [longitude, setLongitude] = useState("");
  const [radius, setRadius] = useState("");
  const [country, setCountry] = useState("");
  const [region, setRegion] = useState("");
  const [locality, setLocality] = useState("");
  const [rationale, setRationale] = useState("");
  const [decision, setDecision] = useState<AdjudicationDecision>("needs_more_evidence");
  const [supporting, setSupporting] = useState<string[]>([]);
  const [invalid, setInvalid] = useState(false);
  const mutation = useMutation({
    mutationFn: () => {
      if (!media) throw new Error("media_required");
      return client.createOperatorHypothesis(caseId, {
        media_id: media.id,
        analysis_id: media.analysis_id,
        actor_id: LOCAL_ANALYST_ID,
        latitude: Number(latitude),
        longitude: Number(longitude),
        uncertainty_radius_m: Number(radius),
        country_code: country.trim() ? country.trim().toUpperCase() : null,
        region_name: region.trim() || null,
        locality_name: locality.trim() || null,
        supporting_evidence_ids: supporting,
        rationale: rationale.trim(),
        decision,
      });
    },
    onSuccess: () => {
      setLatitude("");
      setLongitude("");
      setRadius("");
      setCountry("");
      setRegion("");
      setLocality("");
      setRationale("");
      setSupporting([]);
      setInvalid(false);
      onChanged();
    },
  });

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const lat = Number(latitude);
    const lon = Number(longitude);
    const uncertainty = Number(radius);
    const valid = media
      && Number.isFinite(lat) && lat >= -90 && lat <= 90
      && Number.isFinite(lon) && lon >= -180 && lon <= 180
      && Number.isFinite(uncertainty) && uncertainty > 0 && uncertainty <= 20050000
      && rationale.trim().length > 0
      && (!country.trim() || /^[A-Za-z]{2}$/.test(country.trim()));
    if (!valid) {
      setInvalid(true);
      return;
    }
    mutation.mutate();
  };

  return (
    <section className="operator-correction" aria-labelledby="operator-correction-title">
      <header>
        <p className="eyebrow">{text.operatorCorrection}</p>
        <h2 id="operator-correction-title">{text.correctionTitle}</h2>
        <p>{text.correctionIntro}</p>
      </header>
      <form onSubmit={submit} noValidate>
        <div className="form-field">
          <label htmlFor="correction-latitude">{text.latitude}</label>
          <input id="correction-latitude" inputMode="decimal" value={latitude} onChange={(event) => setLatitude(event.target.value)} />
        </div>
        <div className="form-field">
          <label htmlFor="correction-longitude">{text.longitude}</label>
          <input id="correction-longitude" inputMode="decimal" value={longitude} onChange={(event) => setLongitude(event.target.value)} />
        </div>
        <div className="form-field">
          <label htmlFor="correction-radius">{text.radius}</label>
          <input id="correction-radius" inputMode="numeric" value={radius} onChange={(event) => setRadius(event.target.value)} />
        </div>
        <div className="form-field">
          <label htmlFor="correction-country">{text.country}</label>
          <input id="correction-country" maxLength={2} value={country} onChange={(event) => setCountry(event.target.value)} />
        </div>
        <div className="form-field">
          <label htmlFor="correction-region">{text.region}</label>
          <input id="correction-region" maxLength={160} value={region} onChange={(event) => setRegion(event.target.value)} />
        </div>
        <div className="form-field">
          <label htmlFor="correction-locality">{text.locality}</label>
          <input id="correction-locality" maxLength={160} value={locality} onChange={(event) => setLocality(event.target.value)} />
        </div>
        <fieldset className="operator-evidence">
          <legend>{text.supportingEvidence}</legend>
          {evidence.length === 0 ? <p>{text.noEvidence}</p> : evidence.map((item) => (
            <label key={item.id}>
              <input
                type="checkbox"
                checked={supporting.includes(item.id)}
                onChange={(event) => setSupporting((current) =>
                  event.target.checked ? [...current, item.id] : current.filter((id) => id !== item.id)
                )}
              />
              <span>{item.provider}: {item.summary}</span>
            </label>
          ))}
        </fieldset>
        <div className="form-field operator-correction__wide">
          <label htmlFor="correction-rationale">{text.rationale}</label>
          <textarea id="correction-rationale" rows={3} maxLength={2000} value={rationale} onChange={(event) => setRationale(event.target.value)} />
        </div>
        <div className="form-field">
          <label htmlFor="correction-decision">{text.initialDecision}</label>
          <select id="correction-decision" value={decision} onChange={(event) => setDecision(event.target.value as AdjudicationDecision)}>
            <option value="needs_more_evidence">{text.needsEvidence}</option>
            <option value="accepted">{text.accepted}</option>
            <option value="rejected">{text.rejected}</option>
          </select>
        </div>
        {invalid ? <p className="field-error operator-correction__wide" role="alert">{text.rationaleRequired}</p> : null}
        {mutation.isError ? <p className="field-error operator-correction__wide" role="alert">{text.correctionFailed}</p> : null}
        <button className="primary-button operator-correction__wide" type="submit" disabled={!media || mutation.isPending}>
          {text.addCorrection}
        </button>
      </form>
    </section>
  );
}

export function HypothesisReview({
  client,
  caseId,
  media,
  hypotheses,
  evidence,
  onChanged,
  guided = false,
  betweenMapAndReview,
}: HypothesisReviewProps) {
  const { locale } = useI18n();
  const text = caseCopy[locale];
  const [selectedId, setSelectedId] = useState<string | null>(hypotheses[0]?.id ?? null);

  return (
    <div className="hypothesis-workspace">
      <CaseHypothesisMap
        hypotheses={hypotheses}
        selectedId={selectedId}
        onSelect={setSelectedId}
        sectionId={guided ? "demo-map" : undefined}
      />
      {betweenMapAndReview}
      <section id={guided ? "demo-adjudication" : undefined} className="hypothesis-panel" aria-labelledby="case-hypotheses-title">
        <header>
          <p className="eyebrow">{text.unverified}</p>
          <h2 id="case-hypotheses-title">{text.hypothesesTitle}</h2>
        </header>
        {hypotheses.length === 0 ? (
          <p className="workspace-empty">{text.noHypotheses}</p>
        ) : (
          <div className="hypothesis-list">
            {hypotheses.map((item) => (
              <HypothesisCard
                key={item.id}
                client={client}
                caseId={caseId}
                hypothesis={item}
                selected={selectedId === item.id}
                onSelect={() => setSelectedId(item.id)}
                onChanged={onChanged}
              />
            ))}
          </div>
        )}
      </section>
      <OperatorCorrectionForm
        client={client}
        caseId={caseId}
        media={media}
        evidence={evidence}
        onChanged={onChanged}
      />
    </div>
  );
}
