import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useI18n } from "../i18n";
import { mapillaryDemoClient, type MapillaryDemoClient } from "../mapillary-demo/client";
import { MapillaryDemoMap } from "./MapillaryDemoMap";

const copy = {
  tr: {
    eyebrow: "Faz 3B3 özel demo",
    title: "Ankara Görsel Referans Pilotu",
    subtitle: "Özel teknik demo — üretim sistemi değildir",
    intro: "Ankara referans koleksiyonu içinde görsel benzerlik aramasıdır. Türkiye geneli konum tespiti veya kalibre edilmiş doğruluk değildir.",
    disabled: "Devre dışı",
    notReady: "Hazır değil",
    active: "Etkin",
    unavailable: "Demo durumu alınamadı.",
    city: "İndeks şehri",
    count: "Referans sayısı",
    model: "Model / indeks sürümü",
    license: "Kaynak ve lisans",
    choose: "Demo sorgu görseli seç",
    authorization: "Bu görseli analiz etmeye yetkiliyim.",
    query: "Ankara referans pilotunda sorgula",
    querying: "Yerel MegaLoc sorgusu çalışıyor…",
    private: "Görsel yalnızca yerel sunucuya gönderilir; geçicidir ve Mapillary'ye veya başka bir buluta gönderilmez.",
    results: "Ankara koleksiyonundaki görsel benzerlikler",
    distance: "Kosinüs uzaklığı",
    similarity: "Ham kosinüs benzerliği",
    contributor: "Katkı sahibi",
    capture: "Çekim tarihi",
    source: "Mapillary kaynağı",
    uncertainty: "Sunum belirsizliği yarıçapı",
    semantics: "Bu yarıçap yalnızca harita sunumu içindir; doğruluk, olasılık veya kalibre edilmiş güven değildir.",
    abstained: "Kanıt yetersiz; sistem sonuç üretmekten kaçındı.",
    failed: "Demo sorgusu güvenli biçimde tamamlanamadı.",
    map: "MapLibre üzerinde deneysel Mapillary referansları",
    alternative: "Metinsel harita alternatifi",
    noContributor: "API tarafından sağlanmadı",
  },
  en: {
    eyebrow: "Phase 3B3 private demo",
    title: "Ankara Visual Reference Pilot",
    subtitle: "Private technical demo — not a production system",
    intro: "This is visual similarity search within the Ankara reference collection, not Turkey-wide geolocation or calibrated accuracy.",
    disabled: "Disabled",
    notReady: "Not ready",
    active: "Active",
    unavailable: "Demo status could not be loaded.",
    city: "Index city",
    count: "Reference count",
    model: "Model / index version",
    license: "Source and license",
    choose: "Choose a demo query image",
    authorization: "I am authorized to analyze this image.",
    query: "Run Ankara reference pilot",
    querying: "Running the local MegaLoc query…",
    private: "The image is sent only to the local server, is temporary, and is not sent to Mapillary or another cloud.",
    results: "Visual similarities in the Ankara collection",
    distance: "Cosine distance",
    similarity: "Raw cosine similarity",
    contributor: "Contributor",
    capture: "Capture date",
    source: "Mapillary source",
    uncertainty: "Presentation uncertainty radius",
    semantics: "This radius is for map presentation only; it is not accuracy, probability, or calibrated confidence.",
    abstained: "Evidence was insufficient; the system abstained.",
    failed: "The demo query could not complete safely.",
    map: "Experimental Mapillary references on MapLibre",
    alternative: "Textual map alternative",
    noContributor: "Not provided by the API",
  },
} as const;

export function MapillaryDemoView({ client = mapillaryDemoClient }: { client?: MapillaryDemoClient }) {
  const { locale } = useI18n();
  const text = copy[locale];
  const [file, setFile] = useState<File | null>(null);
  const [authorized, setAuthorized] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement | null>(null);
  const status = useQuery({ queryKey: ["mapillary-demo-status"], queryFn: () => client.getStatus(), retry: 0 });
  const query = useMutation({
    mutationFn: ({ selectedFile, authorizationAcknowledged }: { selectedFile: File; authorizationAcknowledged: boolean }) => client.query(selectedFile, authorizationAcknowledged),
    onSettled: () => {
      setFile(null);
      setAuthorized(false);
      if (fileInput.current) fileInput.current.value = "";
    },
  });
  const demoState = status.data?.state;
  const resetQuery = query.reset;
  const selected = useMemo(() => query.data?.candidates.find((item) => item.mapillary_image_id === selectedId) ?? query.data?.candidates[0], [query.data, selectedId]);
  useEffect(() => {
    setSelectedId(query.data?.candidates[0]?.mapillary_image_id ?? null);
  }, [query.data]);
  useEffect(() => {
    if (demoState !== undefined && demoState !== "active") {
      resetQuery();
      setFile(null);
      setAuthorized(false);
      setSelectedId(null);
      if (fileInput.current) fileInput.current.value = "";
    }
  }, [demoState, resetQuery]);

  const stateLabel = status.data?.state === "active" ? text.active : status.data?.state === "not_ready" ? text.notReady : text.disabled;
  return (
    <main id="main-content" className="workspace mapillary-demo-workspace">
      <section className="mapillary-demo-hero" aria-labelledby="mapillary-demo-heading">
        <p className="eyebrow">{text.eyebrow}</p>
        <h1 id="mapillary-demo-heading">{text.title}</h1>
        <p className="mapillary-demo-subtitle">{text.subtitle}</p>
        <p>{text.intro}</p>
        {status.isError ? <div className="notice notice--warning" role="alert">{text.unavailable}</div> : null}
        {status.data ? (
          <>
            <span className={`mapillary-demo-state mapillary-demo-state--${status.data.state}`} role="status">{stateLabel}</span>
            <dl className="mapillary-demo-facts">
              <div><dt>{text.city}</dt><dd>{status.data.city ?? "—"}</dd></div>
              <div><dt>{text.count}</dt><dd>{status.data.image_count.toLocaleString(locale)}</dd></div>
              <div><dt>{text.model}</dt><dd>{status.data.model_version ?? "—"} · {status.data.index_version ?? "—"}</dd></div>
              <div><dt>{text.license}</dt><dd><a href={status.data.attribution_url} rel="noreferrer" target="_blank">Mapillary</a> · <a href={status.data.license_url} rel="noreferrer" target="_blank">{status.data.license_identifier}</a></dd></div>
            </dl>
            {status.data.reason_code ? <p className="muted"><code>{status.data.reason_code}</code></p> : null}
          </>
        ) : null}
      </section>

      {status.data?.state === "active" ? (
        <section className="mapillary-demo-query" aria-labelledby="mapillary-demo-query-heading">
          <h2 id="mapillary-demo-query-heading">{text.choose}</h2>
          <input ref={fileInput} aria-label={text.choose} type="file" accept="image/jpeg,image/png,image/webp" onChange={(event) => { setFile(event.currentTarget.files?.[0] ?? null); setAuthorized(false); query.reset(); }} />
          <label className="check-row"><input type="checkbox" checked={authorized} onChange={(event) => setAuthorized(event.currentTarget.checked)} />{text.authorization}</label>
          <p className="muted small">{text.private}</p>
          <button type="button" className="button button--primary" disabled={!file || !authorized || query.isPending} onClick={() => { if (file) query.mutate({ selectedFile: file, authorizationAcknowledged: authorized }); }}>{query.isPending ? text.querying : text.query}</button>
          {query.isError ? <div className="notice notice--error" role="alert">{text.failed}</div> : null}
        </section>
      ) : null}

      {demoState === "active" && query.data ? (
        <section className="mapillary-demo-results" aria-labelledby="mapillary-demo-results-heading">
          <h2 id="mapillary-demo-results-heading">{text.results}</h2>
          {query.data.status === "failed" ? <div className="notice notice--error" role="alert">{text.failed} {query.data.reason_code ? <code>{query.data.reason_code}</code> : null}</div> : query.data.abstained ? <div className="notice notice--warning" role="status">{text.abstained} {query.data.reason_code ? <code>{query.data.reason_code}</code> : null}</div> : null}
          {query.data.candidates.length ? <MapillaryDemoMap candidates={query.data.candidates} selectedId={selectedId} onSelect={setSelectedId} mapLabel={text.map} alternativeLabel={text.alternative} /> : null}
          {selected ? (
            <article className="mapillary-demo-reference">
              <h3>#{selected.rank} · {selected.mapillary_image_id}</h3>
              <dl>
                <div><dt>{text.distance}</dt><dd>{selected.cosine_distance.toFixed(6)}</dd></div>
                <div><dt>{text.similarity}</dt><dd>{selected.cosine_similarity.toFixed(6)}</dd></div>
                <div><dt>{text.contributor}</dt><dd>{selected.contributor ?? text.noContributor}</dd></div>
                <div><dt>{text.capture}</dt><dd>{selected.capture_date ?? "—"}</dd></div>
                <div><dt>{text.uncertainty}</dt><dd>{selected.uncertainty_radius_m.toLocaleString(locale)} m</dd></div>
                <div><dt>{text.source}</dt><dd><a href={selected.source_url} rel="noreferrer" target="_blank">Mapillary #{selected.mapillary_image_id}</a> · <a href={selected.license_url} rel="noreferrer" target="_blank">{selected.license_identifier}</a></dd></div>
              </dl>
              <p className="confidence-disclaimer">{text.semantics}</p>
            </article>
          ) : null}
        </section>
      ) : null}
    </main>
  );
}
