import { useQuery } from "@tanstack/react-query";
import type { AtlasLensApiClient } from "../api/client";
import type { InvestigationCase } from "../api/schemas";
import { useI18n } from "../i18n";
import {
  caseCopy,
  decisionLabel,
  purposeLabel,
  sensitivityLabel,
  statusLabel,
} from "./copy";

interface CaseListViewProps {
  client: AtlasLensApiClient;
  onCreate: () => void;
  onOpen: (caseId: string) => void;
}

function CaseCard({
  item,
  onOpen,
}: {
  item: InvestigationCase;
  onOpen: (caseId: string) => void;
}) {
  const { locale } = useI18n();
  const text = caseCopy[locale];
  const updated = new Intl.DateTimeFormat(locale === "tr" ? "tr-TR" : "en-GB", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(item.updated_at));

  return (
    <article className="case-card">
      <header>
        <div>
          <span className={"case-sensitivity case-sensitivity--" + item.sensitivity}>
            {sensitivityLabel(item.sensitivity, locale)}
          </span>
          <h2>{item.title}</h2>
        </div>
        <span className={"status-badge status-badge--" + item.status}>
          {statusLabel(item.status, locale)}
        </span>
      </header>
      {item.description ? <p>{item.description}</p> : null}
      <dl className="case-card__facts">
        <div><dt>{text.casePurpose}</dt><dd>{purposeLabel(item.purpose, locale)}</dd></div>
        <div><dt>{text.mediaCount}</dt><dd>{item.media_count}</dd></div>
        <div><dt>{text.hypothesesCount}</dt><dd>{item.hypothesis_count}</dd></div>
        <div><dt>{text.status}</dt><dd>{decisionLabel(item.latest_adjudication_decision, locale)}</dd></div>
      </dl>
      <footer>
        <span>{text.updated}: <time dateTime={item.updated_at}>{updated}</time></span>
        <button className="secondary-button" type="button" onClick={() => onOpen(item.id)}>
          {text.openCase}
        </button>
      </footer>
    </article>
  );
}

export function CaseListView({ client, onCreate, onOpen }: CaseListViewProps) {
  const { locale } = useI18n();
  const text = caseCopy[locale];
  const query = useQuery({
    queryKey: ["cases"],
    queryFn: () => client.listCases({ limit: 100, offset: 0 }),
    retry: false,
  });

  return (
    <main id="main-content" className="workspace-shell case-list-view">
      <header className="workspace-heading case-page-heading">
        <div>
          <p className="eyebrow">{text.eyebrow}</p>
          <h1>{text.listTitle}</h1>
          <p>{text.listIntro}</p>
        </div>
        <button type="button" className="primary-button" onClick={onCreate}>
          {text.createCase}
        </button>
      </header>

      {query.isPending ? (
        <section className="workspace-state" role="status" aria-live="polite">
          <span className="spinner" aria-hidden="true" />
          <p>{text.loadingCases}</p>
        </section>
      ) : query.isError ? (
        <section className="workspace-state" role="alert">
          <h2>{text.loadError}</h2>
          <button type="button" className="secondary-button" onClick={() => void query.refetch()}>
            {text.retry}
          </button>
        </section>
      ) : query.data.items.length === 0 ? (
        <section className="workspace-state case-empty">
          <span className="case-empty__mark" aria-hidden="true">+</span>
          <h2>{text.emptyTitle}</h2>
          <p>{text.emptyBody}</p>
          <button type="button" className="primary-button" onClick={onCreate}>
            {text.createCase}
          </button>
        </section>
      ) : (
        <section className="case-grid" aria-label={text.listTitle}>
          {query.data.items.map((item) => (
            <CaseCard key={item.id} item={item} onOpen={onOpen} />
          ))}
        </section>
      )}
    </main>
  );
}
