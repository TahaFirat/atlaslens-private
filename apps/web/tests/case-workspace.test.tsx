import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { AtlasLensApp } from "../src/App";
import type { AtlasLensApiClient } from "../src/api/client";
import {
  caseAuditEvents,
  caseAuditIntegrity,
  caseEvidence,
  caseHypothesis,
  caseId,
  caseMedia,
  createFakeClient,
  investigationCase,
  jpegFile,
} from "./fixtures";

function casePage(items = [investigationCase]) {
  return {
    items,
    total: items.length,
    limit: 100,
    offset: 0,
    ordering: "updated_at_desc_id_desc" as const,
  };
}

async function openCases(client: AtlasLensApiClient) {
  const user = userEvent.setup();
  render(<AtlasLensApp client={client} initialLocale="tr" />);
  await user.click(screen.getByRole("button", { name: "Vakalar" }));
  return user;
}

async function openWorkspace(client: AtlasLensApiClient) {
  const user = await openCases(client);
  await screen.findByRole("heading", { name: investigationCase.title });
  await user.click(screen.getByRole("button", { name: "Vakayı aç" }));
  await screen.findByRole("heading", { name: investigationCase.title });
  return user;
}

describe("Phase 2 case and evidence workspace", () => {
  it("renders an honest loading and empty case-list state", async () => {
    let resolveCases: ((value: ReturnType<typeof casePage>) => void) | undefined;
    const listCases = vi.fn(() => new Promise<ReturnType<typeof casePage>>((resolve) => {
      resolveCases = resolve;
    }));
    const { client } = createFakeClient({ clientOverrides: { listCases } });

    await openCases(client);
    expect(screen.getByText("Vakalar yükleniyor…")).toBeVisible();

    resolveCases?.(casePage([]));
    expect(await screen.findByRole("heading", { name: "Henüz inceleme vakası yok" })).toBeVisible();
    expect(screen.getAllByRole("button", { name: "Vaka oluştur" })).toHaveLength(2);
    screen.getAllByRole("button", { name: "Vaka oluştur" }).forEach((button) => {
      expect(button).toBeEnabled();
    });
  });

  it("shows a retryable case-list error", async () => {
    const listCases = vi.fn(() => Promise.reject(new Error("synthetic list failure")));
    const { client } = createFakeClient({ clientOverrides: { listCases } });

    const user = await openCases(client);
    expect(await screen.findByRole("alert")).toHaveTextContent("Vakalar yüklenemedi");
    await user.click(screen.getByRole("button", { name: "Yeniden dene" }));
    await waitFor(() => expect(listCases).toHaveBeenCalledTimes(2));
  });

  it("validates required create-case controls and submits the bounded payload", async () => {
    const createCase = vi.fn(() => Promise.resolve(investigationCase));
    const { client } = createFakeClient({
      clientOverrides: {
        listCases: vi.fn(() => Promise.resolve(casePage([]))),
        createCase,
      },
    });
    const user = await openCases(client);
    await screen.findByRole("heading", { name: "Henüz inceleme vakası yok" });
    await user.click(screen.getAllByRole("button", { name: "Vaka oluştur" })[0] as HTMLElement);
    await user.click(screen.getByRole("button", { name: "Vaka oluştur" }));
    expect(screen.getAllByText("Bu alan zorunludur.")).toHaveLength(2);
    expect(screen.getByText("Yetkilendirme beyanı zorunludur.")).toBeVisible();

    await user.type(screen.getByLabelText("Başlık"), "Sentetik inceleme");
    await user.selectOptions(screen.getByLabelText("Meşru amaç"), "other");
    await user.type(screen.getByLabelText("Kaynak bağlamı"), "Açıkça sentetik, güvenli test bağlamı.");
    await user.click(screen.getByLabelText(/Bu incelemenin ve medyanın yetkili olduğunu/));
    await user.click(screen.getByRole("button", { name: "Vaka oluştur" }));
    expect(screen.getByText("Diğer seçildiğinde amacı açıklayın.")).toBeVisible();

    await user.type(screen.getByLabelText("Amaç ayrıntısı"), "Yetkili ürün doğrulaması");
    await user.selectOptions(screen.getByLabelText("Hassasiyet"), "conflict_related");
    expect(screen.getByRole("alert")).toHaveTextContent("Kesin model hipotezleri");
    await user.click(screen.getByRole("button", { name: "Vaka oluştur" }));

    await waitFor(() => expect(createCase).toHaveBeenCalledWith(expect.objectContaining({
      title: "Sentetik inceleme",
      purpose: "other",
      purpose_detail: "Yetkili ürün doğrulaması",
      sensitivity: "conflict_related",
      authorization_attested: true,
      created_by_actor_id: "local-analyst",
    })));
    expect(await screen.findByRole("heading", { name: investigationCase.title })).toBeVisible();
  });

  it("groups evidence and shows sensitive, expired-media, hypothesis and audit truth in Turkish", async () => {
    vi.stubGlobal("WebGLRenderingContext", undefined);
    const { client } = createFakeClient({
      clientOverrides: { listCases: vi.fn(() => Promise.resolve(casePage())) },
    });
    await openWorkspace(client);

    expect(screen.getByRole("alert")).toHaveTextContent("gerçek zamanlı taktik hedefleme");
    expect(screen.getByText("Görsel saklama politikası gereği artık mevcut değil")).toBeVisible();
    expect(screen.getByRole("heading", { name: "Meta veri" })).toBeVisible();
    expect(screen.getByRole("heading", { name: "Görsel ipucu" })).toBeVisible();
    expect(screen.getByText("synthetic-metadata-fixture")).toBeVisible();
    expect(screen.getAllByText("Model adayı").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Doğrulanmamış").length).toBeGreaterThan(0);
    expect(screen.getByText("Kalibre edilmemiştir; bu bir olasılık değildir.")).toBeVisible();
    expect(screen.queryByText(/%/)).not.toBeInTheDocument();
    expect(screen.getByTestId("case-hypothesis-map")).toBeVisible();
    expect(screen.getByText("Denetim özet zinciri doğrulandı")).toBeVisible();

    const events = screen.getAllByText(/case created|analysis materialized/);
    expect(events[0]).toHaveTextContent("case created");
    expect(events[1]).toHaveTextContent("analysis materialized");
  });

  it("appends hypothesis decisions only with an explicit rationale", async () => {
    const adjudicateCaseHypothesis = vi.fn(() => Promise.resolve({
      ...caseAuditEvents,
    } as never));
    const { client } = createFakeClient({
      clientOverrides: {
        listCases: vi.fn(() => Promise.resolve(casePage())),
        adjudicateCaseHypothesis,
      },
    });
    const user = await openWorkspace(client);
    const card = screen.getByRole("heading", { name: /#1 Synthetic region/ }).closest("article");
    expect(card).not.toBeNull();
    const scope = within(card as HTMLElement);

    await user.click(scope.getByRole("button", { name: "Daha fazla kanıt iste" }));
    expect(scope.getByText("Salt-eklemeli değerlendirme için gerekçe zorunludur.")).toBeVisible();
    await user.type(scope.getByLabelText("Karar gerekçesi"), "İkinci bağımsız sağlayıcı gerekli.");
    await user.click(scope.getByRole("button", { name: "Daha fazla kanıt iste" }));

    await waitFor(() => expect(adjudicateCaseHypothesis).toHaveBeenCalledWith(
      caseId,
      caseHypothesis.id,
      expect.objectContaining({
        actor_id: "local-analyst",
        decision: "needs_more_evidence",
        rationale: "İkinci bağımsız sağlayıcı gerekli.",
      }),
    ));
  });

  it("adds a separate operator correction with positive uncertainty and initial adjudication", async () => {
    const createOperatorHypothesis = vi.fn(() => Promise.resolve({
      hypothesis: { ...caseHypothesis, origin: "operator_correction" as const },
      adjudication: {
        id: "a23e4567-e89b-42d3-a456-426614174009",
        case_id: caseId,
        hypothesis_id: caseHypothesis.id,
        actor_id: "local-analyst",
        decision: "needs_more_evidence" as const,
        rationale: "Sentetik operatör düzeltmesi.",
        created_at: "2026-07-15T10:10:00Z",
        supersedes_adjudication_id: null,
      },
    }));
    const { client } = createFakeClient({
      clientOverrides: {
        listCases: vi.fn(() => Promise.resolve(casePage())),
        createOperatorHypothesis,
      },
    });
    const user = await openWorkspace(client);
    await user.type(screen.getByLabelText("Enlem"), "1.25");
    await user.type(screen.getByLabelText("Boylam"), "2.5");
    await user.type(screen.getByLabelText("Belirsizlik yarıçapı (metre)"), "5000");
    const correction = screen.getByRole("heading", { name: "Operatör tarafından düzeltilmiş konum ekle" }).closest("section");
    expect(correction).not.toBeNull();
    const correctionScope = within(correction as HTMLElement);
    await user.type(correctionScope.getByLabelText("Karar gerekçesi"), "Sentetik operatör düzeltmesi.");
    await user.click(correctionScope.getByRole("button", { name: "Operatör düzeltmesi ekle" }));

    await waitFor(() => expect(createOperatorHypothesis).toHaveBeenCalledWith(caseId, expect.objectContaining({
      media_id: caseMedia.id,
      analysis_id: caseMedia.analysis_id,
      latitude: 1.25,
      longitude: 2.5,
      uncertainty_radius_m: 5000,
      decision: "needs_more_evidence",
    })));
  });

  it("reuses createAnalysis before metadata linking and materializes completed analysis", async () => {
    const createAnalysis = vi.fn(() => Promise.resolve({
      id: caseMedia.analysis_id as string,
      status: "queued" as const,
      status_url: "/analysis",
      events_url: "/events",
      delete_url: "/delete",
    }));
    const createCaseMedia = vi.fn(() => Promise.resolve(caseMedia));
    const linkCaseAnalysis = vi.fn(() => Promise.resolve(caseMedia));
    const materializeCaseAnalysis = vi.fn(() => Promise.resolve({
      case_id: caseId,
      media_id: caseMedia.id,
      analysis_id: caseMedia.analysis_id as string,
      created: true,
      evidence_created: 2,
      hypotheses_created: 1,
      already_materialized: false,
      evidence_ids: caseEvidence.map((item) => item.id),
      hypothesis_ids: [caseHypothesis.id],
    }));
    const { client } = createFakeClient({
      clientOverrides: {
        listCases: vi.fn(() => Promise.resolve(casePage())),
        createAnalysis,
        createCaseMedia,
        linkCaseAnalysis,
        materializeCaseAnalysis,
      },
    });
    const user = await openWorkspace(client);
    await user.upload(screen.getByTestId("case-file-input"), jpegFile("synthetic-generated-fixture.jpg"));
    await user.click(screen.getByLabelText("Bu görseli analiz etmeye yetkiliyim."));
    await user.click(screen.getByRole("button", { name: "Analiz et ve bağla" }));

    await waitFor(() => expect(createAnalysis).toHaveBeenCalledOnce());
    expect(createCaseMedia).toHaveBeenCalledWith(caseId, expect.objectContaining({
      actor_id: "local-analyst",
      original_filename_display: "synthetic-generated-fixture.jpg",
      storage_state: "ephemeral",
      authorization_attested: true,
    }));
    expect(linkCaseAnalysis).toHaveBeenCalledWith(caseId, caseMedia.analysis_id, caseMedia.id, "local-analyst");
    await waitFor(() => expect(materializeCaseAnalysis).toHaveBeenCalledWith(
      caseId,
      caseMedia.analysis_id,
      caseMedia.id,
      "local-analyst",
    ));
  });

  it("recovers materialization after reload and retries only on explicit operator action", async () => {
    const pendingMedia = { ...caseMedia, materialized_at: null };
    const materialized = {
      case_id: caseId,
      media_id: caseMedia.id,
      analysis_id: caseMedia.analysis_id as string,
      created: true,
      evidence_created: 2,
      hypotheses_created: 1,
      already_materialized: false,
      evidence_ids: caseEvidence.map((item) => item.id),
      hypothesis_ids: [caseHypothesis.id],
    };
    const materializeCaseAnalysis = vi.fn()
      .mockRejectedValueOnce(new Error("synthetic materialization failure"))
      .mockResolvedValue(materialized);
    const { client } = createFakeClient({
      clientOverrides: {
        listCases: vi.fn(() => Promise.resolve(casePage())),
        listCaseMedia: vi.fn(() => Promise.resolve({
          items: [pendingMedia],
          total: 1,
          limit: 100,
          offset: 0,
          ordering: "created_at_asc_id_asc" as const,
        })),
        materializeCaseAnalysis,
      },
    });
    const user = await openWorkspace(client);

    expect(await screen.findByText("Tamamlanan analiz vaka kanıtına dönüştürülemedi.")).toBeVisible();
    expect(materializeCaseAnalysis).toHaveBeenCalledTimes(1);
    await user.click(screen.getByRole("button", { name: "Kanıt oluşturmayı yeniden dene" }));
    await waitFor(() => expect(materializeCaseAnalysis).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(screen.queryByText("Tamamlanan analiz vaka kanıtına dönüştürülemedi.")).not.toBeInTheDocument());
    expect(materializeCaseAnalysis).toHaveBeenLastCalledWith(
      caseId,
      caseMedia.analysis_id,
      caseMedia.id,
      "local-analyst",
    );
  });

  it("renders independent empty and integrity-failure workspace states", async () => {
    const { client } = createFakeClient({
      clientOverrides: {
        listCases: vi.fn(() => Promise.resolve(casePage())),
        listCaseMedia: vi.fn(() => Promise.resolve({ items: [], total: 0, limit: 100, offset: 0, ordering: "created_at_asc_id_asc" as const })),
        listCaseEvidence: vi.fn(() => Promise.resolve({ items: [], total: 0, limit: 100, offset: 0, ordering: "created_at_asc_id_asc" as const })),
        listCaseHypotheses: vi.fn(() => Promise.resolve({ items: [], total: 0, limit: 100, offset: 0, ordering: "created_at_asc_id_asc" as const })),
        listCaseAuditEvents: vi.fn(() => Promise.resolve({ items: [], total: 0, limit: 100, offset: 0, ordering: "sequence_number_asc" as const, integrity_scope: "tamper_evident_application_history" as const })),
        getCaseAuditIntegrity: vi.fn(() => Promise.resolve({ ...caseAuditIntegrity, valid: false, reason: "hash_mismatch", first_invalid_sequence: 2 })),
      },
    });
    await openWorkspace(client);
    expect(screen.getAllByText("Bu vakaya bağlı medya yok.").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Normalize edilmiş kanıt yok. AtlasLens kanıt uydurmak yerine çekimser kalır.").length).toBeGreaterThan(0);
    expect(screen.getByText("Konum hipotezi için yeterli kanıt yok.")).toBeVisible();
    expect(screen.getByText("Denetim olayı döndürülmedi.")).toBeVisible();
    expect(screen.getByText("hash_mismatch")).toBeVisible();
  });
});
