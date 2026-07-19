import { expect, test, type Page, type Route } from "@playwright/test";

const ids = {
  case: "1cd56835-46ab-5b0a-9158-b6c326f27b85",
  media: "22222222-2222-4222-8222-222222222222",
  analysis: "33333333-3333-4333-8333-333333333333",
  evidence: "44444444-4444-4444-8444-444444444444",
  modelHypothesis: "55555555-5555-4555-8555-555555555555",
  operatorHypothesis: "66666666-6666-4666-8666-666666666666",
};
const createdAt = "2026-07-15T12:00:00Z";

function json(route: Route, body: unknown, status = 200) {
  return route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
}

async function installSyntheticCaseApi(page: Page, preparedInvestorCase = false) {
  let caseCreated = preparedInvestorCase;
  let materialized = preparedInvestorCase;
  let mediaLinked = preparedInvestorCase;
  const adjudications: Record<string, Array<Record<string, unknown>>> = {};
  const operatorHypotheses: Array<Record<string, unknown>> = [];
  const audit: Array<Record<string, unknown>> = [];

  const addAudit = (eventType: string, actorType: "operator" | "system" | "model") => {
    const sequence = audit.length + 1;
    audit.push({
      id: String(sequence).padStart(8, "0") + "-0000-4000-8000-000000000000",
      case_id: ids.case,
      sequence_number: sequence,
      event_type: eventType,
      actor_id: actorType === "operator" ? "local-analyst" : "atlaslens-system",
      actor_type: actorType,
      payload: { fixture_kind: "explicitly_synthetic" },
      created_at: new Date(Date.parse(createdAt) + sequence * 1000).toISOString(),
      previous_event_hash: sequence === 1 ? null : String(sequence - 1).repeat(64).slice(0, 64),
      event_hash: String(sequence).repeat(64).slice(0, 64),
    });
  };
  if (preparedInvestorCase) {
    addAudit("case_created", "operator");
    addAudit("analysis_materialized", "system");
  }

  const caseRecord = () => ({
    id: ids.case,
    workspace_id: "local-default",
    title: preparedInvestorCase ? "AtlasLens Ankara Pilot — Yatırımcı Özel Demo" : "Sentetik kıyı incelemesi",
    description: "Açıkça sentetik Playwright fixture'ı.",
    purpose: "humanitarian",
    purpose_detail: null,
    source_context: "Test paketi tarafından üretilmiş sentetik görsel.",
    status: "under_review",
    sensitivity: "conflict_related",
    authorization_attested: true,
    created_at: createdAt,
    updated_at: "2026-07-15T12:10:00Z",
    closed_at: null,
    created_by_actor_id: "local-analyst",
    retention_policy: "analysis_metadata_only",
    version: 1,
    media_count: mediaLinked ? 1 : 0,
    evidence_count: materialized ? 1 : 0,
    hypothesis_count: materialized ? 1 + operatorHypotheses.length : 0,
    adjudication_count: Object.values(adjudications).reduce((sum, values) => sum + values.length, 0),
    latest_adjudication_decision: null,
  });
  const media = () => ({
    id: ids.media,
    case_id: ids.case,
    analysis_id: mediaLinked ? ids.analysis : null,
    media_type: "image",
    source_type: "upload",
    original_filename_display: "synthetic-generated-phase2.png",
    mime_type: "image/png",
    byte_size: 68,
    sha256: "a".repeat(64),
    captured_at: null,
    received_at: "2026-07-15T12:01:00Z",
    source_url: null,
    archive_url: null,
    source_description: "Explicitly synthetic fixture.",
    authorization_attested: true,
    storage_state: materialized ? "deleted_after_analysis" : "ephemeral",
    created_at: "2026-07-15T12:01:00Z",
    analysis_status_at_link: mediaLinked ? "completed" : null,
    analysis_created_at: mediaLinked ? "2026-07-15T12:01:00Z" : null,
    analysis_expires_at: mediaLinked ? "2026-07-15T13:01:00Z" : null,
    materialized_at: materialized ? "2026-07-15T12:02:00Z" : null,
  });
  const evidence = {
    id: ids.evidence,
    case_id: ids.case,
    media_id: ids.media,
    analysis_id: ids.analysis,
    evidence_type: "visual_clue",
    provider: "synthetic-fixture-provider",
    provider_family: "synthetic-family",
    summary: "Sentetik kanıt özeti; gerçek OSINT sonucu değildir.",
    structured_payload: { fixture: true },
    provenance: { fixture_kind: "explicitly_synthetic" },
    observed_at: "2026-07-15T12:02:00Z",
    created_at: "2026-07-15T12:02:00Z",
    immutable_source_hash: "b".repeat(64),
  };
  const modelHypothesis = () => ({
    id: ids.modelHypothesis,
    case_id: ids.case,
    media_id: ids.media,
    analysis_id: ids.analysis,
    origin: "model",
    latitude: 0.25,
    longitude: 0.5,
    uncertainty_radius_m: 25000,
    country_code: null,
    region_name: "Synthetic region",
    locality_name: null,
    rank: 1,
    confidence_label: null,
    calibration_state: "uncalibrated",
    supporting_evidence_ids: [ids.evidence],
    model_family_groups: ["synthetic-family"],
    created_at: "2026-07-15T12:03:00Z",
    supersedes_hypothesis_id: null,
    adjudications: adjudications[ids.modelHypothesis] ?? [],
    adjudication_count: (adjudications[ids.modelHypothesis] ?? []).length,
    latest_adjudication: (adjudications[ids.modelHypothesis] ?? []).at(-1) ?? null,
  });
  const analysis = {
    id: ids.analysis,
    status: "completed",
    analysis_mode: "local_only",
    created_at: "2026-07-15T12:01:00Z",
    expires_at: "2026-07-15T13:01:00Z",
    progress: { stage: "completed", percent: 100, message_key: "progress.completed" },
    image: { format: "png", width: 1, height: 1, megapixels: 0.000001, sha256: "a".repeat(64), orientation_normalized: true, exif_present: false },
    quality: { blur_score: 0.5, brightness_score: 0.5, contrast_score: 0.5, resolution_score: 0.1, warnings: [] },
    evidence: [],
    candidates: [],
    abstention: { abstained: true, reason_code: "insufficient_evidence", message_key: "reason.insufficient_evidence" },
    warnings: [],
    timings_ms: { total: 1 },
    fusion_policy_version: "synthetic-e2e",
    result_classification: "real",
    failure: null,
  };

  await page.route("**/api/v1/capabilities", (route) => json(route, {
    supported_formats: ["jpeg", "png", "webp"],
    max_upload_bytes: 20971520,
    max_decoded_pixels: 40000000,
    enabled_analysis_modes: ["local_only"],
    providers: {
      exif: { provider_id: "exif", enabled: true, available: true, execution_boundary: "local", reason_code: null },
      quality: { provider_id: "quality", enabled: true, available: true, execution_boundary: "local", reason_code: null },
      ocr: { provider_id: "ocr", enabled: false, available: false, execution_boundary: "local", reason_code: "disabled" },
      cloud_vision: { provider_id: "nvidia-qwen-vision-reasoning", enabled: false, available: false, execution_boundary: "cloud", reason_code: "disabled", operational_status: "disabled" },
    },
    retention: { keep_uploads: false, ttl_seconds: 3600, originals_deleted_after_analysis: true },
    version: "phase2-synthetic-e2e",
  }));
  await page.route("**/api/v1/analyses", async (route) => {
    if (route.request().method() !== "POST") return route.fallback();
    return json(route, {
      id: ids.analysis,
      status: "queued",
      status_url: "/api/v1/analyses/" + ids.analysis,
      events_url: "/api/v1/analyses/" + ids.analysis + "/events",
      delete_url: "/api/v1/analyses/" + ids.analysis,
    }, 202);
  });
  await page.route("**/api/v1/analyses/" + ids.analysis + "/events", (route) => route.fulfill({
    status: 200,
    contentType: "text/event-stream",
    body: "event: completed\ndata: " + JSON.stringify({
      event_id: "synthetic-terminal",
      event_type: "completed",
      analysis_id: ids.analysis,
      occurred_at: "2026-07-15T12:01:02Z",
      status: "completed",
      progress: analysis.progress,
    }) + "\n\n",
  }));
  await page.route("**/api/v1/analyses/" + ids.analysis, (route) => json(route, analysis));

  await page.route("**/api/v1/cases/**", async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    const method = request.method();
    if (path.endsWith("/media")) {
      if (method === "POST") return json(route, media(), 201);
      return json(route, { items: mediaLinked ? [media()] : [], total: mediaLinked ? 1 : 0, limit: 100, offset: 0, ordering: "created_at_asc_id_asc" });
    }
    if (path.endsWith("/link")) {
      mediaLinked = true;
      addAudit("analysis_linked", "operator");
      return json(route, media());
    }
    if (path.endsWith("/materialize-evidence")) {
      materialized = true;
      addAudit("analysis_materialized", "system");
      return json(route, {
        case_id: ids.case,
        media_id: ids.media,
        analysis_id: ids.analysis,
        created: true,
        evidence_created: 1,
        hypotheses_created: 1,
        already_materialized: false,
        evidence_ids: [ids.evidence],
        hypothesis_ids: [ids.modelHypothesis],
      });
    }
    if (path.endsWith("/evidence")) {
      const items = materialized ? [evidence] : [];
      return json(route, { items, total: items.length, limit: 100, offset: 0, ordering: "created_at_asc_id_asc" });
    }
    if (path.endsWith("/hypotheses")) {
      const items = materialized ? [modelHypothesis(), ...operatorHypotheses] : [];
      return json(route, { items, total: items.length, limit: 100, offset: 0, ordering: "rank_asc_created_at_asc_id_asc" });
    }
    if (path.includes("/hypotheses/") && path.endsWith("/adjudications")) {
      const hypothesisId = path.split("/hypotheses/")[1]?.split("/")[0] ?? "";
      const input = request.postDataJSON() as { actor_id: string; decision: string; rationale: string };
      const history = adjudications[hypothesisId] ?? [];
      const item = {
        id: String(70 + history.length).padStart(8, "0") + "-0000-4000-8000-000000000000",
        case_id: ids.case,
        hypothesis_id: hypothesisId,
        actor_id: input.actor_id,
        decision: input.decision,
        rationale: input.rationale,
        created_at: new Date(Date.parse(createdAt) + (audit.length + 1) * 1000).toISOString(),
        supersedes_adjudication_id: history.at(-1)?.id ?? null,
      };
      adjudications[hypothesisId] = [...history, item];
      const operatorIndex = operatorHypotheses.findIndex((candidate) => candidate.id === hypothesisId);
      if (operatorIndex >= 0) {
        operatorHypotheses[operatorIndex] = {
          ...operatorHypotheses[operatorIndex],
          adjudications: adjudications[hypothesisId],
          adjudication_count: adjudications[hypothesisId]?.length ?? 0,
          latest_adjudication: item,
        };
      }
      addAudit("hypothesis_adjudicated", "operator");
      return json(route, item, 201);
    }
    if (path.endsWith("/operator-hypotheses")) {
      const input = request.postDataJSON() as Record<string, unknown>;
      const adjudication = {
        id: "77777777-7777-4777-8777-777777777777",
        case_id: ids.case,
        hypothesis_id: ids.operatorHypothesis,
        actor_id: "local-analyst",
        decision: input.decision,
        rationale: input.rationale,
        created_at: "2026-07-15T12:07:00Z",
        supersedes_adjudication_id: null,
      };
      adjudications[ids.operatorHypothesis] = [adjudication];
      const hypothesis = {
        id: ids.operatorHypothesis,
        case_id: ids.case,
        media_id: ids.media,
        analysis_id: ids.analysis,
        origin: "operator_correction",
        latitude: input.latitude,
        longitude: input.longitude,
        uncertainty_radius_m: input.uncertainty_radius_m,
        country_code: null,
        region_name: null,
        locality_name: "Synthetic operator point",
        rank: null,
        confidence_label: null,
        calibration_state: "not_applicable",
        supporting_evidence_ids: input.supporting_evidence_ids,
        model_family_groups: [],
        created_at: "2026-07-15T12:07:00Z",
        supersedes_hypothesis_id: null,
        adjudications: [adjudication],
        adjudication_count: 1,
        latest_adjudication: adjudication,
      };
      operatorHypotheses.push(hypothesis);
      addAudit("operator_hypothesis_created", "operator");
      return json(route, { hypothesis, adjudication }, 201);
    }
    if (path.endsWith("/audit-events")) {
      return json(route, { items: audit, total: audit.length, limit: 100, offset: 0, ordering: "sequence_number_asc", integrity_scope: "tamper_evident_application_history" });
    }
    if (path.endsWith("/audit-integrity")) {
      return json(route, { case_id: ids.case, valid: true, checked_event_count: audit.length, first_invalid_sequence: null, reason: null, integrity_scope: "tamper_evident_application_history", legally_certified_evidence: false });
    }
    if (path === "/api/v1/cases/" + ids.case && method === "GET") return json(route, caseRecord());
    return json(route, { code: "synthetic_route_missing" }, 404);
  });
  await page.route(/\/api\/v1\/cases(?:\?.*)?$/, async (route) => {
    if (route.request().method() === "POST") {
      caseCreated = true;
      addAudit("case_created", "operator");
      return json(route, caseRecord(), 201);
    }
    const items = caseCreated ? [caseRecord()] : [];
    return json(route, { items, total: items.length, limit: 100, offset: 0, ordering: "updated_at_desc_id_desc" });
  });
}

test("synthetic case review survives reload and preserves the existing analysis entry point", async ({ page }) => {
  await installSyntheticCaseApi(page);
  await page.goto("/");
  await page.getByRole("button", { name: "TR" }).click();
  await page.getByRole("button", { name: "Vakalar" }).click();
  await expect(page.getByRole("heading", { name: "Henüz inceleme vakası yok" })).toBeVisible();
  await page.getByRole("button", { name: "Vaka oluştur" }).first().click();
  await page.getByLabel("Başlık").fill("Sentetik kıyı incelemesi");
  await page.getByLabel("Meşru amaç").selectOption("humanitarian");
  await page.getByLabel("Hassasiyet").selectOption("conflict_related");
  await page.getByLabel("Kaynak bağlamı").fill("Test paketi tarafından üretilmiş sentetik görsel.");
  await page.getByLabel(/Bu incelemenin ve medyanın yetkili olduğunu/).check();
  await page.getByRole("button", { name: "Vaka oluştur" }).click();
  await expect(page.getByRole("heading", { name: "Sentetik kıyı incelemesi" })).toBeVisible();

  await page.getByTestId("case-file-input").setInputFiles({
    name: "synthetic-generated-phase2.png",
    mimeType: "image/png",
    buffer: Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl2lB8AAAAASUVORK5CYII=", "base64"),
  });
  await page.getByLabel("Bu görseli analiz etmeye yetkiliyim.").check();
  await page.getByRole("button", { name: "Analiz et ve bağla" }).click();
  await expect(page.getByText("synthetic-fixture-provider", { exact: true })).toBeVisible();
  await expect(page.getByText("Model adayı", { exact: true })).toBeVisible();
  await expect(page.getByText("Kalibre edilmemiştir; bu bir olasılık değildir.")).toBeVisible();

  const modelCard = page.locator(".hypothesis-card").filter({ hasText: "Model adayı" });
  await modelCard.getByLabel("Karar gerekçesi").fill("İkinci bağımsız kanıt gerekli.");
  await modelCard.getByRole("button", { name: "Daha fazla kanıt iste" }).click();
  await expect(modelCard.locator(".decision-badge")).toHaveText("Daha fazla kanıt gerekli");

  const correction = page.locator(".operator-correction");
  await correction.getByLabel("Enlem").fill("1.25");
  await correction.getByLabel("Boylam").fill("2.5");
  await correction.getByLabel("Belirsizlik yarıçapı (metre)").fill("5000");
  await correction.getByLabel("Karar gerekçesi").fill("Sentetik operatör düzeltmesi.");
  await correction.getByRole("button", { name: "Operatör düzeltmesi ekle" }).click();
  const operatorCard = page.locator(".hypothesis-card").filter({ hasText: "Operatör düzeltmesi" });
  await expect(operatorCard).toBeVisible();
  await operatorCard.getByLabel("Karar gerekçesi").fill("Yetkili sentetik düzeltme kabul edildi.");
  await operatorCard.getByRole("button", { name: "Kabul et" }).click();
  await expect(operatorCard.locator(".decision-badge")).toHaveText("Kabul edildi");

  const sequence = await page.locator(".audit-sequence").allTextContents();
  expect(sequence).toEqual(sequence.map((_, index) => String(index + 1)));

  await page.reload();
  await page.getByRole("button", { name: "TR" }).click();
  await page.getByRole("button", { name: "Vakalar" }).click();
  await page.getByRole("button", { name: "Vakayı aç" }).click();
  const reloadedOperatorCard = page.locator(".hypothesis-card").filter({ hasText: "Operatör düzeltmesi" });
  await expect(reloadedOperatorCard).toBeVisible();
  await expect(reloadedOperatorCard.locator(".decision-badge")).toHaveText("Kabul edildi");

  await page.getByRole("button", { name: "Yeni analiz" }).click();
  await expect(page.getByRole("heading", { name: "Yalnızca görüntüyü değil, kanıtı konumlandırın" })).toBeVisible();
});

test("guided Turkish investor flow survives disabled NVIDIA and keeps review boundaries visible", async ({ page }) => {
  await installSyntheticCaseApi(page, true);
  await page.goto("/?demo=investor&lang=tr");

  await expect(page.getByRole("heading", { name: "Yetkili kaynaktan incelenebilir konum hipotezlerine" })).toBeVisible();
  await expect(page.getByText("Yalnızca Ankara pilotu", { exact: true })).toBeVisible();
  await expect(page.getByTestId("investor-nvidia-state")).toHaveText("Opsiyonel sağlayıcı kullanılamıyor");
  await expect(page.getByText(/genel analiz hatası değildir/)).toBeVisible();

  await page.getByRole("button", { name: "Hazır demo vakasını aç" }).click();
  await expect(page.getByRole("heading", { name: "AtlasLens Ankara Pilot — Yatırımcı Özel Demo" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Kaynak, yetki ve amaç" })).toBeVisible();
  await expect(page.getByText("Yalnızca yerel", { exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Provider health ve kullanılan kaynaklar" })).toBeVisible();
  await expect(page.getByText("synthetic-fixture-provider", { exact: true }).first()).toBeVisible();
  await expect(page.getByRole("heading", { name: "Hipotez haritası" })).toBeVisible();
  await expect(page.getByText(/25[.,]000 m/).first()).toBeVisible();
  await expect(page.getByRole("heading", { name: "Normalize edilmiş kanıt" })).toBeVisible();

  const modelCard = page.locator(".hypothesis-card").filter({ hasText: "Model adayı" });
  await modelCard.getByLabel("Karar gerekçesi").fill("Yatırımcı demosunda ikinci bağımsız kanıt gerekli.");
  await modelCard.getByRole("button", { name: "Daha fazla kanıt iste" }).click();
  await expect(modelCard.locator(".decision-badge")).toHaveText("Daha fazla kanıt gerekli");

  await expect(page.getByText("Denetim özet zinciri doğrulandı")).toBeVisible();
  const limitations = page.locator("#demo-limitations");
  await expect(limitations.getByRole("heading", { name: "Bu neyi kanıtlamıyor?" })).toBeVisible();
  await expect(limitations.getByText(/Türkiye-geneli doğruluk iddiası bulunmamaktadır/)).toBeVisible();
  await expect(limitations.getByText(/kalibre edilmiş olasılık değildir/)).toBeVisible();
  await expect(limitations.getByText(/hukuken sertifikalı delil değildir/)).toBeVisible();

  await page.reload();
  await expect(page.getByRole("heading", { name: "Yetkili kaynaktan incelenebilir konum hipotezlerine" })).toBeVisible();
  await expect(page).toHaveURL(/demo=investor.*lang=tr/);
});
