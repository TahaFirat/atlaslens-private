import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { MapillaryDemoView } from "../src/components/MapillaryDemoView";
import { mapillaryDemoClient, type MapillaryDemoClient } from "../src/mapillary-demo/client";
import {
  mapillaryDemoStatusSchema,
  type MapillaryDemoQuery,
  type MapillaryDemoStatus,
} from "../src/mapillary-demo/models";
import { I18nProvider } from "../src/i18n";

const activeStatus: MapillaryDemoStatus = {
  state: "active",
  enabled: true,
  available: true,
  reason_code: null,
  city: "Ankara",
  image_count: 1_500,
  model_id: "gberton/MegaLoc",
  model_version: "megaloc-phase3b2-mapillary-private-demo-v1",
  model_artifact_sha256: "d4f9f2bcb60018f91eb6a8e061ed054fd55654e10c2569cf13841ea986ffb4f8",
  descriptor_dimension: 8_448,
  index_version: "mapillary-private-demo-faiss-flatip-v1",
  index_checksum: "a".repeat(64),
  attribution_url: "https://www.mapillary.com/",
  license_identifier: "CC-BY-SA-4.0",
  license_url: "https://creativecommons.org/licenses/by-sa/4.0/",
  experimental_status: "private_technical_demo_not_production",
  coverage_status: "limited_pilot",
  coverage_label: "Ankara reference pilot",
  retrieval_provider: "megaloc_mapillary_faiss",
  retrieval_scope: "ankara_reference_collection",
  result_semantics: "ankara_reference_collection_visual_similarity_not_general_geolocation",
  similarity_semantics: "cosine_similarity_not_confidence",
  supported_region: "Ankara pilot collection only",
  evidence_version: "phase3b3-mapillary-ankara-pilot-v1",
  benchmark_version: "phase3b3-mapillary-benchmark-v1",
  limitations: ["Bounded pilot-city coverage only."],
};

const completedQuery: MapillaryDemoQuery = {
  status: "completed",
  reason_code: null,
  analysis_scope: "ankara_reference_pilot",
  coverage_status: "pilot_eligible",
  coverage_label: "Ankara reference pilot",
  retrieval_provider: "megaloc_mapillary_faiss",
  retrieval_scope: "ankara_reference_collection",
  result_semantics: "ankara_reference_collection_visual_similarity_not_general_geolocation",
  abstained: false,
  abstention_reason: null,
  similarity_semantics: "cosine_similarity_not_confidence",
  supported_region: "Ankara pilot collection only",
  evidence_version: "phase3b3-mapillary-ankara-pilot-v1",
  benchmark_version: "phase3b3-mapillary-benchmark-v1",
  city: "Ankara",
  index_version: "mapillary-private-demo-faiss-flatip-v1",
  candidates: [{
    rank: 1,
    cosine_similarity: 0.75,
    cosine_distance: 0.25,
    latitude: 38.7225,
    longitude: 35.4875,
    mapillary_image_id: "stable-mapillary-id",
    contributor: "mapillary-contributor-id",
    source_url: "https://www.mapillary.com/app/?pKey=stable-mapillary-id",
    license_identifier: "CC-BY-SA-4.0",
    license_url: "https://creativecommons.org/licenses/by-sa/4.0/",
    capture_date: "2024-05-12",
    confidence: null,
    confidence_semantics: "uncalibrated_unavailable",
    similarity_semantics: "cosine_similarity_not_confidence",
    uncertainty_radius_m: 1_000,
    uncertainty_semantics: "presentation_radius_not_accuracy_or_probability",
    experimental_status: "private_technical_demo_not_production",
  }],
  confidence: null,
  confidence_semantics: "uncalibrated_unavailable",
  experimental_status: "private_technical_demo_not_production",
  limitations: ["Bounded pilot-city coverage only."],
};

function renderDemo(client: MapillaryDemoClient, locale: "en" | "tr" = "tr") {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  const view = render(
    <QueryClientProvider client={queryClient}>
      <I18nProvider initialLocale={locale}>
        <MapillaryDemoView client={client} />
      </I18nProvider>
    </QueryClientProvider>,
  );
  return { ...view, queryClient };
}

afterEach(() => vi.unstubAllGlobals());

it("routes the explicit UI action through the Ankara reference pilot scope", async () => {
  const fetchMock = vi.fn<typeof fetch>(() => Promise.resolve(new Response(JSON.stringify(completedQuery), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  })));
  vi.stubGlobal("fetch", fetchMock);

  await mapillaryDemoClient.query(new File(["safe-test"], "neutral.png", { type: "image/png" }), true);

  const options = fetchMock.mock.calls[0]?.[1] as RequestInit;
  expect(options.body).toBeInstanceOf(FormData);
  expect((options.body as FormData).get("analysis_scope")).toBe("ankara_reference_pilot");
});

describe("private Mapillary demo presentation", () => {
  it("shows the truthful Turkish active state, attribution, rank, distance, and uncertainty semantics", async () => {
    const user = userEvent.setup();
    const query = vi.fn(() => Promise.resolve(completedQuery));
    const client: MapillaryDemoClient = {
      getStatus: vi.fn(() => Promise.resolve(activeStatus)),
      query,
    };
    renderDemo(client);

    expect(await screen.findByRole("heading", { name: "Ankara Görsel Referans Pilotu" })).toBeInTheDocument();
    expect(screen.getByText("Özel teknik demo — üretim sistemi değildir")).toBeInTheDocument();
    expect(screen.getByText(/Türkiye geneli konum tespiti veya kalibre edilmiş doğruluk değildir/)).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("Etkin");
    expect(screen.getByRole("link", { name: "Mapillary" })).toHaveAttribute("href", "https://www.mapillary.com/");
    expect(screen.getByRole("link", { name: "CC-BY-SA-4.0" })).toHaveAttribute("href", "https://creativecommons.org/licenses/by-sa/4.0/");

    await user.upload(screen.getByLabelText("Demo sorgu görseli seç", { selector: "input" }), new File(["safe-test"], "fixture.png", { type: "image/png" }));
    await user.click(screen.getByRole("checkbox", { name: "Bu görseli analiz etmeye yetkiliyim." }));
    await user.click(screen.getByRole("button", { name: "Ankara referans pilotunda sorgula" }));

    expect(await screen.findByRole("heading", { name: "Ankara koleksiyonundaki görsel benzerlikler" })).toBeInTheDocument();
    expect(screen.getByText("0.250000")).toBeInTheDocument();
    expect(screen.getByText("0.750000")).toBeInTheDocument();
    expect(screen.getByText(/yalnızca harita sunumu içindir/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Mapillary #stable-mapillary-id" })).toHaveAttribute("href", completedQuery.candidates[0]?.source_url);
    expect(query).toHaveBeenCalledWith(expect.any(File), true);
  });

  it.each([
    ["disabled", "Devre dışı", "explicitly_disabled"],
    ["not_ready", "Hazır değil", "demo_index_incompatible"],
  ] as const)("renders the %s state without an upload action", async (state, label, reason) => {
    const client: MapillaryDemoClient = {
      getStatus: vi.fn(() => Promise.resolve({
        ...activeStatus,
        state,
        enabled: state !== "disabled",
        available: false,
        reason_code: reason,
        city: null,
        image_count: 0,
        model_version: null,
        model_artifact_sha256: null,
        index_version: null,
        index_checksum: null,
      })),
      query: vi.fn(() => Promise.resolve(completedQuery)),
    };
    renderDemo(client);
    expect(await screen.findByRole("status")).toHaveTextContent(label);
    expect(screen.queryByText("Demo sorgu görseli seç")).not.toBeInTheDocument();
  });

  it("requires a fresh authorization acknowledgement whenever the selected file changes", async () => {
    const user = userEvent.setup();
    const client: MapillaryDemoClient = {
      getStatus: vi.fn(() => Promise.resolve(activeStatus)),
      query: vi.fn(() => Promise.resolve(completedQuery)),
    };
    renderDemo(client, "en");
    const input = await screen.findByLabelText("Choose a demo query image", { selector: "input" });
    const checkbox = screen.getByRole("checkbox", { name: "I am authorized to analyze this image." });
    const button = screen.getByRole("button", { name: "Run Ankara reference pilot" });
    await user.upload(input, new File(["first"], "first.png", { type: "image/png" }));
    await user.click(checkbox);
    expect(checkbox).toBeChecked();
    expect(button).toBeEnabled();
    await user.upload(input, new File(["second"], "second.png", { type: "image/png" }));
    expect(checkbox).not.toBeChecked();
    expect(button).toBeDisabled();
  });

  it("renders an execution failure separately from evidence insufficiency", async () => {
    const user = userEvent.setup();
    const failedQuery: MapillaryDemoQuery = {
      ...completedQuery,
      status: "failed",
      reason_code: "demo_query_failed",
      abstention_reason: "demo_query_failed",
      abstained: true,
      candidates: [],
    };
    const client: MapillaryDemoClient = {
      getStatus: vi.fn(() => Promise.resolve(activeStatus)),
      query: vi.fn(() => Promise.resolve(failedQuery)),
    };
    renderDemo(client, "en");
    await user.upload(
      await screen.findByLabelText("Choose a demo query image", { selector: "input" }),
      new File(["safe-test"], "fixture.png", { type: "image/png" }),
    );
    await user.click(screen.getByRole("checkbox", { name: "I am authorized to analyze this image." }));
    await user.click(screen.getByRole("button", { name: "Run Ankara reference pilot" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("could not complete safely");
    expect(screen.queryByText(/Evidence was insufficient/)).not.toBeInTheDocument();
  });

  it("removes stale results when runtime status is no longer active", async () => {
    const user = userEvent.setup();
    const client: MapillaryDemoClient = {
      getStatus: vi.fn(() => Promise.resolve(activeStatus)),
      query: vi.fn(() => Promise.resolve(completedQuery)),
    };
    const { queryClient } = renderDemo(client, "en");
    await user.upload(
      await screen.findByLabelText("Choose a demo query image", { selector: "input" }),
      new File(["safe-test"], "fixture.png", { type: "image/png" }),
    );
    await user.click(screen.getByRole("checkbox", { name: "I am authorized to analyze this image." }));
    await user.click(screen.getByRole("button", { name: "Run Ankara reference pilot" }));
    expect(await screen.findByRole("heading", { name: "Visual similarities in the Ankara collection" })).toBeInTheDocument();
    queryClient.setQueryData(["mapillary-demo-status"], {
      ...activeStatus,
      state: "not_ready",
      available: false,
      reason_code: "demo_index_incompatible",
    });
    await waitFor(() => {
      expect(screen.queryByRole("heading", { name: "Visual similarities in the Ankara collection" })).not.toBeInTheDocument();
    });
    expect(screen.queryByLabelText("Choose a demo query image", { selector: "input" })).not.toBeInTheDocument();
  });

  it("rejects credential-shaped extra response fields", () => {
    expect(mapillaryDemoStatusSchema.safeParse({ ...activeStatus, access_token: "fake-token" }).success).toBe(false);
  });
});
