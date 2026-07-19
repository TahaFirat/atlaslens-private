import { randomUUID } from "node:crypto";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AtlasLensApp } from "../src/App";
import { ApiError } from "../src/api/client";
import {
  capabilities,
  createFakeClient,
  investigationCase,
} from "./fixtures";

const authorizedCase = {
  ...investigationCase,
  title: "Authorized investor fixture case",
  description: "Explicit synthetic frontend fixture; not real demo evidence or an accuracy claim.",
  source_context: "Authorized synthetic frontend test source.",
  sensitivity: "standard" as const,
};

const disabledNvidiaCapabilities = {
  ...capabilities,
  enabled_analysis_modes: ["local_only" as const],
  providers: {
    ...capabilities.providers,
    cloud_vision: {
      provider_id: "nvidia-qwen-vision-reasoning",
      enabled: false,
      available: false,
      execution_boundary: "cloud" as const,
      reason_code: "disabled",
      operational_status: "disabled" as const,
    },
  },
};

function casePage(items = [authorizedCase]) {
  return {
    items,
    total: items.length,
    limit: 100,
    offset: 0,
    ordering: "updated_at_desc_id_desc" as const,
  };
}

async function openFirstAuthorizedCase(user: ReturnType<typeof userEvent.setup>) {
  const authorizedCases = await screen.findByTestId("investor-authorized-cases");
  await user.click(within(authorizedCases).getByText(/Diğer yetkili vakalar/));
  await user.click(within(authorizedCases).getAllByTestId("investor-authorized-case-open")[0]!);
}

describe("Phase 3D guided investor demo", () => {
  beforeEach(() => {
    window.history.replaceState(null, "", "/?demo=investor&lang=tr");
    vi.stubGlobal("WebGLRenderingContext", undefined);
  });

  afterEach(() => {
    window.history.replaceState(null, "", "/");
  });

  it("loads an exact caseId directly, persists it across remount, and keeps claims evidence-backed", async () => {
    window.history.replaceState(
      null,
      "",
      `/?demo=investor&lang=tr&caseId=${authorizedCase.id}`,
    );
    const listCases = vi.fn(() => Promise.resolve(casePage()));
    const getCase = vi.fn(() => Promise.resolve(authorizedCase));
    const { client } = createFakeClient({
      capabilityState: disabledNvidiaCapabilities,
      clientOverrides: { listCases, getCase },
    });

    const firstMount = render(<AtlasLensApp client={client} />);
    expect(await screen.findByRole("heading", { name: "Sorgu görseli" })).toBeVisible();
    expect(document.documentElement).toHaveAttribute("lang", "tr");
    expect(new URL(window.location.href).searchParams.get("caseId")).toBe(authorizedCase.id);
    expect(listCases).not.toHaveBeenCalled();
    expect(screen.getByText("synthetic-metadata-fixture · synthetic-visual-fixture")).toBeVisible();
    expect(screen.getByText("Ankara referans koleksiyonu içinde görsel benzerlik araması.")).toBeVisible();
    expect(screen.getByText("Türkiye-geneli konum tespiti veya kalibre edilmiş doğruluk değildir.")).toBeVisible();
    expect(await screen.findByText(/confidence veya olasılık değildir/)).toBeVisible();
    expect(screen.queryByText(authorizedCase.source_context)).not.toBeInTheDocument();

    firstMount.unmount();
    render(<AtlasLensApp client={client} />);
    expect(await screen.findByRole("heading", { name: "Sorgu görseli" })).toBeVisible();
    expect(new URL(window.location.href).searchParams.get("caseId")).toBe(authorizedCase.id);
    expect(getCase.mock.calls.length).toBeGreaterThanOrEqual(2);
  });

  it("keeps a missing caseId on a warned landing until explicit authorized-case selection", async () => {
    const getCase = vi.fn(() => Promise.resolve(authorizedCase));
    const { client } = createFakeClient({
      capabilityState: disabledNvidiaCapabilities,
      clientOverrides: {
        listCases: vi.fn(() => Promise.resolve(casePage())),
        getCase,
      },
    });
    const user = userEvent.setup();
    render(<AtlasLensApp client={client} />);

    expect(await screen.findByTestId("investor-route-notice")).toHaveTextContent(/hazırlanmış bir vakayı tanımlamıyor/i);
    expect(getCase).not.toHaveBeenCalled();
    await openFirstAuthorizedCase(user);

    expect(await screen.findByRole("heading", { name: "Sorgu görseli" })).toBeVisible();
    expect(getCase).toHaveBeenCalledWith(authorizedCase.id);
    expect(new URL(window.location.href).searchParams.get("caseId")).toBe(authorizedCase.id);
  });

  it("rejects malformed and duplicate caseId values before any detail request", async () => {
    const getCase = vi.fn(() => Promise.resolve(authorizedCase));
    const { client } = createFakeClient({
      capabilityState: disabledNvidiaCapabilities,
      clientOverrides: {
        listCases: vi.fn(() => Promise.resolve(casePage())),
        getCase,
      },
    });
    window.history.replaceState(
      null,
      "",
      `/?demo=investor&lang=tr&caseId=not-a-uuid&caseId=${authorizedCase.id}`,
    );
    const user = userEvent.setup();
    render(<AtlasLensApp client={client} />);

    expect(await screen.findByTestId("investor-route-notice")).toHaveTextContent(/caseId geçersiz/i);
    expect(getCase).not.toHaveBeenCalled();
    expect(new URL(window.location.href).searchParams.has("caseId")).toBe(false);

    await openFirstAuthorizedCase(user);
    expect(await screen.findByRole("heading", { name: "Sorgu görseli" })).toBeVisible();
    expect(getCase).toHaveBeenCalledWith(authorizedCase.id);
  });

  it("keeps a valid nonexistent caseId on the 404 recovery view with retry and back", async () => {
    const nonexistentId = randomUUID();
    window.history.replaceState(
      null,
      "",
      `/?demo=investor&lang=tr&caseId=${nonexistentId}`,
    );
    const getCase = vi.fn(() => Promise.reject(
      new ApiError("errors.api_not_found", undefined, "api_not_found", 404),
    ));
    const { client } = createFakeClient({
      capabilityState: disabledNvidiaCapabilities,
      clientOverrides: { getCase },
    });
    const user = userEvent.setup();
    render(<AtlasLensApp client={client} />);

    const error = await screen.findByRole("alert", undefined, { timeout: 12_000 });
    expect(within(error).getByRole("heading", { name: "İstenen vaka bulunamadı." })).toBeVisible();
    expect(within(error).getByRole("button", { name: "Yeniden dene" })).toBeVisible();
    expect(within(error).getByRole("button", { name: "Vakalara dön" })).toBeVisible();
    expect(new URL(window.location.href).searchParams.get("caseId")).toBe(nonexistentId);
    expect(getCase).toHaveBeenCalledTimes(1);

    const requestsBeforeRetry = getCase.mock.calls.length;
    await user.click(within(error).getByRole("button", { name: "Yeniden dene" }));
    await waitFor(() => expect(getCase.mock.calls.length).toBeGreaterThan(requestsBeforeRetry));
    expect(new URL(window.location.href).searchParams.get("caseId")).toBe(nonexistentId);
  }, 15_000);

  it("lists same-title authorized cases without semantic auto-selection", async () => {
    const secondCase = { ...authorizedCase, id: randomUUID() };
    const getCase = vi.fn((caseId: string) => Promise.resolve(
      caseId === secondCase.id ? secondCase : authorizedCase,
    ));
    const { client } = createFakeClient({
      capabilityState: disabledNvidiaCapabilities,
      clientOverrides: {
        listCases: vi.fn(() => Promise.resolve(casePage([authorizedCase, secondCase]))),
        getCase,
      },
    });
    const user = userEvent.setup();
    render(<AtlasLensApp client={client} />);

    const authorizedCases = await screen.findByTestId("investor-authorized-cases");
    await user.click(within(authorizedCases).getByText(/Diğer yetkili vakalar/));
    const openButtons = within(authorizedCases).getAllByTestId("investor-authorized-case-open");
    expect(openButtons).toHaveLength(2);
    expect(getCase).not.toHaveBeenCalled();
    await user.click(openButtons[1]!);

    expect(await screen.findByRole("heading", { name: "Sorgu görseli" })).toBeVisible();
    expect(getCase).toHaveBeenCalledWith(secondCase.id);
    expect(new URL(window.location.href).searchParams.get("caseId")).toBe(secondCase.id);
  });

  it("keeps the guided flow available when the optional NVIDIA provider fails", async () => {
    const failedNvidiaCapabilities = {
      ...disabledNvidiaCapabilities,
      providers: {
        ...disabledNvidiaCapabilities.providers,
        cloud_vision: {
          ...disabledNvidiaCapabilities.providers.cloud_vision,
          enabled: true,
          reason_code: "upstream_timeout",
          operational_status: "failed" as const,
        },
      },
    };
    window.history.replaceState(
      null,
      "",
      `/?demo=investor&lang=tr&caseId=${authorizedCase.id}`,
    );
    const { client } = createFakeClient({
      capabilityState: failedNvidiaCapabilities,
      clientOverrides: { getCase: vi.fn(() => Promise.resolve(authorizedCase)) },
    });
    render(<AtlasLensApp client={client} />);

    expect(await screen.findByRole("heading", { name: "Sorgu görseli" })).toBeVisible();
    expect(screen.getByText("Ankara referans koleksiyonu içinde görsel benzerlik araması.")).toBeVisible();
    expect(await screen.findByText(/confidence veya olasılık değildir/)).toBeVisible();
  });

  it("does not invent an evidence source when the selected case has none", async () => {
    window.history.replaceState(
      null,
      "",
      `/?demo=investor&lang=tr&caseId=${authorizedCase.id}`,
    );
    const { client } = createFakeClient({
      capabilityState: disabledNvidiaCapabilities,
      clientOverrides: {
        getCase: vi.fn(() => Promise.resolve(authorizedCase)),
        listCaseEvidence: vi.fn(() => Promise.resolve({
          items: [],
          total: 0,
          limit: 100,
          offset: 0,
          ordering: "created_at_asc_id_asc" as const,
        })),
      },
    });
    render(<AtlasLensApp client={client} />);

    const user = userEvent.setup();
    await screen.findByRole("heading", { name: "Sorgu görseli" });
    await user.click(screen.getByRole("button", { name: /Kanıt ve teknik ayrıntılar/ }));
    expect(screen.getByText(/kanıt uydurulmadı/)).toBeVisible();
    expect(screen.queryByText("synthetic-metadata-fixture")).not.toBeInTheDocument();
    expect(screen.queryByText("synthetic-visual-fixture")).not.toBeInTheDocument();
  });

  it("keeps the language switcher active and the landing locale reload-stable", async () => {
    const { client } = createFakeClient({
      capabilityState: disabledNvidiaCapabilities,
      clientOverrides: { listCases: vi.fn(() => Promise.resolve(casePage())) },
    });
    const user = userEvent.setup();
    render(<AtlasLensApp client={client} />);
    await screen.findByRole("heading", { name: "Yetkili kaynaktan incelenebilir konum hipotezlerine" });
    await user.click(screen.getByRole("button", { name: "EN" }));
    expect(screen.getByRole("heading", { name: "From authorized source to reviewable location hypotheses" })).toBeVisible();
    expect(window.location.search).toContain("lang=en");
    expect(window.location.search).toContain("demo=investor");
  });
});
