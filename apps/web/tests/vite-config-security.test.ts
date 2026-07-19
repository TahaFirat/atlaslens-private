// @vitest-environment node

import { describe, expect, it } from "vitest";

import { assertNoFrontendCloudSecrets } from "../vite.config";

describe("Vite cloud credential boundary", () => {
  it.each([
    "VITE_NVIDIA_API_KEY",
    "VITE_NVIDIA_ACCESS_TOKEN",
    "VITE_OPENAI_CLIENT_SECRET",
    "VITE_OPENAI_CREDENTIAL",
  ])("rejects populated frontend credential variable %s", (name) => {
    expect(() => assertNoFrontendCloudSecrets({ [name]: "test-only-secret" })).toThrow(
      "Cloud API credentials are forbidden in frontend-prefixed variables",
    );
  });

  it("allows empty credential placeholders and unrelated public settings", () => {
    expect(() =>
      assertNoFrontendCloudSecrets({
        VITE_NVIDIA_API_KEY: "",
        VITE_DEV_API_TARGET: "http://127.0.0.1:8000",
      }),
    ).not.toThrow();
  });

  it("rejects a backend credential copied under an innocuous VITE name", () => {
    expect(() =>
      assertNoFrontendCloudSecrets({
        NVIDIA_API_KEY: "test-only-backend-secret",
        VITE_PUBLIC_VALUE: "test-only-backend-secret",
      }),
    ).toThrow("Cloud API credentials are forbidden in frontend-prefixed variables");
  });
});
