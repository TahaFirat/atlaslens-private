import { randomUUID } from "node:crypto";
import { fileURLToPath } from "node:url";
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

const repositoryRoot = fileURLToPath(new URL("../../", import.meta.url));

export function assertNoFrontendCloudSecrets(
  ...sources: ReadonlyArray<Record<string, string | undefined>>
): void {
  const backendSecretValues = new Set(
    sources.flatMap((source) =>
      Object.entries(source).flatMap(([name, secret]) => {
        const normalized = name.toUpperCase();
        const selected = secret?.trim();
        return selected &&
          !normalized.startsWith("VITE_") &&
          (normalized.includes("OPENAI") || normalized.includes("NVIDIA")) &&
          ["KEY", "SECRET", "TOKEN", "CREDENTIAL"].some((marker) =>
            normalized.includes(marker),
          )
          ? [selected]
          : [];
      }),
    ),
  );
  const leaked = sources.some((source) =>
    Object.entries(source).some(([name, secret]) => {
      const normalized = name.toUpperCase();
      const selected = secret?.trim();
      if (!normalized.startsWith("VITE_") || !selected) return false;
      return (
        ((normalized.includes("OPENAI") || normalized.includes("NVIDIA")) &&
          ["KEY", "SECRET", "TOKEN", "CREDENTIAL"].some((marker) =>
            normalized.includes(marker),
          )) || backendSecretValues.has(selected)
      );
    }),
  );
  if (leaked) {
    throw new Error("Cloud API credentials are forbidden in frontend-prefixed variables");
  }
}

export default defineConfig(({ mode }) => {
  const ignoreEnvFile = process.env.ATLASLENS_IGNORE_ENV_FILE === "true";
  const fileEnv = ignoreEnvFile ? {} : loadEnv(mode, repositoryRoot, "");
  assertNoFrontendCloudSecrets(process.env, fileEnv);
  const value = (name: string, fallback: string) => process.env[name] ?? fileEnv[name] ?? fallback;
  const apiPort = Number(value("API_PORT", "8000"));
  const webPort = Number(value("VITE_DEV_PORT", "5173"));
  const apiTarget = value("VITE_DEV_API_TARGET", `http://127.0.0.1:${apiPort}`).replace(/\/$/, "");

  return {
    envDir: repositoryRoot,
    plugins: [react()],
    server: {
      host: "127.0.0.1",
      port: webPort,
      strictPort: true,
      proxy: {
        "/api": {
          target: apiTarget,
          changeOrigin: true,
          configure(proxy) {
            proxy.on("error", (_error, _request, response) => {
              if (response.headersSent) return;
              const payload = JSON.stringify({
                type: "about:blank",
                title: "AtlasLens backend unavailable",
                status: 502,
                code: "backend_unreachable",
                message_key: "errors.backend_unreachable",
                request_id: `vite-proxy-${randomUUID()}`,
                retry_after_seconds: null,
              });
              response.writeHead(502, {
                "Content-Type": "application/problem+json",
                "Cache-Control": "no-store",
                "Content-Length": Buffer.byteLength(payload),
              });
              response.end(payload);
            });
          },
        },
      },
    },
    preview: {
      host: "127.0.0.1",
      port: 4173,
    },
  };
});
