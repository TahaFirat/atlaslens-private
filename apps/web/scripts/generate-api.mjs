import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import openapiTS, { astToString } from "openapi-typescript";

const contractUrl = new URL("../../../packages/contracts/openapi.yaml", import.meta.url);
const outputUrl = new URL("../src/api/generated.ts", import.meta.url);
const banner = "// Generated from packages/contracts/openapi.yaml. Do not edit manually.\n";
const generated = `${banner}${astToString(await openapiTS(contractUrl))}`.replaceAll("\r\n", "\n");

if (process.argv.includes("--check")) {
  let current = "";
  try {
    current = (await readFile(outputUrl, "utf8")).replaceAll("\r\n", "\n");
  } catch {
    // A missing output is drift.
  }
  if (current !== generated) {
    console.error("Generated API types are stale. Run npm run generate:api.");
    process.exitCode = 1;
  }
} else {
  await writeFile(fileURLToPath(outputUrl), generated, "utf8");
  console.log("Generated src/api/generated.ts");
}
