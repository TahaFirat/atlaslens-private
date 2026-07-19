import type { Capabilities } from "./api/schemas";

export type FileValidationCode = "empty" | "tooLarge" | "unsupported" | "signatureMismatch";

const FALLBACK_MAX_BYTES = 20 * 1024 * 1024;

function detectFormat(bytes: Uint8Array): "jpeg" | "png" | "webp" | null {
  if (bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) return "jpeg";
  if (
    bytes[0] === 0x89 &&
    bytes[1] === 0x50 &&
    bytes[2] === 0x4e &&
    bytes[3] === 0x47 &&
    bytes[4] === 0x0d &&
    bytes[5] === 0x0a &&
    bytes[6] === 0x1a &&
    bytes[7] === 0x0a
  )
    return "png";
  if (
    String.fromCharCode(...bytes.slice(0, 4)) === "RIFF" &&
    String.fromCharCode(...bytes.slice(8, 12)) === "WEBP"
  )
    return "webp";
  return null;
}

const MIME_FORMATS: Record<string, "jpeg" | "png" | "webp"> = {
  "image/jpeg": "jpeg",
  "image/png": "png",
  "image/webp": "webp",
};

export async function validateImageFile(file: File, capabilities?: Capabilities): Promise<FileValidationCode | null> {
  if (file.size === 0) return "empty";
  if (file.size > (capabilities?.max_upload_bytes ?? FALLBACK_MAX_BYTES)) return "tooLarge";
  const declared = MIME_FORMATS[file.type];
  if (!declared || (capabilities && !capabilities.supported_formats.includes(declared))) return "unsupported";
  const detected = detectFormat(new Uint8Array(await file.slice(0, 12).arrayBuffer()));
  if (!detected || detected !== declared) return "signatureMismatch";
  return null;
}
