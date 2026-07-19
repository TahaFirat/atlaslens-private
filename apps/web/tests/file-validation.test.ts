import { describe, expect, it } from "vitest";
import { validateImageFile } from "../src/file-validation";
import { capabilities, jpegFile } from "./fixtures";

describe("validateImageFile", () => {
  it("accepts a declared JPEG with a matching signature", async () => {
    await expect(validateImageFile(jpegFile(), capabilities)).resolves.toBeNull();
  });

  it("rejects unsupported and mismatched files before upload", async () => {
    const text = new File(["not an image"], "notes.txt", { type: "text/plain" });
    const disguised = new File(["not a jpeg"], "photo.jpg", { type: "image/jpeg" });
    await expect(validateImageFile(text, capabilities)).resolves.toBe("unsupported");
    await expect(validateImageFile(disguised, capabilities)).resolves.toBe("signatureMismatch");
  });

  it("honors the upload limit reported by the server", async () => {
    const tinyLimit = { ...capabilities, max_upload_bytes: 3 };
    await expect(validateImageFile(jpegFile(), tinyLimit)).resolves.toBe("tooLarge");
  });
});
