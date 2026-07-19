const PRIVATE_VALUE = /(?:[a-z]:[\\/]|\\\\|(?:^|[\s("'=])\/(?:[a-z0-9._-]+\/)+[a-z0-9._-]+|\b(?:sk-[a-z0-9_-]{8,}|api[_-]?key|access[_-]?token|bearer\s+[a-z0-9._-]{8,}))/i;

export function safeOperatorText(value: string | null | undefined): string | null {
  if (!value || PRIVATE_VALUE.test(value)) return null;
  return value;
}

export function safeRepositoryUrl(value: string | null | undefined): string | null {
  if (!value || PRIVATE_VALUE.test(value)) return null;
  try {
    const parsed = new URL(value);
    if (parsed.protocol !== "https:" || parsed.username || parsed.password || parsed.search || parsed.hash) return null;
    return parsed.toString();
  } catch {
    return null;
  }
}
