import type { Phase4Assessment } from "./api/schemas";

// A UI-local name for the generated, runtime-validated public contract.
// Keeping this as an alias prevents the panel from drifting from OpenAPI.
export type Phase4AssessmentView = Phase4Assessment;
