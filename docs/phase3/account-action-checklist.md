# Phase 3A account action checklist

**Action now: wait.** Do not create an account, add billing, generate a key,
create storage, or launch compute in Phase 3A. The entries below describe what
would be needed only after an explicit Phase 3B approval.

Evidence date: 2026-07-16.

| Account or credential | Required? | Free to create? | Payment required? | Token/key required? | Create when | Never commit | User action now |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RunPod user account | Yes, for the selected cloud execution | RunPod documents sign-up separately from paid resource deployment; no account-creation fee is stated | Yes, before a Pod or volume. RunPod uses prepaid credits and requires at least one hour of credits for a new Pod | No key for console-only review; keys are separate rows below | Only after source, privacy, backup and $25 Recommended-MVP budget approval | Login recovery codes, billing data, session cookies | **WAIT** |
| RunPod restricted lifecycle API key | Yes if the mandatory external deadline/watchdog is automated | No separate charge | No separate key charge; resource use is paid | Yes. Use least privilege for Pod read/terminate and billing read; no broad or legacy key | After the account is approved and immediately before a paid preflight | API key, API responses containing account identifiers | **WAIT** |
| RunPod S3 API key | Yes for the planned network-volume upload/output synchronization | No separate key charge | Network-volume storage is paid | Yes; this is distinct from the RunPod API key and its secret is shown once | Only after the approved network volume exists | Access key, secret key, endpoint configuration containing secrets | **WAIT** |
| SSH key pair | Conditional; only if console access cannot perform the approved job | Local key generation has no service charge | No | Public key may be uploaded; private key never leaves the operator device | After RunPod account approval, if needed | Private key, passphrase, SSH agent material | **WAIT** |
| Docker Hub/container registry | No for the selected base-image plan | Not applicable | Not applicable | No | Do not create for Phase 3A/initial Phase 3B plan; use the digest-pinned public NVIDIA base | Any future registry token | **DO NOT CREATE** |
| Source-provider acquisition accounts | Conditional on the source-policy decision, not on RunPod | Provider-specific/unknown | Provider-specific/unknown | Provider-specific | Only after that source is `GO`/`GO_WITH_ATTRIBUTION` and acquisition is separately approved | API tokens, cookies, contributor identities, signed URLs | **WAIT** |
| Dedicated backup service account | Conditional: two independent verified copies are required before volume deletion, but an approved encrypted external medium may satisfy either copy without an account | Provider-specific/unknown | Provider-specific/unknown | Provider-specific | Select the complete backup architecture before any upload; create an account only if that approved architecture selects a service | Access keys, bucket credentials, encryption keys | **WAIT** |
| Vast.ai account | No; platform is not selected | No account-creation fee is asserted here | Prepaid credits are required before rental | CLI/API use would require a key | Only after a new architecture and budget decision | Vast API key, billing data | **DO NOT CREATE** |
| Google/Colab account or paid plan | No; platform is not selected | A no-cost Colab tier exists | Paid plans require payment; hardware remains availability-dependent | Google session credentials, not an AtlasLens API key | Not needed for this pilot | Google tokens, Drive share links | **DO NOT CREATE** |
| Hugging Face/model-provider account | No; the verified MegaLoc artifact already exists and no model download is planned | Not applicable | No | No | Do not create for this execution | Any future model token | **DO NOT CREATE** |

## Approved-order checklist for a later phase

Every box remains unchecked in Phase 3A.

- [ ] Source-policy decision permits commercial acquisition, derivative creation,
      cloud processing, caching/indexing and required retention.
- [ ] A legal/privacy reviewer has addressed first-party/partner imagery,
      personal data, blur state, cross-border processing and any required DPA.
- [ ] A professional privacy/security reviewer has reconciled RunPod's
      volume-disk-only encryption toggle with the DPA's broader at-rest
      representation, current subprocessors, selected region and Terms. The
      unresolved `[INSERT]` marker in the DPA security table has been clarified.
- [ ] Client-side application encryption has passed a restore test; no plaintext
      private route imagery will persist on the selected network volume or
      container disk, and the required keys remain outside RunPod and Git.
- [ ] Recommended MVP is the explicitly approved scenario: one A5000, 150 GB
      Standard volume, 11.25 expected/16 maximum GPU-hours, $15 soft/$25 hard.
- [ ] Public price and the signed-in deployment console still show an acceptable
      Secure Cloud A5000 rate and resource shape in the selected EU datacenter.
- [ ] C: and D: disk gates pass and the independent backup restore test succeeds.
- [ ] The encrypted governed source/rebuild archive and two independent backup
      copies are identified and restore-tested; C: is only a selective output
      destination, and the RunPod volume is not counted as a backup copy.
- [ ] RunPod account is created with a unique password and 2FA.
- [ ] Auto-pay is disabled; low-balance and Pod notifications are enabled.
- [ ] Only the approved credits/payment are added. No 3- or 6-month plan is
      purchased.
- [ ] Least-privilege lifecycle and S3 keys are created, stored in an OS credential
      manager/password manager, injected at runtime, and excluded from logs.
- [ ] An absolute UTC termination deadline is recorded before the first Pod.
- [ ] One small paid preflight passes before the corpus job.
- [ ] At completion, keys are revoked/rotated, Pod and volume deletion is verified,
      and the billing page is checked for zero unintended resources.

## Secret handling

- Never place a token in Git, `.env.example`, a manifest, notebook, command
  history, issue, chat, screenshot, log, provenance receipt or report.
- Use a password manager or OS credential manager. Inject secrets as ephemeral
  environment/file credentials with the narrowest permissions and shortest useful
  lifetime.
- Keep source-acquisition credentials outside the descriptor Pod whenever
  possible. Prefer locally governed, encrypted archives or short-lived signed
  URLs.
- Redact account IDs, signed URLs and bearer headers from logs. A receipt records
  only key ID/fingerprint, permission scope, creation/rotation time and revocation
  status.
- Revoke temporary lifecycle/S3 keys after backup and deletion verification.

## Primary account evidence

- [RunPod account creation](https://docs.runpod.io/accounts-billing/manage-accounts)
- [RunPod billing, credits, minimum balance and auto-pay](https://docs.runpod.io/accounts-billing/billing)
- [RunPod API key permissions and handling](https://docs.runpod.io/get-started/api-keys)
- [RunPod S3 API key handling](https://docs.runpod.io/storage/s3-api)
- [RunPod notification fields](https://docs.runpod.io/runpodctl/reference/runpodctl-user)
- [RunPod storage types and encryption-control scope](https://docs.runpod.io/pods/storage/types)
- [RunPod Data Processing Agreement](https://www.runpod.io/legal/data-processing-agreement)
- [RunPod Terms and shared responsibility](https://www.runpod.io/legal/terms-of-service)
- [Vast.ai prepaid billing](https://docs.vast.ai/guides/reference/billing)
- [Colab no-cost and paid-resource limitations](https://research.google.com/colaboratory/faq.html)

This is an operational checklist, not a request to provision anything and not
legal advice.
