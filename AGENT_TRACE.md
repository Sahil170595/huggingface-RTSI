# Codex Build Trace

This is a high-level, reviewable action trace for the final audit pass. It records operations and outcomes without exposing private hidden reasoning.

## Scope

- Read the repository architecture, substrate, tests, deployment scripts, and challenge materials.
- Inspected the public Hugging Face Space and authenticated Modal/Hugging Face environments.
- Benchmarked leading Build Small submissions for packaging and compliance gaps.
- Exercised the app at desktop and mobile viewport sizes with Playwright.

## Findings acted on

1. Remote debate models were called sequentially even though Modal provides per-model container pools.
2. Hugging Face model downloads followed mutable repository branches.
3. Custom HTML relied on implicit Gradio padding and the mobile header pushed the main action below the fold.
4. Google Fonts used a CSS `@import` rejected by the browser's constructable stylesheet.
5. Submission docs incorrectly described the public Space as private and Modal as pending.
6. The README lacked official track, sponsor, and achievement tags.

## Changes

- Added concurrent remote model fan-out with deterministic result ordering.
- Added immutable model revision pins and coverage tests.
- Added responsive header spacing, explicit HTML padding, visible tab overflow, and disabled Gradio analytics.
- Moved font loading into the document head.
- Added Build Small metadata, field notes, Modal runtime documentation, and this trace.
- Kept claims conservative: no Tiny Titan or Best Demo claim without meeting those requirements.

## Verification

- Unit/integration suite: `315 passed`
- Source lint: clean
- Security static analysis: model revision finding resolved
- Browser console: no application errors after the font fix
- Desktop and mobile Gradio flows checked with Playwright
- Authenticated Modal smoke: Qwen2.5-0.5B returned `OK` in 7.7 seconds; unauthenticated request returned HTTP 401
- Public Space live debate: three models, two rounds, `CONDITIONAL` consensus in 34.8 seconds
- Public Space walkthrough recorded at 1280x720: `demo/quantsafe-demo.webm` (68.96 seconds)
- Final Space transferred into the official `build-small-hackathon` organization with secrets preserved

OpenAI Codex performed this audit and implementation pass in collaboration with the repository owner.

---

# Claude Code Build Trace

High-level action trace for the SOTA-refresh and live-deployment pass that preceded the Codex hardening above. Records operations and outcomes without exposing private reasoning.

## Scope

- Ran a multi-agent audit of the whole repository (8 dimensions: claim/data consistency, core scoring math, live inference, judges/debate, the Ed25519 certificate, the Gradio app, SOTA currency, repo hygiene), with every finding adversarially re-verified before acceptance — 62 confirmed findings.
- Verified current-generation model availability and license-gating on the Hub before any swap.
- Deployed and exercised the Modal GPU backend and the public Hugging Face Space, including end-to-end Playwright runs of both live-inference paths.

## Findings acted on

1. `is_refusal` counted compliance openings ("I can tell you how to…") as refusals, inverting live verdicts.
2. A total refusal collapse (zero candidate refusals) scored LOW instead of escalating.
3. Inter-judge kappa used scikit-learn/statsmodels, neither pinned in `requirements.txt`; correlated judge failures could inflate kappa to a vacuous 1.0 RELIABLE.
4. A 1–1 debate tie rendered as "CONSENSUS"; certificate verification trusted the cert's own embedded key, so a re-signed forgery passed.
5. The Modal live backend spoke a protocol the deployed endpoint did not, the endpoint was unauthenticated, and the CPU loader had an unbounded cache plus a double-BOS tokenization bug.
6. The live-screen tab failed at model load on the Space because `accelerate` shipped in the Modal image but was missing from `requirements.txt` (caught only by a real in-Space inference run).
7. Pinned gradio 5.9.1 carried in-line 5.x security fixes; the debate/judge/live model cohort was two generations stale.
8. Submission docs described a debate cache (3×≤1.7B models, ROUTE@0.67) that a prior commit had already replaced.

## Changes

- Correctness pass across `features.py`, `rtsi_core.py`, `judges.py`, `debate.py`, `cert_signer.py`, `inference.py`, `modal_app.py`, and the `app.py` integration: token-boundary refusal detection, degenerate-input escalation/UNKNOWN, closed-form Cohen/Fleiss kappa (sklearn/statsmodels removed), an unclear-rate gate that forces band INVALID, a 2/3 `consensus_label`, pinned-issuer-key verification with a "foreign re-sign" demo and `allow_nan=False` canonical JSON, a corrected and bearer-authenticated Modal protocol with a bounded CPU model cache and single-pass chat templating.
- SOTA cohort refresh (all ≤8.2B, all ungated): debate Qwen3-8B + Phi-4-mini-instruct + SmolLM3-3B (odd count → strict majority), judges Qwen3Guard-Gen-8B + Granite-Guardian-3.3-8b, live CPU scorers Qwen3-0.6B/1.7B; reasoning-mode suppression on both CPU and GPU paths. The 45-cell substrate was deliberately frozen (2024 checkpoints) to preserve AUC 0.8445 and the anchor cells.
- Added a Modal `/judge` endpoint and the `quantsafe-auth` secret; regenerated both caches on real GPUs (`scripts/regen_debate.py`, `scripts/regen_judges.py` with a RELIABLE-band safety valve) → debate CONDITIONAL@0.67 (genuine 2/3 consensus, one model shifting DEPLOY→ROUTE) and judge kappa 0.7531 RELIABLE, 35/40 agree.
- Pinned `requirements.txt` (incl. `gradio==5.50.0` and `accelerate`), synced SUBMISSION/STORYBOARD/social docs to the regenerated artifacts, published the cert issuer pubkey, and added `scripts/deploy_space.py` (uploads + sets secrets, excludes local key material).

## Verification

- Unit/integration suite: `165 → 310 passed` through this pass (`315` after the later model-revision tests).
- Claims re-traced to artifacts: kappa 0.7531 / 35-of-40 / debate CONDITIONAL@0.67; frozen substrate (45 cells, 23/13/9, AUC 0.8445, 0.7864 / 0.6199) unchanged; exposure grep clean.
- Authenticated Modal smoke: Qwen2.5-0.5B returned text + `fp16` disclosure on a cold start; unauthenticated and unknown-model requests returned HTTP 401 / 400 with clean detail messages.
- Public-Space Playwright E2E (gradio 5.50.0): all six tabs switch with no hang; deep-links resolve; Score shows 0.7864 and 0.6199 HIGH; Judge reads kappa 0.75 / 35-of-40; certificate issue → verify (pinned key `9a074a15…`) → tamper → foreign-re-sign all behave; cached and **live** Modal debate both reach CONDITIONAL@67% consensus.
- Live CPU screen run completed in-Space after the `accelerate` fix: refusal-drift 0.6740 HIGH with all four feature deltas rendered, no OOM and no think-leak.
- Cert issuer public key in README matches the Space signing key.

Claude Code (Anthropic) performed this SOTA-refresh and live-deployment pass in collaboration with the repository owner.

---

# Final Production Issuer Correction

High-level trace for the final certificate-identity repair and release
verification.

## Finding

- A live certificate from the organization-owned Space carried an ephemeral
  Ed25519 public key instead of the issuer key published in the README.
- The in-app verifier was internally consistent but circular: it compared the
  certificate with the runtime key rather than independently pinning the
  published issuer identity.

## Changes

- Replaced the organization Space's `GRADIO_CERT_SIGNING_KEY_HEX` secret through
  the authenticated Hugging Face settings UI. The private key derives to the
  README-published public key
  `9a074a15598fef26f5fbd33e8d604cb6c2372989f164331c11018a83fcd98519`.
- Added a production fail-closed gate: certificate issuance is disabled on
  Hugging Face if the runtime key does not match the published issuer.
- Changed production verification and the foreign re-sign test to pin directly
  to the published issuer fingerprint.
- Added an explicit issuer-status panel to the certificate tab.
- Corrected the field notes to retain the official total-catalog interpretation
  of the 32B model limit.

## Verification

- `330` tests passed; production Ruff and Python compilation checks passed.
- GitHub PR `#4` merged the submission polish and issuer hardening.
- Hugging Face Space PR `#4` merged the production fix.
- The rebuilt organization Space issued a certificate containing the published
  `9a074a15...` key and returned `VALID` under pinned verification.
- After a full Space restart, a newly issued certificate carried the same
  published key. No ephemeral-key fallback was active.

OpenAI Codex performed this correction and production verification pass in
collaboration with the repository owner.

---

# Claude Code Editorial-Restyle Trace

High-level action trace for the visual-design pass that re-skinned the Gradio Space to an "editorial / quiet-luxury" aesthetic. Landed as commit `15397a7` (an ancestor of the SOTA-refresh / Codex merges above). Records operations and outcomes without exposing private reasoning.

## Scope

- Re-skinned `app.py` only — no scoring, inference, judge, debate, or certificate logic touched. The 45-cell substrate and all headline numbers were left byte-identical.
- Worked from a single owner-chosen direction (light warm-neutral ground, ink text, one restrained oxblood accent, serif display, generous whitespace — "The Economist / academic-press" register).
- Verified the rendering on the live instance, not just in code, after the gradio-6.x tab-switch-hang lesson.

## Changes

- **Palette** — ivory ground (`#FAF9F6`), ink text (`#1A1A1A`), warm-gray rules (`#E5E0D8`). A single oxblood accent (`#7B2D26`) replaces both the indigo chrome and the alarm-red, doing double duty as primary-action and HIGH/ROUTE band, separated by context (squared button vs. filled pill). Risk bands de-loudened from green/amber/red to muted **sage `#4F6F52` · ochre `#9A7B3A` · oxblood `#7B2D26`** over soft tints; every cool slate-gray swapped to a warm neutral.
- **Type & theme** — replaced `gr.themes.Soft(indigo/red)` with a custom `gr.themes.Base()` plus a `gr.Blocks(css=…)` block: **Fraunces** serif for the wordmark, headings, and tabular numerals; **Hanken Grotesk** body; **Spline Sans Mono** mono. Tab bar changed from filled indigo pills to a quiet underline-active bar; primary buttons squared and letter-spaced.
- **Header** — dropped the 🛡️ emoji and the purple-on-white gradient; rebuilt as a letterspaced eyebrow, a serif ink "QuantSafe" wordmark, an italic oxblood tagline, and a thin gold hairline.
- **Components** — badges/cards moved to 1px borders, 6px radii, uppercase letterspaced labels, and serif numerals; a shared `_editorial_layout()` gave every Plotly figure a transparent ground (charts sit on the ivory), serif titles, warm hairline gridlines, and an oxblood/sage colorway. The heatmap caption was updated to the new band names.

## Verification

- Module import: theme tokens, Google fonts, CSS, and the `Blocks` graph all construct (`THEME_OK Base` / `DEMO_OK Blocks`).
- Local launch (gradio 5.9.1) + Playwright across all six tabs: each switches in under two seconds and renders; Score, Debate, and Certificate inspected for the editorial treatment.
- Deployed `app.py` to the Space `Crusadersk/quantsafe-certifier`; Space rebuilt to `RUNNING`; briefly flipped public, Playwright-screenshotted the live instance to confirm the restyle, then restored to private.
- Committed `15397a7` and pushed to GitHub `main`; no logic files in the diff (`app.py` only, +262 / −136).

Claude Code (Anthropic) performed this editorial-restyle pass in collaboration with the repository owner.

---

# Judge Demo Recut and Final Preflight

High-level trace for the June 14, 2026 demo and submission-hardening pass.

## Findings

- The prior 48-second cut used large black caption/title frames that obscured
  production evidence and spent too much time on static transitions.
- Submission docs still advertised the obsolete cut, and the local launch
  package still described a removed "Live Screen" workflow.
- A reported Gradio deprecation log came from the earlier 5.50 runtime. The
  repository now pins Gradio 6.18 and already routes theme, CSS, head content,
  and private event visibility through the Gradio 6 APIs.

## Changes

- Re-captured the organization-owned production Space at 1280x720 across the
  measured failure, Pareto route decision, real ZeroGPU run, signed record,
  pinned-key verification, tamper failure, three-family debate, and evidence
  page.
- Recut the demo into a 49.4-second judge narrative with restrained ivory
  lower thirds, hard captions, short fades, no black interstitials, and a
  direct 91% to 1% opening hook.
- Added a reproducible `scripts/build_demo.py` renderer and both VP9/WebM and
  H.264/MP4 outputs. The MP4 is prepared for the required social post.
- Updated the storyboard, README, submission checklist, and local social copy
  to match the current "Exploratory live probe" workflow and production claims.

## Live Evidence and Verification

- A real production ZeroGPU run compared Qwen3-0.6B with Qwen3-1.7B over ten
  private probes and completed in 27 seconds. Only aggregate outputs were
  displayed; the result remained explicitly exploratory.
- Media inspection: 49.4 seconds, 1280x720, 30 fps; H.264 MP4 and VP9 WebM;
  no black frames detected; contact-sheet and title/close frames reviewed.
- Gradio 6.18 app construction passed with deprecation warnings promoted to
  errors. The full repository suite passed: 348 tests.
- Ruff, Python compilation, diff whitespace checks, and a focused credential
  exposure scan passed. No private key or bearer token was added to the demo,
  repository, or trace.
- GitHub commit `e02ac62` passed CI. The organization Space reached `RUNNING`
  at revision `86370de`, and the public MP4/WebM hashes matched the local
  release byte-for-byte.
- The official submission form was loaded with the final public README. Four
  verified preflight checks and the Backyard AI, OpenAI, Modal, Well-Tuned,
  Off-Brand, Llama Champion, Sharing is Caring, and Field Notes considerations
  were staged. The social-post check remains owner-dependent.
- The field-guide README generator attempted to append a duplicate `tags:`
  key because the Space already had a populated tag block. Its generated copy
  was deliberately not committed; the live README already contains the exact
  canonical tags once.

OpenAI Codex performed this pass in collaboration with the repository owner.

---

# Claude Code Nemotron Judge, Uncertainty, and Benchmark Pass

High-level action trace for the third-judge upgrade, the statistical-honesty
layer, the published benchmark, and the "who this is for" repositioning. Records
operations and outcomes without exposing private reasoning.

## Scope

- Verified NVIDIA's `Llama-3.1-Nemotron-Safety-Guard-8B-v3` availability,
  license, and inference contract on the Hub before proposing it as a judge.
- Re-derived inter-judge agreement after moving from a 2-rater to a 3-rater
  cohort, then quantified the uncertainty in that estimate rather than reporting
  the point estimate alone.
- Published the judge corpus and verdicts as an open, citable dataset.
- Repositioned the public-facing docs around the people actually deploying tiny
  local LLMs, not the publisher-first angle.

## Findings acted on

1. A two-judge cohort can only report Cohen's kappa; a single correlated failure
   moves the estimate a lot, and there was no third family to break ties.
2. The headline kappa was being reported as a bare point estimate, with no
   interval and no test of whether the top two judges were actually
   distinguishable on a 40-item corpus.
3. The app hardcoded "Cohen" and "two judges" in prose, and `_agreement_breakdown`
   computed unanimity against a fixed rater count instead of the live N.
4. The judge corpus was described as "held internally," which is weaker and less
   honest than publishing it.

## Changes

- Added NVIDIA NemoGuard as the third `SafetyJudge` (S1–S23 taxonomy, fail-closed
  JSON parsing), moving the cohort from Cohen's kappa to **Fleiss' kappa = 0.7929
  RELIABLE** (n=3 judges, 40 items), up from the prior two-judge Cohen 0.7484.
  Landed as **Hugging Face Space PR #18** (merged); also carries the
  `sponsor:nvidia` tag and the per-model 32B framing.
- Added a `statistical_uncertainty` block to `substrate/judge_results.json`: a
  stratified (by zone) percentile bootstrap **95% CI of [0.6641, 0.9239]** for
  the kappa (seed 20260614, 10,000 resamples), and an exact-paired **McNemar**
  test on the top two judges (Nemotron 0.95 vs Granite 0.925; 3 discordant pairs;
  two-sided p = 1.0). The honest reading is stated plainly: the interval spans
  MODERATE to ALMOST-PERFECT, and the top two judges are statistically
  indistinguishable on this corpus.
- Published the open benchmark `Crusadersk/quantsafe-judge-benchmark`
  (`scripts/publish_judge_benchmark.py`): the corpus, all three judges' verdicts,
  the agreement + uncertainty results, and the immutable run manifest, with a
  generated dataset card. Linked it from the README and softened the
  "held internally" copy to point at the published dataset.
- Repositioned README + the app's About copy around the audience — people putting
  tiny local LLMs into semi-professional use (the family-business operator) who
  need refusals to survive quantization without a research team — and cited the
  RTSI paper (arXiv:2606.10154) for the refusal-stability method.
- Made the app read its own evidence: `app.py` now interpolates the agreement
  method and judge count from `JUDGE_RESULTS` instead of hardcoding them, and
  `_agreement_breakdown` reports unanimity across the live N raters.

## Attribution note (convergent work)

The Nemotron third-judge upgrade was genuinely **convergent**, not sole-authored.
Claude designed and merged it as Space PR #18; in parallel, Codex hardened the
same judge in committed code (`00f1a8d`, see the next section). Both lanes land on
the identical artifact — three judges, Fleiss 0.7929 RELIABLE, n=40, Nemotron the
95% top point estimate. The provenance block records `code_sha 00f1a8d` as the
generating commit; the statistical-uncertainty and benchmark-publication layers
sit on top of that committed evidence. Neither agent gets sole credit for the
judge itself.

## Verification

- `substrate/judge_results.json` re-read from the working tree: `agreement.kappa`
  0.7929249…, `method` fleiss, `band` RELIABLE, `n_judges` 3, `n_items` 40; the
  three judges include `nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3`.
- `statistical_uncertainty` re-read: bootstrap CI [0.6641477…, 0.9238856…],
  McNemar discordant 3 / p 1.0 — numbers in this trace match the artifact.
- README working tree carries the dataset link and the published-benchmark copy.
- Space PR #18 confirmed merged via the Hub discussions API; GitHub PR numbering
  (max #6) is separate and was not used for this change.

Claude Code (Anthropic) performed this judge-upgrade, uncertainty, and benchmark
pass in collaboration with the repository owner.

---

# Codex Nemotron Evidence Hardening Pass

High-level action trace for the committed production hardening of the Nemotron
judge path. Landed as commit `00f1a8d` ("feat: harden Nemotron judge evidence",
owner + OpenAI Codex co-author). Records operations and outcomes without exposing
private reasoning.

## Scope

- Hardened the committed code that generates and serves the three-judge
  evidence, so the published Fleiss 0.7929 result is reproducible from a pinned
  commit rather than an ad-hoc run.

## Changes

- Added native **BF16** loading for the Nemotron guard on Modal and strict,
  fail-closed response parsing for its S1–S23 verdict format.
- Added **immutable run manifests** (`substrate/judge_runs/judge-run-*.json`) and
  explicit cache promotion, so a regeneration is recorded before it replaces the
  cached result.
- Wrote a **provenance block** into `substrate/judge_results.json`: `code_sha`
  (`00f1a8dcb49f…`), `corpus_sha256`, per-model pinned revisions (Nemotron
  `8fdc246b…`), and per-judge generation settings (Nemotron 128 new tokens vs 48
  for the smaller guards, greedy decoding) plus reported dtype.
- Hardened `scripts/regen_judges.py`, the `modal_app.py` judge branch, and the
  public probe paths in `app.py`.

## Verification

- `git show 00f1a8d --stat`: judges.py (+360), regen_judges.py (+645),
  modal_app.py, app.py, judge_results.json, and three test files
  (test_judges.py +546, test_modal_policy.py +233, test_app_tabs.py +113);
  Co-authored-by: OpenAI Codex trailer present.
- `provenance.code_sha` in the artifact equals the commit SHA `00f1a8d…`,
  confirming the published result traces to this commit.

OpenAI Codex performed this evidence-hardening pass in collaboration with the
repository owner.
