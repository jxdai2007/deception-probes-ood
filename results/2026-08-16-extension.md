# 04 extension — OOD behavior of Apollo's deception probes at small scale

2026-08-16, branch `feat/common-harness`. Raw JSONs + figures in this
directory; code `p04/` (run: `python -m p04.run_extension`). Vendor pin +
detector sha256s: `../VENDOR-PIN.md`. All SEs: stratified bootstrap over
dialogues, 1000 resamples, seed 42.

## Headline

1. **Cross-dataset transfer collapses at small scale.** Probes refit on one
   deception dataset and evaluated on another — same model, same layer
   fraction — average AUROC **0.468** (30 cells, all 5 models) against
   same-dataset ceilings of **0.88–0.91**. The paper's 70B probes generalize
   across its eval sets; at 1B–9B the linear "deception" direction is
   dataset-local.
2. **One inversion is systematic, not noise: roleplaying→sandbagging AUROC
   0.157 / 0.186 / 0.205 / 0.221 / 0.273** (8B / gemma-2b / 1B / 3B /
   gemma-9b; SEs 0.012–0.016). A monitor fit on roleplaying doesn't merely
   miss sandbagging at these scales — it ranks sandbagging deception as
   *extra honest*. Sign-flipped it would score ~0.8.
3. **The released artifact is architecture-bound — stated as a limitation,
   not a discovery.** The shipped probes live in the 70B's 8192-dim residual
   basis and no smaller model shares that width, so there is no
   principled refit-free port at all; we verify that the one naive option,
   coordinate restriction, scores at chance (per-probe means 0.453–0.489 over
   60 cells; the 7 cells above 0.6, max 0.69, are what 60 comparisons produce
   under the null). A reader may reasonably call this obvious a priori — an
   8192-dim vector cannot dot a 2048-dim residual stream — so we claim only
   that we quantified the failure rather than that we discovered it. The
   consequence still matters for deployment: "apply the released probe to
   your model" is not an available option, and any user of these probes must
   refit, which lands them in findings 1 and 2.
4. **Cross-model refit transfer is also chance** under the same coordinate
   restriction: diag mean 0.842, off-diag mean 0.487, range [0.406, 0.589]
   (roleplaying, 5×5).

## Replication gate (Task 2) — documented miss, instrument validated

Protocol: 5-fold CV at dialogue level, per-token probe fit (Apollo's
protocol), mean prompt score, one AUROC over all out-of-fold dialogues
(n=287, 84 honest / 203 deceptive). An earlier single 80/20 split was
seed-dominated (0.88–0.97 across seeds; val honest arm n≈17) and was
replaced by CV *for power, before any transfer numbers were computed*.

- Llama-3.1-8B roleplaying: **0.9365** vs the 70B-calibrated bar 0.94
  (published range 0.96–0.999 − 0.02 tolerance). Miss ≈ 0.4 pp.
- Same-distribution CV AUROC by model: 1B **0.863**, gemma-2b **0.871**,
  gemma-9b **0.876**, 3B **0.908**, 8B **0.936** — a clean capability-ordered
  trend toward the 70B's published range. A broken extractor sits at ~0.5;
  this is a size effect, consistent with the probe size-sweep literature
  (2604.13386).
- Disposition (project decision): proceed with the miss documented; each
  model's own same-distribution CV AUROC is the reference ceiling for its
  transfer numbers. Runs with 8B below a 0.90 sanity floor still hard-stop.

## Instrument definitions (uniform across models; deviations from vendor)

- Data: Apollo's shipped, graded 70B rollouts, teacher-forced through each
  small model (text constant; representations vary). insider_trading uses the
  on-policy llama-70b file — the repo's "upscale" rollout file is a stub.
- Labels mirror the vendor rules (roleplaying grade ≥6 deceptive / ≤2 honest,
  else excluded; string labels for the rest; AMBIGUOUS/SKIP never enter
  metrics). Counts: roleplaying 84h/203d, insider 80h/91d, sandbagging
  512h/420d.
- Activations: `hidden_states[int(0.275 × num_layers)]` (22-of-80
  convention), mean of assistant-response tokens after the generation-prompt
  prefix. For linear probes with the mean prompt-scorer this pooling is
  exact. Vendor detection masks are per-dataset and more granular; ours is
  uniform by design.
- Released-probe scoring uses the shipped scaler:
  `((x − scaler_mean)/scaler_scale) · w` (the scaler ships inside
  detector.pt; conversions in `results/probes/*_manifest.json`).
- Gemma templates reject system messages; system content is folded into the
  first user turn (all models see identical dialogue text).
- Models are ungated mirrors (unsloth ×4, NousResearch 8B) per the project's
  gate-era mirror-with-provenance pattern; sae_rp probe excluded by
  construction (SAE-latent space).

## Cross-dataset matrix (Task 4) — all off-diagonal cells

| model | rp→it | rp→sb | it→rp | it→sb | sb→rp | sb→it | diag mean |
|---|---|---|---|---|---|---|---|
| Llama-3.2-1B | 0.463 | **0.205** | 0.463 | 0.565 | 0.409 | 0.565 | 0.883 |
| gemma-2-2b | 0.433 | **0.186** | 0.508 | 0.579 | 0.415 | 0.562 | 0.879 |
| Llama-3.2-3B | 0.509 | **0.221** | 0.530 | 0.703 | 0.377 | 0.614 | 0.907 |
| Llama-3.1-8B | 0.539 | **0.157** | 0.594 | 0.540 | 0.340 | 0.522 | 0.898 |
| gemma-2-9b | 0.501 | **0.273** | 0.525 | 0.708 | 0.411 | 0.636 | 0.893 |

(rp=roleplaying, it=insider_trading, sb=sandbagging; off-diag SEs 0.012–0.046;
diagonals are 80/20 held-out.)

Interpretation caution for the inversion: sandbagging responses are
multiple-choice answers while roleplaying is persona dialogue; the uniform
response-span instrument cannot rule out that the probe partially tracks
content register rather than deception per se. Supporting that concern, the
sandbagging *diagonal* is ~0.99–1.00 on every model — within-dataset
separability that clean at 1B strongly suggests shortcut features — while its
transfer to the other datasets is chance. The claim that survives any
reading: a roleplaying-fit linear monitor produces systematically *reversed*
rankings on sandbagging at 1B–8B, so deploying it there would be worse than
useless. The mechanism question is follow-up work.

## Completeness

All 15 (model, dataset) cells extracted and analyzed. gemma-2-9b's first
extraction OOM'd on Gemma's 256k-vocab logits (job 21461609); the headless
base-model re-extraction (21461703) completed and its cells are in every
table above.

## Figures

`heatmap_released.png`, `heatmap_cross_dataset.png`, `heatmap_cross_model.png`.

## Prior art and honest positioning

Checked three ways: arXiv sweep, the full 64-item Semantic Scholar forward
citation graph of 2502.03407, and a separate LessWrong/AF sweep (neither
arXiv nor S2 indexes LW, and this community publishes there). Verdict per
claim, with the citations that must appear in the post:

**Finding 1 is precedented in kind, not in scope.** Kirch et al.
(arXiv:2511.17408, "The Impact of Off-Policy Training Data on Probe
Generalisation") builds its deception behaviour from the *same* Apollo
roleplaying and insider-trading files, tests primarily on Llama-3.2-3B (one
of our five models), and reports that deception suffers the largest
domain-shift penalty of the eight behaviours it studies (−0.416 AUROC),
concluding "current deception probes may fail to generalise to real
monitoring scenarios." What we add: a full 3×3 transfer matrix including
Apollo's actual sandbagging dataset (which they do not use — they build a
separate sandbagging behaviour from WMDP-MMLU), five models spanning 1B–9B
across two families, and per-cell AUROCs with bootstrap intervals.
Supporting: Mahadik & Skapars (arXiv:2605.09391, cross-dataset deception
transfer on Llama-3.2-3B / Llama-3-8B); Nordby, Pais & Parrack
(arXiv:2604.13386, probes "fail entirely" on Insider Trading at a fixed
layer); Ying et al. (arXiv:2602.20273, geometric account of why cross-domain
probe transfer fails).

**Finding 2's phenomenon type is precedented; this specific pair is not, and
notably could not be measured before.** Kumar (arXiv:2605.27958,
"Pressure-Testing Deception Probes") independently reports below-chance,
anti-correlated cross-dataset transfer on Gemma 3 at magnitudes close to ours
(0.147–0.298), and argues the cause is geometric rather than layer-mismatch —
a mechanism our post should engage with rather than ignore. Crucially, that
paper *attempted and failed* to construct a working sandbagging dataset for
this analysis ("effectively not generated… 0–1 examples surviving per model
size… All D-Sand entries in cross-domain analyses are consequently null"), so
the roleplaying→sandbagging direction has not previously been measured by
anyone, including the work most directly aimed at it. Our contribution is
that pair specifically, on Apollo's canonical datasets, replicated on five
models.

**Finding 3 is unclaimed** — every related paper refits per model rather than
porting Apollo's released weights, and neither Apollo's paper nor blog
addresses portability. See the framing caveat in the headline above; claim
verification, not discovery.

Checked and cleared (no collision, post bodies and comment threads read):
"Trusted monitoring, but with deception probes" (Parrack & Heimersheim) and
the LW crosspost of the Apollo paper itself — no commenter raises
cross-dataset transfer, inversion, or dimension mismatch.

Full record: `../PRIOR-ART-2026-08-15.md` and `docs/post-conventions.md` §6.
