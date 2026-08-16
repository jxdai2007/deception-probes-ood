# Deception probes off-distribution

Do released linear "deception probes" still work when applied off the
distribution they were built on? This repo takes Apollo Research's released
probes ([Detecting Strategic Deception Using Linear Probes](https://arxiv.org/abs/2502.03407),
fit on Llama-3.3-70B) and audits how they behave on smaller open models and on
deception datasets they were not trained on.

Replication-and-extend project. LessWrong post: _(link to be added)_

## Motivation

> _(Framing paragraph — author to finalize in their own voice.)_

A deception monitor is only useful if it fires on the deception you actually
care about, not just the kind it was trained on. Apollo's probes reproduce
strong same-distribution AUROCs on a 70B model. The safety-relevant question is
what happens when a practitioner reuses them off-distribution — a smaller model,
a different kind of lie. This project measures that.

## Setup

- **Anchor:** Goldowsky-Dill et al., [arXiv:2502.03407](https://arxiv.org/abs/2502.03407);
  released code and probe weights at
  [ApolloResearch/deception-detection](https://github.com/ApolloResearch/deception-detection)
  (pinned commit `f8ec401`).
- **Data:** Apollo's own shipped, graded 70B rollouts for three deception
  datasets (roleplaying, insider trading, sandbagging), teacher-forced through
  each small model so the text is held constant and only representations vary.
- **Probes:** per-token logistic-regression probes at the paper's layer
  convention (fraction 0.275 of depth = layer 22 of 80 on the 70B, 8192-dim),
  applied with the released scaler.
- **Models:** Llama-3.2-1B/3B, Llama-3.1-8B, Gemma-2-2B/9B (ungated mirrors;
  provenance recorded in the results doc).
- **Metric:** AUROC (deceptive vs honest), stratified bootstrap 95% CIs.

## Replication gate

Apollo's nine published AUROCs re-derive exactly from their shipped per-token
scores. A same-distribution refit on Llama-3.1-8B reaches CV-AUROC 0.9365
against a pre-registered 0.94 bar — a documented miss, with a clean
capability-ordered size trend (1B 0.863 → 8B 0.936) validating the instrument.

## Findings

| # | finding |
|---|---------|
| 1 | **Cross-dataset transfer collapses at small scale** — probes fit on one dataset average AUROC 0.468 across the others (30 cells) vs 0.88–0.91 same-dataset ceilings. |
| 2 | **One direction is systematically inverted** — roleplaying→sandbagging scores AUROC 0.157–0.273 on all five models; the probe ranks sandbagging deception as *more honest* than honest responses. |
| 3 | **The released 70B artifact is architecture-bound** — no smaller model shares its 8192-dim residual basis; naive coordinate restriction scores at chance across 60 cells. |

Full numbers, figures, and prior-art positioning:
[`results/2026-08-16-extension.md`](results/2026-08-16-extension.md).

## Reproduce

```bash
pip install -e common            # shared harness (jrp_common)
export PYTHONPATH=.:common
# 1. convert released probes to numpy (one audited pickle load):
python -m p04.convert_detector --in <detector.pt> --out-dir results/probes
# 2. extract activations (per model) on the shipped rollouts:
python -m p04.extract_activations --model <hf-id> --out-dir <acts-dir>
# 3. full analysis: gate -> transfer legs -> heatmaps:
python -m p04.run_extension --acts-dir <acts-dir>
python -m p04.figures_post   # publication figures
pytest tests/                # 12 synthetic-fixture tests
```

The released `detector.pt` files are plain pickles; `p04/convert_detector.py`
loads them once through a restricted unpickler and ships only numpy artifacts.

## Layout

- `p04/` — loaders, probe conversion, transfer analyses, figures
- `results/` — results doc, per-cell JSONs, figures, converted probe manifests
- `common/jrp_common/` — shared probe/metrics harness
- `tests/` — synthetic-fixture unit tests

## Citation

If you use this, please also cite the anchor paper (arXiv:2502.03407). Prior
work this builds on is credited in the results doc (Kirch et al. 2511.17408,
Kumar 2605.27958, and others).

## License

MIT (this repository's code). The vendored Apollo repo, released under its own
license, is not redistributed here — clone it at the pinned commit to reproduce.
