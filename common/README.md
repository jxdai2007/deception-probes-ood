# common/ — shared infrastructure

Built once, reused across projects. Specced here; implemented when the first project's replication
gate starts.

## env/
- `environment.yml` / `requirements.txt` — one env for all four projects: transformers,
  lm-evaluation-harness, concept-erasure (LEACE), scikit-learn (probes), the anchor repos pinned
  or vendored as submodules.
- SLURM job templates (single-GPU eval, activation-dump). Partition/GPU/account filled at socket-open.

## eval/
- WMDP + MMLU harness wrappers (used by projects 02, and as controls elsewhere).
- Activation-extraction + linear-probe utilities (used by projects 01, 04, and 03's interp hook):
  contrastive activation collection, layer sweep, logistic-regression probe fit/eval, AUROC +
  catch-rate-at-1%-FPR reporting.

Keeping these here stops us reimplementing the probe + WMDP harness four times, and makes numbers
comparable across projects.
