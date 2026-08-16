import datetime
import json
import os
import subprocess

try:
    from lm_eval import simple_evaluate
except ImportError:
    # lm-eval is an optional/heavy dependency; keep jrp_common importable without it.
    # run_capability raises a clear error below if this stays unset at call time.
    simple_evaluate = None


# Honest placeholder for `revision` when the concrete snapshot SHA could not be
# determined (offline, gated repo, or a local checkpoint path). Never None and
# never a value that could be mistaken for a real SHA.
UNRESOLVED_REVISION_MARKER = "unresolved: revision lookup failed (offline, gated repo, or local path)"

# Seconds to wait on the HF Hub lookup before giving up. huggingface_hub 0.36.2's
# `model_info(timeout=...)` defaults to None (no bound), which on a compute node
# whose firewall silently drops outbound packets can stall for the OS TCP timeout
# (a minute or two) before failing. HF_HUB_OFFLINE=1 short-circuits the call entirely.
_REVISION_LOOKUP_TIMEOUT_SECONDS = 5


def run_capability(model_id, tasks, limit=None, model_args="", revision=None, seed=0, num_fewshot=None):
    if simple_evaluate is None:
        raise RuntimeError(
            "lm-eval is not installed. Install it with `pip install lm-eval` (it is a "
            "declared dependency in common/pyproject.toml, so `pip install -e common` "
            "also pulls it in) before calling run_capability."
        )
    if revision is None:
        # Resolve once, here, at evaluation start -- not hours later when the run is
        # recorded. A revision resolved at record time can name a snapshot that isn't
        # the weights actually evaluated, since `main` can move during a long run.
        revision = resolve_revision(model_id)
    if revision == UNRESOLVED_REVISION_MARKER:
        # Single normalization point: whether the marker arrived because resolution
        # just failed above, or because a caller threaded it in from a resolve_revision
        # call that failed earlier, it means the same thing -- no revision -- and must
        # not be pinned into model_args. An eval must never die because provenance
        # lookup failed.
        revision = None
    args = f"pretrained={model_id}"
    if revision:
        # Pins the exact model snapshot evaluated, rather than whatever `main` resolves to.
        args += f",revision={revision}"
    if model_args:
        args += f",{model_args}"
    res = simple_evaluate(
        model="hf",
        model_args=args,
        tasks=tasks,
        limit=limit,
        num_fewshot=num_fewshot,
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
    )
    out = {}
    for t in tasks:
        r = res["results"][t]
        if "acc,none" in r:
            out[t] = float(r["acc,none"])
        elif "acc" in r:
            out[t] = float(r["acc"])
        else:
            # Silently falling back to NaN would let json.dump write a bare `NaN`
            # token -- not valid strict JSON, and it breaks downstream parsers.
            # A task that only reports e.g. acc_norm,none must fail loudly instead
            # of recording an unusable number.
            raise ValueError(
                f"task {t!r} has neither 'acc,none' nor 'acc' in its results; "
                f"available result keys: {sorted(r.keys())}"
            )
    return out


def _git_sha():
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                # Pin to the harness's own directory, not the process cwd -- a job
                # launched from elsewhere (e.g. a SLURM submission dir) would
                # otherwise record "unknown" or, worse, a different repo's SHA.
                # Correct under the documented editable install (`pip install -e common`).
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def resolve_revision(model_id):
    """
    Best-effort resolution of the concrete HF Hub snapshot SHA for model_id. Called by
    run_capability at evaluation start (so the eval itself can be pinned to the
    resolved snapshot) and by record_run as a fallback when a caller does not pass an
    explicit revision, so a run's payload always names a concrete snapshot instead of
    silently describing whatever `main` happened to resolve to that day.

    Never raises: offline machines, gated repos, and local checkpoint paths all fail
    the lookup, and an eval must not die because provenance lookup failed. On any
    failure this returns an explicit honest marker string rather than None or a
    guessed value.

    Public API: this is the function callers should use to thread one resolved
    revision through both calls below, rather than letting each call resolve
    independently.

        rev = resolve_revision(model_id)
        res = run_capability(model_id, tasks, revision=rev)
        record_run(out_dir, model_id, res, revision=rev, ...)

    Threading one resolved value makes the recorded `revision` provably the snapshot
    that was evaluated: `revision_source` on the recorded payload comes out
    "caller-provided" instead of "resolved-at-record-time". Leaving revision=None on
    both calls instead resolves twice -- once inside run_capability at T0, again
    inside record_run at T1 -- and if upstream moved between T0 and T1, the recorded
    SHA is not the SHA that was evaluated.

    Threading is safe even when resolution above fails: if `rev` is
    UNRESOLVED_REVISION_MARKER, run_capability proceeds unpinned and record_run labels
    the run "unresolved" -- the caller does not need to check `rev` before passing it
    on to either call.
    """
    try:
        from huggingface_hub import model_info

        return model_info(model_id, timeout=_REVISION_LOOKUP_TIMEOUT_SECONDS).sha
    except Exception:
        return UNRESOLVED_REVISION_MARKER


def record_run(
    out_dir,
    model_id,
    results,
    timestamp=None,
    seed=0,
    revision=None,
    num_fewshot=None,
    tag=None,
    limit=None,
    model_args="",
    tasks=None,
):
    os.makedirs(out_dir, exist_ok=True)
    if revision is not None:
        # The caller handed us a concrete value directly -- either their own explicit
        # pin, or a SHA they resolved and pinned themselves at evaluation time (e.g.
        # via run_capability's own resolve-and-pin). Either way it did not come from a
        # resolution performed here, at write time, so it is labeled accordingly --
        # unless the value is the marker itself, meaning the caller's own resolution
        # attempt failed and nothing was actually pinned.
        resolved_revision = revision
        revision_source = "unresolved" if revision == UNRESOLVED_REVISION_MARKER else "caller-provided"
    else:
        # Not pinned by the caller -- fall back to resolving here. This happens at
        # record time (T1), which can be hours after the eval ran (T0), so the value
        # is labeled "resolved-at-record-time" rather than conflated with a pin taken
        # at evaluation time. See run_capability for the evaluation-time resolution.
        resolved_revision = resolve_revision(model_id)
        revision_source = (
            "unresolved" if resolved_revision == UNRESOLVED_REVISION_MARKER else "resolved-at-record-time"
        )
    payload = {
        "model_id": model_id,
        "results": results,
        "git_sha": _git_sha(),
        "timestamp": timestamp or datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seed": seed,
        "revision": resolved_revision,
        # Makes the origin of `revision` explicit so a reader never has to guess
        # whether it is the weights actually evaluated: "caller-provided" (explicit
        # pin or eval-time resolution the caller passed through), "resolved-at-record-time"
        # (a fallback lookup done here, which can be stale relative to the eval), or
        # "unresolved" (lookup failed; `revision` is the honest marker, not a SHA).
        "revision_source": revision_source,
        "num_fewshot": num_fewshot,
        # Without these, a limit=100 smoke-test number and a full-dataset accuracy
        # are indistinguishable in the recorded JSON except by the optional free-text
        # tag below.
        "limit": limit,
        "model_args": model_args,
        # Defaults to the tasks actually present in results (in the order
        # run_capability produced them) when the caller doesn't pass tasks explicitly.
        "tasks": tasks if tasks is not None else list(results.keys()),
    }
    name = f"run_{model_id.replace('/', '_')}"
    if tag:
        # Discriminates multiple settings (quantization/pruning) evaluated on the same
        # model_id, which would otherwise all write to the same path and clobber.
        name += f"_{tag}"
    path = os.path.join(out_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path
