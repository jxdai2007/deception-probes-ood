#!/usr/bin/env python3
"""Extract per-token response activations for one model over the transfer datasets.

Usage:
    python -m p04.extract_activations --model google/gemma-2-2b-it \
        --out-dir <activation-output-directory> [--layer-frac 0.275] \
        [--datasets roleplaying insider_trading sandbagging] [--limit N]

Per (model, dataset) writes into out-dir:
    <model_short>__<dataset>__acts.npy    float16 [sum_response_tokens, hidden]
    <model_short>__<dataset>__lengths.npy int32   [n_dialogues]
    <model_short>__<dataset>__labels.npy  int8    [n_dialogues]  (0/1/-1)
    <model_short>__<dataset>__meta.json

Instrument definitions (identical across models, so cross-model comparisons
are internally consistent):
- Activations are hidden_states[L] with L = layer_frac_to_index(num_layers,
  layer_frac) — the paper's 22-of-80 convention at frac 0.275.
- The scored span is every token of the assistant response (tokens after the
  add_generation_prompt prefix), teacher-forced from the 70B rollout text.
- Dialogues over --max-tokens total length are truncated to the first
  max-tokens tokens (vendor default 2048); truncation count goes in meta.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess

import numpy as np
import torch

from p04.condition_io import condition_stem
from p04.loader import (
    DATASET_FILES,
    ROLLOUTS_DIR,
    load_rollouts,
    layer_frac_to_index,
)
from p04.pooling_confirmation import canonical_protocol_sha256
from p04.rendering import (
    FROZEN_PARSE_EXCLUSION_SHA256,
    RENDERINGS,
    immutable_digest,
    parse_selected_answer,
    render_response,
)


_EXTRACTION_SOURCE_PATHS = (
    "projects/04-deception-probes/p04/condition_io.py",
    "projects/04-deception-probes/p04/extract_activations.py",
    "projects/04-deception-probes/p04/loader.py",
    "projects/04-deception-probes/p04/rendering.py",
    "projects/04-deception-probes/slurm/extract_format_geometry.sbatch",
    "projects/04-deception-probes/slurm/run_format_geometry.sh",
)
_REQUIREMENTS_LOCK_PATH = "common/env/requirements.lock.txt"
_PINNED_PYTHON_MAJOR_MINOR = "3.11"
_RUNTIME_PACKAGES = (
    "numpy",
    "scikit-learn",
    "torch",
    "transformers",
)


def _sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pinned_runtime_requirements(*, repo_root=None):
    """Read the analysis/extraction runtime contract from the checked-in lock."""
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[3]
    )
    locked = {}
    for line in (root / _REQUIREMENTS_LOCK_PATH).read_text().splitlines():
        if "==" not in line or line.lstrip().startswith("#"):
            continue
        name, version = line.split("==", 1)
        locked[name.lower()] = version
    missing = sorted(set(_RUNTIME_PACKAGES) - set(locked))
    if missing:
        raise ValueError(
            f"requirements lock is missing runtime packages: {missing}"
        )
    return {
        "python": _PINNED_PYTHON_MAJOR_MINOR,
        **{name: locked[name] for name in _RUNTIME_PACKAGES},
    }


def validate_runtime_versions(
    runtime_versions,
    *,
    required=None,
    artifact_name="runtime",
):
    """Fail unless installed versions satisfy the pinned Project 04 runtime."""
    required = pinned_runtime_requirements() if required is None else required
    if not isinstance(runtime_versions, dict):
        raise ValueError(f"{artifact_name} does not match pinned runtime")
    if set(runtime_versions) != set(required) or any(
        not isinstance(value, str) or not value
        for value in runtime_versions.values()
    ):
        raise ValueError(f"{artifact_name} does not match pinned runtime")
    python_parts = runtime_versions["python"].split(".")
    python_major_minor = ".".join(python_parts[:2])
    mismatches = {}
    if python_major_minor != required["python"]:
        mismatches["python"] = {
            "observed": runtime_versions["python"],
            "required_major_minor": required["python"],
        }
    for name in _RUNTIME_PACKAGES:
        if runtime_versions[name] != required[name]:
            mismatches[name] = {
                "observed": runtime_versions[name],
                "required": required[name],
            }
    if mismatches:
        raise ValueError(
            f"{artifact_name} does not match pinned runtime: {mismatches}"
        )
    return {
        "python": python_major_minor,
        **{name: runtime_versions[name] for name in _RUNTIME_PACKAGES},
    }


def extraction_provenance(*, repo_root=None, runtime_versions=None):
    """Bind extraction artifacts to reviewed code, lockfile, and runtime."""
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[3]
    )
    git_commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if runtime_versions is None:
        runtime_versions = {
            "python": platform.python_version(),
            **{
                name: importlib.metadata.version(name)
                for name in _RUNTIME_PACKAGES
            },
        }
    return {
        "git_commit": git_commit,
        "source_sha256": {
            relative: _sha256_path(root / relative)
            for relative in _EXTRACTION_SOURCE_PATHS
        },
        "requirements_lock_sha256": _sha256_path(
            root / _REQUIREMENTS_LOCK_PATH
        ),
        "runtime_versions": dict(runtime_versions),
    }


def model_short(model_id: str) -> str:
    return model_id.rstrip("/").split("/")[-1]


def load_experiment_protocol(path, required_sha256, expected_model_short):
    """Load and canonically bind one confirmation protocol."""
    try:
        protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid experiment protocol: {path}") from error
    if not isinstance(protocol, dict):
        raise ValueError("experiment protocol must be a JSON object")
    digest = canonical_protocol_sha256(protocol)
    if digest != required_sha256:
        raise ValueError(
            f"protocol digest mismatch: observed {digest}, "
            f"required {required_sha256}"
        )
    model = protocol.get("model")
    if (
        protocol.get("schema_version") != 1
        or protocol.get("experiment")
        != "p04_pooling_mechanism_confirmation"
        or not isinstance(model, dict)
        or model.get("model_short") != expected_model_short
    ):
        raise ValueError("protocol model mismatch")
    identity = {
        "schema_version": 1,
        "selected_k": protocol.get("selected_k"),
        "model_short": model["model_short"],
        "resolved_model_revision": model.get("resolved_model_revision"),
        "resolved_tokenizer_revision": model.get(
            "resolved_tokenizer_revision"
        ),
    }
    return {"record": protocol, "sha256": digest, "identity": identity}


def bind_experiment_protocol_metadata(metadata, binding):
    """Add protocol fields, leaving legacy Phase A metadata untouched."""
    if binding is None:
        return metadata
    metadata["experiment_protocol_sha256"] = binding["sha256"]
    metadata["experiment_protocol_identity"] = dict(binding["identity"])
    return metadata


def _validate_experiment_protocol_environment(binding, args, producer):
    if binding is None:
        return
    from p04.pooling_confirmation import _validate_protocol_record

    protocol = binding["record"]
    _validate_protocol_record(protocol)
    model = protocol["model"]
    if args.model != model["model_id"]:
        raise ValueError("protocol model ID does not match --model")
    if (
        args.model_revision != model["resolved_model_revision"]
        or args.tokenizer_revision != model["resolved_tokenizer_revision"]
    ):
        raise ValueError(
            "protocol-bound extraction requires the frozen model and "
            "tokenizer revisions"
        )
    instrument = protocol["instrument"]
    if (
        args.layer_frac != instrument["layer_frac"]
        or args.max_tokens != instrument["max_tokens"]
    ):
        raise ValueError("extraction instrument does not match protocol")
    if not set(args.datasets).issubset(protocol["dataset_sha256"]):
        raise ValueError("extraction dataset is outside the protocol")
    if producer["source_sha256"] != protocol["source_sha256"]:
        raise ValueError("extraction source identity does not match protocol")
    if (
        producer["requirements_lock_sha256"]
        != protocol["requirements_lock_sha256"]
    ):
        raise ValueError("requirements lock does not match protocol")
    observed_dataset_sha256 = {
        dataset: _sha256_path(ROLLOUTS_DIR / DATASET_FILES[dataset])
        for dataset in protocol["dataset_sha256"]
    }
    if observed_dataset_sha256 != protocol["dataset_sha256"]:
        raise ValueError("rollout dataset identity does not match protocol")


def resolve_hub_revision(
    model_id,
    filenames,
    requested_revision=None,
    *,
    cached_file_fn=None,
):
    """Resolve a Hub revision from a concrete cached/downloaded file path."""
    if isinstance(filenames, str):
        filenames = (filenames,)
    if cached_file_fn is None:
        from transformers.utils.hub import cached_file

        cached_file_fn = cached_file
    from transformers.utils.hub import extract_commit_hash

    for filename in filenames:
        resolved_file = cached_file_fn(
            model_id,
            filename,
            revision=requested_revision,
            _raise_exceptions_for_missing_entries=False,
        )
        if resolved_file is None:
            continue
        commit_hash = extract_commit_hash(resolved_file, None)
        if commit_hash:
            return commit_hash
    raise RuntimeError(
        f"resolved Hub revision is unavailable for {model_id!r} "
        f"from files {tuple(filenames)!r}"
    )


def resolve_extraction_revisions(
    model_id,
    requested_model_revision,
    requested_tokenizer_revision,
    *,
    rendering,
):
    """Resolve immutable Hub SHAs while retaining legacy local-path support."""
    if Path(model_id).expanduser().is_dir():
        if rendering is not None:
            raise RuntimeError(
                "condition extraction requires a Hub model with resolved revision "
                "provenance; local model directories remain supported only by the "
                "legacy extraction path"
            )
        return requested_model_revision, requested_tokenizer_revision

    resolved_model_revision = resolve_hub_revision(
        model_id, "config.json", requested_model_revision
    )
    tokenizer_requested_revision = (
        requested_tokenizer_revision
        if requested_tokenizer_revision is not None
        else resolved_model_revision
    )
    resolved_tokenizer_revision = resolve_hub_revision(
        model_id,
        (
            "config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "tokenizer.model",
            "spiece.model",
            "vocab.json",
            "merges.txt",
        ),
        tokenizer_requested_revision,
    )
    return resolved_model_revision, resolved_tokenizer_revision


def _template_supports_system(tokenizer) -> bool:
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
            add_generation_prompt=True, tokenize=True,
        )
        return True
    except Exception:
        return False


def _fold_system_into_user(msgs):
    """Gemma-style templates reject system role; prepend system content to the
    first user turn instead (applied identically to every dialogue)."""
    sys_parts = [m["content"] for m in msgs if m["role"] == "system"]
    rest = [dict(m) for m in msgs if m["role"] != "system"]
    if sys_parts:
        prefix = "\n\n".join(sys_parts)
        for m in rest:
            if m["role"] == "user":
                m["content"] = f"{prefix}\n\n{m['content']}"
                break
        else:
            rest.insert(0, {"role": "user", "content": prefix})
    return rest


def token_span_from_offsets(offsets, char_span):
    """Map a half-open character span to overlapping response-token indices."""
    char_start, char_end = char_span
    if char_start < 0 or char_end <= char_start:
        raise ValueError(f"invalid character span {char_span}")
    hits = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > char_start and start < char_end
    ]
    if not hits:
        raise ValueError(f"character span {char_span} does not overlap a token")
    return hits[0], hits[-1] + 1


def prepare_rendered_rows(
    rows,
    rendering,
    *,
    parse_exclusion_sha256=None,
    require_all_parse_exclusions=False,
):
    """Validate all renderings, then select one deterministic view per row."""
    if rendering not in RENDERINGS:
        raise ValueError(f"unknown rendering {rendering!r}; expected {RENDERINGS}")
    example_ids = [row.get("example_id") for row in rows]
    if any(not isinstance(value, str) or not value for value in example_ids):
        raise ValueError("condition rows require nonempty example IDs")
    if len(set(example_ids)) != len(example_ids):
        raise ValueError("duplicate example IDs in condition rows")

    registry = (
        FROZEN_PARSE_EXCLUSION_SHA256
        if parse_exclusion_sha256 is None
        else dict(parse_exclusion_sha256)
    )
    prepared = []
    exclusions = []
    seen_exclusions = set()
    for row in rows:
        example_id = row["example_id"]
        if example_id in registry:
            output_sha256 = hashlib.sha256(row["output_str"].encode()).hexdigest()
            if output_sha256 != registry[example_id]:
                raise ValueError(f"{example_id}: parse-exclusion hash mismatch")
            try:
                parse_selected_answer(row["output_str"])
            except ValueError:
                exclusions.append(
                    {
                        "example_id": example_id,
                        "output_sha256": output_sha256,
                        "reason": "invalid_selected_answer_tag",
                    }
                )
                seen_exclusions.add(example_id)
                continue
            raise ValueError(
                f"{example_id}: frozen parse exclusion became parseable"
            )
        try:
            views = {
                name: render_response(row["output_str"], name)
                for name in RENDERINGS
            }
        except ValueError as exc:
            raise ValueError(f"{example_id}: {exc}") from exc
        selected_answers = {view.selected_answer for view in views.values()}
        if len(selected_answers) != 1:
            raise ValueError(f"{example_id}: selected-answer mismatch across renderings")
        digests = {
            immutable_digest(row, view.selected_answer) for view in views.values()
        }
        if len(digests) != 1:
            raise ValueError(f"{example_id}: immutable hash mismatch across renderings")
        selected = views[rendering]
        prepared.append(
            {
                **row,
                "output_str": selected.text,
                "answer_char_span": selected.answer_char_span,
                "selected_answer": selected.selected_answer,
                "immutable_hash": next(iter(digests)),
            }
        )
    if require_all_parse_exclusions:
        missing = sorted(set(registry) - seen_exclusions)
        if missing:
            raise ValueError(f"missing frozen parse exclusions: {missing}")
    return prepared, exclusions


def response_token_spans(tokenizer, rows, max_tokens: int):
    """Tokenize each dialogue; return (input_ids, prompt_len, total_len) per row.

    prompt_len counts the tokens of the chat-templated prompt with the
    generation prefix; everything after it is the teacher-forced response.
    """
    out = []
    fold_system = not _template_supports_system(tokenizer)
    for row in rows:
        msgs = [
            {"role": m["role"], "content": m["content"]}
            for m in row["input_messages"]
        ]
        # Drop the trailing empty assistant stub if the rollout carries one
        if msgs and msgs[-1]["role"] == "assistant" and not msgs[-1]["content"]:
            msgs = msgs[:-1]
        if fold_system:
            msgs = _fold_system_into_user(msgs)
        prompt_ids = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True
        )
        answer_span = None
        if "answer_char_span" in row:
            if not getattr(tokenizer, "is_fast", False):
                raise ValueError("condition extraction requires a fast tokenizer")
            encoded = tokenizer(
                row["output_str"],
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            response_ids = encoded["input_ids"]
            answer_start, answer_end = token_span_from_offsets(
                encoded["offset_mapping"], row["answer_char_span"]
            )
        else:
            response_ids = tokenizer.encode(
                row["output_str"], add_special_tokens=False
            )
        full = prompt_ids + response_ids
        truncated = len(full) > max_tokens
        kept_response = max(
            0, min(len(response_ids), max_tokens - len(prompt_ids))
        )
        if answer_span is None and "answer_char_span" in row:
            answer_span = (
                (answer_start, answer_end)
                if answer_end <= kept_response
                else (-1, -1)
            )
        full = full[:max_tokens]
        out.append(
            {
                "ids": full,
                "prompt_len": min(len(prompt_ids), len(full)),
                "truncated": truncated,
                "answer_span": answer_span,
            }
        )
    return out


@torch.no_grad()
def extract_dataset(model, tokenizer, rows, layer_index, device, batch_size, max_tokens):
    """Return activations, lengths, truncation count, and tokenized row data."""
    tokenized = response_token_spans(tokenizer, rows, max_tokens)
    acts_chunks, lengths = [], []
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    for start in range(0, len(tokenized), batch_size):
        batch = tokenized[start : start + batch_size]
        maxlen = max(len(b["ids"]) for b in batch)
        input_ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
        for j, b in enumerate(batch):
            input_ids[j, : len(b["ids"])] = torch.tensor(b["ids"])
            attn[j, : len(b["ids"])] = 1
        out = model(
            input_ids=input_ids.to(device),
            attention_mask=attn.to(device),
            output_hidden_states=True,
        )
        hs = out.hidden_states[layer_index]  # [batch, seq, hidden]
        for j, b in enumerate(batch):
            span = hs[j, b["prompt_len"] : len(b["ids"])]
            if span.shape[0] == 0:
                # Response fully truncated away; keep a zero-length entry
                lengths.append(0)
                continue
            acts_chunks.append(span.to(torch.float16).cpu())
            lengths.append(span.shape[0])

    acts = (
        torch.cat(acts_chunks).numpy()
        if acts_chunks
        else np.zeros((0, model.config.hidden_size), dtype=np.float16)
    )
    n_truncated = sum(1 for t in tokenized if t["truncated"])
    return acts, np.array(lengths, dtype=np.int32), n_truncated, tokenized


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--layer-frac", type=float, default=0.275)
    parser.add_argument(
        "--datasets", nargs="+", default=list(DATASET_FILES)
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--rendering", choices=RENDERINGS)
    parser.add_argument("--model-revision")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--experiment-protocol")
    parser.add_argument("--require-experiment-protocol-sha256")
    args = parser.parse_args(argv)

    if (args.experiment_protocol is None) != (
        args.require_experiment_protocol_sha256 is None
    ):
        parser.error(
            "experiment protocol path and SHA-256 must be supplied together"
        )
    if args.rendering is not None and args.datasets != ["sandbagging"]:
        parser.error("--rendering requires --datasets sandbagging")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    producer = extraction_provenance()
    validate_runtime_versions(
        producer["runtime_versions"],
        artifact_name="extraction runtime",
    )
    experiment_protocol = None
    if args.experiment_protocol is not None:
        experiment_protocol = load_experiment_protocol(
            args.experiment_protocol,
            args.require_experiment_protocol_sha256,
            model_short(args.model),
        )
    _validate_experiment_protocol_environment(
        experiment_protocol, args, producer
    )

    from transformers import AutoModel, AutoTokenizer

    resolved_model_revision, resolved_tokenizer_revision = (
        resolve_extraction_revisions(
            args.model,
            args.model_revision,
            args.tokenizer_revision,
            rendering=args.rendering,
        )
    )
    if experiment_protocol is not None and (
        resolved_model_revision
        != experiment_protocol["identity"]["resolved_model_revision"]
        or resolved_tokenizer_revision
        != experiment_protocol["identity"]["resolved_tokenizer_revision"]
    ):
        raise ValueError("resolved model/tokenizer revisions do not match protocol")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=resolved_tokenizer_revision
    )
    if args.rendering is not None and not tokenizer.is_fast:
        raise RuntimeError("condition extraction requires a fast tokenizer")

    rows_by_dataset = {}
    source_counts = {}
    parse_exclusions_by_dataset = {}
    for dataset in args.datasets:
        rows = load_rollouts(dataset)
        if args.limit is not None:
            rows = rows[: args.limit]
        source_counts[dataset] = len(rows)
        parse_exclusions = []
        if args.rendering is not None:
            rows, parse_exclusions = prepare_rendered_rows(
                rows,
                args.rendering,
                require_all_parse_exclusions=args.limit is None,
            )
        rows_by_dataset[dataset] = rows
        parse_exclusions_by_dataset[dataset] = parse_exclusions

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device}")
    # Base model, no lm_head: we only need hidden states, and Gemma-2's 256k
    # vocab makes the logits tensor alone ~8GB at batch 4 (OOM'd job 21461609).
    # eager attention: Gemma-2 attention soft-capping is silently skipped by
    # some sdpa paths; eager is correct everywhere at these model sizes
    model = AutoModel.from_pretrained(
        args.model,
        revision=resolved_model_revision,
        dtype=torch.float16,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    num_layers = model.config.num_hidden_layers
    layer_index = layer_frac_to_index(num_layers, args.layer_frac)
    short = model_short(args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        rows = rows_by_dataset[dataset]
        labels = np.array([r["label"] for r in rows], dtype=np.int8)
        print(
            f"{dataset}: n={len(rows)} "
            f"honest={(labels == 0).sum()} deceptive={(labels == 1).sum()} "
            f"excluded={(labels == -1).sum()}"
        )
        acts, lengths, n_trunc, tokenized = extract_dataset(
            model, tokenizer, rows, layer_index, device,
            args.batch_size, args.max_tokens,
        )
        assert len(lengths) == len(rows)
        assert acts.shape[0] == int(lengths.sum())
        assert np.isfinite(acts.astype(np.float32)).all(), "non-finite activations"

        stem = (
            condition_stem(short, args.rendering)
            if args.rendering is not None
            else f"{short}__{dataset}"
        )
        np.save(out_dir / f"{stem}__acts.npy", acts)
        np.save(out_dir / f"{stem}__lengths.npy", lengths)
        np.save(out_dir / f"{stem}__labels.npy", labels)
        row_manifest = None
        if args.rendering is not None:
            row_manifest = f"{stem}__rows.json"
            row_records = [
                {
                    "example_id": row["example_id"],
                    "immutable_hash": row["immutable_hash"],
                    "selected_answer": row["selected_answer"],
                    "label": int(row["label"]),
                    "truncated": bool(tokenized_row["truncated"]),
                    "answer_token_span": list(tokenized_row["answer_span"]),
                }
                for row, tokenized_row in zip(rows, tokenized)
            ]
            (out_dir / row_manifest).write_text(
                json.dumps(row_records, indent=2, sort_keys=True) + "\n"
            )
        meta = bind_experiment_protocol_metadata(
            {
                "model": args.model,
                "dataset": dataset,
                "rollout_file": DATASET_FILES[dataset],
                "layer_frac": args.layer_frac,
                "layer_index": layer_index,
                "num_hidden_layers": num_layers,
                "hidden_size": model.config.hidden_size,
                "hs_convention": (
                    "outputs.hidden_states[layer_index]; index 0 is embeddings"
                ),
                "span": (
                    "assistant response tokens after add_generation_prompt prefix"
                ),
                "n_dialogues": len(rows),
                "n_source_dialogues": source_counts[dataset],
                "n_parse_excluded": len(parse_exclusions_by_dataset[dataset]),
                "parse_exclusions": parse_exclusions_by_dataset[dataset],
                "n_honest": int((labels == 0).sum()),
                "n_deceptive": int((labels == 1).sum()),
                "n_excluded": int((labels == -1).sum()),
                "n_truncated": n_trunc,
                "max_tokens": args.max_tokens,
                "limit": args.limit,
                "rendering": args.rendering,
                "row_manifest": row_manifest,
                "requested_model_revision": args.model_revision,
                "requested_tokenizer_revision": args.tokenizer_revision,
                "resolved_model_revision": resolved_model_revision,
                "resolved_tokenizer_revision": resolved_tokenizer_revision,
                "extraction_provenance": producer,
            },
            experiment_protocol,
        )
        (out_dir / f"{stem}__meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n"
        )
        print(f"  saved {stem}: acts {acts.shape}, truncated={n_trunc}")

    print("Done")


if __name__ == "__main__":
    main()
