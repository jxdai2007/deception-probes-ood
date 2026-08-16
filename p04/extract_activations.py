#!/usr/bin/env python3
"""Extract per-token response activations for one model over the transfer datasets.

Usage:
    python -m p04.extract_activations --model google/gemma-2-2b-it \
        --out-dir /scratch/USER/results/04-activations [--layer-frac 0.275] \
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
import json
from pathlib import Path

import numpy as np
import torch

from p04.loader import DATASET_FILES, load_rollouts, layer_frac_to_index


def model_short(model_id: str) -> str:
    return model_id.rstrip("/").split("/")[-1]


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
        full = prompt_ids + tokenizer.encode(
            row["output_str"], add_special_tokens=False
        )
        truncated = len(full) > max_tokens
        full = full[:max_tokens]
        out.append(
            {
                "ids": full,
                "prompt_len": min(len(prompt_ids), len(full)),
                "truncated": truncated,
            }
        )
    return out


@torch.no_grad()
def extract_dataset(model, tokenizer, rows, layer_index, device, batch_size, max_tokens):
    """Return (acts fp16 [sum_tokens, hidden], lengths int32, n_truncated)."""
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
    return acts, np.array(lengths, dtype=np.int32), n_truncated


def main():
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
    args = parser.parse_args()

    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Base model, no lm_head: we only need hidden states, and Gemma-2's 256k
    # vocab makes the logits tensor alone ~8GB at batch 4 (OOM'd job 21461609).
    # eager attention: Gemma-2 attention soft-capping is silently skipped by
    # some sdpa paths; eager is correct everywhere at these model sizes
    model = AutoModel.from_pretrained(
        args.model, dtype=torch.float16, attn_implementation="eager"
    ).to(device)
    model.eval()

    num_layers = model.config.num_hidden_layers
    layer_index = layer_frac_to_index(num_layers, args.layer_frac)
    short = model_short(args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        rows = load_rollouts(dataset)
        if args.limit:
            rows = rows[: args.limit]
        labels = np.array([r["label"] for r in rows], dtype=np.int8)
        print(
            f"{dataset}: n={len(rows)} "
            f"honest={(labels == 0).sum()} deceptive={(labels == 1).sum()} "
            f"excluded={(labels == -1).sum()}"
        )
        acts, lengths, n_trunc = extract_dataset(
            model, tokenizer, rows, layer_index, device,
            args.batch_size, args.max_tokens,
        )
        assert len(lengths) == len(rows)
        assert acts.shape[0] == int(lengths.sum())
        assert np.isfinite(acts.astype(np.float32)).all(), "non-finite activations"

        stem = f"{short}__{dataset}"
        np.save(out_dir / f"{stem}__acts.npy", acts)
        np.save(out_dir / f"{stem}__lengths.npy", lengths)
        np.save(out_dir / f"{stem}__labels.npy", labels)
        meta = {
            "model": args.model,
            "dataset": dataset,
            "rollout_file": DATASET_FILES[dataset],
            "layer_frac": args.layer_frac,
            "layer_index": layer_index,
            "num_hidden_layers": num_layers,
            "hidden_size": model.config.hidden_size,
            "hs_convention": "outputs.hidden_states[layer_index]; index 0 is embeddings",
            "span": "assistant response tokens after add_generation_prompt prefix",
            "n_dialogues": len(rows),
            "n_honest": int((labels == 0).sum()),
            "n_deceptive": int((labels == 1).sum()),
            "n_excluded": int((labels == -1).sum()),
            "n_truncated": n_trunc,
            "max_tokens": args.max_tokens,
        }
        (out_dir / f"{stem}__meta.json").write_text(json.dumps(meta, indent=2))
        print(f"  saved {stem}: acts {acts.shape}, truncated={n_trunc}")

    print("Done")


if __name__ == "__main__":
    main()
