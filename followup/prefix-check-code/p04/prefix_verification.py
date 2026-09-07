"""Inspectable contract for the frozen and corrected assistant-prefix paths."""

import hashlib
import json
from datetime import date


class PrefixContractError(ValueError):
    """Raised when a proposed prefix correction lacks a matched identity."""


class DateBoundTokenizer:
    """Bind the template's explicit date without modifying its pinned bytes.

    The declared date may be a reconstruction; this adapter does not establish
    historical token identity. English month names avoid locale-dependent input.
    """

    def __init__(self, tokenizer, chat_template_date):
        try:
            parsed = date.fromisoformat(chat_template_date)
            if parsed.isoformat() != chat_template_date:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise PrefixContractError("chat template date must be a valid ISO date (YYYY-MM-DD)") from exc
        self._tokenizer = tokenizer
        self.chat_template_date = chat_template_date
        months = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
        self.date_string = f"{parsed.day:02d} {months[parsed.month - 1]} {parsed.year:04d}"

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)

    def __call__(self, *args, **kwargs):
        return self._tokenizer(*args, **kwargs)

    def apply_chat_template(self, *args, **kwargs):
        if "date_string" in kwargs and kwargs["date_string"] != self.date_string:
            raise PrefixContractError("date_string conflicts with the bound chat template date")
        kwargs["date_string"] = self.date_string
        return self._tokenizer.apply_chat_template(*args, **kwargs)


def _digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def _messages(row):
    messages = row.get("input_messages")
    if not isinstance(messages, list) or any(
        not isinstance(item, dict)
        or not {"role", "content"}.issubset(item)
        or not isinstance(item["role"], str)
        or not isinstance(item["content"], str)
        for item in messages
    ):
        raise PrefixContractError("input messages must be role/content records")
    # Match the frozen extractor's projection; vendor `detect` metadata does
    # not participate in its chat template or this continuation comparison.
    return [{"role": item["role"], "content": item["content"]} for item in messages]


def _truncate(ids, prompt_length, max_tokens, label):
    if len(ids) > max_tokens:
        if prompt_length >= max_tokens:
            raise PrefixContractError(f"{label} truncates the complete scored response")
        raise PrefixContractError(f"{label} truncates part of the scored response")
    return {"input_ids": ids, "prompt_length": prompt_length, "scored_span": [prompt_length, len(ids)], "truncated": False}


def compare_token_contracts(tokenizer, row, *, max_tokens):
    """Use the unchanged extractor as baseline; continue only real prefixes.

    Prefix-only tokens are context. A token crossing the character boundary is
    scored because it overlaps the response; this is explicitly disclosed.
    Truncated rows are rejected, never silently removed from a fit cohort.
    """
    from p04.extract_activations import (
        response_token_spans, _template_supports_system, _fold_system_into_user,
    )

    response = row.get("output_str")
    if not isinstance(response, str) or not response:
        raise PrefixContractError("empty assistant response has no scored span")
    if type(max_tokens) is not int or max_tokens <= 0:
        raise PrefixContractError("max_tokens must be a positive integer")
    messages = _messages(row)
    if messages and messages[-1]["role"] == "assistant" and not messages[-1]["content"]:
        messages = messages[:-1]
    actual = response_token_spans(tokenizer, [row], max_tokens)[0]
    if actual["truncated"]:
        raise PrefixContractError("frozen construction truncates the scored response")
    frozen = _truncate(actual["ids"], actual["prompt_len"], max_tokens, "frozen construction")
    if frozen["scored_span"][0] == frozen["scored_span"][1]:
        raise PrefixContractError("empty assistant response tokenization")
    candidate = dict(frozen)
    has_prefix = bool(messages and messages[-1]["role"] == "assistant" and messages[-1]["content"])
    if has_prefix:
        if not getattr(tokenizer, "is_fast", False):
            raise PrefixContractError("prefix continuation requires fast-tokenizer offsets")
        if not _template_supports_system(tokenizer):
            messages = _fold_system_into_user(messages)
        messages[-1] = dict(messages[-1], content=messages[-1]["content"] + response)
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, continue_final_message=True, tokenize=False,
        )
        if not isinstance(text, str) or not text.endswith(response):
            raise PrefixContractError("template does not preserve the exact response suffix")
        boundary = len(text) - len(response)
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = list(encoded["input_ids"]), list(encoded["offset_mapping"])
        if len(ids) != len(offsets):
            raise PrefixContractError("token offsets are not aligned")
        hits = [i for i, (a, b) in enumerate(offsets) if b > boundary and a < len(text)]
        if not hits or hits != list(range(hits[0], len(ids))):
            raise PrefixContractError("response tokens must be a nonempty contiguous suffix")
        candidate = _truncate(ids, hits[0], max_tokens, "candidate continuation")
        candidate["boundary_spanning_token"] = offsets[hits[0]][0] < boundary
        candidate["response_char_start"] = boundary
    return {
        "schema_version": 2,
        "frozen": frozen, "candidate": candidate,
        "correction_applied": has_prefix,
        "response_token_count": len(actual["ids"]) - actual["prompt_len"],
        "prefix_policy": "context except any token overlapping the response boundary",
        "eos_policy": "baseline unchanged; continuation adds no EOS/end-of-turn token",
    }


def validate_evaluation_reuse(recorded, proposed):
    """Reject reuse unless every token/mask/runtime/source identity agrees."""
    required = {"input_ids_sha256", "scoring_mask_sha256", "layer", "runtime", "row_mapping_sha256"}
    if not isinstance(recorded, dict) or not isinstance(proposed, dict) or set(recorded) != required or set(proposed) != required:
        raise PrefixContractError("evaluation reuse identity has an unexpected schema")
    for key in ("input_ids_sha256", "scoring_mask_sha256", "row_mapping_sha256"):
        if not isinstance(recorded[key], str) or len(recorded[key]) != 64:
            raise PrefixContractError(f"recorded {key} is invalid")
    for key in required:
        if recorded[key] != proposed[key]:
            raise PrefixContractError(f"evaluation reuse {key} mismatch")
    return {"eligible": True, "identity_sha256": _digest(recorded)}
