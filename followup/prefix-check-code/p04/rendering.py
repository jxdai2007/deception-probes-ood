"""Deterministic response renderings for the format/geometry intervention."""

from dataclasses import dataclass
import hashlib
import json
import re


RENDERINGS = ("r0", "r1", "r2")
FROZEN_PARSE_EXCLUSION_SHA256 = {
    "sandbagging:351:0": (
        "ecb90718dfaff0df39a6fef87c36e0fa62ab768a9f5ebac249d69e0423d63a0b"
    ),
    "sandbagging:410:0": (
        "8b869f978243757c42010376ee8d65d4ac688e3a8a8c91f0b85cc70acd0ff24c"
    ),
    "sandbagging:456:0": (
        "205a76e94a6e0c190433d0e7610d5185238fc51ee1df90f796acf6a36d2e8f29"
    ),
}
_ANSWER_RE = re.compile(r"<answer>\s*([A-Da-d])\s*</answer>")
R1_PROTOCOL_TAG_NAMES = (
    "step1",
    "step1-results",
    "step2",
    "step2-results",
    "step3",
    "step4",
    "answer",
    "sandbag-skill",
)
_TAG_RE = re.compile(
    rf"</?(?:{'|'.join(R1_PROTOCOL_TAG_NAMES)})(?:\s+[^<>]*?)?>"
)
_ANSWER_MARKER = "__JRP_SELECTED_ANSWER_MARKER__"


@dataclass(frozen=True)
class RenderedResponse:
    """One response rendering and the selected-answer character span."""

    text: str
    selected_answer: str
    answer_char_span: tuple[int, int]


def parse_selected_answer(output: str) -> str:
    """Return the sole A-D answer tag, failing closed otherwise."""
    matches = list(_ANSWER_RE.finditer(output))
    if len(matches) != 1:
        raise ValueError("expected exactly one <answer> A-D letter tag")
    return matches[0].group(1).upper()


def render_response(output: str, condition: str) -> RenderedResponse:
    """Render an Apollo completion under the frozen R0/R1/R2 protocol."""
    if condition not in RENDERINGS:
        raise ValueError(f"unknown rendering {condition!r}; expected {RENDERINGS}")

    matches = list(_ANSWER_RE.finditer(output))
    if len(matches) != 1:
        raise ValueError("expected exactly one <answer> A-D letter tag")
    match = matches[0]
    answer = match.group(1).upper()

    if condition == "r0":
        start, end = match.span(1)
        return RenderedResponse(output, answer, (start, end))

    if condition == "r2":
        prefix = "After considering the question, my answer is "
        text = f"{prefix}{answer}."
        return RenderedResponse(text, answer, (len(prefix), len(prefix) + 1))

    start, end = match.span(1)
    marked = output[:start] + _ANSWER_MARKER + output[end:]
    normalized = " ".join(_TAG_RE.sub(" ", marked).split())
    marker_start = normalized.index(_ANSWER_MARKER)
    text = normalized.replace(_ANSWER_MARKER, answer, 1)
    return RenderedResponse(text, answer, (marker_start, marker_start + 1))


def immutable_digest(row: dict, selected_answer: str) -> str:
    """Hash condition-invariant fields for cross-rendering identity checks."""
    payload = {
        "example_id": row["example_id"],
        "input_messages": row["input_messages"],
        "selected_answer": selected_answer,
        "label": int(row["label"]),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
