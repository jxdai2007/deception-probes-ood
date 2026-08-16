import json
import os
import subprocess
from unittest import mock

import pytest

from jrp_common import capability
from jrp_common.capability import run_capability, record_run


def test_run_capability_parses_results():
    fake = {"results": {"wmdp_bio": {"acc,none": 0.42}, "mmlu": {"acc,none": 0.55}}}
    # revision is omitted, so run_capability now resolves it at evaluation start;
    # patch that lookup so this test stays offline (see finding: pin at eval time).
    with mock.patch("jrp_common.capability.simple_evaluate", return_value=fake), mock.patch(
        "jrp_common.capability.resolve_revision", return_value=capability.UNRESOLVED_REVISION_MARKER
    ):
        out = run_capability("some/model", ["wmdp_bio", "mmlu"], limit=8)
    assert out == {"wmdp_bio": 0.42, "mmlu": 0.55}


def test_record_run_writes_metadata(tmp_path):
    # revision is passed explicitly (rather than left at the default None) so this
    # test -- which isn't about revision resolution -- doesn't exercise the
    # network-touching resolve_revision path added for finding 2.
    p = record_run(str(tmp_path), "some/model", {"wmdp_bio": 0.42}, revision="unspecified")
    # Read back via a context manager (not the brief's open(p).read()) so the file
    # handle is closed deterministically -- an unclosed handle triggers a
    # ResourceWarning from GC that fails the repo's -W error::Warning test run.
    with open(p) as f:
        data = json.load(f)
    assert data["model_id"] == "some/model"
    assert data["results"]["wmdp_bio"] == 0.42
    assert "git_sha" in data and "timestamp" in data


def test_run_capability_raises_clear_error_when_lm_eval_missing():
    # simple_evaluate is None on a machine without lm-eval installed; run_capability
    # must fail loudly with actionable guidance rather than a bare TypeError/AttributeError.
    with mock.patch("jrp_common.capability.simple_evaluate", None):
        with pytest.raises(RuntimeError, match="lm-eval"):
            run_capability("some/model", ["wmdp_bio"])


def test_run_capability_plumbs_determinism_and_fewshot():
    # Global Constraint: every eval sets a fixed seed and pins the model revision.
    # num_fewshot must reach simple_evaluate too -- the compression gate (Retain MMLU
    # 54.7%) is a 5-shot number, and lm-eval defaults to 0-shot.
    fake = {"results": {"wmdp_bio": {"acc,none": 0.281}}}
    with mock.patch("jrp_common.capability.simple_evaluate", return_value=fake) as m:
        run_capability(
            "some/model",
            ["wmdp_bio"],
            revision="deadbeef",
            seed=1234,
            num_fewshot=5,
        )
    _, kwargs = m.call_args
    assert "pretrained=some/model" in kwargs["model_args"]
    assert "revision=deadbeef" in kwargs["model_args"]
    assert kwargs["random_seed"] == 1234
    assert kwargs["numpy_random_seed"] == 1234
    assert kwargs["torch_random_seed"] == 1234
    assert kwargs["num_fewshot"] == 5


def test_record_run_without_tag_keeps_original_filename(tmp_path):
    p = record_run(str(tmp_path), "some/model", {"wmdp_bio": 0.42}, revision="unspecified")
    assert p == os.path.join(str(tmp_path), "run_some_model.json")


def test_record_run_with_tag_disambiguates_filename(tmp_path):
    # The compression project evaluates one model at several quantization/pruning
    # settings; without a discriminator every setting would clobber the same path.
    p1 = record_run(str(tmp_path), "some/model", {"wmdp_bio": 0.42}, tag="int8", revision="unspecified")
    p2 = record_run(str(tmp_path), "some/model", {"wmdp_bio": 0.40}, tag="pruned50", revision="unspecified")
    assert p1 != p2
    assert os.path.exists(p1)
    assert os.path.exists(p2)
    with open(p1) as f:
        data1 = json.load(f)
    with open(p2) as f:
        data2 = json.load(f)
    assert data1["results"]["wmdp_bio"] == 0.42
    assert data2["results"]["wmdp_bio"] == 0.40


def test_record_run_writes_revision_and_num_fewshot(tmp_path):
    p = record_run(
        str(tmp_path), "some/model", {"mmlu": 0.547}, revision="abc123", num_fewshot=5
    )
    with open(p) as f:
        data = json.load(f)
    assert data["revision"] == "abc123"
    assert data["num_fewshot"] == 5


# --- Finding 1: limit, model_args, tasks recordable -------------------------------


def test_record_run_writes_limit_model_args_and_tasks(tmp_path):
    # A limit=100 smoke-test number and a full-dataset accuracy must be
    # distinguishable in the recorded JSON without relying on the free-text tag.
    p = record_run(
        str(tmp_path),
        "some/model",
        {"wmdp_bio": 0.42, "mmlu": 0.55},
        revision="abc123",
        limit=100,
        model_args="dtype=bfloat16",
        tasks=["wmdp_bio", "mmlu"],
    )
    with open(p) as f:
        data = json.load(f)
    assert data["limit"] == 100
    assert data["model_args"] == "dtype=bfloat16"
    assert data["tasks"] == ["wmdp_bio", "mmlu"]


def test_record_run_defaults_tasks_from_results_when_not_given(tmp_path):
    p = record_run(str(tmp_path), "some/model", {"wmdp_bio": 0.42}, revision="abc123")
    with open(p) as f:
        data = json.load(f)
    assert data["tasks"] == ["wmdp_bio"]
    assert data["limit"] is None
    assert data["model_args"] == ""


# --- Finding 2: revision resolution when not given ---------------------------------


def test_resolve_revision_returns_sha_when_hub_lookup_succeeds():
    fake_info = mock.Mock(sha="deadbeef123")
    with mock.patch("huggingface_hub.model_info", return_value=fake_info):
        assert capability.resolve_revision("some/model") == "deadbeef123"


def test_resolve_revision_returns_marker_when_hub_lookup_fails():
    # Offline, gated, or a local checkpoint path all surface as some exception from
    # huggingface_hub -- resolve_revision must not raise, and must not return None.
    with mock.patch("huggingface_hub.model_info", side_effect=OSError("offline")):
        result = capability.resolve_revision("some/model")
    assert isinstance(result, str)
    assert result != "null"
    assert "unresolved" in result


def test_record_run_uses_resolved_revision_when_none_given(tmp_path):
    with mock.patch(
        "jrp_common.capability.resolve_revision", return_value="resolved-sha-123"
    ) as m:
        p = record_run(str(tmp_path), "some/model", {"wmdp_bio": 0.42})
    m.assert_called_once_with("some/model")
    with open(p) as f:
        data = json.load(f)
    assert data["revision"] == "resolved-sha-123"


def test_record_run_records_marker_when_resolution_unresolved(tmp_path):
    # Asserts against the production module's own marker constant (not a hardcoded
    # copy of the string) so this test cannot pass if the marker text changes but the
    # mock's return value silently drifts along with it.
    with mock.patch(
        "jrp_common.capability.resolve_revision",
        return_value=capability.UNRESOLVED_REVISION_MARKER,
    ):
        p = record_run(str(tmp_path), "some/model", {"wmdp_bio": 0.42})
    with open(p) as f:
        data = json.load(f)
    assert data["revision"] == capability.UNRESOLVED_REVISION_MARKER


def test_record_run_skips_resolution_when_revision_given(tmp_path):
    # An explicit revision must not trigger a hub lookup at all -- no network call.
    with mock.patch("jrp_common.capability.resolve_revision") as m:
        record_run(str(tmp_path), "some/model", {"wmdp_bio": 0.42}, revision="abc123")
    m.assert_not_called()


# --- Follow-up: resolve-and-pin at evaluation time (T0), honest revision_source ------
#
# record_run's own post-hoc resolution (Finding 2, above) runs when the payload is
# written, hours after an overnight eval starts -- the resolved SHA can name a
# snapshot that moved after the eval actually ran. run_capability must instead
# resolve once at evaluation start and pin the fetch itself.


def test_run_capability_pins_resolved_revision_when_none_given():
    # revision omitted and resolution succeeds -- the resolved SHA must be pinned
    # into the model-args string so the evaluation itself is pinned to that snapshot.
    fake = {"results": {"wmdp_bio": {"acc,none": 0.3}}}
    with mock.patch("jrp_common.capability.simple_evaluate", return_value=fake) as m, mock.patch(
        "jrp_common.capability.resolve_revision", return_value="resolved-sha-456"
    ) as resolve_mock:
        run_capability("some/model", ["wmdp_bio"])
    resolve_mock.assert_called_once_with("some/model")
    _, kwargs = m.call_args
    assert kwargs["model_args"] == "pretrained=some/model,revision=resolved-sha-456"


def test_run_capability_proceeds_unpinned_when_resolution_fails():
    # revision omitted and resolution fails -- proceed exactly as before (unpinned),
    # since an eval must never die because provenance lookup failed.
    fake = {"results": {"wmdp_bio": {"acc,none": 0.3}}}
    with mock.patch("jrp_common.capability.simple_evaluate", return_value=fake) as m, mock.patch(
        "jrp_common.capability.resolve_revision", return_value=capability.UNRESOLVED_REVISION_MARKER
    ) as resolve_mock:
        run_capability("some/model", ["wmdp_bio"])
    # Proves the attempt actually happened -- without this, the test would still pass
    # if run_capability never called resolve_revision at all.
    resolve_mock.assert_called_once_with("some/model")
    _, kwargs = m.call_args
    assert kwargs["model_args"] == "pretrained=some/model"


def test_run_capability_explicit_revision_skips_resolution():
    # An explicit caller revision must short-circuit resolution entirely -- no
    # network call, and the caller's exact value is what gets pinned.
    fake = {"results": {"wmdp_bio": {"acc,none": 0.3}}}
    with mock.patch("jrp_common.capability.simple_evaluate", return_value=fake) as m, mock.patch(
        "jrp_common.capability.resolve_revision"
    ) as resolve_mock:
        run_capability("some/model", ["wmdp_bio"], revision="caller-pinned-sha")
    resolve_mock.assert_not_called()
    _, kwargs = m.call_args
    assert kwargs["model_args"] == "pretrained=some/model,revision=caller-pinned-sha"


def test_record_run_revision_source_labels_caller_provided(tmp_path):
    p = record_run(str(tmp_path), "some/model", {"wmdp_bio": 0.42}, revision="abc123")
    with open(p) as f:
        data = json.load(f)
    assert data["revision"] == "abc123"
    assert data["revision_source"] == "caller-provided"


def test_record_run_revision_source_labels_resolved_at_record_time(tmp_path):
    with mock.patch(
        "jrp_common.capability.resolve_revision", return_value="resolved-sha-123"
    ):
        p = record_run(str(tmp_path), "some/model", {"wmdp_bio": 0.42})
    with open(p) as f:
        data = json.load(f)
    assert data["revision"] == "resolved-sha-123"
    assert data["revision_source"] == "resolved-at-record-time"


def test_record_run_revision_source_labels_unresolved(tmp_path):
    with mock.patch(
        "jrp_common.capability.resolve_revision", return_value=capability.UNRESOLVED_REVISION_MARKER
    ):
        p = record_run(str(tmp_path), "some/model", {"wmdp_bio": 0.42})
    with open(p) as f:
        data = json.load(f)
    assert data["revision"] == capability.UNRESOLVED_REVISION_MARKER
    assert data["revision_source"] == "unresolved"


# --- Finding 3: _git_sha must not depend on process cwd -----------------------------


def test_git_sha_ignores_process_cwd(tmp_path, monkeypatch):
    harness_dir = os.path.dirname(os.path.abspath(capability.__file__))
    expected = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=harness_dir)
        .decode()
        .strip()
    )
    monkeypatch.chdir(tmp_path)
    assert capability._git_sha() == expected


# --- Finding 5: model-args comma-placement, remaining combinations -----------------


def test_run_capability_model_args_only():
    fake = {"results": {"wmdp_bio": {"acc,none": 0.3}}}
    # revision is omitted; force resolution to fail so the expected model_args stays
    # unpinned, and so this test doesn't reach the network (see finding: pin at eval time).
    with mock.patch("jrp_common.capability.simple_evaluate", return_value=fake) as m, mock.patch(
        "jrp_common.capability.resolve_revision", return_value=capability.UNRESOLVED_REVISION_MARKER
    ):
        run_capability("some/model", ["wmdp_bio"], model_args="dtype=bfloat16")
    _, kwargs = m.call_args
    assert kwargs["model_args"] == "pretrained=some/model,dtype=bfloat16"


def test_run_capability_revision_and_model_args_both_present():
    fake = {"results": {"wmdp_bio": {"acc,none": 0.3}}}
    with mock.patch("jrp_common.capability.simple_evaluate", return_value=fake) as m:
        run_capability(
            "some/model",
            ["wmdp_bio"],
            revision="deadbeef",
            model_args="dtype=bfloat16",
        )
    _, kwargs = m.call_args
    assert kwargs["model_args"] == "pretrained=some/model,revision=deadbeef,dtype=bfloat16"


# --- Finding 7: missing acc key must raise, not silently write NaN -----------------


def test_run_capability_raises_on_missing_acc_key():
    fake = {"results": {"wmdp_bio": {"acc_norm,none": 0.42}}}
    # revision is omitted; patch resolution so this test doesn't reach the network
    # (see finding: pin at eval time).
    with mock.patch("jrp_common.capability.simple_evaluate", return_value=fake), mock.patch(
        "jrp_common.capability.resolve_revision", return_value=capability.UNRESOLVED_REVISION_MARKER
    ):
        with pytest.raises(ValueError, match="wmdp_bio") as exc_info:
            run_capability("some/model", ["wmdp_bio"])
    assert "acc_norm,none" in str(exc_info.value)


# --- resolve_revision as public API: resolve-once-then-thread ----------------------


def test_threaded_revision_matches_between_run_capability_and_record_run(tmp_path):
    # The invariant this whole change exists to establish: resolve once, pass the
    # same value to both calls, and the SHA pinned into the eval is provably the SHA
    # recorded -- rather than each call resolving independently and possibly drifting.
    fake = {"results": {"wmdp_bio": {"acc,none": 0.3}}}
    with mock.patch(
        "jrp_common.capability.resolve_revision", return_value="threaded-sha-789"
    ) as resolve_mock:
        rev = capability.resolve_revision("some/model")
    resolve_mock.assert_called_once_with("some/model")

    with mock.patch("jrp_common.capability.simple_evaluate", return_value=fake) as m:
        res = run_capability("some/model", ["wmdp_bio"], revision=rev)
    _, kwargs = m.call_args
    assert "revision=threaded-sha-789" in kwargs["model_args"]

    p = record_run(str(tmp_path), "some/model", res, revision=rev)
    with open(p) as f:
        data = json.load(f)
    assert data["revision"] == "threaded-sha-789"
    assert data["revision_source"] == "caller-provided"
    assert f"revision={data['revision']}" in kwargs["model_args"]


# --- Marker hardening: threading the marker itself must be safe --------------------


def test_run_capability_caller_supplied_marker_is_unpinned_and_not_reresolved():
    # A caller who threads resolve_revision's own failure output back in (the
    # documented resolve-once-then-thread pattern) must not have it pinned as a
    # literal revision string, and must not trigger a second lookup.
    fake = {"results": {"wmdp_bio": {"acc,none": 0.3}}}
    with mock.patch("jrp_common.capability.simple_evaluate", return_value=fake) as m, mock.patch(
        "jrp_common.capability.resolve_revision"
    ) as resolve_mock:
        run_capability(
            "some/model", ["wmdp_bio"], revision=capability.UNRESOLVED_REVISION_MARKER
        )
    resolve_mock.assert_not_called()
    _, kwargs = m.call_args
    assert kwargs["model_args"] == "pretrained=some/model"


def test_record_run_caller_supplied_marker_labeled_unresolved(tmp_path):
    # Same threaded-marker scenario on the record_run side: the marker must not be
    # mislabeled as a real caller pin, and must not trigger a second lookup.
    with mock.patch("jrp_common.capability.resolve_revision") as resolve_mock:
        p = record_run(
            str(tmp_path),
            "some/model",
            {"wmdp_bio": 0.42},
            revision=capability.UNRESOLVED_REVISION_MARKER,
        )
    resolve_mock.assert_not_called()
    with open(p) as f:
        data = json.load(f)
    assert data["revision"] == capability.UNRESOLVED_REVISION_MARKER
    assert data["revision_source"] == "unresolved"


# --- Finding 4: real (unmocked) call path -------------------------------------------


@pytest.mark.slow
def test_run_capability_real_call_path():
    # Every other test in this file mocks simple_evaluate. This one exercises the
    # real lm-eval 0.4.12 call path, so a future lm-eval upgrade that renames or
    # drops one of the seed kwargs fails here instead of mid overnight run.
    # Smallest available ungated model + task: tiny random GPT-2 weights, arc_easy,
    # limit=2 so only two examples are scored.
    model_id = "sshleifer/tiny-gpt2"
    task = "arc_easy"

    # Pre-flight fetch only, mirroring test_activations.py: skip (don't fail) if the
    # model or dataset can't be reached, e.g. offline. Once this succeeds, the real
    # run_capability call below is not shielded by the skip, so a genuine breakage
    # (like a renamed kwarg) fails hard.
    try:
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer

        AutoTokenizer.from_pretrained(model_id)
        AutoModelForCausalLM.from_pretrained(model_id)
        load_dataset("allenai/ai2_arc", "ARC-Easy", split="test[:2]")
    except Exception as e:
        pytest.skip(f"could not fetch {model_id} or the arc_easy dataset: {e}")

    out = run_capability(model_id, [task], limit=2, model_args="device=cpu")
    assert isinstance(out, dict)
    assert set(out) == {task}
    assert isinstance(out[task], float)
