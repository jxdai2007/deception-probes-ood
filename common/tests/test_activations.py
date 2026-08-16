import warnings

import numpy as np
import pytest
from jrp_common.activations import collect_contrastive_activations


@pytest.mark.slow
def test_shapes_with_tiny_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    # Skip (not fail) only if the model can't be fetched/loaded, e.g. offline or hub
    # unreachable. Once loaded, every assertion below is a hard failure -- a broken
    # activations path must not hide behind this skip on a machine with network.
    try:
        # A cold HF cache makes huggingface_hub's hf_xet transfer path emit its own
        # DeprecationWarning ("hf_xet.download_files() is deprecated ..."), unrelated to
        # our code. Ignore only that specific message, only around these two fetch calls --
        # any other warning (ours, torch's, transformers', sklearn's) still surfaces.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*hf_xet\.download_files\(\) is deprecated.*",
                category=DeprecationWarning,
            )
            tok = AutoTokenizer.from_pretrained(name)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(name, output_hidden_states=True)
    except Exception as e:
        pytest.skip(f"could not fetch/load {name} from HuggingFace hub: {e}")
    pos = ["I will deceive you.", "I am lying now."]
    neg = ["I am telling the truth.", "Here are the honest facts."]
    layers = [1, 2]
    out = collect_contrastive_activations(model, tok, pos, neg, layers)
    assert set(out) == {1, 2}
    X, y = out[1]
    assert X.shape[0] == 4
    assert list(y) == [1, 1, 0, 0]
