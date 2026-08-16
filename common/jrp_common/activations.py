import numpy as np
import torch


def collect_contrastive_activations(model, tokenizer, pos_prompts, neg_prompts, layers, device="cpu"):
    model.eval().to(device)
    prompts = list(pos_prompts) + list(neg_prompts)
    labels = np.array([1] * len(pos_prompts) + [0] * len(neg_prompts))
    per_layer = {l: [] for l in layers}
    with torch.no_grad():
        for p in prompts:
            enc = tokenizer(p, return_tensors="pt").to(device)
            # output_hidden_states=True returns a tuple of length num_layers + 1: index 0
            # is the embedding output (before any transformer layer runs), and index l
            # (for l >= 1) is the output of transformer layer l. So `layers` here are real
            # 1-based transformer-layer numbers, not 0-based -- this is the convention
            # layer numbers reported elsewhere (e.g. "probe after layer 22 of 80") rely on.
            hs = model(**enc, output_hidden_states=True).hidden_states  # tuple[num_layers+1]
            for l in layers:
                per_layer[l].append(hs[l][0, -1, :].float().cpu().numpy())
    return {l: (np.stack(per_layer[l]), labels) for l in layers}
