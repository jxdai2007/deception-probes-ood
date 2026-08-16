# Vendor pin — Apollo deception-detection

- Repo: https://github.com/ApolloResearch/deception-detection.git
- Pinned commit: `f8ec4010e74927394709dffa22b97bdf8cd5a62f`
- Local path: `projects/04-deception-probes/vendor/deception-detection/` (gitignored —
  full clone with binary artifacts; re-clone at the pinned commit to reproduce)
- Released detector.pt sha256 (all under `example_results/<name>/detector.pt`):
  - roleplaying `c83aa151373d94d1dc8da879ff4a87b18de92a1e209820223a6802067a267487`
  - descriptive `5ed709c4b0f98413a6222153e9d5b777d3c7663ce5de4aa8bce0c8f62f0eff2d`
  - followup `62acc541cdc8ab603fd4dd3ddf27cdc68100ef0d4bb37b7ece01e4e3646c2eda`
  - instructed_pairs `203d8a4a7c2007e2e0e5bdde253f27fd82a3ed90abfeee6468511661848c3ac5`
  - sae_rp `0d8bb2ccbc5b869a6b624ae0fb5d2e2505b7648a590d9fa9c6507138d494b0a7`
    (SAE-latent probe, `use_goodfire_sae_acts: true` in its cfg.yaml — excluded
    from transfer legs by construction)
- Artifact format note: detector.pt files are plain `pickle.dump` outputs (NOT
  torch.save) — outer pickle globals verified to be torch internals only;
  converted once via `p04/convert_detector.py` to .npy + manifest.

## Transfer-model mirrors (gated canonicals; gate-era mirror pattern)

Used (all ungated, auth_check OK anonymous, 2026-08-16):
- `unsloth/gemma-2-2b-it` — tokenizer.json sha256 3f289bc05132635a…, config e3885c56…
- `unsloth/gemma-2-9b-it` — tokenizer.json 3f289bc05132635a… (== 2b, shared Gemma vocab), config 7b881c1a…
- `unsloth/Llama-3.2-1B-Instruct` — tokenizer.json 6b9e4e7fb171f92f…, config dfb67fd8…
- `unsloth/Llama-3.2-3B-Instruct` — tokenizer.json 6b9e4e7fb171f92f… (== 1B), config eadff796…
- `NousResearch/Meta-Llama-3.1-8B-Instruct` — tokenizer.json 79e3e522635f3171…, config 29e4c210…

Cross-mirror check (8B): NousResearch vs `unsloth/Meta-Llama-3.1-8B-Instruct`
tokenizer.json bytes DIFFER (serialization), but vocab dicts are equal
(128256 entries), sample encodings identical. Chat templates differ between
the two mirrors; canonical-template verification requires gated meta-llama
access — flagged, pending a token. Instrument-internal consistency is
unaffected (each model is templated uniformly across all datasets and legs).
