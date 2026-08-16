#!/usr/bin/env python3
"""One-shot detector.pt converter: extracts probe weights to .npy/.json.

THIS IS THE ONLY PLACE weights_only=False APPEARS IN THE REPO.
Containment: subprocess-only execution, provenance checks, sanity gates.

Security note: weights_only=False used for legacy torch pickle format.
Pickle verified safe (only torch._utils._rebuild_tensor_v2,
torch.storage._load_from_bytes GLOBALs). Source: pinned vendor clone of
ApolloResearch/deception-detection.
"""

import argparse
import hashlib
import importlib
import json
import pickle
import torch
import numpy as np
from pathlib import Path


# Restricted unpickler allowlist (STEP A verified globals only)
ALLOWED = {
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch.storage", "_load_from_bytes"),
    ("collections", "OrderedDict"),
}


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickler with explicit allowlist for security."""

    def find_class(self, module, name):
        if (module, name) in ALLOWED:
            return getattr(importlib.import_module(module), name)
        raise pickle.UnpicklingError(f"blocked global: {module}.{name}")


def compute_sha256(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def convert_detector(in_path: Path, out_dir: Path) -> dict:
    """Convert detector.pt to .npy/.json manifest.

    Args:
        in_path: Path to detector.pt (from pinned vendor clone)
        out_dir: Output directory for .npy/.json

    Returns:
        Manifest dict (written to manifest.json)
    """
    # Provenance check: ensure file exists
    if not in_path.exists():
        raise FileNotFoundError(f"Detector not found: {in_path}")

    # Compute source hash for provenance
    source_sha256 = compute_sha256(in_path)

    print(f"Converting {in_path.name}")
    print(f"  Source SHA256: {source_sha256}")

    # ONE-SHOT LOAD (only place weights_only=False appears)
    # Apollo used plain pickle.dump, not torch.save. Using restricted unpickler.
    with open(in_path, "rb") as f:
        detector = RestrictedUnpickler(f).load()

    # Extract components. Released LogisticRegressionDetector dicts carry:
    # layers, directions, scaler_mean, scaler_scale, normalize, reg_coeff.
    # Dropping the scaler silently breaks scoring: their get_score_tensor
    # computes ((x - scaler_mean) / scaler_scale) @ direction per token.
    layers = detector["layers"]  # list[int]
    directions = detector["directions"].cpu().float().numpy()  # [layer, hidden_dim]
    normalize = bool(detector.get("normalize", False))
    reg_coeff = detector.get("reg_coeff")
    scaler_mean = detector.get("scaler_mean")
    scaler_scale = detector.get("scaler_scale")

    # SAE-latent probes (Goodfire SAE of llama-70b-3.3, 131072 latents) are
    # excluded from transfer legs by construction: transfer targets have no
    # matching SAE encoder. Detect by config, gate residual probes on 8192.
    base_name = in_path.parent.name  # e.g., "roleplaying"
    cfg_path = in_path.parent / "cfg.yaml"
    sae_probe = "use_goodfire_sae_acts: true" in cfg_path.read_text()
    hidden_dim = directions.shape[1]
    expected_dim = 8192

    if not sae_probe:
        assert hidden_dim == expected_dim, (
            f"Probe dimension {hidden_dim} != expected {expected_dim} "
            "(Llama-3.3-70B residual stream)"
        )
    if normalize:
        assert scaler_mean is not None and scaler_scale is not None, (
            "normalize=True but scaler tensors missing from detector"
        )

    assert np.all(np.isfinite(directions)), "Probe weights contain NaN/Inf"

    print(f"  Layers: {layers}")
    print(f"  Directions shape: {directions.shape}")
    print(f"  normalize={normalize} reg_coeff={reg_coeff} sae_probe={sae_probe}")

    # Create output dir
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}
    directions_path = out_dir / f"{base_name}_directions.npy"
    np.save(directions_path, directions)
    outputs["directions_npy"] = str(directions_path)

    if scaler_mean is not None:
        sm = scaler_mean.cpu().float().numpy()
        ss = scaler_scale.cpu().float().numpy()
        assert np.all(np.isfinite(sm)) and np.all(np.isfinite(ss)), "scaler NaN/Inf"
        assert np.all(ss != 0), "scaler_scale contains zeros"
        sm_path = out_dir / f"{base_name}_scaler_mean.npy"
        ss_path = out_dir / f"{base_name}_scaler_scale.npy"
        np.save(sm_path, sm)
        np.save(ss_path, ss)
        outputs["scaler_mean_npy"] = str(sm_path)
        outputs["scaler_scale_npy"] = str(ss_path)

    # Create manifest
    manifest = {
        "source_file": str(in_path),
        "source_sha256": source_sha256,
        "layers": layers,
        "directions_shape": list(directions.shape),
        "directions_dtype": str(directions.dtype),
        "hidden_dim": int(hidden_dim),
        "expected_residual_dim": expected_dim,
        "normalize": normalize,
        "reg_coeff": reg_coeff,
        "probe_space": "sae_latent" if sae_probe else "residual",
        "excluded_from_transfer": sae_probe,
        "exclusion_reason": (
            "sae-latent probe; requires source-model SAE; unrefitted "
            "cross-model transfer undefined" if sae_probe else None
        ),
        "finite_values": bool(np.all(np.isfinite(directions))),
        "outputs": outputs,
    }

    # Save manifest
    manifest_path = out_dir / f"{base_name}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  Saved: {directions_path}")
    print(f"  Manifest: {manifest_path}")

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Convert Apollo detector.pt to .npy/.json (one-shot)"
    )
    parser.add_argument("--in", required=True, dest="in_path", help="Path to detector.pt")
    parser.add_argument(
        "--out-dir", required=True, dest="out_dir", help="Output directory"
    )
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir)

    manifest = convert_detector(in_path, out_dir)

    print("\nConversion complete. Manifest:")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
