"""Verify a local Project 04 follow-up release bundle without raw activations."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from p04.figures_pooling_mechanism import (
    _validate_confirmation,
    make_confirmation_figure,
)
from p04.report_pooling_mechanism import write_confirmation_report


EXPECTED_SOURCE_RECORDS = {
    "confirmation_sha256": "d22cd5044d4103e0a5f96920d69df025b99ae9bf3982fce5c785135544c43b74",
    "discovery_sha256": "4a231b95617471287f833a5bba832b7d8332ec80bdadb1a2624cb5d4f928db84",
}
EXPECTED_SCIENTIFIC_HASHES = {
    "pooling_confirmation.generated.md": "8f4b8fa743ae9a4f072f0962b2c0596274c254afe09c80a504ab05adcf0dcb09",
    "pooling_confirmation.png": "effc3f0ded2cb60463fb0510e157afaf83bdb263bcf741b2a8ca817749b236f4",
    "pooling_confirmation.public.json": "8641ceda667711ee8b8a7331c8ca02667f9359d0b8aee9aa230f0424edddb39c",
    "pooling_confirmation_protocol.public.json": "9b77c51f6315c12bfee137229c612a30c6c78638c8bdbee23d5d28092951e128",
    "pooling_mechanism_discovery.public.json": "20ced2cd568acca12680162dd63d6156c9aef664db3aed114f949d476da0cc3d",
}
EXPECTED_PRIMARY_ENDPOINT = {
    "cell_aurocs": {
        "full_r0": 0.22068349774609752,
        "full_treatment": 0.4279503316560482,
        "window_r0": 0.5770718191831605,
        "window_treatment": 0.3846919019637337,
    },
    "ci95": [0.35022102316012177, 0.44821883476395874],
    "common_ids_sha256": "3d400ead3498e97ee9d4220ceb9e383f050534a6344c7373168428244f9b0e2a",
    "estimate": 0.3996467511293775,
    "n_boot": 10000,
    "n_common": 912,
    "se": 0.025258328974485444,
    "seed": 42,
}
EXPECTED_ARTIFACTS = {
    "jrp_common/__init__.py",
    "jrp_common/metrics.py",
    "jrp_common/probes.py",
    "p04/__init__.py",
    "p04/condition_io.py",
    "p04/figures_pooling_mechanism.py",
    "p04/loader.py",
    "p04/pooling.py",
    "p04/pooling_confirmation.py",
    "p04/pooling_mechanism.py",
    "p04/rendering.py",
    "p04/report_pooling_mechanism.py",
    "p04/verify_release_bundle.py",
    "pooling_confirmation.generated.md",
    "pooling_confirmation.png",
    "pooling_confirmation.public.json",
    "pooling_confirmation_protocol.public.json",
    "pooling_mechanism_discovery.public.json",
    "release-environment.json",
    "requirements-release.lock.txt",
    "requirements-release.txt",
    "verify_release.sh",
}


def verify_bundle(bundle_dir):
    bundle_dir = Path(bundle_dir)
    symlinks = sorted(
        path.relative_to(bundle_dir).as_posix()
        for path in bundle_dir.rglob("*")
        if path.is_symlink()
    )
    if symlinks:
        raise ValueError(f"release bundle must not contain symlinks: {symlinks}")
    manifest = json.loads((bundle_dir / "release_manifest.json").read_text())
    if manifest.get("status") != "local_release_staging_only":
        raise ValueError("unexpected release status")

    actual_files = {
        path.relative_to(bundle_dir).as_posix()
        for path in bundle_dir.rglob("*")
        if path.is_file()
    }
    expected_files = EXPECTED_ARTIFACTS | {"release_manifest.json"}
    if actual_files != expected_files:
        extras = sorted(actual_files - expected_files)
        missing = sorted(expected_files - actual_files)
        raise ValueError(f"release file-set mismatch; extras={extras}, missing={missing}")
    if set(manifest.get("artifacts", {})) != EXPECTED_ARTIFACTS:
        raise ValueError("manifest artifact set mismatch")
    if manifest.get("source_records") != EXPECTED_SOURCE_RECORDS:
        raise ValueError("private source-record identities changed")

    for relative, expected in manifest["artifacts"].items():
        artifact = bundle_dir / relative
        observed = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if observed != expected:
            raise ValueError(f"artifact hash mismatch: {relative}")
    for relative, expected in EXPECTED_SCIENTIFIC_HASHES.items():
        observed = hashlib.sha256((bundle_dir / relative).read_bytes()).hexdigest()
        if observed != expected:
            raise ValueError(f"trusted scientific artifact changed: {relative}")

    discovery = json.loads(
        (bundle_dir / "pooling_mechanism_discovery.public.json").read_text()
    )
    confirmation = json.loads(
        (bundle_dir / "pooling_confirmation.public.json").read_text()
    )
    protocol = json.loads(
        (bundle_dir / "pooling_confirmation_protocol.public.json").read_text()
    )
    if protocol != confirmation.get("protocol"):
        raise ValueError("standalone protocol differs from confirmation protocol")
    if discovery.get("public_release", {}).get("private_record_sha256") != (
        EXPECTED_SOURCE_RECORDS["discovery_sha256"]
    ):
        raise ValueError("discovery private-source binding changed")
    if confirmation.get("public_release", {}).get("private_record_sha256") != (
        EXPECTED_SOURCE_RECORDS["confirmation_sha256"]
    ):
        raise ValueError("confirmation private-source binding changed")
    _validate_confirmation(discovery, confirmation)
    if confirmation.get("primary_endpoint") != EXPECTED_PRIMARY_ENDPOINT:
        raise ValueError("frozen primary endpoint record mismatch")
    if confirmation.get("components") != {
        "full_delta": 0.20726683390995065,
        "window_delta": -0.19237991721942682,
    }:
        raise ValueError("frozen component record mismatch")
    if confirmation.get("status") != "confirmed" or confirmation.get(
        "primary_decision"
    ) != "confirmed":
        raise ValueError("confirmation decision changed")
    if confirmation.get("integrity", {}).get("passed") is not True:
        raise ValueError("integrity gate is not passed")
    if confirmation.get("parity", {}).get("passed") is not True:
        raise ValueError("historical parity gate is not passed")

    with TemporaryDirectory(prefix="p04-release-verify-") as temporary:
        temporary = Path(temporary)
        regenerated_figure = temporary / "pooling_confirmation.png"
        regenerated_report = temporary / "pooling_confirmation.generated.md"
        make_confirmation_figure(discovery, confirmation, regenerated_figure)
        write_confirmation_report(
            discovery, confirmation, regenerated_report, "2026-08-20"
        )
        if regenerated_figure.read_bytes() != (
            bundle_dir / "pooling_confirmation.png"
        ).read_bytes():
            raise ValueError("regenerated figure differs from staged figure")
        if regenerated_report.read_bytes() != (
            bundle_dir / "pooling_confirmation.generated.md"
        ).read_bytes():
            raise ValueError("regenerated report differs from staged report")

    return {
        "status": "verified",
        "primary_endpoint": EXPECTED_PRIMARY_ENDPOINT["estimate"],
        "artifact_count": len(manifest["artifacts"]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_dir")
    args = parser.parse_args(argv)
    print(json.dumps(verify_bundle(args.bundle_dir), sort_keys=True))


if __name__ == "__main__":
    main()
