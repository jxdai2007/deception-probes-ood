"""Inspect detector.pt pickle structure without code execution."""

import pickletools
from pathlib import Path

detector_path = Path("projects/04-deception-probes/vendor/deception-detection/example_results/roleplaying/detector.pt")

print("=== Pickle Opcodes ===")
with open(detector_path, "rb") as f:
    pickletools.dis(f)
