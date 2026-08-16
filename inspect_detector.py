import torch
from pathlib import Path

detector_path = Path("projects/04-deception-probes/vendor/deception-detection/example_results/roleplaying/detector.pt")
d = torch.load(detector_path, map_location="cpu", weights_only=False)

print(f"Type: {type(d)}")
print(f"Class: {d.__class__.__name__}")
print(f"Module: {d.__class__.__module__}")

if hasattr(d, "__dict__"):
    print("\nAttributes:")
    for key in list(d.__dict__.keys())[:20]:
        print(f"  {key}: {type(getattr(d, key))}")

print("\nString representation:")
print(str(d)[:500])
