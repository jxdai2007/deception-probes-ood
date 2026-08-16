"""Tests for layer fraction to layer index mapping."""

import pytest


def test_layer_frac_maps_to_int_layer():
    """Assert layer_frac maps to correct integer layer index."""
    from p04.loader import layer_frac_to_index

    # Llama-3.1-70B has 80 layers, paper uses layer 22 (0.275 * 80 ≈ 22)
    assert layer_frac_to_index(depth=80, frac=0.275) == 22

    # Edge cases
    assert layer_frac_to_index(depth=32, frac=0.0) == 0
    assert layer_frac_to_index(depth=32, frac=1.0) == 31

    # Fractional rounds down
    assert layer_frac_to_index(depth=10, frac=0.55) == 5


def test_layer_frac_bounds_check():
    """Assert invalid fractions raise ValueError."""
    from p04.loader import layer_frac_to_index

    with pytest.raises(ValueError, match="layer_frac must be between 0 and 1"):
        layer_frac_to_index(depth=80, frac=1.5)

    with pytest.raises(ValueError, match="layer_frac must be between 0 and 1"):
        layer_frac_to_index(depth=80, frac=-0.1)
