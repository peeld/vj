"""
color_harmony.py — Generate harmonious, muted color palettes using color theory.

Supported schemes
-----------------
complementary       2 colors, 180° apart — high contrast, stable
split_complementary 3 colors, base + flanks of its complement — softer contrast
triadic             3 colors, 120° apart — vibrant but balanced
analogous           4 colors, 30° steps — cohesive, low tension
tetradic            4 colors, 90° apart — rich, needs care
monochromatic       4 tones of one hue — minimal, elegant

Saturation and lightness are kept in a "muted mid-range" by default so
colors look good together without being garish.

Usage
-----
    from color_harmony import ColorScheme, generate_palette, preview_palette

    colors = generate_palette(ColorScheme.TRIADIC, seed=42)
    preview_palette(colors)

    # For Warp rendering
    warp_colors = palette_to_warp(colors)
"""

import colorsys
import random
from enum import Enum
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

RGB = tuple[float, float, float]       # each channel in [0, 1]
RGB8 = tuple[int, int, int]            # each channel in [0, 255]

# ---------------------------------------------------------------------------
# Defaults — "nice" range: moderate saturation, mid lightness
# ---------------------------------------------------------------------------

DEFAULT_SAT: tuple[float, float] = (0.35, 0.60)
DEFAULT_LIT: tuple[float, float] = (0.42, 0.68)


# ---------------------------------------------------------------------------
# Scheme definitions
# ---------------------------------------------------------------------------

class ColorScheme(Enum):
    COMPLEMENTARY       = "complementary"
    SPLIT_COMPLEMENTARY = "split_complementary"
    TRIADIC             = "triadic"
    ANALOGOUS           = "analogous"
    TETRADIC            = "tetradic"
    MONOCHROMATIC       = "monochromatic"


# Hue offsets (in [0, 1] turns) for each scheme, relative to the base hue.
# Each inner list is one color's offset.
_SCHEME_OFFSETS: dict[ColorScheme, list[float]] = {
    ColorScheme.COMPLEMENTARY:       [0.0, 0.500],
    ColorScheme.SPLIT_COMPLEMENTARY: [0.0, 0.417, 0.583],   # base, 150°, 210°
    ColorScheme.TRIADIC:             [0.0, 1/3,  2/3],
    ColorScheme.ANALOGOUS:           [0.0, 1/12, 2/12, 3/12],
    ColorScheme.TETRADIC:            [0.0, 0.25, 0.50, 0.75],
    ColorScheme.MONOCHROMATIC:       [0.0, 0.0,  0.0,  0.0],  # hue fixed; lightness varies
}


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def generate_palette(
    scheme: "ColorScheme | str" = ColorScheme.COMPLEMENTARY,
    seed: Optional[int] = None,
    sat_range: tuple[float, float] = DEFAULT_SAT,
    lit_range: tuple[float, float] = DEFAULT_LIT,
    as_uint8: bool = False,
) -> list[RGB] | list[RGB8]:
    """
    Generate a harmonious color palette.

    Parameters
    ----------
    scheme:    Color harmony rule. String names accepted ('triadic', etc.).
    seed:      RNG seed for reproducibility.
    sat_range: (min, max) saturation clamp.
    lit_range: (min, max) lightness clamp.
    as_uint8:  Return integer RGB tuples (0–255) instead of float (0–1).

    Returns
    -------
    List of RGB tuples — length depends on scheme (2–4 colors).
    """
    if isinstance(scheme, str):
        scheme = ColorScheme(scheme)

    rng = random.Random(seed)

    base_h = rng.random()
    base_s = rng.uniform(*sat_range)
    base_l = rng.uniform(*lit_range)

    offsets = _SCHEME_OFFSETS[scheme]
    palette: list[RGB] = []

    for i, offset in enumerate(offsets):
        h = (base_h + offset) % 1.0

        if scheme == ColorScheme.MONOCHROMATIC:
            # Vary lightness evenly across the allowed range
            t = i / max(len(offsets) - 1, 1)
            l = lit_range[0] + t * (lit_range[1] - lit_range[0])
            s = base_s
        else:
            # Slight per-color jitter so colors aren't robotically uniform
            l = float(np.clip(base_l + (i % 2) * 0.07 - 0.035, *lit_range))
            s = float(np.clip(base_s + (i % 3) * 0.05 - 0.05,  *sat_range))

        # colorsys uses HLS order (hue, lightness, saturation)
        rgb: RGB = colorsys.hls_to_rgb(h, l, s)
        palette.append(rgb)

    if as_uint8:
        return [tuple(round(c * 255) for c in color) for color in palette]  # type: ignore[return-value]
    return palette


def generate_named_palette(
    scheme: "ColorScheme | str" = ColorScheme.COMPLEMENTARY,
    seed: Optional[int] = None,
    **kwargs,
) -> dict[str, RGB]:
    """
    Same as generate_palette but returns ``{"primary": rgb, "secondary": rgb, ...}``.
    """
    colors = generate_palette(scheme, seed, **kwargs)
    roles = ["primary", "secondary", "tertiary", "quaternary"]
    return {roles[i]: c for i, c in enumerate(colors)}


# ---------------------------------------------------------------------------
# Warp integration
# ---------------------------------------------------------------------------

def palette_to_warp(colors: list[RGB]) -> "wp.array":  # type: ignore[name-defined]
    """
    Convert a palette to a ``warp.array`` of shape (N, 3), dtype float32.

    Each row is one color as [r, g, b] in [0, 1].
    """
    import warp as wp  # local import so the module is usable without warp

    arr = np.array(colors, dtype=np.float32)
    return wp.array(arr, dtype=wp.float32).reshape(len(colors), 3)


def palette_to_numpy(colors: list[RGB], dtype=np.float32) -> np.ndarray:
    """Return palette as a (N, 3) numpy array."""
    return np.array(colors, dtype=dtype)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def to_hex(color: RGB) -> str:
    """Convert float RGB to '#RRGGBB' hex string."""
    r, g, b = (round(c * 255) for c in color)
    return f"#{r:02X}{g:02X}{b:02X}"


def preview_palette(colors: list[RGB], label: str = "") -> None:
    """
    Print a color swatch for each palette entry using ANSI escape codes.
    Requires a terminal that supports 24-bit color.
    """
    header = f"Palette ({label})" if label else "Palette"
    print(f"\n{header}  [{len(colors)} colors]\n")
    for i, (r, g, b) in enumerate(colors):
        r8, g8, b8 = round(r * 255), round(g * 255), round(b * 255)
        swatch = f"\033[48;2;{r8};{g8};{b8}m        \033[0m"
        print(f"  [{i}] {swatch}  {to_hex((r, g, b))}  rgb({r8:3d}, {g8:3d}, {b8:3d})")
    print()


def preview_all_schemes(seed: Optional[int] = None) -> None:
    """Print swatches for every scheme using the same seed."""
    for scheme in ColorScheme:
        colors = generate_palette(scheme, seed=seed)
        preview_palette(colors, label=scheme.value)


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7

    print(f"Seed: {seed}")
    preview_all_schemes(seed=seed)

    # Show dict form
    named = generate_named_palette(ColorScheme.TRIADIC, seed=seed)
    print("Named palette (triadic):")
    for role, color in named.items():
        print(f"  {role:12s}  {to_hex(color)}")
