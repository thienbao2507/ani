#!/usr/bin/env python3
"""Create clean RGBA assets for the opening parallax scene.

Some source PNGs generated for this project can contain a *rendered*
checkerboard rather than usable transparency.  The script inspects every input
instead of assuming its mode or alpha state, then deliberately removes only
bright, neutral pixels that are connected to an edge of an image.  It does not
key out every bright pixel, which is important for white/cream petals and warm
sunlight inside an illustration.

Run from the repository root:

    python tools/remove_intro_checkerboard.py

Original files in ``pics/`` are never overwritten.  Clean assets are written
to ``pics/intro/``.  The supplied golden sunbeam image is intentionally not
matteable by this conservative method: its checkerboard is coloured by the
glow.  A fresh, transparent RGBA light overlay is generated instead and the
report makes that exception explicit.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage


Role = Literal["background", "foreground", "unusable-source"]


@dataclass(frozen=True)
class Asset:
    output_name: str
    source_name: str
    role: Role


# The order matches the visual stack in index.html.
ASSETS: tuple[Asset, ...] = (
    Asset("background.png", "sunlit_garden_path_in_a_dreamy_forest.png", "background"),
    Asset("flowers-far.png", "sunlit_winding_path_through_blooms.png", "foreground"),
    Asset("flowers-mid.png", "lush_floral_arrangement_with_soft_pastels.png", "foreground"),
    Asset("canopy.png", "floral_garden_path_with_soft_blooms.png", "foreground"),
    Asset("flowers-left.png", "floral_border_with_transparent_center.png", "foreground"),
    Asset("flowers-right.png", "vibrant_floral_corner_arrangement.png", "foreground"),
    Asset("sunlight.png", "golden_sunbeam_with_ethereal_glow.png", "unusable-source"),
)


def rgba_array(image: Image.Image) -> np.ndarray:
    """Return an RGBA uint8 array without assuming the source has alpha."""

    return np.asarray(image.convert("RGBA"), dtype=np.uint8)


def edge_seed(mask: np.ndarray) -> np.ndarray:
    """Seed every candidate pixel on the four borders of a binary mask."""

    seeds = np.zeros_like(mask, dtype=bool)
    seeds[0, :] = mask[0, :]
    seeds[-1, :] = mask[-1, :]
    seeds[:, 0] |= mask[:, 0]
    seeds[:, -1] |= mask[:, -1]
    return seeds


def matte_baked_checkerboard(rgba: np.ndarray) -> tuple[np.ndarray, int]:
    """Remove only edge-connected white/grey checkerboard pixels.

    ``core`` intentionally requires almost-neutral, high-value pixels.  The
    one-pixel fringe allows a soft edge around the core but remains too narrow
    to eat cream flowers or warm details.  ``binary_propagation`` is a proper
    4-neighbour flood-fill from the image borders, not a global colour key.
    """

    rgb = rgba[..., :3].astype(np.int16)
    source_alpha = rgba[..., 3]
    value = rgb.max(axis=2)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)

    # The actual white/grey tiles in the supplied source images are near
    # 239--255 with chroma close to zero.  White/cream petals tend to have a
    # noticeably warmer chroma, so they are protected by this small threshold.
    # Existing transparent/anti-aliased pixels are already safe.  Restrict the
    # colour test to opaque pixels so an image that genuinely has alpha is not
    # incorrectly reported or processed as a baked checkerboard.
    core = (source_alpha >= 250) & (value >= 225) & (chroma <= 10)
    structure = ndimage.generate_binary_structure(2, 1)  # four neighbours
    connected_core = ndimage.binary_propagation(
        edge_seed(core), structure=structure, mask=core
    )

    # Expand only into a two-pixel, still-neutral bright fringe.  This removes
    # a thin anti-aliased checkerboard halo without treating all pale pixels as
    # background.  It is intentionally not another long flood fill.
    fringe_candidates = (source_alpha >= 250) & (value >= 210) & (chroma <= 16)
    fringe = (
        ndimage.binary_dilation(connected_core, structure=structure, iterations=2)
        & fringe_candidates
    )
    checkerboard = connected_core | fringe
    removed = int(np.count_nonzero(checkerboard))

    # A source that already has a clean alpha channel should pass through
    # byte-for-byte in its visible pixels; there is no reason to blur it.
    if removed == 0:
        return rgba, 0

    alpha = np.where(checkerboard, 0, source_alpha).astype(np.uint8)

    # A sub-pixel Gaussian makes the hard matte transition span roughly 1--2
    # pixels.  Limiting it by the input alpha also preserves genuine alpha if a
    # future source image has one.
    softened = np.asarray(
        Image.fromarray(alpha, mode="L").filter(ImageFilter.GaussianBlur(radius=0.75)),
        dtype=np.uint8,
    )
    alpha = np.minimum(softened, source_alpha)

    # Checkerboard RGB can contaminate semi-transparent edge pixels.  Borrow
    # the nearest fully opaque foreground colour only inside the 2px soft edge
    # so the resulting PNG composites cleanly over a dark garden.
    full_foreground = alpha >= 250
    if np.any(full_foreground):
        distance, indices = ndimage.distance_transform_edt(
            ~full_foreground, return_indices=True
        )
        soft_edge = (alpha > 0) & (alpha < 250) & (distance <= 2.5)
        if np.any(soft_edge):
            nearest_rgb = rgba[..., :3][tuple(indices)]
            rgba[..., :3][soft_edge] = nearest_rgb[soft_edge]

    rgba[..., 3] = alpha
    return rgba, removed


def make_sunlight_fallback(size: tuple[int, int]) -> Image.Image:
    """Create a transparent, checkerboard-free warm light overlay.

    The original golden-sunbeam source is not used here: its baked checkerboard
    is tinted by the very glow that should be retained, so an edge flood-fill
    cannot distinguish them safely.  This replacement keeps the requested
    sunlight layer without risking a rectangular checkerboard panel.
    """

    width, height = size
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    x /= max(width - 1, 1)
    y /= max(height - 1, 1)

    # A compact sun in the upper-right plus two soft diagonal shafts.
    sun = np.exp(-(((x - 0.90) / 0.20) ** 2 + ((y - 0.07) / 0.19) ** 2) * 1.25)
    shaft_a = np.exp(-((y - (0.08 + 0.38 * (0.92 - x))) / 0.13) ** 2)
    shaft_b = np.exp(-((y - (0.12 + 0.74 * (0.90 - x))) / 0.19) ** 2)
    reach = np.clip((x - 0.18) / 0.75, 0.0, 1.0) * np.clip(1.12 - y, 0.0, 1.0)
    glow = np.clip(sun * 1.15 + (shaft_a * 0.34 + shaft_b * 0.18) * reach, 0, 1)

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 1] = 211
    rgba[..., 2] = 132
    rgba[..., 3] = np.round(glow * 185).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA").filter(ImageFilter.GaussianBlur(radius=0.45))


def inspect(image: Image.Image, name: str, forced_checker: bool = False) -> dict[str, object]:
    """Collect the required alpha facts and a conservative checker diagnosis."""

    rgba = rgba_array(image)
    rgb = rgba[..., :3].astype(np.int16)
    value = rgb.max(axis=2)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    # RGB values underneath fully transparent pixels do not render.  Excluding
    # them is what distinguishes a true-alpha PNG from a white checkerboard
    # baked into visible RGB pixels.
    neutral_bright = (rgba[..., 3] >= 250) & (value >= 225) & (chroma <= 10)
    connected = ndimage.binary_propagation(
        edge_seed(neutral_bright),
        structure=ndimage.generate_binary_structure(2, 1),
        mask=neutral_bright,
    )
    coverage = float(np.count_nonzero(connected)) / connected.size
    has_alpha = "A" in image.getbands()
    checker = forced_checker or coverage > 0.08
    return {
        "name": name,
        "mode": image.mode,
        "size": f"{image.width}x{image.height}",
        "has_alpha": has_alpha,
        "native_transparency": bool(np.any(rgba[..., 3] < 255)),
        "alpha_lt_255": int(np.count_nonzero(rgba[..., 3] < 255)),
        "edge_neutral_pixels": int(np.count_nonzero(connected)),
        "edge_neutral_percent": coverage * 100,
        "checkerboard_baked": checker,
    }


def print_report(record: dict[str, object], note: str = "") -> None:
    verdict = "YES" if record["checkerboard_baked"] else "NO"
    print(
        f"{record['name']}: mode={record['mode']}; size={record['size']}; "
        f"alpha_channel={record['has_alpha']}; native_transparency={record['native_transparency']}; "
        f"alpha<255={record['alpha_lt_255']}; "
        f"edge_neutral={record['edge_neutral_pixels']} "
        f"({record['edge_neutral_percent']:.2f}%); baked_checkerboard={verdict}{note}"
    )


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=repo / "pics")
    parser.add_argument("--output-dir", type=Path, default=repo / "pics" / "intro")
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="print the source alpha/checker report without writing output files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    source_images: dict[str, Image.Image] = {}

    print("Source image inspection")
    for asset in ASSETS:
        source_path = source_dir / asset.source_name
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing source image: {source_path}")
        image = Image.open(source_path)
        source_images[asset.output_name] = image
        print_report(
            inspect(image, asset.source_name, forced_checker=asset.role == "unusable-source"),
            "; unsafe to matte conservatively" if asset.role == "unusable-source" else "",
        )

    if args.inspect_only:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting clean assets to {output_dir}")
    for asset in ASSETS:
        source_path = source_dir / asset.source_name
        output_path = output_dir / asset.output_name
        if source_path.resolve() == output_path.resolve():
            raise RuntimeError("Refusing to overwrite a source image")

        source = source_images[asset.output_name]
        if asset.role == "background":
            # This is a complete opaque garden photograph, not a cutout.
            output = source.convert("RGBA")
            detail = "copied as opaque background (no checkerboard detected)"
        elif asset.role == "unusable-source":
            output = make_sunlight_fallback(source.size)
            detail = "generated clean RGBA fallback; original tinted checkerboard was not used"
        else:
            output_array, removed = matte_baked_checkerboard(rgba_array(source).copy())
            output = Image.fromarray(output_array, mode="RGBA")
            detail = f"removed {removed:,} edge-connected neutral checkerboard pixels"

        output.save(output_path, format="PNG", optimize=True)
        print(f"{asset.output_name}: {detail}")

    print(
        "\nNote: golden_sunbeam_with_ethereal_glow.png was intentionally excluded "
        "from the intro because its checkerboard is tinted by the glow. "
        "sunlight.png is a newly generated transparent replacement."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
